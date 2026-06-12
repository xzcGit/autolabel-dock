"""LocateAnything-3B optional auto-labeling backend (out-of-process sidecar).

This module is the **first non-Ultralytics backend** plugged into the
``ModelBackend`` system from ``06-04-model-backends``.

Architecture — **out-of-process sidecar** (2026-06-08 revision):
    The earlier in-process implementation activated LA's CUDA context (torch +
    bitsandbytes 4-bit) inside the Qt GUI process, which killed the X server
    driving the same single GPU (diagnosed root cause; see memory
    ``locateanything-runtime``). The fix moves **all** of LA's torch/CUDA into a
    separate long-lived subprocess (``locateanything_worker.py``), reproducing
    the standalone-script process structure that runs the same model on the same
    card without crashing.

    Consequence — **the GUI process never touches torch/transformers/bitsandbytes
    on the LA path** (not even for the CUDA/VRAM preflight, which becomes a cheap
    ``nvidia-smi`` parse here, with the *real* torch check done by the worker at
    startup). Parsing, class mapping, prompt assembly, and ``set_query`` all stay
    pure-Python in this process. The predictor is a thin **client/proxy** that
    ships ``(image_path, prompt)`` to the worker over newline-delimited JSON and
    receives back ``{raw_text, width, height}``.

Hard constraint — **dependency isolation**:
    The module top level imports ONLY stdlib and ``src.engine.backends.base``.
    Heavy ML deps live exclusively in the worker subprocess. A regression test
    asserts ``torch`` / ``transformers`` / ``bitsandbytes`` are absent from the
    GUI process' ``sys.modules`` even after LA is "enabled".

Cost unlock is four-tier (startup stays at tier 0):
    0 register  — app startup, pure stdlib, register a lightweight object.
    1 probe()   — user clicks toolbar button: cheap version + HF-cache checks.
    2 preflight()— continue after probe: cheap nvidia-smi GPU/VRAM check (no torch).
    3 load_runtime() — preflight passed: spawn the worker subprocess; it does the
                  real torch CUDA check + 4-bit NF4 model load and reports back.

The runtime / inference contract mirrors the ground-truth reference script
``.trellis/tasks/06-08-locateanything-backend/research/locate_anything_4bit_reference.py``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

from src.engine.backends.base import (
    BackendProbe,
    BackendUnavailableError,
)

logger = logging.getLogger(__name__)

# ── Constants (centralized, tunable) ──────────────────────────────────────

BACKEND_ID = "locateanything"
DISPLAY_NAME = "LocateAnything (文本标注)"
MODEL_REPO_ID = "nvidia/LocateAnything-3B"
MODEL_FORMAT = "vlm-hf"
RUNTIME = "subprocess"

# Required optional dependencies — distribution names for importlib.metadata.
_REQUIRED_PACKAGES = ("transformers", "accelerate", "bitsandbytes")

# VRAM gating thresholds (GiB). Centralized so they are easy to tune.
# NOTE: on a single-GPU machine the desktop/X server shares this same card,
# so the inference peak competes with display memory. Keep MIN_FREE_VRAM_GB
# generous to leave headroom for the transient generation peak. The worker
# subprocess re-checks these with torch at startup (see locateanything_worker).
MIN_TOTAL_VRAM_GB = 6.0
MIN_FREE_VRAM_GB = 5.0

# How long (seconds) to wait for the worker to report ``ready`` before giving
# up the load, and how long a single inference request may take.
WORKER_STARTUP_TIMEOUT_S = 600.0
WORKER_INFERENCE_TIMEOUT_S = 120.0
# Grace period for a graceful shutdown before we hard-kill the subprocess.
WORKER_SHUTDOWN_TIMEOUT_S = 10.0

# Advisory appended to probe / preflight messages: on a single-GPU box the
# display and inference share VRAM, which is tight.
_SHARED_GPU_HINT = (
    "提示：若显示器与推理共用同一块 GPU，单卡共享显存较紧张，"
    "建议关闭其他占用显存的程序后再使用。"
)

# Box coordinates emitted by the model are in the [0, 1000] corner space.
_BOX_SCALE = 1000.0

# Matches "<box><x1><y1><x2><y2></box>" optionally preceded by
# "<ref>label</ref>". The ref group is optional so plain boxes still parse.
_REF_BOX_RE = re.compile(
    r"(?:<ref>(?P<label>.*?)</ref>)?\s*"
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)

# Path to the worker entry script (same directory as this module).
_WORKER_SCRIPT = str(Path(__file__).resolve().with_name("locateanything_worker.py"))


def _hf_cache_has_model() -> bool:
    """Cheap filesystem check: is the LA model present in the HF hub cache?

    No network, no import of huggingface_hub. We look for the conventional
    ``models--nvidia--LocateAnything-3B`` directory under the HF cache roots,
    and require at least one ``*.safetensors`` snapshot file inside.
    """
    repo_dir_name = "models--" + MODEL_REPO_ID.replace("/", "--")
    candidates: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        candidates.append(Path(hub_cache))
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in candidates:
        repo_dir = root / repo_dir_name
        if not repo_dir.is_dir():
            continue
        snapshots = repo_dir / "snapshots"
        search_root = snapshots if snapshots.is_dir() else repo_dir
        try:
            if any(search_root.rglob("*.safetensors")):
                return True
        except OSError:
            continue
    return False


def _query_nvidia_smi_vram() -> tuple[bool, float, float, str]:
    """Cheap GPU/VRAM check via ``nvidia-smi`` — **no torch import**.

    Returns ``(ok, total_gb, free_gb, message)``. ``ok`` is False if nvidia-smi
    is missing/unparsable (treated as "no usable GPU detected by the cheap
    path"). This is only an advisory pre-gate in the GUI process — the worker
    subprocess does the authoritative ``torch.cuda`` check at startup, so a
    false negative here just means the user can't even attempt the load, and a
    false positive is caught by the worker's preflight.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (
            False,
            0.0,
            0.0,
            f"未检测到 nvidia-smi / 可用 GPU：{exc}。LocateAnything 需要 NVIDIA GPU。",
        )
    if proc.returncode != 0:
        return (
            False,
            0.0,
            0.0,
            "nvidia-smi 调用失败，未检测到可用的 NVIDIA GPU。LocateAnything 需要 GPU。",
        )

    # Pick the GPU with the most free memory (multi-GPU boxes).
    best_total = best_free = -1.0
    for raw in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            total_mib = float(parts[0])
            free_mib = float(parts[1])
        except ValueError:
            continue
        if free_mib > best_free:
            best_free = free_mib
            best_total = total_mib

    if best_total < 0:
        return (
            False,
            0.0,
            0.0,
            "无法解析 nvidia-smi 显存信息。LocateAnything 需要 NVIDIA GPU。",
        )

    total_gb = best_total / 1024.0
    free_gb = best_free / 1024.0
    return True, total_gb, free_gb, ""


class _WorkerProcess:
    """Owns the LA sidecar subprocess and the newline-delimited JSON IPC.

    All ``send``/``recv`` are serialized by a lock so single-image and batch
    paths (both running on a background QThread, but distinct ones) can never
    interleave protocol frames. Lives entirely in the GUI process but imports
    **no** torch — the heavy deps are inside the spawned subprocess.
    """

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._lock = threading.Lock()

    # ── Startup handshake ─────────────────────────────────────────────────

    def wait_until_ready(
        self,
        progress_cb: Callable[[str], None] | None = None,
        timeout: float = WORKER_STARTUP_TIMEOUT_S,
    ) -> None:
        """Read startup frames until ``ready`` / failure / EOF.

        Raises ``BackendUnavailableError`` on ``preflight_failed`` / ``fatal`` /
        early exit, and ``RuntimeError`` on timeout. ``status`` frames are
        forwarded to ``progress_cb``.

        ``timeout`` bounds the whole startup (the heavy 4-bit load can take ~40s
        cold; the generous default leaves room for a slow first-time disk read).
        """
        import time

        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() > deadline:
                self.terminate()
                raise RuntimeError("LocateAnything 子进程启动超时。")
            line = self._read_line_blocking()
            if line is None:
                # EOF before ready → worker died during startup.
                code = self._proc.poll()
                raise BackendUnavailableError(
                    f"LocateAnything 子进程在加载阶段意外退出 (code={code})。"
                    "详情见日志。"
                )
            try:
                msg = json.loads(line)
            except (ValueError, TypeError):
                continue  # ignore non-protocol noise on stdout (best-effort)
            if not isinstance(msg, dict):
                continue
            if msg.get("ready") is True:
                return
            if "preflight_failed" in msg:
                self.terminate()
                raise BackendUnavailableError(str(msg["preflight_failed"]))
            if "fatal" in msg:
                self.terminate()
                raise BackendUnavailableError(str(msg["fatal"]))
            status = msg.get("status")
            if status and progress_cb is not None:
                try:
                    progress_cb(str(status))
                except Exception:  # noqa: BLE001
                    logger.debug("progress_cb raised", exc_info=True)

    # ── Inference round-trip ──────────────────────────────────────────────

    def infer(
        self, image_path: str, prompt: str, timeout: float = WORKER_INFERENCE_TIMEOUT_S,
    ) -> tuple[str, int, int]:
        """Send one inference request and return ``(raw_text, width, height)``.

        Raises ``RuntimeError`` on worker error / crash / timeout. Thread-safe.
        """
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"LocateAnything 子进程已退出 (code={self._proc.poll()})，无法推理。"
                )
            self._send({"image": str(image_path), "prompt": prompt})
            line = self._read_line_blocking(timeout=timeout)
        if line is None:
            code = self._proc.poll()
            raise RuntimeError(
                f"LocateAnything 子进程在推理时意外退出 (code={code})。详情见日志。"
            )
        try:
            msg = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"子进程返回无法解析: {line!r}") from exc
        if isinstance(msg, dict) and "error" in msg:
            raise RuntimeError(str(msg["error"]))
        if not isinstance(msg, dict) or "raw_text" not in msg:
            raise RuntimeError(f"子进程返回格式错误: {msg!r}")
        return (
            str(msg.get("raw_text", "")),
            int(msg.get("width", 0)),
            int(msg.get("height", 0)),
        )

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    # ── Teardown ──────────────────────────────────────────────────────────

    def terminate(self) -> None:
        """Stop the worker: send shutdown, then terminate/kill if it lingers.

        Best-effort and idempotent. Does NOT import torch — the subprocess owns
        the CUDA context, so killing it frees the VRAM.
        """
        proc = self._proc
        if proc.poll() is not None:
            self._close_streams()
            return
        with self._lock:
            try:
                self._send({"cmd": "shutdown"})
            except Exception:  # noqa: BLE001 - pipe may already be broken
                pass
            try:
                stdin = proc.stdin
                if stdin is not None:
                    stdin.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("LocateAnything worker did not exit; terminating.")
            proc.terminate()
            try:
                proc.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning("LocateAnything worker ignored terminate; killing.")
                proc.kill()
                try:
                    proc.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    logger.error("LocateAnything worker could not be killed.")
        self._close_streams()

    # ── Low-level IPC ─────────────────────────────────────────────────────

    def _send(self, obj: dict) -> None:
        stdin = self._proc.stdin
        if stdin is None:
            raise RuntimeError("子进程 stdin 不可用。")
        stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stdin.flush()

    def _read_line_blocking(self, timeout: float | None = None) -> str | None:
        """Read one line from the worker stdout. Returns ``None`` on EOF.

        ``timeout`` (seconds) bounds the wait by reading on a helper thread and
        joining with the timeout; on timeout the worker is terminated and a
        ``RuntimeError`` is raised so the caller surfaces a readable message
        instead of hanging the GUI forever.
        """
        stdout = self._proc.stdout
        if stdout is None:
            return None
        if timeout is None:
            line = stdout.readline()
            return line if line else None

        result: list[str | None] = []

        def _reader():
            try:
                result.append(stdout.readline())
            except Exception:  # noqa: BLE001
                result.append(None)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            # The read is stuck (worker hung) — kill it so the thread can exit.
            logger.error("LocateAnything worker timed out; killing.")
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                "LocateAnything 推理超时，已终止子进程。请重试或检查日志。"
            )
        if not result:
            return None
        line = result[0]
        return line if line else None

    def _close_streams(self) -> None:
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass


class LocateAnythingBackend:
    """Optional, lazily-loaded open-vocabulary detection backend.

    Implements the ``ModelBackend`` protocol plus two LA-specific runtime
    methods (``preflight`` and ``load_runtime``) used by the dedicated
    controller — they are NOT part of the generic backend protocol because
    only LA needs the GPU/VRAM gating + background heavy load.
    """

    backend_id = BACKEND_ID
    display_name = DISPLAY_NAME

    # ── Tier 1: probe ─────────────────────────────────────────────────────

    def probe(self) -> BackendProbe:
        """Cheap, side-effect-free availability check.

        Checks optional dependency versions via ``importlib.metadata`` and the
        presence of model weights in the HF cache via the filesystem. Never
        imports torch/transformers and never raises — missing pieces return
        ``available=False`` with an actionable Chinese message.
        """
        missing: list[str] = []
        versions: dict[str, str] = {}
        for pkg in _REQUIRED_PACKAGES:
            try:
                versions[pkg] = version(pkg)
            except PackageNotFoundError:
                missing.append(pkg)
            except Exception:  # noqa: BLE001 - probe must never raise
                missing.append(pkg)

        if missing:
            return BackendProbe(
                backend_id=self.backend_id,
                display_name=self.display_name,
                available=False,
                runtime=RUNTIME,
                message=(
                    "缺少可选依赖: "
                    + ", ".join(missing)
                    + "。请运行 `pip install -e .[locateanything]` 安装。"
                ),
                metadata={"missing_packages": missing},
            )

        weights_ready = _hf_cache_has_model()
        if not weights_ready:
            return BackendProbe(
                backend_id=self.backend_id,
                display_name=self.display_name,
                available=False,
                version=versions.get("transformers", ""),
                runtime=RUNTIME,
                message=(
                    f"未在本地 HuggingFace 缓存找到权重 {MODEL_REPO_ID}。"
                    "请先下载模型权重到 HF 缓存后再试。"
                ),
                metadata={"weights_ready": False, "package_versions": versions},
            )

        return BackendProbe(
            backend_id=self.backend_id,
            display_name=self.display_name,
            available=True,
            version=versions.get("transformers", ""),
            runtime=RUNTIME,
            message="依赖与权重就绪。" + _SHARED_GPU_HINT,
            metadata={"weights_ready": True, "package_versions": versions},
        )

    def infer_model_format(self, model_path: str | Path) -> str:
        """LA loads from a repo-id, not a file — report the VLM format label."""
        return MODEL_FORMAT

    def load_predictor(self, model_path: str | Path, model_info):
        """Protocol entry point. LA has no model file, so loading is routed
        through ``load_runtime`` by the dedicated controller instead. This
        method exists to satisfy the protocol and simply delegates.
        """
        return self.load_runtime()

    def create_trainer(self):
        """LA is inference-only (open-vocabulary, no training)."""
        raise BackendUnavailableError(
            "LocateAnything 是开放词汇推理后端，不支持训练。"
        )

    # ── Tier 2: preflight (NO torch — cheap nvidia-smi only) ──────────────

    def preflight(self) -> dict:
        """Cheap GPU/VRAM gate via ``nvidia-smi`` — **never imports torch**.

        Returns a structured dict:
            {"ok": bool, "message": str, "total_gb": float, "free_gb": float}

        The authoritative ``torch.cuda`` check runs inside the worker subprocess
        at startup; this GUI-side gate just avoids spawning the heavy worker when
        there is obviously no usable GPU / not enough free VRAM. Pure CPU is
        intentionally NOT offered (too slow).
        """
        ok, total_gb, free_gb, message = _query_nvidia_smi_vram()
        if not ok:
            return {"ok": False, "message": message, "total_gb": 0.0, "free_gb": 0.0}

        if total_gb < MIN_TOTAL_VRAM_GB:
            return {
                "ok": False,
                "message": (
                    f"显卡总显存 {total_gb:.1f}GB 低于建议的 {MIN_TOTAL_VRAM_GB:.0f}GB，"
                    "无法稳定运行 LocateAnything。"
                ),
                "total_gb": total_gb,
                "free_gb": free_gb,
            }

        if free_gb < MIN_FREE_VRAM_GB:
            return {
                "ok": False,
                "message": (
                    f"空闲显存仅 {free_gb:.1f}GB，低于所需的 ~{MIN_FREE_VRAM_GB:.0f}GB。"
                    "请关闭占用显存的程序后重试。"
                    + _SHARED_GPU_HINT
                ),
                "total_gb": total_gb,
                "free_gb": free_gb,
            }

        return {
            "ok": True,
            "message": (
                f"显存检查通过 (空闲 {free_gb:.1f}GB / 总 {total_gb:.1f}GB)。"
                + _SHARED_GPU_HINT
            ),
            "total_gb": total_gb,
            "free_gb": free_gb,
        }

    # ── Tier 3: load runtime (spawn the sidecar subprocess) ───────────────

    def load_runtime(
        self, progress_cb: Callable[[str], None] | None = None
    ) -> "LocateAnythingPredictor":
        """Spawn the LA worker subprocess and return a client predictor.

        The GUI process imports **no** torch here — it only launches a Python
        subprocess (same interpreter / ``yolov8`` env) running the worker entry
        script, then reads its startup handshake. The worker does the heavy
        4-bit NF4 + fp16 load and the real CUDA check. ``progress_cb`` receives
        the worker's status strings for the UI. Raises on preflight/load failure.
        """
        def _report(msg: str) -> None:
            logger.info("LA load: %s", msg)
            if progress_cb is not None:
                try:
                    progress_cb(msg)
                except Exception:  # noqa: BLE001
                    logger.debug("progress_cb raised", exc_info=True)

        _report("正在启动 LocateAnything 子进程…")

        # Worker stderr → log file so library/model-remote-code chatter and any
        # traceback are captured for diagnosis without polluting the protocol.
        log_dir = Path.home() / ".autolabel" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            stderr_target = open(log_dir / "locateanything_worker.log", "a", encoding="utf-8")
        except OSError:
            stderr_target = subprocess.DEVNULL

        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Mitigate "free VRAM exists but the allocation still OOMs" on this
        # memory-tight single GPU shared with the X server: PyTorch's default
        # caching allocator fragments VRAM (bitsandbytes 4-bit dequant scratch +
        # the transient attention/KV peak need a *contiguous* block, which can be
        # far smaller than total free memory). ``expandable_segments:True`` lets
        # the allocator grow/shrink segments instead of pre-carving fixed blocks,
        # which is the standard fix for fragmentation-driven OOM. ``setdefault``
        # so a value set in the launching shell still wins.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Ensure the worker can import the project package (it imports stdlib only
        # at top level, but is launched as a script).
        repo_root = str(Path(__file__).resolve().parents[3])
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root + (os.pathsep + existing_pp if existing_pp else "")

        proc = subprocess.Popen(
            [sys.executable, "-u", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=repo_root,
        )

        worker = _WorkerProcess(proc)
        try:
            worker.wait_until_ready(progress_cb=_report)
        except Exception:
            worker.terminate()
            raise
        _report("加载完成")
        return LocateAnythingPredictor(worker=worker)


class LocateAnythingPredictor:
    """Client/proxy implementing ``PredictorProtocol`` for LA detection.

    Holds a handle to the sidecar :class:`_WorkerProcess` and ships
    ``(image_path, prompt)`` to it; the heavy torch/CUDA work happens in the
    subprocess. **All parsing / class mapping / prompt assembly / query state is
    pure-Python in this (GUI) process — no torch/transformers/PIL imports.**

    The protocol's ``predict`` / ``predict_with_size`` signatures have no
    ``prompt`` parameter, so the natural-language query is carried as instance
    state via :meth:`set_query` instead of changing the protocol or the
    ``BatchPredictWorker`` signature.

    Natural-language → class mapping is **hybrid**:
        - target_class set  → every returned box is forced into that class.
        - no target_class   → each ``<ref>label</ref>`` is normalized and
          matched back to a project class; unmatched boxes are dropped and
          counted (surfaced via :attr:`last_dropped`).
        - empty prompt      → fall back to building the prompt from the
          project class names (equivalent to the boundary-doc v1 behavior).
    """

    def __init__(self, worker: _WorkerProcess | None = None):
        self._worker = worker
        self._prompt: str = ""
        self._target_class: str | None = None
        # Number of detections dropped by the name-matching rule on the most
        # recent predict call (for status-bar feedback).
        self.last_dropped: int = 0

    # ── Query state (carries natural language without a protocol change) ──

    def set_query(self, prompt: str, target_class: str | None) -> None:
        """Set the natural-language prompt and optional forced target class."""
        self._prompt = (prompt or "").strip()
        self._target_class = (target_class or None)

    # ── Class-name normalization (shared with Ultralytics Predictor) ──────

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        """Normalize class names for tolerant matching (mirrors Predictor)."""
        return " ".join(str(class_name).split()).casefold()

    # ── Prompt assembly ───────────────────────────────────────────────────

    def _build_prompt(self, project_classes: list[str] | None) -> str:
        """Pick the detection prompt body following the hybrid rules."""
        if self._prompt:
            desc = self._prompt
        elif project_classes:
            desc = "</c>".join(project_classes)
        else:
            desc = ""
        return (
            "Locate all the instances that matches the following description: "
            f"{desc}."
        )

    # ── Parsing + hybrid class mapping ────────────────────────────────────

    def _parse_and_map(
        self, raw_text: str, project_classes: list[str] | None
    ) -> list:
        """Parse <ref>/<box> spans and map to project classes (hybrid rules).

        Returns a list of ``Annotation`` with normalized center-format bbox,
        ``source='auto'``, ``confirmed=False``, ``confidence=1.0``. Updates
        :attr:`last_dropped`.
        """
        from src.core.annotation import Annotation  # local: avoid Qt-free import churn

        project_classes = project_classes or []
        # Lookup for name-matching (only used when no target_class).
        name_lookup: dict[str, tuple[str, int]] = {
            self._normalize_class_name(name): (name, idx)
            for idx, name in enumerate(project_classes)
        }

        annotations: list = []
        dropped = 0

        for m in _REF_BOX_RE.finditer(raw_text):
            x1, y1, x2, y2 = (
                int(m.group("x1")),
                int(m.group("y1")),
                int(m.group("x2")),
                int(m.group("y2")),
            )
            label = m.group("label")

            if self._target_class is not None:
                # Force every box into the chosen target class.
                class_name = self._target_class
                if class_name in project_classes:
                    class_id = project_classes.index(class_name)
                else:
                    class_id = 0
            else:
                # Match the returned <ref> label back to a project class.
                norm = self._normalize_class_name(label or "")
                match = name_lookup.get(norm)
                if match is None:
                    dropped += 1
                    continue
                class_name, class_id = match

            # The model sometimes emits corners in reversed order (x1>x2 or
            # y1>y2 — observed on real images). Computing w/h as a raw
            # (x2-x1)/(y2-y1) then yields a NEGATIVE-dimension bbox, which the
            # canvas can neither hit-test (hit_test requires x1<=x2, y1<=y2 — the
            # box becomes unselectable) nor render correctly. Normalize the
            # corner order with min/max before deriving center+size.
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            lo_y, hi_y = min(y1, y2), max(y1, y2)
            cx = round((lo_x + hi_x) / 2.0 / _BOX_SCALE, 6)
            cy = round((lo_y + hi_y) / 2.0 / _BOX_SCALE, 6)
            bw = round((hi_x - lo_x) / _BOX_SCALE, 6)
            bh = round((hi_y - lo_y) / _BOX_SCALE, 6)

            ann = Annotation(
                class_name=class_name,
                class_id=class_id,
                bbox=(cx, cy, bw, bh),
                keypoints=[],
                confidence=1.0,
                confirmed=False,
                source="auto",
            )
            ann.clamp()
            annotations.append(ann)

        self.last_dropped = dropped
        if dropped:
            logger.info(
                "LocateAnything dropped %d detection(s) that did not match "
                "project classes %s",
                dropped,
                project_classes,
            )
        return annotations

    # ── PredictorProtocol surface ─────────────────────────────────────────

    def predict(
        self,
        image_path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
    ) -> list:
        """Run open-vocabulary detection. conf/iou/class_match_mode ignored."""
        annotations, _ = self._run(image_path, project_classes)
        return annotations

    def predict_with_size(
        self,
        image_path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
    ) -> tuple[list, tuple[int, int]]:
        """Run detection and also report image (w, h) reported by the worker."""
        return self._run(image_path, project_classes)

    def predict_classify(
        self,
        image_path,
        project_classes: list[str] | None = None,
        filter_to_project: bool = True,
    ):
        """LA is detect-only — classify is not supported."""
        raise BackendUnavailableError(
            "LocateAnything 仅支持检测任务，不支持分类。"
        )

    def _run(
        self, image_path, project_classes: list[str] | None
    ) -> tuple[list, tuple[int, int]]:
        """Ship (image, prompt) to the sidecar, then parse/map the raw text.

        The client never opens the image or imports PIL/torch — the worker
        reports the original image (w, h) alongside the raw text.
        """
        if self._worker is None:
            raise RuntimeError("LocateAnything 子进程未就绪。")
        prompt = self._build_prompt(project_classes)
        raw_text, width, height = self._worker.infer(str(image_path), prompt)
        annotations = self._parse_and_map(raw_text, project_classes)
        return annotations, (width, height)

    # ── VRAM lifecycle ────────────────────────────────────────────────────

    def release(self) -> None:
        """Terminate the sidecar subprocess (frees its CUDA context / VRAM).

        **Does not import torch** — the subprocess owns the CUDA context, so
        killing it releases the GPU memory. Idempotent and best-effort.
        """
        worker = self._worker
        self._worker = None
        if worker is not None:
            try:
                worker.terminate()
            except Exception:  # noqa: BLE001 - teardown must be best-effort
                logger.debug("worker terminate failed", exc_info=True)
