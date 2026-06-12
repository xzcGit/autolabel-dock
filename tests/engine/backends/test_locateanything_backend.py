"""Tests for the optional LocateAnything backend (out-of-process sidecar).

Critical: these tests must NOT trigger any heavy ML import or model load. The
backend lives in the GUI process and imports NO torch — preflight is a cheap
``nvidia-smi`` parse, and ``load_runtime`` spawns a subprocess (mocked here).
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.engine.backends.base import BackendUnavailableError

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Dependency isolation guard (the most important invariant) ─────────────


def test_importing_module_does_not_import_torch_or_transformers():
    """Importing the LA module must not pull in torch/transformers/bitsandbytes.

    Run in a clean subprocess so we get a pristine ``sys.modules``.
    """
    script = textwrap.dedent(
        """
        import sys
        import src.engine.backends.locateanything  # noqa: F401
        import src.engine.backends.locateanything_worker  # noqa: F401
        heavy = [m for m in ("torch", "transformers", "bitsandbytes")
                 if m in sys.modules]
        print(",".join(heavy))
        """
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"heavy deps leaked at import time: {leaked}"


# ── probe() ───────────────────────────────────────────────────────────────


def _make_backend():
    from src.engine.backends.locateanything import LocateAnythingBackend

    return LocateAnythingBackend()


def test_probe_reports_missing_dependency(monkeypatch):
    import src.engine.backends.locateanything as la

    def fake_version(pkg):
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError(pkg)

    monkeypatch.setattr(la, "version", fake_version)

    probe = _make_backend().probe()
    assert probe.available is False
    assert "缺少可选依赖" in probe.message
    assert probe.metadata["missing_packages"]


def test_probe_reports_missing_weights(monkeypatch):
    import src.engine.backends.locateanything as la

    monkeypatch.setattr(la, "version", lambda pkg: "4.57.0")
    monkeypatch.setattr(la, "_hf_cache_has_model", lambda: False)

    probe = _make_backend().probe()
    assert probe.available is False
    assert "缓存" in probe.message
    assert probe.metadata["weights_ready"] is False


def test_probe_available_when_deps_and_weights_present(monkeypatch):
    import src.engine.backends.locateanything as la

    monkeypatch.setattr(la, "version", lambda pkg: "4.57.0")
    monkeypatch.setattr(la, "_hf_cache_has_model", lambda: True)

    probe = _make_backend().probe()
    assert probe.available is True
    assert probe.backend_id == "locateanything"
    assert probe.metadata["weights_ready"] is True


def test_probe_runtime_is_subprocess(monkeypatch):
    import src.engine.backends.locateanything as la

    monkeypatch.setattr(la, "version", lambda pkg: "4.57.0")
    monkeypatch.setattr(la, "_hf_cache_has_model", lambda: True)
    probe = _make_backend().probe()
    assert probe.runtime == "subprocess"


def test_probe_never_raises_on_unexpected_version_error(monkeypatch):
    import src.engine.backends.locateanything as la

    def boom(pkg):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(la, "version", boom)
    probe = _make_backend().probe()
    assert probe.available is False


# ── infer_model_format / create_trainer ──────────────────────────────────


def test_infer_model_format_is_vlm():
    assert _make_backend().infer_model_format("whatever") == "vlm-hf"


def test_create_trainer_raises_unavailable():
    with pytest.raises(BackendUnavailableError):
        _make_backend().create_trainer()


# ── preflight() — cheap nvidia-smi parse, NO torch ────────────────────────


def _patch_smi(monkeypatch, ok, total_gb, free_gb, message=""):
    import src.engine.backends.locateanything as la

    monkeypatch.setattr(
        la,
        "_query_nvidia_smi_vram",
        lambda: (ok, total_gb, free_gb, message),
    )


def test_preflight_does_not_import_torch(monkeypatch):
    """Calling preflight must never *newly* import torch.

    (Other tests in the suite may have already loaded torch into the shared
    process — the authoritative "torch never enters the process" guard runs in a
    clean subprocess, see test_importing_module_does_not_import_torch_or_transformers.
    Here we assert the *call itself* introduces no torch import.)
    """
    _patch_smi(monkeypatch, True, 8.0, 6.0)
    torch_present_before = "torch" in sys.modules
    _make_backend().preflight()
    if not torch_present_before:
        assert "torch" not in sys.modules


def test_preflight_blocks_without_gpu(monkeypatch):
    _patch_smi(monkeypatch, False, 0.0, 0.0, "未检测到 nvidia-smi / 可用 GPU")
    result = _make_backend().preflight()
    assert result["ok"] is False
    assert "GPU" in result["message"]


def test_preflight_blocks_low_total_vram(monkeypatch):
    _patch_smi(monkeypatch, True, 4.0, 4.0)
    result = _make_backend().preflight()
    assert result["ok"] is False
    assert "总显存" in result["message"]


def test_preflight_blocks_low_free_vram(monkeypatch):
    _patch_smi(monkeypatch, True, 8.0, 2.0)
    result = _make_backend().preflight()
    assert result["ok"] is False
    assert "空闲" in result["message"]


def test_preflight_passes_when_vram_ok(monkeypatch):
    _patch_smi(monkeypatch, True, 8.0, 6.0)
    result = _make_backend().preflight()
    assert result["ok"] is True
    assert result["free_gb"] == pytest.approx(6.0, abs=0.01)


def test_preflight_free_vram_threshold_is_5gb(monkeypatch):
    # 4.5GB free must be blocked by the 5.0 threshold (shared display+inference).
    _patch_smi(monkeypatch, True, 8.0, 4.5)
    result = _make_backend().preflight()
    assert result["ok"] is False
    assert "空闲" in result["message"]


def test_preflight_pass_message_includes_shared_gpu_hint(monkeypatch):
    _patch_smi(monkeypatch, True, 8.0, 6.0)
    result = _make_backend().preflight()
    assert result["ok"] is True
    assert "共用同一块 GPU" in result["message"]


# ── nvidia-smi parsing (no real nvidia-smi on CI) ─────────────────────────


def test_query_nvidia_smi_parses_csv(monkeypatch):
    import src.engine.backends.locateanything as la

    class _Proc:
        returncode = 0
        stdout = "8192, 6144\n"

    monkeypatch.setattr(la.subprocess, "run", lambda *a, **k: _Proc())
    ok, total, free, msg = la._query_nvidia_smi_vram()
    assert ok is True
    assert total == pytest.approx(8.0, abs=0.01)
    assert free == pytest.approx(6.0, abs=0.01)


def test_query_nvidia_smi_picks_most_free_gpu(monkeypatch):
    import src.engine.backends.locateanything as la

    class _Proc:
        returncode = 0
        stdout = "8192, 1024\n24576, 20480\n"

    monkeypatch.setattr(la.subprocess, "run", lambda *a, **k: _Proc())
    ok, total, free, msg = la._query_nvidia_smi_vram()
    assert ok is True
    # The GPU with the most free memory is picked.
    assert total == pytest.approx(24.0, abs=0.01)
    assert free == pytest.approx(20.0, abs=0.01)


def test_query_nvidia_smi_missing_binary(monkeypatch):
    import src.engine.backends.locateanything as la

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(la.subprocess, "run", boom)
    ok, total, free, msg = la._query_nvidia_smi_vram()
    assert ok is False
    assert "GPU" in msg


# ── load_runtime spawns a subprocess (mocked; no real model) ──────────────


class _FakeReadyWorker:
    """Captures the _WorkerProcess constructor and simulates a ready handshake."""

    def __init__(self, proc):
        self.proc = proc
        self.ready_called = False
        self.terminated = False

    def wait_until_ready(self, progress_cb=None, timeout=None):
        self.ready_called = True
        if progress_cb:
            progress_cb("正在加载模型 (4-bit NF4)，约需 40 秒…")

    def terminate(self):
        self.terminated = True


def test_load_runtime_spawns_subprocess_and_returns_client(monkeypatch):
    import src.engine.backends.locateanything as la

    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()  # opaque proc handle; _WorkerProcess is faked too

    created_workers = []

    def fake_worker_cls(proc):
        w = _FakeReadyWorker(proc)
        created_workers.append(w)
        return w

    monkeypatch.setattr(la.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(la, "_WorkerProcess", fake_worker_cls)

    torch_present_before = "torch" in sys.modules
    progress = []
    pred = _make_backend().load_runtime(progress_cb=progress.append)

    # Spawned the same interpreter running the worker script.
    assert captured["args"][0] == sys.executable
    assert captured["args"][-1].endswith("locateanything_worker.py")
    # Client predictor returned, holding the ready worker.
    from src.engine.backends.locateanything import LocateAnythingPredictor

    assert isinstance(pred, LocateAnythingPredictor)
    assert created_workers[0].ready_called is True
    assert any("加载" in p for p in progress)
    # The (mocked) spawn introduces no NEW torch import into the GUI process.
    if not torch_present_before:
        assert "torch" not in sys.modules


def test_load_runtime_terminates_worker_on_startup_failure(monkeypatch):
    import src.engine.backends.locateanything as la

    monkeypatch.setattr(la.subprocess, "Popen", lambda *a, **k: object())

    class _FailWorker(_FakeReadyWorker):
        def wait_until_ready(self, progress_cb=None, timeout=None):
            raise BackendUnavailableError("显存预检未通过")

    created = []

    def cls(proc):
        w = _FailWorker(proc)
        created.append(w)
        return w

    monkeypatch.setattr(la, "_WorkerProcess", cls)

    with pytest.raises(BackendUnavailableError):
        _make_backend().load_runtime()
    # The worker was terminated on failure (no leaked subprocess).
    assert created[0].terminated is True
