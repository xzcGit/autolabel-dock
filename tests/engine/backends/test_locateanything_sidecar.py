"""Sidecar IPC protocol + worker entry-script contract tests.

No real model / torch ever loads. We drive ``_WorkerProcess`` against a fake
worker subprocess (a tiny pure-Python script that replays JSON frames over
stdin/stdout) and assert the startup handshake, inference round-trip, error
frames, EOF/crash handling, and graceful shutdown. We also import the worker
entry module to check its contract (stdlib-only top level) without invoking the
heavy load.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.engine.backends.base import BackendUnavailableError
from src.engine.backends.locateanything import _WorkerProcess

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Fake worker subprocess helpers ─────────────────────────────────────────


def _spawn_fake_worker(script_body: str) -> _WorkerProcess:
    """Spawn a tiny pure-Python worker script, wrap it in _WorkerProcess."""
    script = textwrap.dedent(script_body)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
    )
    return _WorkerProcess(proc)


# A fake worker that: emits two status lines then ready, then echoes one
# canned inference response per request, and exits on shutdown.
_FAKE_READY_WORKER = """
import json, sys
def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
send({"status": "正在导入依赖…"})
send({"status": "正在加载模型 (4-bit NF4)…"})
send({"ready": True})
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("cmd") == "shutdown":
        break
    send({"raw_text": "<ref>cat</ref><box><100><200><300><400></box>",
          "width": 640, "height": 480})
"""


# ── Startup handshake ───────────────────────────────────────────────────────


def test_wait_until_ready_collects_status_then_ready():
    worker = _spawn_fake_worker(_FAKE_READY_WORKER)
    try:
        progress = []
        worker.wait_until_ready(progress_cb=progress.append, timeout=30)
        assert any("导入依赖" in p for p in progress)
        assert any("加载模型" in p for p in progress)
    finally:
        worker.terminate()


def test_wait_until_ready_raises_on_preflight_failed():
    worker = _spawn_fake_worker(
        """
        import json, sys
        sys.stdout.write(json.dumps({"preflight_failed": "未检测到 GPU"}) + "\\n")
        sys.stdout.flush()
        """
    )
    try:
        with pytest.raises(BackendUnavailableError) as exc:
            worker.wait_until_ready(timeout=30)
        assert "GPU" in str(exc.value)
    finally:
        worker.terminate()


def test_wait_until_ready_raises_on_fatal():
    worker = _spawn_fake_worker(
        """
        import json, sys
        sys.stdout.write(json.dumps({"fatal": "模型加载失败: boom"}) + "\\n")
        sys.stdout.flush()
        """
    )
    try:
        with pytest.raises(BackendUnavailableError) as exc:
            worker.wait_until_ready(timeout=30)
        assert "加载失败" in str(exc.value)
    finally:
        worker.terminate()


def test_wait_until_ready_raises_on_early_exit():
    # Worker exits immediately without emitting ready → EOF before ready.
    worker = _spawn_fake_worker("import sys; sys.exit(3)")
    try:
        with pytest.raises(BackendUnavailableError) as exc:
            worker.wait_until_ready(timeout=30)
        assert "意外退出" in str(exc.value)
    finally:
        worker.terminate()


# ── Inference round-trip ────────────────────────────────────────────────────


def test_infer_round_trip():
    worker = _spawn_fake_worker(_FAKE_READY_WORKER)
    try:
        worker.wait_until_ready(timeout=30)
        raw, w, h = worker.infer("img.jpg", "prompt", timeout=30)
        assert raw == "<ref>cat</ref><box><100><200><300><400></box>"
        assert (w, h) == (640, 480)
    finally:
        worker.terminate()


def test_infer_error_frame_raises_runtime_error():
    worker = _spawn_fake_worker(
        """
        import json, sys
        def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        send({"ready": True})
        for line in sys.stdin:
            if not line.strip():
                continue
            send({"error": "推理显存不足 (CUDA OOM)"})
        """
    )
    try:
        worker.wait_until_ready(timeout=30)
        with pytest.raises(RuntimeError) as exc:
            worker.infer("img.jpg", "prompt", timeout=30)
        assert "显存不足" in str(exc.value)
    finally:
        worker.terminate()


def test_infer_after_worker_crash_raises_not_hang():
    """Worker dies mid-serve → client.infer raises, never hangs the GUI."""
    worker = _spawn_fake_worker(
        """
        import json, sys
        def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        send({"ready": True})
        # Read one request then exit hard (simulate crash) without responding.
        for line in sys.stdin:
            if line.strip():
                sys.exit(9)
        """
    )
    try:
        worker.wait_until_ready(timeout=30)
        with pytest.raises(RuntimeError) as exc:
            worker.infer("img.jpg", "prompt", timeout=30)
        assert "退出" in str(exc.value) or "超时" in str(exc.value)
    finally:
        worker.terminate()


def test_infer_when_already_exited_raises():
    worker = _spawn_fake_worker(
        """
        import json, sys
        sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
        sys.exit(0)
        """
    )
    try:
        worker.wait_until_ready(timeout=30)
        worker._proc.wait(timeout=10)
        with pytest.raises(RuntimeError):
            worker.infer("img.jpg", "prompt", timeout=30)
    finally:
        worker.terminate()


# ── Lifecycle / teardown ────────────────────────────────────────────────────


def test_terminate_is_graceful_then_idempotent():
    worker = _spawn_fake_worker(_FAKE_READY_WORKER)
    worker.wait_until_ready(timeout=30)
    assert worker.is_alive() is True
    worker.terminate()
    assert worker.is_alive() is False
    # Idempotent.
    worker.terminate()


def test_terminate_kills_unresponsive_worker():
    # Worker ignores shutdown (busy loop reading nothing) → must be killed.
    worker = _spawn_fake_worker(
        """
        import json, sys, time
        sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
        # Ignore stdin entirely; spin so graceful shutdown can't work.
        while True:
            time.sleep(0.5)
        """
    )
    worker.wait_until_ready(timeout=30)
    # Shorten the grace period so the test doesn't wait 10s before kill.
    import src.engine.backends.locateanything as la
    old = la.WORKER_SHUTDOWN_TIMEOUT_S
    la.WORKER_SHUTDOWN_TIMEOUT_S = 1.0
    try:
        worker.terminate()
        assert worker.is_alive() is False
    finally:
        la.WORKER_SHUTDOWN_TIMEOUT_S = old


# ── Worker entry-script contract (no heavy load triggered) ─────────────────


def test_worker_module_top_level_is_stdlib_only():
    """Importing the worker module must not pull in torch/transformers."""
    script = textwrap.dedent(
        """
        import sys
        import src.engine.backends.locateanything_worker as w  # noqa: F401
        heavy = [m for m in ("torch", "transformers", "bitsandbytes", "PIL")
                 if m in sys.modules]
        print(",".join(heavy))
        # Sanity: the contract constants + main() are present.
        assert callable(w.main)
        assert w.MAX_NEW_TOKENS == 512
        # Regression guard: generation must SAMPLE, not run greedy. This model's
        # custom generate() keys the sampling switch off temperature (0 → greedy)
        # and ignores do_sample; pure greedy runs away to ~85 junk boxes.
        assert w.GEN_TEMPERATURE > 0, "greedy decoding regresses to ~85 junk boxes"
        assert 0 < w.GEN_TOP_P <= 1.0
        """
    )
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy deps leaked: {result.stdout!r}"


def test_worker_preflight_blocks_without_cuda():
    """The worker's _preflight gate returns ok=False with no CUDA (fake torch)."""
    import src.engine.backends.locateanything_worker as w

    class _FakeCuda:
        def is_available(self):
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    result = w._preflight(_FakeTorch())
    assert result["ok"] is False
    assert "GPU" in result["message"]


def test_worker_preflight_passes_with_enough_vram():
    import src.engine.backends.locateanything_worker as w

    class _FakeCuda:
        def is_available(self):
            return True

        def mem_get_info(self):
            return (6 * 1024 ** 3, 8 * 1024 ** 3)

    class _FakeTorch:
        cuda = _FakeCuda()

    result = w._preflight(_FakeTorch())
    assert result["ok"] is True
    assert result["free_gb"] == pytest.approx(6.0, abs=0.01)


def test_worker_readable_error_maps_oom():
    import src.engine.backends.locateanything_worker as w

    class _OOM(RuntimeError):
        pass

    _OOM.__name__ = "OutOfMemoryError"
    msg = w._readable_inference_error(_OOM("CUDA error"))
    assert "显存不足" in msg


def test_worker_downscale_shrinks_large_image():
    import src.engine.backends.locateanything_worker as w

    class _Img:
        def __init__(self, size):
            self.size = size
            self.resized = None

        def resize(self, new_size, resample=None):
            self.resized = new_size
            return _Img(new_size)

    # Pure long-edge behavior: disable the area cap so a wide image is bounded
    # only by its longest edge (1280 → 640 keeps the 2:1 ratio).
    out = w._downscale_for_inference(
        _Img((4000, 2000)), max_long_edge=1280, max_pixels=10 ** 12
    )
    assert max(out.size) == 1280
    assert out.size == (1280, 640)


def test_worker_downscale_caps_by_area():
    """A near-square image UNDER the long-edge cap but OVER the area cap must be
    shrunk by area — this is the case that OOM'd (74×74 patches) before the area
    ceiling existed; the long-edge cap alone left it untouched."""
    import src.engine.backends.locateanything_worker as w

    class _Img:
        def __init__(self, size):
            self.size = size
            self.resized = None

        def resize(self, new_size, resample=None):
            self.resized = new_size
            return _Img(new_size)

    # 1024×1011 = 1,035,264 px: long edge 1024 ≤ cap, but area > 750k.
    src = _Img((1024, 1011))
    out = w._downscale_for_inference(src, max_long_edge=1024, max_pixels=750_000)
    assert out is not src  # was resized
    # Under the area ceiling, allowing a few px of rounding slop: round() can nudge
    # each dimension up by <1px, so the product can sit a hair over the budget. The
    # cap is a VRAM guardrail (≈2G headroom), not an exact pixel contract.
    assert out.size[0] * out.size[1] <= 750_000 * 1.01
    assert max(out.size) <= 1024  # still within the long-edge cap
    # Aspect ratio preserved within rounding.
    assert abs(out.size[0] / out.size[1] - 1024 / 1011) < 0.01


def test_worker_downscale_leaves_small_image():
    import src.engine.backends.locateanything_worker as w

    class _Img:
        def __init__(self, size):
            self.size = size

        def resize(self, *a, **k):  # pragma: no cover - should not be called
            raise AssertionError("should not resize a small image")

    small = _Img((640, 480))
    assert w._downscale_for_inference(small, max_long_edge=1280) is small


# ── End-to-end GUI-process zero-torch guard (clean subprocess) ─────────────


def test_enabled_la_session_imports_no_torch_in_gui_process():
    """Drive the controller to the 'enabled' state with a FAKE sidecar (no real
    subprocess / model) and assert torch/transformers/bitsandbytes never enter
    the GUI process. Runs in a pristine interpreter so the check is reliable.
    """
    import os

    script = textwrap.dedent(
        """
        import sys
        # Import the full LA stack the GUI process touches.
        import src.engine.backends.locateanything as la
        import src.engine.backends.locateanything_worker  # noqa: F401
        import src.controllers.model  # noqa: F401
        import src.controllers.locateanything as la_ctrl

        # Fake backend whose load_runtime returns a client predictor wired to a
        # fake worker — no subprocess, no torch.
        from src.engine.backends.locateanything import LocateAnythingPredictor
        from src.engine.backends.base import BackendProbe

        class FakeWorker:
            def infer(self, image_path, prompt, timeout=None):
                return ("<ref>cat</ref><box><1><2><3><4></box>", 10, 10)
            def terminate(self):
                pass

        class FakeBackend:
            backend_id = "locateanything"
            display_name = "LA"
            def probe(self):
                return BackendProbe(backend_id="locateanything",
                                    display_name="LA", available=True)
            def preflight(self):
                return {"ok": True, "message": "", "total_gb": 8.0, "free_gb": 6.0}
            def load_runtime(self, progress_cb=None):
                return LocateAnythingPredictor(worker=FakeWorker())

        # Minimal stand-ins so we don't need a QApplication / QThread.
        class FakeModelCtrl:
            def __init__(self):
                self._p = None
            @property
            def predictor(self):
                return self._p
            def unload(self):
                if self._p is not None and hasattr(self._p, "release"):
                    self._p.release()
                self._p = None
            def set_predictor(self, p):
                self._p = p

        from PyQt5.QtCore import QObject, pyqtSignal
        class SyncWorker(QObject):
            progress = pyqtSignal(str)
            loaded = pyqtSignal(object)
            error = pyqtSignal(str)
            finished = pyqtSignal()
            def __init__(self, backend, parent=None):
                super().__init__(parent)
                self._b = backend
            def start(self):
                p = self._b.load_runtime(progress_cb=self.progress.emit)
                self.loaded.emit(p)
                self.finished.emit()

        la_ctrl.get_backend = lambda _id: FakeBackend()
        la_ctrl.LocateAnythingLoadWorker = SyncWorker

        mc = FakeModelCtrl()
        ctrl = la_ctrl.LocateAnythingController(mc)
        ctrl.begin_enable()
        assert ctrl.is_active is True
        # Run an inference through the installed client predictor.
        anns = mc.predictor.predict("x.jpg", project_classes=["cat"])
        assert len(anns) == 1
        # And tear down.
        ctrl.disable()

        heavy = [m for m in ("torch", "transformers", "bitsandbytes")
                 if m in sys.modules]
        print(",".join(heavy))
        """
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
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
    assert leaked == "", f"heavy deps leaked in GUI process: {leaked}"
