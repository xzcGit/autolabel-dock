"""LocateAnything enable/disable controller.

Bridges the LocateAnything UI bar to the optional backend, orchestrating the
four-tier lazy unlock (probe → preflight → background load) and the VRAM
lifecycle (unload the resident YOLO predictor before loading LA).

The loaded LA predictor is installed as ``ModelController._predictor`` via
``ModelController.set_predictor``, so the existing single-image / batch
auto-label flows in ``app.py`` work **unchanged** — they only read
``ModelController.predictor`` and call ``predict`` / ``predict_with_size``.
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, pyqtSignal

from src.engine.backends import get_backend
from src.engine.backends.base import BackendError, BackendProbe
from src.utils.workers import LocateAnythingLoadWorker

logger = logging.getLogger(__name__)

BACKEND_ID = "locateanything"


class LocateAnythingController(QObject):
    """Owns the LA enable lifecycle. Holds a reference to ModelController.

    Signals:
        probe_done(BackendProbe): result of the cheap availability probe.
        preflight_blocked(str): preflight failed — message has guidance.
        load_progress(str): heavy-load status updates.
        enabled(): LA runtime is loaded and installed as the active predictor.
        disabled(): LA runtime unloaded; no predictor active.
        failed(str): load failed (background thread).
    """

    probe_done = pyqtSignal(object)        # BackendProbe
    preflight_blocked = pyqtSignal(str)
    load_progress = pyqtSignal(str)
    enabled = pyqtSignal()
    disabled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, model_controller, parent=None):
        super().__init__(parent)
        self._model_ctrl = model_controller
        self._backend = None
        self._load_worker: LocateAnythingLoadWorker | None = None
        self._active: bool = False

    @property
    def is_active(self) -> bool:
        return self._active

    def _resolve_backend(self):
        if self._backend is None:
            self._backend = get_backend(BACKEND_ID)
        return self._backend

    # ── Enable pipeline: probe → preflight → background load ──────────────

    def begin_enable(self) -> None:
        """Run probe + preflight synchronously, then kick off background load.

        Emits ``probe_done`` (always), ``preflight_blocked`` if gated, then
        ``load_progress`` updates and finally ``enabled`` / ``failed``.
        """
        try:
            backend = self._resolve_backend()
        except BackendError as exc:
            self.failed.emit(str(exc))
            return

        # Tier 1: probe (cheap, no heavy import, never raises).
        probe: BackendProbe = backend.probe()
        self.probe_done.emit(probe)
        if not probe.available:
            self.preflight_blocked.emit(probe.message or "LocateAnything 不可用")
            return

        # Tier 2: preflight — unload the resident YOLO first so VRAM check sees
        # the post-unload free memory, then validate GPU / VRAM.
        self._model_ctrl.unload()
        result = backend.preflight()
        if not result.get("ok"):
            self.preflight_blocked.emit(
                result.get("message", "显存预检未通过")
            )
            return

        # Tier 3: heavy load on a background thread.
        self.load_progress.emit("预检通过，正在后台加载模型…")
        self._load_worker = LocateAnythingLoadWorker(backend)
        self._load_worker.progress.connect(self.load_progress.emit)
        self._load_worker.loaded.connect(self._on_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.finished.connect(self._on_worker_finished)
        self._load_worker.start()

    def _on_loaded(self, predictor) -> None:
        self._model_ctrl.set_predictor(predictor)
        self._active = True
        logger.info("LocateAnything runtime enabled")
        self.enabled.emit()

    def _on_load_error(self, message: str) -> None:
        self._active = False
        logger.error("LocateAnything load failed: %s", message)
        self.failed.emit(message)

    def _on_worker_finished(self) -> None:
        self._load_worker = None

    # ── Query injection (carries natural language to the predictor) ───────

    def set_query(self, prompt: str, target_class: str | None) -> None:
        """Forward the natural-language prompt + optional target class to the
        active LA predictor. No-op if LA is not the active predictor."""
        predictor = self._model_ctrl.predictor
        set_query = getattr(predictor, "set_query", None)
        if callable(set_query):
            set_query(prompt, target_class)

    # ── Disable ───────────────────────────────────────────────────────────

    def disable(self) -> None:
        """Unload the LA runtime and return to a no-predictor state."""
        if not self._active and self._model_ctrl.predictor is None:
            return
        self._model_ctrl.unload()
        self._active = False
        logger.info("LocateAnything runtime disabled")
        self.disabled.emit()
