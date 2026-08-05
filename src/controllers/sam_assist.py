"""SAM-assist controller — lazy load/encode state machine on a QThread.

``SamAssistController`` bridges the canvas SAM tool and the Qt-free
``engine.sam_assist.SamAssistant``:

- **Lazy load**: the model is built only on first activation (~2.6 s cold on
  CPU), on a background ``SamPrepareWorker`` QThread with status signals.
- **Per-image encode state machine** (idle → encoding → ready): activating
  the tool or switching images (re-)encodes the focused image off the GUI
  thread (~1 s CPU). While a worker is in flight, new encode requests are
  coalesced into ``_pending`` (only the latest matters) and prompt clicks
  are answered with a Chinese "please wait" status — never a predictor call
  (a promptless / mid-encode call would be wrong or crash).
- **Decode is synchronous**: one prompt ≈ 65 ms CPU, run directly on the GUI
  thread when the state is ready (the worker has finished, so the assistant
  is not shared across threads).

The signal face is data-only (no ultralytics types): polygons/bboxes are
plain lists/tuples, mirroring the worker-in-controller convention of
``controllers/video_import.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from src.engine.sam_assist import (
    BoxPrompt,
    PointPrompt,
    SamAssistant,
    SamAssistError,
)

logger = logging.getLogger(__name__)


class SamPrepareWorker(QThread):
    """Loads the model (idempotent) and encodes one image off the GUI thread.

    Signals:
        ready(): load + encode completed for ``image_path`` (or load-only
            when ``image_path`` is None).
        error(str): Chinese failure message (weights missing, encode failed).
    """

    ready = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        assistant: SamAssistant,
        image_path: Path | None,
        image_size: tuple[int, int] | None,
        parent=None,
    ):
        super().__init__(parent)
        self._assistant = assistant
        self.image_path = image_path
        self.image_size = image_size

    def run(self) -> None:
        try:
            self._assistant.load()
            if self.image_path is not None and self.image_size is not None:
                self._assistant.encode(self.image_path, self.image_size)
            self.ready.emit()
        except SamAssistError as e:
            self.error.emit(str(e))
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently
            # kill the thread.
            logger.exception("SAM prepare worker failed")
            self.error.emit(f"SAM 初始化失败：{e}")


class SamAssistController(QObject):
    """Owns the SAM assistant lifecycle + encode state machine.

    Signals (data-only face — the view renders them):
        load_started(): first-time model build began (show "加载中" status).
        load_failed(str): load/encode failed with a Chinese message.
        encode_started(object): Path — encoding of an image began.
        encode_ready(object): Path — that image is ready for prompts.
        mask_ready(object, object): (polygon, bbox) — normalized open vertex
            list + derived (cx, cy, w, h); invariant bbox ==
            bbox_from_polygon(polygon).
        status_message(str): transient Chinese status text.
    """

    load_started = pyqtSignal()
    load_failed = pyqtSignal(str)
    encode_started = pyqtSignal(object)
    encode_ready = pyqtSignal(object)
    mask_ready = pyqtSignal(object, object)
    status_message = pyqtSignal(str)

    def __init__(self, assistant: SamAssistant | None = None, parent=None):
        super().__init__(parent)
        self._assistant = assistant or SamAssistant()
        self._worker: SamPrepareWorker | None = None
        # Path whose embedding is currently usable for prompts.
        self._ready_path: Path | None = None
        # Latest coalesced encode request queued behind the in-flight worker.
        self._pending: tuple[Path, tuple[int, int]] | None = None

    # ── State ─────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        """True while a load/encode worker is in flight."""
        return self._worker is not None

    @property
    def ready_path(self) -> Path | None:
        return self._ready_path

    # ── Activation / image switching ──────────────────────────

    def activate(self, image_path: Path | None, image_size: tuple[int, int] | None) -> None:
        """SAM tool armed: lazily load the model and encode the focused image.

        ``image_path=None`` (no image focused yet) loads the model only.
        """
        self._ensure_encoded(image_path, image_size)

    def set_image(self, image_path: Path | None, image_size: tuple[int, int] | None) -> None:
        """Focused image changed while the tool is active: pre-encode it."""
        self._ensure_encoded(image_path, image_size)

    def _ensure_encoded(
        self, image_path: Path | None, image_size: tuple[int, int] | None
    ) -> None:
        if image_path is not None and image_path == self._ready_path:
            return  # already encoded — nothing to do
        if image_path is not None and (
            image_size is None or image_size[0] <= 0 or image_size[1] <= 0
        ):
            self.status_message.emit(f"无法读取图像尺寸: {image_path.name}")
            return
        if self._worker is not None:
            # Coalesce: only the latest request matters (image switches can
            # outrun a ~1 s encode).
            if image_path is not None:
                self._pending = (image_path, image_size)
            return
        self._start_worker(image_path, image_size)

    def _start_worker(
        self, image_path: Path | None, image_size: tuple[int, int] | None
    ) -> None:
        self._ready_path = None  # stale embedding must not serve prompts
        if not self._assistant.is_loaded:
            self.load_started.emit()
            self.status_message.emit("正在加载 SAM 模型…")
        if image_path is not None:
            self.encode_started.emit(image_path)
        worker = SamPrepareWorker(self._assistant, image_path, image_size)
        worker.ready.connect(self._on_worker_ready)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    # ── Prompts (synchronous decode, GUI thread) ──────────────

    def request_point(self, x: float, y: float) -> None:
        """Single positive click at normalized (x, y)."""
        self._run_prompt(PointPrompt(x=x, y=y))

    def request_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Box prompt with normalized corners."""
        self._run_prompt(BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2))

    def _run_prompt(self, prompt) -> None:
        if self._worker is not None or self._ready_path is None:
            # Never touch the predictor mid-load/encode (thread ownership) and
            # never let an un-encoded state degenerate into a promptless call.
            self.status_message.emit("SAM 正在准备中，请稍候…")
            return
        try:
            result = self._assistant.predict(prompt)
        except SamAssistError as e:
            self.status_message.emit(str(e))
            return
        if result is None:
            self.status_message.emit("SAM 未能生成有效多边形，请调整点击位置重试")
            return
        self.mask_ready.emit(result.polygon, result.bbox)

    # ── Worker slots (queued → main thread) ───────────────────

    def _on_worker_ready(self) -> None:
        if self._worker is not None:
            self._ready_path = self._worker.image_path
            if self._ready_path is not None:
                self.encode_ready.emit(self._ready_path)
                self.status_message.emit("SAM 就绪：单击或拉框生成多边形")

    def _on_worker_error(self, message: str) -> None:
        self._ready_path = None
        self._pending = None
        self.load_failed.emit(message)
        self.status_message.emit(message)

    def _on_worker_finished(self) -> None:
        """QThread finished (always fires): drop the ref, drain the queue."""
        self._worker = None
        if self._pending is not None:
            path, size = self._pending
            self._pending = None
            if path != self._ready_path:
                self._start_worker(path, size)

    # ── Teardown ──────────────────────────────────────────────

    def shutdown(self, timeout_ms: int = 30000) -> None:
        """Wait out any in-flight worker and release the model (project
        switch / app close). Safe to call repeatedly."""
        self._pending = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(timeout_ms)
        self._worker = None
        self._ready_path = None
        self._assistant.release()
