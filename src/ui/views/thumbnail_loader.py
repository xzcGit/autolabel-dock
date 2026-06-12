"""Async thumbnail loader. One worker thread + FIFO queue."""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

from PyQt5.QtCore import Qt, QMutex, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)


def _default_load_pixmap(path: Path, size: QSize) -> QPixmap | None:
    img = QImage(str(path))
    if img.isNull():
        return None
    scaled = img.scaled(
        size.width(), size.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
    )
    return QPixmap.fromImage(scaled)


class ThumbnailLoader(QThread):
    """Background loader. Emits (path, pixmap) for each completed item."""

    loaded = pyqtSignal(object, object)  # Path, QPixmap

    def __init__(self, loader_fn=None, parent=None):
        super().__init__(parent)
        self._loader_fn = loader_fn or _default_load_pixmap
        self._queue: deque[tuple[Path, QSize]] = deque()
        self._mu = QMutex()
        self._stop = False

    def enqueue(self, path: Path, size: QSize) -> None:
        self._mu.lock()
        try:
            self._queue.append((path, size))
        finally:
            self._mu.unlock()
        if not self.isRunning():
            self.start()

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            self._mu.lock()
            try:
                if not self._queue:
                    return
                path, size = self._queue.popleft()
            finally:
                self._mu.unlock()
            try:
                pix = self._loader_fn(path, size)
            except Exception as e:
                logger.warning("Thumbnail load failed for %s: %s", path, e)
                pix = None
            if pix is not None:
                self.loaded.emit(path, pix)
