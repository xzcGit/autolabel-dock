"""Image loading utilities with caching."""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImageReader, QImageIOHandler

logger = logging.getLogger(__name__)


def load_pixmap(
    path: Path | str, max_size: int | None = None
) -> QPixmap | None:
    """Load image as QPixmap. Returns None if file doesn't exist or fails to load."""
    path = Path(path)
    if not path.exists():
        return None

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return None

    pixmap = QPixmap.fromImage(image)
    if max_size and (pixmap.width() > max_size or pixmap.height() > max_size):
        pixmap = pixmap.scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return pixmap


def get_image_size(path: Path | str) -> tuple[int, int]:
    """Get image dimensions (width, height) without fully loading. Returns (0, 0) on failure.

    Honors EXIF orientation so the reported size matches what ``load_pixmap``
    displays. NOTE: ``QImageReader.size()`` returns the RAW (pre-transform) size
    even with ``setAutoTransform(True)`` (verified on Qt 5.15), so for 90°/270°
    orientations we must swap W/H manually based on ``transformation()``.
    """
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid():
        w, h = size.width(), size.height()
        if int(reader.transformation()) & int(QImageIOHandler.TransformationRotate90):
            w, h = h, w
        return w, h
    return 0, 0


def _pixmap_bytes(pixmap: QPixmap) -> int:
    """Estimate memory usage of a QPixmap in bytes."""
    return pixmap.width() * pixmap.height() * 4  # RGBA


class ImageCache:
    """LRU cache for loaded QPixmaps with memory limit.

    Eviction is based on both item count and total memory usage.
    """

    def __init__(self, max_count: int = 16, max_memory_mb: float = 512.0):
        self._max_count = max_count
        self._max_memory = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._total_bytes: int = 0

    def get(self, path: Path | str, pixmap_max_size: int | None = None) -> QPixmap | None:
        """Get pixmap from cache, loading if necessary."""
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        pixmap = load_pixmap(path, max_size=pixmap_max_size)
        if pixmap is None:
            return None

        self._put(key, pixmap)
        return pixmap

    def preload(self, paths: list[Path], pixmap_max_size: int | None = None) -> None:
        """Preload a list of image paths into cache (skips already cached)."""
        for p in paths:
            key = str(p)
            if key not in self._cache:
                pixmap = load_pixmap(p, max_size=pixmap_max_size)
                if pixmap is not None:
                    self._put(key, pixmap)

    def _put(self, key: str, pixmap: QPixmap) -> None:
        """Add pixmap to cache, evicting oldest if limits exceeded."""
        size = _pixmap_bytes(pixmap)
        self._cache[key] = pixmap
        self._total_bytes += size
        self._evict()

    def _evict(self) -> None:
        """Evict oldest entries until both count and memory limits are satisfied."""
        while self._cache and (
            len(self._cache) > self._max_count or self._total_bytes > self._max_memory
        ):
            _, evicted = self._cache.popitem(last=False)
            self._total_bytes -= _pixmap_bytes(evicted)

    @property
    def memory_usage_mb(self) -> float:
        """Current estimated memory usage in MB."""
        return self._total_bytes / (1024 * 1024)

    @property
    def size(self) -> int:
        """Number of cached pixmaps."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached pixmaps."""
        self._cache.clear()
        self._total_bytes = 0

    def invalidate(self, path: Path | str) -> None:
        """Remove a specific path from cache."""
        key = str(path)
        pixmap = self._cache.pop(key, None)
        if pixmap is not None:
            self._total_bytes -= _pixmap_bytes(pixmap)
