"""Tests for image loading utility."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_test_image(path: Path, width: int = 100, height: int = 80) -> None:
    """Create a minimal valid PNG file for testing."""
    from PyQt5.QtGui import QImage, QColor
    from PyQt5.QtCore import Qt

    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(Qt.red))
    img.save(str(path), "PNG")


class TestLoadPixmap:
    def test_load_valid_image(self, qapp, tmp_path):
        from src.utils.image import load_pixmap

        img_path = tmp_path / "test.png"
        _make_test_image(img_path)
        pixmap = load_pixmap(img_path)
        assert pixmap is not None
        assert not pixmap.isNull()
        assert pixmap.width() == 100
        assert pixmap.height() == 80

    def test_load_nonexistent_returns_none(self, qapp, tmp_path):
        from src.utils.image import load_pixmap

        pixmap = load_pixmap(tmp_path / "nope.png")
        assert pixmap is None

    def test_load_with_max_size(self, qapp, tmp_path):
        from src.utils.image import load_pixmap

        img_path = tmp_path / "big.png"
        _make_test_image(img_path, 2000, 1000)
        pixmap = load_pixmap(img_path, max_size=500)
        assert pixmap is not None
        assert pixmap.width() <= 500
        assert pixmap.height() <= 500


class TestImageCache:
    def test_cache_returns_same_object(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        img_path = tmp_path / "cached.png"
        _make_test_image(img_path)
        cache = ImageCache(max_count=5)
        p1 = cache.get(img_path)
        p2 = cache.get(img_path)
        assert p1 is not None
        assert p1 is p2  # same cached object

    def test_cache_evicts_oldest(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        cache = ImageCache(max_count=2)
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            _make_test_image(p)
            paths.append(p)

        cache.get(paths[0])
        cache.get(paths[1])
        cache.get(paths[2])  # should evict paths[0]
        assert str(paths[0]) not in cache._cache

    def test_cache_clear(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        img_path = tmp_path / "clear.png"
        _make_test_image(img_path)
        cache = ImageCache(max_count=5)
        cache.get(img_path)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_memory_tracking(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        img_path = tmp_path / "mem.png"
        _make_test_image(img_path, 200, 100)
        cache = ImageCache(max_count=10)
        cache.get(img_path)
        # 200*100*4 = 80000 bytes ~= 0.076 MB
        assert cache.memory_usage_mb > 0
        assert cache.size == 1

    def test_memory_limit_eviction(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        # Very low memory limit: 0.01 MB = ~10KB
        cache = ImageCache(max_count=100, max_memory_mb=0.01)
        paths = []
        for i in range(5):
            p = tmp_path / f"big{i}.png"
            _make_test_image(p, 200, 100)  # ~78KB each
            paths.append(p)

        for p in paths:
            cache.get(p)
        # Should have evicted most images due to memory limit
        assert cache.size < 5

    def test_preload(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        cache = ImageCache(max_count=10)
        paths = []
        for i in range(3):
            p = tmp_path / f"pre{i}.png"
            _make_test_image(p)
            paths.append(p)

        cache.preload(paths)
        assert cache.size == 3
        # All should be cache hits now
        for p in paths:
            assert str(p) in cache._cache

    def test_invalidate_updates_memory(self, qapp, tmp_path):
        from src.utils.image import ImageCache

        img_path = tmp_path / "inv.png"
        _make_test_image(img_path, 100, 100)
        cache = ImageCache(max_count=10)
        cache.get(img_path)
        assert cache.memory_usage_mb > 0
        cache.invalidate(img_path)
        assert cache.size == 0
        assert cache.memory_usage_mb == 0


class TestGetImageSize:
    def test_returns_width_height(self, qapp, tmp_path):
        from src.utils.image import get_image_size

        img_path = tmp_path / "size.png"
        _make_test_image(img_path, 320, 240)
        w, h = get_image_size(img_path)
        assert w == 320
        assert h == 240

    def test_nonexistent_returns_zero(self, qapp, tmp_path):
        from src.utils.image import get_image_size

        w, h = get_image_size(tmp_path / "nope.png")
        assert w == 0
        assert h == 0
