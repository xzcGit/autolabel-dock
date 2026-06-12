"""Tests for ThumbnailLoader."""


def test_thumbnail_loader_synchronous(qapp, tmp_path):
    """Inject a synchronous loader_fn to bypass thread async.

    QThread.start() spins up a real thread; we wait() for completion and
    inspect the emitted (path, pixmap) tuples on the main thread.
    """
    from PyQt5.QtCore import QSize
    from PyQt5.QtGui import QImage, QPixmap
    from src.ui.views.thumbnail_loader import ThumbnailLoader

    def fake_load(path, size):
        img = QImage(size.width(), size.height(), QImage.Format_RGB32)
        return QPixmap.fromImage(img)

    loader = ThumbnailLoader(loader_fn=fake_load)
    received = []
    loader.loaded.connect(lambda p, pix: received.append((p, pix)))
    p = tmp_path / "x.png"
    loader.enqueue(p, QSize(64, 64))
    loader.wait(2000)
    qapp.processEvents()
    assert received
    assert received[0][0] == p
    assert not received[0][1].isNull()


def test_thumbnail_loader_skips_failed_loads(qapp, tmp_path):
    """When loader_fn returns None, no signal is emitted."""
    from PyQt5.QtCore import QSize
    from src.ui.views.thumbnail_loader import ThumbnailLoader

    def fail_load(path, size):
        return None

    loader = ThumbnailLoader(loader_fn=fail_load)
    received = []
    loader.loaded.connect(lambda p, pix: received.append((p, pix)))
    loader.enqueue(tmp_path / "x.png", QSize(64, 64))
    loader.wait(2000)
    qapp.processEvents()
    assert received == []
