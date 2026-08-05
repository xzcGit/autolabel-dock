"""FileListWidget video-drop recognition + the view→shell forwarding chain.

Entry 2 of the video-frame-import feature: dropping video files onto the
detect/pose file list emits ``videos_dropped`` (routed up through
DetectPoseView → LabelPanel → MainWindow's import dialog); images in the
same drop keep taking the existing copy path.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt5.QtGui import QDropEvent

from src.ui.file_list import FileListWidget


def _drop(widget, paths) -> None:
    """Build and deliver a file-URL drop.

    The QMimeData must outlive the dropEvent call — QDropEvent does NOT take
    ownership (in the real flow the drag source owns it), so constructing it
    inside a helper that returns only the event segfaults on mimeData().
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(
        QPointF(5.0, 5.0), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier,
    )
    widget.dropEvent(event)


class TestFileListVideoDrop:
    def test_video_drop_emits_videos_dropped_only(self, qapp, tmp_path):
        video = tmp_path / "clip.mp4"
        video.touch()
        widget = FileListWidget()
        images, videos = [], []
        widget.images_dropped.connect(images.append)
        widget.videos_dropped.connect(videos.append)

        _drop(widget, [video])

        assert videos == [[video]]
        assert images == []

    def test_mixed_drop_routes_each_kind(self, qapp, tmp_path):
        img = tmp_path / "a.png"
        img.touch()
        vid = tmp_path / "b.mkv"
        vid.touch()
        widget = FileListWidget()
        images, videos = [], []
        widget.images_dropped.connect(images.append)
        widget.videos_dropped.connect(videos.append)

        _drop(widget, [img, vid])

        assert images == [[img]]
        assert videos == [[vid]]

    def test_uppercase_extension_recognized(self, qapp, tmp_path):
        video = tmp_path / "CLIP.MOV"
        video.touch()
        widget = FileListWidget()
        videos = []
        widget.videos_dropped.connect(videos.append)

        _drop(widget, [video])

        assert videos == [[video]]

    def test_directory_drop_globs_videos_too(self, qapp, tmp_path):
        (tmp_path / "v1.mp4").touch()
        (tmp_path / "v2.mov").touch()
        (tmp_path / "i.jpg").touch()
        widget = FileListWidget()
        images, videos = [], []
        widget.images_dropped.connect(images.append)
        widget.videos_dropped.connect(videos.append)

        _drop(widget, [tmp_path])

        assert images == [[tmp_path / "i.jpg"]]
        assert len(videos) == 1
        assert set(videos[0]) == {tmp_path / "v1.mp4", tmp_path / "v2.mov"}

    def test_unrelated_extension_emits_nothing(self, qapp, tmp_path):
        stray = tmp_path / "notes.txt"
        stray.touch()
        widget = FileListWidget()
        images, videos = [], []
        widget.images_dropped.connect(images.append)
        widget.videos_dropped.connect(videos.append)

        _drop(widget, [stray])

        assert images == [] and videos == []


class TestLabelPanelForwardingChain:
    def test_view_videos_dropped_forwards_to_panel_signal(self, qapp, tmp_path):
        """FileListWidget → DetectPoseView → LabelPanel.videos_dropped."""
        from PyQt5.QtGui import QColor, QImage

        from src.core.project import ProjectManager
        from src.ui.label_panel import LabelPanel

        pm = ProjectManager.create(
            tmp_path / "proj", "p", classes=["cat"], task_type="detect",
        )
        img = QImage(20, 20, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(pm.project_dir / pm.config.image_dir / "img0.png"), "PNG")

        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        received = []
        panel.videos_dropped.connect(received.append)

        video = tmp_path / "clip.mp4"
        video.touch()
        _drop(panel._view._file_list, [video])

        assert received == [[video]]
