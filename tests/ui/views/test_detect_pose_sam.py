"""DetectPoseView SAM-assist tests — tool gating + mask→annotation signal flow.

A fake controller (matching SamAssistController's public face) is injected so
no ultralytics/QThread is involved.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QColor


def _make_view(qapp, sam_controller=None):
    from src.ui.views.detect_pose import DetectPoseView
    from src.utils.image import ImageCache

    return DetectPoseView(
        ImageCache(max_count=2, max_memory_mb=16.0), OrderedDict(),
        sam_controller=sam_controller,
    )


def _make_project(tmp_path, task_type):
    from src.core.project import ProjectManager

    pm = ProjectManager.create(
        tmp_path / f"proj_{task_type}", "p", classes=["leaf", "petal"],
        task_type=task_type,
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(2):
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


class _FakeSamController(QObject):
    """Public face of SamAssistController without any model/thread."""

    load_started = pyqtSignal()
    load_failed = pyqtSignal(str)
    encode_started = pyqtSignal(object)
    encode_ready = pyqtSignal(object)
    mask_ready = pyqtSignal(object, object)
    status_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.activated: list = []
        self.images: list = []
        self.points: list = []
        self.boxes: list = []
        self.shutdowns = 0

    def activate(self, path, size):
        self.activated.append((path, size))

    def set_image(self, path, size):
        self.images.append((path, size))

    def request_point(self, x, y):
        self.points.append((x, y))

    def request_box(self, x1, y1, x2, y2):
        self.boxes.append((x1, y1, x2, y2))

    def shutdown(self, timeout_ms=30000):
        self.shutdowns += 1


POLY = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]


class TestSamToolGating:
    def test_button_visible_only_for_segment(self, qapp, tmp_path):
        view = _make_view(qapp)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            assert not view._btn_sam.isHidden()
            view.set_project(_make_project(tmp_path, "detect"))
            assert view._btn_sam.isHidden()
        finally:
            view.deleteLater()

    def test_no_controller_constructed_until_tool_armed(self, qapp, tmp_path):
        view = _make_view(qapp)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            assert view._sam_ctrl is None  # lazy: zero model impact at load
        finally:
            view.deleteLater()

    def test_arming_tool_activates_controller_with_focused_image(self, qapp, tmp_path):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            assert view._canvas.tool_mode == "sam_prompt"
            assert len(fake.activated) == 1
            path, size = fake.activated[0]
            assert path == view.get_focused_image()
            assert size == (100, 80)
        finally:
            view.deleteLater()

    def test_image_switch_while_armed_pre_encodes(self, qapp, tmp_path):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            view._file_list.setCurrentRow(1)
            assert len(fake.images) == 1
            assert fake.images[0][0] == view.get_focused_image()
        finally:
            view.deleteLater()

    def test_image_switch_without_tool_does_not_touch_controller(self, qapp, tmp_path):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._file_list.setCurrentRow(1)
            assert fake.images == [] and fake.activated == []
        finally:
            view.deleteLater()

    def test_cleanup_shuts_controller_down(self, qapp, tmp_path):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view.cleanup()
            assert fake.shutdowns == 1
        finally:
            view.deleteLater()


class TestSamSignalFlow:
    def test_canvas_prompts_forward_to_controller(self, qapp, tmp_path):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            view._canvas.sam_point_requested.emit(0.3, 0.4)
            view._canvas.sam_box_requested.emit(0.1, 0.1, 0.5, 0.5)
            assert fake.points == [(0.3, 0.4)]
            assert fake.boxes == [(0.1, 0.1, 0.5, 0.5)]
        finally:
            view.deleteLater()

    def test_mask_ready_creates_auto_unconfirmed_polygon(self, qapp, tmp_path, monkeypatch):
        from src.core.polygon import bbox_from_polygon

        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            monkeypatch.setattr(
                view, "_show_class_picker", lambda *a, **k: "leaf",
            )
            created = []
            view._canvas.annotation_created.connect(created.append)

            fake.mask_ready.emit(POLY, bbox_from_polygon(POLY))

            assert len(created) == 1
            ann = created[0]
            assert ann.class_name == "leaf"
            assert ann.source == "auto"
            assert ann.confirmed is False
            assert ann.polygon == POLY
            assert ann.bbox == bbox_from_polygon(ann.polygon)
            # Undo covers the assisted annotation (existing signal flow).
            view.undo()
            assert view._canvas.annotations == []
        finally:
            view.deleteLater()

    def test_mask_ready_after_tool_switch_is_ignored(self, qapp, tmp_path, monkeypatch):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            view._set_tool("select")
            monkeypatch.setattr(view, "_show_class_picker", lambda *a, **k: "leaf")
            fake.mask_ready.emit(POLY, None)
            assert view._canvas.annotations == []
        finally:
            view.deleteLater()

    def test_picker_cancel_drops_staged_polygon(self, qapp, tmp_path, monkeypatch):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            monkeypatch.setattr(view, "_show_class_picker", lambda *a, **k: None)
            fake.mask_ready.emit(POLY, None)
            assert view._canvas.annotations == []
            assert not view._canvas.has_polygon_draft
        finally:
            view.deleteLater()

    def test_status_messages_reach_shell(self, qapp, tmp_path):
        fake = _FakeSamController()
        view = _make_view(qapp, sam_controller=fake)
        try:
            view.set_project(_make_project(tmp_path, "segment"))
            view._set_tool("sam_prompt")
            statuses = []
            view.status_changed.connect(statuses.append)
            fake.status_message.emit("SAM 就绪")
            assert "SAM 就绪" in statuses
        finally:
            view.deleteLater()
