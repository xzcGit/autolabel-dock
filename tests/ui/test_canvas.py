"""Tests for AnnotationCanvas."""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt5.QtGui import QImage, QColor, QMouseEvent


def _make_test_image(path: Path, width: int = 200, height: int = 150) -> None:
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(Qt.blue))
    img.save(str(path), "PNG")


class TestCanvasCoordinates:
    """Test coordinate transformations between normalized and pixel space."""

    def test_norm_to_pixel_identity(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        canvas._image_w = 200
        canvas._image_h = 150
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

        px, py = canvas.norm_to_pixel(0.5, 0.5)
        assert abs(px - 100.0) < 1.0
        assert abs(py - 75.0) < 1.0

    def test_pixel_to_norm_roundtrip(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 150
        canvas._scale = 2.0
        canvas._offset_x = 50.0
        canvas._offset_y = 30.0

        nx, ny = 0.3, 0.7
        px, py = canvas.norm_to_pixel(nx, ny)
        nx2, ny2 = canvas.pixel_to_norm(px, py)
        assert abs(nx2 - nx) < 0.001
        assert abs(ny2 - ny) < 0.001

    def test_norm_to_pixel_with_scale(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        canvas._scale = 2.0
        canvas._offset_x = 10.0
        canvas._offset_y = 20.0

        px, py = canvas.norm_to_pixel(0.0, 0.0)
        assert abs(px - 10.0) < 0.01
        assert abs(py - 20.0) < 0.01

        px, py = canvas.norm_to_pixel(1.0, 1.0)
        assert abs(px - 210.0) < 0.01
        assert abs(py - 220.0) < 0.01


class TestCanvasState:
    def test_initial_tool_mode(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        assert canvas.tool_mode == "select"

    def test_set_tool_mode(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.set_tool_mode("draw_bbox")
        assert canvas.tool_mode == "draw_bbox"
        canvas.set_tool_mode("draw_keypoint")
        assert canvas.tool_mode == "draw_keypoint"

    def test_load_image(self, qapp, tmp_path):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        img_path = tmp_path / "test.png"
        _make_test_image(img_path, 200, 150)
        canvas.load_image(str(img_path))
        assert canvas._image is not None
        assert canvas._image_w == 200
        assert canvas._image_h == 150

    def test_load_image_fit_to_window(self, qapp, tmp_path):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        img_path = tmp_path / "big.png"
        _make_test_image(img_path, 800, 600)
        canvas.load_image(str(img_path))
        assert canvas._scale <= 1.0
        assert canvas._scale > 0

    def test_set_annotations(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100

        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        assert len(canvas._annotations) == 1
        assert canvas._annotations[0].class_name == "cat"

    def test_clear_resets_state(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        canvas.select_annotation(ann.id)
        canvas.clear()
        assert canvas._annotations == []
        assert canvas._selected_id is None
        assert canvas._image is None


class TestCanvasSelection:
    def test_select_annotation(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        canvas.select_annotation(ann.id)
        assert canvas._selected_id == ann.id

    def test_select_none_deselects(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 100
        canvas._image_h = 100
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        canvas.select_annotation(ann.id)
        canvas.select_annotation(None)
        assert canvas._selected_id is None

    def test_hit_test_bbox(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4))
        canvas.set_annotations([ann])

        # Click inside bbox
        result = canvas.hit_test(100.0, 100.0)
        assert result == ann.id

        # Click outside bbox
        result = canvas.hit_test(10.0, 10.0)
        assert result is None


class TestCanvasViewportCulling:
    def test_ann_in_viewport_visible(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0

        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.3))
        assert canvas._ann_in_viewport(ann, 0, 0, 200, 200) is True

    def test_ann_in_viewport_outside(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = -500.0  # image scrolled far left
        canvas._offset_y = 0.0

        # Annotation centered at 0.5 → pixel 100, but offset pushes it to -400
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.3))
        assert canvas._ann_in_viewport(ann, 0, 0, 200, 200) is False

    def test_keypoint_only_always_in_viewport(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp_ann = Annotation(class_name="pt", class_id=0, keypoints=[
            Keypoint(x=0.5, y=0.5, visible=2, label="nose"),
        ])
        # No bbox, should always return True
        assert canvas._ann_in_viewport(kp_ann, 0, 0, 200, 200) is True


class TestCanvasConflicts:
    def test_set_conflict_pairs(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        a1 = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True)
        a2 = Annotation(class_name="person", class_id=0, bbox=(0.52, 0.52, 0.2, 0.2),
                         confirmed=False, source="auto")
        canvas.set_annotations([a1, a2])
        canvas.set_conflict_pairs([(a1.id, a2.id)])

        assert canvas._conflict_pairs[a1.id] == a2.id
        assert canvas._conflict_pairs[a2.id] == a1.id

    def test_resolve_conflict_keep_existing(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        a1 = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True)
        a2 = Annotation(class_name="person", class_id=0, bbox=(0.52, 0.52, 0.2, 0.2),
                         confirmed=False, source="auto")
        canvas.set_annotations([a1, a2])
        canvas.set_conflict_pairs([(a1.id, a2.id)])

        canvas.resolve_conflict(a1.id)  # keep existing

        assert len(canvas.annotations) == 1
        assert canvas.annotations[0].id == a1.id
        assert a1.id not in canvas._conflict_pairs
        assert a2.id not in canvas._conflict_pairs

    def test_resolve_conflict_keep_prediction(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        a1 = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True)
        a2 = Annotation(class_name="person", class_id=0, bbox=(0.52, 0.52, 0.2, 0.2),
                         confirmed=False, source="auto")
        canvas.set_annotations([a1, a2])
        canvas.set_conflict_pairs([(a1.id, a2.id)])

        canvas.resolve_conflict(a2.id)  # keep prediction

        assert len(canvas.annotations) == 1
        assert canvas.annotations[0].id == a2.id
        assert canvas.annotations[0].confirmed is False

    def test_clear_conflicts(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        a1 = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True)
        a2 = Annotation(class_name="person", class_id=0, bbox=(0.52, 0.52, 0.2, 0.2),
                         confirmed=False, source="auto")
        canvas.set_annotations([a1, a2])
        canvas.set_conflict_pairs([(a1.id, a2.id)])
        canvas.clear_conflicts()

        assert len(canvas._conflict_pairs) == 0
        assert len(canvas.annotations) == 2  # both still there

    def test_clear_also_clears_conflicts(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation

        canvas = AnnotationCanvas()
        a1 = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True)
        a2 = Annotation(class_name="person", class_id=0, bbox=(0.52, 0.52, 0.2, 0.2),
                         confirmed=False, source="auto")
        canvas.set_annotations([a1, a2])
        canvas.set_conflict_pairs([(a1.id, a2.id)])
        canvas.clear()

        assert len(canvas._conflict_pairs) == 0
        assert len(canvas.annotations) == 0


class TestCanvasKeypointManagement:
    def test_add_keypoint_to_annotation(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
        canvas.set_annotations([ann])
        assert len(ann.keypoints) == 0

        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        canvas.add_keypoint_to_annotation(ann.id, kp)
        assert len(canvas.annotations[0].keypoints) == 1
        assert canvas.annotations[0].keypoints[0].label == "nose"

    def test_add_keypoint_auto_confirms(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=False)
        canvas.set_annotations([ann])

        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        canvas.add_keypoint_to_annotation(ann.id, kp)
        assert canvas.annotations[0].confirmed is True

    def test_remove_keypoint(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kps = [Keypoint(x=0.1, y=0.2, visible=2, label="nose"),
               Keypoint(x=0.3, y=0.4, visible=1, label="eye")]
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=kps)
        canvas.set_annotations([ann])

        canvas.remove_keypoint(ann.id, 0)
        assert len(canvas.annotations[0].keypoints) == 1
        assert canvas.annotations[0].keypoints[0].label == "eye"

    def test_remove_last_keypoint_on_bbox_ann_keeps_ann(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        canvas.set_annotations([ann])

        canvas.remove_keypoint(ann.id, 0)
        assert len(canvas.annotations) == 1  # bbox still there
        assert len(canvas.annotations[0].keypoints) == 0

    def test_remove_last_keypoint_on_kp_only_ann_deletes_ann(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="point", class_id=0, keypoints=[kp])
        canvas.set_annotations([ann])

        canvas.remove_keypoint(ann.id, 0)
        assert len(canvas.annotations) == 0

    def test_rename_keypoint(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        canvas.set_annotations([ann])

        canvas.rename_keypoint(ann.id, 0, "left_eye")
        assert canvas.annotations[0].keypoints[0].label == "left_eye"

    def test_cycle_keypoint_visibility(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp = Keypoint(x=0.1, y=0.2, visible=0, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        canvas.set_annotations([ann])

        canvas.cycle_keypoint_visibility(ann.id, 0)
        assert canvas.annotations[0].keypoints[0].visible == 1
        canvas.cycle_keypoint_visibility(ann.id, 0)
        assert canvas.annotations[0].keypoints[0].visible == 2
        canvas.cycle_keypoint_visibility(ann.id, 0)
        assert canvas.annotations[0].keypoints[0].visible == 0

    def test_select_keypoint(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        canvas.set_annotations([ann])

        canvas.select_keypoint(ann.id, 0)
        assert canvas._selected_id == ann.id
        assert canvas._selected_kp_idx == 0

    def test_select_annotation_clears_kp_selection(self, qapp):
        from src.ui.canvas import AnnotationCanvas
        from src.core.annotation import Annotation, Keypoint

        canvas = AnnotationCanvas()
        kp = Keypoint(x=0.1, y=0.2, visible=2, label="nose")
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), keypoints=[kp])
        canvas.set_annotations([ann])

        canvas.select_keypoint(ann.id, 0)
        canvas.select_annotation(ann.id)
        assert canvas._selected_kp_idx is None


def _press(canvas, x, y, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), button, button, modifiers)
    canvas.mousePressEvent(evt)


def _move(canvas, x, y, buttons=Qt.NoButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton, buttons, modifiers)
    canvas.mouseMoveEvent(evt)


def _release(canvas, x, y, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseButtonRelease, QPointF(x, y), button, Qt.NoButton, modifiers)
    canvas.mouseReleaseEvent(evt)


class TestCtrlDragPan:
    """Ctrl + left-button drag pans the canvas in all tool modes."""

    def _make_canvas(self, qapp):
        from src.ui.canvas import AnnotationCanvas

        canvas = AnnotationCanvas()
        canvas.resize(400, 300)
        canvas._image_w = 200
        canvas._image_h = 150
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0
        # Stub image so code paths requiring _image is not None activate
        canvas._image = QImage(200, 150, QImage.Format_RGB32)
        return canvas

    def test_ctrl_left_press_enters_panning(self, qapp):
        canvas = self._make_canvas(qapp)
        _press(canvas, 100, 80, Qt.LeftButton, Qt.ControlModifier)
        assert canvas._panning is True
        assert canvas._pan_start == (100, 80)

    def test_ctrl_left_drag_updates_offset(self, qapp):
        canvas = self._make_canvas(qapp)
        _press(canvas, 100, 80, Qt.LeftButton, Qt.ControlModifier)
        _move(canvas, 140, 100, Qt.LeftButton, Qt.ControlModifier)
        assert canvas._offset_x == 40.0
        assert canvas._offset_y == 20.0
        assert canvas._pan_start == (140, 100)

    def test_left_release_ends_panning(self, qapp):
        canvas = self._make_canvas(qapp)
        _press(canvas, 100, 80, Qt.LeftButton, Qt.ControlModifier)
        _release(canvas, 120, 90, Qt.LeftButton)
        assert canvas._panning is False
        assert canvas._pan_start is None

    def test_ctrl_left_does_not_create_bbox_in_draw_mode(self, qapp):
        canvas = self._make_canvas(qapp)
        canvas.set_tool_mode("draw_bbox")
        _press(canvas, 50, 50, Qt.LeftButton, Qt.ControlModifier)
        assert canvas._panning is True
        assert canvas._drawing is False
        _release(canvas, 80, 80, Qt.LeftButton)
        assert canvas._annotations == []

    def test_ctrl_left_does_not_select_in_select_mode(self, qapp):
        from src.core.annotation import Annotation

        canvas = self._make_canvas(qapp)
        ann = Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4))
        canvas.set_annotations([ann])
        # Click inside the bbox with Ctrl — should pan, not select
        _press(canvas, 100, 75, Qt.LeftButton, Qt.ControlModifier)
        assert canvas._panning is True
        assert canvas._selected_id is None

    def test_plain_left_click_still_selects(self, qapp):
        from src.core.annotation import Annotation

        canvas = self._make_canvas(qapp)
        ann = Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.4, 0.4))
        canvas.set_annotations([ann])
        _press(canvas, 100, 75, Qt.LeftButton, Qt.NoModifier)
        assert canvas._panning is False
        assert canvas._selected_id == ann.id

    def test_middle_button_pan_still_works(self, qapp):
        canvas = self._make_canvas(qapp)
        _press(canvas, 100, 80, Qt.MiddleButton)
        assert canvas._panning is True
        _release(canvas, 120, 90, Qt.MiddleButton)
        assert canvas._panning is False
