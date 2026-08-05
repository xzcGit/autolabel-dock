"""Tests for AnnotationCanvas polygon (segment) drawing + editing.

Mirrors the bbox/keypoint interaction tests in test_canvas.py: synthesize
mouse events with QMouseEvent and drive the canvas directly. The canvas holds
image geometry with a 1:1 scale so pixel↔normalized is a plain divide.
"""
import pytest
from PyQt5.QtCore import Qt, QPointF, QEvent
from PyQt5.QtGui import QImage, QMouseEvent


@pytest.fixture
def canvas_factory(qapp):
    """Yield a factory that tracks canvases and deletes them on teardown.

    Leaked top-level widgets pollute the shared QApplication (they perturb
    later geometry-sensitive tests), so every canvas built here is cleaned up.
    """
    from src.ui.canvas import AnnotationCanvas

    created = []

    def _make():
        canvas = AnnotationCanvas()
        canvas.resize(200, 200)
        canvas._image_w = 200
        canvas._image_h = 200
        canvas._scale = 1.0
        canvas._offset_x = 0.0
        canvas._offset_y = 0.0
        canvas._image = QImage(200, 200, QImage.Format_RGB32)
        created.append(canvas)
        return canvas

    yield _make
    for c in created:
        c.deleteLater()


def _press(canvas, x, y, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y), button, button, modifiers)
    canvas.mousePressEvent(evt)


def _move(canvas, x, y, buttons=Qt.NoButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseMove, QPointF(x, y), Qt.NoButton, buttons, modifiers)
    canvas.mouseMoveEvent(evt)


def _release(canvas, x, y, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseButtonRelease, QPointF(x, y), button, Qt.NoButton, modifiers)
    canvas.mouseReleaseEvent(evt)


def _dblclick(canvas, x, y, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    evt = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(x, y), button, button, modifiers)
    canvas.mouseDoubleClickEvent(evt)


class TestPolygonDrawMode:
    def test_set_tool_mode_polygon(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        assert canvas.tool_mode == "draw_polygon"

    def test_clicks_accumulate_vertices(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        _press(canvas, 100, 20)
        _press(canvas, 60, 120)
        assert len(canvas._polygon_points) == 3
        # First vertex seeds _draw_start (satisfies the resize refit guard).
        assert canvas._draw_start is not None
        assert canvas.has_polygon_draft is True

    def test_first_vertex_seeds_draw_start_for_refit_guard(self, canvas_factory):
        from PyQt5.QtGui import QResizeEvent
        from PyQt5.QtCore import QSize

        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        before = (canvas._scale, canvas._offset_x, canvas._offset_y)
        canvas.resizeEvent(QResizeEvent(QSize(400, 400), QSize(200, 200)))
        # Refit must not fire mid-draw (would move the placed vertex under the user).
        assert (canvas._scale, canvas._offset_x, canvas._offset_y) == before

    def test_move_updates_rubber_band(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        _move(canvas, 80, 90)
        assert canvas._draw_current is not None
        assert abs(canvas._draw_current[0] - 0.4) < 0.01
        assert abs(canvas._draw_current[1] - 0.45) < 0.01

    def test_double_click_closes_polygon_and_requests_class(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        requested = []
        canvas.class_requested.connect(lambda px, py: requested.append((px, py)))
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        _press(canvas, 70, 120)
        _dblclick(canvas, 70, 120)
        # class_requested fired → view will pop the class picker.
        assert len(requested) == 1
        # Draft is retained until the class is chosen.
        assert len(canvas._polygon_points) == 3

    def test_double_click_dedupes_press_duplicate_vertex(self, canvas_factory):
        """The dblclick's own press appended a 4th vertex duplicating the 3rd;
        it must be dropped so the closed polygon has 3 vertices, not 4."""
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        canvas.class_requested.connect(lambda px, py: None)
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        _press(canvas, 70, 120)
        # The double-click delivers a press at the same spot first.
        _press(canvas, 70, 120)
        assert len(canvas._polygon_points) == 4
        _dblclick(canvas, 70, 120)
        assert len(canvas._polygon_points) == 3

    def test_create_polygon_from_draw_builds_annotation(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        created = []
        canvas.annotation_created.connect(lambda a: created.append(a))
        _press(canvas, 20, 20)    # (0.1, 0.1)
        _press(canvas, 120, 20)   # (0.6, 0.1)
        _press(canvas, 70, 120)   # (0.35, 0.6)
        ann = canvas.create_polygon_from_draw("leaf", 3)
        assert ann is not None
        assert ann.polygon == [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
        assert ann.confirmed is True
        assert ann.source == "manual"
        # Derived bbox spans x∈[0.1,0.6], y∈[0.1,0.6].
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.35) < 1e-6
        assert abs(w - 0.5) < 1e-6
        assert abs(cy - 0.35) < 1e-6
        assert abs(h - 0.5) < 1e-6
        assert len(created) == 1
        # Draft cleared after creation.
        assert canvas._polygon_points == []

    def test_create_polygon_needs_three_vertices(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        ann = canvas.create_polygon_from_draw("leaf", 3)
        assert ann is None
        assert canvas._polygon_points == []

    def test_finish_polygon_below_three_keeps_draft(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        requested = []
        canvas.class_requested.connect(lambda px, py: requested.append((px, py)))
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        assert canvas.finish_polygon() is False
        assert requested == []
        assert len(canvas._polygon_points) == 2  # draft kept

    def test_set_tool_mode_clears_polygon_draft(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        canvas.set_tool_mode("select")
        assert canvas._polygon_points == []

    def test_clear_draw_state_clears_polygon_draft(self, canvas_factory):
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        canvas.clear_draw_state()
        assert canvas._polygon_points == []
        assert canvas.has_polygon_draft is False

    def test_set_annotations_drops_stale_polygon_draft(self, canvas_factory):
        """A polygon draft is open across events (no button held), so an image
        switch / undo restore / reload_current can happen mid-draft — the
        replaced annotation set must drop the draft, or its vertices would
        render over (and could be committed onto) the wrong image."""
        canvas = canvas_factory()
        canvas.set_tool_mode("draw_polygon")
        _press(canvas, 20, 20)
        _press(canvas, 120, 20)
        assert canvas.has_polygon_draft is True
        canvas.set_annotations([])  # what an image switch delivers
        assert canvas.has_polygon_draft is False
        assert canvas._draw_start is None
        assert canvas._draw_current is None
        # A later finish on the new image must not create anything.
        assert canvas.finish_polygon() is False


class TestPolygonVertexEditing:
    def _make_with_polygon(self, canvas_factory):
        from src.core.annotation import Annotation
        from src.core.polygon import bbox_from_polygon

        canvas = canvas_factory()
        poly = [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
        ann = Annotation(
            class_name="leaf", class_id=0,
            polygon=[list(p) for p in poly],
            bbox=bbox_from_polygon(poly),
            confirmed=False, source="auto",
        )
        canvas.set_annotations([ann])
        canvas.set_tool_mode("select")
        return canvas, ann

    def test_drag_vertex_updates_polygon_and_bbox(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        canvas.select_annotation(ann.id)
        # Vertex 0 is at (0.1, 0.1) → pixel (20, 20). Press on it.
        _press(canvas, 20, 20)
        assert canvas._drag_type == "move_poly_vertex"
        assert canvas._drag_vertex_idx == 0
        # Drag to (0.0, 0.0) → pixel (0, 0).
        _move(canvas, 0, 0, buttons=Qt.LeftButton)
        assert ann.polygon[0] == [0.0, 0.0]
        # Derived bbox rebuilt from new vertices: x∈[0,0.6], y∈[0,0.6].
        cx, cy, w, h = ann.bbox
        assert abs(w - 0.6) < 1e-6
        assert abs(h - 0.6) < 1e-6
        # Manual edit auto-confirms.
        assert ann.confirmed is True

    def test_drag_vertex_emits_modified_once_on_release(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        canvas.select_annotation(ann.id)
        modified = []
        canvas.annotation_modified.connect(lambda aid: modified.append(aid))
        _press(canvas, 20, 20)
        _move(canvas, 10, 10, buttons=Qt.LeftButton)
        _move(canvas, 0, 0, buttons=Qt.LeftButton)
        assert modified == []  # nothing on the way
        _release(canvas, 0, 0)
        assert modified == [ann.id]  # exactly once → single undo push

    def test_vertex_handles_only_on_selected_polygon(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        # Not selected → no vertex hit.
        assert canvas._hit_test_polygon_vertex(20, 20) is None
        canvas.select_annotation(ann.id)
        assert canvas._hit_test_polygon_vertex(20, 20) == (ann.id, 0)

    def test_polygon_has_no_corner_resize_handles(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        canvas.select_annotation(ann.id)
        # Derived bbox corners must NOT expose resize handles (would desync).
        assert canvas._hit_test_handle(20, 20) is None

    def test_insert_vertex_grows_polygon_and_rebuilds_bbox(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        canvas.select_annotation(ann.id)
        # Insert on edge 0 (between vertex 0 and 1) at (0.35, 0.05) — above the box.
        canvas.insert_polygon_vertex(ann.id, 0, 0.35, 0.05)
        assert len(ann.polygon) == 4
        assert ann.polygon[1] == [0.35, 0.05]
        # bbox top now at y=0.05.
        _, cy, _, h = ann.bbox
        assert abs((cy - h / 2) - 0.05) < 1e-6
        assert ann.confirmed is True

    def test_remove_vertex(self, canvas_factory):
        from src.core.annotation import Annotation
        from src.core.polygon import bbox_from_polygon

        canvas = canvas_factory()
        poly = [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]]
        ann = Annotation(
            class_name="leaf", class_id=0,
            polygon=[list(p) for p in poly],
            bbox=bbox_from_polygon(poly),
        )
        canvas.set_annotations([ann])
        canvas.remove_polygon_vertex(ann.id, 2)
        assert len(ann.polygon) == 3
        assert [0.6, 0.6] not in ann.polygon

    def test_remove_vertex_refuses_below_triangle(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        assert len(ann.polygon) == 3
        canvas.remove_polygon_vertex(ann.id, 0)
        assert len(ann.polygon) == 3  # unchanged — a polygon needs ≥3 vertices

    def test_edge_hit_test_targets_selected_polygon(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        canvas.select_annotation(ann.id)
        # Midpoint of edge 0 (vertex0→vertex1): (0.35, 0.1) → pixel (70, 20).
        edge = canvas._hit_test_polygon_edge(70, 20)
        assert edge == 0

    def test_whole_polygon_move_translates_vertices(self, canvas_factory):
        canvas, ann = self._make_with_polygon(canvas_factory)
        canvas.select_annotation(ann.id)
        # Press inside the polygon body (centroid-ish), away from any vertex.
        # Centroid ~ (0.35, 0.27) → pixel (70, 54).
        _press(canvas, 70, 54)
        assert canvas._drag_type == "move"
        # Drag by +0.05 in both axes (10 px).
        _move(canvas, 80, 64, buttons=Qt.LeftButton)
        # Every vertex shifted by the same delta; bbox stays in sync.
        assert abs(ann.polygon[0][0] - 0.15) < 1e-6
        assert abs(ann.polygon[0][1] - 0.15) < 1e-6
        from src.core.polygon import bbox_from_polygon
        assert ann.bbox == bbox_from_polygon(ann.polygon)
        assert ann.confirmed is True

