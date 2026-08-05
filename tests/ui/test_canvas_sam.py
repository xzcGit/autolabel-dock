"""Canvas sam_prompt tool tests — gesture forwarding + polygon staging.

The canvas only relays interactions (point/box, normalized) and stages
externally-computed polygons; it knows nothing about any SAM backend.
"""
from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt, QEvent, QPointF
from PyQt5.QtGui import QMouseEvent, QPixmap

from src.core.polygon import bbox_from_polygon
from src.ui.canvas import AnnotationCanvas


@pytest.fixture
def canvas(qapp):
    c = AnnotationCanvas()
    c.resize(200, 200)
    pix = QPixmap(100, 100)
    pix.fill(Qt.black)
    c.set_pixmap(pix)
    return c


def _press(canvas, x, y):
    canvas.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, QPointF(x, y),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    ))


def _move(canvas, x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QEvent.MouseMove, QPointF(x, y),
        Qt.NoButton, Qt.LeftButton, Qt.NoModifier,
    ))


def _release(canvas, x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, QPointF(x, y),
        Qt.LeftButton, Qt.NoButton, Qt.NoModifier,
    ))


class TestSamGesture:
    def test_click_emits_point_prompt(self, canvas):
        canvas.set_tool_mode("sam_prompt")
        points, boxes = [], []
        canvas.sam_point_requested.connect(lambda *a: points.append(a))
        canvas.sam_box_requested.connect(lambda *a: boxes.append(a))

        px, py = canvas.norm_to_pixel(0.5, 0.5)
        _press(canvas, px, py)
        _release(canvas, px, py)

        assert boxes == []
        assert len(points) == 1
        assert points[0][0] == pytest.approx(0.5, abs=0.02)
        assert points[0][1] == pytest.approx(0.5, abs=0.02)
        assert not canvas._drawing and canvas._draw_start is None

    def test_drag_emits_box_prompt_sorted_corners(self, canvas):
        canvas.set_tool_mode("sam_prompt")
        points, boxes = [], []
        canvas.sam_point_requested.connect(lambda *a: points.append(a))
        canvas.sam_box_requested.connect(lambda *a: boxes.append(a))

        # Drag bottom-right → top-left; emitted corners must be sorted.
        sx, sy = canvas.norm_to_pixel(0.8, 0.7)
        ex, ey = canvas.norm_to_pixel(0.2, 0.3)
        _press(canvas, sx, sy)
        _move(canvas, ex, ey)
        _release(canvas, ex, ey)

        assert points == []
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert x1 == pytest.approx(0.2, abs=0.02) and x2 == pytest.approx(0.8, abs=0.02)
        assert y1 == pytest.approx(0.3, abs=0.02) and y2 == pytest.approx(0.7, abs=0.02)

    def test_tiny_drag_counts_as_point(self, canvas):
        canvas.set_tool_mode("sam_prompt")
        points, boxes = [], []
        canvas.sam_point_requested.connect(lambda *a: points.append(a))
        canvas.sam_box_requested.connect(lambda *a: boxes.append(a))
        px, py = canvas.norm_to_pixel(0.4, 0.4)
        _press(canvas, px, py)
        _release(canvas, px + 1, py + 1)
        assert len(points) == 1 and boxes == []

    def test_other_modes_do_not_emit_sam_signals(self, canvas):
        canvas.set_tool_mode("draw_bbox")
        points, boxes = [], []
        canvas.sam_point_requested.connect(lambda *a: points.append(a))
        canvas.sam_box_requested.connect(lambda *a: boxes.append(a))
        px, py = canvas.norm_to_pixel(0.5, 0.5)
        _press(canvas, px, py)
        _release(canvas, px, py)
        assert points == [] and boxes == []


class TestPolygonStaging:
    POLY = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]

    def test_begin_polygon_from_points_requests_class(self, canvas):
        requested = []
        canvas.class_requested.connect(lambda *a: requested.append(a))
        assert canvas.begin_polygon_from_points(self.POLY)
        assert len(requested) == 1
        assert canvas.has_polygon_draft

    def test_begin_polygon_rejects_degenerate(self, canvas):
        requested = []
        canvas.class_requested.connect(lambda *a: requested.append(a))
        assert not canvas.begin_polygon_from_points([[0.1, 0.1], [0.2, 0.2]])
        assert not canvas.begin_polygon_from_points([])
        assert requested == []
        assert not canvas.has_polygon_draft

    def test_create_polygon_from_staged_mask_auto_unconfirmed(self, canvas):
        canvas.begin_polygon_from_points(self.POLY)
        ann = canvas.create_polygon_from_draw("leaf", 0, source="auto", confirmed=False)
        assert ann is not None
        assert ann.source == "auto"
        assert ann.confirmed is False
        assert ann.polygon == self.POLY
        assert ann.bbox == bbox_from_polygon(self.POLY)
        assert not canvas.has_polygon_draft

    def test_create_polygon_defaults_stay_manual_confirmed(self, canvas):
        canvas.begin_polygon_from_points(self.POLY)
        ann = canvas.create_polygon_from_draw("leaf", 0)
        assert ann.source == "manual" and ann.confirmed is True
