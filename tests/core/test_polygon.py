"""Tests for src/core/polygon.py — Qt-free segment geometry helpers."""
import pytest

from src.core.polygon import (
    bbox_from_polygon,
    bbox_to_polygon,
    nearest_edge,
    nearest_vertex,
    point_in_polygon,
    simplify_polygon,
)


def test_module_is_qt_free():
    """polygon.py must not drag in Qt (unit-testable without a QApplication)."""
    import src.core.polygon as mod

    src_text = open(mod.__file__, encoding="utf-8").read()
    assert "PyQt5" not in src_text
    assert "src.ui" not in src_text
    assert not any(
        name.startswith("PyQt5") for name in getattr(mod, "__dict__", {})
    )


def test_cv2_import_does_not_hijack_qt_plugin_path():
    """opencv-python sets QT_QPA_PLATFORM_PLUGIN_PATH at import; a QApplication
    constructed AFTER that aborts (PyQt5 loads cv2's ABI-incompatible xcb
    plugin). The cv2 path must clean the variable up, or any pytest subset
    running this file before the first qapp-using test dies with a cryptic
    'Fatal Python error: Aborted'."""
    import os

    pts = [[0.05 * i, 0.5 + (0.01 if i % 2 else 0.0)] for i in range(20)]
    simplify_polygon(pts)  # forces the cv2 path when cv2 is installed
    leftover = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    assert "cv2" not in leftover and "opencv" not in leftover


class TestBboxFromPolygon:
    def test_triangle_bounding_box(self):
        # Triangle spanning x∈[0.2,0.8], y∈[0.1,0.7]
        bbox = bbox_from_polygon([[0.2, 0.1], [0.8, 0.1], [0.5, 0.7]])
        cx, cy, w, h = bbox
        assert abs(cx - 0.5) < 1e-9
        assert abs(cy - 0.4) < 1e-9
        assert abs(w - 0.6) < 1e-9
        assert abs(h - 0.6) < 1e-9

    def test_tuple_points(self):
        bbox = bbox_from_polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert bbox == (0.5, 0.5, 1.0, 1.0)

    def test_empty_returns_none(self):
        assert bbox_from_polygon([]) is None


class TestBboxToPolygon:
    def test_rectangle_corners_clockwise(self):
        poly = bbox_to_polygon((0.5, 0.5, 0.4, 0.2))
        # (cx±w/2, cy±h/2): x∈[0.3,0.7], y∈[0.4,0.6]
        assert poly == [[0.3, 0.4], [0.7, 0.4], [0.7, 0.6], [0.3, 0.6]]

    def test_roundtrip_bbox_polygon_bbox(self):
        bbox = (0.4, 0.6, 0.3, 0.5)
        poly = bbox_to_polygon(bbox)
        derived = bbox_from_polygon(poly)
        for a, b in zip(bbox, derived):
            assert abs(a - b) < 1e-9


class TestPointInPolygon:
    def _square(self):
        return [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

    def test_inside(self):
        assert point_in_polygon(0.5, 0.5, self._square()) is True

    def test_outside(self):
        assert point_in_polygon(0.9, 0.5, self._square()) is False
        assert point_in_polygon(0.1, 0.1, self._square()) is False

    def test_degenerate_polygon_is_never_inside(self):
        assert point_in_polygon(0.5, 0.5, [[0.5, 0.5], [0.6, 0.6]]) is False


class TestNearestVertex:
    def _tri(self):
        return [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]

    def test_hits_closest_vertex(self):
        assert nearest_vertex(0.21, 0.19, self._tri(), tolerance=0.05) == 0
        assert nearest_vertex(0.79, 0.21, self._tri(), tolerance=0.05) == 1

    def test_returns_none_beyond_tolerance(self):
        assert nearest_vertex(0.5, 0.5, self._tri(), tolerance=0.01) is None


class TestNearestEdge:
    def _square(self):
        return [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

    def test_hits_top_edge(self):
        # Midpoint of edge 0 (top): (0.5, 0.2)
        assert nearest_edge(0.5, 0.21, self._square(), tolerance=0.05) == 0

    def test_hits_wrapping_edge(self):
        # Left edge wraps from points[3] → points[0]: x≈0.2, y≈0.5
        assert nearest_edge(0.21, 0.5, self._square(), tolerance=0.05) == 3

    def test_returns_none_when_far(self):
        assert nearest_edge(0.5, 0.5, self._square(), tolerance=0.05) is None


class TestSimplifyPolygon:
    def test_short_polygon_unchanged(self):
        tri = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]
        assert simplify_polygon(tri) == tri

    def test_reduces_dense_contour(self):
        # A many-point contour sampled along a square perimeter should collapse
        # near its 4 corners.
        pts = []
        for i in range(25):
            pts.append([0.2 + 0.6 * i / 25, 0.2])  # bottom edge
        for i in range(25):
            pts.append([0.8, 0.2 + 0.6 * i / 25])  # right edge
        for i in range(25):
            pts.append([0.8 - 0.6 * i / 25, 0.8])  # top edge
        for i in range(25):
            pts.append([0.2, 0.8 - 0.6 * i / 25])  # left edge
        simplified = simplify_polygon(pts, epsilon_ratio=0.01)
        assert len(simplified) < len(pts)
        assert len(simplified) >= 3

    def test_output_is_normalized_pairs(self):
        pts = [[0.1 * i, 0.1 * (i % 3)] for i in range(10)]
        out = simplify_polygon(pts)
        for p in out:
            assert len(p) == 2
            assert all(isinstance(v, float) for v in p)

    def test_collinear_run_collapses_to_endpoints_via_fallback(self):
        # Force the pure-Python DP path (no cv2) on a straight run of points:
        # it should keep essentially just the endpoints.
        import src.core.polygon as polymod
        pts = [[0.1 * i, 0.5] for i in range(6)] + [[0.5, 0.9]]
        out = polymod._douglas_peucker(pts, epsilon=0.001)
        assert out[0] == pts[0]
        assert out[-1] == pts[-1]
        assert len(out) < len(pts)

    @staticmethod
    def _jittered_rectangle():
        """Rectangle whose top edge carries a small vertical zigzag.

        The jitter is 0.01 in normalized units: large relative to a normalized
        epsilon, but only ~1 px on a 2000×100 image — so pixel-space DP drops
        it while normalized-space DP keeps it.
        """
        top = [[0.1 + 0.8 * i / 21, 0.2 + (0.01 if i % 2 else 0.0)] for i in range(22)]
        return top + [[0.9, 0.8], [0.1, 0.8]]

    def test_image_size_runs_dp_in_pixel_space(self):
        pts = self._jittered_rectangle()
        norm_space = simplify_polygon(pts)
        pixel_space = simplify_polygon(pts, image_size=(2000, 100))
        # On the wide image the 1-px jitter is noise: pixel-space DP collapses
        # the top edge, normalized-space DP (geometrically distorted) keeps it.
        assert len(pixel_space) < len(norm_space)
        assert len(pixel_space) <= 8

    def test_image_size_output_is_scaled_back_to_normalized(self):
        pts = self._jittered_rectangle()
        out = simplify_polygon(pts, image_size=(2000, 100))
        assert len(out) >= 3
        for x, y in out:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
        # The rectangle's extent survives (corners at x∈[0.1,0.9], y∈[0.2,0.8]).
        xs = [p[0] for p in out]
        ys = [p[1] for p in out]
        assert min(xs) == pytest.approx(0.1, abs=1e-3)
        assert max(xs) == pytest.approx(0.9, abs=1e-3)
        assert min(ys) == pytest.approx(0.2, abs=0.02)
        assert max(ys) == pytest.approx(0.8, abs=1e-3)
