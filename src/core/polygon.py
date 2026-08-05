"""Qt-free polygon geometry helpers for the segment task.

All coordinates are normalized [0, 1] (same convention as ``Annotation``).
These helpers back the canvas polygon-editing UI (Chunk B) and the mask →
polygon ingestion in ``Predictor._run`` — none of them import Qt, PIL, or
torch, so they unit-test without a QApplication or a model.

``simplify_polygon`` lazy-imports cv2 (already a hard ultralytics dependency)
and falls back to a pure-Python Douglas–Peucker when cv2 is unavailable, so a
core import never pays for OpenCV.
"""
from __future__ import annotations

import os


def bbox_from_polygon(
    points: list[list[float]] | list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    """Return the axis-aligned bounding box (cx, cy, w, h) of a polygon.

    Returns ``None`` for an empty point list. Coordinates in and out are
    normalized [0, 1]; the derived bbox is what bbox-based logic (IoU
    conflict detection, viewport culling, stats) consumes for a polygon.
    """
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def bbox_to_polygon(
    bbox: tuple[float, float, float, float],
) -> list[list[float]]:
    """Fold a (cx, cy, w, h) bbox into a 4-corner polygon (TL, TR, BR, BL).

    Used when a bbox-only annotation must be exported to a segment label
    (YOLO-seg forbids mixing 5-field bbox rows with polygon rows in one file,
    so a plain box becomes a rectangular polygon rather than being dropped).
    """
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def point_in_polygon(x: float, y: float, points: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test (even–odd rule).

    ``points`` is a list of [x, y]; the polygon is treated as closed. Returns
    True when (x, y) lies strictly inside (boundary results are unspecified,
    which is fine for hit-testing where an epsilon tolerance is applied by the
    edge/vertex helpers).
    """
    n = len(points)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points[i][0], points[i][1]
        xj, yj = points[j][0], points[j][1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def nearest_vertex(
    x: float, y: float, points: list[list[float]], tolerance: float,
) -> int | None:
    """Return the index of the vertex within ``tolerance`` of (x, y), else None.

    Ties resolve to the closest vertex. Distance is Euclidean in normalized
    coordinate space (the caller passes a tolerance scaled for the current
    zoom).
    """
    best_idx: int | None = None
    best_dist = tolerance
    for i, p in enumerate(points):
        dx = p[0] - x
        dy = p[1] - y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def nearest_edge(
    x: float, y: float, points: list[list[float]], tolerance: float,
) -> int | None:
    """Return the index ``i`` of the edge (points[i] → points[i+1]) whose
    segment passes within ``tolerance`` of (x, y), else None.

    The polygon is closed, so the last edge wraps from ``points[-1]`` to
    ``points[0]``. A new vertex inserted for this edge belongs at position
    ``i + 1``. Ties resolve to the nearest edge.
    """
    n = len(points)
    if n < 2:
        return None
    best_idx: int | None = None
    best_dist = tolerance
    for i in range(n):
        ax, ay = points[i][0], points[i][1]
        bx, by = points[(i + 1) % n][0], points[(i + 1) % n][1]
        dist = _point_segment_distance(x, y, ax, ay, bx, by)
        if dist <= best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float,
) -> float:
    """Shortest distance from point P to segment AB."""
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        # Degenerate segment — distance to the point A.
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def simplify_polygon(
    points: list[list[float]] | list[tuple[float, float]],
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
    image_size: tuple[int, int] | None = None,
) -> list[list[float]]:
    """Douglas–Peucker simplification of a normalized polygon.

    ``epsilon_ratio`` is multiplied by the polygon perimeter to get the DP
    epsilon (≈0.002 keeps 20–40 vertices for a typical mask contour of a few
    hundred points). Input/output are both normalized [0, 1] point lists.

    ``image_size`` (w, h): when given, simplification runs in PIXEL space —
    points are scaled up before DP and scaled back after. Normalized space
    distorts geometry on non-square images (one normalized unit is a
    different pixel length on x vs y, so vertex importance and epsilon lose
    their meaning); callers that know the real size (the predictor) must pass
    it. ``None`` keeps normalized space (fine for unknown sizes).

    Raw mask contours carry hundreds of points; simplification MUST run before
    a polygon is stored (label JSON size, undo deep-copies, canvas hit-testing
    all scale with the vertex count). Fewer than ``min_points`` are returned
    unchanged (nothing to simplify).
    """
    norm_pts = [[float(p[0]), float(p[1])] for p in points]
    if len(norm_pts) <= min_points:
        return norm_pts

    sw = sh = 1.0
    if image_size is not None and image_size[0] > 0 and image_size[1] > 0:
        sw, sh = float(image_size[0]), float(image_size[1])
    pts = [[x * sw, y * sh] for x, y in norm_pts]

    perimeter = 0.0
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        perimeter += ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    if perimeter <= 0.0:
        return norm_pts
    epsilon = epsilon_ratio * perimeter

    simplified = _simplify_cv2(pts, epsilon)
    if simplified is None:
        simplified = _douglas_peucker(pts, epsilon)

    # DP can collapse below a drawable polygon on near-degenerate inputs; keep
    # at least the original when that happens.
    if len(simplified) < min_points:
        return norm_pts
    return [[x / sw, y / sh] for x, y in simplified]


def _strip_cv2_qt_plugin_hijack() -> None:
    """Undo opencv-python's Qt plugin-path hijack.

    Importing cv2 (the full opencv-python build) sets
    ``QT_QPA_PLATFORM_PLUGIN_PATH`` to its bundled Qt plugins; any
    QApplication constructed AFTER that aborts with "Could not load the Qt
    platform plugin xcb" (cv2's Qt is ABI-incompatible with the GUI
    bindings). This app never uses cv2's GUI, so drop the variable when cv2
    set it. The running app is unaffected either way (its QApplication
    predates any cv2 import), but a pytest subset that touches the cv2 path
    before the shared ``qapp`` fixture would die without this.
    """
    path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if path and ("cv2" in path or "opencv" in path):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def _simplify_cv2(pts: list[list[float]], epsilon: float) -> list[list[float]] | None:
    """cv2.approxPolyDP wrapper; returns None if cv2 is unavailable."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    _strip_cv2_qt_plugin_hijack()
    contour = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[float(x), float(y)] for x, y in approx.reshape(-1, 2)]


def _douglas_peucker(pts: list[list[float]], epsilon: float) -> list[list[float]]:
    """Pure-Python Douglas–Peucker fallback for an open point run.

    Operates on the polygon as an open polyline (first→last); the closing edge
    is implicit. Adequate for the few-hundred-point contours we ingest.
    """
    if len(pts) < 3:
        return pts
    dmax = 0.0
    index = 0
    ax, ay = pts[0]
    bx, by = pts[-1]
    for i in range(1, len(pts) - 1):
        d = _point_segment_distance(pts[i][0], pts[i][1], ax, ay, bx, by)
        if d > dmax:
            dmax = d
            index = i
    if dmax > epsilon:
        left = _douglas_peucker(pts[: index + 1], epsilon)
        right = _douglas_peucker(pts[index:], epsilon)
        return left[:-1] + right
    return [pts[0], pts[-1]]
