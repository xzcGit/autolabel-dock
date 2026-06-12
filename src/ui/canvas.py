"""Annotation canvas widget for image display and annotation editing."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QMenu, QAction, QInputDialog
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPixmap,
    QImage,
    QImageReader,
    QFont,
    QCursor,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
)

from src.core.annotation import Annotation, Keypoint
from src.ui.theme import PALETTE

logger = logging.getLogger(__name__)

# Visual constants
HANDLE_SIZE = 6
KEYPOINT_RADIUS = 5
LABEL_FONT_SIZE = 11
LABEL_PADDING = 3
MIN_SCALE = 0.1
MAX_SCALE = 20.0
ZOOM_FACTOR = 1.15


class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and editing annotations.

    Signals:
        annotation_created(Annotation): New annotation drawn by user.
        annotation_modified(str): Annotation with given ID was moved/resized.
        annotation_selected(str): Annotation with given ID was selected (or None).
        annotation_deleted(str): Annotation with given ID should be deleted.
        class_requested(float, float): Request class picker at pixel position (after drawing).
        annotations_changed(): Any change to annotations occurred.
    """

    annotation_created = pyqtSignal(object)   # Annotation
    annotation_modified = pyqtSignal(str)     # annotation id
    annotation_selected = pyqtSignal(object)  # annotation id or None
    annotation_deleted = pyqtSignal(str)      # annotation id
    annotation_copied = pyqtSignal(str)       # annotation id (for copy via right-click)
    class_requested = pyqtSignal(float, float)
    class_change_requested = pyqtSignal(str, float, float)  # ann_id, px, py
    annotations_changed = pyqtSignal()
    keypoint_attach_requested = pyqtSignal(str, float, float)  # ann_id, px, py
    keypoint_selected = pyqtSignal(str, int)  # ann_id, kp_index

    zoom_changed = pyqtSignal(float)  # current scale factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

        # Image state
        self._image: QPixmap | None = None
        self._image_w: int = 0
        self._image_h: int = 0

        # View transform
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Tool mode: "select", "draw_bbox", "draw_keypoint"
        self.tool_mode: str = "select"

        # Annotations
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._selected_kp_idx: int | None = None
        self._class_colors: dict[str, str] = {}

        # Drawing state
        self._drawing: bool = False
        self._draw_start: tuple[float, float] | None = None  # normalized
        self._draw_current: tuple[float, float] | None = None  # normalized

        # Dragging state (move/resize)
        self._dragging: bool = False
        self._drag_type: str = ""  # "move", "resize_tl", "resize_br", etc., "move_kp"
        self._drag_ann_id: str | None = None
        self._drag_kp_idx: int = -1
        self._drag_start_norm: tuple[float, float] | None = None
        self._drag_ann_snapshot: dict | None = None

        # Panning state
        self._panning: bool = False
        self._pan_start: tuple[float, float] | None = None

        # Conflict pairs: {ann_id: paired_ann_id} (bidirectional)
        self._conflict_pairs: dict[str, str] = {}


    # ── Coordinate transforms ──────────────────────────────────

    def norm_to_pixel(self, nx: float, ny: float) -> tuple[float, float]:
        """Convert normalized [0,1] image coords to widget pixel coords."""
        px = nx * self._image_w * self._scale + self._offset_x
        py = ny * self._image_h * self._scale + self._offset_y
        return px, py

    def pixel_to_norm(self, px: float, py: float) -> tuple[float, float]:
        """Convert widget pixel coords to normalized [0,1] image coords."""
        if self._image_w == 0 or self._image_h == 0 or self._scale == 0:
            return 0.0, 0.0
        nx = (px - self._offset_x) / (self._image_w * self._scale)
        ny = (py - self._offset_y) / (self._image_h * self._scale)
        return nx, ny

    def _clamp_norm(self, nx: float, ny: float) -> tuple[float, float]:
        """Clamp normalized coords to [0, 1]."""
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # ── Public API ─────────────────────────────────────────────

    def load_image(self, path: str) -> None:
        """Load and display an image, fit to window."""
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        qimage = reader.read()
        if qimage.isNull():
            logger.warning("Failed to load image: %s", path)
            return
        self.set_pixmap(QPixmap.fromImage(qimage))

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set a pre-loaded pixmap as the display image."""
        self._image = pixmap
        self._image_w = pixmap.width()
        self._image_h = pixmap.height()
        self._fit_to_window()
        self.update()

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Set the annotations to display."""
        self._annotations = list(annotations)
        self._selected_id = None
        self.update()

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class name -> hex color mapping."""
        self._class_colors = colors
        self.update()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation by ID, or deselect with None."""
        self._selected_id = ann_id
        self._selected_kp_idx = None
        self.annotation_selected.emit(ann_id)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        """Set tool mode: 'select', 'draw_bbox', 'draw_keypoint'."""
        self.tool_mode = mode
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        if mode == "select":
            self.setCursor(Qt.ArrowCursor)
        elif mode in ("draw_bbox", "draw_keypoint"):
            self.setCursor(Qt.CrossCursor)

    def set_locked(self, locked: bool) -> None:
        """Set lock state (no-op, kept for API compatibility)."""
        pass

    @property
    def annotations(self) -> list[Annotation]:
        """Return the current annotations list (mutable reference)."""
        return self._annotations

    @annotations.setter
    def annotations(self, value: list[Annotation]) -> None:
        """Replace annotations list."""
        self._annotations = value
        self.update()

    @property
    def is_locked(self) -> bool:
        """Return whether the canvas is in locked (view-only) mode."""
        return False

    def add_annotation(self, ann: Annotation) -> None:
        """Append an annotation to the canvas and repaint."""
        self._annotations.append(ann)
        self.update()

    def add_annotations(self, anns: list[Annotation]) -> None:
        """Append multiple annotations and repaint once."""
        self._annotations.extend(anns)
        self.update()

    def remove_annotation(self, ann_id: str) -> None:
        """Remove annotation by ID and clear selection."""
        self._annotations = [a for a in self._annotations if a.id != ann_id]
        self._selected_id = None
        self.update()

    def clear_draw_state(self) -> None:
        """Clear in-progress drawing state."""
        self._draw_start = None
        self._draw_current = None
        self.update()

    def clear(self) -> None:
        """Clear image and annotations."""
        self._image = None
        self._image_w = 0
        self._image_h = 0
        self._annotations = []
        self._selected_id = None
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._conflict_pairs.clear()
        self.update()

    # ── Conflict pair management ──────────────────────────────

    def set_conflict_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Set conflict pairs. Each pair is (existing_id, pred_id)."""
        for eid, pid in pairs:
            self._conflict_pairs[eid] = pid
            self._conflict_pairs[pid] = eid
        self.update()

    def resolve_conflict(self, keep_id: str) -> None:
        """Keep one annotation from a conflict pair and remove the other."""
        remove_id = self._conflict_pairs.get(keep_id)
        if not remove_id:
            return
        # Clean up mapping (both directions)
        self._conflict_pairs.pop(keep_id, None)
        self._conflict_pairs.pop(remove_id, None)
        # Remove the losing annotation
        self._annotations = [a for a in self._annotations if a.id != remove_id]
        if self._selected_id == remove_id:
            self._selected_id = None
        self.annotation_deleted.emit(remove_id)
        self.annotations_changed.emit()
        self.update()

    def clear_conflicts(self) -> None:
        """Clear all conflict state."""
        self._conflict_pairs.clear()
        self.update()

    def get_selected_annotation(self) -> Annotation | None:
        """Return the currently selected annotation."""
        if self._selected_id is None:
            return None
        for ann in self._annotations:
            if ann.id == self._selected_id:
                return ann
        return None

    def hit_test(self, px: float, py: float) -> str | None:
        """Find annotation at pixel position. Returns annotation ID or None."""
        nx, ny = self.pixel_to_norm(px, py)

        # Check keypoints first (smaller targets, higher priority)
        kp_radius_norm_x = KEYPOINT_RADIUS / (self._image_w * self._scale) if self._image_w * self._scale > 0 else 0
        kp_radius_norm_y = KEYPOINT_RADIUS / (self._image_h * self._scale) if self._image_h * self._scale > 0 else 0

        for ann in reversed(self._annotations):
            for kp in ann.keypoints:
                if abs(kp.x - nx) < kp_radius_norm_x * 2 and abs(kp.y - ny) < kp_radius_norm_y * 2:
                    return ann.id

        # Check bboxes
        for ann in reversed(self._annotations):
            if ann.bbox:
                cx, cy, w, h = ann.bbox
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    return ann.id

        return None

    def _fit_to_window(self) -> None:
        """Scale and offset so image fits in widget."""
        if self._image_w == 0 or self._image_h == 0:
            return
        ww, wh = self.width(), self.height()
        if ww <= 0 or wh <= 0:
            return
        sx = ww / self._image_w
        sy = wh / self._image_h
        self._scale = min(sx, sy)
        # Center the image
        self._offset_x = (ww - self._image_w * self._scale) / 2
        self._offset_y = (wh - self._image_h * self._scale) / 2

    # ── Paint ──────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(PALETTE["canvas"]))

        if self._image is None:
            painter.setPen(QColor(PALETTE["text_subtle"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图片")
            painter.end()
            return

        # Draw image
        dest = QRectF(
            self._offset_x, self._offset_y,
            self._image_w * self._scale, self._image_h * self._scale,
        )
        painter.drawPixmap(dest.toRect(), self._image)

        # Viewport bounds for culling
        vp_left = 0.0
        vp_top = 0.0
        vp_right = float(self.width())
        vp_bottom = float(self.height())

        # LOD: skip labels at very small zoom
        draw_labels = self._scale >= 0.3

        # Draw annotations with viewport culling
        for ann in self._annotations:
            # Cull: skip if annotation is entirely outside viewport
            if ann.bbox and not self._ann_in_viewport(ann, vp_left, vp_top, vp_right, vp_bottom):
                continue
            is_selected = ann.id == self._selected_id
            color = QColor(self._class_colors.get(ann.class_name, PALETTE["primary"]))
            self._paint_annotation(painter, ann, color, is_selected, draw_labels)

        # Draw in-progress bbox
        if self._drawing and self._draw_start and self._draw_current:
            self._paint_drawing_preview(painter)

        # Zoom level indicator + lock badge
        if self._image is not None:
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            painter.setPen(QColor(PALETTE["text_subtle"]))
            zoom_pct = int(self._scale * 100)
            status_text = f"{zoom_pct}%"
            painter.drawText(8, self.height() - 8, status_text)

        painter.end()

    def _ann_in_viewport(
        self, ann: Annotation, vp_left: float, vp_top: float, vp_right: float, vp_bottom: float
    ) -> bool:
        """Check if annotation bbox overlaps the viewport."""
        if not ann.bbox:
            return True  # keypoint-only annotations always drawn
        cx, cy, w, h = ann.bbox
        x1, y1 = self.norm_to_pixel(cx - w / 2, cy - h / 2)
        x2, y2 = self.norm_to_pixel(cx + w / 2, cy + h / 2)
        # Annotation is outside if entirely to the left, right, above, or below viewport
        if x2 < vp_left or x1 > vp_right or y2 < vp_top or y1 > vp_bottom:
            return False
        return True

    def _paint_annotation(
        self, painter: QPainter, ann: Annotation, color: QColor, selected: bool,
        draw_labels: bool = True,
    ) -> None:
        """Paint a single annotation (bbox + keypoints + label)."""
        in_conflict = ann.id in self._conflict_pairs
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            x1, y1 = self.norm_to_pixel(cx - w / 2, cy - h / 2)
            x2, y2 = self.norm_to_pixel(cx + w / 2, cy + h / 2)

            if in_conflict and not ann.confirmed:
                # Conflict prediction: teal dashed, thicker
                pen = QPen(QColor(PALETTE["teal"]), 3, Qt.DashLine)
            else:
                pen = QPen(color, 2)
                if not ann.confirmed:
                    pen.setStyle(Qt.DashLine)
            if selected:
                pen.setWidth(pen.width() + 1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            # Label background (skip at low zoom for performance)
            if draw_labels or selected:
                label_text = ann.class_name
                if in_conflict:
                    label_text += " \u21c4"
                elif not ann.confirmed:
                    label_text += " \u26a1"
                font = QFont()
                font.setPixelSize(LABEL_FONT_SIZE)
                painter.setFont(font)
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(label_text) + LABEL_PADDING * 2
                th = fm.height() + LABEL_PADDING * 2
                label_rect = QRectF(x1, y1 - th, tw, th)
                if label_rect.top() < 0:
                    label_rect.moveTop(y1)
                bg_color = QColor(color)
                bg_color.setAlpha(200)
                painter.fillRect(label_rect, bg_color)
                painter.setPen(QColor(PALETTE["ink"]))
                painter.drawText(label_rect, Qt.AlignCenter, label_text)

            # Control handles when selected
            if selected:
                self._paint_handles(painter, x1, y1, x2, y2)

        # Keypoints
        for i, kp in enumerate(ann.keypoints):
            px, py = self.norm_to_pixel(kp.x, kp.y)
            is_kp_selected = selected and self._selected_kp_idx == i
            r = KEYPOINT_RADIUS + (3 if is_kp_selected else (2 if selected else 0))

            if kp.visible == 0:
                painter.setPen(QPen(QColor(PALETTE["text_subtle"]), 1))
                painter.setBrush(Qt.NoBrush)
            elif kp.visible == 1:
                painter.setPen(QPen(color, 1))
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))

            if is_kp_selected:
                painter.setPen(QPen(QColor(PALETTE["text"]), 2))

            painter.drawEllipse(QPointF(px, py), r, r)

            # Label for keypoint (show when selected or when individual kp is selected)
            if (selected and draw_labels) or is_kp_selected:
                painter.setPen(QColor(PALETTE["text"]))
                font = QFont()
                font.setPixelSize(10)
                painter.setFont(font)
                vis_text = ["inv", "occ", "vis"][kp.visible]
                label_text = f"{kp.label} ({vis_text})" if is_kp_selected else kp.label
                painter.drawText(int(px + r + 2), int(py - 2), label_text)

    def _paint_handles(self, painter: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
        """Paint resize handles on selected bbox corners."""
        painter.setPen(QPen(QColor(PALETTE["text"]), 1))
        painter.setBrush(QBrush(QColor(PALETTE["primary"])))
        hs = HANDLE_SIZE
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))

    def _paint_drawing_preview(self, painter: QPainter) -> None:
        """Paint the bbox being drawn with size HUD."""
        sx, sy = self.norm_to_pixel(*self._draw_start)
        ex, ey = self.norm_to_pixel(*self._draw_current)
        painter.setPen(QPen(QColor(PALETTE["primary"]), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        x = min(sx, ex)
        y = min(sy, ey)
        w = abs(ex - sx)
        h = abs(ey - sy)
        painter.drawRect(QRectF(x, y, w, h))

        # Size HUD (pixel dimensions)
        if self._image_w > 0 and self._image_h > 0:
            ns = self._draw_start
            nc = self._draw_current
            pw = int(abs(nc[0] - ns[0]) * self._image_w)
            ph = int(abs(nc[1] - ns[1]) * self._image_h)
            size_text = f"{pw} x {ph}"
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            bg = QColor(PALETTE["panel_raised"])
            bg.setAlpha(200)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(size_text) + 8
            th = fm.height() + 4
            label_x = x + w / 2 - tw / 2
            label_y = y + h + 4
            painter.fillRect(QRectF(label_x, label_y, tw, th), bg)
            painter.setPen(QColor(PALETTE["text"]))
            painter.drawText(QRectF(label_x, label_y, tw, th), Qt.AlignCenter, size_text)

    # ── Mouse events ───────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        # Middle button, or Ctrl + Left button → pan
        ctrl_left = (
            event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ControlModifier
        )
        if event.button() == Qt.MiddleButton or ctrl_left:
            self._panning = True
            self._pan_start = (px, py)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.tool_mode == "draw_bbox":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._drawing = True
            self._draw_start = (nx, ny)
            self._draw_current = (nx, ny)

        elif self.tool_mode == "draw_keypoint":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_start = (nx, ny)
            # Don't emit class_requested here — do it on mouse release
            # to avoid popup appearing while mouse button is still pressed

        elif self.tool_mode == "select":
            # Check if clicking a handle first (for selected bbox)
            handle = self._hit_test_handle(px, py)
            if handle:
                self._dragging = True
                self._drag_type = handle
                self._drag_ann_id = self._selected_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
                return

            # Check if clicking a keypoint to drag
            kp_hit = self._hit_test_keypoint(px, py)
            if kp_hit:
                ann_id, kp_idx = kp_hit
                self._dragging = True
                self._drag_type = "move_kp"
                self._drag_ann_id = ann_id
                self._drag_kp_idx = kp_idx
                self._drag_start_norm = self.pixel_to_norm(px, py)
                return

            # Hit test annotations
            hit_id = self.hit_test(px, py)
            if hit_id:
                self.select_annotation(hit_id)
                self._dragging = True
                self._drag_type = "move"
                self._drag_ann_id = hit_id
                self._drag_start_norm = self.pixel_to_norm(px, py)
                ann = self.get_selected_annotation()
                if ann:
                    self._drag_ann_snapshot = ann.to_dict()
            else:
                self.select_annotation(None)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        if self._panning and self._pan_start:
            dx = px - self._pan_start[0]
            dy = py - self._pan_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._pan_start = (px, py)
            self.update()
            return

        if self._drawing and self._draw_start:
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            self.update()
            return

        if self._dragging and self._drag_ann_id:
            nx, ny = self.pixel_to_norm(px, py)
            self._handle_drag(nx, ny)
            self.update()
            return

        # Cursor feedback in select mode
        if self.tool_mode == "select" and self._image is not None:
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
                return
            handle = self._hit_test_handle(px, py)
            if handle:
                if "tl" in handle or "br" in handle:
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.SizeBDiagCursor)
            elif self._hit_test_keypoint(px, py):
                self.setCursor(Qt.SizeAllCursor)
            elif self.hit_test(px, py):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif self._image is not None:
            # Draw modes: Ctrl signals pan availability, otherwise crosshair
            if event.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        px, py = event.x(), event.y()

        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor if self.tool_mode == "select" else Qt.CrossCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self._drawing and self._draw_start and self.tool_mode == "draw_bbox":
            nx, ny = self._clamp_norm(*self.pixel_to_norm(px, py))
            self._draw_current = (nx, ny)
            sx, sy = self._draw_start
            w = abs(nx - sx)
            h = abs(ny - sy)
            if w > 0.01 and h > 0.01:
                self.class_requested.emit(px, py)
            else:
                self._draw_start = None
                self._draw_current = None
            self._drawing = False
            self.update()
            return

        if self.tool_mode == "draw_keypoint" and self._draw_start:
            # Check if inside an existing bbox — attach to it
            hit_id = self.hit_test(px, py)
            if hit_id:
                ann = next((a for a in self._annotations if a.id == hit_id), None)
                if ann and ann.bbox:
                    self.keypoint_attach_requested.emit(hit_id, px, py)
                    return
            # Outside any bbox — standalone keypoint (show class picker)
            self.class_requested.emit(px, py)
            return

        if self._dragging:
            if self._drag_type in ("move", "resize_tl", "resize_tr", "resize_bl", "resize_br", "move_kp"):
                self.annotation_modified.emit(self._drag_ann_id)
            self._dragging = False
            self._drag_type = ""
            self._drag_ann_id = None
            self._drag_ann_snapshot = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / ZOOM_FACTOR
            self._apply_zoom(event.x(), event.y(), factor)

    def _apply_zoom(self, center_px: float, center_py: float, factor: float) -> None:
        """Apply zoom by factor around a pixel center point."""
        old_nx, old_ny = self.pixel_to_norm(center_px, center_py)
        self._scale = max(MIN_SCALE, min(self._scale * factor, MAX_SCALE))
        new_px = old_nx * self._image_w * self._scale + self._offset_x
        new_py = old_ny * self._image_h * self._scale + self._offset_y
        self._offset_x += center_px - new_px
        self._offset_y += center_py - new_py
        self.zoom_changed.emit(self._scale)
        self.update()

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        if self._image is None:
            return
        self._apply_zoom(self.width() / 2, self.height() / 2, ZOOM_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        if self._image is None:
            return
        self._apply_zoom(self.width() / 2, self.height() / 2, 1.0 / ZOOM_FACTOR)

    def zoom_fit(self) -> None:
        """Reset zoom to fit image in window."""
        if self._image is None:
            return
        self._fit_to_window()
        self.zoom_changed.emit(self._scale)
        self.update()

    def contextMenuEvent(self, event) -> None:
        """Show right-click context menu."""
        px, py = event.x(), event.y()

        # Check if right-clicking a keypoint specifically
        kp_hit = self._hit_test_keypoint(px, py)

        hit_id = self.hit_test(px, py)
        if not hit_id:
            return

        self.select_annotation(hit_id)
        ann = self.get_selected_annotation()
        if not ann:
            return

        menu = QMenu(self)

        # Keypoint-specific actions (when right-clicking directly on a keypoint)
        if kp_hit and kp_hit[0] == hit_id:
            kp_idx = kp_hit[1]
            kp = ann.keypoints[kp_idx] if kp_idx < len(ann.keypoints) else None
            if kp:
                vis_names = ["不可见", "被遮挡", "可见"]
                vis_label = vis_names[kp.visible] if kp.visible < 3 else "?"
                kp_header = menu.addAction(f"关键点: {kp.label} ({vis_label})")
                kp_header.setEnabled(False)

                rename_kp = menu.addAction("重命名关键点")
                rename_kp.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self._request_rename_keypoint(aid, ki))

                cycle_vis = menu.addAction("切换可见性")
                cycle_vis.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self.cycle_keypoint_visibility(aid, ki))

                del_kp = menu.addAction("删除关键点")
                del_kp.triggered.connect(
                    lambda _, aid=ann.id, ki=kp_idx: self.remove_keypoint(aid, ki))

                menu.addSeparator()

        # Conflict resolution options
        paired_id = self._conflict_pairs.get(ann.id)
        if paired_id:
            paired_ann = next((a for a in self._annotations if a.id == paired_id), None)
            if paired_ann:
                # Determine which is existing (confirmed) and which is prediction
                if ann.confirmed:
                    existing_ann, pred_ann = ann, paired_ann
                else:
                    existing_ann, pred_ann = paired_ann, ann
                keep_existing = menu.addAction(
                    f"保留确认框 (conf={existing_ann.confidence:.2f})")
                keep_existing.triggered.connect(
                    lambda: self.resolve_conflict(existing_ann.id))
                keep_pred = menu.addAction(
                    f"保留预测框 (conf={pred_ann.confidence:.2f})")
                keep_pred.triggered.connect(
                    lambda: self.resolve_conflict(pred_ann.id))
                menu.addSeparator()

        # Modify class
        change_cls = menu.addAction("修改类别")
        change_cls.triggered.connect(lambda: self.class_change_requested.emit(ann.id, px, py))

        if ann.confirmed:
            unconfirm = menu.addAction("取消确认")
            unconfirm.triggered.connect(lambda: self._toggle_confirm(ann, False))
        else:
            confirm = menu.addAction("确认")
            confirm.triggered.connect(lambda: self._toggle_confirm(ann, True))

        menu.addSeparator()

        copy_ann = menu.addAction("复制标注 (Ctrl+C)")
        copy_ann.triggered.connect(lambda: self.annotation_copied.emit(ann.id))

        delete = menu.addAction("删除")
        delete.triggered.connect(lambda: self.annotation_deleted.emit(ann.id))

        menu.exec_(event.globalPos())

    def resizeEvent(self, event: QResizeEvent) -> None:
        if self._image and not self._draw_start:
            self._fit_to_window()
        super().resizeEvent(event)

    def keyPressEvent(self, event) -> None:
        """Let key events propagate to parent (LabelPanel) for handling."""
        event.ignore()

    # ── Drag helpers ───────────────────────────────────────────

    def _handle_drag(self, nx: float, ny: float) -> None:
        """Handle ongoing drag operation."""
        if not self._drag_ann_id or not self._drag_start_norm:
            return

        ann = None
        for a in self._annotations:
            if a.id == self._drag_ann_id:
                ann = a
                break
        if ann is None:
            return

        dx = nx - self._drag_start_norm[0]
        dy = ny - self._drag_start_norm[1]

        if self._drag_type == "move" and ann.bbox and self._drag_ann_snapshot:
            orig_bbox = self._drag_ann_snapshot["bbox"]
            new_cx = orig_bbox[0] + dx
            new_cy = orig_bbox[1] + dy
            w, h = orig_bbox[2], orig_bbox[3]
            new_cx = max(w / 2, min(1.0 - w / 2, new_cx))
            new_cy = max(h / 2, min(1.0 - h / 2, new_cy))
            ann.bbox = (new_cx, new_cy, w, h)
            # Move keypoints by same offset
            if "keypoints" in self._drag_ann_snapshot:
                for i, kp_dict in enumerate(self._drag_ann_snapshot["keypoints"]):
                    if i < len(ann.keypoints):
                        ann.keypoints[i].x = max(0, min(1, kp_dict["x"] + dx))
                        ann.keypoints[i].y = max(0, min(1, kp_dict["y"] + dy))
            if not ann.confirmed:
                ann.confirmed = True
            self.annotations_changed.emit()

        elif self._drag_type == "move_kp":
            if 0 <= self._drag_kp_idx < len(ann.keypoints):
                ann.keypoints[self._drag_kp_idx].x = max(0.0, min(1.0, nx))
                ann.keypoints[self._drag_kp_idx].y = max(0.0, min(1.0, ny))
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotations_changed.emit()

        elif self._drag_type.startswith("resize_") and ann.bbox and self._drag_ann_snapshot:
            orig_bbox = self._drag_ann_snapshot["bbox"]
            ocx, ocy, ow, oh = orig_bbox
            ox1, oy1 = ocx - ow / 2, ocy - oh / 2
            ox2, oy2 = ocx + ow / 2, ocy + oh / 2

            if "tl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - 0.01, ox1 + dx))
                oy1 = max(0.0, min(oy2 - 0.01, oy1 + dy))
            elif "tr" in self._drag_type:
                ox2 = max(ox1 + 0.01, min(1.0, ox2 + dx))
                oy1 = max(0.0, min(oy2 - 0.01, oy1 + dy))
            elif "bl" in self._drag_type:
                ox1 = max(0.0, min(ox2 - 0.01, ox1 + dx))
                oy2 = max(oy1 + 0.01, min(1.0, oy2 + dy))
            elif "br" in self._drag_type:
                ox2 = max(ox1 + 0.01, min(1.0, ox2 + dx))
                oy2 = max(oy1 + 0.01, min(1.0, oy2 + dy))

            ann.bbox = ((ox1 + ox2) / 2, (oy1 + oy2) / 2, ox2 - ox1, oy2 - oy1)
            if not ann.confirmed:
                ann.confirmed = True
            self.annotations_changed.emit()

    def _hit_test_handle(self, px: float, py: float) -> str | None:
        """Check if pixel pos hits a resize handle on the selected bbox."""
        if not self._selected_id:
            return None
        ann = self.get_selected_annotation()
        if not ann or not ann.bbox:
            return None

        cx, cy, w, h = ann.bbox
        corners = {
            "resize_tl": (cx - w / 2, cy - h / 2),
            "resize_tr": (cx + w / 2, cy - h / 2),
            "resize_bl": (cx - w / 2, cy + h / 2),
            "resize_br": (cx + w / 2, cy + h / 2),
        }
        for handle_name, (nx, ny) in corners.items():
            hpx, hpy = self.norm_to_pixel(nx, ny)
            if abs(px - hpx) <= HANDLE_SIZE + 2 and abs(py - hpy) <= HANDLE_SIZE + 2:
                return handle_name
        return None

    def _hit_test_keypoint(self, px: float, py: float) -> tuple[str, int] | None:
        """Check if pixel pos hits a keypoint. Returns (ann_id, kp_index) or None."""
        for ann in reversed(self._annotations):
            for i, kp in enumerate(ann.keypoints):
                kpx, kpy = self.norm_to_pixel(kp.x, kp.y)
                if abs(px - kpx) <= KEYPOINT_RADIUS + 4 and abs(py - kpy) <= KEYPOINT_RADIUS + 4:
                    return ann.id, i
        return None

    def _toggle_confirm(self, ann: Annotation, confirmed: bool) -> None:
        ann.confirmed = confirmed
        self.annotations_changed.emit()
        self.update()

    def _request_rename_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Show input dialog to rename a keypoint."""
        ann = next((a for a in self._annotations if a.id == ann_id), None)
        if not ann or kp_idx >= len(ann.keypoints):
            return
        old_label = ann.keypoints[kp_idx].label
        new_label, ok = QInputDialog.getText(
            self, "重命名关键点", "标签:", text=old_label)
        if ok and new_label.strip():
            self.rename_keypoint(ann_id, kp_idx, new_label.strip())

    # ── Public helpers for external bbox/kp creation ───────────

    def create_bbox_from_draw(self, class_name: str, class_id: int) -> Annotation | None:
        """Create a bbox annotation from the last draw operation."""
        if not self._draw_start or not self._draw_current:
            return None
        sx, sy = self._draw_start
        ex, ey = self._draw_current
        w = abs(ex - sx)
        h = abs(ey - sy)
        if w < 0.01 or h < 0.01:
            return None
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            bbox=(cx, cy, w, h),
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self._draw_current = None
        self.update()
        return ann

    def create_keypoint_at(
        self, class_name: str, class_id: int, label: str = "point"
    ) -> Annotation | None:
        """Create a keypoint annotation at the stored draw position."""
        if not self._draw_start:
            return None
        nx, ny = self._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        ann = Annotation(
            class_name=class_name,
            class_id=class_id,
            keypoints=[kp],
            confirmed=True,
            source="manual",
        )
        self._annotations.append(ann)
        self.select_annotation(ann.id)
        self.annotation_created.emit(ann)
        self.annotations_changed.emit()
        self._draw_start = None
        self.update()
        return ann

    def select_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Select a specific keypoint within an annotation."""
        self._selected_id = ann_id
        self._selected_kp_idx = kp_idx
        self.annotation_selected.emit(ann_id)
        self.keypoint_selected.emit(ann_id, kp_idx)
        self.update()

    def add_keypoint_to_annotation(self, ann_id: str, kp: Keypoint) -> None:
        """Append a keypoint to an existing annotation."""
        for ann in self._annotations:
            if ann.id == ann_id:
                ann.keypoints.append(kp)
                if not ann.confirmed:
                    ann.confirmed = True
                self.annotation_modified.emit(ann_id)
                self.annotations_changed.emit()
                self.update()
                return

    def remove_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Remove a single keypoint from an annotation.

        If the annotation has no bbox and this is the last keypoint, remove the annotation.
        """
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints.pop(kp_idx)
                    if not ann.bbox and not ann.keypoints:
                        self._annotations = [a for a in self._annotations if a.id != ann_id]
                        self.annotation_deleted.emit(ann_id)
                    else:
                        self.annotation_modified.emit(ann_id)
                    if self._selected_kp_idx is not None and self._selected_kp_idx >= len(ann.keypoints):
                        self._selected_kp_idx = None
                    self.annotations_changed.emit()
                    self.update()
                return

    def rename_keypoint(self, ann_id: str, kp_idx: int, new_label: str) -> None:
        """Rename a keypoint's label."""
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints[kp_idx].label = new_label
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return

    def cycle_keypoint_visibility(self, ann_id: str, kp_idx: int) -> None:
        """Cycle keypoint visibility: 0 → 1 → 2 → 0."""
        for ann in self._annotations:
            if ann.id == ann_id:
                if 0 <= kp_idx < len(ann.keypoints):
                    ann.keypoints[kp_idx].visible = (ann.keypoints[kp_idx].visible + 1) % 3
                    self.annotation_modified.emit(ann_id)
                    self.annotations_changed.emit()
                    self.update()
                return
