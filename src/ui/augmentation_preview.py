"""Augmentation preview dialog — shows effect of augmentation params on a sample image."""
from __future__ import annotations

import random
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QWidget,
    QApplication,
)

from src.ui.theme import text_style
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor, QTransform


class AugmentationPreviewDialog(QDialog):
    """Shows augmented versions of a sample image using current parameters."""

    def __init__(self, image_path: Path | str, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据增强预览")
        self.setMinimumSize(700, 500)
        self._image_path = str(image_path)
        self._params = params
        self._init_ui()
        # Defer generation so the dialog paints first
        QTimer.singleShot(0, self._generate)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Parameter summary
        param_text = "  |  ".join(
            f"{k}: {v}" for k, v in self._params.items() if v != 0
        )
        summary = QLabel(f"增强参数: {param_text}")
        summary.setStyleSheet(text_style("hint"))
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # Container for the 2x2 preview grid
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        layout.addWidget(self._grid_container, 1)

        # Loading label (shown while generating)
        self._loading_label = QLabel("正在生成预览…")
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet(text_style("muted"))
        layout.addWidget(self._loading_label)

        # Refresh button
        self._btn_refresh = QPushButton("重新生成")
        self._btn_refresh.clicked.connect(self._generate)
        layout.addWidget(self._btn_refresh)

    def _generate(self) -> None:
        """Generate augmented previews, keeping UI responsive."""
        original = QImage(self._image_path)
        if original.isNull():
            self._loading_label.setText("无法加载图片")
            return

        self._btn_refresh.setEnabled(False)
        self._clear_grid()
        self._loading_label.show()
        QApplication.processEvents()

        thumb_size = 280

        # Original
        self._grid_layout.addWidget(
            self._make_preview("原图", original, thumb_size), 0, 0
        )

        # Generate 3 augmented versions
        for i in range(3):
            aug = _apply_augmentation(original, self._params)
            row, col = divmod(i + 1, 2)
            self._grid_layout.addWidget(
                self._make_preview(f"增强 #{i+1}", aug, thumb_size), row, col
            )
            QApplication.processEvents()

        self._loading_label.hide()
        self._btn_refresh.setEnabled(True)

    def _clear_grid(self) -> None:
        """Remove all widgets from the preview grid."""
        while self._grid_layout.count() > 0:
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _make_preview(self, title: str, image: QImage, max_size: int) -> QWidget:
        """Create a labeled image preview widget."""
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(text_style("section"))
        lbl_title.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl_title)

        pixmap = QPixmap.fromImage(image).scaled(
            max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl_img)
        return w


def _apply_augmentation(image: QImage, params: dict) -> QImage:
    """Apply random augmentations based on params."""
    result = image.copy()

    # HSV augmentation
    hsv_h = params.get("hsv_h", 0)
    hsv_s = params.get("hsv_s", 0)
    hsv_v = params.get("hsv_v", 0)
    if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
        result = _augment_hsv(result, hsv_h, hsv_s, hsv_v)

    # Horizontal flip
    fliplr = params.get("fliplr", 0)
    if fliplr > 0 and random.random() < fliplr:
        result = result.mirrored(True, False)

    # Vertical flip
    flipud = params.get("flipud", 0)
    if flipud > 0 and random.random() < flipud:
        result = result.mirrored(False, True)

    # Rotation
    degrees = params.get("degrees", 0)
    if degrees > 0:
        angle = random.uniform(-degrees, degrees)
        transform = QTransform().rotate(angle)
        result = result.transformed(transform, Qt.SmoothTransformation)

    return result


def _augment_hsv(image: QImage, h_gain: float, s_gain: float, v_gain: float) -> QImage:
    """Apply HSV color jitter to image."""
    result = image.convertToFormat(QImage.Format_RGB32)
    h_delta = random.uniform(-h_gain, h_gain) * 180
    s_factor = random.uniform(max(0, 1 - s_gain), 1 + s_gain)
    v_factor = random.uniform(max(0, 1 - v_gain), 1 + v_gain)

    w, h = result.width(), result.height()
    # Sample-based approach for performance (process every 2nd pixel for preview)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            c = QColor(result.pixel(x, y))
            hue = c.hueF() * 360 + h_delta
            sat = max(0, min(1, c.saturationF() * s_factor))
            val = max(0, min(1, c.valueF() * v_factor))
            c.setHsvF((hue % 360) / 360, sat, val)
            rgb = c.rgb()
            result.setPixel(x, y, rgb)
            if x + 1 < w:
                result.setPixel(x + 1, y, rgb)
            if y + 1 < h:
                result.setPixel(x, y + 1, rgb)
                if x + 1 < w:
                    result.setPixel(x + 1, y + 1, rgb)
    return result
