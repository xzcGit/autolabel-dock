"""Tests for augmentation preview dialog."""
from pathlib import Path

import pytest
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt


def _make_test_image(path: Path, width: int = 200, height: int = 150) -> None:
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(255, 0, 0))  # red
    img.save(str(path), "PNG")


class TestAugmentationPreviewDialog:
    def test_creates(self, qapp, tmp_path):
        from src.ui.augmentation_preview import AugmentationPreviewDialog

        img = tmp_path / "test.png"
        _make_test_image(img)
        params = {"hsv_h": 0.015, "fliplr": 0.5, "degrees": 10.0}
        dlg = AugmentationPreviewDialog(img, params)
        assert dlg.windowTitle() == "数据增强预览"

    def test_apply_augmentation_returns_image(self, qapp, tmp_path):
        from src.ui.augmentation_preview import _apply_augmentation

        img = tmp_path / "test.png"
        _make_test_image(img)
        params = {"fliplr": 1.0, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
                  "degrees": 0, "flipud": 0}
        original = QImage(str(img))
        result = _apply_augmentation(original, params)
        assert not result.isNull()
        assert result.width() > 0

    def test_handles_missing_image(self, qapp, tmp_path):
        from src.ui.augmentation_preview import AugmentationPreviewDialog

        dlg = AugmentationPreviewDialog(tmp_path / "nonexistent.png", {})
        # Should not crash
        assert dlg is not None

    def test_zero_params_returns_copy(self, qapp, tmp_path):
        from src.ui.augmentation_preview import _apply_augmentation

        img = tmp_path / "test.png"
        _make_test_image(img)
        params = {"hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "fliplr": 0,
                  "flipud": 0, "degrees": 0}
        original = QImage(str(img))
        result = _apply_augmentation(original, params)
        assert result.size() == original.size()


class TestTrainPanelAugPreview:
    def test_has_preview_button(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._btn_preview_aug is not None

    def test_get_augmentation_params(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        params = panel.get_augmentation_params()
        assert "hsv_h" in params
        assert "fliplr" in params
        assert "degrees" in params
        assert isinstance(params["hsv_h"], float)

    def test_preview_signal_emits(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        received = []
        panel.preview_augmentation_requested.connect(lambda p: received.append(p))
        panel._on_preview_augmentation()
        assert len(received) == 1
        assert "hsv_h" in received[0]
