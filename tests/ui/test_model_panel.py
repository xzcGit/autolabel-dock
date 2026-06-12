"""Tests for ModelPanel."""
import pytest
from PyQt5.QtCore import Qt


class TestModelPanel:
    def test_creates(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel is not None

    def test_has_model_list(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._model_list is not None

    def test_set_models(self, qapp):
        from src.ui.model_panel import ModelPanel
        from src.engine.model_manager import ModelInfo

        panel = ModelPanel()
        models = [
            ModelInfo(name="det-v1", path="models/det/best.pt", task="detect",
                      base_model="yolov8n.pt", classes=["cat", "dog"]),
            ModelInfo(name="pose-v1", path="models/pose/best.pt", task="pose",
                      base_model="yolov8n-pose.pt", classes=["person"]),
        ]
        panel.set_models(models)
        assert panel._model_list.count() == 2

    def test_has_threshold_controls(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._conf_spin is not None
        assert panel._iou_spin is not None
        assert panel._class_match_mode_combo is not None

    def test_get_thresholds(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        panel._conf_spin.setValue(0.6)
        panel._iou_spin.setValue(0.5)
        assert panel.get_conf_threshold() == 0.6
        assert panel.get_iou_threshold() == 0.5

    def test_displays_backend_info_when_model_selected(self, qapp):
        from src.ui.model_panel import ModelPanel
        from src.engine.model_manager import ModelInfo

        panel = ModelPanel()
        models = [
            ModelInfo(
                name="onnx-model",
                path="models/imported/model.onnx",
                task="detect",
                base_model="imported",
                classes=["person"],
                backend_id="ultralytics",
                model_format="onnx",
                backend_version="8.3.0",
            ),
        ]
        panel.set_models(models)
        panel._model_list.setCurrentRow(0)

        backend_text = panel._detail_backend.text()
        assert "ultralytics" in backend_text
        assert "onnx" in backend_text
        assert "8.3.0" in backend_text

    def test_backend_info_omits_pt_format_suffix(self, qapp):
        from src.ui.model_panel import ModelPanel
        from src.engine.model_manager import ModelInfo

        panel = ModelPanel()
        models = [
            ModelInfo(
                name="default-model",
                path="models/best.pt",
                task="detect",
                base_model="yolov8n.pt",
                classes=["cat"],
                backend_id="ultralytics",
                model_format="pt",
                backend_version="8.2.69",
            ),
        ]
        panel.set_models(models)
        panel._model_list.setCurrentRow(0)

        backend_text = panel._detail_backend.text()
        assert "ultralytics" in backend_text
        assert "8.2.69" in backend_text
        assert "(pt)" not in backend_text  # .pt is default, don't clutter

    def test_class_match_mode_defaults_to_class_id(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel.get_class_match_mode() == "class_id"

    def test_get_class_match_mode(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        panel._class_match_mode_combo.setCurrentIndex(1)
        assert panel.get_class_match_mode() == "class_name"

    def test_has_load_delete_buttons(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._btn_load is not None
        assert panel._btn_delete is not None

    def test_has_compare_button(self, qapp):
        from src.ui.model_panel import ModelPanel

        panel = ModelPanel()
        assert panel._btn_compare is not None

    def test_model_list_multi_select(self, qapp):
        from src.ui.model_panel import ModelPanel
        from PyQt5.QtWidgets import QAbstractItemView

        panel = ModelPanel()
        assert panel._model_list.selectionMode() == QAbstractItemView.ExtendedSelection


class TestModelCompareDialog:
    def test_creates_with_models(self, qapp):
        from src.ui.model_panel import ModelCompareDialog
        from src.engine.model_manager import ModelInfo

        m1 = ModelInfo(name="v1", path="m1.pt", task="detect", base_model="yolov8n.pt",
                       classes=["cat"], metrics={"mAP50": 0.85, "mAP50-95": 0.65}, epochs=100)
        m2 = ModelInfo(name="v2", path="m2.pt", task="detect", base_model="yolov8s.pt",
                       classes=["cat"], metrics={"mAP50": 0.90, "mAP50-95": 0.70}, epochs=200)
        dlg = ModelCompareDialog([m1, m2])
        assert dlg.windowTitle().startswith("模型对比")

    def test_handles_missing_metrics(self, qapp):
        from src.ui.model_panel import ModelCompareDialog
        from src.engine.model_manager import ModelInfo

        m1 = ModelInfo(name="v1", path="m1.pt", task="detect", base_model="yolov8n.pt",
                       classes=["cat"], metrics={"mAP50": 0.85})
        m2 = ModelInfo(name="v2", path="m2.pt", task="detect", base_model="yolov8s.pt",
                       classes=["cat"], metrics={"recall": 0.9})
        dlg = ModelCompareDialog([m1, m2])
        # Should not crash with mismatched metric keys
        assert dlg is not None


class TestFormatTrainParams:
    def test_empty_params_returns_dash(self, qapp):
        from src.ui.model_panel import _format_train_params

        assert _format_train_params({}, "detect") == "无"

    def test_base_groups_render(self, qapp):
        from src.ui.model_panel import _format_train_params

        params = {"epochs": 100, "batch": 16, "optimizer": "auto", "lr0": 0.01,
                  "hsv_h": 0.015, "scale": 0.5}
        text = _format_train_params(params, "detect")

        assert "[基础]" in text
        assert "epochs=100" in text
        assert "batch=16" in text
        assert "[优化器]" in text
        assert "optimizer=auto" in text
        assert "lr0=0.01" in text
        assert "[数据增强（通用）]" in text

    def test_detect_aug_shown_only_when_flag_active(self, qapp):
        from src.ui.model_panel import _format_train_params

        on = _format_train_params(
            {"epochs": 1, "include_detect_params": True, "mosaic": 0.7, "degrees": 5.0},
            "detect",
        )
        off = _format_train_params(
            {"epochs": 1, "include_detect_params": False, "mosaic": 0.7, "degrees": 5.0},
            "detect",
        )

        assert "[Detect 增强]" in on
        assert "mosaic=0.7" in on
        assert "[Detect 增强]" not in off
        assert "mosaic" not in off

    def test_classify_aug_shown_only_when_flag_active(self, qapp):
        from src.ui.model_panel import _format_train_params

        on = _format_train_params(
            {"epochs": 1, "include_classify_params": True, "erasing": 0.4, "auto_augment": "randaugment", "dropout": 0.3},
            "classify",
        )
        off = _format_train_params(
            {"epochs": 1, "include_classify_params": False, "erasing": 0.4, "auto_augment": "randaugment", "dropout": 0.3},
            "classify",
        )

        assert "[Classify 增强]" in on
        assert "erasing=0.4" in on
        assert "auto_augment=randaugment" in on
        assert "dropout=0.3" in on
        assert "[Classify 增强]" not in off

    def test_pose_kpt_shape_always_shown_for_pose(self, qapp):
        from src.ui.model_panel import _format_train_params

        text = _format_train_params(
            {"epochs": 1, "include_pose_params": False, "kpt_shape": [17, 3]},
            "pose",
        )

        assert "[Pose 参数]" in text
        assert "kpt_shape=[17, 3]" in text
        # pose / kobj gated by flag
        assert "pose=" not in text
        assert "kobj=" not in text

    def test_pose_loss_keys_appear_when_flag_active(self, qapp):
        from src.ui.model_panel import _format_train_params

        text = _format_train_params(
            {"epochs": 1, "include_pose_params": True,
             "pose": 12.0, "kobj": 1.0, "kpt_shape": [17, 3]},
            "pose",
        )

        assert "kpt_shape=[17, 3]" in text
        assert "pose=12" in text
        assert "kobj=1" in text

    def test_pose_section_hidden_for_non_pose_task(self, qapp):
        from src.ui.model_panel import _format_train_params

        text = _format_train_params(
            {"epochs": 1, "kpt_shape": [17, 3], "include_pose_params": True, "pose": 12.0},
            "detect",
        )

        assert "[Pose 参数]" not in text
        assert "kpt_shape" not in text


class TestModelPanelDetailDisplay:
    def test_train_params_label_populated_on_select(self, qapp):
        from src.ui.model_panel import ModelPanel
        from src.engine.model_manager import ModelInfo

        panel = ModelPanel()
        info = ModelInfo(
            name="m1", path="m.pt", task="detect", base_model="yolov8n.pt",
            classes=["cat"],
            train_params={"epochs": 50, "batch": 4, "lr0": 0.01},
        )
        panel.set_models([info])
        panel._model_list.setCurrentRow(0)

        text = panel._detail_train_params.text()
        assert "epochs=50" in text
        assert "batch=4" in text

    def test_train_params_label_shows_dash_for_legacy_models(self, qapp):
        from src.ui.model_panel import ModelPanel
        from src.engine.model_manager import ModelInfo

        panel = ModelPanel()
        info = ModelInfo(
            name="m1", path="m.pt", task="detect", base_model="yolov8n.pt",
            classes=["cat"],
        )
        panel.set_models([info])
        panel._model_list.setCurrentRow(0)

        assert panel._detail_train_params.text() == "无"
