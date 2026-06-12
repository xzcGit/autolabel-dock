"""Tests for TrainPanel."""
import pytest
from PyQt5.QtCore import Qt


class TestTrainPanel:
    def test_creates(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel is not None

    def test_has_task_selector(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._task_combo is not None
        # Should have detect, classify, pose
        items = [panel._task_combo.itemText(i) for i in range(panel._task_combo.count())]
        assert "detect" in items
        assert "classify" in items
        assert "pose" in items

    def test_has_hyperparameters(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._epochs_spin is not None
        assert panel._batch_spin is not None
        assert panel._imgsz_spin is not None
        assert panel._lr0_spin is not None
        assert panel._freeze_spin is not None
        assert panel._freeze_default_check is not None
        assert panel._workers_spin is not None
        assert panel._patience_spin is not None

    def test_has_start_stop_buttons(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._btn_start is not None
        assert panel._btn_stop is not None

    def test_has_log_display(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._log_text is not None

    def test_get_train_config(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(50)
        panel._batch_spin.setValue(8)
        panel._imgsz_spin.setValue(320)
        panel._workers_spin.setValue(3)
        panel._patience_spin.setValue(12)

        config = panel.get_train_config(data_yaml="/tmp/data.yaml", model="yolov8n.pt")
        assert config.task == "detect"
        assert config.epochs == 50
        assert config.batch == 8
        assert config.imgsz == 320
        assert config.freeze is None
        assert config.workers == 3
        assert config.patience == 12
        assert config.data_yaml == "/tmp/data.yaml"

    def test_get_train_config_collects_classify_only_params(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("classify")
        panel._freeze_default_check.setChecked(False)
        panel._freeze_spin.setValue(4)
        panel._workers_spin.setValue(6)
        panel._patience_spin.setValue(18)
        panel._erasing_spin.setValue(0.7)
        panel._auto_augment_combo.setCurrentText("augmix")
        panel._dropout_spin.setValue(0.25)
        panel._include_classify_params_check.setChecked(True)

        config = panel.get_train_config(data_yaml="/tmp/data.yaml", model="yolov8n-cls.pt")

        assert config.task == "classify"
        assert config.freeze == 4
        assert config.workers == 6
        assert config.patience == 18
        assert config.erasing == 0.7
        assert config.auto_augment == "augmix"
        assert config.dropout == 0.25
        assert config.include_classify_params is True

    def test_append_log(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.append_log("Epoch 1/50")
        assert "Epoch 1/50" in panel._log_text.toPlainText()

    def test_update_epoch_metrics(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.update_epoch({"epoch": 0, "train_loss": 1.5})
        assert "epoch: 0" in panel._log_text.toPlainText().lower() or "Epoch" in panel._log_text.toPlainText()

    def test_pose_params_visible_on_pose_task(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("pose")
        assert not panel._pose_group.isHidden()

    def test_parameter_groups_stay_visible_on_detect_task(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        assert not panel._common_geo_group.isHidden()
        assert not panel._detect_geo_group.isHidden()
        assert not panel._detect_mix_group.isHidden()
        assert not panel._classify_aug_group.isHidden()
        assert not panel._pose_group.isHidden()

    def test_has_preset_combo(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._preset_combo is not None
        # Only "默认" is built-in; user templates appear via registry injection.
        items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
        assert items == ["默认"]

    def test_preset_default_restores_dataclass_defaults(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        # Mutate some fields away from defaults
        panel._epochs_spin.setValue(7)
        panel._batch_spin.setValue(64)
        panel._include_detect_params_check.setChecked(True)

        # Re-applying 默认 should reset everything; call slot directly because
        # setCurrentText("默认") is a no-op when 默认 is already the current selection.
        panel._on_preset_changed("默认")

        assert panel._epochs_spin.value() == 100
        assert panel._batch_spin.value() == 16
        assert panel._workers_spin.value() == 8
        assert panel._patience_spin.value() == 100
        assert panel._freeze_default_check.isChecked()
        assert panel._include_detect_params_check.isChecked() is False
        assert panel._include_classify_params_check.isChecked() is False
        assert panel._include_pose_params_check.isChecked() is False

    def test_group_include_checkboxes_exist_and_default_off(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._include_detect_params_check.isChecked() is False
        assert panel._include_classify_params_check.isChecked() is False
        assert panel._include_pose_params_check.isChecked() is False

    def test_group_include_checkboxes_are_interactive_at_startup(self, qapp):
        """All three include-in-training checkboxes must be enabled and visible at startup,
        otherwise the user can't toggle them. Regression for the case where placing
        _include_detect_params_check inside a checkable-collapsible QGroupBox left it disabled
        and hidden because Qt cascades disabled state to children of unchecked groupboxes."""
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._include_detect_params_check.isEnabled()
        assert panel._include_detect_params_check.isVisibleTo(panel)
        assert panel._include_classify_params_check.isEnabled()
        assert panel._include_classify_params_check.isVisibleTo(panel)
        assert panel._include_pose_params_check.isEnabled()
        assert panel._include_pose_params_check.isVisibleTo(panel)
        # Spinboxes that share the detect group must also be interactive
        assert panel._degrees_spin.isEnabled()
        assert panel._translate_spin.isEnabled()

    def test_advanced_optimizer_params_editable_at_startup(self, qapp):
        """Advanced optimizer spinboxes must be editable at startup. Regression for
        opt_group originally being a checkable-collapsible QGroupBox that disabled
        children when collapsed."""
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        assert panel._lrf_spin.isEnabled()
        assert panel._momentum_spin.isEnabled()
        assert panel._weight_decay_spin.isEnabled()
        assert panel._warmup_epochs_spin.isEnabled()
        assert panel._warmup_momentum_spin.isEnabled()
        assert panel._warmup_bias_lr_spin.isEnabled()

    def test_all_parameter_groups_stay_visible_across_tasks(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        for task in ("detect", "classify", "pose"):
            panel._task_combo.setCurrentText(task)
            assert not panel._common_geo_group.isHidden()
            assert not panel._detect_geo_group.isHidden()
            assert not panel._detect_mix_group.isHidden()
            assert not panel._classify_aug_group.isHidden()
            assert not panel._pose_group.isHidden()

    def test_get_train_config_collects_group_flags(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._include_detect_params_check.setChecked(True)
        panel._include_classify_params_check.setChecked(True)
        panel._include_pose_params_check.setChecked(True)
        panel._auto_augment_combo.setCurrentText("autoaugment")

        config = panel.get_train_config(data_yaml="/tmp/data.yaml", model="yolov8n.pt")

        assert config.include_detect_params is True
        assert config.include_classify_params is True
        assert config.include_pose_params is True
        assert config.auto_augment == "autoaugment"

    def test_preview_uses_effective_task_specific_augmentations(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._degrees_spin.setValue(15.0)
        panel._mosaic_spin.setValue(0.6)
        panel._erasing_spin.setValue(0.8)
        panel._include_classify_params_check.setChecked(True)

        received = []
        panel.preview_augmentation_requested.connect(lambda params: received.append(params))
        panel._on_preview_augmentation()

        assert len(received) == 1
        assert received[0]["erasing"] == 0.8
        assert "degrees" not in received[0]
        assert "mosaic" not in received[0]

    def test_preview_merges_multiple_enabled_groups(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._degrees_spin.setValue(15.0)
        panel._mosaic_spin.setValue(0.6)
        panel._erasing_spin.setValue(0.8)
        panel._include_detect_params_check.setChecked(True)
        panel._include_classify_params_check.setChecked(True)

        received = []
        panel.preview_augmentation_requested.connect(lambda params: received.append(params))
        panel._on_preview_augmentation()

        assert len(received) == 1
        assert received[0]["degrees"] == 15.0
        assert received[0]["mosaic"] == 0.6
        assert received[0]["erasing"] == 0.8

    def test_set_template_registry_repopulates_combo_with_user_templates(self, qapp, tmp_path):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="my-detect", task="detect", params={}, created_at="t"))
        reg.upsert(TrainTemplate(name="my-classify", task="classify", params={}, created_at="t"))

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)

        items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
        assert items[0] == "默认"
        assert "my-detect" in items
        assert "my-classify" not in items  # filtered out by task

    def test_set_template_registry_with_none_falls_back_to_default_only(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.set_template_registry(None)
        items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
        assert items == ["默认"]

    def test_apply_template_params_writes_only_present_keys(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(50)
        panel._batch_spin.setValue(8)
        panel._lr0_spin.setValue(0.005)

        panel.apply_template_params({"epochs": 200, "mosaic": 0.7, "include_detect_params": True})

        assert panel._epochs_spin.value() == 200       # written
        assert panel._batch_spin.value() == 8           # untouched
        assert panel._lr0_spin.value() == 0.005         # untouched
        assert panel._mosaic_spin.value() == 0.7
        assert panel._include_detect_params_check.isChecked() is True

    def test_apply_template_params_ignores_unknown_keys(self, qapp, caplog):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        with caplog.at_level("WARNING"):
            panel.apply_template_params({"epochs": 50, "totally_made_up_key": 999})
        assert panel._epochs_spin.value() == 50
        assert any("totally_made_up_key" in rec.getMessage() for rec in caplog.records)

    def test_apply_template_params_handles_freeze_none(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._freeze_default_check.setChecked(False)
        panel._freeze_spin.setValue(5)

        panel.apply_template_params({"freeze": None})

        assert panel._freeze_default_check.isChecked() is True

    def test_apply_template_params_handles_freeze_int(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.apply_template_params({"freeze": 7})
        assert panel._freeze_default_check.isChecked() is False
        assert panel._freeze_spin.value() == 7

    def test_apply_template_params_handles_auto_augment_string(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.apply_template_params({"auto_augment": "augmix"})
        assert panel._auto_augment_combo.currentText() == "augmix"

        panel.apply_template_params({"auto_augment": ""})
        assert panel._auto_augment_combo.currentText() == "none"

    def test_apply_template_params_handles_model_string(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel.apply_template_params({"model": "yolov8s.pt"})
        assert panel._model_combo.currentText() == "yolov8s.pt"

    def test_apply_template_params_handles_pose_kpt_shape(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("pose")
        panel.apply_template_params({"kpt_shape": [21, 2]})
        assert panel._kpt_num_spin.value() == 21
        assert panel._kpt_dim_spin.value() == 2

    def test_get_train_template_params_extracts_task_relevant_subset(self, qapp):
        from src.ui.train_panel import TrainPanel

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(42)
        panel._mosaic_spin.setValue(0.6)
        panel._include_detect_params_check.setChecked(True)
        panel._erasing_spin.setValue(0.9)  # classify-only, should be filtered out

        params = panel.get_train_template_params()

        assert params["epochs"] == 42
        assert params["mosaic"] == 0.6
        assert params["include_detect_params"] is True
        assert "erasing" not in params
        assert "task" not in params

    def test_selecting_user_template_applies_its_params(self, qapp, tmp_path):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(
            name="big-batch",
            task="detect",
            params={"epochs": 250, "batch": 64, "include_detect_params": True, "mosaic": 0.9},
            created_at="t",
        ))

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)

        panel._preset_combo.setCurrentText("big-batch")

        assert panel._epochs_spin.value() == 250
        assert panel._batch_spin.value() == 64
        assert panel._include_detect_params_check.isChecked() is True
        assert panel._mosaic_spin.value() == 0.9

    def test_selecting_default_resets_to_dataclass_defaults(self, qapp, tmp_path):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(
            name="big-batch",
            task="detect",
            params={"epochs": 250, "batch": 64},
            created_at="t",
        ))

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)
        panel._preset_combo.setCurrentText("big-batch")

        panel._preset_combo.setCurrentText("默认")

        assert panel._epochs_spin.value() == 100
        assert panel._batch_spin.value() == 16

    def test_save_template_button_invokes_registry_upsert(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry

        reg = TemplateRegistry(tmp_path / "tpl.json")
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(77)
        panel.set_template_registry(reg)

        monkeypatch.setattr(
            "src.ui.train_panel.QInputDialog.getText",
            lambda *args, **kwargs: ("我的模板", True),
        )

        panel._on_save_template()

        saved = reg.get("我的模板", "detect")
        assert saved is not None
        assert saved.params["epochs"] == 77
        items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
        assert "我的模板" in items
        assert panel._preset_combo.currentText() == "我的模板"

    def test_save_template_with_empty_name_rejected(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry

        reg = TemplateRegistry(tmp_path / "tpl.json")
        panel = TrainPanel()
        panel.set_template_registry(reg)

        monkeypatch.setattr(
            "src.ui.train_panel.QInputDialog.getText",
            lambda *args, **kwargs: ("   ", True),
        )
        warning_calls: list = []
        monkeypatch.setattr(
            "src.ui.train_panel.QMessageBox.warning",
            lambda *args, **kwargs: warning_calls.append(args) or 0,
        )

        panel._on_save_template()

        assert reg.list(task="detect") == [t for t in reg.list(task="detect") if t.builtin]
        assert warning_calls, "expected QMessageBox.warning to be called for empty name"

    def test_save_template_with_existing_name_asks_to_overwrite(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate
        from PyQt5.QtWidgets import QMessageBox

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="dup", task="detect", params={"epochs": 1}, created_at="t"))
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(999)
        panel.set_template_registry(reg)

        monkeypatch.setattr(
            "src.ui.train_panel.QInputDialog.getText",
            lambda *args, **kwargs: ("dup", True),
        )
        monkeypatch.setattr(
            "src.ui.train_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Yes,
        )

        panel._on_save_template()

        assert reg.get("dup", "detect").params["epochs"] == 999

    def test_save_template_overwrite_cancelled_keeps_old(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate
        from PyQt5.QtWidgets import QMessageBox

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="dup", task="detect", params={"epochs": 1}, created_at="t"))
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(999)
        panel.set_template_registry(reg)

        monkeypatch.setattr(
            "src.ui.train_panel.QInputDialog.getText",
            lambda *args, **kwargs: ("dup", True),
        )
        monkeypatch.setattr(
            "src.ui.train_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.No,
        )

        panel._on_save_template()

        assert reg.get("dup", "detect").params == {"epochs": 1}

    def test_save_template_persists_to_disk(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry

        path = tmp_path / "tpl.json"
        reg = TemplateRegistry(path)
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel._epochs_spin.setValue(33)
        panel.set_template_registry(reg)

        monkeypatch.setattr(
            "src.ui.train_panel.QInputDialog.getText",
            lambda *args, **kwargs: ("disk-test", True),
        )
        panel._on_save_template()

        assert path.exists()
        reg2 = TemplateRegistry(path)
        reg2.load()
        loaded = reg2.get("disk-test", "detect")
        assert loaded is not None
        assert loaded.params["epochs"] == 33

    def test_delete_template_button_disabled_for_default(self, qapp, tmp_path):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="my", task="detect", params={}, created_at="t"))
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)

        panel._preset_combo.setCurrentText("默认")
        assert panel._btn_delete_template.isEnabled() is False

        panel._preset_combo.setCurrentText("my")
        assert panel._btn_delete_template.isEnabled() is True

    def test_delete_template_removes_and_resets_to_default(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate
        from PyQt5.QtWidgets import QMessageBox

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="my", task="detect", params={}, created_at="t"))
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)
        panel._preset_combo.setCurrentText("my")

        monkeypatch.setattr(
            "src.ui.train_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Yes,
        )

        panel._on_delete_template()

        assert reg.get("my", "detect") is None
        assert panel._preset_combo.currentText() == "默认"
        items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
        assert "my" not in items

    def test_delete_template_cancelled_keeps_template(self, qapp, tmp_path, monkeypatch):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate
        from PyQt5.QtWidgets import QMessageBox

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="keep", task="detect", params={}, created_at="t"))
        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)
        panel._preset_combo.setCurrentText("keep")

        monkeypatch.setattr(
            "src.ui.train_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.No,
        )

        panel._on_delete_template()

        assert reg.get("keep", "detect") is not None

    def test_task_switch_repopulates_combo_and_resets_to_default(self, qapp, tmp_path):
        from src.ui.train_panel import TrainPanel
        from src.core.train_templates import TemplateRegistry, TrainTemplate

        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(
            name="detect-only",
            task="detect",
            params={"epochs": 250},
            created_at="t",
        ))
        reg.upsert(TrainTemplate(
            name="classify-only",
            task="classify",
            params={"epochs": 75},
            created_at="t",
        ))

        panel = TrainPanel()
        panel._task_combo.setCurrentText("detect")
        panel.set_template_registry(reg)
        panel._preset_combo.setCurrentText("detect-only")
        assert panel._epochs_spin.value() == 250

        panel._task_combo.setCurrentText("classify")

        items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
        assert "detect-only" not in items
        assert "classify-only" in items
        # After repopulate the previous (detect-only) selection is gone, so combo shows 默认
        assert panel._preset_combo.currentText() == "默认"
        # Defaults applied (epochs reset to 100)
        assert panel._epochs_spin.value() == 100


def test_train_panel_filter_changed_signal_reemits_inner_bar(qapp):
    """TrainPanel.filter_changed re-emits TagFilterBar.filter_changed."""
    from src.core.tags import TagFilter
    from src.ui.train_panel import TrainPanel

    panel = TrainPanel()
    received: list = []
    panel.filter_changed.connect(received.append)

    panel._tag_filter_bar.set_available_tags(["a", "b"])
    panel._tag_filter_bar._advance_state("a")

    assert received
    assert isinstance(received[-1], TagFilter)
    assert received[-1].includes == ("a",)


def test_train_panel_set_filter_breakdown_renders_label(qapp):
    from src.ui.train_panel import TrainPanel

    panel = TrainPanel()

    # No counts → label hidden
    panel.set_filter_breakdown(None)
    assert panel._filter_breakdown_label.isHidden()

    # Match + drops, no conflicts
    panel.set_filter_breakdown(
        {"match": 5, "excluded": 2, "no_include": 1, "conflict": 0}
    )
    assert panel._filter_breakdown_label.text() == "命中 5 张，排除 3 张"
    assert not panel._filter_breakdown_label.isHidden()

    # Conflicts present → suffix appended
    panel.set_filter_breakdown(
        {"match": 5, "excluded": 1, "no_include": 0, "conflict": 2}
    )
    assert panel._filter_breakdown_label.text() == (
        "命中 5 张，排除 3 张（其中 2 张同时命中 include 和 exclude）"
    )

    # Back to None → hidden again
    panel.set_filter_breakdown(None)
    assert panel._filter_breakdown_label.isHidden()


def test_train_panel_set_filter_breakdown_all_zero_is_hidden(qapp):
    from src.ui.train_panel import TrainPanel

    panel = TrainPanel()
    panel.set_filter_breakdown(
        {"match": 0, "excluded": 0, "no_include": 0, "conflict": 0}
    )
    assert panel._filter_breakdown_label.isHidden()
