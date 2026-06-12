"""Tests for dialogs."""
import os
import pytest
from pathlib import Path
from PyQt5.QtCore import Qt


class TestNewProjectDialog:
    def test_creates(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        assert dlg is not None

    def test_has_name_field(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        assert dlg._name_edit is not None

    def test_has_classes_field(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        assert dlg._classes_edit is not None

    def test_get_values(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test_project")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._classes_edit.setText("cat, dog, bird")

        name, proj_dir, image_dir, classes, task_type = dlg.get_values()
        assert name == "test_project"
        assert proj_dir == str(tmp_path)
        assert image_dir == ""
        assert classes == ["cat", "dog", "bird"]
        assert task_type == "detect"  # default

    def test_rejects_empty_name(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert "项目名称" in dlg._error_label.text()

    def test_rejects_invalid_chars_in_name(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test<project")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert "非法字符" in dlg._error_label.text()

    def test_rejects_long_name(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("x" * 65)
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert "64" in dlg._error_label.text()

    def test_rejects_nonexistent_dir(self, qapp):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test")
        dlg._dir_edit.setText("/nonexistent/path/12345")
        dlg._validate_and_accept()
        assert "不存在" in dlg._error_label.text()

    def test_rejects_nonexistent_image_dir(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("test")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._image_dir_edit.setText("/nonexistent/img/dir")
        dlg._validate_and_accept()
        assert "图片目录" in dlg._error_label.text()

    def test_accepts_valid_input(self, qapp, tmp_path):
        from src.ui.dialogs import NewProjectDialog

        dlg = NewProjectDialog()
        dlg._name_edit.setText("my_project")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert dlg._error_label.text() == ""


class TestExportDialog:
    def test_creates(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        assert dlg is not None

    def test_has_format_selector(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        items = [dlg._format_combo.itemText(i) for i in range(dlg._format_combo.count())]
        assert "YOLO" in items
        assert "COCO" in items
        assert "labelme" in items

    def test_get_values(self, qapp, tmp_path):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._format_combo.setCurrentText("COCO")
        dlg._dir_edit.setText(str(tmp_path))
        dlg._confirmed_only.setChecked(True)

        fmt, out_dir, only_confirmed = dlg.get_values()
        assert fmt == "COCO"
        assert out_dir == str(tmp_path)
        assert only_confirmed is True

    def test_rejects_nonexistent_output_dir(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._dir_edit.setText("/nonexistent/output/12345")
        dlg._validate_and_accept()
        assert "不存在" in dlg._error_label.text()

    def test_rejects_empty_output_dir(self, qapp):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._dir_edit.setText("")
        dlg._validate_and_accept()
        assert "输出目录" in dlg._error_label.text()

    def test_accepts_valid_dir(self, qapp, tmp_path):
        from src.ui.dialogs import ExportDialog

        dlg = ExportDialog()
        dlg._dir_edit.setText(str(tmp_path))
        dlg._validate_and_accept()
        assert dlg._error_label.text() == ""


class TestClassManagerDialog:
    def test_creates(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat", "dog"], colors={"cat": "#a6e3a1"})
        assert dlg._class_list.count() == 2

    def test_add_class(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat"], colors={})
        dlg._new_class_edit.setText("dog")
        dlg._on_add()
        assert dlg._class_list.count() == 2

    def test_get_classes(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat", "dog"], colors={})
        result = dlg.get_classes()
        assert result == ["cat", "dog"]

    def test_rejects_long_class_name(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=[], colors={})
        dlg._new_class_edit.setText("x" * 33)
        dlg._on_add()
        assert "32" in dlg._status_label.text()
        assert dlg._class_list.count() == 0

    def test_rejects_invalid_chars_in_class_name(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=[], colors={})
        dlg._new_class_edit.setText("cat*dog")
        dlg._on_add()
        assert "非法字符" in dlg._status_label.text()
        assert dlg._class_list.count() == 0

    def test_rejects_duplicate_class(self, qapp):
        from src.ui.dialogs import ClassManagerDialog

        dlg = ClassManagerDialog(classes=["cat"], colors={})
        dlg._new_class_edit.setText("cat")
        dlg._on_add()
        assert "已存在" in dlg._status_label.text()
        assert dlg._class_list.count() == 1


class TestClassRegisterDialog:
    """Dialog shown before batch auto-label to confirm new model classes."""

    def _items(self, *specs):
        from src.controllers.project import ClassPreviewItem
        return [
            ClassPreviewItem(model_name=name, is_blacklisted=bl, default_checked=def_)
            for name, bl, def_ in specs
        ]

    def test_default_checks_match_default_checked(self, qapp):
        from src.ui.dialogs import ClassRegisterDialog
        dlg = ClassRegisterDialog(self._items(
            ("cat", False, True),
            ("dog", False, True),
            ("n01440764", True, False),
        ))
        assert dlg.get_selected() == ["cat", "dog"]
        dlg.deleteLater()

    def test_select_only_valid_button(self, qapp):
        from src.ui.dialogs import ClassRegisterDialog
        dlg = ClassRegisterDialog(self._items(
            ("cat", False, False),
            ("n01440764", True, True),
        ))
        dlg._on_only_valid_clicked()
        assert dlg.get_selected() == ["cat"]
        dlg.deleteLater()

    def test_select_all_button(self, qapp):
        from src.ui.dialogs import ClassRegisterDialog
        dlg = ClassRegisterDialog(self._items(
            ("cat", False, False),
            ("n01440764", True, False),
        ))
        dlg._on_select_all_clicked()
        assert set(dlg.get_selected()) == {"cat", "n01440764"}
        dlg.deleteLater()

    def test_invert_button(self, qapp):
        from src.ui.dialogs import ClassRegisterDialog
        dlg = ClassRegisterDialog(self._items(
            ("cat", False, True),
            ("dog", False, False),
        ))
        dlg._on_invert_clicked()
        assert dlg.get_selected() == ["dog"]
        dlg.deleteLater()

    def test_summary_label_updates_on_check(self, qapp):
        from src.ui.dialogs import ClassRegisterDialog
        items = self._items(("cat", False, True), ("dog", False, True))
        dlg = ClassRegisterDialog(items)
        assert "2 项" in dlg._summary_label.text()
        assert "已勾选 2 项" in dlg._summary_label.text()
        # Uncheck one
        dlg._checkboxes[0].setChecked(False)
        assert "已勾选 1 项" in dlg._summary_label.text()
        dlg.deleteLater()

    def test_blacklist_marker_shown(self, qapp):
        from src.ui.dialogs import ClassRegisterDialog
        items = self._items(("n01440764", True, False))
        dlg = ClassRegisterDialog(items)
        text = dlg._checkboxes[0].text()
        assert "n01440764" in text
        # Tooltip / suffix mentions ImageNet
        combined = text + " " + (dlg._checkboxes[0].toolTip() or "")
        assert "ImageNet" in combined
        dlg.deleteLater()
