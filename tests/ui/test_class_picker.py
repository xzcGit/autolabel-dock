"""Tests for ClassPickerPopup."""
import pytest
from PyQt5.QtCore import Qt


class TestClassPickerPopup:
    def test_creates_with_classes(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog", "bird"],
            colors={"cat": "#a6e3a1", "dog": "#89b4fa", "bird": "#f38ba8"},
        )
        assert picker._list.count() == 3

    def test_default_selection(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog"],
            colors={},
            default_class="dog",
        )
        assert picker._list.currentRow() == 1

    def test_default_first_if_no_default(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog"],
            colors={},
        )
        assert picker._list.currentRow() == 0

    def test_get_selected_class(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(
            classes=["cat", "dog", "bird"],
            colors={},
        )
        picker._list.setCurrentRow(2)
        assert picker.get_selected_class() == "bird"
        assert picker.get_selected_index() == 2

    def test_empty_classes(self, qapp):
        from src.ui.class_picker import ClassPickerPopup

        picker = ClassPickerPopup(classes=[], colors={})
        assert picker._list.count() == 0
        assert picker.get_selected_class() is None
