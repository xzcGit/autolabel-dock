"""Class picker popup for annotation class assignment."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QLineEdit,
    QPushButton,
)

from src.ui.theme import PALETTE, text_style
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class ClassPickerPopup(QDialog):
    """Popup dialog for selecting an annotation class.

    Like labelimg: shows existing classes + input box for new class.
    Single-click or Enter to confirm, Escape to cancel.
    """

    def __init__(
        self,
        classes: list[str],
        colors: dict[str, str],
        default_class: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("选择类别")
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setMinimumWidth(180)
        self.setFocusPolicy(Qt.StrongFocus)

        self._classes = list(classes)
        self._new_class: str | None = None  # track if user typed a new class

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Input box for new class (like labelimg)
        self._input = QLineEdit()
        self._input.setPlaceholderText("输入新类别或搜索...")
        self._input.textChanged.connect(self._on_input_changed)
        self._input.returnPressed.connect(self._on_input_confirmed)
        layout.addWidget(self._input)

        # Class list
        self._list = QListWidget()
        self._list.setMaximumHeight(220)

        self._colors = dict(colors)
        default_row = 0
        for i, cls_name in enumerate(classes):
            item = QListWidgetItem(cls_name)
            color = colors.get(cls_name, PALETTE["primary"])
            item.setForeground(QColor(color))
            self._list.addItem(item)
            if cls_name == default_class:
                default_row = i

        if classes:
            self._list.setCurrentRow(default_row)

        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Hint
        hint = QLabel("单击 / 1-9键选择 / 输入新类别后回车")
        hint.setStyleSheet(text_style("hint"))
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        # Focus the input for immediate typing
        self._input.setFocus()

    def _on_input_changed(self, text: str) -> None:
        """Filter list as user types."""
        text_lower = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text_lower != "" and text_lower not in item.text().lower())
        # Select first visible match
        for i in range(self._list.count()):
            if not self._list.item(i).isHidden():
                self._list.setCurrentRow(i)
                break

    def _on_input_confirmed(self) -> None:
        """Handle Enter in the input box."""
        text = self._input.text().strip()
        if not text:
            # Empty input — use selected item from list
            if self._list.currentItem() and not self._list.currentItem().isHidden():
                self.accept()
            return

        # Check if text matches an existing class
        for i in range(self._list.count()):
            if self._list.item(i).text() == text:
                self._list.setCurrentRow(i)
                self._new_class = None
                self.accept()
                return

        # New class — accept with the typed name
        self._new_class = text
        self.accept()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Accept on single click."""
        self._new_class = None
        self._input.clear()  # clear filter
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.reject()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._on_input_confirmed()
        elif Qt.Key_1 <= event.key() <= Qt.Key_9 and not self._input.hasFocus():
            idx = event.key() - Qt.Key_1
            if 0 <= idx < self._list.count() and not self._list.item(idx).isHidden():
                self._list.setCurrentRow(idx)
                self.accept()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        """Close picker when it loses focus."""
        super().focusOutEvent(event)
        # Small delay to allow click on list item to register
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._check_focus)

    def _check_focus(self) -> None:
        if not self.isActiveWindow():
            self.reject()

    def get_selected_class(self) -> str | None:
        """Return the selected class name (existing or newly typed)."""
        if self._new_class:
            return self._new_class
        item = self._list.currentItem()
        if item and not item.isHidden():
            return item.text()
        return None

    def get_selected_index(self) -> int:
        """Return the selected class index, or -1 for new classes."""
        if self._new_class:
            return -1
        return self._list.currentRow()

    def is_new_class(self) -> bool:
        """Return True if the user typed a new class name."""
        return self._new_class is not None


class KeypointLabelPicker(QDialog):
    """Popup dialog for selecting a keypoint label.

    Shows existing keypoint labels from current annotations and allows free-text.
    """

    def __init__(
        self,
        existing_labels: list[str],
        default_label: str = "point",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("关键点标签")
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setMinimumWidth(160)
        self.setFocusPolicy(Qt.StrongFocus)

        self._labels = list(existing_labels)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._input = QLineEdit()
        self._input.setPlaceholderText("输入标签或搜索...")
        self._input.setText(default_label)
        self._input.selectAll()
        self._input.textChanged.connect(self._on_input_changed)
        self._input.returnPressed.connect(self._on_confirm)
        layout.addWidget(self._input)

        self._list = QListWidget()
        self._list.setMaximumHeight(180)
        for label in existing_labels:
            self._list.addItem(QListWidgetItem(label))
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        hint = QLabel("单击选择 / 输入后回车")
        hint.setStyleSheet(text_style("hint"))
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        self._input.setFocus()

    def _on_input_changed(self, text: str) -> None:
        text_lower = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text_lower != "" and text_lower not in item.text().lower())
        for i in range(self._list.count()):
            if not self._list.item(i).isHidden():
                self._list.setCurrentRow(i)
                break

    def _on_confirm(self) -> None:
        text = self._input.text().strip()
        if text:
            self.accept()
        elif self._list.currentItem() and not self._list.currentItem().isHidden():
            self.accept()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._input.setText(item.text())
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.reject()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._on_confirm()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._check_focus)

    def _check_focus(self) -> None:
        if not self.isActiveWindow():
            self.reject()

    def get_label(self) -> str | None:
        """Return the selected/typed label, or None if cancelled."""
        text = self._input.text().strip()
        return text if text else None
