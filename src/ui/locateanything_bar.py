"""LocateAnything toolbar bar — natural-language open-vocabulary labeling.

Two visual states, driven entirely through the public API (no external reads
of private widgets):

    not-enabled : a single button (「文本标注」). Clicking emits
                  ``enable_requested`` — the controller then runs
                  probe → preflight → background load.
    loading     : the button is disabled and shows a status string.
    enabled     : a natural-language QLineEdit + an optional 「目标类别」
                  QComboBox (project classes plus a「(按名称自动匹配)」item)
                  + a 「关闭」 button. Text/combo changes emit
                  ``query_changed(prompt, target_class)``; 关闭 emits
                  ``disable_requested``.

The actual labeling actions still go through the existing 自动标注 / 批量标注
buttons — once LA is enabled it becomes the active predictor, so those flows
work unchanged.
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from src.ui.icons import icon
from src.ui.theme import set_button_role, text_style

# Sentinel combo entry meaning "no forced target — match each box by name".
_AUTO_MATCH_LABEL = "(按名称自动匹配)"


class LocateAnythingBar(QWidget):
    """Toolbar widget for the LocateAnything text-labeling backend."""

    enable_requested = pyqtSignal()
    disable_requested = pyqtSignal()
    query_changed = pyqtSignal(str, object)  # (prompt, target_class | None)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._classes: list[str] = []
        self._enabled_state = False

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        # Collapsed (not enabled) — single button.
        self._enable_btn = QPushButton(icon("auto_label"), "文本标注")
        self._enable_btn.setToolTip(
            "启用 LocateAnything 文本标注后端（开放词汇，用自然语言描述要检测的目标）"
        )
        set_button_role(self._enable_btn, "secondary")
        self._enable_btn.clicked.connect(self.enable_requested.emit)
        self._layout.addWidget(self._enable_btn)

        # Loading / status label.
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(text_style("hint"))
        self._status_label.hide()
        self._layout.addWidget(self._status_label)

        # Expanded (enabled) controls — created hidden.
        self._prompt_edit = QLineEdit()
        self._prompt_edit.setPlaceholderText("描述要检测的目标，如: cat, dog（留空则用项目类别）")
        self._prompt_edit.setMinimumWidth(220)
        self._prompt_edit.setClearButtonEnabled(True)
        self._prompt_edit.textChanged.connect(self._on_query_changed)
        self._prompt_edit.hide()
        self._layout.addWidget(self._prompt_edit)

        self._target_label = QLabel(" 目标类别: ")
        self._target_label.setStyleSheet(text_style("muted"))
        self._target_label.hide()
        self._layout.addWidget(self._target_label)

        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(120)
        self._target_combo.currentIndexChanged.connect(self._on_query_changed)
        self._target_combo.hide()
        self._layout.addWidget(self._target_combo)

        self._disable_btn = QPushButton(icon("cancel"), "关闭")
        self._disable_btn.setToolTip("关闭并卸载 LocateAnything")
        set_button_role(self._disable_btn, "danger")
        self._disable_btn.clicked.connect(self.disable_requested.emit)
        self._disable_btn.hide()
        self._layout.addWidget(self._disable_btn)

        self._rebuild_target_combo()

    # ── Public API ────────────────────────────────────────────────────────

    def set_classes(self, classes) -> None:
        """Update the project class list used by the 目标类别 dropdown."""
        self._classes = list(classes)
        self._rebuild_target_combo()

    def set_enabled_state(self, enabled: bool) -> None:
        """Toggle between the collapsed (single button) and expanded layouts."""
        self._enabled_state = bool(enabled)
        self._enable_btn.setVisible(not self._enabled_state)
        self._enable_btn.setEnabled(True)
        self._status_label.setVisible(False)
        self._status_label.setText("")
        self._prompt_edit.setVisible(self._enabled_state)
        self._target_label.setVisible(self._enabled_state)
        self._target_combo.setVisible(self._enabled_state)
        self._disable_btn.setVisible(self._enabled_state)

    def set_feature_visible(self, visible: bool) -> None:
        """Fully show/hide the whole LA feature (the experimental master switch).

        When ``visible`` is False the entire bar is hidden, so the 文本标注 entry
        point disappears — backing the ``AppConfig.enable_locateanything``
        "完全隐藏" contract.
        """
        self.setVisible(bool(visible))

    def set_status(self, message: str) -> None:
        """Show a transient status string (used during background load).

        While loading we keep the single button visible but disabled and show
        the message next to it.
        """
        self._status_label.setText(message or "")
        self._status_label.setVisible(bool(message))
        if not self._enabled_state:
            self._enable_btn.setEnabled(not message)

    def get_query(self) -> tuple[str, object]:
        """Return (prompt, target_class | None) from the current controls."""
        prompt = self._prompt_edit.text().strip()
        target = self._target_combo.currentData()
        return prompt, target

    # ── Internals ───────────────────────────────────────────────────────--

    def _rebuild_target_combo(self) -> None:
        prev = self._target_combo.currentData()
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        # First item = auto-match (no forced target). Stored data = None.
        self._target_combo.addItem(_AUTO_MATCH_LABEL, None)
        for cls in self._classes:
            self._target_combo.addItem(cls, cls)
        # Restore previous selection if still present.
        if prev is not None:
            idx = self._target_combo.findData(prev)
            if idx >= 0:
                self._target_combo.setCurrentIndex(idx)
        self._target_combo.blockSignals(False)

    def _on_query_changed(self, *args) -> None:
        prompt, target = self.get_query()
        self.query_changed.emit(prompt, target)
