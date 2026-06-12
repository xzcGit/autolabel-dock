"""Collapsible group box: chevron header + hide-able body. Qt-native."""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import PALETTE

# Standard Qt "no maximum" sentinel (QWIDGETSIZE_MAX is not exposed in PyQt5).
_NO_MAX_HEIGHT = (1 << 24) - 1


class CollapsibleGroupBox(QWidget):
    """A QGroupBox-like container with a clickable header that hides body.

    When collapsed, the widget's ``maximumHeight`` is pinned to the header's
    height so a parent ``QSplitter`` releases the freed vertical space to
    sibling panes. Expanding restores the unbounded height.

    Signals:
        toggled(bool): emitted with the new expanded state when the user (or
            ``setExpanded``) changes it. Not emitted for no-op calls.
    """

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow)
        self._toggle.setAutoRaise(True)
        self._toggle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._toggle.setStyleSheet(
            "QToolButton { border: none; padding: 2px 4px; font-weight: bold; "
            f"color: {PALETTE['text']}; text-align: left; }}"
        )
        self._toggle.clicked.connect(self._on_toggle_clicked)
        outer.addWidget(self._toggle)

        self._body = QFrame(self)
        self._body.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._body)

    def title(self) -> str:
        return self._title

    def set_content_layout(self, layout: QLayout) -> None:
        """Install the body's layout. May only be called once."""
        self._body.setLayout(layout)

    def isExpanded(self) -> bool:
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._toggle.setChecked(expanded)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._body.setVisible(expanded)
        if expanded:
            self.setMaximumHeight(_NO_MAX_HEIGHT)
        else:
            # Clamp to just the header so a parent QSplitter reclaims the
            # freed space and redistributes it to sibling panes.
            self.setMaximumHeight(self._toggle.sizeHint().height())
        self.toggled.emit(expanded)

    def _on_toggle_clicked(self) -> None:
        self.setExpanded(self._toggle.isChecked())
