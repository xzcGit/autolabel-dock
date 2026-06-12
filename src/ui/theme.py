"""Industrial dark theme and shared style helpers for PyQt5."""
from __future__ import annotations

from PyQt5.QtCore import Qt


FONT_STACK = (
    '"Segoe UI", "Microsoft YaHei UI", "PingFang SC", '
    '"Noto Sans CJK SC", "Source Han Sans SC", "Helvetica Neue", Arial, sans-serif'
)

PALETTE = {
    "bg": "#101418",
    "bg_deep": "#0b0f12",
    "canvas": "#0c1116",
    "panel": "#151b20",
    "panel_alt": "#1b2229",
    "panel_raised": "#202932",
    "line": "#2a343d",
    "line_strong": "#3b4752",
    "text": "#d8dee9",
    "text_muted": "#9aa7b2",
    "text_subtle": "#6f7b86",
    "ink": "#081016",
    "primary": "#38bdf8",
    "primary_hover": "#67d5ff",
    "primary_pressed": "#0ea5e9",
    "primary_soft": "#102836",
    "success": "#7dd3a8",
    "success_soft": "#173527",
    "danger": "#f87171",
    "danger_soft": "#3a1d20",
    "warning": "#fbbf77",
    "warning_soft": "#3a2a18",
    "violet": "#b6a7ff",
    "teal": "#5eead4",
}

# Backwards-compatible alias for older modules/tests that still describe the
# palette as Catppuccin. New code should use PALETTE directly.
MOCHA = {
    "base": PALETTE["bg"],
    "mantle": PALETTE["panel"],
    "crust": PALETTE["bg_deep"],
    "surface0": PALETTE["panel_alt"],
    "surface1": PALETTE["line"],
    "surface2": PALETTE["line_strong"],
    "overlay0": PALETTE["text_subtle"],
    "text": PALETTE["text"],
    "subtext0": PALETTE["text_muted"],
    "green": PALETTE["success"],
    "blue": PALETTE["primary"],
    "red": PALETTE["danger"],
    "peach": PALETTE["warning"],
    "yellow": PALETTE["warning"],
    "mauve": PALETTE["violet"],
    "teal": PALETTE["teal"],
    "sky": PALETTE["primary_hover"],
    "lavender": PALETTE["violet"],
}


def _refresh_style(widget) -> None:
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_widget_role(widget, role: str):
    """Attach a QSS role dynamic property and refresh the widget style."""
    widget.setProperty("role", role)
    if hasattr(widget, "setCursor") and role not in {"passive", "list-item"}:
        widget.setCursor(Qt.PointingHandCursor)
    _refresh_style(widget)
    return widget


def set_button_role(button, role: str):
    """Mark a button as a named action role for global QSS styling."""
    return set_widget_role(button, role)


def set_surface(widget, surface: str):
    """Mark a widget as a named surface for global QSS styling."""
    widget.setProperty("surface", surface)
    _refresh_style(widget)
    return widget


def text_style(role: str = "muted") -> str:
    styles = {
        "display": (
            f"color: {PALETTE['text']}; font-size: 30px; font-weight: 700;"
        ),
        "title": (
            f"color: {PALETTE['text']}; font-size: 16px; font-weight: 650;"
        ),
        "section": (
            f"color: {PALETTE['text_muted']}; font-size: 12px; font-weight: 650;"
        ),
        "body": f"color: {PALETTE['text']}; font-size: 13px;",
        "small": f"color: {PALETTE['text']}; font-size: 11px;",
        "muted": f"color: {PALETTE['text_muted']}; font-size: 12px;",
        "hint": f"color: {PALETTE['text_subtle']}; font-size: 11px;",
        "success": f"color: {PALETTE['success']}; font-weight: 650;",
        "warning": f"color: {PALETTE['warning']}; font-weight: 650;",
        "error": f"color: {PALETTE['danger']}; font-size: 11px;",
    }
    return styles.get(role, styles["muted"])


def chip_style(active: bool = False) -> str:
    bg = PALETTE["primary_soft"] if active else PALETTE["panel_raised"]
    fg = PALETTE["primary"] if active else PALETTE["text"]
    border = PALETTE["primary"] if active else PALETTE["line_strong"]
    weight = "700" if active else "500"
    return (
        "QToolButton {"
        f" background-color: {bg};"
        f" color: {fg};"
        f" border: 1px solid {border};"
        " border-radius: 7px;"
        " padding: 3px 9px;"
        " font-size: 11px;"
        f" font-weight: {weight};"
        "}"
        f"QToolButton:hover {{ background-color: {PALETTE['panel_alt']};"
        f" border-color: {PALETTE['primary']}; }}"
    )


STYLESHEET = f"""
QWidget {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: {FONT_STACK};
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {PALETTE['bg']};
}}

QWidget[surface="panel"] {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
}}

QTabWidget::pane {{
    border: 1px solid {PALETTE['line']};
    background-color: {PALETTE['bg']};
}}

QTabBar::tab {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text_muted']};
    padding: 8px 18px;
    border: 1px solid {PALETTE['line']};
    border-bottom: none;
    margin-right: 2px;
    min-height: 22px;
}}

QTabBar::tab:selected {{
    background-color: {PALETTE['panel']};
    color: {PALETTE['text']};
    border-bottom: 2px solid {PALETTE['primary']};
}}

QTabBar::tab:hover {{
    color: {PALETTE['text']};
    background-color: {PALETTE['panel_alt']};
}}

QToolBar {{
    background-color: {PALETTE['panel']};
    border: none;
    border-bottom: 1px solid {PALETTE['line']};
    spacing: 5px;
    padding: 5px 6px;
}}

QToolBar::separator {{
    background-color: {PALETTE['line']};
    width: 1px;
    margin: 4px 6px;
}}

QPushButton {{
    background-color: {PALETTE['panel_raised']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line_strong']};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {PALETTE['panel_alt']};
    border-color: {PALETTE['primary']};
}}

QPushButton:focus {{
    border-color: {PALETTE['primary']};
}}

QPushButton:pressed {{
    background-color: {PALETTE['bg_deep']};
    padding-top: 7px;
    padding-bottom: 5px;
}}

QPushButton:checked {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
}}

QPushButton:disabled {{
    color: {PALETTE['text_subtle']};
    background-color: {PALETTE['panel']};
    border-color: {PALETTE['line']};
}}

QPushButton[role="primary"] {{
    background-color: {PALETTE['primary']};
    color: {PALETTE['ink']};
    border-color: {PALETTE['primary']};
}}

QPushButton[role="primary"]:hover {{
    background-color: {PALETTE['primary_hover']};
    border-color: {PALETTE['primary_hover']};
}}

QPushButton[role="primary"]:pressed {{
    background-color: {PALETTE['primary_pressed']};
    border-color: {PALETTE['primary_pressed']};
}}

QPushButton[role="danger"] {{
    background-color: {PALETTE['danger_soft']};
    color: {PALETTE['danger']};
    border-color: {PALETTE['danger']};
}}

QPushButton[role="danger"]:hover {{
    background-color: {PALETTE['danger']};
    color: {PALETTE['ink']};
}}

QPushButton[role="success"] {{
    background-color: {PALETTE['success_soft']};
    color: {PALETTE['success']};
    border-color: {PALETTE['success']};
}}

QPushButton[role="secondary"] {{
    background-color: {PALETTE['panel_raised']};
    color: {PALETTE['text']};
}}

QPushButton[role="icon"] {{
    min-width: 26px;
    max-width: 32px;
    padding: 4px;
}}

QPushButton[role="icon-danger"] {{
    min-width: 24px;
    max-width: 28px;
    padding: 3px;
    color: {PALETTE['danger']};
}}

QPushButton[role="list-item"] {{
    text-align: left;
    padding: 6px 8px;
    border-color: transparent;
    background-color: transparent;
    font-weight: 500;
}}

QPushButton[role="list-item"]:checked {{
    background-color: {PALETTE['primary_soft']};
    border-color: {PALETTE['primary']};
    color: {PALETTE['primary']};
}}

QPushButton[role="primary"]:disabled,
QPushButton[role="danger"]:disabled,
QPushButton[role="success"]:disabled,
QPushButton[role="secondary"]:disabled {{
    color: {PALETTE['text_subtle']};
    background-color: {PALETTE['panel']};
    border-color: {PALETTE['line']};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line_strong']};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {PALETTE['primary_soft']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {PALETTE['primary']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QTextEdit, QPlainTextEdit {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    padding: 6px;
    selection-background-color: {PALETTE['primary_soft']};
}}

QLabel {{
    background-color: transparent;
}}

QGroupBox {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 8px 8px 8px;
    font-weight: 650;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {PALETTE['text_muted']};
    background-color: {PALETTE['panel']};
}}

QListWidget, QTreeWidget, QTableWidget {{
    background-color: {PALETTE['bg_deep']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    outline: none;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 4px 8px;
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {PALETTE['primary_soft']};
    color: {PALETTE['text']};
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {PALETTE['panel_alt']};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {PALETTE['bg_deep']};
    width: 10px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {PALETTE['line_strong']};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {PALETTE['text_subtle']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {PALETTE['bg_deep']};
    height: 10px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {PALETTE['line_strong']};
    border-radius: 5px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {PALETTE['text_subtle']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QStatusBar {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text_muted']};
    border-top: 1px solid {PALETTE['line']};
}}

QMenuBar {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text']};
}}

QMenuBar::item:selected {{
    background-color: {PALETTE['panel_alt']};
}}

QMenu {{
    background-color: {PALETTE['panel_raised']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
}}

QMenu::item {{
    padding: 5px 22px;
}}

QMenu::item:selected {{
    background-color: {PALETTE['primary_soft']};
}}

QProgressBar {{
    background-color: {PALETTE['bg_deep']};
    border: 1px solid {PALETTE['line']};
    border-radius: 6px;
    text-align: center;
    color: {PALETTE['text']};
}}

QProgressBar::chunk {{
    background-color: {PALETTE['primary']};
    border-radius: 5px;
}}

QToolTip {{
    background-color: {PALETTE['panel_raised']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line_strong']};
    padding: 5px;
}}

QSplitter::handle {{
    background-color: {PALETTE['line']};
}}

QSplitter::handle:hover {{
    background-color: {PALETTE['primary']};
}}

QHeaderView::section {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['line']};
    padding: 5px;
}}

QCheckBox, QRadioButton {{
    color: {PALETTE['text']};
    spacing: 6px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid {PALETTE['line_strong']};
    background-color: {PALETTE['bg_deep']};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {PALETTE['primary']};
    border-color: {PALETTE['primary']};
}}

QSlider::groove:horizontal {{
    background-color: {PALETTE['line']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {PALETTE['primary']};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
"""


def apply_theme(app) -> None:
    """Apply the shared industrial dark theme to a QApplication."""
    app.setStyleSheet(STYLESHEET)
