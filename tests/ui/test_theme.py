"""Tests for shared UI theme contracts."""
from PyQt5.QtWidgets import QPushButton


def test_theme_exposes_industrial_palette_and_font_stack():
    from src.ui import theme

    assert theme.PALETTE["primary"] == "#38bdf8"
    assert theme.PALETTE["bg"] == "#101418"
    assert "Microsoft YaHei UI" in theme.FONT_STACK
    assert "Noto Sans CJK SC" in theme.FONT_STACK
    assert "font-family" in theme.STYLESHEET
    assert 'QPushButton[role="primary"]' in theme.STYLESHEET
    assert "QToolBar" in theme.STYLESHEET


def test_set_button_role_marks_buttons_for_global_qss(qapp):
    from src.ui.theme import set_button_role

    btn = QPushButton("Run")
    set_button_role(btn, "primary")

    assert btn.property("role") == "primary"
    assert btn.cursor().shape() is not None
