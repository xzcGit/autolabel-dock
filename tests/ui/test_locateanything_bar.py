"""UI tests for the LocateAnything toolbar bar."""
from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox

from src.ui.locateanything_bar import LocateAnythingBar, _AUTO_MATCH_LABEL


def test_collapsed_state_shows_only_enable_button(qapp):
    bar = LocateAnythingBar()
    bar.set_enabled_state(False)
    assert bar._enable_btn.isVisible() or not bar.isVisible()
    # Expanded controls hidden in collapsed state.
    assert bar._prompt_edit.isHidden()
    assert bar._target_combo.isHidden()
    assert bar._disable_btn.isHidden()


def test_enabled_state_shows_query_controls(qapp):
    bar = LocateAnythingBar()
    bar.show()
    bar.set_enabled_state(True)
    assert not bar._prompt_edit.isHidden()
    assert not bar._target_combo.isHidden()
    assert not bar._disable_btn.isHidden()
    assert bar._enable_btn.isHidden()


def test_enable_request_signal(qapp, monkeypatch):
    # Clicking 文本标注 first shows a confirm dialog; accept it.
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    bar = LocateAnythingBar()
    fired = []
    bar.enable_requested.connect(lambda: fired.append(True))
    bar._enable_btn.click()
    assert fired == [True]


def test_enable_request_cancelled_does_not_emit(qapp, monkeypatch):
    # Declining the confirm dialog must not emit enable_requested.
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    bar = LocateAnythingBar()
    fired = []
    bar.enable_requested.connect(lambda: fired.append(True))
    bar._enable_btn.click()
    assert fired == []


def test_disable_request_signal(qapp):
    bar = LocateAnythingBar()
    bar.set_enabled_state(True)
    fired = []
    bar.disable_requested.connect(lambda: fired.append(True))
    bar._disable_btn.click()
    assert fired == [True]


def test_target_combo_first_item_is_auto_match(qapp):
    bar = LocateAnythingBar()
    bar.set_classes(["cat", "dog"])
    assert bar._target_combo.itemText(0) == _AUTO_MATCH_LABEL
    assert bar._target_combo.itemData(0) is None
    assert bar._target_combo.itemData(1) == "cat"
    assert bar._target_combo.itemData(2) == "dog"


def test_get_query_returns_prompt_and_target(qapp):
    bar = LocateAnythingBar()
    bar.set_classes(["cat", "dog"])
    bar.set_enabled_state(True)
    bar._prompt_edit.setText("a fluffy cat")
    # Default selection = auto-match -> target None.
    prompt, target = bar.get_query()
    assert prompt == "a fluffy cat"
    assert target is None
    # Select an explicit target class.
    bar._target_combo.setCurrentIndex(1)
    prompt, target = bar.get_query()
    assert target == "cat"


def test_query_changed_emits_on_text_change(qapp):
    bar = LocateAnythingBar()
    bar.set_classes(["cat"])
    bar.set_enabled_state(True)
    received = []
    bar.query_changed.connect(lambda p, t: received.append((p, t)))
    bar._prompt_edit.setText("dog")
    assert received and received[-1][0] == "dog"


def test_set_status_disables_enable_button_while_loading(qapp):
    bar = LocateAnythingBar()
    bar.set_enabled_state(False)
    bar.set_status("正在加载…")
    assert bar._enable_btn.isEnabled() is False
    bar.set_status("")
    assert bar._enable_btn.isEnabled() is True
