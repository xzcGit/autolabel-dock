"""UI tests for the tag widget module."""
from __future__ import annotations

from src.core.tags import TagFilter
from src.ui.tag_widget import TagChipBar, TagFilterBar, TagManagerDialog


def test_tag_chip_bar_emits_on_add_and_remove(qapp):
    bar = TagChipBar()
    bar.set_available_tags(["a", "b", "c"])
    received: list[list[str]] = []
    bar.tags_changed.connect(lambda t: received.append(list(t)))

    bar._add("a")
    bar._add("b")
    assert received[-1] == ["a", "b"]
    bar._remove("a")
    assert received[-1] == ["b"]
    assert bar.get_tags() == ["b"]


def test_tag_chip_bar_set_tags_does_not_emit(qapp):
    """Programmatic load (e.g. switching image) must not look like an edit."""
    bar = TagChipBar()
    received: list = []
    bar.tags_changed.connect(received.append)
    bar.blockSignals(True)
    try:
        bar.set_tags(["x", "y"])
    finally:
        bar.blockSignals(False)
    assert received == []


def test_tag_filter_bar_empty_by_default(qapp):
    bar = TagFilterBar()
    f = bar.current_filter()
    assert isinstance(f, TagFilter)
    assert f.is_empty()


def test_tag_filter_bar_cycles_three_states(qapp):
    bar = TagFilterBar()
    bar.set_available_tags(["a", "b"])
    received: list[TagFilter] = []
    bar.filter_changed.connect(received.append)

    # neither -> include
    bar._advance_state("a")
    f = bar.current_filter()
    assert f.includes == ("a",)
    assert f.excludes == ()

    # include -> exclude
    bar._advance_state("a")
    f = bar.current_filter()
    assert f.includes == ()
    assert f.excludes == ("a",)

    # exclude -> neither
    bar._advance_state("a")
    assert bar.current_filter().is_empty()

    # three signals fired (one per state advance)
    assert len(received) == 3


def test_tag_filter_bar_carries_both_sets(qapp):
    bar = TagFilterBar()
    bar.set_available_tags(["a", "b", "c"])

    bar._advance_state("a")  # a -> include
    bar._advance_state("b"); bar._advance_state("b")  # b -> exclude

    f = bar.current_filter()
    assert f.includes == ("a",)
    assert f.excludes == ("b",)


def test_tag_filter_bar_mode_visibility_depends_on_includes_only(qapp):
    bar = TagFilterBar()
    bar.set_available_tags(["a", "b", "c"])

    # 0 includes: mode hidden
    assert bar._mode_or.isHidden()
    assert bar._mode_and.isHidden()

    # 1 include: still hidden
    bar._advance_state("a")
    assert bar._mode_or.isHidden()

    # 2 includes: visible
    bar._advance_state("b")
    assert not bar._mode_or.isHidden()
    assert not bar._mode_and.isHidden()

    # 2 excludes (no includes): mode hidden again
    bar.clear()
    bar._advance_state("a"); bar._advance_state("a")  # a -> exclude
    bar._advance_state("b"); bar._advance_state("b")  # b -> exclude
    assert bar._mode_or.isHidden()


def test_tag_filter_bar_button_label_summary(qapp):
    bar = TagFilterBar()
    bar.set_available_tags(["a", "b", "c"])
    assert bar._btn.text() == "Tag 筛选: 全部"

    bar._advance_state("a")
    assert bar._btn.text() == "Tag: 含 a"

    bar._advance_state("a")  # a now in excludes
    assert bar._btn.text() == "Tag: 不含 a"

    bar._advance_state("b")  # b in includes, a in excludes
    assert bar._btn.text() == "Tag: 含 1 项 / 不含 1 项"

    bar._advance_state("c")  # c in includes too
    assert bar._btn.text() == "Tag: 含 2 项 / 不含 1 项"


def test_tag_filter_bar_drops_stale_selections(qapp):
    bar = TagFilterBar()
    bar.set_available_tags(["a", "b"])
    bar._advance_state("a")  # a -> include
    bar._advance_state("b"); bar._advance_state("b")  # b -> exclude

    bar.set_available_tags(["a", "c"])
    f = bar.current_filter()
    assert f.includes == ("a",)
    assert f.excludes == ()


def test_tag_filter_bar_clear_resets_both_sets(qapp):
    bar = TagFilterBar()
    bar.set_available_tags(["a", "b"])
    bar._advance_state("a")
    bar._advance_state("b"); bar._advance_state("b")

    received: list[TagFilter] = []
    bar.filter_changed.connect(received.append)
    bar.clear()

    assert bar.current_filter().is_empty()
    assert len(received) == 1


def test_tag_manager_dialog_diffs(qapp):
    dlg = TagManagerDialog(["existing", "to_remove"])
    # Add new
    dlg._edit.setText("new_tag")
    dlg._on_add()
    # Remove "to_remove" (row index 1)
    dlg._list.setCurrentRow(1)
    dlg._on_remove()
    final = dlg.get_tags()
    assert "existing" in final
    assert "new_tag" in final
    assert "to_remove" not in final
    assert dlg.get_renames() == {}


def test_tag_apply_bar_arm_disarm_cycle(qapp):
    from src.ui.tag_widget import TagApplyBar
    bar = TagApplyBar()
    bar.set_available_tags(["a", "b", "c"])
    received: list = []
    bar.armed_changed.connect(received.append)

    bar._on_chip_clicked("a")
    assert bar.get_armed() == "a"
    assert received == ["a"]

    bar._on_chip_clicked("a")
    assert bar.get_armed() is None
    assert received == ["a", None]


def test_tag_apply_bar_switch_arm(qapp):
    from src.ui.tag_widget import TagApplyBar
    bar = TagApplyBar()
    bar.set_available_tags(["a", "b"])
    received: list = []
    bar.armed_changed.connect(received.append)

    bar._on_chip_clicked("a")
    bar._on_chip_clicked("b")
    assert bar.get_armed() == "b"
    assert received == ["a", "b"]


def test_tag_apply_bar_auto_disarm_on_registry_change(qapp):
    from src.ui.tag_widget import TagApplyBar
    bar = TagApplyBar()
    bar.set_available_tags(["a", "b", "c"])
    bar._on_chip_clicked("a")
    received: list = []
    bar.armed_changed.connect(received.append)

    bar.set_available_tags(["b", "c"])
    assert bar.get_armed() is None
    assert received == [None]


def test_tag_apply_bar_clear_armed_idempotent(qapp):
    from src.ui.tag_widget import TagApplyBar
    bar = TagApplyBar()
    bar.set_available_tags(["a"])
    received: list = []
    bar.armed_changed.connect(received.append)

    bar.clear_armed()
    assert received == []
    bar._on_chip_clicked("a")
    bar.clear_armed()
    assert received == ["a", None]
    bar.clear_armed()  # idempotent — no re-emit
    assert received == ["a", None]


def test_tag_apply_bar_empty_registry(qapp):
    from src.ui.tag_widget import TagApplyBar
    bar = TagApplyBar()
    bar.set_available_tags([])
    assert bar.get_armed() is None
    assert bar._chip_count() == 0
