"""Reusable UI widgets for the tag subsystem.

Three independent widgets — none depends on the others — designed to be
mounted into existing panels with minimal coupling:

  - ``TagChipBar``: per-image editor (used inside AnnotationPanel)
  - ``TagFilterBar``: multi-select + AND/OR mode (used by file list,
    train panel, or any other view that needs to filter by tag)
  - ``TagManagerDialog``: project-level CRUD modal

All three communicate via Qt signals carrying plain Python objects
(``list[str]`` / ``TagFilter``) so they have no implicit coupling to the
controller layer — wire them up in MainWindow.
"""
from __future__ import annotations

from typing import Iterable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from src.core.tags import TagError, TagFilter, normalize
from src.ui.theme import chip_style, text_style


_CHIP_STYLE = chip_style()


class TagChipBar(QWidget):
    """Flow of clickable "chips" + "+ tag" button.

    Click a chip to remove that tag from the image. Click "+" to open a
    popup that lists known project tags and a free-input box for new ones.
    """

    tags_changed = pyqtSignal(list)  # list[str]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._available: list[str] = []
        self._current: list[str] = []
        self._read_only: bool = False
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._add_btn = QToolButton(self)
        self._add_btn.setText("+ Tag")
        self._add_btn.setStyleSheet(_CHIP_STYLE)
        self._add_btn.setPopupMode(QToolButton.InstantPopup)
        self._add_btn.clicked.connect(self._show_add_popup)
        self._layout.addWidget(self._add_btn)
        self._layout.addStretch(1)
        self._rebuild()

    def set_available_tags(self, tags: Iterable[str]) -> None:
        self._available = list(tags)

    def set_tags(self, tags: Iterable[str]) -> None:
        self._current = list(tags)
        self._rebuild()

    def get_tags(self) -> list[str]:
        return list(self._current)

    def set_read_only(self, ro: bool) -> None:
        self._read_only = ro
        self._add_btn.setVisible(not ro)
        self._rebuild()

    # ── internals ────────────────────────────────────────────

    def _rebuild(self) -> None:
        # remove chip widgets (keep add_btn + stretch)
        while self._layout.count() > 2:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for t in self._current:
            chip = QToolButton(self)
            chip.setText(f"{t}  ×" if not self._read_only else t)
            chip.setStyleSheet(_CHIP_STYLE)
            chip.setEnabled(not self._read_only)
            chip.clicked.connect(lambda _checked=False, tag=t: self._remove(tag))
            self._layout.insertWidget(self._layout.count() - 2, chip)

    def _remove(self, tag: str) -> None:
        if tag in self._current:
            self._current = [t for t in self._current if t != tag]
            self._rebuild()
            self.tags_changed.emit(list(self._current))

    def _show_add_popup(self) -> None:
        menu = QMenu(self)
        # Available tags not yet selected
        unused = [t for t in self._available if t not in self._current]
        if unused:
            for t in unused:
                act = menu.addAction(t)
                act.triggered.connect(lambda _checked=False, tag=t: self._add(tag))
            menu.addSeparator()
        # Free-input row
        custom = QLineEdit()
        custom.setPlaceholderText("输入新 tag 并回车")
        custom.returnPressed.connect(lambda: self._on_custom(custom.text(), menu))
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(custom)
        menu.addAction(wa)
        menu.exec_(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    def _on_custom(self, text: str, menu: QMenu) -> None:
        try:
            t = normalize(text)
        except TagError as e:
            QMessageBox.warning(self, "Tag 无效", str(e))
            return
        self._add(t)
        menu.close()

    def _add(self, tag: str) -> None:
        if tag in self._current:
            return
        self._current.append(tag)
        self._rebuild()
        self.tags_changed.emit(list(self._current))


class TagFilterBar(QWidget):
    """Tri-state tag filter: each tag chip cycles through none/include/exclude.

    Emits a ``TagFilter`` payload on every state change. Empty selection
    emits an empty filter (treated as "no filter").

    Conflict by construction: a single tag cannot be both included and
    excluded (cycling enforces ``includes ∩ excludes = ∅``). Conflicts
    between *different* tags on the same image (image has A and B; A in
    includes, B in excludes) are resolved at match-time by ``TagFilter``:
    exclude wins.
    """

    filter_changed = pyqtSignal(object)  # TagFilter

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._available: list[str] = []
        self._includes: set[str] = set()
        self._excludes: set[str] = set()
        self._mode: str = "or"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._btn = QToolButton(self)
        self._btn.setText("Tag 筛选: 全部")
        self._btn.setPopupMode(QToolButton.InstantPopup)
        self._btn.setStyleSheet(_CHIP_STYLE)
        self._btn.clicked.connect(self._show_menu)
        layout.addWidget(self._btn)

        # Mode toggle (OR / AND) — applies only to includes.
        self._mode_or = QRadioButton("任一")
        self._mode_and = QRadioButton("全部")
        self._mode_or.setChecked(True)
        self._mode_or.toggled.connect(self._on_mode_changed)
        grp = QButtonGroup(self)
        grp.addButton(self._mode_or)
        grp.addButton(self._mode_and)
        layout.addWidget(self._mode_or)
        layout.addWidget(self._mode_and)
        self._update_mode_visibility()
        layout.addStretch(1)

    def set_available_tags(self, tags: Iterable[str]) -> None:
        new_set = set(tags)
        self._available = sorted(new_set)
        # Drop selections that no longer exist on either side.
        self._includes &= new_set
        self._excludes &= new_set
        self._update_button_text()

    def current_filter(self) -> TagFilter:
        if not self._includes and not self._excludes:
            return TagFilter()
        return TagFilter(
            includes=tuple(sorted(self._includes)),
            excludes=tuple(sorted(self._excludes)),
            mode="and" if self._mode == "and" else "or",
        )

    def clear(self) -> None:
        if self._includes or self._excludes:
            self._includes.clear()
            self._excludes.clear()
            self._update_button_text()
            self.filter_changed.emit(self.current_filter())

    # ── internals ────────────────────────────────────────────

    def _show_menu(self) -> None:
        menu = QMenu(self)
        if not self._available:
            act = menu.addAction("（无可用 tag）")
            act.setEnabled(False)
        else:
            actions: dict[str, object] = {}
            for t in self._available:
                act = menu.addAction(self._action_text(t))
                act.triggered.connect(
                    lambda _checked=False, tag=t: self._on_cycle(
                        tag, actions[tag]
                    )
                )
                actions[t] = act
            menu.addSeparator()
            clear_act = menu.addAction("清空选择")
            clear_act.triggered.connect(self.clear)
        menu.exec_(self._btn.mapToGlobal(self._btn.rect().bottomLeft()))

    def _action_text(self, tag: str) -> str:
        if tag in self._includes:
            return f"✓  {tag}"
        if tag in self._excludes:
            return f"✗  {tag}"
        return f"     {tag}"

    def _advance_state(self, tag: str) -> None:
        """Cycle tag state: neither → include → exclude → neither.

        Pure state mutation + signal emission. The popup action's text is
        refreshed separately (see ``_on_cycle``) so this method is callable
        from tests without a live QAction.
        """
        if tag in self._includes:
            self._includes.discard(tag)
            self._excludes.add(tag)
        elif tag in self._excludes:
            self._excludes.discard(tag)
        else:
            self._includes.add(tag)
        self._update_button_text()
        self.filter_changed.emit(self.current_filter())

    def _on_cycle(self, tag: str, act) -> None:
        self._advance_state(tag)
        act.setText(self._action_text(tag))

    def _on_mode_changed(self) -> None:
        new_mode = "and" if self._mode_and.isChecked() else "or"
        if new_mode == self._mode:
            return
        self._mode = new_mode
        if self._includes:
            self.filter_changed.emit(self.current_filter())

    def _update_mode_visibility(self) -> None:
        visible = len(self._includes) >= 2
        self._mode_or.setVisible(visible)
        self._mode_and.setVisible(visible)

    def _update_button_text(self) -> None:
        inc = len(self._includes)
        exc = len(self._excludes)
        if inc == 0 and exc == 0:
            self._btn.setText("Tag 筛选: 全部")
        elif inc == 1 and exc == 0:
            (only,) = tuple(self._includes)
            self._btn.setText(f"Tag: 含 {only}")
        elif inc == 0 and exc == 1:
            (only,) = tuple(self._excludes)
            self._btn.setText(f"Tag: 不含 {only}")
        elif inc >= 2 and exc == 0:
            self._btn.setText(f"Tag: 含 {inc} 项")
        elif inc == 0 and exc >= 2:
            self._btn.setText(f"Tag: 不含 {exc} 项")
        else:
            self._btn.setText(f"Tag: 含 {inc} 项 / 不含 {exc} 项")
        self._update_mode_visibility()


_ARMED_CHIP_STYLE = chip_style(active=True)


class TagApplyBar(QWidget):
    """Always-visible strip of project tag chips. Click to arm one tag;
    LabelPanel reads ``get_armed()`` when the user presses T to apply.

    State is local — ``armed_changed`` emits whenever the armed value
    transitions (including auto-disarm when ``set_available_tags``
    removes the current tag from the registry).
    """

    armed_changed = pyqtSignal(object)  # str | None

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._available: list[str] = []
        self._armed: str | None = None
        self._chips: dict[str, QToolButton] = {}

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._hint = QLabel("按 T 应用到选中")
        self._hint.setStyleSheet(text_style("hint"))
        self._layout.addWidget(self._hint)

        self._empty_label = QLabel("无项目 tag——用「编辑 → Tag 管理」添加")
        self._empty_label.setStyleSheet(text_style("hint"))
        self._empty_label.hide()
        self._layout.addWidget(self._empty_label)

        self._layout.addStretch(1)
        self._rebuild()

    # ── Public API ───────────────────────────────────────────

    def set_available_tags(self, tags: Iterable[str]) -> None:
        new_list = list(tags)
        self._available = new_list
        # Auto-disarm if the armed tag is gone.
        if self._armed is not None and self._armed not in new_list:
            self._armed = None
            self._rebuild()
            self.armed_changed.emit(None)
            return
        self._rebuild()

    def get_armed(self) -> str | None:
        return self._armed

    def clear_armed(self) -> None:
        if self._armed is None:
            return
        self._armed = None
        self._restyle_all()
        self.armed_changed.emit(None)

    # ── Test helper (not for production callers) ─────────────

    def _chip_count(self) -> int:
        return len(self._chips)

    # ── Internals ────────────────────────────────────────────

    def _on_chip_clicked(self, tag: str) -> None:
        if self._armed == tag:
            self._armed = None
            self._restyle_all()
            self.armed_changed.emit(None)
            return
        self._armed = tag
        self._restyle_all()
        self.armed_changed.emit(tag)

    def _rebuild(self) -> None:
        # Remove existing chip widgets (keep hint, empty_label, stretch).
        for chip in list(self._chips.values()):
            self._layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()

        if not self._available:
            self._hint.hide()
            self._empty_label.show()
            return

        self._empty_label.hide()
        self._hint.show()
        # Insert chips between hint and stretch (stretch is the last item).
        insert_at = self._layout.count() - 1
        for tag in self._available:
            chip = QToolButton(self)
            chip.setText(tag)
            chip.clicked.connect(
                lambda _checked=False, t=tag: self._on_chip_clicked(t)
            )
            self._layout.insertWidget(insert_at, chip)
            self._chips[tag] = chip
            insert_at += 1
        self._restyle_all()

    def _restyle_all(self) -> None:
        for tag, chip in self._chips.items():
            chip.setStyleSheet(
                _ARMED_CHIP_STYLE if tag == self._armed else _CHIP_STYLE
            )


class TagManagerDialog(QDialog):
    """Project-level tag CRUD. Returns the desired final tag list via ``get_tags``.

    Editing is local until the dialog is accepted — the calling controller
    diffs against the original list and applies add/remove/rename.
    """

    def __init__(self, tags: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Tag 管理")
        self.setMinimumWidth(340)
        self._original: list[str] = list(tags)
        # Track renames: original -> new
        self._renames: dict[str, str] = {}
        self._tags: list[str] = list(tags)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        for t in self._tags:
            self._list.addItem(QListWidgetItem(t))
        self._list.itemDoubleClicked.connect(self._on_rename)
        layout.addWidget(self._list)

        add_row = QHBoxLayout()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("新 tag 名称")
        self._edit.returnPressed.connect(self._on_add)
        add_row.addWidget(self._edit)
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self._on_add)
        add_row.addWidget(btn_add)
        layout.addLayout(add_row)

        self._status = QLabel("")
        self._status.setStyleSheet(text_style("error"))
        layout.addWidget(self._status)

        btn_remove = QPushButton("删除选中 tag")
        btn_remove.clicked.connect(self._on_remove)
        layout.addWidget(btn_remove)

        hint = QLabel("双击 tag 可重命名。删除会同步移除所有图片上的该 tag。")
        hint.setWordWrap(True)
        hint.setStyleSheet(text_style("hint"))
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_add(self) -> None:
        try:
            t = normalize(self._edit.text())
        except TagError as e:
            self._status.setText(str(e))
            return
        if t in self._tags:
            self._status.setText(f"Tag \"{t}\" 已存在")
            return
        self._status.setText("")
        self._tags.append(t)
        self._list.addItem(QListWidgetItem(t))
        self._edit.clear()

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        removed = self._tags.pop(row)
        self._list.takeItem(row)
        # If this was a rename target, drop that rename mapping
        for k, v in list(self._renames.items()):
            if v == removed:
                del self._renames[k]
        self._status.setText(f"已删除 \"{removed}\"")

    def _on_rename(self, item: QListWidgetItem) -> None:
        old = item.text()
        new, ok = QInputDialog.getText(self, "重命名 tag", "新名称:", text=old)
        if not ok:
            return
        try:
            new_norm = normalize(new)
        except TagError as e:
            QMessageBox.warning(self, "Tag 无效", str(e))
            return
        if new_norm == old:
            return
        if new_norm in self._tags:
            QMessageBox.warning(self, "冲突", f"Tag \"{new_norm}\" 已存在")
            return
        idx = self._tags.index(old)
        self._tags[idx] = new_norm
        item.setText(new_norm)
        # Resolve rename chain: if old was itself a renamed-from, update
        original = old
        for k, v in self._renames.items():
            if v == old:
                original = k
                break
        self._renames[original] = new_norm

    def get_tags(self) -> list[str]:
        return list(self._tags)

    def get_renames(self) -> dict[str, str]:
        """Return {original_name: new_name} for tags that were renamed."""
        return {k: v for k, v in self._renames.items() if k != v}
