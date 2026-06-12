"""File list widget for image navigation and status display."""
from __future__ import annotations

import shutil
from pathlib import Path

from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QMenu, QAction
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent, QMouseEvent

from src.core.project import IMAGE_EXTENSIONS
from src.core.tags import TagFilter
from src.ui.theme import PALETTE

# Status colors
STATUS_COLORS = {
    "confirmed": PALETTE["success"],
    "pending": PALETTE["warning"],
    "unlabeled": PALETTE["text_subtle"],
}

STATUS_ICONS = {
    "confirmed": "\u2713",    # ✓
    "pending": "\u26a1",      # ⚡
    "unlabeled": "\u25cb",    # ○
}


class FileListWidget(QListWidget):
    """Image file list with status indicators, filtering, and drag-and-drop.

    Signals:
        image_selected(Path): Emitted when user clicks a different image.
        images_dropped(list[Path]): Emitted when image files are dropped onto the list.
        batch_confirm_requested(list): Emitted with list of Paths to batch confirm.
        batch_delete_requested(list): Emitted with list of Paths to batch delete annotations.
        delete_images_requested(list): Emitted with list of Paths to delete (image file + label).
    """

    image_selected = pyqtSignal(object)  # Path
    images_dropped = pyqtSignal(list)    # list[Path]
    batch_confirm_requested = pyqtSignal(list)   # list[Path]
    batch_delete_requested = pyqtSignal(list)    # list[Path]
    delete_images_requested = pyqtSignal(list)   # list[Path]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._statuses: dict[str, str] = {}  # path_str -> status
        self._image_classes: dict[str, set[str]] = {}  # path_str -> set of class names
        self._image_tags: dict[str, set[str]] = {}  # path_str -> set of user tags
        self._filter: str | None = None  # None = show all
        self._class_filter: str | None = None  # None = show all classes
        self._tag_filter: TagFilter = TagFilter()  # empty = no tag filter

        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)

        # Disable Qt's automatic scroll-to-item behavior.
        # When items are hidden (filtered), Qt's ensureVisible() calculates
        # scroll positions incorrectly, causing the "scroll jump on click" bug.
        # We manage scroll position manually in _apply_filter instead.
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)

        self.currentRowChanged.connect(self._on_row_changed)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            event.accept()
            return

        # Save scroll position before Qt processes the click.
        # Qt's default mousePressEvent calls scrollToItem() on the clicked item,
        # which causes scroll jumps when items are hidden (filtered).
        sb = self.verticalScrollBar()
        scroll_before = sb.value()

        super().mousePressEvent(event)

        # Restore scroll position after Qt's processing.
        # This prevents the automatic scrollToItem() from jumping the viewport.
        # The clicked item is already visible (user just clicked it), so
        # Qt's auto-scroll is unnecessary and harmful when items are hidden.
        sb.setValue(scroll_before)

    # ── Drag and drop ─────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        image_paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
            elif path.is_dir():
                for ext in IMAGE_EXTENSIONS:
                    image_paths.extend(path.glob(f"*{ext}"))
        if image_paths:
            self.images_dropped.emit(image_paths)
        event.acceptProposedAction()

    def set_image_paths(self, paths: list[Path]) -> None:
        """Set the list of image paths to display."""
        self.blockSignals(True)
        self.clear()
        self._paths = list(paths)
        for path in paths:
            status = self._statuses.get(str(path), "unlabeled")
            icon = STATUS_ICONS.get(status, "○")
            item = QListWidgetItem(f"{icon} {path.name}")
            item.setData(Qt.UserRole, str(path))
            item.setForeground(QColor(STATUS_COLORS.get(status, PALETTE["text_subtle"])))
            self.addItem(item)
        self._apply_filter()
        self.blockSignals(False)

    def set_status(self, path: Path, status: str) -> None:
        """Update the status of an image file."""
        self._statuses[str(path)] = status
        # Update the item display
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.UserRole) == str(path):
                icon = STATUS_ICONS.get(status, "○")
                item.setText(f"{icon} {path.name}")
                item.setForeground(QColor(STATUS_COLORS.get(status, PALETTE["text_subtle"])))
                break

    def set_filter(self, status: str | None) -> None:
        """Filter items by status. None shows all."""
        self._filter = status
        self._apply_filter()

    def set_class_filter(self, class_name: str | None) -> None:
        """Filter items by class name. None shows all."""
        self._class_filter = class_name
        self._apply_filter()

    def set_tag_filter(self, tag_filter: TagFilter | None) -> None:
        """Filter items by user-defined tags. Empty / None disables the dimension."""
        self._tag_filter = tag_filter if tag_filter is not None else TagFilter()
        self._apply_filter()

    def set_image_classes(self, path: Path, classes: set[str]) -> None:
        """Set the class names present in an image's annotations."""
        self._image_classes[str(path)] = classes

    def set_image_tags(self, path: Path, tags: set[str]) -> None:
        """Set the user-defined tags assigned to an image (drives the tag filter).

        Also re-evaluates this row's visibility under the current filters so
        the tag filter stays consistent after a batch tag apply. Scroll is
        NOT touched — the bulk recompute in ``_apply_filter`` is the only
        path that re-positions the scrollbar.
        """
        self._image_tags[str(path)] = set(tags)
        self._refresh_row_visibility(path)

    def _refresh_row_visibility(self, path: Path) -> None:
        """Update one row's hidden flag from current cached state.

        Short-circuits when no filter is engaged. Never touches scroll.
        """
        if (
            self._filter is None
            and self._class_filter is None
            and self._tag_filter.is_empty()
        ):
            return
        path_str = str(path)
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.UserRole) == path_str:
                item.setHidden(self._compute_hidden_for_path(path_str))
                return

    def set_all_image_tags(self, tag_map: dict[str, set[str]]) -> None:
        """Bulk-replace the per-image tag cache (used on project open)."""
        self._image_tags = {k: set(v) for k, v in tag_map.items()}
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Apply current status and class filters to items.

        After hiding items, explicitly position the scrollbar so the current
        item sits in the upper portion of the viewport. Without this, Qt
        clamps the existing scroll value to the new (smaller) maximum,
        snapping the current item to the very bottom of the visible items —
        the user perceives this as "the list scrolling to the end".
        """
        current_row = self.currentRow()
        for i in range(self.count()):
            item = self.item(i)
            path_str = item.data(Qt.UserRole)
            item.setHidden(self._compute_hidden_for_path(path_str))

        sb = self.verticalScrollBar()
        if current_row < 0 or self.item(current_row).isHidden():
            sb.setValue(0)
            return
        # Visible-list index of the current row.
        cur_visible_idx = sum(
            1 for i in range(current_row + 1) if not self.item(i).isHidden()
        ) - 1
        item_h = self.sizeHintForRow(current_row) or 1
        viewport_items = max(1, self.viewport().height() // item_h)
        # Place current near viewport center so the user sees context above
        # and below — and crucially, so the LAST visible item is NOT pinned
        # to the viewport bottom (which reads as "list scrolled to the end").
        target = cur_visible_idx - viewport_items // 2
        sb.setValue(max(0, min(target, sb.maximum())))

    def _compute_hidden_for_path(self, path_str: str) -> bool:
        """Pure predicate: should the row at ``path_str`` be hidden right now?"""
        if self._filter is not None:
            item_status = self._statuses.get(path_str, "unlabeled")
            if item_status != self._filter:
                return True
        if self._class_filter is not None:
            img_classes = self._image_classes.get(path_str, set())
            if self._class_filter not in img_classes:
                return True
        if not self._tag_filter.is_empty():
            img_tags = self._image_tags.get(path_str, set())
            if not self._tag_filter.matches(img_tags):
                return True
        return False

    def get_current_path(self) -> Path | None:
        """Get the currently selected image path."""
        item = self.currentItem()
        if item is None:
            return None
        path_str = item.data(Qt.UserRole)
        return Path(path_str) if path_str else None

    def get_index_info(self) -> tuple[int, int]:
        """Get current 1-based index and visible count (respects active filter)."""
        row = self.currentRow()
        visible_count = sum(1 for i in range(self.count()) if not self.item(i).isHidden())
        if row < 0:
            return 0, visible_count
        # Compute 1-based visible index
        visible_idx = 0
        for i in range(row + 1):
            if not self.item(i).isHidden():
                visible_idx += 1
        return visible_idx, visible_count

    def go_next(self) -> None:
        """Navigate to next visible image."""
        row = self.currentRow()
        for i in range(row + 1, self.count()):
            if not self.item(i).isHidden():
                self.setCurrentRow(i)
                return

    def go_prev(self) -> None:
        """Navigate to previous visible image."""
        row = self.currentRow()
        for i in range(row - 1, -1, -1):
            if not self.item(i).isHidden():
                self.setCurrentRow(i)
                return

    def refresh_paths(self, paths: list[Path]) -> None:
        """Refresh file list, preserving statuses, selection, and scroll position.

        Skips the rebuild when ``paths`` matches the currently-displayed paths
        in the same order — clearing and re-populating a QListWidget resets
        scroll to top and Qt's ``EnsureVisible`` behaviour on the subsequent
        ``setCurrentRow`` would snap the previously-current item to the bottom
        edge of the viewport, which the user perceives as "scrolling to the
        end" of the list.
        """
        new_keys = [str(p) for p in paths]
        old_keys = [str(p) for p in self._paths]
        if new_keys == old_keys:
            return
        current = self.get_current_path()
        scroll_value = self.verticalScrollBar().value()
        self._paths = list(paths)
        self.set_image_paths(paths)
        if current:
            for i in range(self.count()):
                if self.item(i).data(Qt.UserRole) == str(current):
                    self.setCurrentRow(i)
                    break
        self.verticalScrollBar().setValue(scroll_value)

    def get_paths(self) -> list[Path]:
        """Return a copy of the current image paths list."""
        return list(self._paths)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            path = self.get_current_path()
            if path:
                self.image_selected.emit(path)

    def get_selected_paths(self) -> list[Path]:
        """Get all selected image paths."""
        paths = []
        for item in self.selectedItems():
            path_str = item.data(Qt.UserRole)
            if path_str:
                paths.append(Path(path_str))
        return paths

    def get_visible_paths(self) -> list[Path]:
        """Get paths of all currently visible (non-hidden) items."""
        paths = []
        for i in range(self.count()):
            item = self.item(i)
            if not item.isHidden():
                path_str = item.data(Qt.UserRole)
                if path_str:
                    paths.append(Path(path_str))
        return paths

    def contextMenuEvent(self, event) -> None:
        """Show right-click context menu for file list items."""
        item = self.itemAt(event.pos())
        if not item:
            return
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return

        selected = self.get_selected_paths()
        menu = QMenu(self)

        # Batch operations (when multiple selected)
        if len(selected) > 1:
            batch_confirm = menu.addAction(f"批量确认 ({len(selected)} 张)")
            batch_confirm.triggered.connect(lambda: self.batch_confirm_requested.emit(selected))

            batch_delete = menu.addAction(f"批量删除标注 ({len(selected)} 张)")
            batch_delete.triggered.connect(lambda: self.batch_delete_requested.emit(selected))

            delete_imgs = menu.addAction(f"删除图片 ({len(selected)} 张)")
            delete_imgs.triggered.connect(
                lambda: self.delete_images_requested.emit(selected)
            )

            menu.addSeparator()
        else:
            delete_imgs = menu.addAction("删除图片")
            delete_imgs.triggered.connect(
                lambda: self.delete_images_requested.emit([Path(path_str)])
            )
            menu.addSeparator()

        open_folder = menu.addAction("在文件管理器中打开")
        open_folder.triggered.connect(lambda: self._open_in_explorer(Path(path_str)))

        copy_path = menu.addAction("复制文件路径")
        copy_path.triggered.connect(lambda: self._copy_path(path_str))

        menu.exec_(event.globalPos())

    def forget_paths(self, paths) -> None:
        """Drop status / class entries for paths no longer in the list.

        Call after deleting images to prevent the internal dicts from growing
        unbounded over long sessions.
        """
        for p in paths:
            key = str(p)
            self._statuses.pop(key, None)
            self._image_classes.pop(key, None)
            self._image_tags.pop(key, None)

    def _open_in_explorer(self, path: Path) -> None:
        """Open the containing folder in the system file manager."""
        import subprocess, sys
        folder = str(path.parent)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", folder])

    def _copy_path(self, path_str: str) -> None:
        """Copy file path to clipboard."""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(path_str)
