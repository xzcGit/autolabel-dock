"""Tests for FileListWidget."""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt, QPointF, QEvent
from PyQt5.QtGui import QImage, QColor, QMouseEvent


def _make_test_image(path: Path, width: int = 100, height: int = 80) -> None:
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(Qt.red))
    img.save(str(path), "PNG")


class TestFileListWidget:
    def test_set_image_paths(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(5)]
        widget.set_image_paths(paths)
        assert widget.count() == 5

    def test_get_current_path(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg")]
        widget.set_image_paths(paths)
        widget.setCurrentRow(1)
        assert widget.get_current_path() == paths[1]

    def test_get_current_path_none_when_empty(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        assert widget.get_current_path() is None

    def test_set_status(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg")]
        widget.set_image_paths(paths)

        widget.set_status(paths[0], "confirmed")
        widget.set_status(paths[1], "pending")

        assert widget._statuses[str(paths[0])] == "confirmed"
        assert widget._statuses[str(paths[1])] == "pending"

    def test_filter_by_status(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(4)]
        widget.set_image_paths(paths)
        widget.set_status(paths[0], "confirmed")
        widget.set_status(paths[1], "pending")
        widget.set_status(paths[2], "unlabeled")
        widget.set_status(paths[3], "confirmed")

        widget.set_filter("confirmed")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 2

    def test_filter_all_shows_everything(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)
        widget.set_status(paths[0], "confirmed")
        widget.set_filter("confirmed")
        widget.set_filter(None)  # show all
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 3

    def test_navigate_next_prev(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)
        widget.setCurrentRow(0)

        widget.go_next()
        assert widget.currentRow() == 1
        widget.go_next()
        assert widget.currentRow() == 2
        widget.go_next()
        assert widget.currentRow() == 2  # stays at end

        widget.go_prev()
        assert widget.currentRow() == 1
        widget.go_prev()
        assert widget.currentRow() == 0
        widget.go_prev()
        assert widget.currentRow() == 0  # stays at start

    def test_current_index_info(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(5)]
        widget.set_image_paths(paths)
        widget.setCurrentRow(2)
        idx, total = widget.get_index_info()
        assert idx == 3  # 1-based
        assert total == 5

    def test_extended_selection_mode(self, qapp):
        from src.ui.file_list import FileListWidget
        from PyQt5.QtWidgets import QAbstractItemView

        widget = FileListWidget()
        assert widget.selectionMode() == QAbstractItemView.ExtendedSelection

    def test_get_selected_paths(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(5)]
        widget.set_image_paths(paths)
        # Select multiple items
        widget.item(1).setSelected(True)
        widget.item(3).setSelected(True)
        selected = widget.get_selected_paths()
        assert len(selected) == 2

    def test_batch_confirm_signal(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        results = []
        widget.batch_confirm_requested.connect(lambda p: results.append(p))
        # Simulate signal emission
        widget.batch_confirm_requested.emit([Path("/img1.jpg")])
        assert len(results) == 1

    def test_batch_delete_signal(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        results = []
        widget.batch_delete_requested.connect(lambda p: results.append(p))
        widget.batch_delete_requested.emit([Path("/img1.jpg")])
        assert len(results) == 1

    def test_delete_images_signal(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        results = []
        widget.delete_images_requested.connect(lambda p: results.append(p))
        widget.delete_images_requested.emit([Path("/a.jpg"), Path("/b.jpg")])
        assert len(results) == 1
        assert results[0] == [Path("/a.jpg"), Path("/b.jpg")]

    def test_forget_paths_drops_status_and_classes(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)
        widget.set_status(paths[0], "confirmed")
        widget.set_image_classes(paths[0], {"cat"})

        widget.forget_paths([paths[0]])

        assert str(paths[0]) not in widget._statuses
        assert str(paths[0]) not in widget._image_classes

    def test_class_filter(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(4)]
        widget.set_image_paths(paths)

        # Set classes for images
        widget.set_image_classes(paths[0], {"cat", "dog"})
        widget.set_image_classes(paths[1], {"cat"})
        widget.set_image_classes(paths[2], {"dog"})
        widget.set_image_classes(paths[3], set())  # no annotations

        # Filter by "cat"
        widget.set_class_filter("cat")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 2  # img0 and img1

        # Filter by "dog"
        widget.set_class_filter("dog")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 2  # img0 and img2

        # Clear filter
        widget.set_class_filter(None)
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 4

    def test_combined_status_and_class_filter(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        widget.set_image_paths(paths)

        widget.set_status(paths[0], "confirmed")
        widget.set_status(paths[1], "pending")
        widget.set_status(paths[2], "confirmed")
        widget.set_image_classes(paths[0], {"cat"})
        widget.set_image_classes(paths[1], {"cat"})
        widget.set_image_classes(paths[2], {"dog"})

        # Filter: confirmed + cat
        widget.set_filter("confirmed")
        widget.set_class_filter("cat")
        visible = [i for i in range(widget.count()) if not widget.item(i).isHidden()]
        assert len(visible) == 1  # only img0

    def test_get_paths_returns_copy(self, qapp):
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg"), Path("/imgs/c.jpg")]
        widget.set_image_paths(paths)

        got = widget.get_paths()
        assert got == paths
        got.append(Path("/imgs/x.jpg"))
        assert widget.get_paths() == paths

    def test_refresh_paths_noop_when_paths_unchanged(self, qapp):
        """Regression: refresh_paths must skip the rebuild when paths are identical.

        Otherwise QListWidget.clear()+addItem*N resets scroll to top, then the
        subsequent setCurrentRow(prev_idx) auto-scrolls so the previously-current
        row sits at the BOTTOM edge of the viewport — perceived as "the list
        scrolled to the end".
        """
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        paths = [Path(f"/imgs/img_{i:03}.png") for i in range(50)]
        widget.set_image_paths(paths)
        widget.setCurrentRow(40)

        # Mark the existing items so we can detect a rebuild
        for i in range(widget.count()):
            widget.item(i).setData(Qt.UserRole + 99, "sentinel")

        widget.refresh_paths(list(paths))

        # If items were rebuilt, the sentinel data would be gone.
        for i in range(widget.count()):
            assert widget.item(i).data(Qt.UserRole + 99) == "sentinel", \
                f"refresh_paths rebuilt item {i} despite paths being unchanged"
        assert widget.currentRow() == 40

    def test_apply_filter_does_not_clamp_scroll_to_max(self, qapp):
        """Regression: applying a filter must keep the current item visible
        instead of letting Qt clamp scroll to the new (smaller) maximum.

        Bug: when a status filter hid many items, the scrollbar maximum
        shrank, and Qt clamped the existing scroll value to that new max,
        snapping the viewport to the very bottom of the visible items —
        the user perceived this as "the list scrolling to the end".
        """
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        widget.resize(200, 300)
        widget.show()
        try:
            paths = [Path(f"/imgs/img_{i:03}.png") for i in range(100)]
            widget.set_image_paths(paths)
            # Mark every 5th as confirmed; current image is confirmed (visible).
            for i in range(0, 100, 5):
                widget.set_status(paths[i], "confirmed")
            widget.setCurrentRow(50)
            qapp.processEvents()

            sb = widget.verticalScrollBar()
            old_value = sb.value()
            old_max = sb.maximum()
            assert old_value > 0, "test setup: scroll should be non-zero before filter"

            widget.set_filter("confirmed")
            qapp.processEvents()
            new_max = sb.maximum()
            new_value = sb.value()

            assert new_max < old_max, "test setup: filter should shrink max"
            assert new_max > 0, "test setup: filter should still leave scrollable range"
            assert not widget.item(50).isHidden(), "row 50 must still be visible"
            # The bug clamps scroll to new_max (current item ends up at the
            # bottom of the viewport, which user reads as "scrolled to end").
            assert new_value < new_max, (
                f"after filter, scroll={new_value} == max={new_max}: "
                "bug — scroll was clamped to bottom of list"
            )
        finally:
            widget.close()

    def test_right_click_after_pending_filter_does_not_navigate_or_scroll(self, qapp):
        """Regression: right-clicking a filtered list item is only a context action.

        QListWidget's default right-button handling changes the current row.
        In a filtered list that image switch can make Qt ensure a far-away
        source row is visible, so the visible pending list appears to flip.
        """
        from src.ui.file_list import FileListWidget

        widget = FileListWidget()
        widget.resize(220, 260)
        widget.show()
        try:
            paths = [Path(f"/imgs/img_{i:03}.png") for i in range(100)]
            widget.set_image_paths(paths)
            for i, path in enumerate(paths):
                if i % 3 == 0:
                    widget.set_status(path, "pending")

            widget.setCurrentRow(30)
            widget.set_filter("pending")
            qapp.processEvents()

            sb = widget.verticalScrollBar()
            sb.setValue(4)
            qapp.processEvents()
            before_row = widget.currentRow()
            before_path = widget.get_current_path()
            before_scroll = sb.value()
            selected = []
            widget.image_selected.connect(lambda path: selected.append(path))
            assert before_row == 30

            click_pos = widget.visualItemRect(widget.item(45)).center()
            assert widget.itemAt(click_pos) is widget.item(45)
            assert before_row != 45

            event = QMouseEvent(
                QEvent.MouseButtonPress,
                QPointF(click_pos),
                Qt.RightButton,
                Qt.RightButton,
                Qt.NoModifier,
            )
            widget.mousePressEvent(event)
            qapp.processEvents()

            assert widget.currentRow() == before_row
            assert widget.get_current_path() == before_path
            assert sb.value() == before_scroll
            assert selected == []
        finally:
            widget.close()
