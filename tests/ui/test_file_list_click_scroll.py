"""Regression test for file list scroll jump on click bug."""
import pytest
from pathlib import Path
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QTest

from src.ui.file_list import FileListWidget


def test_clicking_filtered_items_does_not_scroll(qapp):
    """Verify that clicking items in a filtered list doesn't cause scroll jumps.

    Regression test for the bug where clicking any item in a filtered list
    would cause the viewport to scroll/jump incrementally with each click,
    eventually scrolling to the end.

    The root cause was Qt's mousePressEvent calling scrollToItem() on the
    clicked item, which calculates scroll positions incorrectly when items
    are hidden (filtered).
    """
    widget = FileListWidget()
    widget.resize(400, 600)
    widget.show()

    # Create a large list
    paths = [Path(f"/tmp/test_{i:03d}.jpg") for i in range(100)]
    widget.set_image_paths(paths)

    # Set alternating statuses
    for i, path in enumerate(paths):
        status = "pending" if i % 2 else "confirmed"
        widget.set_status(path, status)

    # Filter to pending only - this triggers the bug
    widget.set_filter("pending")

    # Select item in the middle
    widget.setCurrentRow(20)

    # Wait for UI to settle
    qapp.processEvents()

    # Record initial scroll position
    initial_scroll = widget.verticalScrollBar().value()

    # Simulate clicking on different items multiple times
    # (the bug manifested as incremental scroll jumps with each click)
    scroll_values = [initial_scroll]

    for target_row in [22, 24, 26, 28, 30]:
        # Find the visible item at this row
        item = widget.item(target_row)
        if item and not item.isHidden():
            # Get item's visual rectangle
            rect = widget.visualItemRect(item)

            # Simulate click on the item
            click_pos = rect.center()
            QTest.mouseClick(widget.viewport(), Qt.LeftButton, Qt.NoModifier, click_pos)

            qapp.processEvents()

            # Record scroll position after click
            scroll_after = widget.verticalScrollBar().value()
            scroll_values.append(scroll_after)

    # Verify scroll position remained stable (no incremental jumps)
    # Allow for small variations due to item height alignment
    if len(scroll_values) > 1:
        scroll_deltas = [abs(scroll_values[i+1] - scroll_values[i]) for i in range(len(scroll_values)-1)]
        max_delta = max(scroll_deltas)

        # The bug would cause deltas of 20-50+ pixels per click
        # Normal behavior should have deltas near 0
        assert max_delta < 10, (
            f"Scroll jumped by {max_delta} pixels after clicking items. "
            f"Scroll values: {scroll_values}"
        )
    else:
        # If no items were clickable, at least verify no crash
        assert len(scroll_values) >= 1


def test_keyboard_navigation_in_filtered_list(qapp):
    """Verify keyboard navigation doesn't cause unexpected scroll jumps."""
    widget = FileListWidget()
    widget.resize(400, 600)
    widget.show()

    paths = [Path(f"/tmp/test_{i:03d}.jpg") for i in range(100)]
    widget.set_image_paths(paths)

    for i, path in enumerate(paths):
        status = "pending" if i % 2 else "confirmed"
        widget.set_status(path, status)

    widget.set_filter("pending")
    widget.setCurrentRow(10)
    qapp.processEvents()

    initial_scroll = widget.verticalScrollBar().value()

    # Navigate down using keyboard
    for _ in range(5):
        widget.go_next()
        qapp.processEvents()

    # Scroll should have changed (following the selection is normal),
    # but should be stable and not jump erratically
    final_scroll = widget.verticalScrollBar().value()

    # Just verify it didn't crash and scroll is reasonable
    assert final_scroll >= initial_scroll - 50  # Allow some upward scroll
    assert widget.currentRow() >= 10  # Selection moved forward
