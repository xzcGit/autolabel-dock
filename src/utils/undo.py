"""Undo/redo stack using command pattern with deep-copied snapshots."""
from __future__ import annotations

import copy


class UndoStack:
    """Per-image undo/redo stack with configurable max depth."""

    def __init__(self, max_depth: int = 50):
        self._max_depth = max_depth
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 1

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def push(self, state: dict) -> None:
        """Push a new state snapshot (deep-copied). Clears redo stack."""
        self._undo_stack.append(copy.deepcopy(state))
        self._redo_stack.clear()
        # Enforce max depth
        while len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)

    def undo(self) -> dict | None:
        """Undo to previous state. Returns the restored state or None."""
        if not self.can_undo:
            return None
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        return copy.deepcopy(self._undo_stack[-1])

    def redo(self) -> dict | None:
        """Redo to next state. Returns the restored state or None."""
        if not self.can_redo:
            return None
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        return copy.deepcopy(state)

    def clear(self) -> None:
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
