"""Tests for undo/redo stack."""
from src.utils.undo import UndoStack


class TestUndoStack:
    def test_initial_state(self):
        stack = UndoStack(max_depth=50)
        assert not stack.can_undo
        assert not stack.can_redo

    def test_push_and_undo(self):
        stack = UndoStack()
        state = {"annotations": []}
        stack.push(state)
        state2 = {"annotations": [{"id": "1"}]}
        stack.push(state2)
        assert stack.can_undo
        restored = stack.undo()
        assert restored == state

    def test_redo_after_undo(self):
        stack = UndoStack()
        s1 = {"a": 1}
        s2 = {"a": 2}
        stack.push(s1)
        stack.push(s2)
        stack.undo()
        assert stack.can_redo
        restored = stack.redo()
        assert restored == s2

    def test_push_clears_redo(self):
        stack = UndoStack()
        stack.push({"a": 1})
        stack.push({"a": 2})
        stack.undo()
        assert stack.can_redo
        stack.push({"a": 3})
        assert not stack.can_redo

    def test_max_depth(self):
        stack = UndoStack(max_depth=3)
        for i in range(5):
            stack.push({"v": i})
        # Should only keep last 3 states
        count = 0
        while stack.can_undo:
            stack.undo()
            count += 1
        assert count == 2  # 3 states means 2 undos (current->prev->prev)

    def test_undo_empty_returns_none(self):
        stack = UndoStack()
        assert stack.undo() is None

    def test_redo_empty_returns_none(self):
        stack = UndoStack()
        assert stack.redo() is None

    def test_clear(self):
        stack = UndoStack()
        stack.push({"a": 1})
        stack.push({"a": 2})
        stack.clear()
        assert not stack.can_undo
        assert not stack.can_redo

    def test_deep_copy_isolation(self):
        """Ensure pushed states are deep-copied to avoid external mutation."""
        stack = UndoStack()
        data = {"items": [1, 2, 3]}
        stack.push(data)
        data["items"].append(4)  # mutate original
        stack.push(data)
        restored = stack.undo()
        assert restored["items"] == [1, 2, 3]  # should be the original
