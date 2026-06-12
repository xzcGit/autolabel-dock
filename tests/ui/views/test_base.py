"""Smoke test for TaskView abstract interface."""
import pytest


def test_task_view_default_methods_raise(qapp):
    from src.ui.views.base import TaskView

    view = TaskView()
    with pytest.raises(NotImplementedError):
        view.set_project(None)
    with pytest.raises(NotImplementedError):
        view.set_class_colors({})
    with pytest.raises(NotImplementedError):
        view.set_classes([])
    with pytest.raises(NotImplementedError):
        view.set_filter(None)
    with pytest.raises(NotImplementedError):
        view.set_class_filter(None)
    with pytest.raises(NotImplementedError):
        view.get_focused_image()
    with pytest.raises(NotImplementedError):
        view.get_visible_paths()
    with pytest.raises(NotImplementedError):
        view.reload_current()
    with pytest.raises(NotImplementedError):
        view.commit_pending_save()
    with pytest.raises(NotImplementedError):
        view.undo()
    with pytest.raises(NotImplementedError):
        view.redo()
    with pytest.raises(NotImplementedError):
        view.add_auto_annotations([], 0.5)
    with pytest.raises(NotImplementedError):
        view.add_auto_class_prediction(None, "x", 0.0)


def test_task_view_signals_exist(qapp):
    from src.ui.views.base import TaskView

    view = TaskView()
    assert hasattr(view, "annotations_changed")
    assert hasattr(view, "status_changed")
    assert hasattr(view, "image_focus_changed")
    assert hasattr(view, "auto_label_single_requested")
    assert hasattr(view, "auto_label_batch_requested")
    assert hasattr(view, "images_dropped")
