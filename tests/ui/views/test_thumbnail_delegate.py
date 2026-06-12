"""Tests for ThumbnailDelegate visual state computation."""


def test_compute_visual_state_unlabeled():
    from src.ui.views.classify import _compute_visual_state
    from src.core.annotation import ImageAnnotation
    ia = ImageAnnotation(image_path="x.jpg", image_size=(10, 10))
    state = _compute_visual_state(ia, class_colors={}, default_color="#6c7086")
    assert state.label == "未标"
    assert state.bg_color == "#6c7086"
    assert state.status_glyph == "○"
    assert state.show_question_badge is False


def test_compute_visual_state_confirmed_manual():
    from src.ui.views.classify import _compute_visual_state
    from src.core.annotation import ImageAnnotation
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10),
        image_tags=["cat"], image_tags_confirmed=True, image_tags_source="manual",
    )
    state = _compute_visual_state(ia, class_colors={"cat": "#a6e3a1"}, default_color="#6c7086")
    assert state.label == "cat"
    assert state.bg_color == "#a6e3a1"
    assert state.status_glyph == "✓"
    assert state.show_question_badge is False


def test_compute_visual_state_pending_auto():
    from src.ui.views.classify import _compute_visual_state
    from src.core.annotation import ImageAnnotation
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10),
        image_tags=["cat"], image_tags_confirmed=False, image_tags_source="auto",
    )
    state = _compute_visual_state(ia, class_colors={"cat": "#a6e3a1"}, default_color="#6c7086")
    assert state.label == "cat"
    assert state.status_glyph == "⚡"
    assert state.show_question_badge is True
