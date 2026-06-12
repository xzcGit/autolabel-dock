"""Tests for detection class mapping helpers."""

from src.core.annotation import Annotation, ImageAnnotation
from src.core.class_mapping import resolve_detection_class_map


def test_resolve_detection_class_map_filters_to_actual_bbox_classes():
    image_annotations = [
        ImageAnnotation(
            image_path="a.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
                Annotation(class_name="outdoor", class_id=2, bbox=None, confirmed=True),
            ],
        ),
    ]

    class_map = resolve_detection_class_map(
        image_annotations,
        project_classes=["cat", "dog", "outdoor"],
        only_confirmed=False,
    )

    assert class_map.names == ["cat"]
    assert class_map.id_by_name == {"cat": 0}


def test_resolve_detection_class_map_keeps_project_order_and_appends_unknowns():
    image_annotations = [
        ImageAnnotation(
            image_path="a.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="dog", class_id=1, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
                Annotation(class_name="zebra", class_id=9, bbox=(0.3, 0.3, 0.1, 0.1), confirmed=True),
            ],
        ),
    ]

    class_map = resolve_detection_class_map(
        image_annotations,
        project_classes=["cat", "dog", "bird"],
        only_confirmed=False,
    )

    assert class_map.names == ["dog", "zebra"]
    assert class_map.id_by_name == {"dog": 0, "zebra": 1}


def test_resolve_detection_class_map_honors_only_confirmed():
    image_annotations = [
        ImageAnnotation(
            image_path="a.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
                Annotation(class_name="dog", class_id=1, bbox=(0.3, 0.3, 0.1, 0.1), confirmed=False),
            ],
        ),
    ]

    class_map = resolve_detection_class_map(
        image_annotations,
        project_classes=["cat", "dog"],
        only_confirmed=True,
    )

    assert class_map.names == ["cat"]
