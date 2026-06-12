"""Tests for tags field on ImageAnnotation + label_io empty-record handling."""
from __future__ import annotations

from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation, save_annotation


def test_image_annotation_tags_roundtrip():
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10), tags=["a", "b"],
    )
    d = ia.to_dict()
    ia2 = ImageAnnotation.from_dict(d)
    assert ia2.tags == ["a", "b"]


def test_image_annotation_tags_default_when_missing_in_old_file():
    # Simulate an older project's label file with no `tags` key.
    d = {"image_path": "x.jpg", "image_size": [10, 10]}
    ia = ImageAnnotation.from_dict(d)
    assert ia.tags == []


def test_save_annotation_keeps_record_for_tag_only(tmp_path):
    label_path = tmp_path / "x.json"
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10), tags=["important"],
    )
    save_annotation(ia, label_path)
    # File must exist — tags-only record is meaningful and must persist.
    assert label_path.exists()
    loaded = load_annotation(label_path)
    assert loaded is not None
    assert loaded.tags == ["important"]


def test_save_annotation_deletes_when_truly_empty(tmp_path):
    label_path = tmp_path / "x.json"
    label_path.write_text("{}")
    ia = ImageAnnotation(image_path="x.jpg", image_size=(10, 10))
    save_annotation(ia, label_path)
    assert not label_path.exists()
