"""Tests for label I/O (internal JSON format)."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.label_io import load_annotation, save_annotation


class TestSaveAnnotation:
    def test_save_creates_json_file(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6)),
            ],
            image_tags=["outdoor"],
        )
        save_annotation(ia, tmp_path / "img_001.json")
        assert (tmp_path / "img_001.json").exists()

    def test_saved_json_structure(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6)),
            ],
        )
        path = tmp_path / "img_001.json"
        save_annotation(ia, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["image_path"] == "img_001.jpg"
        assert data["image_size"] == [1920, 1080]
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["class_name"] == "person"

    def test_save_empty_unlinks_existing_file(self, tmp_path):
        path = tmp_path / "img.json"
        save_annotation(
            ImageAnnotation(
                image_path="img.jpg",
                image_size=(10, 10),
                annotations=[
                    Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2)),
                ],
            ),
            path,
        )
        assert path.exists()
        save_annotation(
            ImageAnnotation(image_path="img.jpg", image_size=(10, 10)),
            path,
        )
        assert not path.exists()

    def test_save_empty_does_not_create_file(self, tmp_path):
        path = tmp_path / "sub" / "img.json"
        save_annotation(
            ImageAnnotation(image_path="img.jpg", image_size=(10, 10)),
            path,
        )
        assert not path.exists()

    def test_save_with_only_image_tags_writes_file(self, tmp_path):
        path = tmp_path / "img.json"
        save_annotation(
            ImageAnnotation(
                image_path="img.jpg",
                image_size=(10, 10),
                image_tags=["dog"],
            ),
            path,
        )
        assert path.exists()


class TestLoadAnnotation:
    def test_load_roundtrip(self, tmp_path):
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="car",
                    class_id=1,
                    bbox=(0.6, 0.3, 0.25, 0.35),
                    keypoints=[Keypoint(0.5, 0.5, 2, "center")],
                    confidence=0.87,
                    confirmed=False,
                    source="auto",
                ),
            ],
            image_tags=["daytime"],
        )
        path = tmp_path / "test.json"
        save_annotation(ia, path)
        loaded = load_annotation(path)
        assert loaded.image_path == "test.jpg"
        assert loaded.annotations[0].confidence == 0.87
        assert loaded.annotations[0].confirmed is False
        assert loaded.annotations[0].keypoints[0].label == "center"
        assert loaded.image_tags == ["daytime"]

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_annotation(tmp_path / "missing.json")
        assert result is None

    def test_load_corrupted_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{", encoding="utf-8")
        result = load_annotation(path)
        assert result is None
