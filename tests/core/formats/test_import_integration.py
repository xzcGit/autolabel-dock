"""Integration tests for annotation import workflow."""
import json

import pytest
import yaml
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.label_io import save_annotation, load_annotation
from src.core.project import ProjectManager
from src.core.formats.yolo import import_yolo_detection, import_yolo_auto
from src.core.formats.coco import import_coco
from src.core.formats.labelme import import_labelme


def _create_project(tmp_path, classes=None, images=None):
    """Helper: create a minimal project with optional image files."""
    if classes is None:
        classes = ["person", "car"]
    pm = ProjectManager.create(tmp_path / "proj", "test", classes=classes)
    img_dir = pm.project_dir / pm.config.image_dir
    for name in (images or ["img_001.jpg", "img_002.jpg"]):
        (img_dir / name).write_bytes(b"\xff\xd8")  # minimal JPEG header
    return pm


class TestYoloImportIntegration:
    def test_import_matches_by_stem(self, tmp_path):
        pm = _create_project(tmp_path)
        labels_dir = tmp_path / "yolo_labels"
        labels_dir.mkdir()
        (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        (labels_dir / "img_002.txt").write_text("1 0.2 0.2 0.1 0.1\n")
        (labels_dir / "img_999.txt").write_text("0 0.1 0.1 0.1 0.1\n")  # no match

        imported = import_yolo_detection(labels_dir, pm.config.classes)
        assert len(imported) == 3  # all parsed

        # Match by stem
        by_stem = {Path(ia.image_path).stem: ia for ia in imported}
        assert "img_001" in by_stem
        assert "img_002" in by_stem
        assert "img_999" in by_stem

    def test_import_and_save_to_project(self, tmp_path):
        pm = _create_project(tmp_path)
        labels_dir = tmp_path / "yolo_labels"
        labels_dir.mkdir()
        (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        imported = import_yolo_detection(labels_dir, pm.config.classes)
        ia = imported[0]
        # Write to project label path
        label_path = pm.label_path_for(pm.list_images()[0])
        ia.image_path = pm.list_images()[0].name
        ia.image_size = (640, 480)
        save_annotation(ia, label_path)

        loaded = load_annotation(label_path)
        assert loaded is not None
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].class_name == "person"

    def test_yolo_with_data_yaml(self, tmp_path):
        labels_dir = tmp_path / "yolo"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
        data_yaml = labels_dir / "data.yaml"
        data_yaml.write_text(yaml.dump({"names": ["cat", "dog"], "nc": 2}))

        imported = import_yolo_detection(labels_dir, data_yaml=data_yaml)
        assert len(imported) == 1
        assert imported[0].annotations[0].class_name == "cat"
        assert imported[0].annotations[1].class_name == "dog"


class TestCocoImportIntegration:
    def test_import_coco_basic(self, tmp_path):
        coco = {
            "images": [
                {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }
        coco_path = tmp_path / "coco.json"
        coco_path.write_text(json.dumps(coco))

        imported = import_coco(coco_path)
        assert len(imported) == 1
        ia = imported[0]
        assert ia.image_path == "img_001.jpg"
        assert len(ia.annotations) == 1
        ann = ia.annotations[0]
        assert ann.class_name == "person"
        assert ann.bbox is not None
        # Verify center format: (100+200/2)/640, (100+150/2)/480
        cx, cy, w, h = ann.bbox
        assert abs(cx - (100 + 100) / 640) < 0.01
        assert abs(w - 200 / 640) < 0.01


class TestLabelmeImportIntegration:
    def test_import_labelme_rectangles(self, tmp_path):
        labelme_dir = tmp_path / "labelme"
        labelme_dir.mkdir()
        data = {
            "version": "5.0.0",
            "shapes": [
                {
                    "label": "person",
                    "shape_type": "rectangle",
                    "points": [[100, 50], [300, 250]],
                    "flags": {},
                },
            ],
            "imagePath": "img_001.jpg",
            "imageData": None,
            "imageWidth": 640,
            "imageHeight": 480,
        }
        (labelme_dir / "img_001.json").write_text(json.dumps(data))

        imported = import_labelme(labelme_dir)
        assert len(imported) == 1
        ia = imported[0]
        assert ia.image_path == "img_001.jpg"
        ann = ia.annotations[0]
        assert ann.class_name == "person"
        cx, cy, w, h = ann.bbox
        assert abs(cx - 200 / 640) < 0.01
        assert abs(cy - 150 / 480) < 0.01

    def test_pose_labelme_import_does_not_pollute_project_classes(self, tmp_path):
        """Regression test: importing pose-labeled JSON must not add keypoint
        names (nose, left_eye, ...) into the project's class list, because the
        controller auto-adds any annotation.class_name it doesn't recognize.
        """
        from src.core.formats.labelme import export_labelme
        pm = _create_project(tmp_path, classes=["person"])
        original = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person", class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[
                        Keypoint(0.45, 0.30, 2, "nose"),
                        Keypoint(0.50, 0.35, 1, "left_eye"),
                    ],
                ),
            ],
        )
        labelme_dir = tmp_path / "exported"
        export_labelme([original], labelme_dir)

        imported = import_labelme(labelme_dir)
        assert len(imported) == 1
        class_names_in_anns = {a.class_name for a in imported[0].annotations}
        # The only annotation class_name should be the bbox class — keypoint names
        # must live inside Annotation.keypoints[*].label, not as new annotations.
        assert class_names_in_anns == {"person"}
        assert len(imported[0].annotations) == 1
        ann = imported[0].annotations[0]
        assert ann.bbox is not None
        assert len(ann.keypoints) == 2


class TestConflictModes:
    """Test the three conflict resolution strategies."""

    def test_skip_existing(self, tmp_path):
        pm = _create_project(tmp_path)
        # Create existing annotation
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        existing = ImageAnnotation(
            image_path=img.name,
            image_size=(640, 480),
            annotations=[Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2))],
        )
        save_annotation(existing, label_path)

        # With skip mode, existing should be untouched
        loaded = load_annotation(label_path)
        assert loaded is not None
        assert len(loaded.annotations) == 1
        assert loaded.annotations[0].class_name == "person"

    def test_merge_appends(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        existing = ImageAnnotation(
            image_path=img.name,
            image_size=(640, 480),
            annotations=[Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2))],
        )
        save_annotation(existing, label_path)

        # Simulate merge
        loaded = load_annotation(label_path)
        new_ann = Annotation(class_name="car", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1))
        loaded.annotations.append(new_ann)
        save_annotation(loaded, label_path)

        final = load_annotation(label_path)
        assert len(final.annotations) == 2
        names = {a.class_name for a in final.annotations}
        assert names == {"person", "car"}

    def test_overwrite_replaces(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        existing = ImageAnnotation(
            image_path=img.name,
            image_size=(640, 480),
            annotations=[Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2))],
        )
        save_annotation(existing, label_path)

        # Simulate overwrite
        new_ia = ImageAnnotation(
            image_path=img.name,
            image_size=(640, 480),
            annotations=[Annotation(class_name="car", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1))],
        )
        save_annotation(new_ia, label_path)

        final = load_annotation(label_path)
        assert len(final.annotations) == 1
        assert final.annotations[0].class_name == "car"


class TestNewClassAutoAdd:
    def test_new_classes_detected(self, tmp_path):
        pm = _create_project(tmp_path, classes=["person"])
        labels_dir = tmp_path / "yolo"
        labels_dir.mkdir()
        # class_id 0 = person (existing), class_id 1 = bicycle (new in data.yaml)
        (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
        data_yaml = labels_dir / "data.yaml"
        data_yaml.write_text(yaml.dump({"names": ["person", "bicycle"], "nc": 2}))

        imported = import_yolo_detection(labels_dir, data_yaml=data_yaml)
        # Check that the imported annotations contain the new class name
        class_names = {a.class_name for ia in imported for a in ia.annotations}
        assert "bicycle" in class_names

        # Simulate what the controller does: auto-add new classes
        existing_set = set(pm.config.classes)
        new_classes = []
        for ia in imported:
            for ann in ia.annotations:
                if ann.class_name not in existing_set:
                    existing_set.add(ann.class_name)
                    new_classes.append(ann.class_name)
        pm.config.classes.extend(new_classes)
        pm.save()

        reopened = ProjectManager.open(pm.project_dir)
        assert "bicycle" in reopened.config.classes


class TestImportRegistryIntegration:
    def test_import_registry_has_all_formats(self):
        from src.core.formats import get_import_registry
        registry = get_import_registry()
        names = registry.list_names()
        assert "YOLO" in names
        assert "COCO" in names
        assert "labelme" in names

    def test_coco_is_file_input(self):
        from src.core.formats import get_import_registry
        registry = get_import_registry()
        info = registry.get("COCO")
        assert info.input_is_file is True

    def test_yolo_is_dir_input(self):
        from src.core.formats import get_import_registry
        registry = get_import_registry()
        info = registry.get("YOLO")
        assert info.input_is_file is False


class TestYoloAutoImport:
    """Tests for import_yolo_auto: auto-detection, data.yaml search, fallback."""

    def test_auto_detect_detection(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        results = import_yolo_auto(labels_dir, classes=["person"])
        assert len(results) == 1
        assert len(results[0].annotations[0].keypoints) == 0

    def test_auto_detect_pose(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # 5 bbox fields + 2 keypoints * 3 dims = 11 fields
        (labels_dir / "img.txt").write_text(
            "0 0.5 0.5 0.3 0.6 0.45 0.3 2 0.50 0.35 1\n"
        )
        results = import_yolo_auto(labels_dir, classes=["person"])
        assert len(results) == 1
        ann = results[0].annotations[0]
        assert len(ann.keypoints) == 2
        assert ann.keypoints[0].label == "kp_0"
        assert ann.keypoints[1].label == "kp_1"
        assert ann.keypoints[0].visible == 2

    def test_data_yaml_in_parent_dir(self, tmp_path):
        """data.yaml in parent directory should be found automatically."""
        labels_dir = tmp_path / "dataset" / "labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
        data_yaml = tmp_path / "dataset" / "data.yaml"
        data_yaml.write_text(yaml.dump({"names": ["cat", "dog"], "nc": 2}))

        results = import_yolo_auto(labels_dir)
        assert results[0].annotations[0].class_name == "cat"
        assert results[0].annotations[1].class_name == "dog"

    def test_no_data_yaml_no_classes_uses_numeric(self, tmp_path):
        """Without data.yaml or classes, should use numeric class names."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.1 0.1\n")

        results = import_yolo_auto(labels_dir)
        assert len(results) == 1
        names = {a.class_name for a in results[0].annotations}
        assert names == {"0", "2"}

    def test_auto_pose_with_kpt_labels(self, tmp_path):
        """Providing kpt_labels should use them instead of generic names."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "p.txt").write_text(
            "0 0.5 0.5 0.3 0.6 0.45 0.3 2 0.50 0.35 1\n"
        )
        results = import_yolo_auto(
            labels_dir, classes=["person"],
            kpt_labels=["nose", "left_eye"],
        )
        ann = results[0].annotations[0]
        assert ann.keypoints[0].label == "nose"
        assert ann.keypoints[1].label == "left_eye"

    def test_empty_dir(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        results = import_yolo_auto(labels_dir, classes=["a"])
        assert results == []

    def test_classes_txt_is_ignored_as_label_file_and_used_for_names(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
        (labels_dir / "img.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        results = import_yolo_auto(labels_dir)

        assert len(results) == 1
        assert results[0].image_path == "img"
        assert results[0].annotations[0].class_name == "dog"

    def test_invalid_yolo_line_reports_file_and_line(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "bad.txt").write_text("oops\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"bad\.txt:1"):
            import_yolo_auto(labels_dir, classes=["cat"])
