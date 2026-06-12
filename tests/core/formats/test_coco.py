"""Tests for COCO format import/export."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.coco import export_coco, import_coco


class TestCocoExport:
    def test_export_detection_bbox_values(self, tmp_path):
        """Verify exact pixel conversion: normalized center -> COCO pixel top-left."""
        ia = ImageAnnotation(
            image_path="v.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.5, 0.5), confirmed=True),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person"])
        data = json.loads(out_path.read_text())
        bbox = data["annotations"][0]["bbox"]
        # cx=0.5, cy=0.5, w=0.5, h=0.5 on 640x480 -> x_tl=160, y_tl=120, pw=320, ph=240
        assert bbox[0] == 160.0
        assert bbox[1] == 120.0
        assert bbox[2] == 320.0
        assert bbox[3] == 240.0

    def test_export_detection(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6), confirmed=True),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person", "car"])
        data = json.loads(out_path.read_text())
        assert len(data["images"]) == 1
        assert data["images"][0]["file_name"] == "img_001.jpg"
        assert data["images"][0]["width"] == 1920
        assert len(data["annotations"]) == 1
        ann = data["annotations"][0]
        # COCO bbox is [x_top_left, y_top_left, width, height] in pixels
        assert ann["category_id"] == 1  # COCO is 1-indexed
        assert len(ann["bbox"]) == 4
        assert len(data["categories"]) == 2

    def test_export_with_keypoints(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[
                        Keypoint(0.45, 0.3, 2, "nose"),
                        Keypoint(0.50, 0.35, 1, "left_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person"])
        data = json.loads(out_path.read_text())
        ann = data["annotations"][0]
        assert "keypoints" in ann
        assert ann["num_keypoints"] == 2
        # COCO keypoints: [x, y, v, x, y, v, ...]
        assert len(ann["keypoints"]) == 6

    def test_export_only_confirmed(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="a", class_id=0, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["a"], only_confirmed=True)
        data = json.loads(out_path.read_text())
        assert len(data["annotations"]) == 1


class TestCocoImport:
    def test_import_detection(self, tmp_path):
        coco_data = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [192, 96, 192, 288],  # x,y,w,h in pixels
                    "area": 55296,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }
        coco_path = tmp_path / "coco.json"
        coco_path.write_text(json.dumps(coco_data))
        results = import_coco(coco_path)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img.jpg"
        assert ia.image_size == (640, 480)
        assert len(ia.annotations) == 1
        ann = ia.annotations[0]
        assert ann.class_name == "person"
        # Verify normalized bbox (center format)
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.45) < 0.01
        assert abs(cy - 0.5) < 0.01
        assert abs(w - 0.3) < 0.01
        assert abs(h - 0.6) < 0.01
        assert ann.confirmed is True

    def test_import_with_classes_resolves_class_id(self, tmp_path):
        coco_data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            ],
            "categories": [{"id": 5, "name": "dog"}],
        }
        coco_path = tmp_path / "c.json"
        coco_path.write_text(json.dumps(coco_data))
        results = import_coco(coco_path, classes=["cat", "dog", "bird"])
        ann = results[0].annotations[0]
        assert ann.class_name == "dog"
        assert ann.class_id == 1  # index in provided classes list


class TestCocoKeypointNames:
    """Keypoint label names must survive export/import via category metadata."""

    def test_export_writes_keypoints_into_category(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[
                        Keypoint(0.45, 0.30, 2, "nose"),
                        Keypoint(0.50, 0.35, 1, "left_eye"),
                        Keypoint(0.55, 0.35, 2, "right_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        out_path = tmp_path / "coco.json"
        export_coco([ia], out_path, classes=["person"])
        data = json.loads(out_path.read_text())
        cat = next(c for c in data["categories"] if c["name"] == "person")
        assert "keypoints" in cat
        assert cat["keypoints"] == ["nose", "left_eye", "right_eye"]

    def test_import_reads_keypoint_names_from_category(self, tmp_path):
        coco_data = {
            "images": [{"id": 1, "file_name": "pose.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1, "image_id": 1, "category_id": 1,
                    "bbox": [25, 25, 50, 50], "area": 2500, "iscrowd": 0,
                    "keypoints": [50, 30, 2, 45, 35, 1, 55, 35, 2],
                    "num_keypoints": 3,
                },
            ],
            "categories": [
                {
                    "id": 1, "name": "person",
                    "keypoints": ["nose", "left_eye", "right_eye"],
                },
            ],
        }
        coco_path = tmp_path / "p.json"
        coco_path.write_text(json.dumps(coco_data))
        results = import_coco(coco_path)
        ann = results[0].annotations[0]
        assert len(ann.keypoints) == 3
        assert [kp.label for kp in ann.keypoints] == ["nose", "left_eye", "right_eye"]
        assert ann.keypoints[1].visible == 1

    def test_import_falls_back_to_generic_names_when_category_lacks_keypoints(self, tmp_path):
        coco_data = {
            "images": [{"id": 1, "file_name": "p.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1, "image_id": 1, "category_id": 1,
                    "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0,
                    "keypoints": [20, 20, 2, 30, 30, 2],
                    "num_keypoints": 2,
                },
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        coco_path = tmp_path / "p.json"
        coco_path.write_text(json.dumps(coco_data))
        results = import_coco(coco_path)
        ann = results[0].annotations[0]
        assert [kp.label for kp in ann.keypoints] == ["kp_0", "kp_1"]

    def test_pose_export_import_roundtrip_preserves_keypoint_names(self, tmp_path):
        original = ImageAnnotation(
            image_path="rt.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[
                        Keypoint(0.45, 0.30, 2, "nose"),
                        Keypoint(0.50, 0.35, 1, "left_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        out_path = tmp_path / "rt.json"
        export_coco([original], out_path, classes=["person"])
        results = import_coco(out_path)
        ann = results[0].annotations[0]
        assert ann.class_name == "person"
        assert [kp.label for kp in ann.keypoints] == ["nose", "left_eye"]
