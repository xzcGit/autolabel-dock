"""Tests for YOLO format import/export."""
import yaml

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.yolo import (
    export_yolo_detection,
    import_yolo_detection,
    export_yolo_pose,
    import_yolo_pose,
)


class TestYoloDetectionExport:
    def test_export_single_image(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6), confirmed=True),
                Annotation(class_name="car", class_id=1, bbox=(0.2, 0.3, 0.1, 0.2), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["person", "car"])
        txt_path = out_dir / "labels" / "img.txt"
        assert txt_path.exists()
        lines = txt_path.read_text().strip().split("\n")
        assert len(lines) == 2
        parts = lines[0].split()
        assert parts[0] == "0"  # class_id
        assert len(parts) == 5  # id cx cy w h

    def test_export_generates_data_yaml(self, tmp_path):
        ia = ImageAnnotation(
            image_path="a.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["person", "car"])
        yaml_path = out_dir / "data.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["nc"] == 1
        assert data["names"] == ["person"]

    def test_export_only_confirmed(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
            ],
        )
        out_dir = tmp_path / "output"
        export_yolo_detection([ia], out_dir, classes=["a", "b"], only_confirmed=True)
        lines = (out_dir / "labels" / "img.txt").read_text().strip().split("\n")
        assert len(lines) == 1

    def test_export_data_yaml_uses_only_actual_detection_classes(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.3), confirmed=True),
            ],
            image_tags=["outdoor"],
        )

        export_yolo_detection([ia], tmp_path / "out", classes=["cat", "dog", "outdoor"])

        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
        assert data["names"] == ["cat"]
        assert data["nc"] == 1

    def test_export_reindexes_label_ids_against_effective_classes(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="dog", class_id=1, bbox=(0.5, 0.5, 0.3, 0.3), confirmed=True),
            ],
        )

        export_yolo_detection([ia], tmp_path / "out", classes=["cat", "dog", "outdoor"])

        txt = (tmp_path / "out" / "labels" / "img.txt").read_text().strip()
        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())

        assert txt.split()[0] == "0"
        assert data["names"] == ["dog"]

    def test_export_only_confirmed_shrinks_names(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="dog", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False),
            ],
        )

        export_yolo_detection([ia], tmp_path / "out", classes=["cat", "dog"], only_confirmed=True)

        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
        assert data["names"] == ["cat"]

    def test_export_falls_back_to_class_id_when_class_name_missing(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
                Annotation(class_name="cat", class_id=1, bbox=(0.3, 0.3, 0.1, 0.1), confirmed=True),
            ],
        )

        export_yolo_detection([ia], tmp_path / "out", classes=["dog", "cat"])

        txt = (tmp_path / "out" / "labels" / "img.txt").read_text().strip().splitlines()
        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())

        assert txt[0].split()[0] == "0"
        assert txt[1].split()[0] == "1"
        assert data["names"] == ["dog", "cat"]
        assert data["nc"] == 2

    def test_export_uses_numeric_placeholder_when_class_name_missing_and_class_id_unmapped(self, tmp_path):
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="", class_id=5, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
            ],
        )

        export_yolo_detection([ia], tmp_path / "out", classes=["dog", "cat"])

        txt = (tmp_path / "out" / "labels" / "img.txt").read_text().strip()
        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())

        assert txt.split()[0] == "0"
        assert data["names"] == ["5"]
        assert data["nc"] == 1


class TestYoloDetectionImport:
    def test_import_single_file(self, tmp_path):
        # Create YOLO structure
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img.txt").write_text("0 0.5 0.4 0.3 0.6\n1 0.2 0.3 0.1 0.2\n")
        classes = ["person", "car"]
        results = import_yolo_detection(labels_dir, classes)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img"
        assert len(ia.annotations) == 2
        assert ia.annotations[0].class_name == "person"
        assert ia.annotations[0].bbox == (0.5, 0.4, 0.3, 0.6)
        assert ia.annotations[0].confirmed is True
        assert ia.annotations[0].source == "manual"

    def test_import_with_data_yaml(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(yaml.dump({"names": ["cat", "dog"], "nc": 2}))
        results = import_yolo_detection(labels_dir, classes=None, data_yaml=yaml_path)
        assert results[0].annotations[0].class_name == "cat"


class TestYoloPoseExport:
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
        out_dir = tmp_path / "output"
        export_yolo_pose([ia], out_dir, classes=["person"], kpt_dim=3)
        txt = (out_dir / "labels" / "pose.txt").read_text().strip()
        parts = txt.split()
        # class_id + 4 bbox + 2 keypoints * 3 dims = 11
        assert len(parts) == 11
        assert parts[0] == "0"

    def test_export_writes_pose_data_yaml_with_only_actual_classes(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[Keypoint(0.45, 0.3, 2, "nose")],
                    confirmed=True,
                ),
            ],
        )

        export_yolo_pose([ia], tmp_path / "out", classes=["person", "dog", "outdoor"], kpt_dim=3)

        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
        assert data["names"] == ["person"]
        assert data["nc"] == 1

    def test_export_reindexes_pose_ids_against_effective_classes(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="dog",
                    class_id=1,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[Keypoint(0.45, 0.3, 2, "nose")],
                    confirmed=True,
                ),
            ],
        )

        export_yolo_pose([ia], tmp_path / "out", classes=["person", "dog", "outdoor"], kpt_dim=3)

        txt = (tmp_path / "out" / "labels" / "pose.txt").read_text().strip()
        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())

        assert txt.split()[0] == "0"
        assert txt.split()[5:] == ["0.450000", "0.300000", "2"]
        assert data["names"] == ["dog"]
        assert data["nc"] == 1

    def test_export_pose_falls_back_to_class_id_when_class_name_missing(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="",
                    class_id=1,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[Keypoint(0.45, 0.3, 2, "nose")],
                    confirmed=True,
                ),
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.4, 0.4, 0.2, 0.2),
                    keypoints=[Keypoint(0.35, 0.25, 1, "nose")],
                    confirmed=True,
                ),
            ],
        )

        export_yolo_pose([ia], tmp_path / "out", classes=["person", "dog"], kpt_dim=3)

        txt_lines = (tmp_path / "out" / "labels" / "pose.txt").read_text().strip().splitlines()
        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())

        assert txt_lines[0].split()[0] == "1"
        assert txt_lines[1].split()[0] == "0"
        assert txt_lines[0].split()[5:] == ["0.450000", "0.300000", "2"]
        assert txt_lines[1].split()[5:] == ["0.350000", "0.250000", "1"]
        assert data["names"] == ["person", "dog"]
        assert data["nc"] == 2

    def test_export_pose_uses_numeric_placeholder_when_class_name_missing_and_class_id_unmapped(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name="",
                    class_id=5,
                    bbox=(0.5, 0.5, 0.3, 0.6),
                    keypoints=[Keypoint(0.45, 0.3, 2, "nose")],
                    confirmed=True,
                ),
            ],
        )

        export_yolo_pose([ia], tmp_path / "out", classes=["person", "dog"], kpt_dim=3)

        txt = (tmp_path / "out" / "labels" / "pose.txt").read_text().strip()
        data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())

        assert txt.split()[0] == "0"
        assert txt.split()[5:] == ["0.450000", "0.300000", "2"]
        assert data["names"] == ["5"]
        assert data["nc"] == 1


class TestYoloPoseImport:
    def test_import_pose(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
        (labels_dir / "p.txt").write_text("0 0.5 0.5 0.3 0.6 0.45 0.3 2 0.50 0.35 1\n")
        results = import_yolo_pose(
            labels_dir,
            classes=["person"],
            kpt_labels=["nose", "left_eye"],
            kpt_dim=3,
        )
        assert len(results) == 1
        ann = results[0].annotations[0]
        assert ann.bbox == (0.5, 0.5, 0.3, 0.6)
        assert len(ann.keypoints) == 2
        assert ann.keypoints[0].label == "nose"
        assert ann.keypoints[0].visible == 2
