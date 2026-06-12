"""Tests for labelme format import/export."""
import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.formats.labelme import export_labelme, import_labelme


class TestLabelmeExport:
    def test_export_bbox(self, tmp_path):
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="person", class_id=0, bbox=(0.5, 0.5, 0.5, 0.5), confirmed=True),
            ],
        )
        out_dir = tmp_path / "output"
        export_labelme([ia], out_dir)
        json_path = out_dir / "test.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["imagePath"] == "test.jpg"
        assert data["imageWidth"] == 640
        assert data["imageHeight"] == 480
        assert len(data["shapes"]) == 1
        shape = data["shapes"][0]
        assert shape["label"] == "person"
        assert shape["shape_type"] == "rectangle"
        # labelme rectangle: [[x1, y1], [x2, y2]] in pixels
        assert len(shape["points"]) == 2

    def test_export_keypoints(self, tmp_path):
        ia = ImageAnnotation(
            image_path="kp.jpg",
            image_size=(100, 100),
            annotations=[
                Annotation(
                    class_name="point",
                    class_id=0,
                    bbox=None,
                    keypoints=[Keypoint(0.5, 0.3, 2, "nose")],
                    confirmed=True,
                ),
            ],
        )
        out_dir = tmp_path / "output"
        export_labelme([ia], out_dir)
        data = json.loads((out_dir / "kp.json").read_text())
        assert len(data["shapes"]) == 1
        assert data["shapes"][0]["shape_type"] == "point"
        assert data["shapes"][0]["label"] == "nose"


class TestLabelmeImport:
    def test_import_rectangle(self, tmp_path):
        labelme_data = {
            "imagePath": "img.jpg",
            "imageWidth": 640,
            "imageHeight": 480,
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[160, 120], [480, 360]],
                    "flags": {},
                },
            ],
        }
        json_path = tmp_path / "img.json"
        json_path.write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        assert len(results) == 1
        ia = results[0]
        assert ia.image_path == "img.jpg"
        assert ia.image_size == (640, 480)
        ann = ia.annotations[0]
        assert ann.class_name == "car"
        # Check normalized center bbox
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01
        assert abs(w - 0.5) < 0.01
        assert abs(h - 0.5) < 0.01

    def test_import_point(self, tmp_path):
        labelme_data = {
            "imagePath": "kp.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[50, 30]],
                    "flags": {},
                },
            ],
        }
        (tmp_path / "kp.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        ia = results[0]
        assert len(ia.annotations) == 1
        ann = ia.annotations[0]
        assert ann.bbox is None
        assert len(ann.keypoints) == 1
        assert ann.keypoints[0].label == "nose"
        assert abs(ann.keypoints[0].x - 0.5) < 0.01


class TestLabelmePoseRoundtrip:
    """Pose annotations (bbox + keypoints) must survive export -> import."""

    def test_export_pose_uses_group_id_to_link_bbox_and_keypoints(self, tmp_path):
        ia = ImageAnnotation(
            image_path="pose.jpg",
            image_size=(200, 100),
            annotations=[
                Annotation(
                    class_name="person",
                    class_id=0,
                    bbox=(0.5, 0.5, 0.4, 0.4),
                    keypoints=[
                        Keypoint(0.5, 0.3, 2, "nose"),
                        Keypoint(0.45, 0.35, 2, "left_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        out_dir = tmp_path / "out"
        export_labelme([ia], out_dir)
        data = json.loads((out_dir / "pose.json").read_text())

        # Bbox shape and both keypoint shapes share a single group_id.
        rect_shapes = [s for s in data["shapes"] if s["shape_type"] == "rectangle"]
        point_shapes = [s for s in data["shapes"] if s["shape_type"] == "point"]
        assert len(rect_shapes) == 1
        assert len(point_shapes) == 2

        gid = rect_shapes[0].get("group_id")
        assert gid is not None, "bbox shape must carry a group_id when keypoints exist"
        for s in point_shapes:
            assert s.get("group_id") == gid

    def test_import_pose_reconstructs_bbox_and_keypoints(self, tmp_path):
        labelme_data = {
            "imagePath": "pose.jpg",
            "imageWidth": 200,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "person",
                    "shape_type": "rectangle",
                    "points": [[60, 30], [140, 70]],
                    "group_id": 1,
                    "flags": {},
                },
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[100, 30]],
                    "group_id": 1,
                    "flags": {},
                },
                {
                    "label": "left_eye",
                    "shape_type": "point",
                    "points": [[90, 35]],
                    "group_id": 1,
                    "flags": {},
                },
            ],
        }
        (tmp_path / "pose.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        assert len(results) == 1
        ia = results[0]
        assert len(ia.annotations) == 1, "grouped shapes must merge into a single Annotation"
        ann = ia.annotations[0]
        assert ann.class_name == "person"
        assert ann.bbox is not None
        assert len(ann.keypoints) == 2
        labels = [kp.label for kp in ann.keypoints]
        assert labels == ["nose", "left_eye"]

    def test_pose_export_import_roundtrip(self, tmp_path):
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
                        Keypoint(0.55, 0.35, 2, "right_eye"),
                    ],
                    confirmed=True,
                ),
            ],
        )
        out_dir = tmp_path / "rt"
        export_labelme([original], out_dir)
        results = import_labelme(out_dir)
        assert len(results) == 1
        ann = results[0].annotations[0]
        # bbox preserved
        assert ann.class_name == "person"
        assert ann.bbox is not None
        cx, cy, w, h = ann.bbox
        assert abs(cx - 0.5) < 1e-3
        assert abs(cy - 0.5) < 1e-3
        # keypoints preserved with labels
        assert len(ann.keypoints) == 3
        assert [kp.label for kp in ann.keypoints] == ["nose", "left_eye", "right_eye"]

    def test_import_legacy_shapes_without_group_id_keep_per_shape_behavior(self, tmp_path):
        """Shapes without group_id with MULTIPLE rectangles still keep per-shape
        behavior — the auto-merge heuristic only triggers on single-instance pose.
        """
        labelme_data = {
            "imagePath": "legacy.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [50, 50]],
                    "flags": {},
                },
                {
                    "label": "person",
                    "shape_type": "rectangle",
                    "points": [[60, 10], [90, 50]],
                    "flags": {},
                },
                {
                    "label": "nose",
                    "shape_type": "point",
                    "points": [[60, 30]],
                    "flags": {},
                },
            ],
        }
        (tmp_path / "legacy.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        ia = results[0]
        # 2 rectangles -> multi-instance -> per-shape
        assert len(ia.annotations) == 3
        names = {a.class_name for a in ia.annotations}
        assert names == {"car", "person", "nose"}

    def test_single_rectangle_no_points_still_per_shape(self, tmp_path):
        """A pure detection JSON (no points) should not change behavior."""
        labelme_data = {
            "imagePath": "det.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [50, 50]],
                    "flags": {},
                },
            ],
        }
        (tmp_path / "det.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        ia = results[0]
        assert len(ia.annotations) == 1
        assert ia.annotations[0].class_name == "car"
        assert ia.annotations[0].bbox is not None
        assert ia.annotations[0].keypoints == []

    def test_auto_merge_single_instance_pose_with_null_group_id(self, tmp_path):
        """1 rectangle + N points, all ungrouped, should auto-merge into a
        single pose Annotation. This matches labelme files exported by external
        tools that don't set group_id (common single-instance-per-image case).
        """
        labelme_data = {
            "imagePath": "pose.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {"label": "0", "shape_type": "rectangle",
                 "points": [[10, 10], [90, 90]], "group_id": None, "flags": {}},
                {"label": "nose", "shape_type": "point",
                 "points": [[50, 30]], "group_id": None, "flags": {}},
                {"label": "left_eye", "shape_type": "point",
                 "points": [[40, 40]], "group_id": None, "flags": {}},
                {"label": "right_eye", "shape_type": "point",
                 "points": [[60, 40]], "group_id": None, "flags": {}},
            ],
        }
        (tmp_path / "pose.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        ia = results[0]
        assert len(ia.annotations) == 1, "single bbox + points should auto-merge"
        ann = ia.annotations[0]
        assert ann.class_name == "0"
        assert ann.bbox is not None
        assert [kp.label for kp in ann.keypoints] == ["nose", "left_eye", "right_eye"]


class TestLabelmeImagePathNormalization:
    """imagePath written by Windows labelme uses backslashes and a relative
    prefix like '..\\images\\foo.jpg'. The controller matches images by
    ``Path(imagePath).stem`` and skips when it can't find a match. The importer
    must therefore strip directory components and normalize separators so the
    extracted stem is just the image basename without extension.
    """

    def test_imagepath_with_windows_backslashes_is_normalized(self, tmp_path):
        labelme_data = {
            "imagePath": "..\\images\\20251215080121350939737_crop0_cls0.jpg",
            "imageWidth": 401,
            "imageHeight": 395,
            "shapes": [
                {
                    "label": "0",
                    "shape_type": "rectangle",
                    "points": [[30, 29], [367, 369]],
                    "flags": {},
                },
            ],
        }
        (tmp_path / "20251215080121350939737_crop0_cls0.json").write_text(
            json.dumps(labelme_data)
        )
        results = import_labelme(tmp_path)
        from pathlib import Path as _Path
        assert _Path(results[0].image_path).stem == "20251215080121350939737_crop0_cls0"

    def test_imagepath_with_posix_relative_prefix_is_normalized(self, tmp_path):
        labelme_data = {
            "imagePath": "../images/foo.jpg",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {"label": "x", "shape_type": "rectangle", "points": [[10, 10], [50, 50]], "flags": {}},
            ],
        }
        (tmp_path / "foo.json").write_text(json.dumps(labelme_data))
        results = import_labelme(tmp_path)
        from pathlib import Path as _Path
        assert _Path(results[0].image_path).stem == "foo"
