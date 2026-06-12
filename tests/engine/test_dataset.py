"""Tests for dataset preparation (train/val split + data.yaml generation)."""
import json
import yaml
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.engine.dataset import DatasetPreparer


def _make_project_with_images(tmp_path, num_images=10, classes=None):
    """Helper: create a project with fake images and annotations."""
    classes = classes or ["cat", "dog"]
    pm = ProjectManager.create(
        project_dir=tmp_path / "proj",
        name="test",
        image_dir="images",
        classes=classes,
    )
    img_dir = tmp_path / "proj" / "images"
    for i in range(num_images):
        (img_dir / f"img_{i:03d}.jpg").write_bytes(b"fake")
        cls_name = classes[i % len(classes)]
        cls_id = classes.index(cls_name)
        ia = ImageAnnotation(
            image_path=f"img_{i:03d}.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(
                    class_name=cls_name,
                    class_id=cls_id,
                    bbox=(0.5, 0.5, 0.3, 0.4),
                    confirmed=True,
                ),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / f"img_{i:03d}.jpg"))
    return pm


class TestDatasetPreparer:
    def test_prepare_detection_creates_structure(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=10)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        data_yaml = preparer.prepare(output_dir, task="detect", val_ratio=0.2)

        assert data_yaml.exists()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()
        assert (output_dir / "val" / "images").is_dir()
        assert (output_dir / "val" / "labels").is_dir()

    def test_prepare_detection_data_yaml(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=10)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        data_yaml = preparer.prepare(output_dir, task="detect", val_ratio=0.2)

        data = yaml.safe_load(data_yaml.read_text())
        assert data["nc"] == 2
        assert data["names"] == ["cat", "dog"]
        assert "train" in data
        assert "val" in data

    def test_prepare_detection_data_yaml_uses_only_actual_detection_classes(self, tmp_path):
        pm = _make_project_with_images(
            tmp_path,
            num_images=4,
            classes=["cat", "dog", "outdoor"],
        )
        img_dir = tmp_path / "proj" / "images"
        for i in range(4):
            ia = ImageAnnotation(
                image_path=f"img_{i:03d}.jpg",
                image_size=(640, 480),
                annotations=[
                    Annotation(
                        class_name="dog",
                        class_id=1,
                        bbox=(0.5, 0.5, 0.3, 0.4),
                        confirmed=True,
                    ),
                    Annotation(
                        class_name="outdoor",
                        class_id=2,
                        bbox=None,
                        confirmed=True,
                    ),
                ],
            )
            save_annotation(ia, pm.label_path_for(img_dir / f"img_{i:03d}.jpg"))

        data_yaml = DatasetPreparer(pm).prepare(tmp_path / "dataset", task="detect", val_ratio=0.0)
        data = yaml.safe_load(data_yaml.read_text())

        assert data["names"] == ["dog"]
        assert data["nc"] == 1

    def test_prepare_detection_reindexes_txt_ids_against_effective_classes(self, tmp_path):
        pm = _make_project_with_images(
            tmp_path,
            num_images=2,
            classes=["cat", "dog", "outdoor"],
        )
        img_dir = tmp_path / "proj" / "images"
        for i in range(2):
            ia = ImageAnnotation(
                image_path=f"img_{i:03d}.jpg",
                image_size=(640, 480),
                annotations=[
                    Annotation(
                        class_name="dog",
                        class_id=1,
                        bbox=(0.5, 0.5, 0.3, 0.4),
                        confirmed=True,
                    ),
                ],
            )
            save_annotation(ia, pm.label_path_for(img_dir / f"img_{i:03d}.jpg"))

        output_dir = tmp_path / "dataset"
        DatasetPreparer(pm).prepare(output_dir, task="detect", val_ratio=0.0)

        first_line = (output_dir / "train" / "labels" / "img_000.txt").read_text().strip()
        assert first_line.split()[0] == "0"

    def test_prepare_detection_data_yaml_keeps_val_only_bbox_class(self, tmp_path):
        pm = _make_project_with_images(
            tmp_path,
            num_images=2,
            classes=["cat", "dog", "outdoor"],
        )
        img_dir = tmp_path / "proj" / "images"
        save_annotation(
            ImageAnnotation(
                image_path="img_000.jpg",
                image_size=(640, 480),
                annotations=[
                    Annotation(
                        class_name="cat",
                        class_id=0,
                        bbox=(0.5, 0.5, 0.3, 0.4),
                        confirmed=True,
                    ),
                ],
            ),
            pm.label_path_for(img_dir / "img_000.jpg"),
        )
        save_annotation(
            ImageAnnotation(
                image_path="img_001.jpg",
                image_size=(640, 480),
                annotations=[
                    Annotation(
                        class_name="cat",
                        class_id=0,
                        bbox=(0.5, 0.5, 0.3, 0.4),
                        confirmed=True,
                    ),
                    Annotation(
                        class_name="dog",
                        class_id=1,
                        bbox=(0.4, 0.4, 0.2, 0.2),
                        confirmed=True,
                    ),
                ],
            ),
            pm.label_path_for(img_dir / "img_001.jpg"),
        )

        output_dir = tmp_path / "dataset"
        data_yaml = DatasetPreparer(pm).prepare(output_dir, task="detect", val_ratio=0.5, seed=42)

        data = yaml.safe_load(data_yaml.read_text())
        assert data["names"] == ["cat", "dog"]
        assert data["nc"] == 2

        train_first_line = (output_dir / "train" / "labels" / "img_000.txt").read_text().strip()
        assert train_first_line.split()[0] == "0"

        val_lines = (output_dir / "val" / "labels" / "img_001.txt").read_text().strip().splitlines()
        assert [line.split()[0] for line in val_lines] == ["0", "1"]

    def test_prepare_detection_falls_back_to_class_id_when_class_name_is_blank(self, tmp_path):
        pm = _make_project_with_images(
            tmp_path,
            num_images=1,
            classes=["cat", "dog"],
        )
        img_dir = tmp_path / "proj" / "images"
        save_annotation(
            ImageAnnotation(
                image_path="img_000.jpg",
                image_size=(640, 480),
                annotations=[
                    Annotation(
                        class_name="",
                        class_id=1,
                        bbox=(0.5, 0.5, 0.3, 0.4),
                        confirmed=True,
                    ),
                ],
            ),
            pm.label_path_for(img_dir / "img_000.jpg"),
        )

        output_dir = tmp_path / "dataset"
        data_yaml = DatasetPreparer(pm).prepare(output_dir, task="detect", val_ratio=0.0)

        data = yaml.safe_load(data_yaml.read_text())
        assert data["names"] == ["dog"]
        assert data["nc"] == 1

        first_line = (output_dir / "train" / "labels" / "img_000.txt").read_text().strip()
        assert first_line.split()[0] == "0"

    def test_prepare_detection_uses_numeric_placeholder_when_class_name_is_blank_and_class_id_unmapped(self, tmp_path):
        pm = _make_project_with_images(
            tmp_path,
            num_images=1,
            classes=["cat", "dog"],
        )
        img_dir = tmp_path / "proj" / "images"
        save_annotation(
            ImageAnnotation(
                image_path="img_000.jpg",
                image_size=(640, 480),
                annotations=[
                    Annotation(
                        class_name="",
                        class_id=5,
                        bbox=(0.5, 0.5, 0.3, 0.4),
                        confirmed=True,
                    ),
                ],
            ),
            pm.label_path_for(img_dir / "img_000.jpg"),
        )

        output_dir = tmp_path / "dataset"
        data_yaml = DatasetPreparer(pm).prepare(output_dir, task="detect", val_ratio=0.0)

        data = yaml.safe_load(data_yaml.read_text())
        assert data["names"] == ["5"]
        assert data["nc"] == 1

        first_line = (output_dir / "train" / "labels" / "img_000.txt").read_text().strip()
        assert first_line.split()[0] == "0"

    def test_train_val_split_ratio(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=20)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.2)

        train_imgs = list((output_dir / "train" / "images").iterdir())
        val_imgs = list((output_dir / "val" / "images").iterdir())
        total = len(train_imgs) + len(val_imgs)
        assert total == 20
        assert len(val_imgs) >= 2
        assert len(val_imgs) <= 6

    def test_uses_symlinks(self, tmp_path):
        pm = _make_project_with_images(tmp_path, num_images=4)
        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.5)

        for img_path in (output_dir / "train" / "images").iterdir():
            assert img_path.is_symlink()

    def test_only_confirmed_annotations(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "img.jpg").write_bytes(b"fake")
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
                Annotation(class_name="a", class_id=0, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.7),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "img.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.0)

        label_files = list((output_dir / "train" / "labels").glob("*.txt"))
        assert len(label_files) == 1
        lines = label_files[0].read_text().strip().split("\n")
        assert len(lines) == 1

    def test_prepare_only_confirmed_annotations_shrink_names(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["a", "b"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "img.jpg").write_bytes(b"fake")
        ia = ImageAnnotation(
            image_path="img.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
                Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "img.jpg"))

        output_dir = tmp_path / "dataset"
        DatasetPreparer(pm).prepare(output_dir, task="detect", val_ratio=0.0)

        data = yaml.safe_load((output_dir / "data.yaml").read_text())
        assert data["names"] == ["a"]

    def test_skip_unlabeled_images(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "labeled.jpg").write_bytes(b"fake")
        (img_dir / "unlabeled.jpg").write_bytes(b"fake")
        ia = ImageAnnotation(
            image_path="labeled.jpg",
            image_size=(640, 480),
            annotations=[
                Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=True),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "labeled.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="detect", val_ratio=0.0)

        train_imgs = list((output_dir / "train" / "images").iterdir())
        assert len(train_imgs) == 1

    def test_prepare_pose_includes_keypoints(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["person"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "pose.jpg").write_bytes(b"fake")
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
        save_annotation(ia, pm.label_path_for(img_dir / "pose.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        data_yaml = preparer.prepare(output_dir, task="pose", val_ratio=0.0, kpt_shape=[2, 3])

        label_files = list((output_dir / "train" / "labels").glob("*.txt"))
        assert len(label_files) == 1
        parts = label_files[0].read_text().strip().split()
        # class_id + 4 bbox + 2 kpts * 3 dims = 11
        assert len(parts) == 11

        data = yaml.safe_load(data_yaml.read_text())
        assert data["kpt_shape"] == [2, 3]

    def test_prepare_pose_keeps_kpt_shape_and_effective_names(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=["person", "background"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "pose.jpg").write_bytes(b"fake")
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
                Annotation(
                    class_name="background",
                    class_id=1,
                    bbox=None,
                    confirmed=True,
                ),
            ],
        )
        save_annotation(ia, pm.label_path_for(img_dir / "pose.jpg"))

        data_yaml = DatasetPreparer(pm).prepare(
            tmp_path / "dataset",
            task="pose",
            val_ratio=0.0,
            kpt_shape=[2, 3],
        )
        data = yaml.safe_load(data_yaml.read_text())

        assert data["names"] == ["person"]
        assert data["kpt_shape"] == [2, 3]

    def test_prepare_classify_creates_symlink_dirs(self, tmp_path):
        # Create project with image_tags for classification
        classes = ["cat", "dog"]
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="test",
            image_dir="images",
            classes=classes,
            task_type="classify",
        )
        img_dir = tmp_path / "proj" / "images"
        for i in range(6):
            (img_dir / f"img_{i:03d}.jpg").write_bytes(b"fake")
            cls_name = classes[i % len(classes)]
            ia = ImageAnnotation(
                image_path=f"img_{i:03d}.jpg",
                image_size=(640, 480),
                image_tags=[cls_name],
                annotations=[],
            )
            save_annotation(ia, pm.label_path_for(img_dir / f"img_{i:03d}.jpg"))

        preparer = DatasetPreparer(pm)
        output_dir = tmp_path / "dataset"
        preparer.prepare(output_dir, task="classify", val_ratio=0.0)

        assert (output_dir / "train" / "cat").is_dir()
        assert (output_dir / "train" / "dog").is_dir()
        cat_imgs = list((output_dir / "train" / "cat").iterdir())
        dog_imgs = list((output_dir / "train" / "dog").iterdir())
        assert len(cat_imgs) + len(dog_imgs) == 6
