"""Tests for classification dataset preparation."""
import pytest
from pathlib import Path

from src.core.annotation import ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.engine.dataset import DatasetPreparer


def test_stratified_split_by_image_tags(tmp_path):
    """Stratified split should use image_tags[0] for classification projects."""
    # Create a classify project
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test Classify",
        classes=["cat", "dog"],
        task_type="classify",
    )

    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    # Create images and labels with image_tags
    for i in range(10):
        class_name = "cat" if i < 5 else "dog"
        img_path = images_dir / f"img_{i:03d}.jpg"
        img_path.write_text("fake image")

        ia = ImageAnnotation(
            image_path=img_path.name,
            image_size=(100, 100),
            image_tags=[class_name],
            annotations=[],
        )
        save_annotation(ia, labels_dir / f"img_{i:03d}.json")

    # Prepare dataset
    preparer = DatasetPreparer(pm)
    dataset_dir = project_dir / "datasets" / "test_split"
    preparer.prepare(dataset_dir, val_ratio=0.3, task="classify")

    # Verify train/val split exists
    assert (dataset_dir / "train").exists()
    assert (dataset_dir / "val").exists()

    # Count images in each split (classification uses folder-per-class structure)
    train_images = []
    for class_dir in (dataset_dir / "train").iterdir():
        if class_dir.is_dir():
            train_images.extend(list(class_dir.glob("*")))

    val_images = []
    for class_dir in (dataset_dir / "val").iterdir():
        if class_dir.is_dir():
            val_images.extend(list(class_dir.glob("*")))

    # Should have roughly 70/30 split (allow ±1 due to rounding)
    assert 6 <= len(train_images) <= 8
    assert 2 <= len(val_images) <= 4
    assert len(train_images) + len(val_images) == 10

    # Verify stratification: both classes should be in both splits
    train_classes = {p.parent.name for p in train_images}
    val_classes = {p.parent.name for p in val_images}
    assert "cat" in train_classes and "dog" in train_classes
    assert "cat" in val_classes and "dog" in val_classes


def test_dataset_classify_filters_unconfirmed_image_tags(tmp_path):
    """image_tags_confirmed=False 的图不应进入 train/val 软链。"""
    project_dir = tmp_path / "classify_proj"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test",
        classes=["cat", "dog"],
        task_type="classify",
    )
    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    def _make(name: str, cls: str, confirmed: bool):
        (images_dir / name).write_text("fake image")
        ia = ImageAnnotation(
            image_path=name,
            image_size=(100, 100),
            image_tags=[cls],
            image_tags_confirmed=confirmed,
            image_tags_source="manual",
        )
        save_annotation(ia, labels_dir / (Path(name).stem + ".json"))

    # 4 confirmed (2 per class) + 1 pending
    _make("a.jpg", "cat", True)
    _make("b.jpg", "cat", True)
    _make("c.jpg", "dog", True)
    _make("d.jpg", "dog", True)
    _make("pending.jpg", "cat", False)

    out = project_dir / "datasets" / "filtered"
    DatasetPreparer(pm).prepare(out, task="classify", val_ratio=0.5, seed=0)

    all_links: list[Path] = []
    for split in ("train", "val"):
        split_dir = out / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    all_links.extend(class_dir.glob("*"))

    names = {p.name for p in all_links}
    assert "pending.jpg" not in names
    assert "a.jpg" in names


def test_classify_prepare_returns_dataset_root_directory(tmp_path):
    """For classify, prepare() must return the dataset root directory (the parent of train/),
    not a data.yaml file path. Ultralytics' check_cls_dataset expects to do `data_dir / "train"`
    on the returned value — passing a yaml file path triggers NotADirectoryError.
    """
    project_dir = tmp_path / "classify_proj"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test",
        classes=["cat", "dog"],
        task_type="classify",
    )
    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    for i in range(4):
        cls = "cat" if i < 2 else "dog"
        img_path = images_dir / f"img_{i}.jpg"
        img_path.write_text("fake image")
        save_annotation(
            ImageAnnotation(
                image_path=img_path.name,
                image_size=(100, 100),
                image_tags=[cls],
                image_tags_confirmed=True,
            ),
            labels_dir / f"img_{i}.json",
        )

    out = project_dir / "datasets" / "current"
    result = DatasetPreparer(pm).prepare(out, task="classify", val_ratio=0.5, seed=0)

    # Result must be a directory (ultralytics expects this for classify)
    assert result.is_dir(), f"Expected directory, got file: {result}"
    # And the directory must contain the train/ subdir ultralytics will iterate
    assert (result / "train").is_dir()
    # Each train subdir must be a class folder containing images
    train_classes = [p.name for p in (result / "train").iterdir() if p.is_dir()]
    assert set(train_classes) == {"cat", "dog"}


def test_classify_backfills_empty_class_folders(tmp_path):
    """Project classes without any images must still appear as empty folders
    under train/ and val/ so ultralytics' alphabetical class indexing stays
    stable across runs.
    """
    project_dir = tmp_path / "classify_proj"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test",
        classes=["cat", "dog", "fox"],  # fox has no images
        task_type="classify",
    )
    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    for i in range(4):
        cls = "cat" if i < 2 else "dog"
        img_path = images_dir / f"img_{i}.jpg"
        img_path.write_text("fake image")
        save_annotation(
            ImageAnnotation(
                image_path=img_path.name,
                image_size=(100, 100),
                image_tags=[cls],
                image_tags_confirmed=True,
            ),
            labels_dir / f"img_{i}.json",
        )

    out = project_dir / "datasets" / "current"
    result = DatasetPreparer(pm).prepare(out, task="classify", val_ratio=0.5, seed=0)

    for split in ("train", "val"):
        split_dir = result / split
        assert split_dir.is_dir()
        sub_dirs = {p.name for p in split_dir.iterdir() if p.is_dir()}
        assert sub_dirs == {"cat", "dog", "fox"}
        # fox folder must exist but be empty
        assert list((split_dir / "fox").iterdir()) == []
