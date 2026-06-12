"""Tests for ImageFolder format import/export."""
from pathlib import Path

import pytest

from src.core.annotation import ImageAnnotation
from src.core.formats.imagefolder import ImageFolderImporter
from src.core.project import ProjectManager


@pytest.fixture
def sample_imagefolder(tmp_path):
    """Create a sample ImageFolder structure."""
    root = tmp_path / "imagefolder_source"

    # Create class directories
    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir(parents=True)
    (root / "bird").mkdir(parents=True)

    # Create dummy images
    (root / "cat" / "cat_001.jpg").write_text("fake image")
    (root / "cat" / "cat_002.jpg").write_text("fake image")
    (root / "dog" / "dog_001.jpg").write_text("fake image")
    (root / "dog" / "dog_002.jpg").write_text("fake image")
    (root / "bird" / "bird_001.jpg").write_text("fake image")

    return root


@pytest.fixture
def sample_imagefolder_with_splits(tmp_path):
    """Create ImageFolder with train/val splits."""
    root = tmp_path / "imagefolder_splits"

    # Train split
    (root / "train" / "cat").mkdir(parents=True)
    (root / "train" / "dog").mkdir(parents=True)
    (root / "train" / "cat" / "cat_001.jpg").write_text("fake")
    (root / "train" / "cat" / "cat_002.jpg").write_text("fake")
    (root / "train" / "dog" / "dog_001.jpg").write_text("fake")

    # Val split
    (root / "val" / "cat").mkdir(parents=True)
    (root / "val" / "dog").mkdir(parents=True)
    (root / "val" / "cat" / "cat_003.jpg").write_text("fake")
    (root / "val" / "dog" / "dog_002.jpg").write_text("fake")

    return root


@pytest.fixture
def sample_imagefolder_with_conflict(tmp_path):
    """Create ImageFolder with name conflicts."""
    root = tmp_path / "imagefolder_conflict"

    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir(parents=True)

    # Same filename in different classes
    (root / "cat" / "image_001.jpg").write_text("cat image")
    (root / "dog" / "image_001.jpg").write_text("dog image")

    return root


def test_scan_imagefolder_structure(sample_imagefolder):
    """Scan should detect class directories and return class->images mapping."""
    importer = ImageFolderImporter()
    structure = importer.scan_structure(sample_imagefolder)

    assert "cat" in structure
    assert "dog" in structure
    assert "bird" in structure
    assert len(structure["cat"]) == 2
    assert len(structure["dog"]) == 2
    assert len(structure["bird"]) == 1


def test_scan_imagefolder_with_splits(sample_imagefolder_with_splits):
    """Scan should merge train/val splits when merge_splits=True."""
    importer = ImageFolderImporter()
    structure = importer.scan_structure(sample_imagefolder_with_splits, merge_splits=True)

    assert "cat" in structure
    assert "dog" in structure
    assert len(structure["cat"]) == 3  # 2 from train + 1 from val
    assert len(structure["dog"]) == 2  # 1 from train + 1 from val


def test_import_to_project(tmp_path, sample_imagefolder):
    """Import should copy images and generate labels."""
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test Classify",
        classes=[],
        task_type="classify",
    )

    importer = ImageFolderImporter()
    result = importer.import_to_project(sample_imagefolder, pm, copy_mode=True)

    # Check result stats
    assert result["imported"] == 5
    assert result["skipped"] == 0
    assert set(result["classes"]) == {"cat", "dog", "bird"}

    # Check images copied
    images_dir = project_dir / "images"
    assert len(list(images_dir.glob("*.jpg"))) == 5

    # Check labels generated
    labels_dir = project_dir / "labels"
    assert len(list(labels_dir.glob("*.json"))) == 5

    # Check project classes updated
    assert set(pm.config.classes) == {"cat", "dog", "bird"}


def test_import_with_name_conflict(tmp_path, sample_imagefolder_with_conflict):
    """Import should rename conflicting files with class prefix."""
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test Conflict",
        classes=[],
        task_type="classify",
    )

    importer = ImageFolderImporter()
    result = importer.import_to_project(sample_imagefolder_with_conflict, pm, copy_mode=True)

    assert result["imported"] == 2

    images_dir = project_dir / "images"
    image_files = [f.name for f in images_dir.glob("*.jpg")]

    # Should have renamed to avoid conflict
    assert "cat_image_001.jpg" in image_files or "image_001.jpg" in image_files
    assert "dog_image_001.jpg" in image_files or len(image_files) == 2


def test_import_merge_splits(tmp_path, sample_imagefolder_with_splits):
    """Import should merge train/val splits by default."""
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test Splits",
        classes=[],
        task_type="classify",
    )

    importer = ImageFolderImporter()
    result = importer.import_to_project(
        sample_imagefolder_with_splits, pm, copy_mode=True, merge_splits=True
    )

    assert result["imported"] == 5  # All images from train + val
    assert set(result["classes"]) == {"cat", "dog"}


def test_export_to_imagefolder(tmp_path):
    """Export should create folder-per-class structure."""
    from src.core.formats.imagefolder import export_imagefolder
    from src.core.label_io import save_annotation

    # Create a classify project with labeled images
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test Export",
        classes=["cat", "dog"],
        task_type="classify",
    )

    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    # Create dummy images and labels
    (images_dir / "cat_001.jpg").write_text("cat image")
    (images_dir / "cat_002.jpg").write_text("cat image")
    (images_dir / "dog_001.jpg").write_text("dog image")

    ia1 = ImageAnnotation(
        image_path="cat_001.jpg", image_size=(100, 100), image_tags=["cat"]
    )
    ia2 = ImageAnnotation(
        image_path="cat_002.jpg", image_size=(100, 100), image_tags=["cat"]
    )
    ia3 = ImageAnnotation(
        image_path="dog_001.jpg", image_size=(100, 100), image_tags=["dog"]
    )

    save_annotation(ia1, labels_dir / "cat_001.json")
    save_annotation(ia2, labels_dir / "cat_002.json")
    save_annotation(ia3, labels_dir / "dog_001.json")

    # Export
    output_dir = tmp_path / "export_output"
    export_imagefolder(pm, output_dir)

    # Verify structure
    assert (output_dir / "cat").is_dir()
    assert (output_dir / "dog").is_dir()
    assert (output_dir / "cat" / "cat_001.jpg").exists()
    assert (output_dir / "cat" / "cat_002.jpg").exists()
    assert (output_dir / "dog" / "dog_001.jpg").exists()


def test_export_to_csv(tmp_path):
    """Export should create CSV with filename,class columns."""
    from src.core.formats.imagefolder import export_csv
    from src.core.label_io import save_annotation

    # Create a classify project with labeled images
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Test CSV Export",
        classes=["cat", "dog"],
        task_type="classify",
    )

    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    # Create dummy images and labels
    (images_dir / "cat_001.jpg").write_text("cat image")
    (images_dir / "dog_001.jpg").write_text("dog image")

    ia1 = ImageAnnotation(
        image_path="cat_001.jpg", image_size=(100, 100), image_tags=["cat"]
    )
    ia2 = ImageAnnotation(
        image_path="dog_001.jpg", image_size=(100, 100), image_tags=["dog"]
    )

    save_annotation(ia1, labels_dir / "cat_001.json")
    save_annotation(ia2, labels_dir / "dog_001.json")

    # Export
    output_dir = tmp_path / "csv_export"
    output_dir.mkdir()
    export_csv(pm, output_dir)

    # Verify CSV file
    csv_path = output_dir / "labels.csv"
    assert csv_path.exists()

    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == "filename,class"
    assert "cat_001.jpg,cat" in lines
    assert "dog_001.jpg,dog" in lines


def test_imagefolder_import_marks_manual_confirmed(sample_imagefolder, tmp_path):
    """Imports from ImageFolder are treated as manually confirmed labels."""
    from src.core.label_io import load_annotation

    pm = ProjectManager.create(
        project_dir=tmp_path / "proj",
        name="p",
        classes=[],
        task_type="classify",
    )
    ImageFolderImporter().import_to_project(sample_imagefolder, pm)

    images = pm.list_images()
    assert images
    for img in images:
        ia = load_annotation(pm.label_path_for(img))
        assert ia is not None
        assert ia.image_tags  # 都应有标签
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "manual"


# ── Registry-mode (annotations list + source_image_dir) tests ──


def test_export_imagefolder_via_registry(tmp_path):
    """ImageFolder export via ExportRegistry should produce correct structure (Bug #2)."""
    from src.core.formats import get_export_registry
    from src.core.label_io import save_annotation

    project_dir = tmp_path / "proj"
    pm = ProjectManager.create(
        project_dir=project_dir, name="p",
        classes=["cat", "dog"], task_type="classify",
    )
    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"
    (images_dir / "a.jpg").write_text("a")
    (images_dir / "b.jpg").write_text("b")
    save_annotation(
        ImageAnnotation(image_path="a.jpg", image_size=(10, 10), image_tags=["cat"]),
        labels_dir / "a.json",
    )
    save_annotation(
        ImageAnnotation(image_path="b.jpg", image_size=(10, 10), image_tags=["dog"]),
        labels_dir / "b.json",
    )

    annotations = [
        ImageAnnotation(image_path="a.jpg", image_size=(10, 10), image_tags=["cat"]),
        ImageAnnotation(image_path="b.jpg", image_size=(10, 10), image_tags=["dog"]),
    ]
    out = tmp_path / "export"
    get_export_registry().export(
        "ImageFolder", annotations, out,
        source_image_dir=images_dir,
    )

    assert (out / "cat" / "a.jpg").exists()
    assert (out / "dog" / "b.jpg").exists()


def test_export_imagefolder_only_confirmed_skips_pending(tmp_path):
    """only_confirmed=True must skip image_tags_confirmed=False entries (Bug #4)."""
    from src.core.formats import get_export_registry
    from src.core.label_io import save_annotation

    project_dir = tmp_path / "proj"
    pm = ProjectManager.create(
        project_dir=project_dir, name="p",
        classes=["cat", "dog"], task_type="classify",
    )
    images_dir = project_dir / "images"
    (images_dir / "a.jpg").write_text("a")
    (images_dir / "b.jpg").write_text("b")

    annotations = [
        ImageAnnotation(
            image_path="a.jpg", image_size=(10, 10),
            image_tags=["cat"], image_tags_confirmed=True,
        ),
        ImageAnnotation(
            image_path="b.jpg", image_size=(10, 10),
            image_tags=["dog"], image_tags_confirmed=False, image_tags_source="auto",
        ),
    ]
    out = tmp_path / "export_only_confirmed"
    get_export_registry().export(
        "ImageFolder", annotations, out,
        source_image_dir=images_dir,
        only_confirmed=True,
    )

    assert (out / "cat" / "a.jpg").exists()
    assert not (out / "dog").exists() or not (out / "dog" / "b.jpg").exists()


def test_export_csv_via_registry(tmp_path):
    """CSV export via ExportRegistry should not raise NotImplementedError (Bug #2)."""
    from src.core.formats import get_export_registry

    project_dir = tmp_path / "proj"
    pm = ProjectManager.create(
        project_dir=project_dir, name="p",
        classes=["cat"], task_type="classify",
    )
    images_dir = project_dir / "images"
    (images_dir / "a.jpg").write_text("a")

    annotations = [
        ImageAnnotation(image_path="a.jpg", image_size=(10, 10), image_tags=["cat"]),
    ]
    out = tmp_path / "csv_out"
    out.mkdir()
    get_export_registry().export(
        "CSV", annotations, out,
        source_image_dir=images_dir,
    )

    csv_path = out / "labels.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == "filename,class"
    assert "a.jpg,cat" in lines


def test_export_csv_only_confirmed_skips_pending(tmp_path):
    """CSV only_confirmed=True must skip pending entries (Bug #4)."""
    from src.core.formats import get_export_registry

    project_dir = tmp_path / "proj"
    pm = ProjectManager.create(
        project_dir=project_dir, name="p",
        classes=["cat", "dog"], task_type="classify",
    )
    images_dir = project_dir / "images"
    (images_dir / "a.jpg").write_text("a")
    (images_dir / "b.jpg").write_text("b")

    annotations = [
        ImageAnnotation(
            image_path="a.jpg", image_size=(10, 10),
            image_tags=["cat"], image_tags_confirmed=True,
        ),
        ImageAnnotation(
            image_path="b.jpg", image_size=(10, 10),
            image_tags=["dog"], image_tags_confirmed=False, image_tags_source="auto",
        ),
    ]
    out = tmp_path / "csv_oc"
    out.mkdir()
    get_export_registry().export(
        "CSV", annotations, out,
        source_image_dir=images_dir,
        only_confirmed=True,
    )

    body = (out / "labels.csv").read_text().strip().split("\n")
    assert "a.jpg,cat" in body
    assert all("b.jpg" not in line for line in body[1:])
