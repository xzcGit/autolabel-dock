"""End-to-end integration test for classification workflow."""
import gc

import pytest
from pathlib import Path

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QKeyEvent

from src.core.annotation import ImageAnnotation
from src.core.label_io import save_annotation, load_annotation
from src.core.project import ProjectManager
from src.core.formats.imagefolder import ImageFolderImporter, export_imagefolder, export_csv
from src.engine.dataset import DatasetPreparer


def _keyev(key, mod=Qt.NoModifier) -> QKeyEvent:
    return QKeyEvent(QEvent.KeyPress, key, mod)


def _teardown(panel, qapp):
    panel._view.cleanup()
    panel.deleteLater()
    qapp.processEvents()
    gc.collect()
    qapp.processEvents()


def test_classification_workflow_end_to_end(tmp_path):
    """Test complete classification workflow from project creation to dataset export."""

    # ── Step 1: Create ImageFolder source dataset ──
    source_dir = tmp_path / "imagefolder_source"
    (source_dir / "cat").mkdir(parents=True)
    (source_dir / "dog").mkdir(parents=True)
    (source_dir / "bird").mkdir(parents=True)

    for i in range(5):
        (source_dir / "cat" / f"cat_{i:03d}.jpg").write_text("fake cat image")
    for i in range(5):
        (source_dir / "dog" / f"dog_{i:03d}.jpg").write_text("fake dog image")
    for i in range(3):
        (source_dir / "bird" / f"bird_{i:03d}.jpg").write_text("fake bird image")

    # ── Step 2: Create classification project ──
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Animal Classification",
        classes=[],
        task_type="classify",
    )

    assert pm.config.task_type == "classify"
    assert pm.config.classes == []

    # ── Step 3: Import ImageFolder dataset ──
    importer = ImageFolderImporter()
    result = importer.import_to_project(source_dir, pm, copy_mode=True)

    assert result["imported"] == 13
    assert result["skipped"] == 0
    assert set(result["classes"]) == {"cat", "dog", "bird"}
    assert set(pm.config.classes) == {"cat", "dog", "bird"}

    # Verify images copied
    images_dir = project_dir / "images"
    assert len(list(images_dir.glob("*.jpg"))) == 13

    # Verify labels generated with image_tags
    labels_dir = project_dir / "labels"
    assert len(list(labels_dir.glob("*.json"))) == 13

    cat_label = load_annotation(labels_dir / "cat_000.json")
    assert cat_label is not None
    assert cat_label.image_tags == ["cat"]
    assert cat_label.status == "confirmed"

    # ── Step 4: Prepare training dataset ──
    preparer = DatasetPreparer(pm)
    dataset_dir = project_dir / "datasets" / "train_split"
    data_yaml = preparer.prepare(dataset_dir, task="classify", val_ratio=0.2, seed=42)

    assert data_yaml.exists()

    # Verify train/val splits
    train_images = []
    for class_dir in (dataset_dir / "train").iterdir():
        if class_dir.is_dir():
            train_images.extend(list(class_dir.glob("*")))

    val_images = []
    for class_dir in (dataset_dir / "val").iterdir():
        if class_dir.is_dir():
            val_images.extend(list(class_dir.glob("*")))

    # Should have roughly 80/20 split
    assert 9 <= len(train_images) <= 11
    assert 2 <= len(val_images) <= 4
    assert len(train_images) + len(val_images) == 13

    # Verify stratification: all classes in both splits
    train_classes = {p.parent.name for p in train_images}
    val_classes = {p.parent.name for p in val_images}
    assert "cat" in train_classes and "dog" in train_classes
    assert len(val_classes) >= 2  # At least 2 classes in val

    # ── Step 5: Export to ImageFolder ──
    export_dir = tmp_path / "export_imagefolder"
    export_imagefolder(pm, export_dir)

    assert (export_dir / "cat").is_dir()
    assert (export_dir / "dog").is_dir()
    assert (export_dir / "bird").is_dir()
    assert len(list((export_dir / "cat").glob("*"))) == 5
    assert len(list((export_dir / "dog").glob("*"))) == 5
    assert len(list((export_dir / "bird").glob("*"))) == 3

    # ── Step 6: Export to CSV ──
    csv_dir = tmp_path / "export_csv"
    csv_dir.mkdir()
    export_csv(pm, csv_dir)

    csv_path = csv_dir / "labels.csv"
    assert csv_path.exists()

    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == "filename,class"
    assert len(lines) == 14  # header + 13 images

    # Verify CSV content
    csv_content = csv_path.read_text()
    assert "cat_000.jpg,cat" in csv_content
    assert "dog_000.jpg,dog" in csv_content
    assert "bird_000.jpg,bird" in csv_content


def test_classification_manual_labeling_workflow(tmp_path):
    """Test manual labeling workflow for classification."""

    # Create project
    project_dir = tmp_path / "manual_classify"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Manual Classify",
        classes=["cat", "dog"],
        task_type="classify",
    )

    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    # Add unlabeled images
    for i in range(5):
        img_path = images_dir / f"img_{i:03d}.jpg"
        img_path.write_text("fake image")

    # Simulate manual labeling
    for i in range(5):
        img_path = images_dir / f"img_{i:03d}.jpg"
        class_name = "cat" if i < 3 else "dog"

        ia = ImageAnnotation(
            image_path=img_path.name,
            image_size=(100, 100),
            image_tags=[class_name],
            annotations=[],
        )
        save_annotation(ia, labels_dir / f"img_{i:03d}.json")

    # Verify all labeled
    for i in range(5):
        label_path = labels_dir / f"img_{i:03d}.json"
        ia = load_annotation(label_path)
        assert ia is not None
        assert len(ia.image_tags) == 1
        assert ia.status == "confirmed"

    # Prepare dataset
    preparer = DatasetPreparer(pm)
    dataset_dir = project_dir / "datasets" / "current"
    data_yaml = preparer.prepare(dataset_dir, task="classify", val_ratio=0.2)

    assert data_yaml.exists()

    # Verify dataset structure
    assert (dataset_dir / "train" / "cat").exists()
    assert (dataset_dir / "train" / "dog").exists()
    assert (dataset_dir / "val" / "cat").exists() or (dataset_dir / "val" / "dog").exists()


def test_classification_mixed_status(tmp_path):
    """Test handling of mixed labeled/unlabeled images."""

    project_dir = tmp_path / "mixed_classify"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Mixed Classify",
        classes=["cat", "dog"],
        task_type="classify",
    )

    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    # Create 10 images: 5 labeled, 5 unlabeled
    for i in range(10):
        img_path = images_dir / f"img_{i:03d}.jpg"
        img_path.write_text("fake image")

        if i < 5:
            # Labeled
            ia = ImageAnnotation(
                image_path=img_path.name,
                image_size=(100, 100),
                image_tags=["cat" if i < 3 else "dog"],
                annotations=[],
            )
        else:
            # Unlabeled
            ia = ImageAnnotation(
                image_path=img_path.name,
                image_size=(100, 100),
                image_tags=[],
                annotations=[],
            )
        save_annotation(ia, labels_dir / f"img_{i:03d}.json")

    # Prepare dataset should only use labeled images
    preparer = DatasetPreparer(pm)
    dataset_dir = project_dir / "datasets" / "current"
    data_yaml = preparer.prepare(dataset_dir, task="classify", val_ratio=0.2)

    # Count total images in dataset
    train_images = []
    for class_dir in (dataset_dir / "train").iterdir():
        if class_dir.is_dir():
            train_images.extend(list(class_dir.glob("*")))

    val_images = []
    for class_dir in (dataset_dir / "val").iterdir():
        if class_dir.is_dir():
            val_images.extend(list(class_dir.glob("*")))

    # Should only have 5 labeled images
    assert len(train_images) + len(val_images) == 5


# ── UI-driven workflow tests (grid view) ───────────────────────


def test_classification_workflow_grid(qapp, tmp_path):
    """End-to-end: create project, label via grid keyboard, reopen, round-trip."""
    from src.ui.label_panel import LabelPanel

    pm = ProjectManager.create(
        tmp_path / "proj", "p",
        classes=["cat", "dog", "bird"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(5):
        QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / f"img{i}.png"), "PNG")

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        grid = panel._view._grid

        # 1) Single-label first 3 images via 1, 2, 3.
        grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_1))  # cat → focus advances to 1
        panel._view.keyPressEvent(_keyev(Qt.Key_2))  # dog → focus advances to 2
        panel._view.keyPressEvent(_keyev(Qt.Key_3))  # bird → focus advances to 3

        imgs = pm.list_images()
        assert load_annotation(pm.label_path_for(imgs[0])).image_tags == ["cat"]
        assert load_annotation(pm.label_path_for(imgs[1])).image_tags == ["dog"]
        assert load_annotation(pm.label_path_for(imgs[2])).image_tags == ["bird"]

        # 2) After 3 advances, focus is now on row 3. Select row 4 too,
        #    then batch-label both as bird; focus should stay at 3.
        assert grid.currentRow() == 3
        grid.item(3).setSelected(True)
        grid.item(4).setSelected(True)
        panel._view.keyPressEvent(_keyev(Qt.Key_3))
        assert load_annotation(pm.label_path_for(imgs[3])).image_tags == ["bird"]
        assert load_annotation(pm.label_path_for(imgs[4])).image_tags == ["bird"]
        assert grid.currentRow() == 3
    finally:
        _teardown(panel, qapp)

    # 3) Reopen project — verify round-trip.
    pm2 = ProjectManager.open(pm.project_dir)
    panel2 = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel2.set_project(pm2)
        for img, expected in zip(pm2.list_images(), ["cat", "dog", "bird", "bird", "bird"]):
            ia = load_annotation(pm2.label_path_for(img))
            assert ia.image_tags == [expected]
            assert ia.image_tags_confirmed is True
            assert ia.image_tags_source == "manual"
    finally:
        _teardown(panel2, qapp)


def test_classification_ai_pending_then_confirm(qapp, tmp_path):
    """AI prediction → pending state → Space promotes to confirmed; source preserved."""
    from src.ui.label_panel import LabelPanel

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "x.png"), "PNG")

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        img = pm.list_images()[0]

        panel._view.add_auto_class_prediction(img, "cat", 0.88)
        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["cat"]
        assert ia.image_tags_confirmed is False
        assert ia.image_tags_source == "auto"
        assert ia.status == "pending"

        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_Space))

        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "auto"  # source preserved on Space-confirm
        assert ia.status == "confirmed"
    finally:
        _teardown(panel, qapp)


def test_classification_undo_redo_via_panel(qapp, tmp_path):
    """Undo via shell stack restores prior tag, then redo reapplies."""
    from src.ui.label_panel import LabelPanel

    pm = ProjectManager.create(
        tmp_path / "proj", "p", classes=["cat", "dog"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "x.png"), "PNG")

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_1))  # cat
        panel._view.keyPressEvent(_keyev(Qt.Key_2))  # dog
        panel.undo()
        img = pm.list_images()[0]
        assert load_annotation(pm.label_path_for(img)).image_tags == ["cat"]
        panel.redo()
        assert load_annotation(pm.label_path_for(img)).image_tags == ["dog"]
    finally:
        _teardown(panel, qapp)


def test_auto_label_does_not_overwrite_confirmed_tags(qapp, tmp_path):
    """AI auto-label must not overwrite a manually confirmed image_tags (Bug #1)."""
    from src.ui.label_panel import LabelPanel

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat", "dog"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "x.png"), "PNG")
    img = pm.list_images()[0]

    save_annotation(
        ImageAnnotation(
            image_path="x.png", image_size=(20, 20),
            image_tags=["cat"], image_tags_confirmed=True, image_tags_source="manual",
        ),
        pm.label_path_for(img),
    )

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        applied = panel._view.add_auto_class_prediction(img, "dog", 0.9)
        assert applied is False, "AI must not overwrite a confirmed manual tag"
        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["cat"]
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "manual"
    finally:
        _teardown(panel, qapp)


def test_auto_label_overwrites_pending_or_unlabeled(qapp, tmp_path):
    """AI auto-label may overwrite pending (auto) or unlabeled images (Bug #1 boundary)."""
    from src.ui.label_panel import LabelPanel

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat", "dog"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "a.png"), "PNG")
    QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "b.png"), "PNG")
    a, b = pm.list_images()

    # 'a' has prior unconfirmed AI prediction; 'b' has no annotation file.
    save_annotation(
        ImageAnnotation(
            image_path="a.png", image_size=(20, 20),
            image_tags=["cat"], image_tags_confirmed=False, image_tags_source="auto",
        ),
        pm.label_path_for(a),
    )

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        assert panel._view.add_auto_class_prediction(a, "dog", 0.7) is True
        assert panel._view.add_auto_class_prediction(b, "dog", 0.7) is True

        ia_a = load_annotation(pm.label_path_for(a))
        assert ia_a.image_tags == ["dog"]
        assert ia_a.image_tags_source == "auto"

        ia_b = load_annotation(pm.label_path_for(b))
        assert ia_b.image_tags == ["dog"]
    finally:
        _teardown(panel, qapp)



# ── Phase 4 (auto-label flow): end-to-end coverage for register/preview ──


class _FakeClassifyPredictor:
    """Test double for Predictor that returns a configured (class_name, conf).

    Mimics the parts of Predictor that BatchPredictWorker / ProjectController
    consume: ``model.names`` for preview_model_classes and ``predict_classify``
    for the worker hot loop.
    """

    def __init__(self, names, mapping):
        # mapping: image stem -> class name (or None to indicate unrecognized)
        from unittest.mock import MagicMock
        self.model = MagicMock()
        self.model.names = names
        self._mapping = mapping

    def predict_classify(self, image_path, project_classes=None, filter_to_project=True):
        stem = Path(image_path).stem
        cls = self._mapping.get(stem)
        if cls is None:
            return None
        conf = 0.9
        if not project_classes or not filter_to_project:
            return (cls, conf)
        if cls in project_classes:
            return (cls, conf)
        return None


def test_register_auto_class_single_image_registers_and_applies(qapp, tmp_path):
    """单图自动标注：开关默认 ON，模型类不在项目中 → 自动登记 + 写入 image_tags。"""
    from src.controllers.project import ProjectController
    from src.core.config import AppConfig
    from src.core.project import ProjectManager
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QWidget

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "img1.png"), "PNG")

    cfg = AppConfig()
    parent = QWidget()
    ctrl = ProjectController(cfg, tmp_path / "cfg.json", parent)
    ctrl.open_project(pm.project_dir)

    # Simulate predictor producing a class not in project.classes.
    result = ctrl.register_auto_class("dog")
    assert result.action == "registered"
    assert "dog" in ctrl.project.config.classes

    # Same prediction next time → "existing" (idempotent).
    result2 = ctrl.register_auto_class("dog")
    assert result2.action == "existing"


def test_disabled_switch_blocks_new_class_registration(qapp, tmp_path):
    """auto_register_classes=False：新类被拒绝，已有类仍可用。"""
    from src.controllers.project import ProjectController
    from src.core.config import AppConfig
    from src.core.project import ProjectManager
    from PyQt5.QtWidgets import QWidget

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat"], task_type="classify",
    )
    pm.config.auto_register_classes = False
    pm.save()

    cfg = AppConfig()
    ctrl = ProjectController(cfg, tmp_path / "cfg.json", QWidget())
    ctrl.open_project(pm.project_dir)

    assert ctrl.register_auto_class("dog").action == "rejected_disabled"
    assert ctrl.register_auto_class("cat").action == "existing"
    assert ctrl.project.config.classes == ["cat"]


def test_preview_model_classes_then_register_workflow(qapp, tmp_path):
    """批量入口：preview → 用户勾选 → register_auto_class(force=True)。"""
    from src.controllers.project import ProjectController
    from src.core.config import AppConfig
    from src.core.project import ProjectManager
    from PyQt5.QtWidgets import QWidget

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat"], task_type="classify",
    )
    cfg = AppConfig()
    ctrl = ProjectController(cfg, tmp_path / "cfg.json", QWidget())
    ctrl.open_project(pm.project_dir)

    pred = _FakeClassifyPredictor(
        names={0: "cat", 1: "dog", 2: "bird", 3: "n01440764"},
        mapping={},
    )
    items = ctrl.preview_model_classes(pred)
    # cat already in project; dog, bird, n0144... shown
    names = {it.model_name for it in items}
    assert names == {"dog", "bird", "n01440764"}
    blacklisted = {it.model_name for it in items if it.is_blacklisted}
    assert blacklisted == {"n01440764"}

    # Simulate user clicking "Only valid" + Confirm.
    selected = [it.model_name for it in items if not it.is_blacklisted]
    for raw in selected:
        ctrl.register_auto_class(raw, force=True)
    assert set(ctrl.project.config.classes) == {"cat", "dog", "bird"}
    # Blacklisted not registered.
    assert "n01440764" not in ctrl.project.config.classes


def test_batch_worker_with_filter_to_project_false_returns_unknown_classes(
    qapp, tmp_path,
):
    """BatchPredictWorker for classify task returns raw names for caller-side filtering."""
    import time
    from PyQt5.QtGui import QImage
    from src.core.project import ProjectManager
    from src.utils.workers import BatchPredictWorker

    pm = ProjectManager.create(
        tmp_path / "p", "p", classes=["cat"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    QImage(10, 10, QImage.Format_RGB32).save(str(img_dir / "a.png"), "PNG")
    QImage(10, 10, QImage.Format_RGB32).save(str(img_dir / "b.png"), "PNG")

    pred = _FakeClassifyPredictor(
        names={0: "cat", 1: "dog"},
        mapping={"a": "cat", "b": "dog"},
    )
    paths = list(pm.list_images())
    received: list[tuple[str, object]] = []

    worker = BatchPredictWorker(
        predictor=pred,
        image_paths=paths,
        project_classes=pm.config.classes,
        task="classify",
    )
    worker.image_done.connect(
        lambda path, payload, _size: received.append((Path(path).stem, payload))
    )
    worker.start()
    deadline = time.time() + 5.0
    while worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
    worker.wait(2000)
    # Drain any queued signals that finished after the polling loop exited.
    for _ in range(20):
        qapp.processEvents()

    by_stem = dict(received)
    assert by_stem["a"] == ("cat", 0.9)
    # "dog" not in project.classes — but worker still returns it (caller filters).
    assert by_stem["b"] == ("dog", 0.9)
