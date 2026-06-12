"""End-to-end test: arm tag → multi-select → press T → all tags written."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QColor, QKeyEvent

from src.controllers.tags import TagController
from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.ui.label_panel import LabelPanel


def _make_proj(tmp_path: Path) -> ProjectManager:
    pm = ProjectManager.create(
        tmp_path / "proj", "test", classes=["a"], task_type="detect",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(3):
        img = QImage(80, 60, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    pm.config.tags = ["blur"]
    pm.save()
    return pm


def test_arm_select_T_writes_tag_to_all(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(
        config_path=tmp_path / "config.json", tag_controller=ctrl
    )
    try:
        panel.set_project(pm)
        panel._apply_bar._on_chip_clicked("blur")
        panel._view._file_list.selectAll()

        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))

        for p in pm.list_images():
            ia = load_annotation(pm.label_path_for(p))
            assert ia is not None and "blur" in ia.tags
        for p in pm.list_images():
            assert panel._view._file_list._image_tags[str(p)] == {"blur"}
    finally:
        panel.deleteLater()


def test_second_T_is_idempotent(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(
        config_path=tmp_path / "config.json", tag_controller=ctrl
    )
    try:
        panel.set_project(pm)
        panel._apply_bar._on_chip_clicked("blur")
        panel._view._file_list.selectAll()
        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        for p in pm.list_images():
            ia = load_annotation(pm.label_path_for(p))
            assert "blur" in ia.tags

        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        for p in pm.list_images():
            ia = load_annotation(pm.label_path_for(p))
            assert ia.tags.count("blur") == 1
    finally:
        panel.deleteLater()


def test_tag_removed_from_registry_auto_disarms(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(
        config_path=tmp_path / "config.json", tag_controller=ctrl
    )
    try:
        panel.set_project(pm)
        panel._apply_bar._on_chip_clicked("blur")
        assert panel._apply_bar.get_armed() == "blur"

        ctrl.remove_tag("blur")
        panel.refresh_available_tags()

        assert panel._apply_bar.get_armed() is None
    finally:
        panel.deleteLater()


# ── Scroll-preservation regressions ────────────────────────────────
#
# Background: `apply_tag_to_images` fires `image_tags_changed` once per
# modified image. Previously, classify's `refresh_image_tags` re-ran the
# full `_apply_filters()` pass per signal — which calls
# `scrollToItem(current, PositionAtCenter)` and re-reads every label JSON
# — snapping the user's scrollbar on every batch tag apply when a tag
# filter was engaged. Detect/pose's `set_image_tags` updated only its
# cache and never re-evaluated visibility, leaving tag-filtered lists
# stale. Both paths now do single-item visibility updates that never
# touch scroll.


def _make_classify_proj(tmp_path: Path, n: int = 60) -> ProjectManager:
    pm = ProjectManager.create(
        tmp_path / "proj", "cls", classes=["cat", "dog"], task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(n):
        img = QImage(40, 30, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i:03d}.png"), "PNG")
        cls = "cat" if i % 2 == 0 else "dog"
        ia = ImageAnnotation(
            image_path=f"img{i:03d}.png",
            image_size=(40, 30),
            image_tags=[cls],
            image_tags_confirmed=True,
            tags=["existing"] if i % 3 == 0 else [],
        )
        save_annotation(ia, pm.label_path_for(img_dir / f"img{i:03d}.png"))
    pm.config.tags = ["blur", "existing"]
    pm.save()
    return pm


def _make_detect_proj(tmp_path: Path, n: int = 60) -> ProjectManager:
    pm = ProjectManager.create(
        tmp_path / "proj", "det", classes=["cat", "dog"], task_type="detect",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(n):
        img = QImage(40, 30, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i:03d}.png"), "PNG")
        cls = "cat" if i % 2 == 0 else "dog"
        cls_id = 0 if cls == "cat" else 1
        ann = Annotation(
            id="a1", class_id=cls_id, class_name=cls,
            bbox=(0.5, 0.5, 0.3, 0.3), confirmed=True,
        )
        ia = ImageAnnotation(
            image_path=f"img{i:03d}.png", image_size=(40, 30),
            annotations=[ann],
            tags=["existing"] if i % 3 == 0 else [],
        )
        save_annotation(ia, pm.label_path_for(img_dir / f"img{i:03d}.png"))
    pm.config.tags = ["blur", "existing"]
    pm.save()
    return pm


def _scroll_value(view, attr: str) -> tuple[int, int]:
    sb = getattr(view, attr).verticalScrollBar()
    return sb.value(), sb.maximum()


def test_classify_tag_filter_apply_preserves_scroll(qapp, tmp_path):
    """classify + tag filter active + batch T → scroll must not move."""
    pm = _make_classify_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(config_path=tmp_path / "cfg.json", tag_controller=ctrl)
    panel.resize(800, 600)
    panel.show()
    try:
        panel.set_project(pm)
        view = panel._view
        view.set_tag_filter(TagFilter(includes=("existing",)))
        qapp.processEvents()

        sb = view._grid.verticalScrollBar()
        sb.setValue(sb.maximum())
        qapp.processEvents()
        before = sb.value()

        # Select the 5 trailing visible items and arm "blur".
        visible = [view._grid.item(i) for i in range(view._grid.count())
                   if not view._grid.item(i).isHidden()]
        for it in visible[-5:]:
            it.setSelected(True)
        panel._apply_bar._on_chip_clicked("blur")

        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        qapp.processEvents()

        after = sb.value()
        assert after == before, f"scrollbar moved: {before} → {after}"
    finally:
        panel.deleteLater()


def test_classify_tag_filter_apply_updates_visibility(qapp, tmp_path):
    """classify + include filter + apply matching tag → newly-tagged
    images become visible without disturbing scroll."""
    pm = _make_classify_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(config_path=tmp_path / "cfg.json", tag_controller=ctrl)
    panel.resize(800, 600)
    panel.show()
    try:
        panel.set_project(pm)
        view = panel._view
        view.set_tag_filter(TagFilter(includes=("blur",)))
        qapp.processEvents()
        # No image has "blur" yet → all hidden.
        assert all(view._grid.item(i).isHidden()
                   for i in range(view._grid.count()))

        # Tag the first three images with "blur" via apply path.
        targets = sorted(pm.list_images())[:3]
        for p in targets:
            view._grid.item_for_path(p).setSelected(True)
        panel._apply_bar._on_chip_clicked("blur")
        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        qapp.processEvents()

        visible_paths = {
            Path(view._grid.item(i).data(Qt.UserRole))
            for i in range(view._grid.count())
            if not view._grid.item(i).isHidden()
        }
        assert visible_paths == set(targets)
    finally:
        panel.deleteLater()


def test_classify_class_filter_apply_preserves_scroll(qapp, tmp_path):
    """Defensive: class-filter-only + apply tag → scroll unchanged."""
    pm = _make_classify_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(config_path=tmp_path / "cfg.json", tag_controller=ctrl)
    panel.resize(800, 600)
    panel.show()
    try:
        panel.set_project(pm)
        view = panel._view
        view.set_class_filter("cat")
        qapp.processEvents()

        sb = view._grid.verticalScrollBar()
        sb.setValue(sb.maximum())
        qapp.processEvents()
        before = sb.value()

        visible = [view._grid.item(i) for i in range(view._grid.count())
                   if not view._grid.item(i).isHidden()]
        for it in visible[-5:]:
            it.setSelected(True)
        panel._apply_bar._on_chip_clicked("blur")
        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        qapp.processEvents()

        assert sb.value() == before


    finally:
        panel.deleteLater()


def test_detect_tag_filter_apply_preserves_scroll_and_refreshes(qapp, tmp_path):
    """detect/pose + include filter + apply matching tag → no scroll move
    AND newly-tagged rows become visible (was previously stale)."""
    pm = _make_detect_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(config_path=tmp_path / "cfg.json", tag_controller=ctrl)
    panel.resize(800, 600)
    panel.show()
    try:
        panel.set_project(pm)
        view = panel._view
        view.set_tag_filter(TagFilter(includes=("blur",)))
        qapp.processEvents()
        fl = view._file_list
        # No image has "blur" yet → none visible.
        assert all(fl.item(i).isHidden() for i in range(fl.count()))

        sb = fl.verticalScrollBar()
        before = sb.value()  # at top, 0

        targets = sorted(pm.list_images())[:3]
        for p in targets:
            for i in range(fl.count()):
                if fl.item(i).data(Qt.UserRole) == str(p):
                    fl.item(i).setSelected(True)
                    break
        panel._apply_bar._on_chip_clicked("blur")
        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        qapp.processEvents()

        assert sb.value() == before, f"scrollbar moved {before} → {sb.value()}"
        visible_paths = {
            Path(fl.item(i).data(Qt.UserRole))
            for i in range(fl.count()) if not fl.item(i).isHidden()
        }
        assert visible_paths == set(targets)
    finally:
        panel.deleteLater()


def test_detect_class_filter_apply_preserves_scroll(qapp, tmp_path):
    """detect/pose + class-filter-only + apply tag → scroll unchanged."""
    pm = _make_detect_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    panel = LabelPanel(config_path=tmp_path / "cfg.json", tag_controller=ctrl)
    panel.resize(800, 600)
    panel.show()
    try:
        panel.set_project(pm)
        view = panel._view
        view.set_class_filter("cat")
        qapp.processEvents()

        fl = view._file_list
        sb = fl.verticalScrollBar()
        sb.setValue(sb.maximum())
        qapp.processEvents()
        before = sb.value()

        visible = [fl.item(i) for i in range(fl.count())
                   if not fl.item(i).isHidden()]
        for it in visible[-5:]:
            it.setSelected(True)
        panel._apply_bar._on_chip_clicked("blur")
        panel.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
        qapp.processEvents()

        assert sb.value() == before
    finally:
        panel.deleteLater()
