"""Tests for ClassifyView."""
import gc
from pathlib import Path

import pytest
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QKeyEvent, QPixmap


def _keyev(key) -> QKeyEvent:
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)


def _keyev_with_text(key, text: str) -> QKeyEvent:
    """Realistic key event — Qt fills `text` from the actual keypress.
    The bare-key form (no text) is what the existing tests use, but it
    bypasses QListWidget's keyboardSearch which only fires on non-empty text.
    """
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text)


def _make_classify_project(tmp_path, n_imgs=4, classes=None):
    from src.core.project import ProjectManager
    classes = classes or ["cat", "dog"]
    pm = ProjectManager.create(
        tmp_path / "proj", "p", classes=classes, task_type="classify",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(n_imgs):
        QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


def _show_classify_panel_for_scroll_test(qapp, tmp_path, pm):
    from src.ui.label_panel import LabelPanel

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    panel.resize(900, 700)
    panel.show()
    panel.set_project(pm)
    qapp.processEvents()
    return panel


def test_thumbnail_grid_updates_visual_by_path_without_row_scan(qapp, tmp_path):
    from src.ui.views.classify import ThumbnailGridWidget, _ThumbnailVisual

    class CountingGrid(ThumbnailGridWidget):
        def __init__(self):
            super().__init__()
            self.item_calls = 0

        def item(self, row):
            self.item_calls += 1
            return super().item(row)

    grid = CountingGrid()
    try:
        items = []
        for i in range(3):
            items.append(
                grid.add_image_item(
                    tmp_path / f"img{i}.png",
                    _ThumbnailVisual("unlabeled", "#6c7086", "o", False),
                )
            )

        visual = _ThumbnailVisual("cat", "#89b4fa", "ok", False)
        grid.item_calls = 0
        grid.update_visual(tmp_path / "img2.png", visual)

        assert grid.item_calls == 0
        assert items[-1].data(Qt.UserRole + 2) == visual
    finally:
        grid.deleteLater()


def test_thumbnail_grid_updates_pixmap_by_path_without_row_scan(qapp, tmp_path):
    from src.ui.views.classify import ThumbnailGridWidget, _ThumbnailVisual

    class CountingGrid(ThumbnailGridWidget):
        def __init__(self):
            super().__init__()
            self.item_calls = 0

        def item(self, row):
            self.item_calls += 1
            return super().item(row)

    grid = CountingGrid()
    try:
        item = grid.add_image_item(
            tmp_path / "img0.png",
            _ThumbnailVisual("unlabeled", "#6c7086", "o", False),
        )
        pixmap = QPixmap(8, 8)

        assert hasattr(grid, "update_thumbnail")
        grid.item_calls = 0
        grid.update_thumbnail(tmp_path / "img0.png", pixmap)

        assert grid.item_calls == 0
        assert item.data(Qt.DecorationRole).cacheKey() == pixmap.cacheKey()
    finally:
        grid.deleteLater()


def test_classify_view_lists_all_images(qapp, tmp_path):
    from src.ui.label_panel import LabelPanel
    from src.ui.views.classify import ClassifyView
    pm = _make_classify_project(tmp_path, n_imgs=5)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        assert isinstance(panel._view, ClassifyView)
        assert panel._view._grid.count() == 5
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_view_cleanup_tolerates_loader_already_deleted(qapp, tmp_path):
    from PyQt5 import sip
    from src.ui.label_panel import LabelPanel

    pm = _make_classify_project(tmp_path, n_imgs=1)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        view = panel._view
        loader = view._loader
        view.cleanup()
        sip.delete(loader)
        view._loader = loader

        view.cleanup()
    finally:
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_single_select_number_key_labels_and_advances(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=3, classes=["cat", "dog"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        grid = panel._view._grid

        grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_2))
        panel._view.commit_pending_save()

        img0 = pm.list_images()[0]
        ia = load_annotation(pm.label_path_for(img0))
        assert ia.image_tags == ["dog"]
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "manual"
        assert grid.currentRow() == 1
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_multi_select_number_key_batch_labels(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=4, classes=["cat", "dog"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        grid = panel._view._grid

        grid.setCurrentRow(0)
        for r in [0, 2, 3]:
            grid.item(r).setSelected(True)

        panel._view.keyPressEvent(_keyev(Qt.Key_1))
        for r in [0, 2, 3]:
            img = pm.list_images()[r]
            ia = load_annotation(pm.label_path_for(img))
            assert ia.image_tags == ["cat"]
        img1 = pm.list_images()[1]
        ia1 = load_annotation(pm.label_path_for(img1))
        assert ia1 is None or not ia1.image_tags
        assert grid.currentRow() == 0
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_space_confirms_pending(qapp, tmp_path):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import load_annotation, save_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=2, classes=["cat"])
    img = pm.list_images()[0]
    save_annotation(
        ImageAnnotation(
            image_path=img.name,
            image_size=(20, 20),
            image_tags=["cat"],
            image_tags_confirmed=False,
            image_tags_source="auto",
        ),
        pm.label_path_for(img),
    )
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_Space))

        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "auto"
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_delete_clears_tag(qapp, tmp_path):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import load_annotation, save_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat"])
    img = pm.list_images()[0]
    label_path = pm.label_path_for(img)
    save_annotation(
        ImageAnnotation(
            image_path=img.name,
            image_size=(20, 20),
            image_tags=["cat"],
            image_tags_confirmed=True,
            image_tags_source="manual",
        ),
        label_path,
    )
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_Delete))

        # Clearing the only tag leaves no semantic state — the label file is
        # removed from disk so load_annotation returns None (== unlabeled).
        assert not label_path.exists()
        assert load_annotation(label_path) is None
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_view_bulk_auto_label_defers_confirm_count_rescan(
    qapp, tmp_path, monkeypatch,
):
    from src.ui.label_panel import LabelPanel

    pm = _make_classify_project(tmp_path, n_imgs=3, classes=["cat"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        view = panel._view

        calls = 0
        original = view._count_visible_unconfirmed_auto

        def wrapped():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(view, "_count_visible_unconfirmed_auto", wrapped)
        calls = 0

        view.begin_bulk_auto_label_update()
        try:
            for img in pm.list_images()[:2]:
                assert view.add_auto_class_prediction(img, "cat", 0.9) is True
            assert calls == 0
        finally:
            view.end_bulk_auto_label_update()

        assert calls == 1
        assert view._confirm_all_btn.text() == "确认全部 (2)"
        assert not view._confirm_all_btn.icon().isNull()
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_view_emits_focus_changed(qapp, tmp_path):
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=2)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)

        received = []
        panel._view.image_focus_changed.connect(lambda p: received.append(p))
        panel._view._grid.setCurrentRow(1)
        qapp.processEvents()
        assert received and received[-1] == pm.list_images()[1]
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_view_get_focused_and_visible(qapp, tmp_path):
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=3)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        view = panel._view
        view._grid.setCurrentRow(2)
        assert view.get_focused_image() == pm.list_images()[2]
        assert view.get_visible_paths() == pm.list_images()
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_undo_restores_previous_tag(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat", "dog"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_1))  # cat
        panel._view.keyPressEvent(_keyev(Qt.Key_2))  # dog
        panel.undo()

        img = pm.list_images()[0]
        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["cat"]
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_redo_reapplies_tag(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat", "dog"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_1))  # cat
        panel._view.keyPressEvent(_keyev(Qt.Key_2))  # dog
        panel.undo()
        panel.redo()

        img = pm.list_images()[0]
        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["dog"]
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_preview_pane_closes_and_saves(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.config import AppConfig
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    pm = _make_classify_project(tmp_path / "p", n_imgs=2)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        assert panel._view._preview.isHidden() is False

        panel._view._on_preview_close()

        assert panel._view._preview.isHidden() is True
        cfg = AppConfig.load(tmp_path / ".autolabel" / "config.json")
        assert cfg.classify_preview_visible is False
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_preview_pane_show_persists_visible(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.config import AppConfig
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    cfg_path = tmp_path / ".autolabel" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    cfg.classify_preview_visible = False
    cfg.save(cfg_path)

    pm = _make_classify_project(tmp_path / "p", n_imgs=2)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        assert panel._view._preview.isHidden() is True

        panel._view.show_preview()

        assert panel._view._preview.isHidden() is False
        cfg2 = AppConfig.load(cfg_path)
        assert cfg2.classify_preview_visible is True
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_preview_toggle_button_reopens_preview(qapp, tmp_path, monkeypatch):
    """After closing via ✕, the toolbar toggle button must reopen the preview.

    Regression: previously there was no UI path back from a closed preview.
    """
    import pathlib
    from src.core.config import AppConfig
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    pm = _make_classify_project(tmp_path / "p", n_imgs=2)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        view = panel._view
        assert view._preview.isHidden() is False
        assert view._preview_toggle_btn.isChecked() is True
        assert view._preview_toggle_btn.text() == "预览"
        assert not view._preview_toggle_btn.icon().isNull()

        # Close via the preview's ✕ — toolbar button should reflect new state.
        view._on_preview_close()
        assert view._preview.isHidden() is True
        assert view._preview_toggle_btn.isChecked() is False

        # Click toggle to reopen — preview should become visible again.
        view._preview_toggle_btn.click()
        assert view._preview.isHidden() is False
        assert view._preview_toggle_btn.isChecked() is True
        cfg = AppConfig.load(tmp_path / ".autolabel" / "config.json")
        assert cfg.classify_preview_visible is True

        # Click again to hide via toggle — verifies symmetric behavior.
        view._preview_toggle_btn.click()
        assert view._preview.isHidden() is True
        assert view._preview_toggle_btn.isChecked() is False
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_preview_toggle_button_syncs_with_persisted_hidden_state(qapp, tmp_path, monkeypatch):
    """When project loads with preview hidden in config, toggle starts unchecked."""
    import pathlib
    from src.core.config import AppConfig
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    cfg_path = tmp_path / ".autolabel" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    cfg.classify_preview_visible = False
    cfg.save(cfg_path)

    pm = _make_classify_project(tmp_path / "p", n_imgs=2)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        view = panel._view
        assert view._preview.isHidden() is True
        assert view._preview_toggle_btn.isChecked() is False
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_preview_pane_set_image_updates_meta(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    pm = _make_classify_project(tmp_path / "p", n_imgs=2, classes=["cat", "dog"])
    img0 = pm.list_images()[0]
    save_annotation(
        ImageAnnotation(
            image_path=img0.name,
            image_size=(20, 20),
            image_tags=["cat"],
            image_tags_confirmed=False,
            image_tags_source="auto",
        ),
        pm.label_path_for(img0),
    )

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        qapp.processEvents()
        meta = panel._view._preview._meta_lbl.text()
        assert "cat" in meta and "待确认" in meta and "auto" in meta
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_class_bar_lists_buttons(qapp, tmp_path):
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["a", "b", "c"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        bar = panel._view._class_bar
        assert set(bar._buttons.keys()) == {"a", "b", "c"}
        assert bar._buttons["a"].text().startswith("1")
        assert bar._buttons["b"].text().startswith("2")
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_class_bar_click_labels_focused(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._grid.setCurrentRow(0)
        panel._view._class_bar.class_clicked.emit("cat")

        img = pm.list_images()[0]
        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["cat"]
        assert ia.image_tags_source == "manual"
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_class_bar_rebuilds_after_set_classes(qapp, tmp_path):
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        bar = panel._view._class_bar
        assert "cat" in bar._buttons
        panel._view.set_class_colors({"cat": "#ff0000", "dog": "#00ff00"})
        panel._view.set_classes(["cat", "dog"])
        assert "dog" in bar._buttons
        assert bar._buttons["dog"].text().startswith("2")
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_density_slider_updates_icon_size(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.config import AppConfig
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    pm = _make_classify_project(tmp_path / "p", n_imgs=1)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._density_slider.setValue(128)
        assert panel._view._grid.iconSize().width() == 128
        cfg = AppConfig.load(tmp_path / ".autolabel" / "config.json")
        assert cfg.classify_grid_density == 128
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_density_persisted_value_applied_on_set_project(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.config import AppConfig
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    cfg_path = tmp_path / ".autolabel" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    cfg.classify_grid_density = 144
    cfg.save(cfg_path)

    pm = _make_classify_project(tmp_path / "p", n_imgs=1)
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        assert panel._view._density_slider.value() == 144
        assert panel._view._grid.iconSize().width() == 144
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_sort_by_class_groups(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.annotation import ImageAnnotation
    from src.core.config import AppConfig
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    pm = _make_classify_project(tmp_path / "p", n_imgs=4, classes=["cat", "dog"])
    imgs = pm.list_images()
    save_annotation(
        ImageAnnotation(image_path=imgs[0].name, image_size=(20, 20), image_tags=["cat"]),
        pm.label_path_for(imgs[0]),
    )
    save_annotation(
        ImageAnnotation(image_path=imgs[2].name, image_size=(20, 20), image_tags=["cat"]),
        pm.label_path_for(imgs[2]),
    )
    save_annotation(
        ImageAnnotation(image_path=imgs[1].name, image_size=(20, 20), image_tags=["dog"]),
        pm.label_path_for(imgs[1]),
    )

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view._sort_combo.setCurrentIndex(1)  # 按类别

        grid = panel._view._grid
        order = [Path(grid.item(i).data(Qt.UserRole)).name for i in range(grid.count())]
        assert order[0] == imgs[3].name  # unlabeled first
        cat_pos = [order.index(imgs[0].name), order.index(imgs[2].name)]
        dog_pos = order.index(imgs[1].name)
        assert max(cat_pos) < dog_pos

        cfg = AppConfig.load(tmp_path / ".autolabel" / "config.json")
        assert cfg.classify_grid_sort == "class"
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_sort_persisted_class_applied_on_set_project(qapp, tmp_path, monkeypatch):
    import pathlib
    from src.core.annotation import ImageAnnotation
    from src.core.config import AppConfig
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    cfg_path = tmp_path / ".autolabel" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    cfg.classify_grid_sort = "class"
    cfg.save(cfg_path)

    pm = _make_classify_project(tmp_path / "p", n_imgs=2, classes=["cat"])
    imgs = pm.list_images()
    save_annotation(
        ImageAnnotation(image_path=imgs[1].name, image_size=(20, 20), image_tags=["cat"]),
        pm.label_path_for(imgs[1]),
    )

    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        grid = panel._view._grid
        order = [Path(grid.item(i).data(Qt.UserRole)).name for i in range(grid.count())]
        # Unlabeled (imgs[0]) before cat (imgs[1])
        assert order == [imgs[0].name, imgs[1].name]
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_status_filter_hides_items(qapp, tmp_path):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=3, classes=["cat"])
    imgs = pm.list_images()
    save_annotation(
        ImageAnnotation(image_path=imgs[0].name, image_size=(20, 20), image_tags=["cat"]),
        pm.label_path_for(imgs[0]),
    )
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view.set_filter("unlabeled")
        grid = panel._view._grid
        visible = [
            grid.item(i).data(Qt.UserRole)
            for i in range(grid.count())
            if not grid.item(i).isHidden()
        ]
        assert str(imgs[0]) not in visible
        assert str(imgs[1]) in visible
        assert str(imgs[2]) in visible
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_class_filter_hides_other_classes(qapp, tmp_path):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=3, classes=["cat", "dog"])
    imgs = pm.list_images()
    save_annotation(
        ImageAnnotation(image_path=imgs[0].name, image_size=(20, 20), image_tags=["cat"]),
        pm.label_path_for(imgs[0]),
    )
    save_annotation(
        ImageAnnotation(image_path=imgs[1].name, image_size=(20, 20), image_tags=["dog"]),
        pm.label_path_for(imgs[1]),
    )
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view.set_class_filter("cat")
        grid = panel._view._grid
        visible = [
            grid.item(i).data(Qt.UserRole)
            for i in range(grid.count())
            if not grid.item(i).isHidden()
        ]
        assert str(imgs[0]) in visible
        assert str(imgs[1]) not in visible
        assert str(imgs[2]) not in visible
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_filter_cleared_shows_all(qapp, tmp_path):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=2, classes=["cat"])
    imgs = pm.list_images()
    save_annotation(
        ImageAnnotation(image_path=imgs[0].name, image_size=(20, 20), image_tags=["cat"]),
        pm.label_path_for(imgs[0]),
    )
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        panel._view.set_filter("confirmed")
        panel._view.set_filter(None)
        grid = panel._view._grid
        for i in range(grid.count()):
            assert not grid.item(i).isHidden()
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_unlabeled_filter_keeps_visible_current_item_away_from_scroll_bottom(
    qapp, tmp_path
):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation

    pm = _make_classify_project(tmp_path, n_imgs=120, classes=["cat"])
    imgs = pm.list_images()
    for i, img in enumerate(imgs):
        if i % 5 != 0:
            save_annotation(
                ImageAnnotation(
                    image_path=img.name,
                    image_size=(20, 20),
                    image_tags=["cat"],
                    image_tags_confirmed=True,
                    image_tags_source="manual",
                ),
                pm.label_path_for(img),
            )

    panel = _show_classify_panel_for_scroll_test(qapp, tmp_path, pm)
    try:
        grid = panel._view._grid
        grid.setCurrentRow(50)
        qapp.processEvents()

        sb = grid.verticalScrollBar()
        old_value = sb.value()
        old_max = sb.maximum()
        assert old_value > 0, "test setup: scroll should be non-zero before filter"

        panel._view.set_filter("unlabeled")
        qapp.processEvents()

        new_max = sb.maximum()
        new_value = sb.value()
        assert new_max < old_max, "test setup: filter should shrink max"
        assert new_max > 0, "test setup: filter should still leave scrollable range"
        assert not grid.item(50).isHidden(), "row 50 must remain visible"
        assert new_value < new_max, (
            f"after filter, scroll={new_value} == max={new_max}: "
            "bug — scroll was clamped to bottom of grid"
        )
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_unlabeled_filter_resets_scroll_when_current_item_becomes_hidden(
    qapp, tmp_path
):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation

    pm = _make_classify_project(tmp_path, n_imgs=120, classes=["cat"])
    imgs = pm.list_images()
    for i, img in enumerate(imgs):
        if i % 5 != 0:
            save_annotation(
                ImageAnnotation(
                    image_path=img.name,
                    image_size=(20, 20),
                    image_tags=["cat"],
                    image_tags_confirmed=True,
                    image_tags_source="manual",
                ),
                pm.label_path_for(img),
            )

    panel = _show_classify_panel_for_scroll_test(qapp, tmp_path, pm)
    try:
        grid = panel._view._grid
        grid.setCurrentRow(51)
        qapp.processEvents()

        sb = grid.verticalScrollBar()
        assert sb.value() > 0, "test setup: scroll should be non-zero before filter"

        panel._view.set_filter("unlabeled")
        qapp.processEvents()

        assert grid.item(51).isHidden(), "row 51 must be hidden by unlabeled filter"
        assert sb.value() == 0, (
            f"after filter, scroll={sb.value()} instead of 0: "
            "bug — hidden current item left viewport pinned near list bottom"
        )
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_class_filter_keeps_visible_current_item_away_from_scroll_bottom(
    qapp, tmp_path
):
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation

    pm = _make_classify_project(tmp_path, n_imgs=120, classes=["cat", "dog"])
    imgs = pm.list_images()
    for i, img in enumerate(imgs):
        save_annotation(
            ImageAnnotation(
                image_path=img.name,
                image_size=(20, 20),
                image_tags=["cat" if i % 5 == 0 else "dog"],
                image_tags_confirmed=True,
                image_tags_source="manual",
            ),
            pm.label_path_for(img),
        )

    panel = _show_classify_panel_for_scroll_test(qapp, tmp_path, pm)
    try:
        grid = panel._view._grid
        grid.setCurrentRow(50)
        qapp.processEvents()

        sb = grid.verticalScrollBar()
        old_value = sb.value()
        old_max = sb.maximum()
        assert old_value > 0, "test setup: scroll should be non-zero before filter"

        panel._view.set_class_filter("cat")
        qapp.processEvents()

        new_max = sb.maximum()
        new_value = sb.value()
        assert new_max < old_max, "test setup: filter should shrink max"
        assert new_max > 0, "test setup: filter should still leave scrollable range"
        assert not grid.item(50).isHidden(), "row 50 must remain visible"
        assert new_value < new_max, (
            f"after class filter, scroll={new_value} == max={new_max}: "
            "bug — scroll was clamped to bottom of grid"
        )
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_view_add_auto_prediction_marks_pending(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat", "dog"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        img = pm.list_images()[0]
        panel._view.add_auto_class_prediction(img, "cat", 0.9)

        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["cat"]
        assert ia.image_tags_confirmed is False
        assert ia.image_tags_source == "auto"
        assert ia.status == "pending"
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


def test_classify_view_auto_prediction_then_space_confirms(qapp, tmp_path):
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=1, classes=["cat"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        img = pm.list_images()[0]
        panel._view.add_auto_class_prediction(img, "cat", 0.85)
        panel._view._grid.setCurrentRow(0)
        panel._view.keyPressEvent(_keyev(Qt.Key_Space))

        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "auto"  # source preserved
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


# ── Regression: thumbnail grid must not swallow ClassifyView shortcut keys ──
#
# Bug: QListWidget's default keyPressEvent (inherited from QAbstractItemView)
# calls keyboardSearch() for any key whose `text` is non-empty, and accepts
# the event. That swallowed 1-9 ("1".."9") and Space (" ") before they could
# bubble up to ClassifyView.keyPressEvent — so the user's class shortcuts
# stopped working as soon as focus landed on the grid (i.e. after they
# clicked any thumbnail). The pre-existing tests used `_keyev` which omits
# `text`, so QListWidget's default ignored those events and the bug went
# unnoticed.

@pytest.mark.parametrize("key, text", [
    (Qt.Key_1, "1"),
    (Qt.Key_2, "2"),
    (Qt.Key_9, "9"),
    (Qt.Key_Space, " "),
    (Qt.Key_Backspace, "\b"),
])
def test_thumbnail_grid_does_not_consume_class_shortcut_keys(qapp, key, text):
    """ThumbnailGridWidget must let class-shortcut keys bubble to its parent."""
    from src.ui.views.classify import ThumbnailGridWidget
    grid = ThumbnailGridWidget()
    try:
        ev = _keyev_with_text(key, text)
        grid.keyPressEvent(ev)
        assert not ev.isAccepted(), (
            f"Grid consumed key={key} text={text!r}; "
            "event will not bubble up to ClassifyView.keyPressEvent"
        )
    finally:
        grid.deleteLater()


def test_thumbnail_grid_keeps_navigation_keys(qapp, tmp_path):
    """Sanity: arrow keys must still reach QListWidget (we only forward the
    class-shortcut subset). Verifies the override calls super() rather than
    blanket-ignoring everything."""
    from src.ui.views.classify import ThumbnailGridWidget, _ThumbnailVisual
    grid = ThumbnailGridWidget()
    try:
        for i in range(3):
            grid.add_image_item(
                tmp_path / f"img{i}.png",
                _ThumbnailVisual("unlabeled", "#6c7086", "o", False),
            )
        # Down arrow: QListWidget accepts it. After our fix, we forward to
        # super() so the accepted state must remain True.
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
        grid.keyPressEvent(ev)
        assert ev.isAccepted(), (
            "Arrow key was not consumed by QListWidget — fix over-broadly "
            "ignored events."
        )
    finally:
        grid.deleteLater()


def test_classify_digit_key_routed_through_grid_applies_class(qapp, tmp_path):
    """End-to-end: with focus on the grid, pressing '1' must still tag the
    current image. Mimics QApplication::notify's focus-then-bubble routing
    by walking the parent chain manually until the event is accepted."""
    from src.core.label_io import load_annotation
    from src.ui.label_panel import LabelPanel
    pm = _make_classify_project(tmp_path, n_imgs=2, classes=["cat", "dog"])
    panel = LabelPanel(config_path=tmp_path / "cfg.json")
    try:
        panel.set_project(pm)
        grid = panel._view._grid
        grid.setCurrentRow(0)

        ev = _keyev_with_text(Qt.Key_1, "1")
        target = grid
        while target is not None:
            target.keyPressEvent(ev)
            if ev.isAccepted():
                break
            target = target.parentWidget()

        panel._view.commit_pending_save()
        img0 = pm.list_images()[0]
        ia = load_annotation(pm.label_path_for(img0))
        assert ia is not None and ia.image_tags == ["cat"], (
            "Class shortcut '1' did not propagate from grid to ClassifyView"
        )
    finally:
        panel._view.cleanup()
        panel.deleteLater()
        qapp.processEvents()
        del panel
        gc.collect()
        qapp.processEvents()


# ── Phase 4 (auto-label flow): "确认全部 (N)" toolbar button ──


def _seed_label_for_confirm_all(pm, img_path, *, tag, confirmed, source):
    """Helper: write an ImageAnnotation file with given state."""
    from src.core.annotation import ImageAnnotation
    from src.core.label_io import save_annotation
    ia = ImageAnnotation(image_path=img_path.name, image_size=(20, 20))
    if tag is not None:
        ia.image_tags = [tag]
    ia.image_tags_confirmed = confirmed
    ia.image_tags_source = source
    save_annotation(ia, pm.label_path_for(img_path))


def _teardown_view(view, qapp):
    import gc
    view.cleanup()
    view.deleteLater()
    qapp.processEvents()
    gc.collect()
    qapp.processEvents()


def test_confirm_all_button_count_reflects_visible_unconfirmed_auto(qapp, tmp_path):
    from collections import OrderedDict
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache

    pm = _make_classify_project(tmp_path, n_imgs=4)
    imgs = pm.list_images()
    _seed_label_for_confirm_all(pm, imgs[0], tag="cat", confirmed=False, source="auto")
    _seed_label_for_confirm_all(pm, imgs[1], tag="cat", confirmed=False, source="auto")
    _seed_label_for_confirm_all(pm, imgs[2], tag="cat", confirmed=True, source="auto")
    _seed_label_for_confirm_all(pm, imgs[3], tag="cat", confirmed=False, source="manual")

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        # 2 visible + auto + unconfirmed
        assert "(2)" in view._confirm_all_btn.text()
        assert view._confirm_all_btn.isEnabled()
    finally:
        _teardown_view(view, qapp)


def test_confirm_all_button_disabled_when_zero(qapp, tmp_path):
    from collections import OrderedDict
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache

    pm = _make_classify_project(tmp_path, n_imgs=2)
    imgs = pm.list_images()
    _seed_label_for_confirm_all(pm, imgs[0], tag="cat", confirmed=True, source="auto")
    _seed_label_for_confirm_all(pm, imgs[1], tag="cat", confirmed=True, source="manual")

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        assert view._confirm_all_btn.isEnabled() is False
        assert "(0)" not in view._confirm_all_btn.text()
    finally:
        _teardown_view(view, qapp)


def test_confirm_all_clicked_confirms_only_visible_unconfirmed_auto(qapp, tmp_path):
    from collections import OrderedDict
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache
    from src.core.label_io import load_annotation

    pm = _make_classify_project(tmp_path, n_imgs=4)
    imgs = pm.list_images()
    _seed_label_for_confirm_all(pm, imgs[0], tag="cat", confirmed=False, source="auto")
    _seed_label_for_confirm_all(pm, imgs[1], tag="dog", confirmed=False, source="auto")
    _seed_label_for_confirm_all(pm, imgs[2], tag="cat", confirmed=False, source="manual")
    _seed_label_for_confirm_all(pm, imgs[3], tag="cat", confirmed=True, source="auto")

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        view._on_confirm_all_clicked()

        ia0 = load_annotation(pm.label_path_for(imgs[0]))
        ia1 = load_annotation(pm.label_path_for(imgs[1]))
        ia2 = load_annotation(pm.label_path_for(imgs[2]))
        ia3 = load_annotation(pm.label_path_for(imgs[3]))
        assert ia0.image_tags_confirmed is True
        assert ia1.image_tags_confirmed is True
        # manual untouched
        assert ia2.image_tags_confirmed is False
        # already-confirmed unchanged
        assert ia3.image_tags_confirmed is True
        assert "(0)" not in view._confirm_all_btn.text()
        assert view._confirm_all_btn.isEnabled() is False
    finally:
        _teardown_view(view, qapp)


def test_confirm_all_respects_class_filter(qapp, tmp_path):
    from collections import OrderedDict
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache
    from src.core.label_io import load_annotation

    pm = _make_classify_project(tmp_path, n_imgs=2, classes=["cat", "dog"])
    imgs = pm.list_images()
    _seed_label_for_confirm_all(pm, imgs[0], tag="cat", confirmed=False, source="auto")
    _seed_label_for_confirm_all(pm, imgs[1], tag="dog", confirmed=False, source="auto")

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        view.set_class_filter("cat")
        # Only "cat" image visible
        assert "(1)" in view._confirm_all_btn.text()
        view._on_confirm_all_clicked()
        assert load_annotation(pm.label_path_for(imgs[0])).image_tags_confirmed is True
        # dog still unconfirmed (filtered out)
        assert load_annotation(pm.label_path_for(imgs[1])).image_tags_confirmed is False
    finally:
        _teardown_view(view, qapp)


def test_confirm_all_pushes_undo_per_image(qapp, tmp_path):
    from collections import OrderedDict
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache

    pm = _make_classify_project(tmp_path, n_imgs=2)
    imgs = pm.list_images()
    _seed_label_for_confirm_all(pm, imgs[0], tag="cat", confirmed=False, source="auto")
    _seed_label_for_confirm_all(pm, imgs[1], tag="cat", confirmed=False, source="auto")

    undo_stacks = OrderedDict()
    view = ClassifyView(ImageCache(), undo_stacks)
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        view._on_confirm_all_clicked()
        # Per-image undo: each path has a stack with at least one snapshot.
        # (can_undo requires >1 entries; tests seed labels directly so there
        # is only the post-confirm snapshot — that's still proof that the
        # confirm path went through _push_undo_for.)
        assert str(imgs[0]) in undo_stacks
        assert str(imgs[1]) in undo_stacks
        assert len(undo_stacks[str(imgs[0])]._undo_stack) >= 1
        assert len(undo_stacks[str(imgs[1])]._undo_stack) >= 1
    finally:
        _teardown_view(view, qapp)


def test_delete_images_removes_from_grid_and_disk(qapp, tmp_path, monkeypatch):
    from collections import OrderedDict
    from PyQt5.QtWidgets import QMessageBox
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache

    pm = _make_classify_project(tmp_path, n_imgs=3, classes=["cat"])
    imgs = pm.list_images()
    _seed_label_for_confirm_all(pm, imgs[0], tag="cat", confirmed=True, source="manual")

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        assert view._grid.count() == 3

        view._on_delete_images([imgs[0], imgs[1]])
        qapp.processEvents()

        assert view._grid.count() == 1
        assert not imgs[0].exists()
        assert not imgs[1].exists()
        assert imgs[2].exists()
        assert not pm.label_path_for(imgs[0]).exists()
    finally:
        _teardown_view(view, qapp)


def test_delete_images_aborts_when_user_says_no(qapp, tmp_path, monkeypatch):
    from collections import OrderedDict
    from PyQt5.QtWidgets import QMessageBox
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache

    pm = _make_classify_project(tmp_path, n_imgs=2, classes=["cat"])
    imgs = pm.list_images()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)

        view._on_delete_images([imgs[0]])

        assert view._grid.count() == 2
        assert imgs[0].exists()
    finally:
        _teardown_view(view, qapp)


def test_delete_images_clears_preview_when_focused_image_deleted(qapp, tmp_path, monkeypatch):
    from collections import OrderedDict
    from PyQt5.QtWidgets import QMessageBox
    from src.ui.views.classify import ClassifyView
    from src.utils.image import ImageCache

    pm = _make_classify_project(tmp_path, n_imgs=2, classes=["cat"])
    imgs = pm.list_images()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    view = ClassifyView(ImageCache(), OrderedDict())
    try:
        view.set_classes(pm.config.classes)
        view.set_class_colors({c: "#89b4fa" for c in pm.config.classes})
        view.set_project(pm)
        view._preview.set_image(imgs[0])
        assert view._preview._current_path == imgs[0]

        view._on_delete_images([imgs[0]])

        assert view._preview._current_path is None
    finally:
        _teardown_view(view, qapp)


def test_thumbnail_grid_context_menu_emits_delete_signal(qapp, tmp_path):
    """Right-clicking on the grid emits delete_images_requested for selected paths."""
    from src.ui.views.classify import ThumbnailGridWidget, _ThumbnailVisual

    grid = ThumbnailGridWidget()
    try:
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            grid.add_image_item(p, _ThumbnailVisual("unlabeled", "#6c7086", "o", False))
            paths.append(p)

        results = []
        grid.delete_images_requested.connect(lambda ps: results.append(ps))
        grid.delete_images_requested.emit(paths[:2])

        assert results == [paths[:2]]
    finally:
        grid.deleteLater()


def test_thumbnail_grid_remove_path_drops_item_and_index(qapp, tmp_path):
    from src.ui.views.classify import ThumbnailGridWidget, _ThumbnailVisual

    grid = ThumbnailGridWidget()
    try:
        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        for p in paths:
            grid.add_image_item(p, _ThumbnailVisual("unlabeled", "#6c7086", "o", False))

        grid.remove_path(paths[1])

        assert grid.count() == 2
        assert grid.item_for_path(paths[1]) is None
        assert grid.item_for_path(paths[0]) is not None
        assert grid.item_for_path(paths[2]) is not None
    finally:
        grid.deleteLater()
