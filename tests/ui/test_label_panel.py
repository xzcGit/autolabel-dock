"""Tests for LabelPanel shell + view routing."""
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QColor


def _make_test_project(tmp_path, task_type: str = "detect"):
    """Create a minimal project with 3 images."""
    from src.core.project import ProjectManager

    pm = ProjectManager.create(
        tmp_path / "proj", "test", classes=["cat", "dog"], task_type=task_type,
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(3):
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


class TestLabelPanelDetectViewRouting:
    def test_creates_detect_view_for_detect_project(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from src.ui.views.detect_pose import DetectPoseView

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert isinstance(panel._view, DetectPoseView)
        assert panel._view._file_list.count() == 3

    def test_creates_classify_view_for_classify_project(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from src.ui.views.classify import ClassifyView

        pm = _make_test_project(tmp_path, task_type="classify")
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert isinstance(panel._view, ClassifyView)

    def test_view_swap_replaces_previous_view(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from src.ui.views.detect_pose import DetectPoseView
        from src.ui.views.classify import ClassifyView

        # Detect first
        pm_det = _make_test_project(tmp_path / "det")
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm_det)
        assert isinstance(panel._view, DetectPoseView)

        # Switch to classify project — view should swap
        pm_cls = _make_test_project(tmp_path / "cls", task_type="classify")
        panel.set_project(pm_cls)
        assert isinstance(panel._view, ClassifyView)

    def test_save_and_cleanup_routes_to_view(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        called = []
        panel._view.commit_pending_save = lambda: called.append(True)
        panel.save_and_cleanup()
        assert called

    def test_undo_redo_route_to_view(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        undone = []
        redone = []
        panel._view.undo = lambda: undone.append(True)
        panel._view.redo = lambda: redone.append(True)
        panel.undo()
        panel.redo()
        assert undone and redone


class TestLabelPanelDetectInternals:
    """Pre-refactor tests adapted for the shell+view split."""

    def test_creates(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._view._file_list.count() == 3

    def test_has_toolbar(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._toolbar is not None  # shell-level toolbar

    def test_has_canvas(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._view._canvas is not None

    def test_has_annotation_panel(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._view._ann_panel is not None

    def test_tool_mode_buttons(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        # Drawing tool buttons live on the view in detect/pose mode
        assert panel._view._btn_select is not None
        assert panel._view._btn_bbox is not None
        assert panel._view._btn_keypoint is not None


class TestLabelPanelRescan:
    def test_rescan_images_finds_new_files(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._view._file_list.count() == 3

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.green))
        img.save(str(img_dir / "img_new.png"), "PNG")

        added = panel.rescan_images()
        assert added == 1
        assert panel._view._file_list.count() == 4

    def test_rescan_images_returns_zero_when_nothing_new(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel.rescan_images() == 0

    def test_rescan_images_zero_when_status_filter_hides_some(self, qapp, tmp_path):
        """Regression: rescan must compare disk against ALL paths, not visible ones.

        Bug: rescan used view.get_visible_paths(), so any active status filter
        excluded hidden images and made rescan think those were "new" -> false
        refresh -> file list rebuild -> auto-scroll snapped current item to the
        bottom of the viewport on every Label-tab switch / F5 / refresh click.
        """
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        # Activate a filter that hides every image (none are confirmed yet)
        panel._view.set_filter("confirmed")
        assert panel._view.get_visible_paths() == []
        # No file actually appeared on disk -> rescan must NOT report additions
        assert panel.rescan_images() == 0

    def test_rescan_images_returns_zero_when_no_project(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        panel = LabelPanel(config_path=tmp_path / "config.json")
        assert panel.rescan_images() == 0

    def test_refresh_button_disabled_initially(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        panel = LabelPanel(config_path=tmp_path / "config.json")
        assert panel._refresh_btn.isEnabled() is False

    def test_refresh_button_enabled_after_set_project(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._refresh_btn.isEnabled() is True

    def test_refresh_button_click_finds_new_images(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        msgs: list[str] = []
        panel.status_changed.connect(msgs.append)
        panel.set_project(pm)

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.green))
        img.save(str(img_dir / "img_new.png"), "PNG")

        panel._refresh_btn.click()

        assert panel._view._file_list.count() == 4
        assert any("发现 1 张新图片" in m for m in msgs)

    def test_refresh_button_click_zero_message(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        msgs: list[str] = []
        panel.status_changed.connect(msgs.append)
        panel.set_project(pm)

        panel._refresh_btn.click()

        assert any("未发现新图片" in m for m in msgs)

    def test_f5_triggers_rescan(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.magenta))
        img.save(str(img_dir / "img_f5.png"), "PNG")

        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F5, Qt.NoModifier)
        panel.keyPressEvent(ev)

        assert panel._view._file_list.count() == 4

    def test_rescan_images_updates_project_stats_total_images(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._view._ann_panel._project_total_label.text() == "总图片: 3"

        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.yellow))
        img.save(str(img_dir / "img_new.png"), "PNG")

        panel.rescan_images()

        assert panel._view._ann_panel._project_total_label.text() == "总图片: 4"

    def test_dropped_images_update_project_stats_total_images(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel

        pm = _make_test_project(tmp_path)
        panel = LabelPanel(config_path=tmp_path / "config.json")
        panel.set_project(pm)
        assert panel._view._ann_panel._project_total_label.text() == "总图片: 3"

        external = tmp_path / "external.png"
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.green))
        img.save(str(external), "PNG")

        panel._on_images_dropped([external])

        assert panel._view._file_list.count() == 4
        assert panel._view._ann_panel._project_total_label.text() == "总图片: 4"


def _combo_items(combo) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


def test_classify_view_add_class_refreshes_filter_combo(qapp, tmp_path, monkeypatch):
    """Adding a class via ClassifyView must refresh LabelPanel's class filter combo (Bug #5)."""
    from src.ui.label_panel import LabelPanel
    from PyQt5.QtWidgets import QInputDialog

    pm = _make_test_project(tmp_path, task_type="classify")
    panel = LabelPanel(config_path=tmp_path / "config.json")
    try:
        panel.set_project(pm)
        before = _combo_items(panel._class_filter_combo)
        assert "rabbit" not in before

        monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("rabbit", True))
        panel._view._on_add_class_clicked()

        after = _combo_items(panel._class_filter_combo)
        assert "rabbit" in after, f"new class missing from combo: {after}"
        # 「所有类别」+ 原来 cat/dog + 新 rabbit = 4
        assert len(after) == 4
    finally:
        if hasattr(panel._view, "cleanup"):
            panel._view.cleanup()
        panel.deleteLater()


def test_detect_view_add_class_refreshes_filter_combo(qapp, tmp_path, monkeypatch):
    """Same coverage for detect view's in-canvas add-class path (Bug #5)."""
    from src.ui.label_panel import LabelPanel

    pm = _make_test_project(tmp_path, task_type="detect")
    panel = LabelPanel(config_path=tmp_path / "config.json")
    panel.set_project(pm)
    before = _combo_items(panel._class_filter_combo)
    assert "wolf" not in before

    # Simulate the add-class side-effect from DetectPoseView._show_class_picker
    pm.add_class("wolf")
    pm.save()
    panel._view.classes_changed.emit()

    after = _combo_items(panel._class_filter_combo)
    assert "wolf" in after


def test_detect_view_delete_images_removes_files_and_refreshes_list(
    qapp, tmp_path, monkeypatch,
):
    """Right-click delete removes image + label and updates the file list."""
    from PyQt5.QtWidgets import QMessageBox
    from src.core.annotation import Annotation, ImageAnnotation
    from src.core.label_io import save_annotation
    from src.ui.label_panel import LabelPanel

    pm = _make_test_project(tmp_path)
    imgs = pm.list_images()
    # Seed an annotation on img0 so the labeled-count branch is exercised.
    ia = ImageAnnotation(image_path=imgs[0].name, image_size=(100, 80))
    ia.annotations.append(Annotation(
        id="ann-0", class_name="cat", class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.3), confirmed=True,
    ))
    save_annotation(ia, pm.label_path_for(imgs[0]))

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    panel = LabelPanel(config_path=tmp_path / "config.json")
    try:
        panel.set_project(pm)
        view = panel._view
        assert view._file_list.count() == 3

        view._on_delete_images([imgs[0], imgs[1]])

        assert view._file_list.count() == 1
        assert not imgs[0].exists()
        assert not imgs[1].exists()
        assert imgs[2].exists()
        assert not pm.label_path_for(imgs[0]).exists()
    finally:
        panel.deleteLater()


def test_detect_view_delete_images_aborts_when_user_says_no(
    qapp, tmp_path, monkeypatch,
):
    from PyQt5.QtWidgets import QMessageBox
    from src.ui.label_panel import LabelPanel

    pm = _make_test_project(tmp_path)
    imgs = pm.list_images()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

    panel = LabelPanel(config_path=tmp_path / "config.json")
    try:
        panel.set_project(pm)
        view = panel._view

        view._on_delete_images([imgs[0]])

        assert view._file_list.count() == 3
        assert imgs[0].exists()
    finally:
        panel.deleteLater()


def test_detect_view_delete_current_image_clears_canvas(
    qapp, tmp_path, monkeypatch,
):
    """Deleting the currently focused image clears canvas and selects a new one."""
    from PyQt5.QtWidgets import QMessageBox
    from src.ui.label_panel import LabelPanel

    pm = _make_test_project(tmp_path)
    imgs = pm.list_images()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    panel = LabelPanel(config_path=tmp_path / "config.json")
    try:
        panel.set_project(pm)
        view = panel._view
        view._file_list.setCurrentRow(0)
        qapp.processEvents()
        assert view._current_image_path == imgs[0]

        view._on_delete_images([imgs[0]])
        qapp.processEvents()

        assert not imgs[0].exists()
        # After deletion, the file list rebuilds and the first surviving image
        # becomes current (via setCurrentRow(0)).
        assert view._current_image_path == imgs[1]
    finally:
        panel.deleteLater()


class TestDetectPoseViewSelectionApi:
    def test_get_selected_image_paths_returns_file_list_selection(
        self, qapp, tmp_path
    ):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path, task_type="detect")
        panel = LabelPanel(config_path=tmp_path / "config.json")
        try:
            panel.set_project(pm)
            view = panel._view
            view._file_list.selectAll()
            paths = view.get_selected_image_paths()
            assert len(paths) == 3
            assert all(isinstance(p, Path) for p in paths)
        finally:
            panel.deleteLater()

    def test_refresh_image_tags_updates_file_list_cache_and_panel(
        self, qapp, tmp_path
    ):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path, task_type="detect")
        panel = LabelPanel(config_path=tmp_path / "config.json")
        try:
            panel.set_project(pm)
            view = panel._view

            target = sorted(pm.list_images())[0]
            view._current_image_path = target  # simulate focus

            view.refresh_image_tags(target, ["x", "y"])

            assert view._file_list._image_tags[str(target)] == {"x", "y"}
            assert view._ann_panel._tag_bar.get_tags() == ["x", "y"]
        finally:
            panel.deleteLater()


class TestClassifyViewSelectionApi:
    def test_get_selected_image_paths_returns_grid_selection(
        self, qapp, tmp_path
    ):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path, task_type="classify")
        panel = LabelPanel(config_path=tmp_path / "config.json")
        try:
            panel.set_project(pm)
            view = panel._view
            view._grid.selectAll()
            paths = view.get_selected_image_paths()
            assert len(paths) == 3
            assert all(isinstance(p, Path) for p in paths)
        finally:
            panel.deleteLater()

    def test_refresh_image_tags_updates_preview_when_current(
        self, qapp, tmp_path
    ):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path, task_type="classify")
        panel = LabelPanel(config_path=tmp_path / "config.json")
        try:
            panel.set_project(pm)
            view = panel._view

            target = sorted(pm.list_images())[0]
            view._preview._current_path = target  # simulate preview focus

            view.refresh_image_tags(target, ["x", "y"])
            assert view._preview._tag_bar.get_tags() == ["x", "y"]
        finally:
            panel.deleteLater()

    def test_grid_forwards_T_key_to_parent(self, qapp):
        from src.ui.views.classify import ThumbnailGridWidget
        assert Qt.Key_T in ThumbnailGridWidget._PARENT_HANDLED_KEYS


class TestLabelPanelTagApply:
    def test_mounts_tag_apply_bar(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from src.ui.tag_widget import TagApplyBar
        panel = LabelPanel(config_path=tmp_path / "config.json")
        try:
            assert isinstance(panel._apply_bar, TagApplyBar)
        finally:
            panel.deleteLater()

    def test_set_project_clears_armed_and_pushes_tags(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        pm = _make_test_project(tmp_path)
        pm.config.tags = ["alpha", "beta"]
        panel = LabelPanel(config_path=tmp_path / "config.json")
        try:
            panel._apply_bar._armed = "old"  # simulate stale armed state
            panel.set_project(pm)
            assert panel._apply_bar.get_armed() is None
            assert panel._apply_bar._available == ["alpha", "beta"]
        finally:
            panel.deleteLater()

    def test_T_with_no_armed_is_silent_noop(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from src.controllers.tags import TagController
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent

        pm = _make_test_project(tmp_path)
        ctrl = TagController()
        ctrl.set_project(pm)
        called: list = []
        ctrl.apply_tag_to_images = lambda tag, paths: called.append((tag, paths)) or 0

        panel = LabelPanel(
            config_path=tmp_path / "config.json", tag_controller=ctrl
        )
        try:
            panel.set_project(pm)
            evt = QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier)
            panel.keyPressEvent(evt)
            assert called == []
        finally:
            panel.deleteLater()

    def test_T_with_armed_and_selection_calls_controller(
        self, qapp, tmp_path
    ):
        from src.ui.label_panel import LabelPanel
        from src.controllers.tags import TagController
        from src.core.label_io import load_annotation
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent

        pm = _make_test_project(tmp_path)
        pm.config.tags = ["blur"]
        pm.save()
        ctrl = TagController()
        ctrl.set_project(pm)

        panel = LabelPanel(
            config_path=tmp_path / "config.json", tag_controller=ctrl
        )
        try:
            panel.set_project(pm)
            panel._apply_bar._on_chip_clicked("blur")
            view = panel._view
            view._file_list.selectAll()

            evt = QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier)
            panel.keyPressEvent(evt)

            for p in pm.list_images():
                ia = load_annotation(pm.label_path_for(p))
                assert ia is not None and "blur" in ia.tags
        finally:
            panel.deleteLater()

    def test_T_with_modifier_does_not_trigger_apply(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        from src.controllers.tags import TagController
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent

        pm = _make_test_project(tmp_path)
        pm.config.tags = ["blur"]
        pm.save()
        ctrl = TagController()
        ctrl.set_project(pm)
        called: list = []
        ctrl.apply_tag_to_images = lambda tag, paths: called.append((tag, paths)) or 0

        panel = LabelPanel(
            config_path=tmp_path / "config.json", tag_controller=ctrl
        )
        try:
            panel.set_project(pm)
            panel._apply_bar._on_chip_clicked("blur")
            panel._view._file_list.selectAll()

            for mod in (Qt.ControlModifier, Qt.ShiftModifier, Qt.AltModifier):
                evt = QKeyEvent(QEvent.KeyPress, Qt.Key_T, mod)
                panel.keyPressEvent(evt)
            assert called == []
        finally:
            panel.deleteLater()

    def test_external_image_tags_changed_refreshes_view(
        self, qapp, tmp_path
    ):
        from src.ui.label_panel import LabelPanel
        from src.controllers.tags import TagController
        pm = _make_test_project(tmp_path)
        ctrl = TagController()
        ctrl.set_project(pm)
        panel = LabelPanel(
            config_path=tmp_path / "config.json", tag_controller=ctrl
        )
        try:
            panel.set_project(pm)
            target = sorted(pm.list_images())[0]
            ctrl.image_tags_changed.emit(target, ["fresh"])
            assert panel._view._file_list._image_tags[str(target)] == {"fresh"}
        finally:
            panel.deleteLater()


class TestLabelPanelAnnotationPanelState:
    def _make_panel(self, qapp, tmp_path):
        from src.ui.label_panel import LabelPanel
        return LabelPanel(config_path=tmp_path / "cfg.json")

    def test_state_cached_before_any_view(self, qapp, tmp_path):
        panel = self._make_panel(qapp, tmp_path)
        try:
            state = {"sizes": [10, 20, 30, 40, 50], "collapsed": {"Tag": True}}
            panel.set_annotation_panel_state(state)
            snapshot = panel.get_annotation_panel_state()
            assert snapshot["sizes"] == [10, 20, 30, 40, 50]
            assert snapshot["collapsed"] == {"Tag": True}
        finally:
            panel.deleteLater()

    def test_state_pushed_to_detect_view(self, qapp, tmp_path):
        from src.core.project import ProjectManager
        panel = self._make_panel(qapp, tmp_path)
        try:
            proj_dir = tmp_path / "p"
            ProjectManager.create(proj_dir, "p", classes=["cat", "dog"], task_type="detect")
            pm = ProjectManager.open(proj_dir)

            state = {"sizes": [200, 100, 50, 50, 200], "collapsed": {"项目统计": True}}
            panel.set_annotation_panel_state(state)
            panel.set_project(pm)

            live = panel.get_annotation_panel_state()
            assert isinstance(live["sizes"], list)
            assert len(live["sizes"]) == 5
            assert live["collapsed"]["项目统计"] is True
        finally:
            panel.deleteLater()

    def test_state_noop_on_classify_view(self, qapp, tmp_path):
        from src.core.project import ProjectManager
        panel = self._make_panel(qapp, tmp_path)
        try:
            proj_dir = tmp_path / "c"
            ProjectManager.create(proj_dir, "c", classes=["a", "b"], task_type="classify")
            pm = ProjectManager.open(proj_dir)

            state = {"sizes": [200, 100, 50, 50, 200], "collapsed": {"属性": True}}
            panel.set_annotation_panel_state(state)
            panel.set_project(pm)  # classify view has no _ann_panel

            snapshot = panel.get_annotation_panel_state()
            assert snapshot == state
        finally:
            panel.deleteLater()
