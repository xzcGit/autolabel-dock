"""Tests for MainWindow."""
import json
import pytest
from pathlib import Path


def _create_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "images").mkdir()
    (proj / "labels").mkdir()
    config = {
        "name": "test", "image_dir": "images", "label_dir": "labels",
        "classes": ["cat", "dog"], "version": "1.0", "created_at": "2026-01-01",
        "class_colors": {}, "keypoint_templates": {},
        "default_model": "", "auto_label_conf": 0.5, "auto_label_iou": 0.45,
    }
    (proj / "project.json").write_text(json.dumps(config), encoding="utf-8")
    return proj


class TestMainWindow:
    def test_window_creates(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win.windowTitle() == "AutoLabel Dock"
        win.close()

    def test_has_tab_widget(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win.tab_widget is not None
        assert win.tab_widget.count() >= 1  # at least welcome tab
        win.close()

    def test_has_status_bar(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        sb = win.statusBar()
        assert sb is not None
        win.close()

    def test_has_menu_bar(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        mb = win.menuBar()
        assert mb is not None
        # Check key menus exist
        menus = [a.text() for a in mb.actions()]
        assert any("文件" in m for m in menus)
        win.close()

    def test_open_project_sets_title(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "test_proj", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert "test_proj" in win.windowTitle()
        win.close()

    def test_welcome_page_is_first_tab(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win.tab_widget.tabText(0) == "欢迎"
        win.close()

    def test_welcome_page_uses_startup_workspace_roles(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        try:
            welcome = win._welcome
            assert welcome.objectName() == "welcomePage"
            assert welcome.btn_new.property("role") == "primary"
            assert welcome.btn_open.property("role") == "secondary"
            assert welcome.recent_list.property("surface") == "panel"
        finally:
            win.close()

    def test_open_project_creates_panels(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        proj = _create_project(tmp_path)
        pm = ProjectManager.open(proj)
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert win._label_panel is not None
        assert win._train_panel is not None
        assert win._model_panel is not None
        assert win._script_tool_panel is not None
        assert win.tab_widget.count() == 5  # welcome + label + train + model + tools
        win.close()

    def test_open_classify_project_sets_train_task_to_classify(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "proj",
            "classify_project",
            classes=["cat"],
            task_type="classify",
        )
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        assert win._train_panel._task_combo.currentText() == "classify"
        win.close()

    def test_open_project_shows_status(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        proj = _create_project(tmp_path)
        pm = ProjectManager.open(proj)
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        status = win._status_label.text()
        assert "test" in status
        assert "类别" in status
        win.close()

    def test_close_saves_config(self, qapp, tmp_path):
        from src.app import MainWindow

        config_path = tmp_path / "config.json"
        win = MainWindow(config_path=config_path)
        win.close()
        # Config should have been saved
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "window_geometry" in data

    def test_training_finished_for_other_project_skips_model_load(self, qapp, tmp_path):
        """When user switched projects mid-training, completion must NOT auto-load
        the model into the current (wrong) project — just status-bar notify."""
        from unittest.mock import MagicMock
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from src.engine.model_manager import ModelInfo

        proj_a = tmp_path / "A"
        ProjectManager.create(proj_a, "A", classes=["cat"])
        pm_a = ProjectManager.open(proj_a)
        proj_b = tmp_path / "B"
        ProjectManager.create(proj_b, "B", classes=["dog"])
        pm_b = ProjectManager.open(proj_b)

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm_b)  # current project is B

        fake_info = ModelInfo(
            name="detect-1", path="models/detect-x/weights/best.pt",
            task="detect", base_model="yolov8n.pt", classes=["cat"],
        )
        win._train_ctrl.register_model_after_training = MagicMock(return_value=fake_info)
        # Pretend training was started on A.
        win._train_ctrl._project_at_start = pm_a
        win._on_model_load = MagicMock()
        win._refresh_model_lists = MagicMock()

        win._on_training_finished({"mAP50": 0.9})

        win._train_ctrl.register_model_after_training.assert_called_once()
        win._on_model_load.assert_not_called()
        win._refresh_model_lists.assert_not_called()
        assert "A" in win._status_label.text()
        assert "detect-1" in win._status_label.text()
        win.close()

    def test_training_finished_same_project_loads_model(self, qapp, tmp_path):
        from unittest.mock import MagicMock
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from src.engine.model_manager import ModelInfo

        proj = tmp_path / "A"
        ProjectManager.create(proj, "A", classes=["cat"])
        pm = ProjectManager.open(proj)

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        fake_info = ModelInfo(
            name="detect-1", path="models/detect-x/weights/best.pt",
            task="detect", base_model="yolov8n.pt", classes=["cat"],
        )
        win._train_ctrl.register_model_after_training = MagicMock(return_value=fake_info)
        win._train_ctrl._project_at_start = pm
        win._on_model_load = MagicMock()
        win._refresh_model_lists = MagicMock()

        win._on_training_finished({"mAP50": 0.9})

        win._on_model_load.assert_called_once_with(fake_info.id)
        win._refresh_model_lists.assert_called_once()
        assert "已自动加载" in win._status_label.text()
        win.close()

    def test_training_finished_reloads_in_memory_registry(self, qapp, tmp_path):
        """register_model_after_training writes via a fresh ModelRegistry; the
        in-memory ``self._model_registry`` must be reloaded so the model panel
        sees the new entry."""
        from unittest.mock import MagicMock
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from src.engine.model_manager import ModelInfo, ModelRegistry
        from src.engine.trainer import TrainConfig

        proj = tmp_path / "A"
        ProjectManager.create(proj, "A", classes=["cat"])
        pm = ProjectManager.open(proj)

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert win._model_registry.list_models() == []

        # Set up the controller as if a real start() had been called.
        win._train_ctrl._project_at_start = pm
        win._train_ctrl._run_name = "detect-test"
        win._train_ctrl._task_at_start = "detect"
        win._train_ctrl._base_model_at_start = "yolov8n.pt"
        win._train_ctrl._has_prepared_classes = True
        win._train_ctrl._prepared_classes = ["cat"]
        win._train_ctrl._train_config = TrainConfig(
            data_yaml="/tmp/d.yaml", model="yolov8n.pt", task="detect", epochs=1,
        )

        # Stub auto-load — it would try to read a non-existent best.pt.
        win._on_model_load = MagicMock()

        win._on_training_finished({"mAP50": 0.5})

        # The in-memory registry must reflect what register_model_after_training
        # wrote to disk; otherwise _refresh_model_lists would push a stale list
        # to the model panel.
        models = win._model_registry.list_models()
        assert len(models) == 1
        assert models[0].task == "detect"
        win.close()

    def test_close_event_cancel_keeps_window_open(self, qapp, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from PyQt5.QtGui import QCloseEvent
        from PyQt5.QtWidgets import QMessageBox
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        win._train_ctrl._worker = fake_worker
        win._train_ctrl.stop = MagicMock()

        monkeypatch.setattr(
            "src.app.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.No,
        )

        event = QCloseEvent()
        win.closeEvent(event)

        assert not event.isAccepted()
        win._train_ctrl.stop.assert_not_called()
        fake_worker.wait.assert_not_called()
        win._train_ctrl._worker = None
        win.close()

    def test_close_event_accept_stops_and_waits(self, qapp, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from PyQt5.QtGui import QCloseEvent
        from PyQt5.QtWidgets import QMessageBox
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        win._train_ctrl._worker = fake_worker
        win._train_ctrl.stop = MagicMock()

        monkeypatch.setattr(
            "src.app.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Yes,
        )

        event = QCloseEvent()
        win.closeEvent(event)

        win._train_ctrl.stop.assert_called_once()
        fake_worker.wait.assert_called_once_with(30000)
        win._train_ctrl._worker = None
        win.close()

    def test_start_training_warns_when_worker_already_running(self, qapp, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "p", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        win._train_ctrl._worker = fake_worker
        win._train_ctrl.validate_and_prepare = MagicMock(
            side_effect=AssertionError("should not be called")
        )

        warning_calls: list = []
        monkeypatch.setattr(
            "src.app.QMessageBox.warning",
            lambda *args, **kwargs: warning_calls.append(args) or 0,
        )

        win._on_start_training()

        assert len(warning_calls) == 1
        win._train_ctrl.validate_and_prepare.assert_not_called()
        win._train_ctrl._worker = None
        win.close()

    def test_tab_switch_to_label_triggers_rescan(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage, QColor
        from PyQt5.QtCore import Qt

        pm = ProjectManager.create(tmp_path / "proj", "t", classes=["a"])
        img_dir = pm.project_dir / pm.config.image_dir
        for i in range(2):
            img = QImage(40, 40, QImage.Format_RGB32)
            img.fill(QColor(Qt.red))
            img.save(str(img_dir / f"i{i}.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        assert win._label_panel is not None
        assert win._label_panel._view._file_list.count() == 2

        img = QImage(40, 40, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / "new.png"), "PNG")

        welcome_idx = win.tab_widget.indexOf(win._welcome)
        label_idx = win.tab_widget.indexOf(win._label_panel)
        win.tab_widget.setCurrentIndex(welcome_idx)
        win.tab_widget.setCurrentIndex(label_idx)

        assert win._label_panel._view._file_list.count() == 3
        assert "发现 1 张新图片" in win._status_label.text()
        win.close()

    def test_tab_switch_to_label_no_message_when_nothing_new(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage, QColor
        from PyQt5.QtCore import Qt

        pm = ProjectManager.create(tmp_path / "proj", "t", classes=["a"])
        img_dir = pm.project_dir / pm.config.image_dir
        img = QImage(40, 40, QImage.Format_RGB32)
        img.fill(QColor(Qt.red))
        img.save(str(img_dir / "i0.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        win._status_label.setText("sentinel")

        welcome_idx = win.tab_widget.indexOf(win._welcome)
        label_idx = win.tab_widget.indexOf(win._label_panel)
        win.tab_widget.setCurrentIndex(welcome_idx)
        win.tab_widget.setCurrentIndex(label_idx)

        assert win._status_label.text() == "sentinel"
        win.close()

    def test_new_project_refreshes_welcome_recent_list(self, qapp, tmp_path, monkeypatch):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "test_proj", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")
        assert win._welcome.recent_list.count() == 0

        def fake_create_project():
            win._app_config.add_recent_project(str(pm.project_dir))
            return pm

        monkeypatch.setattr(win._project_ctrl, "create_project", fake_create_project)

        win._on_new_project()

        assert win._welcome.recent_list.count() == 1
        assert win._welcome.recent_list.item(0).text() == str(pm.project_dir)
        win.close()

    def test_open_project_shows_project_dir_in_status_bar_left(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(tmp_path / "proj", "test_proj", classes=["cat"])
        win = MainWindow(config_path=tmp_path / "config.json")

        win.open_project(pm)

        assert str(pm.project_dir) in win._project_dir_label.text()
        assert win._project_dir_label.toolTip() == str(pm.project_dir)
        win.close()

    def test_controllers_initialized(self, qapp, tmp_path):
        from src.app import MainWindow

        win = MainWindow(config_path=tmp_path / "config.json")
        assert win._project_ctrl is not None
        assert win._model_ctrl is not None
        assert win._train_ctrl is not None
        win.close()

    def test_auto_label_single_classify_dispatches_classify_path(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.label_io import load_annotation
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage

        pm = ProjectManager.create(
            tmp_path / "proj", "cls", classes=["cat", "dog"], task_type="classify",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "img0.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        # Stub model controller so detect path can't accidentally fire.
        called = {}

        def fake_classify(path, classes):
            called["classify"] = (path, list(classes))
            return ("cat", 0.87)

        def fake_detect(*args, **kwargs):
            called["detect"] = True
            return []

        win._model_ctrl.predict_single_classify = fake_classify
        win._model_ctrl.predict_single = fake_detect

        # Focus an image so get_current_image_path returns it.
        win._label_panel._view._grid.setCurrentRow(0)
        win._on_auto_label_single()

        assert "classify" in called and "detect" not in called
        ia = load_annotation(pm.label_path_for(pm.list_images()[0]))
        assert ia.image_tags == ["cat"]
        assert ia.image_tags_confirmed is False
        assert ia.image_tags_source == "auto"
        win.close()

    def test_auto_label_single_detect_unchanged(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.annotation import Annotation
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage

        pm = ProjectManager.create(
            tmp_path / "proj", "det", classes=["cat"], task_type="detect",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / "img0.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        called = {}

        def fake_classify(*args, **kwargs):
            called["classify"] = True
            return None

        def fake_detect(path, classes, **kwargs):
            called["detect"] = True
            return [Annotation(class_id=0, class_name="cat", bbox=(0.5, 0.5, 0.2, 0.2))]

        win._model_ctrl.predict_single_classify = fake_classify
        win._model_ctrl.predict_single = fake_detect

        win._label_panel._view._file_list.setCurrentRow(0)
        win._on_auto_label_single()

        assert "detect" in called and "classify" not in called
        win.close()

    def test_batch_image_done_updates_filelist_status_detect(self, qapp, tmp_path):
        """Regression: per-image batch result must mark image as 'pending' in file list.

        After the shell+view refactor _file_list moved inside the inner view, but
        _on_batch_image_done still referenced ``_label_panel._file_list`` directly,
        causing an AttributeError that Qt swallowed -> file list status stayed
        'unlabeled' even though the JSON was saved correctly.
        """
        from src.app import MainWindow
        from src.core.annotation import Annotation
        from src.core.label_io import load_annotation
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage

        pm = ProjectManager.create(
            tmp_path / "proj", "det", classes=["cat"], task_type="detect",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        img_path = img_dir / "img0.png"
        QImage(40, 40, QImage.Format_RGB32).save(str(img_path), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        file_list = win._label_panel._view._file_list
        assert file_list._statuses.get(str(img_path), "unlabeled") == "unlabeled"

        pred = Annotation(
            class_id=0, class_name="cat",
            bbox=(0.5, 0.5, 0.2, 0.2),
            confirmed=False, source="auto",
        )
        win._on_batch_image_done(str(img_path), [pred], (40, 40))

        # JSON saved with unconfirmed annotation
        ia = load_annotation(pm.label_path_for(img_path))
        assert ia is not None and len(ia.annotations) == 1
        assert ia.status == "pending"

        # File list must mirror the new status (this was the bug)
        assert file_list._statuses[str(img_path)] == "pending"
        win.close()

    def test_batch_finished_keeps_predictions_for_current_image(self, qapp, tmp_path):
        """Regression: predictions on the focused unlabeled image must survive _on_batch_finished.

        Bug: DetectPoseView.reload_current() routed through _on_image_selected,
        which calls _save_current() first. After the batch worker wrote
        predictions for the focused image to disk, _save_current() overwrote
        them with the stale empty in-memory canvas state — the focused image
        appeared unlabeled even though every other image got labeled.
        """
        from src.app import MainWindow
        from src.core.annotation import Annotation
        from src.core.label_io import load_annotation
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage

        pm = ProjectManager.create(
            tmp_path / "proj", "det", classes=["cat"], task_type="detect",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        img_path = img_dir / "img0.png"
        QImage(40, 40, QImage.Format_RGB32).save(str(img_path), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        view = win._label_panel._view
        assert view._current_image_path == img_path
        assert len(view._canvas.annotations) == 0

        pred = Annotation(
            class_id=0, class_name="cat",
            bbox=(0.5, 0.5, 0.2, 0.2),
            confirmed=False, source="auto",
        )
        win._on_batch_image_done(str(img_path), [pred], (40, 40))

        ia = load_annotation(pm.label_path_for(img_path))
        assert len(ia.annotations) == 1

        # The reload after batch finishes must NOT clobber disk with stale empty state
        win._on_batch_finished()

        ia2 = load_annotation(pm.label_path_for(img_path))
        assert len(ia2.annotations) == 1, "reload_current() overwrote disk with stale empty state"
        assert len(view._canvas.annotations) == 1
        win.close()

    def test_batch_confirm_persists_on_current_image(self, qapp, tmp_path):
        """Regression: right-click batch-confirm must persist on the focused image.

        Bug: _on_batch_confirm / _on_batch_delete / _batch_confirm_visible /
        _batch_revert_visible all called _on_image_selected(current) at the end,
        which invokes _save_current() first and overwrites the on-disk modification
        with the stale canvas state -> the current image's batch op was rolled back.
        """
        from src.app import MainWindow
        from src.core.annotation import Annotation, ImageAnnotation
        from src.core.label_io import save_annotation, load_annotation
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage

        pm = ProjectManager.create(
            tmp_path / "proj", "det", classes=["cat"], task_type="detect",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        img_path = img_dir / "img0.png"
        QImage(40, 40, QImage.Format_RGB32).save(str(img_path), "PNG")

        ia = ImageAnnotation(image_path="img0.png", image_size=(40, 40))
        ia.annotations.append(Annotation(
            class_id=0, class_name="cat",
            bbox=(0.5, 0.5, 0.2, 0.2),
            confirmed=False, source="auto",
        ))
        save_annotation(ia, pm.label_path_for(img_path))

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)

        view = win._label_panel._view
        assert view._current_image_path == img_path
        assert len(view._canvas.annotations) == 1
        assert view._canvas.annotations[0].confirmed is False

        view._on_batch_confirm([img_path])

        ia2 = load_annotation(pm.label_path_for(img_path))
        assert ia2.annotations[0].confirmed is True, \
            "batch confirm rolled back via _save_current() on current image"
        assert view._canvas.annotations[0].confirmed is True
        win.close()

    def test_batch_classify_summary_includes_failed_count(self, qapp, tmp_path):
        """Bug #9: classify batch must surface count of unidentified images."""
        from src.app import MainWindow
        from src.core.project import ProjectManager
        from PyQt5.QtGui import QImage

        pm = ProjectManager.create(
            tmp_path / "p", "p", classes=["cat"], task_type="classify",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        for i in range(2):
            QImage(20, 20, QImage.Format_RGB32).save(str(img_dir / f"img{i}.png"), "PNG")

        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        win._batch_skipped = 0
        win._batch_failed = 0

        # Simulate worker emitting None payloads (predictor couldn't match class)
        for img in pm.list_images():
            win._on_batch_image_done(str(img), None, (0, 0))
        win._on_batch_finished()

        assert win._batch_failed == 2
        assert "失败" in win._status_label.text() or "未识别" in win._status_label.text()
        win.close()


class TestMainWindowTemplateRegistry:
    def test_template_registry_path_uses_autolabel_dir(self, qapp, tmp_path):
        from src.app import MainWindow

        config_path = tmp_path / "config.json"
        win = MainWindow(config_path=config_path)

        assert win._template_registry is not None
        path_str = str(win._template_registry._path)
        assert path_str.endswith("train_templates.json")
        assert ".autolabel" in path_str
        win.close()

    def test_template_registry_injected_into_train_panel(self, qapp, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        config_path = tmp_path / "config.json"
        win = MainWindow(config_path=config_path)

        pm = ProjectManager.create(
            tmp_path / "proj", "p", classes=["a"], task_type="detect",
        )
        win.open_project(pm)

        assert win._train_panel is not None
        assert win._train_panel._template_registry is not None
        assert win._train_panel._template_registry is win._template_registry
        win.close()


class _SyncSingleWorker:
    """Stand-in for SinglePredictWorker that runs synchronously on start()."""

    def __init__(self, annotations=None, error=None):
        from PyQt5.QtCore import QObject, pyqtSignal

        class _Sig(QObject):
            done = pyqtSignal(object)
            error = pyqtSignal(str)
            finished = pyqtSignal()

        self._sig = _Sig()
        self.done = self._sig.done
        self.error = self._sig.error
        self.finished = self._sig.finished
        self._annotations = annotations or []
        self._error = error
        self._running = False

    def isRunning(self):
        return self._running

    def start(self):
        if self._error is not None:
            self.error.emit(self._error)
        else:
            self.done.emit(self._annotations)
        self.finished.emit()


class _StubPredictor:
    last_dropped = 0


class TestLocateAnythingSingleAsync:
    """Single-image auto-label must run off the main thread for the LA backend."""

    def _open_detect_project(self, tmp_path):
        from src.app import MainWindow
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "proj", "p", classes=["cat", "dog"], task_type="detect",
        )
        win = MainWindow(config_path=tmp_path / "config.json")
        win.open_project(pm)
        return win, pm

    def test_la_active_routes_single_through_worker(self, qapp, tmp_path, monkeypatch):
        from src.core.annotation import Annotation

        win, pm = self._open_detect_project(tmp_path)
        try:
            # Pretend a current image is focused.
            img_path = pm.project_dir / "images" / "x.jpg"
            monkeypatch.setattr(
                win._label_panel, "get_current_image_path", lambda: img_path,
            )
            # Mark LA active and install a stub predictor.
            win._la_ctrl._active = True
            win._model_ctrl.set_predictor(_StubPredictor())

            ann = Annotation(
                class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2),
                keypoints=[], confidence=1.0, confirmed=False, source="auto",
            )
            sync_worker = _SyncSingleWorker(annotations=[ann])
            monkeypatch.setattr(
                win._model_ctrl,
                "create_single_predict_worker",
                lambda *a, **k: sync_worker,
            )

            applied = []
            monkeypatch.setattr(
                win._label_panel,
                "add_auto_annotations",
                lambda anns, overlap_iou=0.5: applied.extend(anns),
            )

            win._on_auto_label_single()

            # The result was applied via add_auto_annotations (worker path).
            assert applied == [ann]
            # Buttons re-enabled after the worker finished.
            assert win._label_panel._btn_auto_single.isEnabled() is True
            assert win._label_panel._btn_auto_batch.isEnabled() is True
        finally:
            win.close()

    def test_la_inactive_keeps_synchronous_path(self, qapp, tmp_path, monkeypatch):
        win, pm = self._open_detect_project(tmp_path)
        try:
            img_path = pm.project_dir / "images" / "x.jpg"
            monkeypatch.setattr(
                win._label_panel, "get_current_image_path", lambda: img_path,
            )
            # LA NOT active → must not build a worker; sync predict_single used.
            win._la_ctrl._active = False
            worker_built = []
            monkeypatch.setattr(
                win._model_ctrl,
                "create_single_predict_worker",
                lambda *a, **k: worker_built.append(True),
            )
            monkeypatch.setattr(
                win._model_ctrl, "predict_single", lambda *a, **k: [],
            )
            win._on_auto_label_single()
            assert worker_built == []
        finally:
            win.close()

    def test_la_single_worker_error_surfaces_message(self, qapp, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox

        win, pm = self._open_detect_project(tmp_path)
        try:
            img_path = pm.project_dir / "images" / "x.jpg"
            monkeypatch.setattr(
                win._label_panel, "get_current_image_path", lambda: img_path,
            )
            win._la_ctrl._active = True
            win._model_ctrl.set_predictor(_StubPredictor())

            sync_worker = _SyncSingleWorker(error="推理显存不足 (CUDA OOM)")
            monkeypatch.setattr(
                win._model_ctrl,
                "create_single_predict_worker",
                lambda *a, **k: sync_worker,
            )
            shown = []
            monkeypatch.setattr(
                QMessageBox, "warning",
                lambda *a, **k: shown.append(a[-1] if a else None),
            )

            win._on_auto_label_single()

            # Error surfaced and buttons restored.
            assert any("OOM" in str(s) for s in shown)
            assert win._label_panel._btn_auto_single.isEnabled() is True
        finally:
            win.close()

    def test_la_single_reentrancy_guard_blocks_second_worker(
        self, qapp, tmp_path, monkeypatch,
    ):
        """A second trigger (e.g. Shift+A shortcut bypassing disabled buttons)
        while a worker is in flight must NOT build/start another worker."""
        win, pm = self._open_detect_project(tmp_path)
        try:
            img_path = pm.project_dir / "images" / "x.jpg"
            monkeypatch.setattr(
                win._label_panel, "get_current_image_path", lambda: img_path,
            )
            win._la_ctrl._active = True
            win._model_ctrl.set_predictor(_StubPredictor())

            # An in-flight worker that reports itself as still running and does
            # not emit finished on start() — mimics a real QThread mid-inference.
            class _RunningWorker:
                def __init__(self):
                    self.started = 0

                def isRunning(self):
                    return True

                def start(self):
                    self.started += 1

            in_flight = _RunningWorker()
            win._single_worker = in_flight

            built = []

            def _fake_build(*a, **k):
                built.append(True)
                return _RunningWorker()

            monkeypatch.setattr(
                win._model_ctrl, "create_single_predict_worker", _fake_build,
            )

            win._on_auto_label_single()

            # Guard tripped: no new worker built, in-flight worker untouched.
            assert built == []
            assert win._single_worker is in_flight
        finally:
            win.close()


def _win_with_project(tmp_path: Path):
    """Build a MainWindow with a fresh detect project opened (panels live)."""
    from src.app import MainWindow
    from src.core.project import ProjectManager

    pm = ProjectManager.create(tmp_path / "proj", "p", classes=["cat"])
    win = MainWindow(config_path=tmp_path / "config.json")
    win.open_project(pm)
    return win


class TestLocateAnythingYoloMutex:
    """YOLO↔LocateAnything load mutual exclusion (the YOLO→LA direction).

    The LA→YOLO direction lives in LocateAnythingController.begin_enable (covered
    elsewhere). These cover MainWindow._confirm_disable_la_for_yolo gating both
    YOLO entry points: loading an inference model and starting training.
    """

    # ── _on_model_load ────────────────────────────────────────────────────

    def test_model_load_when_la_inactive_skips_confirm(self, qapp, tmp_path, monkeypatch):
        """LA off → no dialog, plain YOLO load path is unchanged."""
        from unittest.mock import MagicMock

        win = _win_with_project(tmp_path)
        try:
            assert win._la_ctrl.is_active is False
            win._model_ctrl.load_model = MagicMock(return_value=False)
            win._la_ctrl.disable = MagicMock()
            asked: list = []
            monkeypatch.setattr(
                "src.app.QMessageBox.question",
                lambda *a, **k: asked.append(a) or 0,
            )

            win._on_model_load("m-1")

            assert asked == []                      # no confirmation shown
            win._la_ctrl.disable.assert_not_called()
            win._model_ctrl.load_model.assert_called_once_with("m-1")
        finally:
            win.close()

    def test_model_load_when_la_active_confirm_yes_disables_then_loads(self, qapp, tmp_path, monkeypatch):
        """LA on + user agrees → LA disabled first, then YOLO loads."""
        from unittest.mock import MagicMock
        from PyQt5.QtWidgets import QMessageBox

        win = _win_with_project(tmp_path)
        try:
            win._la_ctrl._active = True
            win._la_ctrl.disable = MagicMock()
            win._model_ctrl.load_model = MagicMock(return_value=False)
            monkeypatch.setattr(
                "src.app.QMessageBox.question", lambda *a, **k: QMessageBox.Yes,
            )

            win._on_model_load("m-1")

            win._la_ctrl.disable.assert_called_once()
            win._model_ctrl.load_model.assert_called_once_with("m-1")
        finally:
            win.close()

    def test_model_load_when_la_active_confirm_no_aborts(self, qapp, tmp_path, monkeypatch):
        """LA on + user declines → neither LA disabled nor YOLO loaded."""
        from unittest.mock import MagicMock
        from PyQt5.QtWidgets import QMessageBox

        win = _win_with_project(tmp_path)
        try:
            win._la_ctrl._active = True
            win._la_ctrl.disable = MagicMock()
            win._model_ctrl.load_model = MagicMock(return_value=False)
            monkeypatch.setattr(
                "src.app.QMessageBox.question", lambda *a, **k: QMessageBox.No,
            )

            win._on_model_load("m-1")

            win._la_ctrl.disable.assert_not_called()
            win._model_ctrl.load_model.assert_not_called()
        finally:
            win.close()

    # ── _on_start_training ────────────────────────────────────────────────

    def test_start_training_when_la_inactive_skips_confirm(self, qapp, tmp_path, monkeypatch):
        """LA off → no dialog; training proceeds into validate_and_prepare."""
        from unittest.mock import MagicMock

        win = _win_with_project(tmp_path)
        try:
            assert win._la_ctrl.is_active is False
            win._label_panel.save_and_cleanup = MagicMock()
            win._train_ctrl.validate_and_prepare = MagicMock(return_value=None)
            asked: list = []
            monkeypatch.setattr(
                "src.app.QMessageBox.question",
                lambda *a, **k: asked.append(a) or 0,
            )

            win._on_start_training()

            assert asked == []
            win._train_ctrl.validate_and_prepare.assert_called_once()
        finally:
            win.close()

    def test_start_training_when_la_active_confirm_yes_disables_la(self, qapp, tmp_path, monkeypatch):
        """LA on + user agrees → LA disabled, then training proceeds."""
        from unittest.mock import MagicMock
        from PyQt5.QtWidgets import QMessageBox

        win = _win_with_project(tmp_path)
        try:
            win._la_ctrl._active = True
            win._la_ctrl.disable = MagicMock()
            win._label_panel.save_and_cleanup = MagicMock()
            win._train_ctrl.validate_and_prepare = MagicMock(return_value=None)
            monkeypatch.setattr(
                "src.app.QMessageBox.question", lambda *a, **k: QMessageBox.Yes,
            )

            win._on_start_training()

            win._la_ctrl.disable.assert_called_once()
            win._train_ctrl.validate_and_prepare.assert_called_once()
        finally:
            win.close()

    def test_start_training_when_la_active_confirm_no_aborts_and_resets_button(self, qapp, tmp_path, monkeypatch):
        """LA on + user declines → LA stays, training aborts, start button is
        restored to idle (``_on_start`` had flipped it to the running state)."""
        from unittest.mock import MagicMock
        from PyQt5.QtWidgets import QMessageBox

        win = _win_with_project(tmp_path)
        try:
            win._la_ctrl._active = True
            win._la_ctrl.disable = MagicMock()
            # validate must never run if we abort at the confirmation.
            win._train_ctrl.validate_and_prepare = MagicMock(
                side_effect=AssertionError("should not validate after decline")
            )
            # Simulate train_panel._on_start having flipped the button to running.
            win._train_panel._btn_start.setEnabled(False)
            win._train_panel._btn_start.setText("训练中")
            win._train_panel._btn_stop.setEnabled(True)
            monkeypatch.setattr(
                "src.app.QMessageBox.question", lambda *a, **k: QMessageBox.No,
            )

            win._on_start_training()

            win._la_ctrl.disable.assert_not_called()
            win._train_ctrl.validate_and_prepare.assert_not_called()
            # reset_start_button_idle restored the idle state.
            assert win._train_panel._btn_start.isEnabled() is True
            assert win._train_panel._btn_start.text() == "开始训练"
            assert win._train_panel._btn_stop.isEnabled() is False
        finally:
            win.close()
