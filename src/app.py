"""Main application window."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QAction,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation
from src.ui.label_panel import LabelPanel
from src.ui.train_panel import TrainPanel
from src.ui.model_panel import ModelPanel
from src.ui.script_tool_panel import ScriptToolPanel
from src.ui.dialogs import BatchProgressDialog
from src.ui.theme import set_button_role, set_surface, text_style
from src.engine.model_manager import ModelRegistry
from src.utils.workers import BatchPredictWorker
from src.controllers.project import ProjectController
from src.controllers.model import ModelController
from src.controllers.train import TrainController
from src.controllers.tags import TagController
from src.controllers.locateanything import LocateAnythingController
from src.core.train_templates import TemplateRegistry
from src.ui.icons import icon
from src.ui.tag_widget import TagManagerDialog

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".autolabel" / "config.json"
TEMPLATES_PATH = Path.home() / ".autolabel" / "train_templates.json"


class WelcomePage(QWidget):
    """Startup welcome page with recent projects and create/open buttons."""

    def __init__(self, app_config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = app_config
        self._init_ui()

    def _init_ui(self) -> None:
        self.setObjectName("welcomePage")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(56, 44, 56, 44)
        layout.setSpacing(36)

        intro = QWidget()
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(0, 0, 0, 0)
        intro_layout.setSpacing(10)

        title = QLabel("AutoLabel Dock")
        title.setStyleSheet(text_style("display"))
        title.setAlignment(Qt.AlignLeft)
        intro_layout.addWidget(title)

        subtitle = QLabel("图像标注 · 模型训练 · 自动标注")
        subtitle.setStyleSheet(text_style("muted"))
        subtitle.setAlignment(Qt.AlignLeft)
        intro_layout.addWidget(subtitle)
        intro_layout.addSpacing(16)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_new = QPushButton(icon("new_project"), "新建项目")
        self.btn_new.setMinimumWidth(132)
        set_button_role(self.btn_new, "primary")
        btn_layout.addWidget(self.btn_new)

        self.btn_open = QPushButton(icon("open_project"), "打开项目")
        self.btn_open.setMinimumWidth(132)
        set_button_role(self.btn_open, "secondary")
        btn_layout.addWidget(self.btn_open)

        btn_layout.addStretch()
        intro_layout.addLayout(btn_layout)
        intro_layout.addStretch(1)
        layout.addWidget(intro, 1)

        recent_panel = QWidget()
        set_surface(recent_panel, "panel")
        recent_layout = QVBoxLayout(recent_panel)
        recent_layout.setContentsMargins(16, 14, 16, 16)
        recent_layout.setSpacing(10)

        recent_label = QLabel("最近项目")
        recent_label.setStyleSheet(text_style("section"))
        recent_layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setMinimumWidth(360)
        self.recent_list.setMinimumHeight(220)
        set_surface(self.recent_list, "panel")
        self.refresh_recent_projects()
        recent_layout.addWidget(self.recent_list, 1)

        layout.addWidget(recent_panel, 1)

    def refresh_recent_projects(self) -> None:
        """Refresh the recent project list from app config."""
        self.recent_list.clear()
        for project_path in self._config.recent_projects:
            self.recent_list.addItem(QListWidgetItem(project_path))


class MainWindow(QMainWindow):
    """Application main window with tab-based layout.

    Business logic is delegated to controllers:
    - ProjectController: create, open, export, class management
    - ModelController: load, delete, import, single inference
    - TrainController: validate, start, stop, model registration
    """

    def __init__(self, config_path: Path | str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoLabel Dock")

        self._config_path = Path(config_path) if config_path else CONFIG_PATH
        self._app_config = AppConfig.load(self._config_path)
        geo = self._app_config.window_geometry
        self.setGeometry(geo["x"], geo["y"], geo["width"], geo["height"])

        # Controllers
        self._project_ctrl = ProjectController(self._app_config, self._config_path, self)
        self._model_ctrl = ModelController(self)
        self._train_ctrl = TrainController(self)
        self._tag_ctrl = TagController(self)
        self._tag_ctrl.tags_changed.connect(self._refresh_train_tag_breakdown)
        self._tag_ctrl.image_tags_changed.connect(self._refresh_train_tag_breakdown)
        # LocateAnything optional backend controller — drives probe/preflight/load.
        self._la_ctrl = LocateAnythingController(self._model_ctrl, self)
        self._la_ctrl.probe_done.connect(self._on_la_probe_done)
        self._la_ctrl.preflight_blocked.connect(self._on_la_preflight_blocked)
        self._la_ctrl.load_progress.connect(self._on_la_load_progress)
        self._la_ctrl.enabled.connect(self._on_la_enabled)
        self._la_ctrl.disabled.connect(self._on_la_disabled)
        self._la_ctrl.failed.connect(self._on_la_failed)

        # Global template registry — shared across projects
        self._template_registry = TemplateRegistry(TEMPLATES_PATH)
        self._template_registry.load()

        # State
        self._project: ProjectManager | None = None
        self._label_panel: LabelPanel | None = None
        self._train_panel: TrainPanel | None = None
        self._model_panel: ModelPanel | None = None
        self._script_tool_panel: ScriptToolPanel | None = None
        self._model_registry: ModelRegistry | None = None
        self._batch_worker: BatchPredictWorker | None = None
        self._batch_dialog: BatchProgressDialog | None = None
        # Background worker for single-image inference on slow backends (LA).
        self._single_worker = None

        # Central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Welcome page
        self._welcome = WelcomePage(self._app_config)
        self._welcome.btn_new.clicked.connect(self._on_new_project)
        self._welcome.btn_open.clicked.connect(self._on_open_project)
        self._welcome.recent_list.itemDoubleClicked.connect(self._on_recent_clicked)
        self.tab_widget.addTab(self._welcome, icon("welcome"), "欢迎")

        self._setup_menus()

        self._project_dir_label = QLabel()
        self.statusBar().addWidget(self._project_dir_label, 1)
        self._status_label = QLabel("就绪")
        self.statusBar().addPermanentWidget(self._status_label)
        self._set_project_dir_label(None)

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int) -> None:
        """Auto-rescan images when switching to the Label tab."""
        if self._label_panel is None:
            return
        if self.tab_widget.widget(index) is self._label_panel:
            n = self._label_panel.rescan_images()
            if n > 0:
                self._status_label.setText(f"发现 {n} 张新图片")

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("文件")

        new_action = QAction(icon("new_project"), "新建项目", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction(icon("open_project"), "打开项目", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction(icon("export"), "导出...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        import_action = QAction(icon("import"), "导入标注...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        exit_action = QAction(icon("exit"), "退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = mb.addMenu("编辑")

        classes_action = QAction(icon("classes"), "类别管理...", self)
        classes_action.triggered.connect(self._on_class_manager)
        edit_menu.addAction(classes_action)

        tags_action = QAction("Tag 管理...", self)
        tags_action.triggered.connect(self._on_tag_manager)
        edit_menu.addAction(tags_action)

        tools_menu = mb.addMenu("工具")

        structure_action = QAction("模型结构查看器", self)
        structure_action.triggered.connect(self._on_open_structure_viewer)
        tools_menu.addAction(structure_action)

    def _set_project_dir_label(self, project_dir: Path | None) -> None:
        """Update the status bar label for the current project directory."""
        if project_dir is None:
            text = "项目目录: 未打开项目"
            tooltip = text
        else:
            tooltip = str(project_dir)
            text = f"项目目录: {tooltip}"
        self._project_dir_label.setText(text)
        self._project_dir_label.setToolTip(tooltip)

    # ── Project ──────────────────────────────────────────────

    def open_project(self, project_manager: ProjectManager) -> None:
        """Open a project and switch to labeling workspace."""
        self._project = project_manager
        # Keep ProjectController in sync — methods like register_auto_class
        # depend on the controller's internal _project reference.
        self._project_ctrl._project = project_manager
        self.setWindowTitle(f"AutoLabel Dock — {project_manager.config.name}")
        self._set_project_dir_label(project_manager.project_dir)
        self._welcome.refresh_recent_projects()

        self._model_registry = ModelRegistry(project_manager.project_dir / "models")
        self._model_registry.load()
        self._model_ctrl.set_context(project_manager, self._model_registry)

        if self._label_panel is None:
            self._label_panel = LabelPanel(
                config_path=self._config_path,
                tag_controller=self._tag_ctrl,
            )
            self._label_panel.auto_label_single_requested.connect(self._on_auto_label_single)
            self._label_panel.auto_label_batch_requested.connect(self._on_auto_label_batch)
            self._label_panel.status_changed.connect(self._status_label.setText)
            self._label_panel.user_tags_changed.connect(self._on_label_user_tags_changed)
            self._label_panel.la_enable_requested.connect(self._on_la_enable_requested)
            self._label_panel.la_disable_requested.connect(self._la_ctrl.disable)
            self._label_panel.la_query_changed.connect(self._la_ctrl.set_query)
            # Experimental master switch: fully hide the LA bar when disabled.
            self._label_panel.set_la_feature_visible(
                self._app_config.enable_locateanything
            )
            self._label_panel.set_annotation_panel_state({
                "sizes": list(self._app_config.annotation_panel_splitter_sizes),
                "collapsed": dict(self._app_config.annotation_panel_collapsed),
            })
            self.tab_widget.addTab(self._label_panel, icon("label_tab"), "标注")
        self._tag_ctrl.set_project(project_manager, self._project_ctrl.backup_manager)
        # Switching projects: unload any active LA runtime and reset the bar so
        # the new project starts from the collapsed (not-enabled) state.
        if self._la_ctrl.is_active:
            self._la_ctrl.disable()
        self._label_panel.set_la_enabled_state(False)
        self._label_panel.set_project(project_manager)

        if self._train_panel is None:
            self._train_panel = TrainPanel()
            self._train_panel._btn_start.clicked.connect(self._on_start_training)
            self._train_panel.stop_requested.connect(self._on_stop_training)
            self._train_panel.preview_augmentation_requested.connect(self._on_preview_augmentation)
            self._train_panel.filter_changed.connect(self._on_train_tag_filter_changed)
            self._train_panel.inspect_structure_requested.connect(self._on_train_inspect_requested)
            self._train_panel.set_template_registry(self._template_registry)
            self.tab_widget.addTab(self._train_panel, icon("train_tab"), "训练")
        # Push current tag registry into both label panel + train panel.
        self._sync_available_tags()
        # Re-sync when the registry mutates.
        try:
            self._tag_ctrl.tags_changed.disconnect(self._sync_available_tags)
        except TypeError:
            pass
        self._tag_ctrl.tags_changed.connect(self._sync_available_tags)

        if self._model_panel is None:
            self._model_panel = ModelPanel()
            self._model_panel.model_load_requested.connect(self._on_model_load)
            self._model_panel.model_delete_requested.connect(self._on_model_delete)
            self._model_panel.model_rename_requested.connect(self._on_model_rename)
            self._model_panel.model_import_requested.connect(self._on_model_import)
            self._model_panel.model_inspect_requested.connect(self._on_model_inspect_requested)
            self.tab_widget.addTab(self._model_panel, icon("model_tab"), "模型")

        if self._script_tool_panel is None:
            self._script_tool_panel = ScriptToolPanel(
                app_config=self._app_config,
                config_path=self._config_path,
            )
            self._script_tool_panel.status_changed.connect(self._status_label.setText)
            self.tab_widget.addTab(self._script_tool_panel, icon("script_tab"), "小工具")

        self._script_tool_panel.set_working_directory(project_manager.project_dir)
        self._train_panel._task_combo.setCurrentText(project_manager.config.task_type)
        self._model_panel.set_models(self._model_registry.list_models())
        self._train_panel.set_registered_models(self._model_registry.list_models())

        self.tab_widget.setCurrentWidget(self._label_panel)
        self._status_label.setText(
            f"项目: {project_manager.config.name} | "
            f"图片: {len(project_manager.list_images())} | "
            f"类别: {len(project_manager.config.classes)}"
        )

    def _on_new_project(self) -> None:
        pm = self._project_ctrl.create_project()
        if pm:
            self.open_project(pm)

    def _on_open_project(self) -> None:
        pm = self._project_ctrl.open_project_dialog()
        if pm:
            self.open_project(pm)

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        pm = self._project_ctrl.open_recent(item)
        if pm:
            self.open_project(pm)
        else:
            row = self._welcome.recent_list.row(item)
            if row >= 0:
                self._welcome.recent_list.takeItem(row)

    def _on_export(self) -> None:
        if not self._project:
            return
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        try:
            self._project_ctrl.export(self._project)
            self._status_label.setText("导出完成")
        except (OSError, ValueError, KeyError):
            pass  # Error already shown by controller

    def _on_import(self) -> None:
        if not self._project:
            return
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        count = self._project_ctrl.import_annotations(self._project)
        if count is not None and count > 0:
            # Refresh label panel to show imported annotations
            if self._label_panel:
                self._label_panel.set_project(self._project)
            self._status_label.setText(f"导入完成: {count} 个图片")
        elif count == 0:
            self._status_label.setText("导入完成: 无匹配图片")

    def _on_class_manager(self) -> None:
        if not self._project:
            return
        if self._project_ctrl.manage_classes(self._project):
            if self._label_panel:
                self._label_panel.set_project(self._project)

    def _on_tag_manager(self) -> None:
        """Open the project-level tag CRUD dialog."""
        if not self._project:
            return
        original = list(self._project.config.tags)
        dlg = TagManagerDialog(original, self)
        if not dlg.exec_():
            return
        new_tags = dlg.get_tags()
        renames = dlg.get_renames()
        # Apply renames first so subsequent diffs see the renamed values.
        for old, new in renames.items():
            try:
                self._tag_ctrl.rename_tag(old, new)
            except Exception:
                logger.warning("Rename tag failed: %s -> %s", old, new, exc_info=True)
        # Diff sets: removed = original (post-rename) − new; added = new − original.
        after_rename = [renames.get(t, t) for t in original]
        for t in after_rename:
            if t not in new_tags:
                self._tag_ctrl.remove_tag(t, cascade=True)
        for t in new_tags:
            if t not in after_rename:
                try:
                    self._tag_ctrl.add_tag(t)
                except Exception:
                    logger.warning("Add tag failed: %s", t, exc_info=True)

    def _sync_available_tags(self) -> None:
        """Push current project tag registry to LabelPanel + TrainPanel."""
        if not self._project:
            return
        tags = list(self._project.config.tags)
        if self._label_panel is not None:
            self._label_panel.refresh_available_tags()
        if self._train_panel is not None:
            self._train_panel.set_available_tags(tags)
            self._refresh_train_tag_breakdown()

    def _on_label_user_tags_changed(self, path, tags) -> None:
        """Per-image tag edits — the view already saved the JSON; we only
        need to ensure newly-typed tags exist in the project registry so
        future autocomplete + tag-filter dropdowns include them."""
        if not self._project:
            return
        try:
            self._tag_ctrl.register_new_tags(list(tags))
        except Exception:
            logger.warning("register_new_tags failed for %s", path, exc_info=True)

    # ── LocateAnything text-labeling backend ─────────────────

    def _on_la_enable_requested(self) -> None:
        """User clicked the 文本标注 button — run probe → preflight → load."""
        if not self._app_config.enable_locateanything:
            QMessageBox.information(
                self, "提示", "LocateAnything 文本标注后端已在设置中关闭。"
            )
            return
        if self._label_panel:
            self._label_panel.set_la_status("正在检测运行条件…")
        # probe + preflight run synchronously here; the heavy load is offloaded
        # to a background worker inside the controller.
        self._la_ctrl.begin_enable()

    def _on_la_probe_done(self, probe) -> None:
        if not probe.available:
            return  # preflight_blocked carries the actionable message
        if self._label_panel:
            self._label_panel.set_la_status(probe.message or "依赖就绪")

    def _on_la_preflight_blocked(self, message: str) -> None:
        if self._label_panel:
            self._label_panel.set_la_status("")
        QMessageBox.warning(self, "无法启用 LocateAnything", message)
        self._status_label.setText("LocateAnything 未启用")

    def _on_la_load_progress(self, message: str) -> None:
        if self._label_panel:
            self._label_panel.set_la_status(message)
        self._status_label.setText(f"LocateAnything: {message}")

    def _on_la_enabled(self) -> None:
        if self._label_panel:
            self._label_panel.set_la_enabled_state(True)
            # Push the bar's current query into the freshly-loaded predictor.
            prompt, target = self._label_panel.get_la_query()
            self._la_ctrl.set_query(prompt, target)
        # Sync the current model-panel name display so the user sees LA active.
        if self._model_panel:
            self._model_panel.set_current_model_name("LocateAnything (文本标注)")
        self._status_label.setText("LocateAnything 已启用，可用『自动标注 / 批量标注』")

    def _on_la_disabled(self) -> None:
        if self._label_panel:
            self._label_panel.set_la_enabled_state(False)
            self._label_panel.set_la_status("")
        if self._model_panel:
            self._model_panel.set_current_model_name("无")
        self._status_label.setText("LocateAnything 已关闭")

    def _on_la_failed(self, message: str) -> None:
        if self._label_panel:
            self._label_panel.set_la_enabled_state(False)
            self._label_panel.set_la_status("")
        QMessageBox.warning(self, "LocateAnything 加载失败", message)
        self._status_label.setText("LocateAnything 加载失败")

    # ── Model ────────────────────────────────────────────────

    def _confirm_disable_la_for_yolo(self, action_label: str) -> bool:
        """Gate a YOLO load/train on closing LocateAnything first.

        LA runs an out-of-process VLM that holds the GPU; the in-process
        YOLO/Ultralytics runtime (inference predictor *or* a training worker)
        must not coexist with it on a single card. This is the YOLO→LA half of
        the mutual exclusion (the LA→YOLO half already lives in
        ``LocateAnythingController.begin_enable`` which unloads YOLO before its
        VRAM preflight).

        If LA is not active there is nothing to do — return ``True`` silently so
        the plain YOLO path is unchanged. If LA *is* active, ask the user to
        confirm closing it (LA is an explicitly-enabled, VRAM-expensive mode, so
        unlike the silent LA→YOLO unload it warrants a confirmation). On
        agreement, ``disable()`` tears down the sidecar (frees its VRAM) and
        resets the toolbar; returns ``True`` to proceed. On decline, returns
        ``False`` so the caller aborts without loading/training.

        Coordination lives here in ``MainWindow`` — which owns both
        controllers — so ``ModelController`` never depends on
        ``LocateAnythingController`` (each layer knows only its neighbours).
        """
        if not self._la_ctrl.is_active:
            return True
        reply = QMessageBox.question(
            self,
            "关闭 LocateAnything",
            f"LocateAnything（文本标注）正在运行，{action_label}需要先关闭它。\n是否关闭并继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False
        self._la_ctrl.disable()
        return True

    def _on_model_load(self, model_id: str) -> None:
        if not self._confirm_disable_la_for_yolo("加载模型"):
            return
        if self._model_ctrl.load_model(model_id):
            model_info = self._model_ctrl.registry.get(model_id)
            if self._model_panel and model_info:
                self._model_panel.set_current_model_name(model_info.name)
            self._status_label.setText(f"已加载模型: {model_info.name}" if model_info else "")

    def _on_model_delete(self, model_id: str) -> None:
        if self._model_ctrl.delete_model(model_id):
            self._refresh_model_lists()

    def _on_model_rename(self, model_id: str) -> None:
        if self._model_ctrl.rename_model(model_id):
            self._refresh_model_lists()

    def _on_model_import(self) -> None:
        model_info = self._model_ctrl.import_model()
        if model_info:
            self._refresh_model_lists()
            self._status_label.setText(f"已导入模型: {model_info.name}")

    def _refresh_model_lists(self) -> None:
        if self._model_registry:
            models = self._model_registry.list_models()
            if self._model_panel:
                self._model_panel.set_models(models)
            if self._train_panel:
                self._train_panel.set_registered_models(models)

    # ── Model structure viewer ───────────────────────────────

    def _make_structure_dialog(self) -> "ModelStructureDialog":
        """Build a ModelStructureDialog seeded with the current registry models.

        MainWindow owns the ModelController + registry, so it constructs the
        dialog (per the three-layer principle the panels only emit intent).
        """
        from src.ui.model_structure_dialog import ModelStructureDialog

        models = self._model_registry.list_models() if self._model_registry else []
        return ModelStructureDialog(self._model_ctrl, models, self)

    def _on_open_structure_viewer(self) -> None:
        """Menu entry: open the viewer with no preload (user picks a model)."""
        dlg = self._make_structure_dialog()
        dlg.exec_()

    def _on_model_inspect_requested(self, model_id: str) -> None:
        """ModelPanel「查看结构」: preload the selected registry model."""
        dlg = self._make_structure_dialog()
        dlg.load_from_registry(model_id)
        dlg.exec_()

    def _on_train_inspect_requested(self, model_path: str) -> None:
        """TrainPanel「查看结构」: preload the current base model.

        The resolved base-model string may be a registered model's stored path
        or a pretrained weight name (e.g. "yolov8n.pt"). We load it as a file
        path — the dialog surfaces a friendly error if it can't be found/parsed.
        """
        dlg = self._make_structure_dialog()
        if model_path:
            # Registered models store a path relative to the project dir; a
            # pretrained weight (e.g. "yolov8n.pt") stays as-is. Resolve
            # relative paths against the project so the parse can find the file.
            p = Path(model_path)
            if not p.is_absolute() and self._project is not None:
                candidate = self._project.project_dir / p
                if candidate.exists():
                    p = candidate
            dlg.load_from_path(p)
        dlg.exec_()

    # ── Auto-label ───────────────────────────────────────────

    def _on_auto_label_single(self) -> None:
        if not self._label_panel or not self._project:
            return
        # Re-entrancy guard: a slow-backend (LA) single-image inference is still
        # running on the worker thread. Disabling the toolbar buttons blocks
        # mouse clicks, but the Shift+A shortcut emits the request directly, so
        # guard here to avoid stacking overlapping workers (and leaking the
        # in-flight QThread reference).
        if self._single_worker is not None and self._single_worker.isRunning():
            return
        img_path = self._label_panel.get_current_image_path()
        if not img_path:
            return
        if self._project.config.task_type == "classify":
            result = self._model_ctrl.predict_single_classify(
                img_path, self._project.config.classes,
            )
            if result is None:
                self._status_label.setText("自动标注: 未识别")
                return
            raw_name, conf = result
            reg = self._project_ctrl.register_auto_class(raw_name)
            if reg.action in ("registered", "existing"):
                applied_name = reg.applied_name
                if reg.action == "registered" and self._label_panel:
                    # Refresh class buttons & filter combo so the new class is visible.
                    self._label_panel.set_project(self._project)
                applied = self._label_panel.add_auto_class_prediction(
                    img_path, applied_name, conf,
                )
                if not applied:
                    self._status_label.setText("自动标注: 已存在确认标签，跳过")
                else:
                    suffix = (
                        f" (新增类别 '{applied_name}')"
                        if reg.action == "registered" else ""
                    )
                    self._status_label.setText(
                        f"自动标注: {applied_name} ({conf:.2f}){suffix}"
                    )
            elif reg.action == "rejected_disabled":
                if raw_name in self._project.config.classes:
                    applied = self._label_panel.add_auto_class_prediction(
                        img_path, raw_name, conf,
                    )
                    if not applied:
                        self._status_label.setText("自动标注: 已存在确认标签，跳过")
                    else:
                        self._status_label.setText(
                            f"自动标注: {raw_name} ({conf:.2f})"
                        )
                else:
                    self._status_label.setText(
                        f"自动标注: 类别 '{raw_name}' 不在项目中，已跳过（未开启自动登记）"
                    )
            elif reg.action == "rejected_blacklist":
                self._status_label.setText(
                    f"自动标注: 模型类名 '{raw_name}' 不可用（疑似 ImageNet ID），已跳过"
                )
            else:
                self._status_label.setText("自动标注: 模型类名无效，已跳过")
            return
        conf = self._model_panel.get_conf_threshold() if self._model_panel else 0.5
        iou = self._model_panel.get_iou_threshold() if self._model_panel else 0.45
        overlap_iou = self._model_panel.get_overlap_iou_threshold() if self._model_panel else 0.5
        class_match_mode = self._model_panel.get_class_match_mode() if self._model_panel else "class_id"

        # Slow backends (LocateAnything) block for seconds inside predict(). On a
        # single-GPU box the X server shares the same card, so running that on
        # the Qt/X event loop stalls — and can crash — the desktop. Route slow
        # backends through a background worker; keep YOLO synchronous (fast, and
        # existing tests depend on the synchronous path).
        if self._la_ctrl.is_active:
            self._start_async_single_auto_label(
                img_path, conf, iou, overlap_iou, class_match_mode,
            )
            return

        annotations = self._model_ctrl.predict_single(
            img_path,
            self._project.config.classes,
            conf=conf,
            iou=iou,
            class_match_mode=class_match_mode,
        )
        self._apply_single_auto_label_result(annotations, overlap_iou)

    def _start_async_single_auto_label(
        self, img_path, conf, iou, overlap_iou, class_match_mode,
    ) -> None:
        """Run single-image inference on a worker thread (slow backends).

        Disables the auto-label buttons for the duration so the user can't stack
        overlapping inference requests, then applies the result on the main
        thread via the same ``add_auto_annotations`` path as the sync flow.
        """
        worker = self._model_ctrl.create_single_predict_worker(
            img_path,
            self._project.config.classes,
            conf=conf,
            iou=iou,
            class_match_mode=class_match_mode,
        )
        if worker is None:
            return
        if self._label_panel:
            self._label_panel.set_auto_label_busy(True)
        self._status_label.setText("自动标注进行中…")
        self._single_worker = worker
        # Bind overlap_iou for the result slot without re-reading the panel.
        worker.done.connect(
            lambda anns: self._on_single_auto_label_done(anns, overlap_iou)
        )
        worker.error.connect(self._on_single_auto_label_error)
        worker.finished.connect(self._on_single_auto_label_worker_finished)
        worker.start()

    def _on_single_auto_label_done(self, annotations, overlap_iou: float) -> None:
        self._apply_single_auto_label_result(annotations, overlap_iou)

    def _on_single_auto_label_error(self, message: str) -> None:
        self._status_label.setText("自动标注失败")
        QMessageBox.warning(self, "自动标注失败", message)

    def _on_single_auto_label_worker_finished(self) -> None:
        if self._label_panel:
            self._label_panel.set_auto_label_busy(False)
        self._single_worker = None

    def _apply_single_auto_label_result(self, annotations, overlap_iou: float) -> None:
        """Apply single-image predictions to the canvas (shared sync/async).

        Surfaces the open-vocabulary 'dropped unmatched class' count when the
        active predictor reports one (LocateAnything).
        """
        if not self._label_panel:
            return
        # Open-vocabulary backends (e.g. LocateAnything) drop detections whose
        # name didn't match any project class — surface that count if present.
        dropped = getattr(self._model_ctrl.predictor, "last_dropped", 0) or 0
        drop_suffix = f"，丢弃 {dropped} 个未匹配类别" if dropped else ""
        if annotations:
            self._label_panel.add_auto_annotations(annotations, overlap_iou=overlap_iou)
            self._status_label.setText(
                f"自动标注: 检测到 {len(annotations)} 个目标{drop_suffix}"
            )
        else:
            self._status_label.setText(f"自动标注: 未检测到目标{drop_suffix}")

    def _on_auto_label_batch(self) -> None:
        if not self._model_ctrl.predictor:
            QMessageBox.information(self, "提示", "请先在模型面板中加载一个模型")
            return
        if not self._label_panel or not self._project:
            return
        self._label_panel.save_and_cleanup()

        # Classification pre-flight: validate / register new model classes.
        if self._project.config.task_type == "classify":
            cfg = self._project.config
            if not cfg.classes and not cfg.auto_register_classes:
                QMessageBox.information(
                    self, "提示",
                    "项目当前没有类别，且未开启自动登记。\n"
                    "请先在『类别管理』中添加类别，或开启自动登记后重试。",
                )
                return
            if cfg.auto_register_classes:
                preview_items = self._project_ctrl.preview_model_classes(
                    self._model_ctrl.predictor,
                )
                if preview_items:
                    from src.ui.dialogs import ClassRegisterDialog
                    dlg = ClassRegisterDialog(preview_items, parent=self)
                    if not dlg.exec_():
                        return
                    selected = dlg.get_selected()
                    new_count = 0
                    for raw in selected:
                        result = self._project_ctrl.register_auto_class(raw, force=True)
                        if result.action == "registered":
                            new_count += 1
                    if new_count and self._label_panel:
                        # Refresh class bar / filter combo for the newly registered classes.
                        self._label_panel.set_project(self._project)

        all_images = self._project.list_images()
        unlabeled = self._label_panel.get_unlabeled_image_paths()

        items = ["仅未标注图片", "全部图片"]
        from PyQt5.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(
            self, "批量自动标注", f"选择范围 (未标注: {len(unlabeled)} / 全部: {len(all_images)})",
            items, 0, False,
        )
        if not ok:
            return

        target_images = unlabeled if choice == items[0] else all_images
        if not target_images:
            QMessageBox.information(self, "提示", "没有需要处理的图片")
            return

        conf = self._model_panel.get_conf_threshold() if self._model_panel else 0.5
        iou = self._model_panel.get_iou_threshold() if self._model_panel else 0.45
        class_match_mode = self._model_panel.get_class_match_mode() if self._model_panel else "class_id"

        self._batch_worker = BatchPredictWorker(
            predictor=self._model_ctrl.predictor,
            image_paths=target_images,
            conf=conf, iou=iou,
            project_classes=self._project.config.classes,
            class_match_mode=class_match_mode,
            task=self._project.config.task_type,
        )
        self._batch_skipped = 0
        self._batch_failed = 0
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.image_done.connect(self._on_batch_image_done)
        self._batch_worker.finished_ok.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._on_batch_worker_finished)
        if self._project.config.task_type == "classify":
            self._label_panel.begin_bulk_auto_label_update()
        self._batch_worker.start()

        self._batch_dialog = BatchProgressDialog("批量自动标注", len(target_images), self)
        self._batch_dialog.cancelled.connect(self._batch_worker.cancel)
        self._batch_dialog.show()
        self._status_label.setText(f"批量标注进行中: 0/{len(target_images)}")

    def _on_batch_progress(self, current: int, total: int) -> None:
        self._status_label.setText(f"批量标注进行中: {current}/{total}")
        if self._batch_dialog:
            self._batch_dialog.update_progress(current, total)

    def _on_batch_image_done(self, path_str: str, payload, img_size) -> None:
        if not self._project:
            return
        img_path = Path(path_str)
        if self._project.config.task_type == "classify":
            if payload is None:
                self._batch_failed += 1
                return
            if not self._label_panel:
                return
            raw_name, conf = payload
            # Worker returns raw class names (filter_to_project=False); names
            # not in project.classes (either user did not approve in pre-dialog
            # or auto_register is OFF) are skipped here. We deliberately do not
            # call register_auto_class — that would mutate project state from a
            # non-GUI context if/when this slot is ever invoked off the main
            # thread.
            if raw_name not in self._project.config.classes:
                self._batch_skipped += 1
                return
            applied = self._label_panel.add_auto_class_prediction(
                img_path, raw_name, conf,
            )
            if not applied:
                self._batch_skipped += 1
            return
        annotations = payload
        label_path = self._project.label_path_for(img_path)
        ia = load_annotation(label_path)
        if ia is None:
            ia = ImageAnnotation(image_path=img_path.name, image_size=img_size)
        # Filter out predictions that overlap with existing confirmed annotations
        from src.core.annotation import find_conflicts
        overlap_iou = self._model_panel.get_overlap_iou_threshold() if self._model_panel else 0.5
        _, clean_preds = find_conflicts(ia.annotations, annotations, overlap_iou)
        for ann in clean_preds:
            ia.annotations.append(ann)
        from src.core.label_io import save_annotation
        save_annotation(ia, label_path)
        if self._label_panel and self._label_panel._view is not None:
            self._label_panel._view._file_list.set_status(img_path, ia.status)

    def _on_batch_finished(self) -> None:
        skipped = getattr(self, "_batch_skipped", 0)
        failed = getattr(self, "_batch_failed", 0)
        notes = []
        if skipped > 0:
            notes.append(f"跳过 {skipped} 张已确认")
        if failed > 0:
            notes.append(f"失败 {failed} 张未识别")
        if notes:
            self._status_label.setText("批量自动标注完成（" + "，".join(notes) + "）")
        else:
            self._status_label.setText("批量自动标注完成")
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        if self._label_panel and self._label_panel.get_current_image_path():
            self._label_panel._view.reload_current()

    def _on_batch_error(self, msg: str) -> None:
        self._status_label.setText("批量标注失败")
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        QMessageBox.warning(self, "批量标注失败", msg)

    def _on_batch_worker_finished(self) -> None:
        if self._label_panel:
            self._label_panel.end_bulk_auto_label_update()
        # On cancel, neither finished_ok nor error fired, so the dialog is still
        # open. Close it here and surface a status message.
        if self._batch_dialog is not None:
            cancelled = self._batch_dialog.is_cancelled
            self._batch_dialog.close()
            self._batch_dialog = None
            if cancelled:
                self._status_label.setText("批量标注已取消")

    # ── Training ─────────────────────────────────────────────

    def _on_preview_augmentation(self, params: dict) -> None:
        """Show augmentation preview dialog."""
        if not self._project or not self._label_panel:
            return
        img_path = self._label_panel.get_current_image_path()
        if not img_path:
            # Use first image in project
            images = self._project.list_images()
            if not images:
                QMessageBox.information(self, "提示", "没有可用图片")
                return
            img_path = images[0]
        from src.ui.augmentation_preview import AugmentationPreviewDialog
        dlg = AugmentationPreviewDialog(img_path, params, self)
        dlg.exec_()

    def _on_train_tag_filter_changed(self, _filt) -> None:
        self._refresh_train_tag_breakdown()

    def _refresh_train_tag_breakdown(self, *_args) -> None:
        """Recompute and push the tag-filter breakdown to TrainPanel.

        Fires from three sources: the TrainPanel's own filter widget,
        project-level tag mutations, and per-image tag mutations.
        Tolerates a missing TrainPanel (the train tab is lazily created
        on first project open — see ``_ensure_train_panel``-style block
        in the project-open path).
        """
        if self._train_panel is None:
            return
        filt = self._train_panel.get_tag_filter()
        if filt.is_empty():
            self._train_panel.set_filter_breakdown(None)
            return
        self._train_panel.set_filter_breakdown(
            self._tag_ctrl.compute_filter_breakdown(filt)
        )

    def _on_start_training(self) -> None:
        if not self._project or not self._train_panel:
            return
        # Controller-level guard: the disabled button is the primary UX signal,
        # but project switches can rebind the start button to fresh state.
        if self._train_ctrl.worker is not None and self._train_ctrl.worker.isRunning():
            QMessageBox.warning(
                self, "训练进行中",
                "已有训练任务正在运行，请等待完成或先停止。",
            )
            # Restore button state in case the click slipped through.
            self._train_panel._btn_start.setEnabled(False)
            self._train_panel._btn_stop.setEnabled(True)
            return
        # YOLO↔LocateAnything mutual exclusion: training loads CUDA in *this*
        # process, so a resident LA sidecar would contend for the same GPU.
        # Close LA first (with user confirmation). ``_on_start`` already flipped
        # the start button to the running state, so a decline must restore idle.
        if not self._confirm_disable_la_for_yolo("开始训练"):
            self._train_panel.reset_start_button_idle()
            return
        try:
            if self._label_panel:
                self._label_panel.save_and_cleanup()

            task = self._train_panel._task_combo.currentText()
            val_ratio = self._train_panel.get_val_ratio()
            kpt_shape = None
            if task == "pose":
                kpt_shape = [self._train_panel._kpt_num_spin.value(), self._train_panel._kpt_dim_spin.value()]
            data_yaml = self._train_ctrl.validate_and_prepare(
                self._project, task, val_ratio,
                kpt_shape=kpt_shape,
                tag_filter=self._train_panel.get_tag_filter(),
            )
            if data_yaml is None:
                self._train_panel._btn_start.setEnabled(True)
                self._train_panel._btn_stop.setEnabled(False)
                return

            config = self._train_panel.get_train_config(data_yaml=data_yaml)
            base_model = self._train_panel._model_combo.currentText()
            worker = self._train_ctrl.start(config, self._project, task, base_model=base_model)
            worker.epoch_update.connect(self._train_panel.update_epoch)
            worker.finished_ok.connect(self._on_training_finished)
            worker.cancelled.connect(self._on_training_cancelled)
            worker.error.connect(self._train_panel.on_training_error)

            self._train_panel.append_log(f"开始训练: {task} | {config.model} | {config.epochs} epochs")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to start training: %s", e, exc_info=True)
            self._train_panel.on_training_error(str(e))

    def _on_stop_training(self) -> None:
        self._train_ctrl.stop()
        if self._train_panel:
            self._train_panel.append_log("正在停止训练...")
            self._train_panel._btn_stop.setEnabled(False)

    def _on_training_cancelled(self) -> None:
        if self._train_panel:
            self._train_panel.on_training_cancelled()

    def _on_training_finished(self, metrics: dict) -> None:
        if self._train_panel:
            self._train_panel.on_training_finished(metrics)
        model_info = self._train_ctrl.register_model_after_training(metrics)
        if model_info is None:
            return
        project_at_start = self._train_ctrl.project_at_start
        same_project = (
            self._project is not None
            and project_at_start is not None
            and project_at_start.project_dir == self._project.project_dir
        )
        if same_project:
            # register_model_after_training wrote to disk via a fresh registry
            # bound to the snapshot project; reload our in-memory copy so the
            # model panel picks up the new entry.
            if self._model_registry is not None:
                self._model_registry.load()
            self._refresh_model_lists()
            self._on_model_load(model_info.id)
            self._status_label.setText(f"训练完成，已自动加载模型: {model_info.name}")
        else:
            proj_name = project_at_start.config.name if project_at_start else "原项目"
            self._status_label.setText(
                f"训练完成: 模型已注册到项目「{proj_name}」({model_info.name})"
            )

    # ── Lifecycle ────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        worker = self._train_ctrl.worker
        if worker is not None and worker.isRunning():
            reply = QMessageBox.question(
                self, "训练进行中",
                "训练正在进行，是否取消训练并退出？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self._train_ctrl.stop()
            worker.wait(30000)
        if self._script_tool_panel and not self._script_tool_panel.prepare_close():
            event.ignore()
            return
        # Wait for any in-flight single-image inference (slow backend) so the
        # worker isn't using the predictor while we release it below.
        if self._single_worker is not None and self._single_worker.isRunning():
            self._single_worker.wait(30000)
        # Free the LocateAnything runtime (GPU model) if it is still resident.
        if self._la_ctrl.is_active:
            self._la_ctrl.disable()
        if self._label_panel:
            self._label_panel.save_and_cleanup()
        geo = self.geometry()
        self._app_config.window_geometry = {
            "x": geo.x(), "y": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }
        if self._label_panel is not None:
            state = self._label_panel.get_annotation_panel_state()
            sizes = state.get("sizes", [])
            collapsed = state.get("collapsed", {})
            if isinstance(sizes, list):
                self._app_config.annotation_panel_splitter_sizes = [
                    int(s) for s in sizes if isinstance(s, int)
                ]
            if isinstance(collapsed, dict):
                self._app_config.annotation_panel_collapsed = {
                    str(k): bool(v) for k, v in collapsed.items()
                    if isinstance(v, bool)
                }
        self._app_config.save(self._config_path)
        super().closeEvent(event)
