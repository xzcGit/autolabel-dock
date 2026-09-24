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
from src.core.label_store import LabelStore
from src.core.autolabel import InferenceParams
from src.ui.label_panel import LabelPanel
from src.ui.train_panel import TrainPanel
from src.ui.model_panel import ModelPanel
from src.ui.script_tool_panel import ScriptToolPanel
from src.ui.dialogs import BatchProgressDialog
from src.ui.theme import set_button_role, set_surface, text_style
from src.engine.model_manager import ModelRegistry
from src.controllers.project import ProjectController
from src.controllers.model import ModelController
from src.controllers.train import TrainController
from src.controllers.tags import TagController
from src.controllers.autolabel import AutoLabelController
from src.controllers.locateanything import LocateAnythingController
from src.controllers.video_import import VideoImportController
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

        # Shared LabelStore — the single read/write face for label records.
        # Its flush callback is installed by LabelPanel's constructor
        # (save_and_cleanup, which delegates to the active view); every
        # store-mediated read flushes pending edits first.
        self._label_store = LabelStore()

        # Controllers
        self._project_ctrl = ProjectController(
            self._app_config, self._config_path, self,
            label_store=self._label_store,
        )
        self._model_ctrl = ModelController(self)
        self._train_ctrl = TrainController(self, label_store=self._label_store)
        self._tag_ctrl = TagController(self, label_store=self._label_store)
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
        # AutoLabel deep module — owns the predict → merge → persist pipeline
        # for single/batch auto-labeling. MainWindow keeps only the forwarding
        # slots and the progress-dialog shell (signals wired after the status
        # bar exists, below). Thread decision (sync YOLO vs async slow backend)
        # lives in the controller, driven by the injected predicate.
        self._autolabel_ctrl = AutoLabelController(
            self._model_ctrl,
            self._project_ctrl,
            self._label_store,
            slow_backend_active=lambda: self._la_ctrl.is_active,
            params_provider=self._collect_inference_params,
            parent_widget=self,
        )
        # Video frame import — dialog-driven extraction into the project image
        # dir. MainWindow keeps only the progress-dialog shell (below); the
        # worker lifecycle lives in the controller.
        self._video_import_ctrl = VideoImportController(self)

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
        # Progress-dialog shell for batch auto-label (workers/counters live in
        # AutoLabelController; only the dialog belongs to the window).
        self._batch_dialog: BatchProgressDialog | None = None
        # Progress-dialog shell for video frame import (same pattern).
        self._video_progress_dialog: BatchProgressDialog | None = None

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

        # AutoLabelController signal wiring (needs the status label above).
        self._autolabel_ctrl.status_message.connect(self._status_label.setText)
        self._autolabel_ctrl.batch_started.connect(self._on_autolabel_batch_started)
        self._autolabel_ctrl.batch_progress.connect(self._on_autolabel_batch_progress)
        self._autolabel_ctrl.batch_finished.connect(self._on_autolabel_batch_finished)
        self._autolabel_ctrl.classes_registered.connect(
            self._on_autolabel_classes_registered
        )
        # VideoImportController signal wiring (needs the status label above).
        self._video_import_ctrl.started.connect(self._on_video_import_started)
        self._video_import_ctrl.progress.connect(self._on_video_import_progress)
        self._video_import_ctrl.video_progress.connect(self._on_video_import_detail)
        self._video_import_ctrl.finished.connect(self._on_video_import_finished)
        self._video_import_ctrl.error.connect(self._on_video_import_error)

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

        video_import_action = QAction(icon("import"), "导入视频帧...", self)
        video_import_action.triggered.connect(self._on_import_video_frames)
        file_menu.addAction(video_import_action)

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
            # The panel installs its save_and_cleanup as the shared store's
            # flush callback (stable across view swaps).
            self._label_panel = LabelPanel(
                config_path=self._config_path,
                tag_controller=self._tag_ctrl,
                label_store=self._label_store,
                app_config=self._app_config,
            )
            self._label_panel.auto_label_single_requested.connect(self._on_auto_label_single)
            self._label_panel.auto_label_batch_requested.connect(self._on_auto_label_batch)
            self._label_panel.status_changed.connect(self._status_label.setText)
            self._label_panel.user_tags_changed.connect(self._on_label_user_tags_changed)
            self._label_panel.la_enable_requested.connect(self._on_la_enable_requested)
            self._label_panel.la_disable_requested.connect(self._la_ctrl.disable)
            self._label_panel.la_query_changed.connect(self._la_ctrl.set_query)
            self._label_panel.videos_dropped.connect(self._on_videos_dropped)
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
        # Bind the auto-label pipeline to the (new) project + panel.
        self._autolabel_ctrl.set_context(project_manager, self._label_panel)

        if self._train_panel is None:
            self._train_panel = TrainPanel()
            # The panel's own _on_start slot flips it to the running state and
            # only then emits start_requested — orchestration runs second.
            self._train_panel.start_requested.connect(self._on_start_training)
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
        self._train_panel.set_task_type(project_manager.config.task_type)
        self._model_panel.set_models(self._model_registry.list_models())
        self._train_panel.set_registered_models(self._model_registry.list_models())

        self.tab_widget.setCurrentWidget(self._label_panel)
        n_images = len(project_manager.list_images())
        self._status_label.setText(
            f"项目: {project_manager.config.name} | "
            f"图片: {n_images} | "
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
        # No explicit flush: the export read path (ProjectController.export)
        # collects records through the LabelStore, which flushes first.
        try:
            self._project_ctrl.export(self._project)
            self._status_label.setText("导出完成")
        except (OSError, ValueError, KeyError):
            pass  # Error already shown by controller

    def _on_import(self) -> None:
        if not self._project:
            return
        if self._label_panel:
            # Import is a Supersede: commit the user's pending edits BEFORE
            # the importer advances the records past the in-memory view.
            self._label_panel.save_and_cleanup()
        count = self._project_ctrl.import_annotations(self._project)
        if count is not None and count > 0:
            # Refresh label panel to show imported annotations. discard_pending:
            # disk was intentionally advanced (Supersede) — flushing the stale
            # view during teardown would overwrite the merged records.
            if self._label_panel:
                self._label_panel.set_project(self._project, discard_pending=True)
            self._status_label.setText(f"导入完成: {count} 个图片")
        elif count == 0:
            self._status_label.setText("导入完成: 无匹配图片")

    # ── Video frame import (dialog + progress-dialog shell) ────
    # Extraction runs in VideoImportController; MainWindow opens the dialog,
    # resolves the project image dir (ProjectManager.list_images semantics),
    # hosts the modal progress dialog, and rescans the file list on finish.

    def _on_import_video_frames(self) -> None:
        """文件 menu「导入视频帧...」: open the dialog with no pre-filled videos."""
        self._open_video_import_dialog(initial_videos=None)

    def _on_videos_dropped(self, paths: list) -> None:
        """Videos dropped onto the file list → dialog pre-filled with them."""
        self._open_video_import_dialog(initial_videos=[Path(p) for p in paths])

    def _open_video_import_dialog(self, initial_videos) -> None:
        if not self._project:
            return
        if self._video_import_ctrl.is_running:
            QMessageBox.information(self, "提示", "视频抽帧正在进行，请稍候。")
            return
        from src.ui.video_import_dialog import VideoImportDialog

        dlg = VideoImportDialog(self, initial_videos=initial_videos)
        if not dlg.exec_():
            return
        videos = dlg.get_videos()
        if not videos:
            return
        self._video_import_ctrl.start_import(
            videos, self._project_image_dir(), dlg.get_params(),
        )

    def _project_image_dir(self) -> Path:
        """Resolve the project image dir (relative-or-absolute, matching
        ProjectManager.list_images)."""
        img_dir = Path(self._project.config.image_dir)
        if not img_dir.is_absolute():
            img_dir = self._project.project_dir / img_dir
        return img_dir

    def _on_video_import_started(self, total: int) -> None:
        self._video_progress_dialog = BatchProgressDialog("导入视频帧", total, self)
        self._video_progress_dialog.cancelled.connect(self._video_import_ctrl.cancel)
        self._video_progress_dialog.show()

    def _on_video_import_progress(self, current: int, total: int) -> None:
        if self._video_progress_dialog:
            self._video_progress_dialog.update_progress(current, total)

    def _on_video_import_detail(self, name: str, written: int, skipped: int) -> None:
        if self._video_progress_dialog:
            self._video_progress_dialog.set_detail(
                f"{name}: 已写入 {written} / 跳过 {skipped}"
            )

    def _on_video_import_finished(self, summary: str) -> None:
        """Terminal for every outcome (completed / failed / cancelled)."""
        if self._video_progress_dialog:
            self._video_progress_dialog.close()
            self._video_progress_dialog = None
        self._status_label.setText(summary)
        # New frames are plain images — rescan the file list (F5 path). No
        # label records were touched, so this needs no Supersede flush/reload.
        if self._label_panel:
            self._label_panel.rescan_images()

    def _on_video_import_error(self, message: str) -> None:
        QMessageBox.warning(self, "视频抽帧失败", message)

    def _on_class_manager(self) -> None:
        if not self._project:
            return
        if self._project_ctrl.manage_classes(self._project):
            if self._label_panel:
                # Default set_project flushes the old view's pending edit
                # before teardown — the rebuild must not drop canvas edits.
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

    # ── Auto-label (forwarding slots + progress-dialog shell) ──
    # Orchestration lives in AutoLabelController (src/controllers/autolabel.py):
    # task_type dispatch, sync/async thread decision, conflict merging and
    # LabelStore persistence. MainWindow only forwards the toolbar intents and
    # hosts the modal progress dialog.

    def _on_auto_label_single(self) -> None:
        self._autolabel_ctrl.label_current()

    def _on_auto_label_batch(self) -> None:
        self._autolabel_ctrl.label_batch()

    def _collect_inference_params(self) -> InferenceParams:
        """Assemble inference thresholds from ModelPanel's public getters.

        The panel is created lazily on first project open; before that the
        ``InferenceParams`` defaults mirror the historical fallbacks.
        """
        if not self._model_panel:
            return InferenceParams()
        return InferenceParams(
            conf=self._model_panel.get_conf_threshold(),
            iou=self._model_panel.get_iou_threshold(),
            overlap_iou=self._model_panel.get_overlap_iou_threshold(),
            class_match_mode=self._model_panel.get_class_match_mode(),
        )

    def _on_autolabel_batch_started(self, total: int) -> None:
        self._batch_dialog = BatchProgressDialog("批量自动标注", total, self)
        self._batch_dialog.cancelled.connect(self._autolabel_ctrl.cancel_batch)
        self._batch_dialog.show()

    def _on_autolabel_batch_progress(self, current: int, total: int) -> None:
        if self._batch_dialog:
            self._batch_dialog.update_progress(current, total)

    def _on_autolabel_batch_finished(self, summary: str) -> None:
        """Terminal for every batch outcome (completed / failed / cancelled)."""
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        self._status_label.setText(summary)

    def _on_autolabel_classes_registered(self) -> None:
        """New classes registered from model output → rebuild class widgets.

        Supersede-adjacent rebuild: classify writes through (nothing pending
        to flush), and the records/classes on disk were advanced behind the
        view — discard pending instead of flushing stale memory over them.
        """
        if self._label_panel and self._project:
            self._label_panel.set_project(self._project, discard_pending=True)

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
            self._train_panel.set_running_state(True)
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
                # Keep this explicit flush: downstream DatasetPreparer reads
                # label files raw (outside the LabelStore) by design.
                self._label_panel.save_and_cleanup()

            req = self._train_panel.get_train_request()
            data_yaml = self._train_ctrl.validate_and_prepare(
                self._project, req.task, req.val_ratio,
                kpt_shape=req.kpt_shape,
                tag_filter=req.tag_filter,
            )
            if data_yaml is None:
                self._train_panel.set_running_state(False)
                return

            config = self._train_panel.get_train_config(data_yaml=data_yaml)
            worker = self._train_ctrl.start(config, self._project, req.task, base_model=req.base_model)
            worker.epoch_update.connect(self._train_panel.update_epoch)
            worker.finished_ok.connect(self._on_training_finished)
            worker.cancelled.connect(self._on_training_cancelled)
            worker.error.connect(self._train_panel.on_training_error)

            self._train_panel.append_log(f"开始训练: {req.task} | {config.model} | {config.epochs} epochs")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to start training: %s", e, exc_info=True)
            self._train_panel.on_training_error(str(e))

    def _on_stop_training(self) -> None:
        self._train_ctrl.stop()
        if self._train_panel:
            # The panel's _on_stop already disabled the stop button before
            # emitting stop_requested — only the log line belongs here.
            self._train_panel.append_log("正在停止训练...")

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
        self._autolabel_ctrl.shutdown(30000)
        # Stop any in-flight video frame extraction (cancel is checked per
        # frame, so this returns quickly; written frames stay on disk).
        self._video_import_ctrl.shutdown(30000)
        # Free the LocateAnything runtime (GPU model) if it is still resident.
        if self._la_ctrl.is_active:
            self._la_ctrl.disable()
        if self._label_panel:
            # Keep this explicit flush: persist pending edits before exit —
            # no store-mediated read runs after this point.
            self._label_panel.save_and_cleanup()
            # Release view-held background workers (SAM assist / thumbnails /
            # label scan) so they never outlive their owning widgets.
            self._label_panel.shutdown_view()
        if self._train_panel is not None:
            # Wait out the async device probe while the panel is still alive
            # (a straggling QThread aborts process teardown on quick exits).
            self._train_panel.shutdown()
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
