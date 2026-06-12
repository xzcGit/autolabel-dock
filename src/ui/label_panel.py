"""Label panel shell — hosts a task-specific view based on project task_type.

The shell owns project state, the undo-stack pool, the image cache, and the
shared toolbar (auto-label, batch, undo/redo, save, filters). The actual UI
for editing annotations lives inside a `TaskView` instance: `DetectPoseView`
for detect/pose projects and `ClassifyView` for classify projects.
"""
from __future__ import annotations

import logging
import shutil
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QToolBar,
    QToolButton,
    QLabel,
    QComboBox,
)

from src.core.label_io import load_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.ui.icons import icon
from src.ui.locateanything_bar import LocateAnythingBar
from src.ui.tag_widget import TagApplyBar, TagFilterBar
from src.ui.theme import set_button_role
from src.ui.views.base import TaskView
from src.utils.image import ImageCache
from src.utils.undo import UndoStack

logger = logging.getLogger(__name__)


class LabelPanel(QWidget):
    """Main annotation workspace shell.

    Layout:
        +--------------------------------------------------+
        | shared toolbar (auto/batch/undo/redo/save/filter)|
        +--------------------------------------------------+
        | refresh button strip                              |
        +--------------------------------------------------+
        | inner view (DetectPoseView or ClassifyView)       |
        +--------------------------------------------------+
    """

    auto_label_single_requested = pyqtSignal()
    auto_label_batch_requested = pyqtSignal()
    status_changed = pyqtSignal(str)
    # Bubbled from the active view — MainWindow listens to sync new tags
    # into the project tag registry via TagController.
    user_tags_changed = pyqtSignal(object, list)
    # LocateAnything toolbar — forwarded to MainWindow → LocateAnythingController.
    la_enable_requested = pyqtSignal()
    la_disable_requested = pyqtSignal()
    la_query_changed = pyqtSignal(str, object)  # (prompt, target_class | None)

    _UNDO_MAX_IMAGES = 20

    def __init__(self, config_path=None, parent=None, tag_controller=None):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._undo_stacks: "OrderedDict[str, UndoStack]" = OrderedDict()
        self._image_cache = ImageCache(max_count=16, max_memory_mb=512.0)
        self._config_path = config_path
        self._view: TaskView | None = None
        self._tag_ctrl = tag_controller  # may be None during isolated UI tests
        self._pending_ann_panel_state: dict = {}

        self._init_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)

        self._btn_auto_single = QPushButton(icon("auto_label"), "自动标注")
        self._btn_auto_single.setToolTip("对当前图片执行自动标注 (Shift+A)")
        set_button_role(self._btn_auto_single, "primary")
        self._toolbar.addWidget(self._btn_auto_single)

        self._btn_auto_batch = QPushButton(icon("batch"), "批量标注")
        self._btn_auto_batch.setToolTip("对多张图片执行批量自动标注 (Ctrl+Shift+A)")
        set_button_role(self._btn_auto_batch, "secondary")
        self._toolbar.addWidget(self._btn_auto_batch)

        # LocateAnything text-labeling bar (collapsed to one button until enabled).
        self._la_bar = LocateAnythingBar()
        self._toolbar.addWidget(self._la_bar)

        self._toolbar.addSeparator()

        self._btn_undo = QPushButton(icon("undo"), "")
        self._btn_undo.setToolTip("撤销 (Ctrl+Z)")
        self._btn_undo.setFixedWidth(36)
        set_button_role(self._btn_undo, "icon")
        self._toolbar.addWidget(self._btn_undo)

        self._btn_redo = QPushButton(icon("redo"), "")
        self._btn_redo.setToolTip("重做 (Ctrl+Y)")
        self._btn_redo.setFixedWidth(36)
        set_button_role(self._btn_redo, "icon")
        self._toolbar.addWidget(self._btn_redo)

        self._btn_save = QPushButton(icon("save"), "")
        self._btn_save.setToolTip("保存 (Ctrl+S)")
        self._btn_save.setFixedWidth(36)
        set_button_role(self._btn_save, "icon")
        self._toolbar.addWidget(self._btn_save)

        self._toolbar.addSeparator()

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["全部", "已确认", "待确认", "未标注"])
        self._filter_combo.setMinimumWidth(80)
        self._toolbar.addWidget(QLabel(" 筛选: "))
        self._toolbar.addWidget(self._filter_combo)

        self._class_filter_combo = QComboBox()
        self._class_filter_combo.addItem("所有类别")
        self._class_filter_combo.setMinimumWidth(80)
        self._toolbar.addWidget(QLabel(" 类别: "))
        self._toolbar.addWidget(self._class_filter_combo)

        self._tag_filter_bar = TagFilterBar()
        self._toolbar.addWidget(self._tag_filter_bar)

        layout.addWidget(self._toolbar)

        # Tag apply strip — chip-click + T key for batch tagging.
        self._apply_bar = TagApplyBar()
        layout.addWidget(self._apply_bar)

        # Refresh strip
        refresh_strip = QWidget()
        rs_layout = QHBoxLayout(refresh_strip)
        rs_layout.setContentsMargins(4, 2, 4, 2)
        rs_layout.setSpacing(4)
        self._refresh_btn = QToolButton()
        self._refresh_btn.setIcon(icon("refresh"))
        self._refresh_btn.setToolTip("刷新图像列表 (F5)")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        rs_layout.addWidget(self._refresh_btn)
        rs_layout.addStretch(1)
        layout.addWidget(refresh_strip)

        # View container — holds whichever TaskView we route to
        self._view_container = QWidget()
        self._view_layout = QVBoxLayout(self._view_container)
        self._view_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view_container, 1)

    def _connect_signals(self) -> None:
        self._btn_auto_single.clicked.connect(self.auto_label_single_requested.emit)
        self._btn_auto_batch.clicked.connect(self.auto_label_batch_requested.emit)
        self._btn_undo.clicked.connect(self.undo)
        self._btn_redo.clicked.connect(self.redo)
        self._btn_save.clicked.connect(self.save_and_cleanup)
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self._class_filter_combo.currentTextChanged.connect(self._on_class_filter_changed)
        self._tag_filter_bar.filter_changed.connect(self._on_tag_filter_changed)

        # Forward LocateAnything bar signals out to MainWindow.
        self._la_bar.enable_requested.connect(self.la_enable_requested.emit)
        self._la_bar.disable_requested.connect(self.la_disable_requested.emit)
        self._la_bar.query_changed.connect(self.la_query_changed.emit)

        if self._tag_ctrl is not None:
            self._tag_ctrl.image_tags_changed.connect(
                self._on_external_image_tags_changed
            )

    # ── Project routing ────────────────────────────────────────

    def set_project(self, project: ProjectManager) -> None:
        """Load a project and switch to the appropriate view by task_type."""
        self._project = project
        self._undo_stacks.clear()

        # Tear down previous view
        if self._view is not None:
            if hasattr(self._view, "cleanup"):
                self._view.cleanup()
            self._view_layout.removeWidget(self._view)
            self._view.setParent(None)
            self._view.deleteLater()
            self._view = None

        # Construct new view based on task_type
        from src.ui.views.detect_pose import DetectPoseView
        from src.ui.views.classify import ClassifyView

        task_type = project.config.task_type
        if task_type == "classify":
            self._view = ClassifyView(self._image_cache, self._undo_stacks)
        else:
            self._view = DetectPoseView(self._image_cache, self._undo_stacks)
        self._view_layout.addWidget(self._view)

        # Push cached AnnotationPanel state into the new view (no-op for classify).
        if hasattr(self._view, "_ann_panel") and self._pending_ann_panel_state:
            self._view._ann_panel.restore_state(self._pending_ann_panel_state)

        # Wire view → shell signals
        self._view.status_changed.connect(self.status_changed.emit)
        self._view.images_dropped.connect(self._on_images_dropped)
        self._view.classes_changed.connect(self._refresh_class_filter)
        self._view.user_tags_changed.connect(self._on_view_user_tags_changed)

        # Push project state to the view
        colors = {cls: project.config.get_class_color(cls) for cls in project.config.classes}
        self._view.set_class_colors(colors)
        self._view.set_classes(project.config.classes)
        self._view.set_project(project)

        # Refresh class filter combo
        self._refresh_class_filter()
        self._apply_bar.clear_armed()
        self.refresh_available_tags()
        # Sync project classes into the LocateAnything target dropdown.
        self._la_bar.set_classes(project.config.classes)

        self._refresh_btn.setEnabled(True)
        logger.info("Project loaded: %s (task=%s)", project.config.name, task_type)

    def _refresh_class_filter(self) -> None:
        """Rebuild the class filter combo from the current project's classes.

        Preserves the current selection if still present; otherwise reverts to
        「所有类别」.
        """
        if self._project is None:
            return
        prev = self._class_filter_combo.currentText()
        self._class_filter_combo.blockSignals(True)
        self._class_filter_combo.clear()
        self._class_filter_combo.addItem("所有类别")
        for cls in self._project.config.classes:
            self._class_filter_combo.addItem(cls)
        idx = self._class_filter_combo.findText(prev)
        self._class_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._class_filter_combo.blockSignals(False)

    # ── Forwarders to current view ─────────────────────────────

    def undo(self) -> None:
        if self._view:
            self._view.undo()

    def redo(self) -> None:
        if self._view:
            self._view.redo()

    def save_and_cleanup(self) -> None:
        if self._view:
            self._view.commit_pending_save()

    # ── AnnotationPanel state persistence ──────────────────────

    def set_annotation_panel_state(self, state: dict) -> None:
        """Cache state to apply to the next-mounted AnnotationPanel.

        Pushes immediately if a view is already active and that view
        embeds an AnnotationPanel (detect/pose views only; classify
        views are no-ops).
        """
        if not isinstance(state, dict):
            return
        self._pending_ann_panel_state = dict(state)
        if self._view is not None and hasattr(self._view, "_ann_panel"):
            self._view._ann_panel.restore_state(state)

    def get_annotation_panel_state(self) -> dict:
        """Return the live AnnotationPanel state, or the last-cached
        pending state if no view is mounted (or the active view has no
        AnnotationPanel — e.g., classify)."""
        if self._view is not None and hasattr(self._view, "_ann_panel"):
            return self._view._ann_panel.save_state()
        return dict(self._pending_ann_panel_state)

    def get_current_image_path(self) -> Path | None:
        return self._view.get_focused_image() if self._view else None

    def add_auto_annotations(self, anns, overlap_iou: float = 0.5) -> None:
        if self._view:
            self._view.add_auto_annotations(anns, overlap_iou)

    def add_auto_class_prediction(self, path, class_name: str, confidence: float) -> bool:
        if self._view:
            return bool(self._view.add_auto_class_prediction(path, class_name, confidence))
        return False

    def begin_bulk_auto_label_update(self) -> None:
        if self._view is None:
            return
        begin = getattr(self._view, "begin_bulk_auto_label_update", None)
        if callable(begin):
            begin()

    def end_bulk_auto_label_update(self) -> None:
        if self._view is None:
            return
        end = getattr(self._view, "end_bulk_auto_label_update", None)
        if callable(end):
            end()

    def get_unlabeled_image_paths(self) -> list[Path]:
        """Project-wide list of unlabeled images, semantics depend on task_type."""
        if not self._project:
            return []
        result: list[Path] = []
        is_classify = self._project.config.task_type == "classify"
        for img_path in self._project.list_images():
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                result.append(img_path)
                continue
            if is_classify:
                if not ia.image_tags:
                    result.append(img_path)
            else:
                if not ia.annotations:
                    result.append(img_path)
        return result

    # ── Filter combo handlers ─────────────────────────────────

    def _on_filter_changed(self, text: str) -> None:
        mapping = {"全部": None, "已确认": "confirmed", "待确认": "pending", "未标注": "unlabeled"}
        if self._view:
            self._view.set_filter(mapping.get(text))

    def _on_class_filter_changed(self, text: str) -> None:
        cls = None if text == "所有类别" else text
        if self._view:
            self._view.set_class_filter(cls)

    def _on_tag_filter_changed(self, tag_filter) -> None:
        if self._view:
            self._view.set_tag_filter(tag_filter)

    def refresh_available_tags(self) -> None:
        """Push the current project's tag registry into all tag-aware widgets."""
        if self._project is None:
            return
        tags = list(self._project.config.tags)
        self._tag_filter_bar.set_available_tags(tags)
        self._apply_bar.set_available_tags(tags)
        if self._view is not None:
            self._view.set_available_tags(tags)

    def get_tag_filter(self) -> TagFilter:
        """Current tag filter as selected in the toolbar bar."""
        return self._tag_filter_bar.current_filter()

    # ── LocateAnything bar — public forwarding API ────────────────────────

    def set_la_enabled_state(self, enabled: bool) -> None:
        """Toggle the LA bar between collapsed button and expanded controls."""
        self._la_bar.set_enabled_state(enabled)

    def set_la_feature_visible(self, visible: bool) -> None:
        """Fully show/hide the LA toolbar (experimental master switch)."""
        self._la_bar.set_feature_visible(visible)

    def set_la_status(self, message: str) -> None:
        """Show a transient status string on the LA bar (during load)."""
        self._la_bar.set_status(message)

    def get_la_query(self) -> tuple[str, object]:
        """Return the LA bar's (prompt, target_class | None)."""
        return self._la_bar.get_query()

    def set_auto_label_busy(self, busy: bool) -> None:
        """Disable/enable the auto-label buttons during background inference.

        Prevents the user from stacking overlapping inference requests while a
        slow backend (LocateAnything) runs on a worker thread. The LA bar shows
        a transient status during the busy window.
        """
        self._btn_auto_single.setEnabled(not busy)
        self._btn_auto_batch.setEnabled(not busy)
        if busy:
            self.set_la_status("正在标注当前图片…")
        else:
            self.set_la_status("")

    def _on_view_user_tags_changed(self, path, tags) -> None:
        """Bubble the view's per-image tag edits up to MainWindow."""
        self.user_tags_changed.emit(path, list(tags))

    def _on_external_image_tags_changed(self, path: Path, tags: list[str]) -> None:
        """Controller-originated write — push refresh into the active view."""
        if self._view is not None:
            self._view.refresh_image_tags(path, tags)

    def _apply_armed_tag(self) -> None:
        if self._project is None or self._view is None or self._tag_ctrl is None:
            return
        tag = self._apply_bar.get_armed()
        if not tag:
            self.status_changed.emit("先点 tag chip 装载，再按 T")
            return
        paths = self._view.get_selected_image_paths()
        if not paths:
            self.status_changed.emit("先选择图片，再按 T")
            return
        self._view.commit_pending_save()
        n = self._tag_ctrl.apply_tag_to_images(tag, paths)
        self.status_changed.emit(f"已为 {n}/{len(paths)} 张图加上 tag: {tag}")

    # ── Shell-level keyboard shortcuts ────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key_A and (mod & Qt.ControlModifier) and (mod & Qt.ShiftModifier):
            self.auto_label_batch_requested.emit()
        elif key == Qt.Key_A and (mod & Qt.ShiftModifier):
            self.auto_label_single_requested.emit()
        elif key == Qt.Key_Z and mod & Qt.ControlModifier:
            self.undo()
        elif key == Qt.Key_Y and mod & Qt.ControlModifier:
            self.redo()
        elif key == Qt.Key_S and mod & Qt.ControlModifier:
            self.save_and_cleanup()
        elif key == Qt.Key_F5:
            self.rescan_images()
        elif key == Qt.Key_T and not (mod & (
            Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier
        )):
            self._apply_armed_tag()
        else:
            super().keyPressEvent(event)

    # ── Refresh / rescan / drop ────────────────────────────────

    def rescan_images(self) -> int:
        """Re-scan project image dir; refresh current view if new files found."""
        if not self._project or self._view is None:
            return 0
        current = {str(p) for p in self._view.get_all_paths()}
        latest = self._project.list_images()
        added = len({str(p) for p in latest} - current)
        if added:
            self._refresh_image_list(latest)
        return added

    def _refresh_image_list(self, images: list[Path]) -> None:
        if self._view is None:
            return
        # Detect/Pose view exposes refresh_image_list; classify view rebuilds via set_project
        refresh = getattr(self._view, "refresh_image_list", None)
        if callable(refresh):
            refresh(images)
        else:
            # Fallback: re-set project to re-populate view
            if self._project:
                self._view.set_project(self._project)

    def _on_refresh_clicked(self) -> None:
        n = self.rescan_images()
        if n > 0:
            self.status_changed.emit(f"发现 {n} 张新图片")
        else:
            self.status_changed.emit("未发现新图片")

    def _on_images_dropped(self, paths: list[Path]) -> None:
        if not self._project:
            return
        image_dir = self._project.project_dir / self._project.config.image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for src in paths:
            dst = image_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                added += 1
        if added > 0:
            images = self._project.list_images()
            self._refresh_image_list(images)
            self.status_changed.emit(f"已导入 {added} 张图片")
