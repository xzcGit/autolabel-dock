"""Detect/Pose view — extracted from LabelPanel for the shell/view refactor.

Behavior matches the pre-refactor LabelPanel for detect/pose projects.
The shell (LabelPanel) supplies the shared `image_cache` and `_undo_stacks`
dictionary so view switches preserve cache reuse but per-view undo state.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QToolBar,
    QToolButton,
    QLabel,
    QMessageBox,
)

from src.core.annotation import Annotation, ImageAnnotation
from src.core.autolabel import merge_predictions
from src.core.label_store import LabelStore
from src.core.project import ProjectManager
from src.ui.canvas import AnnotationCanvas
from src.ui.file_list import FileListWidget
from src.ui.properties import AnnotationPanel
from src.ui.class_picker import ClassPickerPopup, KeypointLabelPicker
from src.ui.icons import icon
from src.ui.theme import set_button_role
from src.ui.views.base import TaskView
from src.utils.image import get_image_size, ImageCache
from src.utils.undo import UndoStack

logger = logging.getLogger(__name__)


class DetectPoseView(TaskView):
    """Detect/Pose UI: drawing tools + file list + canvas + annotation panel."""

    _UNDO_MAX_IMAGES = 20

    # Shell-level signals exposed for backwards compat with MainWindow wiring
    batch_confirm_visible_requested = pyqtSignal()
    batch_revert_visible_requested = pyqtSignal()

    def __init__(
        self,
        image_cache: ImageCache,
        undo_stacks: "OrderedDict[str, UndoStack]",
        label_store: LabelStore | None = None,
        sam_controller=None,
        parent=None,
    ):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._current_image_path: Path | None = None
        self._current_annotation: ImageAnnotation | None = None
        self._image_cache = image_cache
        self._undo_stacks = undo_stacks
        # SAM assist controller — lazily constructed on first SAM tool
        # activation (keeps startup free of any model cost); ``sam_controller``
        # is a test injection point (a fake avoids ultralytics entirely).
        self._sam_ctrl = sam_controller
        self._sam_wired = False
        # Shared LabelStore (injected by the shell) — reads flush pending
        # edits first. A private, callback-less store keeps direct view
        # construction (tests) on plain label IO semantics.
        self._store = label_store or LabelStore()
        self._last_class: str | None = None
        self._clipboard: list[dict] | None = None
        self._stats_cache: dict = {}
        self._prev_annotations_snapshot: list[tuple] | None = None
        # Full-record snapshot of the current image as of its last disk
        # load/save. _save_current skips the disk write when the would-be
        # record equals this baseline (the flush callback must be cheap and
        # idempotent when clean — store reads in scan loops trigger it once
        # per image). None means "unknown → must write".
        self._last_saved_record: dict | None = None
        self._scan_worker = None

        self._init_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # View-local toolbar: drawing tools + per-image / per-visible confirm actions
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)

        self._btn_select = QPushButton(icon("cursor"), "移动")
        self._btn_select.setCheckable(True)
        self._btn_select.setChecked(True)
        self._btn_select.setToolTip("选择/移动工具 (V)")
        set_button_role(self._btn_select, "secondary")
        self._btn_bbox = QPushButton(icon("bbox"), "矩形框")
        self._btn_bbox.setCheckable(True)
        self._btn_bbox.setToolTip("绘制矩形框 (W)")
        set_button_role(self._btn_bbox, "secondary")
        self._btn_keypoint = QPushButton(icon("keypoint"), "关键点")
        self._btn_keypoint.setCheckable(True)
        self._btn_keypoint.setToolTip("绘制关键点 (K)")
        set_button_role(self._btn_keypoint, "secondary")
        self._btn_polygon = QPushButton(icon("polygon"), "多边形")
        self._btn_polygon.setCheckable(True)
        self._btn_polygon.setToolTip("绘制多边形 (P) — 逐点点击，双击或回车闭合，Esc 取消")
        set_button_role(self._btn_polygon, "secondary")
        self._btn_sam = QPushButton(icon("auto_label"), "SAM 辅助")
        self._btn_sam.setCheckable(True)
        self._btn_sam.setToolTip("SAM 辅助标注 — 单击或拉框自动生成多边形（CPU 推理）")
        set_button_role(self._btn_sam, "secondary")

        for btn in [
            self._btn_select, self._btn_bbox, self._btn_keypoint,
            self._btn_polygon, self._btn_sam,
        ]:
            btn.setMinimumWidth(80)
            self._toolbar.addWidget(btn)

        self._toolbar.addSeparator()

        self._btn_confirm_all = QPushButton(icon("check_all"), "全部确认")
        self._btn_confirm_all.setToolTip("确认当前图片所有标注 (Ctrl+Space)")
        set_button_role(self._btn_confirm_all, "secondary")
        self._toolbar.addWidget(self._btn_confirm_all)

        self._btn_confirm_visible = QPushButton(icon("confirm_visible"), "确认可见预标注")
        self._btn_confirm_visible.setToolTip("确认当前可见图片的所有未确认标注")
        set_button_role(self._btn_confirm_visible, "primary")
        self._toolbar.addWidget(self._btn_confirm_visible)

        self._btn_revert_visible = QPushButton(icon("revert_visible"), "撤销可见预标注")
        self._btn_revert_visible.setToolTip("删除当前可见图片的所有未确认标注")
        set_button_role(self._btn_revert_visible, "danger")
        self._toolbar.addWidget(self._btn_revert_visible)

        layout.addWidget(self._toolbar)

        # Splitter: file_list | canvas | properties
        self._splitter = QSplitter(Qt.Horizontal)

        self._file_list = FileListWidget()
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.addWidget(self._file_list, 1)
        left_pane.setMaximumWidth(250)
        self._splitter.addWidget(left_pane)

        self._canvas = AnnotationCanvas()
        self._splitter.addWidget(self._canvas)

        self._ann_panel = AnnotationPanel()
        self._ann_panel.setMaximumWidth(280)
        self._splitter.addWidget(self._ann_panel)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([200, 800, 250])

        layout.addWidget(self._splitter, 1)

    def _connect_signals(self) -> None:
        # Tool buttons
        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_bbox.clicked.connect(lambda: self._set_tool("draw_bbox"))
        self._btn_keypoint.clicked.connect(lambda: self._set_tool("draw_keypoint"))
        self._btn_polygon.clicked.connect(lambda: self._set_tool("draw_polygon"))
        self._btn_sam.clicked.connect(lambda: self._set_tool("sam_prompt"))

        # File list
        self._file_list.image_selected.connect(self._on_image_selected)
        self._file_list.images_dropped.connect(self.images_dropped.emit)
        self._file_list.videos_dropped.connect(self.videos_dropped.emit)
        self._file_list.batch_confirm_requested.connect(self._on_batch_confirm)
        self._file_list.batch_delete_requested.connect(self._on_batch_delete)
        self._file_list.delete_images_requested.connect(self._on_delete_images)

        # Canvas signals
        self._canvas.annotation_selected.connect(self._on_annotation_selected)
        self._canvas.annotation_created.connect(self._on_annotation_created)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)
        self._canvas.annotation_deleted.connect(self._on_annotation_deleted)
        self._canvas.class_requested.connect(self._on_class_requested)
        self._canvas.class_change_requested.connect(self._on_class_change_requested)
        self._canvas.annotations_changed.connect(self._on_annotations_changed)
        self._canvas.annotation_copied.connect(self._on_annotation_copied)
        self._canvas.keypoint_attach_requested.connect(self._on_keypoint_attach_requested)
        self._canvas.keypoint_selected.connect(self._ann_panel.select_keypoint)
        self._canvas.sam_point_requested.connect(self._on_sam_point_requested)
        self._canvas.sam_box_requested.connect(self._on_sam_box_requested)

        # Properties panel
        self._ann_panel.annotation_clicked.connect(self._canvas.select_annotation)
        self._ann_panel.keypoint_clicked.connect(self._on_panel_keypoint_clicked)
        self._ann_panel.keypoint_rename_requested.connect(self._on_keypoint_rename)
        self._ann_panel.keypoint_visibility_requested.connect(self._on_keypoint_visibility)
        self._ann_panel.keypoint_delete_requested.connect(self._on_keypoint_delete)
        self._ann_panel.default_class_changed.connect(self._on_default_class_changed)
        self._ann_panel.image_user_tags_changed.connect(self._on_user_tags_edited)

        # Confirm buttons
        self._btn_confirm_all.clicked.connect(self._confirm_all)
        self._btn_confirm_visible.clicked.connect(self._batch_confirm_visible)
        self._btn_revert_visible.clicked.connect(self._batch_revert_visible)

    # ── TaskView protocol ──────────────────────────────────────

    def set_project(self, project: ProjectManager) -> None:
        self._project = project
        self._current_image_path = None
        self._current_annotation = None
        self._canvas.clear()

        # Show drawing tools by task_type (pose hides keypoint button only via task_type="detect")
        self._btn_keypoint.setVisible(project.config.task_type == "pose")
        self._btn_polygon.setVisible(project.config.task_type == "segment")
        self._btn_sam.setVisible(project.config.task_type == "segment")

        images = project.list_images()
        # Populate the file list immediately so project open never blocks on
        # label IO; per-image statuses/classes/tags arrive when the
        # background scan finishes (large projects: thousands of JSON reads).
        # Empty maps reset caches left over from a previous project.
        self._file_list.set_image_paths(images, statuses={}, classes={}, tags={})
        if images:
            self._file_list.setCurrentRow(0)
        stats = self._new_stats(len(images))
        self._stats_cache = stats
        self._ann_panel.set_project_stats(stats)
        self._start_label_scan(project)
        logger.info("DetectPoseView loaded: %s (%d images)", project.config.name, len(images))

    # ── Background label scan (project open) ───────────────────

    def _start_label_scan(self, project: ProjectManager) -> None:
        self._stop_label_scan()
        from src.ui.views.label_scan_worker import LabelScanWorker, scan_labels
        # No flush here: LabelPanel.set_project flushed before building this
        # view, and the flush callback touches Qt widgets (never off-thread).
        if os.environ.get("AUTOLABEL_SYNC_SCAN"):  # test seam: deterministic tests
            self._on_label_scan_finished(*scan_labels(project, self._store.load_unflushed))
            return
        worker = LabelScanWorker(project, self._store.load_unflushed, self)
        worker.scan_done.connect(self._on_label_scan_finished)
        self._scan_worker = worker
        worker.start()

    def _stop_label_scan(self) -> None:
        if self._scan_worker is not None:
            self._scan_worker.stop()
            self._scan_worker.wait(2000)
            self._scan_worker = None

    def _on_label_scan_finished(self, project, statuses, classes, tags, records) -> None:
        # Ignore a stale scan that outlived a project switch/teardown.
        if project is not self._project:
            return
        # Merge in place: keeps the current row, and edits made while the
        # scan ran win over its older snapshot.
        self._file_list.merge_metadata(statuses, classes, tags)
        stats = self._new_stats(len(project.list_images()))
        for ia in records.values():
            self._accumulate_stats(stats, ia)
        self._stats_cache = stats
        self._ann_panel.set_project_stats(stats)

    def set_class_colors(self, colors: dict[str, str]) -> None:
        self._canvas.set_class_colors(colors)
        self._ann_panel.set_class_colors(colors)

    def set_classes(self, classes: list[str]) -> None:
        self._ann_panel.set_classes(classes)

    def set_available_tags(self, tags: list[str]) -> None:
        """Push the project's known-tag registry into the AnnotationPanel chip popup."""
        self._ann_panel.set_available_tags(tags)

    def set_filter(self, status: str | None) -> None:
        self._file_list.set_filter(status)

    def set_class_filter(self, cls: str | None) -> None:
        self._file_list.set_class_filter(cls)

    def set_tag_filter(self, tag_filter) -> None:
        self._file_list.set_tag_filter(tag_filter)

    def get_selected_image_paths(self) -> list[Path]:
        return list(self._file_list.get_selected_paths())

    def refresh_image_tags(self, path: Path, tags: list[str]) -> None:
        self._file_list.set_image_tags(path, set(tags))
        if path == self._current_image_path:
            self._ann_panel.set_image_user_tags(list(tags))
            # Sync to memory so next _save_current() doesn't overwrite disk
            if self._current_annotation is not None:
                self._current_annotation.tags = list(tags)

    def set_image_status(self, path: Path, status: str) -> None:
        """External writer (batch auto-label) persisted a record for ``path``;
        mirror the new status into the file list."""
        self._file_list.set_status(path, status)

    def save_annotation_panel_state(self) -> dict | None:
        """Snapshot the embedded AnnotationPanel's splitter/collapse state."""
        return self._ann_panel.save_state()

    def restore_annotation_panel_state(self, state: dict) -> None:
        """Reapply a shell-cached AnnotationPanel state snapshot."""
        self._ann_panel.restore_state(state)

    def get_focused_image(self) -> Path | None:
        return self._current_image_path

    def get_visible_paths(self) -> list[Path]:
        return self._file_list.get_visible_paths()

    def get_all_paths(self) -> list[Path]:
        return self._file_list.get_paths()

    def reload_current(self) -> None:
        """Discard in-memory state and reload current image's annotations from disk.

        Used after external writers (e.g. batch worker) supersede the on-disk
        record for the focused image. The store read below still triggers the
        flush callback, but _save_current's clean-skip (memory unchanged since
        its last baseline) keeps the stale in-memory state from overwriting
        the newer record — memory is then replaced by the reload.
        """
        if not self._project or not self._current_image_path:
            return
        label_path = self._project.label_path_for(self._current_image_path)
        ia = self._store.load_or_empty(
            label_path,
            self._current_image_path.name,
            image_size=get_image_size(self._current_image_path),
        )
        self._current_annotation = ia
        self._canvas.set_annotations(list(ia.annotations))
        self._ann_panel.set_annotations(list(ia.annotations))
        self._emit_status()
        self._prev_annotations_snapshot = self._stats_snapshot(ia.annotations)
        self._last_saved_record = self._record_snapshot(ia)

    def commit_pending_save(self) -> None:
        self._save_current()

    def add_auto_class_prediction(self, path, class_name, confidence):
        raise NotImplementedError("DetectPoseView does not support classify predictions")

    def add_auto_annotations(self, anns: list[Annotation], iou: float = 0.5) -> None:
        # Single merge implementation point (core.autolabel wraps
        # find_conflicts): the interactive surface adds the conflicting
        # predictions too and highlights the pairs red for the user.
        outcome = merge_predictions(self._canvas.annotations, anns, iou)
        self._canvas.add_annotations(outcome.accepted)
        if outcome.conflict_pairs:
            self._canvas.add_annotations(outcome.conflict_predictions)
            self._canvas.set_conflict_pairs(
                [(e.id, p.id) for e, p in outcome.conflict_pairs]
            )
        self._push_undo()
        self._sync_annotations_to_panel()
        if self._current_image_path is not None:
            self.annotations_changed.emit(self._current_image_path)

    # ── Tool management ────────────────────────────────────────

    def _set_tool(self, mode: str) -> None:
        self._btn_select.setChecked(mode == "select")
        self._btn_bbox.setChecked(mode == "draw_bbox")
        self._btn_keypoint.setChecked(mode == "draw_keypoint")
        self._btn_polygon.setChecked(mode == "draw_polygon")
        self._btn_sam.setChecked(mode == "sam_prompt")
        self._canvas.set_tool_mode(mode)
        if mode == "sam_prompt":
            self._activate_sam_tool()

    def _is_segment(self) -> bool:
        return self._project is not None and self._project.config.task_type == "segment"

    # ── SAM assist (lazy controller, view-local collaborator) ──

    def _ensure_sam_controller(self):
        """Construct + wire the SamAssistController on first use.

        Lazy on purpose: importing/constructing it costs nothing heavy (the
        engine lazy-imports ultralytics inside ``load()``), but deferring
        keeps non-segment sessions from even touching the module.
        """
        if self._sam_ctrl is None:
            from src.controllers.sam_assist import SamAssistController

            self._sam_ctrl = SamAssistController(parent=self)
        if not self._sam_wired:
            self._sam_ctrl.mask_ready.connect(self._on_sam_mask_ready)
            self._sam_ctrl.status_message.connect(self.status_changed.emit)
            self._sam_wired = True
        return self._sam_ctrl

    def _activate_sam_tool(self) -> None:
        ctrl = self._ensure_sam_controller()
        path = self._current_image_path
        size = get_image_size(path) if path is not None else None
        ctrl.activate(path, size)

    def _on_sam_point_requested(self, x: float, y: float) -> None:
        if self._sam_ctrl is not None:
            self._sam_ctrl.request_point(x, y)

    def _on_sam_box_requested(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._sam_ctrl is not None:
            self._sam_ctrl.request_box(x1, y1, x2, y2)

    def _on_sam_mask_ready(self, polygon, bbox) -> None:
        """SAM produced a polygon: stage it as the canvas draft and reuse the
        class-picker → create_polygon_from_draw flow (undo/auto-save ride the
        existing annotation_created signals for free)."""
        if self._canvas.tool_mode != "sam_prompt":
            return  # user switched tools while the status was still fresh
        self._canvas.begin_polygon_from_points(polygon)

    def cleanup(self) -> None:
        """Release view-held background resources (view teardown / app close)."""
        self._stop_label_scan()
        if self._sam_ctrl is not None:
            self._sam_ctrl.shutdown()

    # ── Image switching ────────────────────────────────────────

    def _on_image_selected(self, path: Path) -> None:
        self._save_current()
        self._current_image_path = path
        # Switch window: _current_image_path now names the NEW image while the
        # canvas still shows the OLD one. The store read below re-triggers the
        # flush callback (_save_current); clearing the record makes that flush
        # a provable no-op instead of relying on the clean-skip alone.
        self._current_annotation = None

        pixmap = self._image_cache.get(path)
        if pixmap:
            self._canvas.set_pixmap(pixmap)
        else:
            self._canvas.load_image(str(path))
        logger.debug("Image selected: %s", path.name)

        self._preload_neighbors(path)

        if self._project:
            label_path = self._project.label_path_for(path)
            self._current_annotation = self._store.load_or_empty(
                label_path, path.name, image_size=get_image_size(path),
            )

            self._canvas.set_annotations(list(self._current_annotation.annotations))
            self._ann_panel.set_annotations(list(self._current_annotation.annotations))
            self._ann_panel.set_image_user_tags(list(self._current_annotation.tags))

            self._emit_status()

            key = str(path)
            if key not in self._undo_stacks:
                self._undo_stacks[key] = UndoStack()
                self._undo_stacks[key].push(self._current_annotation.to_dict())
            else:
                self._undo_stacks.move_to_end(key)
            while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
                self._undo_stacks.popitem(last=False)

            self._prev_annotations_snapshot = self._stats_snapshot(self._current_annotation.annotations)
            self._last_saved_record = self._record_snapshot(self._current_annotation)

        self.image_focus_changed.emit(path)

        # SAM tool active: pre-encode the newly-focused image in the
        # background so the first click doesn't stall.
        if self._sam_ctrl is not None and self._canvas.tool_mode == "sam_prompt":
            self._sam_ctrl.set_image(path, get_image_size(path))

    def _preload_neighbors(self, current: Path) -> None:
        if not self._project:
            return
        images = self._project.list_images()
        try:
            idx = images.index(current)
        except ValueError:
            return
        neighbors = []
        for offset in [1, 2, -1]:
            ni = idx + offset
            if 0 <= ni < len(images):
                neighbors.append(images[ni])
        if neighbors:
            self._image_cache.preload(neighbors)

    @staticmethod
    def _record_snapshot(ia: ImageAnnotation) -> dict:
        """Full-record dict used as the clean-check baseline.

        ``ImageAnnotation.to_dict`` aliases the live ``image_tags`` / ``tags``
        lists; detach them so a later in-place mutation of the record can't
        silently equalize the baseline (which would skip a needed write).
        """
        d = ia.to_dict()
        d["image_tags"] = list(d["image_tags"])
        d["tags"] = list(d["tags"])
        return d

    def _save_current(self) -> None:
        if not self._project or not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas.annotations)
        record = self._record_snapshot(self._current_annotation)
        if record == self._last_saved_record:
            # Clean: nothing changed since the last disk load/save. Skip the
            # write — this method doubles as the store's flush callback, so
            # scan loops (one store.load per image) hit it N times.
            return
        label_path = self._project.label_path_for(self._current_image_path)
        self._store.save(self._current_annotation, label_path)
        self._last_saved_record = record
        logger.debug("Saved annotations for %s", self._current_image_path.name)
        self._file_list.set_status(self._current_image_path, self._current_annotation.status)
        old_snap = self._prev_annotations_snapshot or []
        new_snap = self._stats_snapshot(self._current_annotation.annotations)
        if old_snap != new_snap:
            self._update_stats_incremental(old_snap, new_snap)
            self._prev_annotations_snapshot = new_snap
            # Notify shell so it can mirror stats / push undo if needed
            self.annotations_changed.emit(self._current_image_path)

    # ── Annotation events ──────────────────────────────────────

    def _on_annotation_selected(self, ann_id) -> None:
        self._ann_panel.select_annotation(ann_id)

    def _on_annotation_created(self, ann) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_modified(self, ann_id: str) -> None:
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotation_deleted(self, ann_id: str) -> None:
        self._canvas.remove_annotation(ann_id)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_annotations_changed(self) -> None:
        self._sync_annotations_to_panel()

    def _show_class_picker(self, default_class: str | None, px: float, py: float) -> str | None:
        if not self._project:
            return None
        classes = self._project.config.classes
        colors = {cls: self._project.config.get_class_color(cls) for cls in classes}

        picker = ClassPickerPopup(
            classes=classes,
            colors=colors,
            default_class=default_class,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move(global_pos)
        if not picker.exec_():
            return None

        cls_name = picker.get_selected_class()
        if cls_name is None:
            return None

        if picker.is_new_class():
            self._project.add_class(cls_name)
            self._project.save()
            colors[cls_name] = self._project.config.get_class_color(cls_name)
            self._canvas.set_class_colors(colors)
            self._ann_panel.set_class_colors(colors)
            self._ann_panel.set_classes(self._project.config.classes)
            self.classes_changed.emit()

        return cls_name

    def _on_class_requested(self, px: float, py: float) -> None:
        cls_name = self._show_class_picker(self._last_class, px, py)
        if cls_name is None:
            self._clear_draw_state()
            return

        cls_id = self._project.config.get_class_id(cls_name)
        self._set_default_class(cls_name)
        if self._canvas.tool_mode == "draw_bbox":
            self._canvas.create_bbox_from_draw(cls_name, cls_id)
        elif self._canvas.tool_mode == "draw_keypoint":
            self._canvas.create_keypoint_at(cls_name, cls_id)
        elif self._canvas.tool_mode == "draw_polygon":
            self._canvas.create_polygon_from_draw(cls_name, cls_id)
        elif self._canvas.tool_mode == "sam_prompt":
            # SAM-assisted polygon: auto-sourced and unconfirmed, matching the
            # auto-label lifecycle (dashed border until the user confirms).
            self._canvas.create_polygon_from_draw(
                cls_name, cls_id, source="auto", confirmed=False,
            )

    def _on_default_class_changed(self, cls_name) -> None:
        """Class set/cleared via the right-side project class list.

        cls_name is the class name when set, or None when the user toggled
        the current default off by re-double-clicking it.
        """
        self._last_class = cls_name
        if cls_name:
            self.status_changed.emit(f"默认类别: {cls_name}")
        else:
            self.status_changed.emit("已取消默认类别")

    def _on_user_tags_edited(self, new_tags: list) -> None:
        """User added/removed a chip in the per-image Tag bar.

        Persists the change, updates file_list filter cache, and fans out a
        signal so the shell can sync the project tag registry.
        """
        if not self._project or not self._current_image_path or self._current_annotation is None:
            return
        tags = [str(t) for t in new_tags]
        self._current_annotation.tags = list(tags)
        # Reuse the canonical save path so annotations stay in sync.
        self._save_current()
        self._file_list.set_image_tags(self._current_image_path, set(tags))
        self.user_tags_changed.emit(self._current_image_path, list(tags))

    def _set_default_class(self, cls_name: str) -> None:
        """Update `_last_class` and keep the side-panel highlight in sync."""
        self._last_class = cls_name
        self._ann_panel.set_default_class(cls_name)

    def _clear_draw_state(self) -> None:
        self._canvas.clear_draw_state()

    def _on_class_change_requested(self, ann_id: str, px: float, py: float) -> None:
        ann = None
        for a in self._canvas.annotations:
            if a.id == ann_id:
                ann = a
                break
        if ann is None:
            return

        cls_name = self._show_class_picker(ann.class_name, px, py)
        if cls_name is None or cls_name == ann.class_name:
            return

        ann.class_name = cls_name
        ann.class_id = self._project.config.get_class_id(cls_name)
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    def _on_keypoint_attach_requested(self, ann_id: str, px: float, py: float) -> None:
        from src.core.annotation import Keypoint

        # Consume the draw-origin at entry: later canvas events (while the
        # picker below is open) can no longer rewrite it under us.
        start = self._canvas.consume_draw_start()
        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None or not start:
            self._clear_draw_state()
            return

        existing_labels: list[str] = []
        seen: set[str] = set()
        for a in self._canvas.annotations:
            for kp in a.keypoints:
                if kp.label not in seen:
                    existing_labels.append(kp.label)
                    seen.add(kp.label)

        default_label = f"kp_{len(ann.keypoints)}"

        picker = KeypointLabelPicker(
            existing_labels=existing_labels,
            default_label=default_label,
            parent=self,
        )
        global_pos = self._canvas.mapToGlobal(QPoint(int(px), int(py)))
        picker.move(global_pos)

        if not picker.exec_():
            self._clear_draw_state()
            return

        label = picker.get_label()
        if not label:
            self._clear_draw_state()
            return

        nx, ny = start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        self._canvas.add_keypoint_to_annotation(ann_id, kp)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_panel_keypoint_clicked(self, ann_id: str, kp_idx: int) -> None:
        self._canvas.select_keypoint(ann_id, kp_idx)

    def _on_keypoint_rename(self, ann_id: str, kp_idx: int, new_label: str) -> None:
        self._canvas.rename_keypoint(ann_id, kp_idx, new_label)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_keypoint_visibility(self, ann_id: str, kp_idx: int) -> None:
        self._canvas.cycle_keypoint_visibility(ann_id, kp_idx)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _on_keypoint_delete(self, ann_id: str, kp_idx: int) -> None:
        self._canvas.remove_keypoint(ann_id, kp_idx)
        self._push_undo()
        self._sync_annotations_to_panel()

    def _sync_annotations_to_panel(self) -> None:
        self._ann_panel.set_annotations(list(self._canvas.annotations))
        self._emit_status()

    def _emit_status(self) -> None:
        if not self._current_image_path:
            return
        idx, total = self._file_list.get_index_info()
        n_ann = len(self._canvas.annotations)
        n_confirmed = sum(1 for a in self._canvas.annotations if a.confirmed)
        n_pending = n_ann - n_confirmed
        parts = [
            self._current_image_path.name,
            f"{idx}/{total}",
            f"标注: {n_ann}",
        ]
        if n_pending > 0:
            parts.append(f"确认: {n_confirmed} 待确认: {n_pending}")
        self.status_changed.emit(" | ".join(parts))

    # ── Stats ─────────────────────────────────────────────────

    @staticmethod
    def _new_stats(total_images: int) -> dict:
        return {
            "total_images": total_images,
            "labeled_images": 0,
            "confirmed_images": 0,
            "total_annotations": 0,
            "class_counts": {},
        }

    @staticmethod
    def _accumulate_stats(stats: dict, ia) -> None:
        """Fold one label record into a stats dict.

        Single implementation shared by the set_project scan and
        _compute_project_stats so the two paths can't drift.
        """
        if len(ia.annotations) == 0:
            return
        stats["labeled_images"] += 1
        if all(a.confirmed for a in ia.annotations):
            stats["confirmed_images"] += 1
        for ann in ia.annotations:
            stats["total_annotations"] += 1
            stats["class_counts"][ann.class_name] = stats["class_counts"].get(ann.class_name, 0) + 1

    def _compute_project_stats(self) -> dict:
        if not self._project:
            return {}
        images = self._project.list_images()
        stats = self._new_stats(len(images))
        for img_path in images:
            label_path = self._project.label_path_for(img_path)
            ia = self._store.load(label_path)
            if ia is None:
                continue
            self._accumulate_stats(stats, ia)
        return stats

    def _init_stats_cache(self) -> None:
        self._stats_cache = self._compute_project_stats()
        self._ann_panel.set_project_stats(self._stats_cache)

    def _update_stats_incremental(self, old_snap: list[tuple], new_snap: list[tuple]) -> None:
        if not self._stats_cache:
            return
        had_old = len(old_snap) > 0
        has_new = len(new_snap) > 0

        if had_old and not has_new:
            self._stats_cache["labeled_images"] -= 1
        elif not had_old and has_new:
            self._stats_cache["labeled_images"] += 1

        old_all_confirmed = had_old and all(c for _, c in old_snap)
        new_all_confirmed = has_new and all(c for _, c in new_snap)
        if old_all_confirmed and not new_all_confirmed:
            self._stats_cache["confirmed_images"] -= 1
        elif not old_all_confirmed and new_all_confirmed:
            self._stats_cache["confirmed_images"] += 1

        for cls, _ in old_snap:
            self._stats_cache["total_annotations"] -= 1
            self._stats_cache["class_counts"][cls] = self._stats_cache["class_counts"].get(cls, 1) - 1
            if self._stats_cache["class_counts"][cls] <= 0:
                del self._stats_cache["class_counts"][cls]

        for cls, _ in new_snap:
            self._stats_cache["total_annotations"] += 1
            self._stats_cache["class_counts"][cls] = self._stats_cache["class_counts"].get(cls, 0) + 1

        self._ann_panel.set_project_stats(self._stats_cache)

    @staticmethod
    def _stats_snapshot(anns) -> list[tuple]:
        return [(a.class_name, a.confirmed) for a in anns]

    def _refresh_project_stats(self) -> None:
        """Recompute and refresh stats panel after a project-wide change."""
        self._init_stats_cache()

    def _confirm_all(self) -> None:
        for ann in self._canvas.annotations:
            ann.confirmed = True
        self._push_undo()
        self._canvas.update()
        self._sync_annotations_to_panel()

    # ── Copy / Paste ──────────────────────────────────────────

    def _copy_annotation(self) -> None:
        ann = self._canvas.get_selected_annotation()
        if ann:
            self._clipboard = [ann.to_dict()]
            logger.debug("Copied annotation: %s", ann.class_name)

    def _on_annotation_copied(self, ann_id: str) -> None:
        for ann in self._canvas.annotations:
            if ann.id == ann_id:
                self._clipboard = [ann.to_dict()]
                logger.debug("Copied annotation via menu: %s", ann.class_name)
                break

    def _paste_annotation(self) -> None:
        if not self._clipboard or self._canvas.is_locked:
            return
        import uuid as _uuid
        new_anns = []
        for ann_dict in self._clipboard:
            new_dict = dict(ann_dict)
            new_dict["id"] = str(_uuid.uuid4())
            new_dict["confirmed"] = False
            new_anns.append(Annotation.from_dict(new_dict))
        self._canvas.add_annotations(new_anns)
        self._push_undo()
        self._sync_annotations_to_panel()
        logger.debug("Pasted %d annotations", len(self._clipboard))

    # ── Undo/Redo ──────────────────────────────────────────────

    def _push_undo(self) -> None:
        if not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas.annotations)
        key = str(self._current_image_path)
        if key not in self._undo_stacks:
            self._undo_stacks[key] = UndoStack()
        else:
            self._undo_stacks.move_to_end(key)
        self._undo_stacks[key].push(self._current_annotation.to_dict())
        while len(self._undo_stacks) > self._UNDO_MAX_IMAGES:
            self._undo_stacks.popitem(last=False)

    def undo(self) -> None:
        if not self._current_image_path:
            return
        key = str(self._current_image_path)
        stack = self._undo_stacks.get(key)
        if not stack or not stack.can_undo:
            return
        state = stack.undo()
        if state:
            self._restore_state(state)

    def redo(self) -> None:
        if not self._current_image_path:
            return
        key = str(self._current_image_path)
        stack = self._undo_stacks.get(key)
        if not stack or not stack.can_redo:
            return
        state = stack.redo()
        if state:
            self._restore_state(state)

    def _restore_state(self, state: dict) -> None:
        ia = ImageAnnotation.from_dict(state)
        self._current_annotation = ia
        self._canvas.set_annotations(list(ia.annotations))
        self._ann_panel.set_annotations(list(ia.annotations))

    # ── Keyboard shortcuts (view-internal) ─────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key_W:
            self._set_tool("draw_bbox")
        elif key == Qt.Key_K:
            self._set_tool("draw_keypoint")
        elif key == Qt.Key_P and self._is_segment():
            self._set_tool("draw_polygon")
        elif key in (Qt.Key_Return, Qt.Key_Enter) and self._canvas.tool_mode == "draw_polygon":
            # Enter closes an in-progress polygon (fallback to the double-click
            # gesture); only consumed while a draft is open.
            if not self._canvas.finish_polygon():
                super().keyPressEvent(event)
        elif key == Qt.Key_Escape and self._canvas.tool_mode == "draw_polygon" and self._canvas.has_polygon_draft:
            self._canvas.clear_draw_state()
        elif key == Qt.Key_V and not (mod & Qt.ControlModifier):
            self._set_tool("select")
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self._save_current()
            self._file_list.go_next()
        elif (key == Qt.Key_A and not (mod & (Qt.ShiftModifier | Qt.ControlModifier))) or key == Qt.Key_Left:
            self._save_current()
            self._file_list.go_prev()
        elif key == Qt.Key_Space and mod & Qt.ControlModifier:
            self._confirm_all()
        elif key == Qt.Key_Space:
            ann = self._canvas.get_selected_annotation()
            if ann:
                ann.confirmed = True
                self._push_undo()
                self._canvas.update()
                self._sync_annotations_to_panel()
        elif key == Qt.Key_Delete:
            ann = self._canvas.get_selected_annotation()
            if ann:
                self._on_annotation_deleted(ann.id)
        elif key == Qt.Key_C and mod & Qt.ControlModifier:
            self._copy_annotation()
        elif key == Qt.Key_V and mod & Qt.ControlModifier:
            self._paste_annotation()
        elif key in (Qt.Key_Plus, Qt.Key_Equal) and mod & Qt.ControlModifier:
            self._canvas.zoom_in()
        elif key == Qt.Key_Minus and mod & Qt.ControlModifier:
            self._canvas.zoom_out()
        elif key == Qt.Key_0 and mod & Qt.ControlModifier:
            self._canvas.zoom_fit()
        else:
            super().keyPressEvent(event)

    # ── Batch operations on visible images ─────────────────────

    def _on_batch_confirm(self, paths: list[Path]) -> None:
        if not self._project:
            return
        self._save_current()
        count = 0
        for img_path in paths:
            label_path = self._project.label_path_for(img_path)
            ia = self._store.load(label_path)
            if ia and ia.annotations:
                old_snap = self._stats_snapshot(ia.annotations)
                for ann in ia.annotations:
                    ann.confirmed = True
                self._store.save(ia, label_path)
                self._file_list.set_status(img_path, ia.status)
                self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
                count += 1
        if self._current_image_path and self._current_image_path in paths:
            self.reload_current()
        self.status_changed.emit(f"批量确认: {count} 张图片")
        logger.info("Batch confirmed %d images", count)

    def _on_batch_delete(self, paths: list[Path]) -> None:
        if not self._project:
            return
        self._save_current()
        count = 0
        for img_path in paths:
            label_path = self._project.label_path_for(img_path)
            ia = self._store.load(label_path)
            if ia and ia.annotations:
                old_snap = self._stats_snapshot(ia.annotations)
                ia.annotations.clear()
                self._store.save(ia, label_path)
                self._file_list.set_status(img_path, "unlabeled")
                self._update_stats_incremental(old_snap, [])
                count += 1
        if self._current_image_path and self._current_image_path in paths:
            self.reload_current()
        self.status_changed.emit(f"批量删除标注: {count} 张图片")
        logger.info("Batch deleted annotations for %d images", count)

    def _on_delete_images(self, paths: list[Path]) -> None:
        """Delete image files and their labels from disk after confirmation."""
        if not self._project or not paths:
            return
        paths = list(paths)

        # Labeled-count scan: the store flushes pending edits on first read,
        # so the count below reflects the in-memory canvas too.
        labeled_count = 0
        for p in paths:
            ia = self._store.load(self._project.label_path_for(p))
            if ia and ia.annotations:
                labeled_count += 1

        n = len(paths)
        if labeled_count:
            msg = (
                f"确定要删除 {n} 张图片吗？\n"
                f"其中 {labeled_count} 张包含标注，将一并删除。\n\n"
                f"此操作不可撤销。"
            )
        else:
            msg = f"确定要删除 {n} 张图片吗？\n\n此操作不可撤销。"
        reply = QMessageBox.question(
            self, "删除图片", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        current_in_deleted = (
            self._current_image_path is not None
            and self._current_image_path in paths
        )

        img_n, lbl_n = self._project.delete_images(paths)

        for p in paths:
            self._image_cache.invalidate(p)
            self._undo_stacks.pop(str(p), None)
        self._file_list.forget_paths(paths)

        if current_in_deleted:
            self._current_image_path = None
            self._current_annotation = None
            self._prev_annotations_snapshot = None
            self._last_saved_record = None
            self._canvas.clear()
            self._ann_panel.set_annotations([])

        remaining = self._project.list_images()
        self._file_list.refresh_paths(remaining)

        if current_in_deleted and remaining:
            self._file_list.setCurrentRow(0)

        self._refresh_project_stats()
        self.status_changed.emit(
            f"已删除 {img_n} 张图片，{lbl_n} 个标注文件"
        )
        logger.info("Deleted %d images, %d labels", img_n, lbl_n)

    def _collect_unconfirmed(self, visible_paths: list[Path]):
        affected = []
        total = 0
        for img_path in visible_paths:
            label_path = self._project.label_path_for(img_path)
            ia = self._store.load(label_path)
            if ia:
                unconfirmed = sum(1 for a in ia.annotations if not a.confirmed)
                if unconfirmed > 0:
                    affected.append((img_path, label_path, ia))
                    total += unconfirmed
        return affected, total

    def _batch_confirm_visible(self) -> None:
        if not self._project:
            return
        visible_paths = self._file_list.get_visible_paths()
        if not visible_paths:
            return

        # No explicit pre-scan flush: _collect_unconfirmed reads through the
        # store, which flushes the focused image's pending annotations first
        # (so they are counted/confirmed and stale disk can't win).
        affected, total = self._collect_unconfirmed(visible_paths)
        if total == 0:
            self.status_changed.emit("没有需要确认的预标注")
            return

        reply = QMessageBox.question(
            self, "确认可见预标注",
            f"将确认 {len(affected)} 张图片中的 {total} 个未确认标注，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            for ann in ia.annotations:
                if not ann.confirmed:
                    ann.confirmed = True
            self._store.save(ia, label_path)
            self._file_list.set_status(img_path, ia.status)
            self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
            count += 1

        if self._current_image_path and self._current_image_path in visible_paths:
            self.reload_current()
        self.status_changed.emit(f"已确认可见预标注: {count} 张图片")
        logger.info("Batch confirmed visible unconfirmed annotations for %d images", count)

    def _batch_revert_visible(self) -> None:
        if not self._project:
            return
        visible_paths = self._file_list.get_visible_paths()
        if not visible_paths:
            return

        # Same as _batch_confirm_visible: the store-mediated scan flushes.
        affected, total = self._collect_unconfirmed(visible_paths)
        if total == 0:
            self.status_changed.emit("没有需要撤销的预标注")
            return

        reply = QMessageBox.question(
            self, "撤销可见预标注",
            f"将删除 {len(affected)} 张图片中的 {total} 个未确认标注，此操作不可撤销，是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            ia.annotations = [a for a in ia.annotations if a.confirmed]
            self._store.save(ia, label_path)
            self._file_list.set_status(img_path, ia.status)
            self._update_stats_incremental(old_snap, self._stats_snapshot(ia.annotations))
            count += 1

        if self._current_image_path and self._current_image_path in visible_paths:
            self.reload_current()
        self.status_changed.emit(f"已撤销可见预标注: {count} 张图片")
        logger.info("Batch reverted visible unconfirmed annotations for %d images", count)

    # ── Helpers used by the shell ─────────────────────────────

    def get_unlabeled_image_paths(self) -> list[Path]:
        if not self._project:
            return []
        result = []
        for img_path in self._project.list_images():
            label_path = self._project.label_path_for(img_path)
            ia = self._store.load(label_path)
            if ia is None or len(ia.annotations) == 0:
                result.append(img_path)
        return result

    def refresh_image_list(self, images: list[Path]) -> None:
        """Called by shell after rescan / drop to refresh the file list."""
        self._file_list.refresh_paths(images)
        # Recompute stats from scratch (project-wide change)
        self._refresh_project_stats()
