"""Detect/Pose view — extracted from LabelPanel for the shell/view refactor.

Behavior matches the pre-refactor LabelPanel for detect/pose projects.
The shell (LabelPanel) supplies the shared `image_cache` and `_undo_stacks`
dictionary so view switches preserve cache reuse but per-view undo state.
"""
from __future__ import annotations

import logging
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
from src.core.label_io import save_annotation, load_annotation
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
        parent=None,
    ):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._current_image_path: Path | None = None
        self._current_annotation: ImageAnnotation | None = None
        self._image_cache = image_cache
        self._undo_stacks = undo_stacks
        self._last_class: str | None = None
        self._clipboard: list[dict] | None = None
        self._stats_cache: dict = {}
        self._prev_annotations_snapshot: list[tuple] | None = None

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

        for btn in [self._btn_select, self._btn_bbox, self._btn_keypoint]:
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

        # File list
        self._file_list.image_selected.connect(self._on_image_selected)
        self._file_list.images_dropped.connect(self.images_dropped.emit)
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

        images = project.list_images()
        self._file_list.set_image_paths(images)
        for img_path in images:
            label_path = project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia:
                self._file_list.set_status(img_path, ia.status)
                classes_in_img = {a.class_name for a in ia.annotations}
                self._file_list.set_image_classes(img_path, classes_in_img)
                self._file_list.set_image_tags(img_path, set(ia.tags))
        if images:
            self._file_list.setCurrentRow(0)
        logger.info("DetectPoseView loaded: %s (%d images)", project.config.name, len(images))
        self._init_stats_cache()

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

    def get_focused_image(self) -> Path | None:
        return self._current_image_path

    def get_visible_paths(self) -> list[Path]:
        return self._file_list.get_visible_paths()

    def get_all_paths(self) -> list[Path]:
        return self._file_list.get_paths()

    def reload_current(self) -> None:
        """Discard in-memory state and reload current image's annotations from disk.

        Used after external writers (e.g. batch worker) modify the on-disk JSON
        for the focused image. We must NOT call _save_current() first, because
        the canvas/in-memory state is stale relative to those external writes.
        """
        if not self._project or not self._current_image_path:
            return
        label_path = self._project.label_path_for(self._current_image_path)
        ia = load_annotation(label_path)
        if ia is None:
            w, h = get_image_size(self._current_image_path)
            ia = ImageAnnotation(
                image_path=self._current_image_path.name, image_size=(w, h),
            )
        self._current_annotation = ia
        self._canvas.set_annotations(list(ia.annotations))
        self._ann_panel.set_annotations(list(ia.annotations))
        self._emit_status()
        self._prev_annotations_snapshot = self._stats_snapshot(ia.annotations)

    def commit_pending_save(self) -> None:
        self._save_current()

    def add_auto_class_prediction(self, path, class_name, confidence):
        raise NotImplementedError("DetectPoseView does not support classify predictions")

    def add_auto_annotations(self, anns: list[Annotation], iou: float = 0.5) -> None:
        from src.core.annotation import find_conflicts
        existing = self._canvas.annotations
        conflicts, clean = find_conflicts(existing, anns, iou)
        self._canvas.add_annotations(clean)
        if conflicts:
            self._canvas.add_annotations([p for _, p in conflicts])
            self._canvas.set_conflict_pairs([(e.id, p.id) for e, p in conflicts])
        self._push_undo()
        self._sync_annotations_to_panel()
        if self._current_image_path is not None:
            self.annotations_changed.emit(self._current_image_path)

    # ── Tool management ────────────────────────────────────────

    def _set_tool(self, mode: str) -> None:
        self._btn_select.setChecked(mode == "select")
        self._btn_bbox.setChecked(mode == "draw_bbox")
        self._btn_keypoint.setChecked(mode == "draw_keypoint")
        self._canvas.set_tool_mode(mode)

    # ── Image switching ────────────────────────────────────────

    def _on_image_selected(self, path: Path) -> None:
        self._save_current()
        self._current_image_path = path

        pixmap = self._image_cache.get(path)
        if pixmap:
            self._canvas.set_pixmap(pixmap)
        else:
            self._canvas.load_image(str(path))
        logger.debug("Image selected: %s", path.name)

        self._preload_neighbors(path)

        if self._project:
            label_path = self._project.label_path_for(path)
            ia = load_annotation(label_path)
            if ia:
                self._current_annotation = ia
            else:
                w, h = get_image_size(path)
                self._current_annotation = ImageAnnotation(
                    image_path=path.name,
                    image_size=(w, h),
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

        self.image_focus_changed.emit(path)

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

    def _save_current(self) -> None:
        if not self._project or not self._current_image_path or not self._current_annotation:
            return
        self._current_annotation.annotations = list(self._canvas.annotations)
        label_path = self._project.label_path_for(self._current_image_path)
        save_annotation(self._current_annotation, label_path)
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

        ann = next((a for a in self._canvas.annotations if a.id == ann_id), None)
        if ann is None or not self._canvas._draw_start:
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

        nx, ny = self._canvas._draw_start
        kp = Keypoint(x=nx, y=ny, visible=2, label=label)
        self._canvas.add_keypoint_to_annotation(ann_id, kp)
        self._canvas._draw_start = None
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

    def _compute_project_stats(self) -> dict:
        if not self._project:
            return {}
        stats = {
            "total_images": 0,
            "labeled_images": 0,
            "confirmed_images": 0,
            "total_annotations": 0,
            "class_counts": {},
        }
        images = self._project.list_images()
        stats["total_images"] = len(images)
        for img_path in images:
            label_path = self._project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None or len(ia.annotations) == 0:
                continue
            stats["labeled_images"] += 1
            all_confirmed = all(a.confirmed for a in ia.annotations)
            if all_confirmed:
                stats["confirmed_images"] += 1
            for ann in ia.annotations:
                stats["total_annotations"] += 1
                stats["class_counts"][ann.class_name] = stats["class_counts"].get(ann.class_name, 0) + 1
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
            ia = load_annotation(label_path)
            if ia and ia.annotations:
                old_snap = self._stats_snapshot(ia.annotations)
                for ann in ia.annotations:
                    ann.confirmed = True
                save_annotation(ia, label_path)
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
            ia = load_annotation(label_path)
            if ia and ia.annotations:
                old_snap = self._stats_snapshot(ia.annotations)
                ia.annotations.clear()
                save_annotation(ia, label_path)
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

        labeled_count = 0
        for p in paths:
            ia = load_annotation(self._project.label_path_for(p))
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

        self._save_current()
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
            ia = load_annotation(label_path)
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

        self._save_current()
        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            for ann in ia.annotations:
                if not ann.confirmed:
                    ann.confirmed = True
            save_annotation(ia, label_path)
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

        self._save_current()
        count = 0
        for img_path, label_path, ia in affected:
            old_snap = self._stats_snapshot(ia.annotations)
            ia.annotations = [a for a in ia.annotations if a.confirmed]
            save_annotation(ia, label_path)
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
            ia = load_annotation(label_path)
            if ia is None or len(ia.annotations) == 0:
                result.append(img_path)
        return result

    def refresh_image_list(self, images: list[Path]) -> None:
        """Called by shell after rescan / drop to refresh the file list."""
        self._file_list.refresh_paths(images)
        # Recompute stats from scratch (project-wide change)
        self._refresh_project_stats()
