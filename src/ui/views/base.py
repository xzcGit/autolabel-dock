"""TaskView protocol — interface for pluggable task views inside LabelPanel."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.core.annotation import Annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter


class TaskView(QWidget):
    """Abstract interface for a task-specific view inside the LabelPanel shell.

    Subclasses MUST implement all methods. The shell (LabelPanel) calls these
    to drive the view; the view emits the signals to notify the shell.
    """

    # 视图 → 壳
    annotations_changed = pyqtSignal(object)         # path: 触发壳 push_undo + save
    status_changed = pyqtSignal(str)                 # 状态栏文本
    image_focus_changed = pyqtSignal(object)         # path: 当前焦点图
    auto_label_single_requested = pyqtSignal()
    auto_label_batch_requested = pyqtSignal()
    images_dropped = pyqtSignal(list)                # list[Path]
    classes_changed = pyqtSignal()                   # 视图修改了 project.config.classes
    # Per-image user-tag edits — payload (path, new_tags).
    # The view persists the change locally; MainWindow listens to merge any
    # newly-typed tag into the project tag registry.
    user_tags_changed = pyqtSignal(object, list)

    # ── 壳 → 视图 ─────────────────────────────────────
    def set_project(self, project: ProjectManager) -> None:
        raise NotImplementedError

    def set_class_colors(self, colors: dict[str, str]) -> None:
        raise NotImplementedError

    def set_classes(self, classes: list[str]) -> None:
        raise NotImplementedError

    def set_filter(self, status: str | None) -> None:
        raise NotImplementedError

    def set_class_filter(self, cls: str | None) -> None:
        raise NotImplementedError

    def set_available_tags(self, tags: list[str]) -> None:
        """Optional override: push project-level known tags to per-image UI."""
        return None

    def set_tag_filter(self, tag_filter: TagFilter | None) -> None:
        """Optional override: empty / None means "no tag filter"."""
        # Default no-op so views that don't care don't need to implement it.
        return None

    def get_selected_image_paths(self) -> list[Path]:
        """Return paths of images the user currently has selected in this view.

        detect/pose: file_list.get_selected_paths()
        classify:    grid.selectedItems() → Path list
        Both use QListWidget.ExtendedSelection so single-image "selection"
        works without branching. Default is empty; subclasses implement.
        """
        return []

    def refresh_image_tags(self, path: Path, tags: list[str]) -> None:
        """Update this view's local caches/visuals after an external write
        to ``path``'s user tags.

        detect/pose: file_list.set_image_tags(path, set(tags)) updates the
                     tag cache AND re-evaluates this row's visibility under
                     the active filters (without touching scroll); if path
                     == current focus, push to AnnotationPanel.
        classify:    if path == preview pane current, sync its TagChipBar;
                     update only this image's grid-item visibility under
                     the active filters (no scroll movement).
        Default: no-op.
        """
        return None

    def get_focused_image(self) -> Path | None:
        raise NotImplementedError

    def get_visible_paths(self) -> list[Path]:
        raise NotImplementedError

    def get_all_paths(self) -> list[Path]:
        """All image paths the view currently knows about (incl. filtered out)."""
        raise NotImplementedError

    def reload_current(self) -> None:
        raise NotImplementedError

    def commit_pending_save(self) -> None:
        raise NotImplementedError

    def undo(self) -> None:
        raise NotImplementedError

    def redo(self) -> None:
        raise NotImplementedError

    # ── 自动标注接口（按任务类型分两个，避免参数歧义）─────────
    def add_auto_annotations(self, anns: list[Annotation], iou: float = 0.5) -> None:
        """detect/pose 视图实现。classify 视图 raise NotImplementedError"""
        raise NotImplementedError

    def add_auto_class_prediction(
        self, path: Path, class_name: str, confidence: float,
    ) -> bool:
        """classify 视图实现。detect/pose 视图 raise NotImplementedError.

        Returns True if applied; False if skipped (e.g. existing confirmed tag).
        """
        raise NotImplementedError
