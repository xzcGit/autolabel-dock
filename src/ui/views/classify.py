"""Classification view — thumbnail grid (Phase 6+)."""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5 import sip

from src.core.annotation import ImageAnnotation
from src.core.config import AppConfig
from src.core.label_io import load_annotation, save_annotation
from src.core.project import ProjectManager
from src.ui.icons import icon
from src.ui.theme import PALETTE, set_button_role, text_style
from src.ui.views.base import TaskView
from src.ui.views.thumbnail_loader import ThumbnailLoader
from src.utils.image import ImageCache, get_image_size, load_pixmap
from src.utils.undo import UndoStack

logger = logging.getLogger(__name__)


_APP_CONFIG_PATH = lambda: Path.home() / ".autolabel" / "config.json"


# ── Visual state ───────────────────────────────────────────────


@dataclass
class _ThumbnailVisual:
    label: str
    bg_color: str
    status_glyph: str
    show_question_badge: bool


def _compute_visual_state(
    ia: ImageAnnotation,
    class_colors: dict[str, str],
    default_color: str = "#6c7086",
) -> _ThumbnailVisual:
    """Compute thumbnail visual state from ImageAnnotation. Pure function."""
    if ia.image_tags:
        cls = ia.image_tags[0]
        bg = class_colors.get(cls, default_color)
        if ia.image_tags_confirmed:
            return _ThumbnailVisual(
                label=cls, bg_color=bg, status_glyph="✓", show_question_badge=False,
            )
        return _ThumbnailVisual(
            label=cls,
            bg_color=bg,
            status_glyph="⚡",
            show_question_badge=(ia.image_tags_source == "auto"),
        )
    return _ThumbnailVisual(
        label="未标", bg_color=default_color, status_glyph="○", show_question_badge=False,
    )


# ── Thumbnail delegate ─────────────────────────────────────────


_LABEL_BAR_H = 16


class ThumbnailDelegate(QStyledItemDelegate):
    """Renders thumbnail + bottom colored label bar + optional ? badge."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self._view = view  # for iconSize() lookup

    def paint(self, painter: QPainter, option, index):
        painter.save()
        rect = option.rect

        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, QColor(PALETTE["primary_soft"]))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(rect, QColor(PALETTE["panel_alt"]))

        pix_area = QRect(
            rect.x() + 2,
            rect.y() + 2,
            rect.width() - 4,
            rect.height() - _LABEL_BAR_H - 4,
        )
        pixmap = index.data(Qt.DecorationRole)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                pix_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            x = pix_area.x() + (pix_area.width() - scaled.width()) // 2
            y = pix_area.y() + (pix_area.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(pix_area, QColor(PALETTE["bg_deep"]))
            painter.setPen(QColor(PALETTE["text_subtle"]))
            painter.drawText(pix_area, Qt.AlignCenter, "...")

        visual: _ThumbnailVisual | None = index.data(Qt.UserRole + 2)
        if visual is None:
            visual = _ThumbnailVisual("未标", PALETTE["text_subtle"], "○", False)

        bar = QRect(rect.x(), rect.bottom() - _LABEL_BAR_H + 1, rect.width(), _LABEL_BAR_H)
        painter.fillRect(bar, QColor(visual.bg_color))
        painter.setPen(QColor(PALETTE["ink"]))
        f = QFont()
        f.setPixelSize(11)
        painter.setFont(f)
        painter.drawText(
            bar.adjusted(4, 0, -4, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"{visual.status_glyph} {visual.label}",
        )

        if visual.show_question_badge:
            badge = QRect(rect.right() - 18, rect.y() + 2, 16, 14)
            painter.fillRect(badge, QColor(PALETTE["danger"]))
            painter.setPen(QColor(PALETTE["ink"]))
            painter.drawText(badge, Qt.AlignCenter, "?")

        painter.restore()

    def sizeHint(self, option, index):
        sz = self._view.iconSize()
        return QSize(sz.width() + 4, sz.height() + _LABEL_BAR_H + 4)


# ── Thumbnail grid widget ──────────────────────────────────────


class ThumbnailGridWidget(QListWidget):
    """QListWidget in IconMode rendering thumbnails with custom delegate."""

    delete_images_requested = pyqtSignal(list)  # list[Path]

    # Keys that ClassifyView wants — must NOT be consumed by QListWidget's
    # default keyboardSearch (which fires on any key with non-empty text).
    # See keyPressEvent below.
    _PARENT_HANDLED_KEYS = frozenset({
        Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5,
        Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
        Qt.Key_Space, Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Escape,
        Qt.Key_T,
    })

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items_by_path: dict[str, QListWidgetItem] = {}
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setWrapping(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setSelectionRectVisible(True)
        self.setUniformItemSizes(True)
        self.setSpacing(6)
        self.setIconSize(QSize(96, 96))
        self.setItemDelegate(ThumbnailDelegate(self, self))

    def keyPressEvent(self, event):  # noqa: N802 (Qt naming)
        """Forward ClassifyView shortcut keys to the parent.

        QListWidget's default keyPressEvent (inherited from QAbstractItemView)
        invokes keyboardSearch() on any key whose `text` is non-empty and then
        accepts the event. That swallows '1'..'9' and Space before they can
        bubble up to ClassifyView.keyPressEvent — breaking the class
        shortcuts as soon as the grid takes focus (i.e. after the user clicks
        any thumbnail). Calling event.ignore() here lets QApplication's
        notify() walk the parent chain and deliver the key to ClassifyView.
        """
        if event.key() in self._PARENT_HANDLED_KEYS:
            event.ignore()
            return
        super().keyPressEvent(event)

    def add_image_item(
        self,
        path: Path,
        visual: _ThumbnailVisual,
        pixmap: QPixmap | None = None,
    ) -> QListWidgetItem:
        item = QListWidgetItem(self)
        path_key = str(path)
        item.setData(Qt.UserRole, path_key)
        item.setData(Qt.UserRole + 2, visual)
        if pixmap is not None:
            item.setData(Qt.DecorationRole, pixmap)
        item.setText("")
        self._items_by_path[path_key] = item
        return item

    def clear(self) -> None:
        self._items_by_path.clear()
        super().clear()

    def get_path(self, item: QListWidgetItem) -> Path:
        return Path(item.data(Qt.UserRole))

    def item_for_path(self, path: Path) -> QListWidgetItem | None:
        return self._items_by_path.get(str(path))

    def update_visual(self, path: Path, visual: _ThumbnailVisual) -> None:
        item = self.item_for_path(path)
        if item is None:
            return
        item.setData(Qt.UserRole + 2, visual)
        self.update(self.indexFromItem(item))

    def update_thumbnail(self, path: Path, pixmap: QPixmap) -> None:
        item = self.item_for_path(path)
        if item is None:
            return
        item.setData(Qt.DecorationRole, pixmap)
        self.update(self.indexFromItem(item))

    def remove_path(self, path: Path) -> None:
        """Remove the item for ``path`` from the grid and internal index."""
        item = self.item_for_path(path)
        if item is None:
            return
        row = self.row(item)
        self.takeItem(row)
        self._items_by_path.pop(str(path), None)

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        if item is None:
            return
        selected = self.selectedItems()
        if item not in selected:
            paths = [Path(item.data(Qt.UserRole))]
        else:
            paths = [Path(it.data(Qt.UserRole)) for it in selected]
        menu = QMenu(self)
        label = f"删除图片 ({len(paths)} 张)" if len(paths) > 1 else "删除图片"
        act = menu.addAction(label)
        act.triggered.connect(lambda: self.delete_images_requested.emit(paths))
        menu.exec_(event.globalPos())


def _empty_ia(path: Path) -> ImageAnnotation:
    return ImageAnnotation(image_path=path.name, image_size=(1, 1))


# ── Preview pane ───────────────────────────────────────────────


class PreviewPane(QFrame):
    """Right-side resizable, closable preview panel."""

    closed = pyqtSignal()
    # Per-image user-tag edits — payload is the new list[str].
    user_tags_changed = pyqtSignal(list)

    def __init__(self, image_cache: ImageCache, parent=None):
        super().__init__(parent)
        self._image_cache = image_cache
        self._project: ProjectManager | None = None
        self._current_path: Path | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        from src.ui.tag_widget import TagChipBar

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        self._filename_lbl = QLabel("—")
        self._filename_lbl.setStyleSheet(text_style("small"))
        top.addWidget(self._filename_lbl, 1)
        self._btn_close = QPushButton(icon("cancel"), "")
        self._btn_close.setFixedSize(20, 20)
        set_button_role(self._btn_close, "icon-danger")
        self._btn_close.clicked.connect(self.closed.emit)
        top.addWidget(self._btn_close)
        layout.addLayout(top)

        self._image_lbl = QLabel()
        self._image_lbl.setAlignment(Qt.AlignCenter)
        self._image_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_lbl.setMinimumSize(120, 120)
        layout.addWidget(self._image_lbl, 1)

        self._meta_lbl = QLabel("—")
        self._meta_lbl.setStyleSheet(text_style("hint"))
        layout.addWidget(self._meta_lbl)

        # Per-image user tag editor — wires its chip-bar signal upward.
        self._tag_bar = TagChipBar()
        self._tag_bar.tags_changed.connect(self.user_tags_changed)
        layout.addWidget(self._tag_bar)

    def set_available_tags(self, tags: list[str]) -> None:
        self._tag_bar.set_available_tags(tags)

    def set_user_tags(self, tags: list[str]) -> None:
        """Programmatically set the preview pane's user-tag chips.

        Blocks signals so this push doesn't look like a user edit.
        """
        self._tag_bar.blockSignals(True)
        try:
            self._tag_bar.set_tags(list(tags))
        finally:
            self._tag_bar.blockSignals(False)

    def set_project(self, project: ProjectManager | None) -> None:
        self._project = project

    def set_image(self, path: Path | None) -> None:
        self._current_path = path
        if path is None:
            self._filename_lbl.setText("—")
            self._image_lbl.clear()
            self._meta_lbl.setText("—")
            return
        self._filename_lbl.setText(path.name)
        pix = self._image_cache.get(path)
        if pix is None:
            pix = load_pixmap(path)
        if pix is not None and not pix.isNull():
            scaled = pix.scaled(
                self._image_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._image_lbl.setPixmap(scaled)
        else:
            self._image_lbl.clear()
        ia = (
            load_annotation(self._project.label_path_for(path))
            if self._project is not None
            else None
        )
        if ia is not None and ia.image_tags:
            cls = ia.image_tags[0]
            status = "已确认" if ia.image_tags_confirmed else "待确认"
            self._meta_lbl.setText(
                f"类别: {cls}  |  状态: {status}  |  来源: {ia.image_tags_source}"
            )
        else:
            self._meta_lbl.setText("未标")
        # Update tag chips — block signals so loading doesn't look like an edit.
        self._tag_bar.blockSignals(True)
        try:
            self._tag_bar.set_tags(list(ia.tags) if ia is not None else [])
        finally:
            self._tag_bar.blockSignals(False)


# ── Class button bar ───────────────────────────────────────────


class ClassButtonBar(QWidget):
    """Horizontal strip of class buttons, with 1-9 hint for the first 9."""

    class_clicked = pyqtSignal(str)
    add_class_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: list[str] = []
        self._colors: dict[str, str] = {}
        self._buttons: dict[str, QPushButton] = {}

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._order_hint = QLabel("?")
        self._order_hint.setAlignment(Qt.AlignCenter)
        self._order_hint.setFixedSize(18, 18)
        self._order_hint.setToolTip(
            "此处类别顺序仅代表快捷键顺序，训练时采用 yolo 分类训练的默认类别顺序（按字母排序）"
        )
        self._order_hint.setStyleSheet(
            f"QLabel {{ color:{PALETTE['text_muted']}; border:1px solid {PALETTE['line_strong']}; "
            "border-radius:9px; font-size:11px; font-weight:600; }"
            f"QLabel:hover {{ color:{PALETTE['primary']}; border-color:{PALETTE['primary']}; }}"
        )
        self._add_btn = QPushButton("+ 添加类别")
        self._add_btn.clicked.connect(self.add_class_clicked.emit)

    def set_classes(self, classes: list[str], colors: dict[str, str]) -> None:
        self._classes = list(classes)
        self._colors = dict(colors)

        # Clear current items but keep _add_btn alive for reuse.
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None and w not in (self._add_btn, self._order_hint):
                w.deleteLater()
        self._buttons.clear()

        for idx, cls in enumerate(self._classes):
            label = f"{idx + 1}  {cls}" if idx < 9 else cls
            btn = QPushButton(label)
            color = self._colors.get(cls, PALETTE["primary"])
            btn.setStyleSheet(
                f"QPushButton {{ background:{color}; color:{PALETTE['ink']}; padding:6px 10px; "
                f"border:1px solid {color}; border-radius:5px; font-weight:650; }}"
                f"QPushButton:hover {{ border-color:{PALETTE['text']}; }}"
            )
            btn.clicked.connect(lambda _checked=False, c=cls: self.class_clicked.emit(c))
            self._layout.addWidget(btn)
            self._buttons[cls] = btn

        self._layout.addStretch(1)
        self._layout.addWidget(self._order_hint)
        self._layout.addWidget(self._add_btn)


# ── ClassifyView ───────────────────────────────────────────────


class ClassifyView(TaskView):
    """Thumbnail grid view for classification projects."""

    def __init__(
        self,
        image_cache: ImageCache,
        undo_stacks: "OrderedDict[str, UndoStack]",
        parent=None,
    ):
        super().__init__(parent)
        self._image_cache = image_cache
        self._undo_stacks = undo_stacks
        self._project: ProjectManager | None = None
        self._classes: list[str] = []
        self._class_colors: dict[str, str] = {}
        self._status_filter: str | None = None
        self._class_filter: str | None = None
        self._tag_filter = None  # TagFilter | None
        self._bulk_auto_label_updates = 0
        self._confirm_count_dirty = False
        self._loader = ThumbnailLoader()
        self._loader.loaded.connect(self._on_thumbnail_loaded)

        # Auto-cleanup on C++ destruction. The closure captures the loader so
        # it works even after Python attribute lookup is unsafe.
        loader_ref = self._loader

        def _cleanup_on_destroy(*_):
            if sip.isdeleted(loader_ref):
                return
            loader_ref.stop()
            loader_ref.wait(2000)

        self.destroyed.connect(_cleanup_on_destroy)
        self._init_ui()
        self.image_focus_changed.connect(self._preview.set_image)

    def cleanup(self) -> None:
        """Stop background loader. Call before discarding view."""
        loader = getattr(self, "_loader", None)
        if loader is None:
            return
        if sip.isdeleted(loader):
            self._loader = None
            return
        loader.stop()
        loader.wait(2000)

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        self.cleanup()
        super().closeEvent(event)

    def hideEvent(self, event):  # noqa: N802 (Qt naming)
        # Drain loader when the view is hidden (e.g. tab switch). Restarts on next enqueue.
        loader = getattr(self, "_loader", None)
        if loader is not None and loader.isRunning():
            loader.stop()
            loader.wait(2000)
            loader._stop = False  # rearm for future enqueues
        super().hideEvent(event)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # View toolbar (density slider + sort combo wired in 10.2).
        view_tb = QToolBar()
        view_tb.addWidget(QLabel(" 密度: "))
        self._density_slider = QSlider(Qt.Horizontal)
        self._density_slider.setMinimum(64)
        self._density_slider.setMaximum(192)
        self._density_slider.setSingleStep(8)
        self._density_slider.setPageStep(32)
        self._density_slider.setFixedWidth(120)
        self._density_slider.setValue(96)
        self._density_slider.valueChanged.connect(self._on_density_changed)
        view_tb.addWidget(self._density_slider)

        view_tb.addSeparator()
        view_tb.addWidget(QLabel(" 排序: "))
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("按文件名", "filename")
        self._sort_combo.addItem("按类别", "class")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        view_tb.addWidget(self._sort_combo)

        view_tb.addSeparator()
        self._confirm_all_btn = QPushButton(icon("check_all"), "确认全部")
        self._confirm_all_btn.setToolTip("确认所有当前可见的、由模型自动标注且未确认的图片")
        self._confirm_all_btn.setEnabled(False)
        self._confirm_all_btn.clicked.connect(self._on_confirm_all_clicked)
        set_button_role(self._confirm_all_btn, "primary")
        view_tb.addWidget(self._confirm_all_btn)

        self._preview_toggle_btn = QPushButton(icon("eye"), "预览")
        self._preview_toggle_btn.setCheckable(True)
        self._preview_toggle_btn.setChecked(True)
        self._preview_toggle_btn.setToolTip("显示或隐藏右侧大图预览面板")
        self._preview_toggle_btn.toggled.connect(self._on_preview_toggle)
        set_button_role(self._preview_toggle_btn, "secondary")
        view_tb.addWidget(self._preview_toggle_btn)

        layout.addWidget(view_tb)

        self._splitter = QSplitter(Qt.Horizontal, self)
        self._grid = ThumbnailGridWidget()
        self._splitter.addWidget(self._grid)

        self._preview = PreviewPane(self._image_cache)
        self._preview.closed.connect(self._on_preview_close)
        self._preview.user_tags_changed.connect(self._on_preview_user_tags_changed)
        self._splitter.addWidget(self._preview)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        layout.addWidget(self._splitter, 1)

        self._class_bar = ClassButtonBar()
        self._class_bar.class_clicked.connect(self._apply_class)
        self._class_bar.add_class_clicked.connect(self._on_add_class_clicked)
        layout.addWidget(self._class_bar)

        self._grid.currentItemChanged.connect(self._on_grid_focus_changed)
        self._grid.delete_images_requested.connect(self._on_delete_images)

    def _on_density_changed(self, value: int) -> None:
        self._grid.setIconSize(QSize(value, value))
        self._grid.doItemsLayout()
        cfg_path = _APP_CONFIG_PATH()
        cfg = AppConfig.load(cfg_path)
        cfg.classify_grid_density = value
        cfg.save(cfg_path)

    def _on_sort_changed(self, _idx: int) -> None:
        if self._project is None:
            return
        sort_key = self._sort_combo.currentData()
        self._resort_items(sort_key)
        cfg_path = _APP_CONFIG_PATH()
        cfg = AppConfig.load(cfg_path)
        cfg.classify_grid_sort = sort_key
        cfg.save(cfg_path)

    def _resort_items(self, sort_key: str) -> None:
        if self._project is None or self._grid.count() == 0:
            return
        rows = []
        for i in range(self._grid.count()):
            it = self._grid.item(i)
            path = Path(it.data(Qt.UserRole))
            ia = load_annotation(self._project.label_path_for(path))
            tag = ia.image_tags[0] if (ia and ia.image_tags) else ""
            rows.append((path, tag, it))

        if sort_key == "class":
            # Unlabeled (empty tag) first; within each tag, sort by filename.
            rows.sort(key=lambda r: (0 if r[1] == "" else 1, r[1], r[0].name))
        else:
            rows.sort(key=lambda r: r[0].name)

        snapshots = []
        for _, _, it in rows:
            snapshots.append((
                Path(it.data(Qt.UserRole)),
                it.data(Qt.UserRole + 2),
                it.data(Qt.DecorationRole),
            ))
        self._grid.clear()
        for path, visual, pix in snapshots:
            self._grid.add_image_item(path, visual, pixmap=pix)

    def _on_grid_focus_changed(self, current, _prev) -> None:
        if current is None:
            return
        self.image_focus_changed.emit(Path(current.data(Qt.UserRole)))

    # ── TaskView protocol ──────────────────────────────────────

    def set_project(self, project: ProjectManager) -> None:
        self._project = project
        self._preview.set_project(project)

        # Apply persisted density before loading items so iconSize is correct.
        cfg = AppConfig.load(_APP_CONFIG_PATH())
        if self._density_slider.value() != cfg.classify_grid_density:
            self._density_slider.setValue(cfg.classify_grid_density)
        else:
            self._on_density_changed(cfg.classify_grid_density)

        self._grid.clear()
        icon_size = self._grid.iconSize()
        for img in project.list_images():
            ia = load_annotation(project.label_path_for(img)) or _empty_ia(img)
            visual = _compute_visual_state(ia, self._class_colors)
            self._grid.add_image_item(img, visual, pixmap=None)
            self._loader.enqueue(img, icon_size)

        # Apply persisted sort (after items are present).
        target_idx = 0 if cfg.classify_grid_sort == "filename" else 1
        if self._sort_combo.currentIndex() != target_idx:
            self._sort_combo.setCurrentIndex(target_idx)
        elif cfg.classify_grid_sort == "class":
            # Same index → no signal, but we still need to reorder.
            self._resort_items("class")

        self._apply_persisted_preview_state()
        self._update_confirm_all_count()

    def _apply_persisted_preview_state(self) -> None:
        cfg = AppConfig.load(_APP_CONFIG_PATH())
        self._preview.setVisible(cfg.classify_preview_visible)
        if cfg.classify_preview_visible:
            total = sum(self._splitter.sizes()) or 800
            self._splitter.setSizes([max(total - cfg.classify_preview_width, 100),
                                     cfg.classify_preview_width])
        self._sync_preview_toggle(cfg.classify_preview_visible)

    def _on_preview_close(self) -> None:
        sizes = self._splitter.sizes()
        width = sizes[1] if len(sizes) >= 2 and sizes[1] > 50 else 320
        self._save_preview_state(width=width, visible=False)
        self._preview.setVisible(False)
        self._sync_preview_toggle(False)

    def _on_preview_user_tags_changed(self, new_tags: list) -> None:
        """Persist user-tag edits made through the preview pane's chip bar."""
        if self._project is None:
            return
        path = self._preview._current_path
        if path is None:
            return
        label_path = self._project.label_path_for(path)
        ia = load_annotation(label_path)
        if ia is None:
            from src.utils.image import get_image_size
            w, h = get_image_size(path)
            ia = ImageAnnotation(image_path=path.name, image_size=(w, h))
        ia.tags = [str(t) for t in new_tags]
        save_annotation(ia, label_path)
        self.user_tags_changed.emit(path, list(ia.tags))

    def show_preview(self) -> None:
        cfg = AppConfig.load(_APP_CONFIG_PATH())
        w = cfg.classify_preview_width
        total = sum(self._splitter.sizes()) or 800
        self._splitter.setSizes([max(total - w, 100), w])
        self._preview.setVisible(True)
        self._save_preview_state(width=w, visible=True)
        self._sync_preview_toggle(True)

    def _on_preview_toggle(self, checked: bool) -> None:
        if checked:
            self.show_preview()
        else:
            self._on_preview_close()

    def _sync_preview_toggle(self, checked: bool) -> None:
        btn = getattr(self, "_preview_toggle_btn", None)
        if btn is None or btn.isChecked() == checked:
            return
        btn.blockSignals(True)
        btn.setChecked(checked)
        btn.blockSignals(False)

    def _save_preview_state(self, width: int, visible: bool) -> None:
        cfg_path = _APP_CONFIG_PATH()
        cfg = AppConfig.load(cfg_path)
        cfg.classify_preview_width = width
        cfg.classify_preview_visible = visible
        cfg.save(cfg_path)

    # ── Keyboard input ─────────────────────────────────────────

    def keyPressEvent(self, event):  # noqa: N802 (Qt naming)
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            if idx < len(self._classes):
                self._apply_class(self._classes[idx])
                return
        if key == Qt.Key_Space:
            self._confirm_focused_or_selected()
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._clear_selected_tags()
            return
        if key == Qt.Key_Escape:
            self._grid.clearSelection()
            return
        super().keyPressEvent(event)

    def _apply_class(self, class_name: str) -> None:
        if not self._project:
            return
        selected_items = self._grid.selectedItems()
        if len(selected_items) <= 1:
            current = self._grid.currentItem()
            if current is None:
                return
            self._set_image_tag(current, class_name, source="manual", confirmed=True)
            row = self._grid.currentRow()
            for next_row in range(row + 1, self._grid.count()):
                if not self._grid.item(next_row).isHidden():
                    self._grid.setCurrentRow(next_row)
                    break
        else:
            for item in selected_items:
                self._set_image_tag(item, class_name, source="manual", confirmed=True)

    def _set_image_tag(
        self,
        item,
        class_name: str,
        source: str,
        confirmed: bool,
    ) -> None:
        path = Path(item.data(Qt.UserRole))
        label_path = self._project.label_path_for(path)
        ia = load_annotation(label_path)
        if ia is None:
            w, h = get_image_size(path)
            ia = ImageAnnotation(image_path=path.name, image_size=(w, h))
        ia.image_tags = [class_name]
        ia.image_tags_confirmed = confirmed
        ia.image_tags_source = source
        save_annotation(ia, label_path)
        visual = _compute_visual_state(ia, self._class_colors)
        self._grid.update_visual(path, visual)
        self._push_undo_for(path)
        self.annotations_changed.emit(path)
        self._request_confirm_all_count_update()

    def _selected_or_focused_items(self) -> list:
        items = self._grid.selectedItems()
        if items:
            return items
        current = self._grid.currentItem()
        return [current] if current is not None else []

    def _confirm_focused_or_selected(self) -> None:
        if not self._project:
            return
        for item in self._selected_or_focused_items():
            path = Path(item.data(Qt.UserRole))
            label_path = self._project.label_path_for(path)
            ia = load_annotation(label_path)
            if ia is None or not ia.image_tags or ia.image_tags_confirmed:
                continue
            ia.image_tags_confirmed = True
            save_annotation(ia, label_path)
            self._grid.update_visual(path, _compute_visual_state(ia, self._class_colors))
            self._push_undo_for(path)
            self.annotations_changed.emit(path)
        self._request_confirm_all_count_update()

    def _clear_selected_tags(self) -> None:
        if not self._project:
            return
        for item in self._selected_or_focused_items():
            path = Path(item.data(Qt.UserRole))
            label_path = self._project.label_path_for(path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            ia.image_tags = []
            ia.image_tags_confirmed = True
            ia.image_tags_source = "manual"
            save_annotation(ia, label_path)
            self._grid.update_visual(path, _compute_visual_state(ia, self._class_colors))
            self._push_undo_for(path)
            self.annotations_changed.emit(path)
        self._request_confirm_all_count_update()

    def _on_delete_images(self, paths: list[Path]) -> None:
        """Delete image files and their labels from disk after confirmation."""
        if not self._project or not paths:
            return
        paths = list(paths)

        labeled_count = 0
        for p in paths:
            ia = load_annotation(self._project.label_path_for(p))
            if ia and ia.image_tags:
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

        preview_path = self._preview._current_path
        preview_in_deleted = preview_path is not None and preview_path in paths

        img_n, lbl_n = self._project.delete_images(paths)

        for p in paths:
            self._image_cache.invalidate(p)
            self._undo_stacks.pop(str(p), None)
            self._grid.remove_path(p)

        if preview_in_deleted:
            self._preview.set_image(None)

        self._request_confirm_all_count_update()
        self.status_changed.emit(
            f"已删除 {img_n} 张图片，{lbl_n} 个标注文件"
        )
        logger.info("Deleted %d images, %d labels", img_n, lbl_n)

    def _on_thumbnail_loaded(self, path: Path, pixmap: QPixmap) -> None:
        self._grid.update_thumbnail(path, pixmap)

    def set_class_colors(self, colors: dict[str, str]) -> None:
        self._class_colors = dict(colors)
        if hasattr(self, "_class_bar"):
            self._class_bar.set_classes(self._classes, self._class_colors)

    def set_classes(self, classes: list[str]) -> None:
        self._classes = list(classes)
        if hasattr(self, "_class_bar"):
            self._class_bar.set_classes(self._classes, self._class_colors)

    def set_available_tags(self, tags: list[str]) -> None:
        if hasattr(self, "_preview"):
            self._preview.set_available_tags(tags)

    def _on_add_class_clicked(self) -> None:
        if self._project is None:
            return
        name, ok = QInputDialog.getText(self, "新增类别", "类别名:")
        if not (ok and name.strip()):
            return
        cls = name.strip()
        if cls in self._project.config.classes:
            return
        self._project.add_class(cls)
        self._project.save()
        colors = {
            c: self._project.config.get_class_color(c)
            for c in self._project.config.classes
        }
        self.set_class_colors(colors)
        self.set_classes(self._project.config.classes)
        self.classes_changed.emit()

    def set_filter(self, status: str | None) -> None:
        self._status_filter = status
        self._apply_filters()

    def set_class_filter(self, cls: str | None) -> None:
        self._class_filter = cls
        self._apply_filters()

    def get_selected_image_paths(self) -> list[Path]:
        return [Path(it.data(Qt.UserRole)) for it in self._grid.selectedItems()]

    def refresh_image_tags(self, path: Path, tags: list[str]) -> None:
        if path == self._preview._current_path:
            self._preview.set_user_tags(tags)
        # Per-item visibility update — never re-positions scroll, so a
        # batch tag-apply (which fires this signal once per modified image)
        # leaves the user's view stable. Full _apply_filters() is reserved
        # for explicit filter combo / chip changes.
        self._refresh_item_visibility(path)

    def set_tag_filter(self, tag_filter) -> None:
        self._tag_filter = tag_filter
        self._apply_filters()

    def _apply_filters(self) -> None:
        if self._project is None:
            return
        current_item = self._grid.currentItem()
        for i in range(self._grid.count()):
            it = self._grid.item(i)
            path = Path(it.data(Qt.UserRole))
            ia = load_annotation(self._project.label_path_for(path))
            it.setHidden(self._compute_hidden(ia))
        self._restore_scroll_after_filter(current_item)
        self._request_confirm_all_count_update()

    def _compute_hidden(self, ia: ImageAnnotation | None) -> bool:
        """Pure predicate: should an image be hidden under the active filters?"""
        if self._status_filter is not None:
            cur_status = ia.status if ia is not None else "unlabeled"
            if cur_status != self._status_filter:
                return True
        if self._class_filter is not None:
            tag = ia.image_tags[0] if (ia is not None and ia.image_tags) else None
            if tag != self._class_filter:
                return True
        if (
            self._tag_filter is not None
            and not self._tag_filter.is_empty()
        ):
            user_tags = ia.tags if ia is not None else []
            if not self._tag_filter.matches(user_tags):
                return True
        return False

    def _refresh_item_visibility(self, path: Path) -> None:
        """Re-evaluate one grid item's visibility under current filters.

        Used after per-image state mutations (tag apply, class change). Does
        NOT touch the scrollbar — the bulk recompute path in _apply_filters
        is the one that re-centers on the current item.
        """
        if self._project is None:
            return
        if (
            self._status_filter is None
            and self._class_filter is None
            and (self._tag_filter is None or self._tag_filter.is_empty())
        ):
            return  # No filter active — visibility cannot change.
        item = self._grid.item_for_path(path)
        if item is None:
            return
        ia = load_annotation(self._project.label_path_for(path))
        item.setHidden(self._compute_hidden(ia))

    def _restore_scroll_after_filter(self, current_item: QListWidgetItem | None) -> None:
        """Keep filtering from leaving the thumbnail grid pinned at the bottom."""
        self._grid.doItemsLayout()
        sb = self._grid.verticalScrollBar()
        if current_item is None or current_item.isHidden():
            sb.setValue(0)
            return
        self._grid.scrollToItem(current_item, QAbstractItemView.PositionAtCenter)

    def _count_visible_unconfirmed_auto(self) -> int:
        if self._project is None:
            return 0
        n = 0
        for i in range(self._grid.count()):
            it = self._grid.item(i)
            if it.isHidden():
                continue
            path = Path(it.data(Qt.UserRole))
            ia = load_annotation(self._project.label_path_for(path))
            if (
                ia is not None
                and ia.image_tags
                and not ia.image_tags_confirmed
                and ia.image_tags_source == "auto"
            ):
                n += 1
        return n

    def _update_confirm_all_count(self) -> None:
        n = self._count_visible_unconfirmed_auto()
        self._confirm_all_btn.setText(f"确认全部 ({n})" if n else "确认全部")
        self._confirm_all_btn.setEnabled(n > 0)
        self._confirm_count_dirty = False

    def _request_confirm_all_count_update(self) -> None:
        if self._bulk_auto_label_updates > 0:
            self._confirm_count_dirty = True
            return
        self._update_confirm_all_count()

    def begin_bulk_auto_label_update(self) -> None:
        self._bulk_auto_label_updates += 1

    def end_bulk_auto_label_update(self) -> None:
        if self._bulk_auto_label_updates == 0:
            return
        self._bulk_auto_label_updates -= 1
        if self._bulk_auto_label_updates == 0 and self._confirm_count_dirty:
            self._update_confirm_all_count()

    def _on_confirm_all_clicked(self) -> None:
        if self._project is None:
            return
        for i in range(self._grid.count()):
            it = self._grid.item(i)
            if it.isHidden():
                continue
            path = Path(it.data(Qt.UserRole))
            label_path = self._project.label_path_for(path)
            ia = load_annotation(label_path)
            if (
                ia is None
                or not ia.image_tags
                or ia.image_tags_confirmed
                or ia.image_tags_source != "auto"
            ):
                continue
            ia.image_tags_confirmed = True
            save_annotation(ia, label_path)
            self._grid.update_visual(path, _compute_visual_state(ia, self._class_colors))
            self._push_undo_for(path)
            self.annotations_changed.emit(path)
        self._request_confirm_all_count_update()

    def get_focused_image(self) -> Path | None:
        item = self._grid.currentItem()
        return Path(item.data(Qt.UserRole)) if item is not None else None

    def get_visible_paths(self) -> list[Path]:
        return [
            Path(self._grid.item(i).data(Qt.UserRole))
            for i in range(self._grid.count())
            if not self._grid.item(i).isHidden()
        ]

    def get_all_paths(self) -> list[Path]:
        return [
            Path(self._grid.item(i).data(Qt.UserRole))
            for i in range(self._grid.count())
        ]

    def reload_current(self) -> None:
        cur = self.get_focused_image()
        if cur is None or self._project is None:
            return
        ia = load_annotation(self._project.label_path_for(cur))
        visual = _compute_visual_state(ia or _empty_ia(cur), self._class_colors)
        self._grid.update_visual(cur, visual)

    def commit_pending_save(self) -> None:
        # Grid operations write through immediately; nothing to flush.
        return

    # ── Undo / redo ────────────────────────────────────────────

    def _push_undo_for(self, path: Path) -> None:
        if self._project is None:
            return
        ia = load_annotation(self._project.label_path_for(path))
        if ia is None:
            return
        key = str(path)
        stack = self._undo_stacks.get(key)
        if stack is None:
            stack = UndoStack()
            self._undo_stacks[key] = stack
        stack.push(ia.to_dict())
        self._undo_stacks.move_to_end(key)
        from src.ui.label_panel import LabelPanel  # avoid circular import
        while len(self._undo_stacks) > LabelPanel._UNDO_MAX_IMAGES:
            self._undo_stacks.popitem(last=False)

    def undo(self) -> None:
        path = self.get_focused_image()
        if path is None:
            return
        stack = self._undo_stacks.get(str(path))
        if stack is None or not stack.can_undo:
            return
        state = stack.undo()
        self._restore_state(path, state)

    def redo(self) -> None:
        path = self.get_focused_image()
        if path is None:
            return
        stack = self._undo_stacks.get(str(path))
        if stack is None or not stack.can_redo:
            return
        state = stack.redo()
        self._restore_state(path, state)

    def _restore_state(self, path: Path, state: dict) -> None:
        ia = ImageAnnotation.from_dict(state)
        save_annotation(ia, self._project.label_path_for(path))
        self._grid.update_visual(path, _compute_visual_state(ia, self._class_colors))
        self.annotations_changed.emit(path)
        self._request_confirm_all_count_update()

    def add_auto_annotations(self, anns, iou: float = 0.5) -> None:
        raise NotImplementedError("Use add_auto_class_prediction for classify view")

    def add_auto_class_prediction(
        self, path: Path, class_name: str, confidence: float,
    ) -> bool:
        """Set image_tags from AI prediction with confirmed=False, source=auto.

        Returns True if applied; False if skipped because the image already has
        a confirmed manual tag (protects user-verified labels from overwrite).
        """
        if self._project is None:
            return False
        label_path = self._project.label_path_for(path)
        ia = load_annotation(label_path)
        if ia is not None and ia.image_tags and ia.image_tags_confirmed:
            return False
        if ia is None:
            w, h = get_image_size(path)
            ia = ImageAnnotation(image_path=path.name, image_size=(w, h))
        ia.image_tags = [class_name]
        ia.image_tags_confirmed = False
        ia.image_tags_source = "auto"
        save_annotation(ia, label_path)
        self._grid.update_visual(path, _compute_visual_state(ia, self._class_colors))
        self._push_undo_for(path)
        self.annotations_changed.emit(path)
        self._request_confirm_all_count_update()
        return True
