"""Annotation list and properties panel."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QInputDialog,
    QSplitter,
    QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from src.core.annotation import Annotation
from src.ui.collapsible_group import CollapsibleGroupBox
from src.ui.tag_widget import TagChipBar
from src.ui.theme import PALETTE, text_style


class _ClassRow(QWidget):
    """Per-row widget for the project class list.

    QListWidget.itemDoubleClicked does not fire when setItemWidget is used —
    the item widget covers the viewport area and intercepts mouse events even
    with WA_TransparentForMouseEvents. So each row carries its own signal.
    """

    double_clicked = pyqtSignal(str)  # class name

    def __init__(self, cls_name: str, parent=None):
        super().__init__(parent)
        self._cls_name = cls_name

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self._cls_name)
        super().mouseDoubleClickEvent(event)


class AnnotationPanel(QWidget):
    """Right-side panel showing project class list, annotation tree, and properties.

    Signals:
        annotation_clicked(str): Annotation ID clicked in the tree.
        keypoint_clicked(str, int): Keypoint clicked — (ann_id, kp_index).
        keypoint_rename_requested(str, int, str): (ann_id, kp_idx, new_label).
        keypoint_visibility_requested(str, int): (ann_id, kp_idx) — cycle visibility.
        keypoint_delete_requested(str, int): (ann_id, kp_idx).
        default_class_changed(str): Class name double-clicked in the project class
            list — caller should treat this as the new default for drawing.
    """

    annotation_clicked = pyqtSignal(str)
    keypoint_clicked = pyqtSignal(str, int)
    keypoint_rename_requested = pyqtSignal(str, int, str)
    keypoint_visibility_requested = pyqtSignal(str, int)
    keypoint_delete_requested = pyqtSignal(str, int)
    # str class name when set, None when cleared (toggled off)
    default_class_changed = pyqtSignal(object)
    # Emitted when the user edits the per-image dataset Tag chip bar.
    image_user_tags_changed = pyqtSignal(list)  # list[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._annotations: list[Annotation] = []
        self._selected_id: str | None = None
        self._selected_kp_idx: int | None = None
        self._classes: list[str] = []
        self._class_colors: dict[str, str] = {}
        self._default_class: str | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        self._splitter = QSplitter(Qt.Vertical, self)
        self._splitter.setChildrenCollapsible(False)
        outer.addWidget(self._splitter, 1)

        self._sections: dict[str, CollapsibleGroupBox] = {}

        # ── [0] 项目类别 ─────────────────────────────────────
        classes_box = CollapsibleGroupBox("项目类别")
        classes_layout = QVBoxLayout()
        classes_layout.setContentsMargins(4, 2, 4, 2)
        self._classes_list = QListWidget()
        self._classes_list.setToolTip(
            "双击设为下次画框/关键点的默认类别；再次双击当前类可取消"
        )
        classes_layout.addWidget(self._classes_list, 1)
        hint = QLabel("双击设/取消默认类")
        hint.setStyleSheet(text_style("hint"))
        classes_layout.addWidget(hint)
        classes_box.set_content_layout(classes_layout)
        self._splitter.addWidget(classes_box)
        self._sections["项目类别"] = classes_box

        # ── [1] 标注列表 ─────────────────────────────────────
        ann_box = CollapsibleGroupBox("标注列表")
        ann_layout = QVBoxLayout()
        ann_layout.setContentsMargins(4, 2, 4, 2)
        self._ann_tree = QTreeWidget()
        self._ann_tree.setHeaderHidden(True)
        self._ann_tree.setIndentation(16)
        self._ann_tree.currentItemChanged.connect(self._on_tree_item_changed)
        self._ann_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._ann_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        ann_layout.addWidget(self._ann_tree, 1)
        ann_box.set_content_layout(ann_layout)
        self._splitter.addWidget(ann_box)
        self._sections["标注列表"] = ann_box

        # ── [2] 属性 ─────────────────────────────────────────
        attr_box = CollapsibleGroupBox("属性")
        attr_layout = QVBoxLayout()
        attr_layout.setContentsMargins(4, 2, 4, 2)
        attr_layout.setSpacing(2)
        # Folded-in: current-image stats line (was a standalone QLabel below the
        # 属性 group in the old layout).
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(text_style("hint"))
        attr_layout.addWidget(self._stats_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {PALETTE['line_strong']};")
        attr_layout.addWidget(sep)

        self._class_label = QLabel("")
        self._conf_label = QLabel("")
        self._status_label = QLabel("")
        self._source_label = QLabel("")
        self._bbox_label = QLabel("")
        for lbl in [self._class_label, self._conf_label, self._status_label,
                    self._source_label, self._bbox_label]:
            lbl.setStyleSheet(text_style("muted"))
            attr_layout.addWidget(lbl)
        attr_layout.addStretch(1)
        attr_box.set_content_layout(attr_layout)
        self._splitter.addWidget(attr_box)
        self._sections["属性"] = attr_box

        # ── [3] Tag ──────────────────────────────────────────
        tag_box = CollapsibleGroupBox("Tag")
        tag_layout = QVBoxLayout()
        tag_layout.setContentsMargins(6, 4, 6, 4)
        self._tag_bar = TagChipBar()
        self._tag_bar.tags_changed.connect(self.image_user_tags_changed)
        tag_layout.addWidget(self._tag_bar)
        tag_hint = QLabel("用于按 tag 筛选数据/训练子集，与分类标签独立。")
        tag_hint.setWordWrap(True)
        tag_hint.setStyleSheet(text_style("hint"))
        tag_layout.addWidget(tag_hint)
        tag_box.set_content_layout(tag_layout)
        self._splitter.addWidget(tag_box)
        self._sections["Tag"] = tag_box
        # Keep the historical attribute name alive in case external code reads it.
        self._tag_group = tag_box

        # ── [4] 项目统计 ─────────────────────────────────────
        stats_box = CollapsibleGroupBox("项目统计")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(4, 2, 4, 2)
        self._project_total_label = QLabel("总图片: 0")
        self._project_labeled_label = QLabel("已标注: 0")
        self._project_confirmed_label = QLabel("全确认: 0")
        self._project_ann_count_label = QLabel("总标注数: 0")
        for lbl in [self._project_total_label, self._project_labeled_label,
                    self._project_confirmed_label, self._project_ann_count_label]:
            lbl.setStyleSheet(text_style("small"))
            stats_layout.addWidget(lbl)
        self._class_dist_label = QLabel("类别分布:")
        self._class_dist_label.setStyleSheet(text_style("hint"))
        stats_layout.addWidget(self._class_dist_label)
        self._class_dist_list = QListWidget()
        self._class_dist_list.setMaximumHeight(120)
        self._class_dist_list.setStyleSheet("font-size: 11px;")
        stats_layout.addWidget(self._class_dist_list)
        stats_box.set_content_layout(stats_layout)
        self._splitter.addWidget(stats_box)
        self._sections["项目统计"] = stats_box
        # Keep historical attribute alive.
        self._stats_group = stats_box

        # Stretch factors: lists absorb expansion, aux panes stay compact.
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setStretchFactor(3, 0)
        self._splitter.setStretchFactor(4, 0)
        self._splitter.setSizes([120, 220, 90, 80, 160])

        # When a section toggles, QSplitter does not auto-redistribute the
        # freed slot to siblings — we have to call setSizes() explicitly.
        for box in self._sections.values():
            box.toggled.connect(self._rebalance_sections)

    def set_classes(self, classes: list[str]) -> None:
        """Set the project's class list (drives the project class panel)."""
        self._classes = list(classes)
        self._rebuild_classes_list()

    # ── State persistence ──────────────────────────────────────

    def _rebalance_sections(self) -> None:
        """Recompute splitter sizes so collapsed panes shrink to their
        header and the freed space is split among expanded panes
        proportional to their current sizes (with stretch factors as a
        fallback when nothing is expanded yet)."""
        count = self._splitter.count()
        sizes = list(self._splitter.sizes())
        new_sizes = [0] * count
        expanded_idx: list[int] = []
        collapsed_total = 0
        for i in range(count):
            box = self._splitter.widget(i)
            if not isinstance(box, CollapsibleGroupBox):
                new_sizes[i] = sizes[i]
                expanded_idx.append(i)
                continue
            if box.isExpanded():
                expanded_idx.append(i)
            else:
                h = max(box.maximumHeight(), box._toggle.sizeHint().height())
                new_sizes[i] = h
                collapsed_total += h
        total = sum(sizes)
        budget = max(total - collapsed_total, 0)
        if expanded_idx:
            current = sum(sizes[i] for i in expanded_idx) or 1
            assigned = 0
            for j, i in enumerate(expanded_idx):
                if j == len(expanded_idx) - 1:
                    new_sizes[i] = max(budget - assigned, 0)
                else:
                    share = int(round(budget * sizes[i] / current))
                    new_sizes[i] = share
                    assigned += share
        self._splitter.setSizes(new_sizes)

    def save_state(self) -> dict:
        """Snapshot splitter sizes + per-section collapsed flags."""
        return {
            "sizes": list(self._splitter.sizes()),
            "collapsed": {
                name: (not box.isExpanded()) for name, box in self._sections.items()
            },
        }

    def restore_state(self, state: dict) -> None:
        """Reapply a previous snapshot. Missing or malformed keys are no-ops."""
        if not isinstance(state, dict):
            return
        # Apply collapsed flags FIRST so any per-pane maximumHeight clamp is
        # in place before QSplitter.setSizes runs (otherwise the freshly-set
        # size for a now-collapsed pane is immediately clipped away).
        collapsed = state.get("collapsed", {})
        if isinstance(collapsed, dict):
            for name, is_collapsed in collapsed.items():
                box = self._sections.get(name)
                if box is None or not isinstance(is_collapsed, bool):
                    continue
                box.setExpanded(not is_collapsed)
        sizes = state.get("sizes")
        if isinstance(sizes, list) and len(sizes) == self._splitter.count():
            try:
                int_sizes = [int(s) for s in sizes]
            except (TypeError, ValueError):
                int_sizes = None
            if int_sizes is not None:
                self._splitter.setSizes(int_sizes)

    def set_class_colors(self, colors: dict[str, str]) -> None:
        """Set class color mapping. Refreshes the project class panel icons."""
        self._class_colors = dict(colors)
        self._rebuild_classes_list()

    def set_default_class(self, class_name: str | None) -> None:
        """Mark a class as the current drawing default; bold it in the list."""
        self._default_class = class_name
        self._refresh_default_highlight()

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Update the annotation tree."""
        self._annotations = list(annotations)
        self._ann_tree.blockSignals(True)
        self._ann_tree.clear()

        for ann in annotations:
            color = QColor(self._class_colors.get(ann.class_name, PALETTE["primary"]))
            status_icon = "\u2713" if ann.confirmed else "\u26a1"
            type_hint = ""
            if ann.bbox and ann.keypoints:
                type_hint = f" [bbox+kp\u00d7{len(ann.keypoints)}]"
            elif ann.bbox:
                type_hint = " [bbox]"
            elif ann.keypoints:
                type_hint = f" [kp\u00d7{len(ann.keypoints)}]"

            top_item = QTreeWidgetItem([f"{status_icon} {ann.class_name}{type_hint}"])
            top_item.setData(0, Qt.UserRole, ann.id)
            top_item.setData(0, Qt.UserRole + 1, -1)  # -1 = annotation level
            top_item.setForeground(0, color)
            self._ann_tree.addTopLevelItem(top_item)

            # Add keypoint children
            for i, kp in enumerate(ann.keypoints):
                vis_names = ["\u25cb", "\u25d1", "\u25cf"]  # empty, half, full circle
                vis_icon = vis_names[kp.visible] if kp.visible < 3 else "?"
                child = QTreeWidgetItem([f"  {vis_icon} {kp.label}"])
                child.setData(0, Qt.UserRole, ann.id)
                child.setData(0, Qt.UserRole + 1, i)  # keypoint index
                child.setForeground(0, color)
                top_item.addChild(child)

            if ann.keypoints:
                top_item.setExpanded(True)

        self._ann_tree.blockSignals(False)
        self._update_stats()
        self._refresh_class_counts()

    def select_annotation(self, ann_id: str | None) -> None:
        """Select an annotation in the tree and show its properties."""
        self._selected_id = ann_id
        self._selected_kp_idx = None
        if ann_id is None:
            self._ann_tree.clearSelection()
            self._clear_properties()
            return

        for i in range(self._ann_tree.topLevelItemCount()):
            item = self._ann_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == ann_id:
                self._ann_tree.blockSignals(True)
                self._ann_tree.setCurrentItem(item)
                self._ann_tree.blockSignals(False)
                break

        ann = self._find_annotation(ann_id)
        if ann:
            self._show_properties(ann)

    def select_keypoint(self, ann_id: str, kp_idx: int) -> None:
        """Select a specific keypoint in the tree."""
        self._selected_id = ann_id
        self._selected_kp_idx = kp_idx

        for i in range(self._ann_tree.topLevelItemCount()):
            top = self._ann_tree.topLevelItem(i)
            if top.data(0, Qt.UserRole) == ann_id:
                top.setExpanded(True)
                if 0 <= kp_idx < top.childCount():
                    self._ann_tree.blockSignals(True)
                    self._ann_tree.setCurrentItem(top.child(kp_idx))
                    self._ann_tree.blockSignals(False)
                break

        ann = self._find_annotation(ann_id)
        if ann and 0 <= kp_idx < len(ann.keypoints):
            self._show_keypoint_properties(ann, kp_idx)

    def set_project_stats(self, stats: dict) -> None:
        """Update project-level statistics."""
        self._project_total_label.setText(f"总图片: {stats.get('total_images', 0)}")
        self._project_labeled_label.setText(f"已标注: {stats.get('labeled_images', 0)}")
        self._project_confirmed_label.setText(f"全确认: {stats.get('confirmed_images', 0)}")
        self._project_ann_count_label.setText(f"总标注数: {stats.get('total_annotations', 0)}")

        self._class_dist_list.clear()
        class_counts = stats.get("class_counts", {})
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            color = self._class_colors.get(cls_name, PALETTE["primary"])
            item = QListWidgetItem(f"{cls_name}: {count}")
            item.setForeground(QColor(color))
            self._class_dist_list.addItem(item)

    def clear(self) -> None:
        """Clear all state."""
        self._annotations = []
        self._selected_id = None
        self._selected_kp_idx = None
        self._ann_tree.clear()
        self._refresh_class_counts()
        self._clear_properties()
        self._stats_label.setText("")
        self._class_dist_list.clear()
        self._tag_bar.set_tags([])

    # ── Per-image user tags (dataset Tag) ─────────────────────

    def set_available_tags(self, tags: list[str]) -> None:
        """Set the project's known-tag registry (populates the chip popup)."""
        self._tag_bar.set_available_tags(tags)

    def set_image_user_tags(self, tags: list[str]) -> None:
        """Populate the per-image chip bar with the loaded image's tags."""
        # Block our own signal so loading a new image doesn't look like an edit.
        self._tag_bar.blockSignals(True)
        try:
            self._tag_bar.set_tags(tags)
        finally:
            self._tag_bar.blockSignals(False)

    def get_image_user_tags(self) -> list[str]:
        return self._tag_bar.get_tags()

    def _show_properties(self, ann: Annotation) -> None:
        self._class_label.setText(f"类别: {ann.class_name}")
        self._conf_label.setText(f"置信度: {ann.confidence:.2f}")
        self._status_label.setText(f"状态: {'已确认' if ann.confirmed else '待确认'}")
        self._source_label.setText(f"来源: {'手动' if ann.source == 'manual' else '自动'}")
        if ann.bbox:
            cx, cy, w, h = ann.bbox
            self._bbox_label.setText(f"Bbox: ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")
        elif ann.keypoints:
            self._bbox_label.setText(f"关键点: {len(ann.keypoints)} 个")
        else:
            self._bbox_label.setText("")

    def _show_keypoint_properties(self, ann: Annotation, kp_idx: int) -> None:
        kp = ann.keypoints[kp_idx]
        vis_names = ["不可见", "被遮挡", "可见"]
        vis = vis_names[kp.visible] if kp.visible < 3 else "?"
        self._class_label.setText(f"关键点: {kp.label}")
        self._conf_label.setText(f"所属: {ann.class_name}")
        self._status_label.setText(f"可见性: {vis}")
        self._source_label.setText(f"坐标: ({kp.x:.4f}, {kp.y:.4f})")
        self._bbox_label.setText(f"索引: {kp_idx}/{len(ann.keypoints)}")

    def _clear_properties(self) -> None:
        self._class_label.setText("")
        self._conf_label.setText("")
        self._status_label.setText("")
        self._source_label.setText("")
        self._bbox_label.setText("")

    def _update_stats(self) -> None:
        total = len(self._annotations)
        confirmed = sum(1 for a in self._annotations if a.confirmed)
        pending = total - confirmed
        self._stats_label.setText(f"标注: {total} | 确认: {confirmed} | 待确认: {pending}")

    def _find_annotation(self, ann_id: str) -> Annotation | None:
        for ann in self._annotations:
            if ann.id == ann_id:
                return ann
        return None

    def _on_tree_item_changed(self, current, previous) -> None:
        if current is None:
            return
        ann_id = current.data(0, Qt.UserRole)
        kp_idx = current.data(0, Qt.UserRole + 1)
        if ann_id is None:
            return
        if kp_idx is not None and kp_idx >= 0:
            self._selected_kp_idx = kp_idx
            self.keypoint_clicked.emit(ann_id, kp_idx)
            ann = self._find_annotation(ann_id)
            if ann:
                self._show_keypoint_properties(ann, kp_idx)
        else:
            self._selected_kp_idx = None
            self.annotation_clicked.emit(ann_id)
            ann = self._find_annotation(ann_id)
            if ann:
                self._show_properties(ann)

    def _on_tree_context_menu(self, pos) -> None:
        item = self._ann_tree.itemAt(pos)
        if not item:
            return
        ann_id = item.data(0, Qt.UserRole)
        kp_idx = item.data(0, Qt.UserRole + 1)
        if ann_id is None:
            return

        ann = self._find_annotation(ann_id)
        if not ann:
            return

        if kp_idx is not None and kp_idx >= 0 and kp_idx < len(ann.keypoints):
            kp = ann.keypoints[kp_idx]
            menu = QMenu(self)

            rename = menu.addAction(f"重命名 ({kp.label})")
            rename.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx, old=kp.label: self._rename_keypoint(aid, ki, old))

            vis_names = ["不可见", "被遮挡", "可见"]
            vis = vis_names[kp.visible] if kp.visible < 3 else "?"
            toggle_vis = menu.addAction(f"切换可见性 ({vis})")
            toggle_vis.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx: self.keypoint_visibility_requested.emit(aid, ki))

            delete = menu.addAction("删除关键点")
            delete.triggered.connect(
                lambda _, aid=ann_id, ki=kp_idx: self.keypoint_delete_requested.emit(aid, ki))

            menu.exec_(self._ann_tree.viewport().mapToGlobal(pos))

    def _rename_keypoint(self, ann_id: str, kp_idx: int, old_label: str) -> None:
        new_label, ok = QInputDialog.getText(self, "重命名关键点", "标签:", text=old_label)
        if ok and new_label.strip():
            self.keypoint_rename_requested.emit(ann_id, kp_idx, new_label.strip())

    # ── Project class panel helpers ────────────────────────────

    @staticmethod
    def _swatch_style(color_hex: str) -> str:
        """Return QSS for a small color swatch that visually echoes the bbox
        stroke color rendered on canvas (same hex source)."""
        return (
            f"background-color: {color_hex};"
            f"border: 1px solid {PALETTE['line_strong']};"
            "border-radius: 2px;"
        )

    def _make_class_row(self, idx: int, cls_name: str, color: str) -> QWidget:
        """Build the per-row widget: swatch + index/name (left) + count (right).

        The row owns its own `double_clicked(str)` signal — see _ClassRow.
        """
        row = _ClassRow(cls_name)
        row.double_clicked.connect(self._on_class_double_clicked)

        hl = QHBoxLayout(row)
        hl.setContentsMargins(4, 1, 6, 1)
        hl.setSpacing(6)

        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(self._swatch_style(color))
        swatch.setObjectName("swatch")

        name_lbl = QLabel(f"{idx}  {cls_name}")
        name_lbl.setStyleSheet(f"color: {color};")
        name_lbl.setObjectName("name_lbl")

        count_lbl = QLabel("×0")
        count_lbl.setStyleSheet(f"color: {color};")
        count_lbl.setObjectName("count_lbl")

        hl.addWidget(swatch)
        hl.addWidget(name_lbl)
        hl.addStretch(1)
        hl.addWidget(count_lbl)
        return row

    def _rebuild_classes_list(self) -> None:
        """Populate the project class panel from `_classes` + `_class_colors`."""
        self._classes_list.blockSignals(True)
        self._classes_list.clear()
        for idx, cls_name in enumerate(self._classes):
            color = self._class_colors.get(cls_name, PALETTE["primary"])
            item = QListWidgetItem()
            item.setData(Qt.UserRole, cls_name)
            self._classes_list.addItem(item)
            row = self._make_class_row(idx, cls_name, color)
            item.setSizeHint(row.sizeHint())
            self._classes_list.setItemWidget(item, row)
        self._classes_list.blockSignals(False)
        self._refresh_class_counts()
        self._refresh_default_highlight()

    def _class_count_text(self, cls_name: str) -> str:
        """`×N` for annotations, suffixed with `(K kp)` when keypoints exist."""
        ann_count = 0
        kp_count = 0
        for a in self._annotations:
            if a.class_name != cls_name:
                continue
            ann_count += 1
            kp_count += len(a.keypoints)
        text = f"×{ann_count}"
        if kp_count > 0:
            text += f"  ({kp_count} kp)"
        return text

    def _refresh_class_counts(self) -> None:
        """Update per-class count labels without rebuilding rows."""
        if self._classes_list.count() != len(self._classes):
            self._rebuild_classes_list()
            return
        for idx, cls_name in enumerate(self._classes):
            item = self._classes_list.item(idx)
            row = self._classes_list.itemWidget(item)
            if row is None:
                continue
            count_lbl = row.findChild(QLabel, "count_lbl")
            if count_lbl is not None:
                count_lbl.setText(self._class_count_text(cls_name))

    def _refresh_default_highlight(self) -> None:
        """Bold the name label of the row matching `_default_class`."""
        for i in range(self._classes_list.count()):
            item = self._classes_list.item(i)
            row = self._classes_list.itemWidget(item)
            if row is None:
                continue
            name_lbl = row.findChild(QLabel, "name_lbl")
            if name_lbl is None:
                continue
            is_default = item.data(Qt.UserRole) == self._default_class
            font = name_lbl.font()
            font.setBold(is_default)
            name_lbl.setFont(font)

    def _on_class_double_clicked(self, cls_name: str) -> None:
        if not cls_name:
            return
        if cls_name == self._default_class:
            # Toggle off: clear default
            self._default_class = None
            self._refresh_default_highlight()
            self.default_class_changed.emit(None)
        else:
            self._default_class = cls_name
            self._refresh_default_highlight()
            self.default_class_changed.emit(cls_name)
