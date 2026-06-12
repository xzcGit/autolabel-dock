"""Dialogs — new project, export, class management."""
from __future__ import annotations

import os
import re
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QGroupBox,
    QDialogButtonBox,
    QProgressBar,
    QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from src.ui.theme import PALETTE, text_style

_ERROR_STYLE = text_style("error")
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_PROJECT_NAME_LEN = 64
_MAX_CLASS_NAME_LEN = 32


class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("项目名称")
        form.addRow("项目名称:", self._name_edit)

        # Task type selector
        self._task_type_combo = QComboBox()
        self._task_type_combo.addItems(["detect", "pose", "classify"])
        self._task_type_combo.setCurrentText("detect")
        form.addRow("任务类型:", self._task_type_combo)

        dir_layout = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("选择项目目录")
        dir_layout.addWidget(self._dir_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        form.addRow("项目目录:", dir_layout)

        img_dir_layout = QHBoxLayout()
        self._image_dir_edit = QLineEdit()
        self._image_dir_edit.setPlaceholderText("留空则在项目目录下创建 images/")
        img_dir_layout.addWidget(self._image_dir_edit)
        btn_img_browse = QPushButton("浏览...")
        btn_img_browse.clicked.connect(self._browse_image_dir)
        img_dir_layout.addWidget(btn_img_browse)
        form.addRow("图片目录:", img_dir_layout)

        self._classes_edit = QLineEdit()
        self._classes_edit.setPlaceholderText("逗号分隔，如: cat, dog, bird")
        form.addRow("初始类别:", self._classes_edit)

        layout.addLayout(form)

        # Validation error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(_ERROR_STYLE)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting."""
        name = self._name_edit.text().strip()
        proj_dir = self._dir_edit.text().strip()
        if not name:
            self._error_label.setText("请输入项目名称")
            self._name_edit.setFocus()
            return
        if len(name) > _MAX_PROJECT_NAME_LEN:
            self._error_label.setText(f"项目名称不能超过{_MAX_PROJECT_NAME_LEN}个字符")
            self._name_edit.setFocus()
            return
        if _INVALID_NAME_CHARS.search(name):
            self._error_label.setText("项目名称包含非法字符")
            self._name_edit.setFocus()
            return
        if not proj_dir:
            self._error_label.setText("请选择项目目录")
            self._dir_edit.setFocus()
            return
        proj_path = Path(proj_dir)
        if not proj_path.exists():
            self._error_label.setText("项目目录不存在")
            self._dir_edit.setFocus()
            return
        if not os.access(str(proj_path), os.W_OK):
            self._error_label.setText("项目目录没有写入权限")
            self._dir_edit.setFocus()
            return
        # Validate image dir if provided
        image_dir = self._image_dir_edit.text().strip()
        if image_dir:
            img_path = Path(image_dir)
            if not img_path.exists():
                self._error_label.setText("图片目录不存在")
                self._image_dir_edit.setFocus()
                return
        self._error_label.setText("")
        self.accept()

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self._dir_edit.setText(path)

    def _browse_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if path:
            self._image_dir_edit.setText(path)

    def get_values(self) -> tuple[str, str, str, list[str], str]:
        """Return (name, project_dir, image_dir, classes, task_type)."""
        name = self._name_edit.text().strip()
        proj_dir = self._dir_edit.text().strip()
        image_dir = self._image_dir_edit.text().strip()
        classes_text = self._classes_edit.text().strip()
        classes = [c.strip() for c in classes_text.split(",") if c.strip()] if classes_text else []
        task_type = self._task_type_combo.currentText()
        return name, proj_dir, image_dir, classes, task_type


class ExportDialog(QDialog):
    """Dialog for exporting annotations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出标注")
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._format_combo = QComboBox()
        from src.core.formats import get_export_registry
        self._format_combo.addItems(get_export_registry().list_names())
        form.addRow("导出格式:", self._format_combo)

        dir_layout = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("选择输出目录")
        dir_layout.addWidget(self._dir_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        form.addRow("输出目录:", dir_layout)

        self._confirmed_only = QCheckBox("仅导出已确认标注")
        self._confirmed_only.setChecked(False)
        form.addRow("", self._confirmed_only)

        layout.addLayout(form)

        # Validation error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(_ERROR_STYLE)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting."""
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            self._error_label.setText("请选择输出目录")
            self._dir_edit.setFocus()
            return
        out_path = Path(out_dir)
        if not out_path.exists():
            self._error_label.setText("输出目录不存在")
            self._dir_edit.setFocus()
            return
        if not os.access(out_dir, os.W_OK):
            self._error_label.setText("输出目录没有写入权限")
            self._dir_edit.setFocus()
            return
        self._error_label.setText("")
        self.accept()

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self._dir_edit.setText(path)

    def get_values(self) -> tuple[str, str, bool]:
        """Return (format, output_dir, only_confirmed)."""
        return (
            self._format_combo.currentText(),
            self._dir_edit.text().strip(),
            self._confirmed_only.isChecked(),
        )


class ClassManagerDialog(QDialog):
    """Dialog for managing project classes."""

    def __init__(self, classes: list[str], colors: dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别管理")
        self.setMinimumWidth(350)
        self._classes = list(classes)
        self._colors = dict(colors)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Class list
        self._class_list = QListWidget()
        for cls in self._classes:
            color = self._colors.get(cls, PALETTE["primary"])
            item = QListWidgetItem(cls)
            item.setForeground(QColor(color))
            self._class_list.addItem(item)
        layout.addWidget(self._class_list)

        # Add class
        add_layout = QHBoxLayout()
        self._new_class_edit = QLineEdit()
        self._new_class_edit.setPlaceholderText("新类别名称")
        self._new_class_edit.returnPressed.connect(self._on_add)
        add_layout.addWidget(self._new_class_edit)
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self._on_add)
        add_layout.addWidget(btn_add)
        layout.addLayout(add_layout)

        # Status label for duplicate warning
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_ERROR_STYLE)
        layout.addWidget(self._status_label)

        # Remove button
        btn_remove = QPushButton("删除选中类别")
        btn_remove.clicked.connect(self._on_remove)
        layout.addWidget(btn_remove)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_add(self) -> None:
        name = self._new_class_edit.text().strip()
        if not name:
            return
        if len(name) > _MAX_CLASS_NAME_LEN:
            self._status_label.setText(f"类别名称不能超过{_MAX_CLASS_NAME_LEN}个字符")
            return
        if _INVALID_NAME_CHARS.search(name):
            self._status_label.setText("类别名称包含非法字符")
            return
        if name in self._classes:
            self._status_label.setText(f"类别 \"{name}\" 已存在")
            return
        self._status_label.setText("")
        self._classes.append(name)
        item = QListWidgetItem(name)
        item.setForeground(QColor(PALETTE["primary"]))
        self._class_list.addItem(item)
        self._new_class_edit.clear()

    def _on_remove(self) -> None:
        row = self._class_list.currentRow()
        if row >= 0:
            cls = self._classes.pop(row)
            self._colors.pop(cls, None)
            self._class_list.takeItem(row)
            self._status_label.setText("")

    def get_classes(self) -> list[str]:
        """Return the current class list."""
        return list(self._classes)


class ImportDialog(QDialog):
    """Dialog for importing annotations from external formats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入标注")
        self.setMinimumWidth(450)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Format selector
        self._format_combo = QComboBox()
        from src.core.formats import get_import_registry
        self._import_registry = get_import_registry()
        for info in self._import_registry.list_info():
            self._format_combo.addItem(info.label, info.name)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("导入格式:", self._format_combo)

        # Path selector
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("选择标注文件或目录")
        path_layout.addWidget(self._path_edit)
        self._btn_browse = QPushButton("浏览...")
        self._btn_browse.clicked.connect(self._browse_path)
        path_layout.addWidget(self._btn_browse)
        form.addRow("标注路径:", path_layout)

        # Conflict mode
        self._conflict_combo = QComboBox()
        self._conflict_combo.addItem("跳过已有标注", "skip")
        self._conflict_combo.addItem("覆盖已有标注", "overwrite")
        self._conflict_combo.addItem("合并（追加）", "merge")
        form.addRow("冲突处理:", self._conflict_combo)

        layout.addLayout(form)

        # Help text
        self._help_label = QLabel("")
        self._help_label.setStyleSheet(text_style("hint"))
        self._help_label.setWordWrap(True)
        layout.addWidget(self._help_label)
        self._on_format_changed()

        # Validation error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(_ERROR_STYLE)
        layout.addWidget(self._error_label)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_format_changed(self) -> None:
        name = self._format_combo.currentData()
        info = self._import_registry.get(name)
        if not info:
            return
        is_full = getattr(info, "is_full_import", False)
        self._conflict_combo.setEnabled(not is_full)
        if is_full:
            self._path_edit.setPlaceholderText("选择 ImageFolder 数据集根目录")
            self._help_label.setText(
                "ImageFolder: 选择含 train/cat/, train/dog/ 等子目录或直接 cat/, dog/ 子目录的根目录。"
                "图片将被复制到项目并自动生成分类标注。"
            )
        elif info.input_is_file:
            self._path_edit.setPlaceholderText("选择标注 JSON 文件")
            self._help_label.setText("COCO: 选择包含标注信息的 JSON 文件")
        elif name == "YOLO":
            self._path_edit.setPlaceholderText("选择包含 txt 标注的目录")
            self._help_label.setText(
                "YOLO: 选择包含 .txt 标注文件的目录。"
                "自动识别检测/关键点格式。"
                "若目录或上级目录有 data.yaml 会自动读取类别名。"
            )
        else:
            self._path_edit.setPlaceholderText("选择包含 JSON 标注的目录")
            self._help_label.setText("labelme: 选择包含 .json 标注文件的目录")
        self._path_edit.clear()

    def _browse_path(self) -> None:
        name = self._format_combo.currentData()
        info = self._import_registry.get(name)
        if not info:
            return
        if info.input_is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择标注文件", "",
                info.file_filter or "所有文件 (*)",
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "选择标注目录")
        if path:
            self._path_edit.setText(path)

    def _validate_and_accept(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            self._error_label.setText("请选择标注路径")
            self._path_edit.setFocus()
            return

        name = self._format_combo.currentData()
        info = self._import_registry.get(name)
        if not info:
            return

        p = Path(path)
        if info.input_is_file:
            if not p.is_file():
                self._error_label.setText("文件不存在")
                self._path_edit.setFocus()
                return
        else:
            if not p.is_dir():
                self._error_label.setText("目录不存在")
                self._path_edit.setFocus()
                return

        self._error_label.setText("")
        self.accept()

    def get_values(self) -> tuple[str, str, str]:
        """Return (format_name, path, conflict_mode)."""
        return (
            self._format_combo.currentData(),
            self._path_edit.text().strip(),
            self._conflict_combo.currentData(),
        )


class BatchProgressDialog(QDialog):
    """Progress dialog for batch operations with cancel support."""

    cancelled = pyqtSignal()

    def __init__(self, title: str, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setModal(True)
        self._cancelled = False

        layout = QVBoxLayout(self)

        self._info_label = QLabel(f"处理中: 0/{total}")
        layout.addWidget(self._info_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet(text_style("hint"))
        layout.addWidget(self._detail_label)

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self._btn_cancel)

    def update_progress(self, current: int, total: int) -> None:
        """Update progress bar and label."""
        if self._cancelled:
            return
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._info_label.setText(f"处理中: {current}/{total}")

    def set_detail(self, text: str) -> None:
        """Set detail text (e.g. current file name)."""
        self._detail_label.setText(text)

    def _on_cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._info_label.setText("正在取消，等待当前批次完成...")
        self._btn_cancel.setEnabled(False)
        # Indeterminate "busy" indicator so the user sees the cancel was registered.
        self._progress.setRange(0, 0)
        self.cancelled.emit()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class ClassRegisterDialog(QDialog):
    """Confirm which model classes to register before a batch auto-label run.

    Input: list[ClassPreviewItem]. Output via get_selected(): list of model
    names the user wants to register.
    """

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("登记新类别")
        self.setMinimumWidth(420)
        self._items = list(items)
        self._checkboxes: list[QCheckBox] = []
        self._init_ui()
        self._update_summary()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel(
            f"发现 {len(self._items)} 个模型类别不在当前项目中。\n勾选要登记到项目的类别："
        )
        layout.addWidget(header)

        list_widget = QListWidget()
        for item in self._items:
            cb = QCheckBox(item.model_name)
            cb.setChecked(item.default_checked)
            if item.is_blacklisted:
                cb.setText(f"{item.model_name}    ⚠ 看起来是 ImageNet ID")
                cb.setToolTip("ImageNet 风格 ID（n + 8 位数字），通常需要清洗后再使用。")
                cb.setStyleSheet(text_style("warning"))
            cb.toggled.connect(self._update_summary)
            self._checkboxes.append(cb)
            lw_item = QListWidgetItem(list_widget)
            list_widget.setItemWidget(lw_item, cb)
            lw_item.setSizeHint(cb.sizeHint())
        list_widget.setMinimumHeight(200)
        layout.addWidget(list_widget)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet(text_style("hint"))
        layout.addWidget(self._summary_label)

        btn_row = QHBoxLayout()
        self._btn_select_all = QPushButton("全部勾选")
        self._btn_select_all.clicked.connect(self._on_select_all_clicked)
        btn_row.addWidget(self._btn_select_all)
        self._btn_invert = QPushButton("反选")
        self._btn_invert.clicked.connect(self._on_invert_clicked)
        btn_row.addWidget(self._btn_invert)
        self._btn_only_valid = QPushButton("仅勾选有效项")
        self._btn_only_valid.clicked.connect(self._on_only_valid_clicked)
        btn_row.addWidget(self._btn_only_valid)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("确认")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _update_summary(self) -> None:
        total = len(self._checkboxes)
        checked = sum(1 for cb in self._checkboxes if cb.isChecked())
        self._summary_label.setText(f"共 {total} 项，已勾选 {checked} 项")

    def _on_select_all_clicked(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _on_invert_clicked(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(not cb.isChecked())

    def _on_only_valid_clicked(self) -> None:
        for cb, item in zip(self._checkboxes, self._items):
            cb.setChecked(not item.is_blacklisted)

    def get_selected(self) -> list[str]:
        return [
            item.model_name
            for cb, item in zip(self._checkboxes, self._items)
            if cb.isChecked()
        ]
