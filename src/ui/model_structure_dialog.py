"""Model structure viewer — read-only dialog showing a YOLO model's layers.

Displays the top-level modules of a YOLO model (index / type / params /
output shape / cumulative param ratio) in a QTreeWidget to help users decide
the training ``freeze`` value. The layer index matches Ultralytics ``freeze=N``
semantics. Pure display, no回填/副作用.

Loading is synchronous (CPU-only parse, ~1-2s incl. a forward pass) with a wait
cursor. Errors are surfaced as a friendly Chinese QMessageBox.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.model_structure import LayerInfo, ModelStructureError
from src.ui.theme import set_button_role, text_style

logger = logging.getLogger(__name__)


def _format_params(n: int) -> str:
    """Human-friendly param count (e.g. 1,234,567 → '1.23M')."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


class ModelStructureDialog(QDialog):
    """Read-only viewer for a YOLO model's layer hierarchy.

    Entry points construct it with the ``ModelController`` and the current
    registry model list, then call :meth:`load_from_path` or
    :meth:`load_from_registry` to preload. The user can also switch models via
    the combo (registry models) or the "打开文件..." button (external .pt).
    """

    _COLUMNS = ["层索引", "类型", "参数量", "输出 shape", "累计参数占比"]

    def __init__(self, controller, model_list=None, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._models = list(model_list or [])
        self.setWindowTitle("模型结构查看器")
        self.setMinimumSize(680, 520)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Model selection row ──
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QLabel("已注册模型:"))

        self._model_combo = QComboBox()
        self._model_combo.addItem("（请选择）", None)
        for m in self._models:
            self._model_combo.addItem(f"[{m.task}] {m.name}", m.id)
        self._model_combo.currentIndexChanged.connect(self._on_combo_changed)
        top_row.addWidget(self._model_combo, 1)

        self._btn_open = QPushButton("打开文件...")
        set_button_role(self._btn_open, "secondary")
        self._btn_open.setToolTip("加载外部 .pt 模型文件查看结构")
        self._btn_open.clicked.connect(self._on_open_file)
        top_row.addWidget(self._btn_open)

        layout.addLayout(top_row)

        # ── Layer tree ──
        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(self._COLUMNS))
        self._tree.setHeaderLabels(self._COLUMNS)
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self._tree, 1)

        # ── Summary + close ──
        bottom_row = QHBoxLayout()
        self._summary = QLabel("提示：层索引与训练 freeze 参数一致（freeze=N 冻结第 0 ~ N-1 层）")
        self._summary.setStyleSheet(text_style("hint"))
        self._summary.setWordWrap(True)
        bottom_row.addWidget(self._summary, 1)

        btn_close = QPushButton("关闭")
        set_button_role(btn_close, "secondary")
        btn_close.clicked.connect(self.accept)
        bottom_row.addWidget(btn_close)
        layout.addLayout(bottom_row)

    # ── Public preload API ──────────────────────────────────

    def load_from_registry(self, model_id: str) -> None:
        """Preselect a registry model in the combo and load its structure."""
        idx = self._model_combo.findData(model_id)
        if idx >= 0:
            # Setting the index triggers _on_combo_changed which loads.
            if self._model_combo.currentIndex() == idx:
                self._load(lambda: self._controller.inspect_registered_model(model_id))
            else:
                self._model_combo.setCurrentIndex(idx)
        else:
            # Not in the registry list — load directly without combo selection.
            self._load(lambda: self._controller.inspect_registered_model(model_id))

    def load_from_path(self, path: str | Path) -> None:
        """Load an external model file's structure directly."""
        p = str(path)
        self._load(lambda: self._controller.inspect_model_structure(p))

    # ── Internal handlers ───────────────────────────────────

    def _on_combo_changed(self, _idx: int) -> None:
        model_id = self._model_combo.currentData()
        if not model_id:
            return
        self._load(lambda: self._controller.inspect_registered_model(model_id))

    def _on_open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch 模型 (*.pt);;所有文件 (*)"
        )
        if not file_path:
            return
        # Reset combo to placeholder so it doesn't misrepresent the source.
        self._model_combo.blockSignals(True)
        self._model_combo.setCurrentIndex(0)
        self._model_combo.blockSignals(False)
        self._load(lambda: self._controller.inspect_model_structure(file_path))

    def _load(self, parse_fn) -> None:
        """Run a parse callable behind a wait cursor and populate the tree."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            layers = parse_fn()
        except ModelStructureError as e:
            self._tree.clear()
            self._summary.setText("加载失败")
            QMessageBox.warning(self, "无法解析模型", str(e))
            return
        except Exception as e:  # noqa: BLE001 - surface any parse failure友好地
            logger.error("Model structure parse failed: %s", e, exc_info=True)
            self._tree.clear()
            self._summary.setText("加载失败")
            QMessageBox.warning(self, "无法解析模型", f"解析模型结构时出错：{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._populate(layers)

    def _populate(self, layers: list[LayerInfo]) -> None:
        self._tree.clear()
        for layer in layers:
            item = QTreeWidgetItem([
                str(layer.index),
                layer.module_type,
                _format_params(layer.params),
                layer.output_shape,
                f"{layer.params_ratio * 100:.1f}%",
            ])
            item.setTextAlignment(0, Qt.AlignCenter)
            item.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
            item.setTextAlignment(4, Qt.AlignRight | Qt.AlignVCenter)
            self._tree.addTopLevelItem(item)

        # Total params = sum of per-layer params.
        total = sum(layer.params for layer in layers)
        self._summary.setText(
            f"共 {len(layers)} 层 | 总参数量 {_format_params(total)} | "
            "层索引与 freeze 参数一致（freeze=N 冻结第 0 ~ N-1 层）"
        )
