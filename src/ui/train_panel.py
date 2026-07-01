"""Training panel — parameter config, training curves, and log display."""
from __future__ import annotations

import logging
import sys

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QSplitter,
    QCheckBox,
    QProgressBar,
    QScrollArea,
    QInputDialog,
    QMessageBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.engine.trainer import (
    DEFAULT_ERASING,
    DEFAULT_PATIENCE,
    DEFAULT_WORKERS,
    TRAIN_PRESETS,
    TrainConfig,
)
from src.core.train_templates import TemplateRegistry
from src.core.tags import TagFilter
from src.ui.icons import icon
from src.ui.tag_widget import TagFilterBar
from src.ui.theme import PALETTE, set_button_role, text_style

logger = logging.getLogger(__name__)


def _detect_available_devices() -> list[str]:
    """Return device options for the training combo box.

    Always starts with "auto" (maps to "" at train time, letting Ultralytics
    pick the best available device). Then appends detected accelerators:
    CUDA GPUs by index on Linux/Windows, MPS on macOS Apple Silicon.
    Always ends with "cpu".
    """
    devices = ["auto"]
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append(f"{i}")
                logger.debug("Detected CUDA device %d: %s", i, name)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
            logger.debug("Detected MPS (Apple Silicon) device")
    except ImportError:
        if sys.platform == "darwin":
            devices.append("mps")
    devices.append("cpu")
    return devices


# Default pretrained models per task
_TASK_MODELS: dict[str, list[str]] = {
    "detect": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    "pose": ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt"],
    "classify": ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"],
    "segment": ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
    "obb": ["yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt"],
}

# Mapping from TrainConfig field names to spinbox/checkbox attribute names on TrainPanel.
# Special-cased fields (freeze, auto_augment, model, kpt_shape) handled in apply_template_params.
_NUMERIC_FIELD_MAP: dict[str, str] = {
    "epochs": "_epochs_spin",
    "batch": "_batch_spin",
    "imgsz": "_imgsz_spin",
    "workers": "_workers_spin",
    "patience": "_patience_spin",
    "lr0": "_lr0_spin",
    "lrf": "_lrf_spin",
    "momentum": "_momentum_spin",
    "weight_decay": "_weight_decay_spin",
    "warmup_epochs": "_warmup_epochs_spin",
    "warmup_momentum": "_warmup_momentum_spin",
    "warmup_bias_lr": "_warmup_bias_lr_spin",
    "hsv_h": "_hsv_h_spin",
    "hsv_s": "_hsv_s_spin",
    "hsv_v": "_hsv_v_spin",
    "degrees": "_degrees_spin",
    "translate": "_translate_spin",
    "scale": "_scale_spin",
    "shear": "_shear_spin",
    "perspective": "_perspective_spin",
    "flipud": "_flipud_spin",
    "fliplr": "_fliplr_spin",
    "mosaic": "_mosaic_spin",
    "mixup": "_mixup_spin",
    "copy_paste": "_copy_paste_spin",
    "erasing": "_erasing_spin",
    "dropout": "_dropout_spin",
    "pose": "_pose_weight_spin",
    "kobj": "_kobj_spin",
}

_BOOL_FIELD_MAP: dict[str, str] = {
    "include_detect_params": "_include_detect_params_check",
    "include_classify_params": "_include_classify_params_check",
    "include_pose_params": "_include_pose_params_check",
}

_COMBO_FIELD_MAP: dict[str, str] = {
    "optimizer": "_optimizer_combo",
    "device": "_device_combo",
}


# Quality metric per task: (display title, list of (curve label, metric-key candidates))
# The first key that appears in the emitted metrics dict wins.
_TASK_QUALITY_METRICS: dict[str, tuple[str, list[tuple[str, list[str]]]]] = {
    "detect": (
        "mAP",
        [
            ("mAP50", ["metrics/mAP50(B)", "mAP50"]),
            ("mAP50-95", ["metrics/mAP50-95(B)", "mAP50-95"]),
        ],
    ),
    "pose": (
        "mAP (Pose)",
        [
            ("Pose mAP50", ["metrics/mAP50(P)", "metrics/mAP50(B)"]),
            ("Pose mAP50-95", ["metrics/mAP50-95(P)", "metrics/mAP50-95(B)"]),
        ],
    ),
    "classify": (
        "Accuracy",
        [
            ("Top-1", ["metrics/accuracy_top1", "accuracy_top1"]),
            ("Top-5", ["metrics/accuracy_top5", "accuracy_top5"]),
        ],
    ),
}


def _pick_metric(metrics: dict, candidates: list[str]) -> float | None:
    """Return the first matching numeric value among candidate keys, else None."""
    for key in candidates:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def _compute_val_loss(metrics: dict) -> float | None:
    """Sum all val/* loss-like keys. Falls back to legacy 'val_loss' if present."""
    if "val_loss" in metrics:
        try:
            return float(metrics["val_loss"])
        except (TypeError, ValueError):
            pass
    total = 0.0
    found = False
    for k, v in metrics.items():
        if k.startswith("val/") and "loss" in k.lower():
            try:
                total += float(v)
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else None


class TrainPanel(QWidget):
    """Training panel with parameter configuration and monitoring.

    Signals:
        start_requested(TrainConfig): User clicked start training.
        stop_requested(): User clicked stop training.
    """

    start_requested = pyqtSignal(object)  # TrainConfig
    stop_requested = pyqtSignal()
    preview_augmentation_requested = pyqtSignal(dict)  # augmentation params dict
    filter_changed = pyqtSignal(object)  # TagFilter — re-emitted from inner TagFilterBar
    inspect_structure_requested = pyqtSignal(str)  # base model path to inspect

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registered_model_paths: dict[str, str] = {}  # display_name -> path
        self._template_registry: TemplateRegistry | None = None
        self._init_ui()
        self._connect_signals()
        self._on_freeze_default_toggled(self._freeze_default_check.isChecked())
        self._on_task_changed(self._task_combo.currentText())

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left: parameter config (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left = QWidget()
        left.setMinimumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # ── Task config ──
        task_group = QGroupBox("任务配置")
        task_form = QFormLayout(task_group)
        self._task_combo = QComboBox()
        self._task_combo.addItems(["detect", "classify", "pose"])
        task_form.addRow("任务类型:", self._task_combo)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(_TASK_MODELS.get("detect", []))
        task_form.addRow("基础模型:", self._model_combo)

        self._device_combo = QComboBox()
        self._device_combo.setEditable(True)
        self._device_combo.addItems(_detect_available_devices())
        task_form.addRow("设备:", self._device_combo)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(TRAIN_PRESETS.keys()))
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)
        preset_layout.addWidget(self._preset_combo)
        self._btn_save_template = QPushButton("保存为模板")
        self._btn_save_template.setToolTip("将当前训练参数保存为模板")
        preset_layout.addWidget(self._btn_save_template)
        self._btn_delete_template = QPushButton("删除模板")
        self._btn_delete_template.setToolTip("删除当前选中的模板")
        self._btn_delete_template.setEnabled(False)
        preset_layout.addWidget(self._btn_delete_template)
        task_form.addRow("预设:", preset_row)

        left_layout.addWidget(task_group)

        # ── Dataset filter (by user-defined tags) ──
        filter_group = QGroupBox("数据筛选")
        filter_form = QFormLayout(filter_group)
        self._tag_filter_bar = TagFilterBar()
        self._tag_filter_bar.filter_changed.connect(self.filter_changed)
        filter_form.addRow("Tag:", self._tag_filter_bar)

        self._filter_breakdown_label = QLabel("")
        self._filter_breakdown_label.setStyleSheet(text_style("hint"))
        self._filter_breakdown_label.setWordWrap(True)
        self._filter_breakdown_label.setVisible(False)
        filter_form.addRow("", self._filter_breakdown_label)

        filter_hint = QLabel(
            "可选：仅用带有指定 tag 的图片构建训练集。留空使用全部已确认数据。"
        )
        filter_hint.setWordWrap(True)
        filter_hint.setStyleSheet(text_style("hint"))
        filter_form.addRow("", filter_hint)
        left_layout.addWidget(filter_group)

        # ── Basic hyperparameters ──
        hyper_group = QGroupBox("训练参数")
        hyper_form = QFormLayout(hyper_group)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 10000)
        self._epochs_spin.setValue(100)
        hyper_form.addRow("Epochs:", self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(16)
        hyper_form.addRow("Batch:", self._batch_spin)

        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(32, 4096)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        hyper_form.addRow("ImgSz:", self._imgsz_spin)

        freeze_row = QWidget()
        freeze_layout = QHBoxLayout(freeze_row)
        freeze_layout.setContentsMargins(0, 0, 0, 0)
        freeze_layout.setSpacing(8)

        self._freeze_spin = QSpinBox()
        self._freeze_spin.setRange(0, 1000)
        self._freeze_spin.setValue(0)
        freeze_layout.addWidget(self._freeze_spin)

        self._freeze_default_check = QCheckBox("使用默认值")
        self._freeze_default_check.setChecked(True)
        freeze_layout.addWidget(self._freeze_default_check)

        self._btn_inspect_structure = QPushButton("查看结构")
        set_button_role(self._btn_inspect_structure, "secondary")
        self._btn_inspect_structure.setToolTip("查看当前基础模型的层级结构（辅助设置 freeze）")
        self._btn_inspect_structure.clicked.connect(
            lambda: self.inspect_structure_requested.emit(self._resolve_model_path())
        )
        freeze_layout.addWidget(self._btn_inspect_structure)
        freeze_layout.addStretch()
        hyper_form.addRow("Freeze:", freeze_row)

        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(0, 128)
        self._workers_spin.setValue(DEFAULT_WORKERS)
        hyper_form.addRow("Workers:", self._workers_spin)

        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(0, 10000)
        self._patience_spin.setValue(DEFAULT_PATIENCE)
        hyper_form.addRow("Patience:", self._patience_spin)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW"])
        hyper_form.addRow("优化器:", self._optimizer_combo)

        self._lr0_spin = QDoubleSpinBox()
        self._lr0_spin.setRange(0.0001, 1.0)
        self._lr0_spin.setDecimals(4)
        self._lr0_spin.setSingleStep(0.001)
        self._lr0_spin.setValue(0.01)
        hyper_form.addRow("学习率:", self._lr0_spin)

        self._val_ratio_spin = QDoubleSpinBox()
        self._val_ratio_spin.setRange(0.05, 0.5)
        self._val_ratio_spin.setDecimals(2)
        self._val_ratio_spin.setSingleStep(0.05)
        self._val_ratio_spin.setValue(0.2)
        hyper_form.addRow("验证集比例:", self._val_ratio_spin)

        left_layout.addWidget(hyper_group)

        # ── Advanced optimizer params ──
        opt_group = QGroupBox("优化器高级参数")
        opt_form = QFormLayout(opt_group)

        self._lrf_spin = QDoubleSpinBox()
        self._lrf_spin.setRange(0.0001, 1.0)
        self._lrf_spin.setDecimals(4)
        self._lrf_spin.setSingleStep(0.001)
        self._lrf_spin.setValue(0.01)
        opt_form.addRow("最终学习率(lrf):", self._lrf_spin)

        self._momentum_spin = QDoubleSpinBox()
        self._momentum_spin.setRange(0.0, 1.0)
        self._momentum_spin.setDecimals(3)
        self._momentum_spin.setSingleStep(0.01)
        self._momentum_spin.setValue(0.937)
        opt_form.addRow("动量:", self._momentum_spin)

        self._weight_decay_spin = QDoubleSpinBox()
        self._weight_decay_spin.setRange(0.0, 0.1)
        self._weight_decay_spin.setDecimals(5)
        self._weight_decay_spin.setSingleStep(0.0001)
        self._weight_decay_spin.setValue(0.0005)
        opt_form.addRow("权重衰减:", self._weight_decay_spin)

        self._warmup_epochs_spin = QDoubleSpinBox()
        self._warmup_epochs_spin.setRange(0.0, 10.0)
        self._warmup_epochs_spin.setDecimals(1)
        self._warmup_epochs_spin.setSingleStep(0.5)
        self._warmup_epochs_spin.setValue(3.0)
        opt_form.addRow("Warmup Epochs:", self._warmup_epochs_spin)

        self._warmup_momentum_spin = QDoubleSpinBox()
        self._warmup_momentum_spin.setRange(0.0, 1.0)
        self._warmup_momentum_spin.setDecimals(2)
        self._warmup_momentum_spin.setSingleStep(0.05)
        self._warmup_momentum_spin.setValue(0.8)
        opt_form.addRow("Warmup 动量:", self._warmup_momentum_spin)

        self._warmup_bias_lr_spin = QDoubleSpinBox()
        self._warmup_bias_lr_spin.setRange(0.0, 1.0)
        self._warmup_bias_lr_spin.setDecimals(2)
        self._warmup_bias_lr_spin.setSingleStep(0.01)
        self._warmup_bias_lr_spin.setValue(0.1)
        opt_form.addRow("Warmup Bias LR:", self._warmup_bias_lr_spin)

        left_layout.addWidget(opt_group)

        # ── Data augmentation — color ──
        aug_group = QGroupBox("数据增强 — 颜色")
        aug_form = QFormLayout(aug_group)

        self._hsv_h_spin = QDoubleSpinBox()
        self._hsv_h_spin.setRange(0, 1)
        self._hsv_h_spin.setDecimals(3)
        self._hsv_h_spin.setSingleStep(0.005)
        self._hsv_h_spin.setValue(0.015)
        aug_form.addRow("HSV-H (色调):", self._hsv_h_spin)

        self._hsv_s_spin = QDoubleSpinBox()
        self._hsv_s_spin.setRange(0, 1)
        self._hsv_s_spin.setDecimals(1)
        self._hsv_s_spin.setSingleStep(0.1)
        self._hsv_s_spin.setValue(0.7)
        aug_form.addRow("HSV-S (饱和度):", self._hsv_s_spin)

        self._hsv_v_spin = QDoubleSpinBox()
        self._hsv_v_spin.setRange(0, 1)
        self._hsv_v_spin.setDecimals(1)
        self._hsv_v_spin.setSingleStep(0.1)
        self._hsv_v_spin.setValue(0.4)
        aug_form.addRow("HSV-V (亮度):", self._hsv_v_spin)

        left_layout.addWidget(aug_group)

        # ── Data augmentation — scale & flip ──
        self._common_geo_group = QGroupBox("数据增强 — 缩放与翻转")
        common_geo_form = QFormLayout(self._common_geo_group)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0, 1)
        self._scale_spin.setDecimals(2)
        self._scale_spin.setSingleStep(0.1)
        self._scale_spin.setValue(0.5)
        common_geo_form.addRow("缩放:", self._scale_spin)

        self._fliplr_spin = QDoubleSpinBox()
        self._fliplr_spin.setRange(0, 1)
        self._fliplr_spin.setDecimals(1)
        self._fliplr_spin.setSingleStep(0.1)
        self._fliplr_spin.setValue(0.5)
        common_geo_form.addRow("水平翻转:", self._fliplr_spin)

        self._flipud_spin = QDoubleSpinBox()
        self._flipud_spin.setRange(0, 1)
        self._flipud_spin.setDecimals(1)
        self._flipud_spin.setSingleStep(0.1)
        self._flipud_spin.setValue(0.0)
        common_geo_form.addRow("垂直翻转:", self._flipud_spin)

        left_layout.addWidget(self._common_geo_group)

        # ── Data augmentation — detect geometry ──
        self._detect_geo_group = QGroupBox("数据增强 — 检测几何变换")
        detect_geo_form = QFormLayout(self._detect_geo_group)

        self._include_detect_params_check = QCheckBox("训练时传入本组参数")
        detect_geo_form.addRow("检测参数:", self._include_detect_params_check)

        detect_hint = QLabel("本组用于检测类几何与混合增强。未勾选时训练使用框架默认值。")
        detect_hint.setWordWrap(True)
        detect_geo_form.addRow("", detect_hint)

        self._degrees_spin = QDoubleSpinBox()
        self._degrees_spin.setRange(0, 180)
        self._degrees_spin.setDecimals(1)
        self._degrees_spin.setSingleStep(5)
        self._degrees_spin.setValue(0.0)
        detect_geo_form.addRow("旋转角度:", self._degrees_spin)

        self._translate_spin = QDoubleSpinBox()
        self._translate_spin.setRange(0, 1)
        self._translate_spin.setDecimals(2)
        self._translate_spin.setSingleStep(0.05)
        self._translate_spin.setValue(0.1)
        detect_geo_form.addRow("平移:", self._translate_spin)

        self._shear_spin = QDoubleSpinBox()
        self._shear_spin.setRange(0, 90)
        self._shear_spin.setDecimals(1)
        self._shear_spin.setSingleStep(1)
        self._shear_spin.setValue(0.0)
        detect_geo_form.addRow("剪切:", self._shear_spin)

        self._perspective_spin = QDoubleSpinBox()
        self._perspective_spin.setRange(0.0, 0.001)
        self._perspective_spin.setDecimals(4)
        self._perspective_spin.setSingleStep(0.0001)
        self._perspective_spin.setValue(0.0)
        detect_geo_form.addRow("透视:", self._perspective_spin)

        left_layout.addWidget(self._detect_geo_group)

        # ── Data augmentation — detect mix ──
        self._detect_mix_group = QGroupBox("数据增强 — 检测混合")
        detect_mix_form = QFormLayout(self._detect_mix_group)

        self._mosaic_spin = QDoubleSpinBox()
        self._mosaic_spin.setRange(0, 1)
        self._mosaic_spin.setDecimals(1)
        self._mosaic_spin.setSingleStep(0.1)
        self._mosaic_spin.setValue(1.0)
        detect_mix_form.addRow("Mosaic:", self._mosaic_spin)

        self._mixup_spin = QDoubleSpinBox()
        self._mixup_spin.setRange(0, 1)
        self._mixup_spin.setDecimals(1)
        self._mixup_spin.setSingleStep(0.1)
        self._mixup_spin.setValue(0.0)
        detect_mix_form.addRow("MixUp:", self._mixup_spin)

        self._copy_paste_spin = QDoubleSpinBox()
        self._copy_paste_spin.setRange(0, 1)
        self._copy_paste_spin.setDecimals(1)
        self._copy_paste_spin.setSingleStep(0.1)
        self._copy_paste_spin.setValue(0.0)
        detect_mix_form.addRow("Copy-Paste:", self._copy_paste_spin)

        left_layout.addWidget(self._detect_mix_group)

        # ── Data augmentation — classify only ──
        self._classify_aug_group = QGroupBox("分类专用参数")
        classify_aug_form = QFormLayout(self._classify_aug_group)

        self._include_classify_params_check = QCheckBox("训练时传入本组参数")
        classify_aug_form.addRow("分类参数:", self._include_classify_params_check)

        classify_hint = QLabel("本组含分类专用增强与正则化。未勾选时训练使用框架默认值。")
        classify_hint.setWordWrap(True)
        classify_aug_form.addRow("", classify_hint)

        self._erasing_spin = QDoubleSpinBox()
        self._erasing_spin.setRange(0, 1)
        self._erasing_spin.setDecimals(2)
        self._erasing_spin.setSingleStep(0.05)
        self._erasing_spin.setValue(DEFAULT_ERASING)
        classify_aug_form.addRow("Erasing:", self._erasing_spin)

        self._auto_augment_combo = QComboBox()
        self._auto_augment_combo.addItems(["randaugment", "augmix", "autoaugment", "none"])
        classify_aug_form.addRow("Auto Augment:", self._auto_augment_combo)

        self._dropout_spin = QDoubleSpinBox()
        self._dropout_spin.setRange(0.0, 1.0)
        self._dropout_spin.setDecimals(2)
        self._dropout_spin.setSingleStep(0.05)
        self._dropout_spin.setValue(0.0)
        self._dropout_spin.setToolTip("Dropout 正则化率（Ultralytics 仅在分类训练中生效）")
        classify_aug_form.addRow("Dropout:", self._dropout_spin)

        left_layout.addWidget(self._classify_aug_group)

        # ── Pose-specific params ──
        self._pose_group = QGroupBox("关键点参数")
        pose_form = QFormLayout(self._pose_group)

        self._include_pose_params_check = QCheckBox("训练时传入本组参数")
        pose_form.addRow("关键点训练参数:", self._include_pose_params_check)

        pose_hint = QLabel("`Pose权重` 和 `Kobj权重` 未勾选时训练使用框架默认值。关键点数和维度始终用于 pose 数据集准备。")
        pose_hint.setWordWrap(True)
        pose_form.addRow("", pose_hint)

        self._kpt_num_spin = QSpinBox()
        self._kpt_num_spin.setRange(1, 100)
        self._kpt_num_spin.setValue(17)
        pose_form.addRow("关键点数:", self._kpt_num_spin)

        self._kpt_dim_spin = QSpinBox()
        self._kpt_dim_spin.setRange(2, 3)
        self._kpt_dim_spin.setValue(3)
        self._kpt_dim_spin.setToolTip("2=xy, 3=xy+可见性")
        pose_form.addRow("关键点维度:", self._kpt_dim_spin)

        self._pose_weight_spin = QDoubleSpinBox()
        self._pose_weight_spin.setRange(0, 100)
        self._pose_weight_spin.setSingleStep(0.5)
        self._pose_weight_spin.setValue(12.0)
        pose_form.addRow("Pose权重:", self._pose_weight_spin)

        self._kobj_spin = QDoubleSpinBox()
        self._kobj_spin.setRange(0, 10)
        self._kobj_spin.setSingleStep(0.1)
        self._kobj_spin.setValue(1.0)
        pose_form.addRow("Kobj权重:", self._kobj_spin)

        left_layout.addWidget(self._pose_group)

        # Resume checkbox
        self._resume_check = QCheckBox("断点续训 (Resume)")
        left_layout.addWidget(self._resume_check)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_start = QPushButton(icon("start", PALETTE["ink"]), "开始训练")
        set_button_role(self._btn_start, "primary")
        self._btn_start.setToolTip("开始训练模型")
        self._btn_stop = QPushButton(icon("stop"), "停止训练")
        set_button_role(self._btn_stop, "danger")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip("停止当前训练")
        self._btn_preview_aug = QPushButton("预览增强")
        set_button_role(self._btn_preview_aug, "secondary")
        self._btn_preview_aug.setToolTip("预览当前数据增强参数效果")
        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_preview_aug)
        left_layout.addLayout(btn_layout)

        # Epoch progress bar
        self._epoch_progress = QProgressBar()
        self._epoch_progress.setRange(0, 100)
        self._epoch_progress.setValue(0)
        self._epoch_progress.setFormat("Epoch %v / %m")
        self._epoch_progress.setVisible(False)
        left_layout.addWidget(self._epoch_progress)

        left_layout.addStretch()
        scroll.setWidget(left)

        # Right: curves + log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Placeholder shown until pyqtgraph is available / training starts
        self._curve_label = QLabel("训练曲线 (训练开始后显示)")
        self._curve_label.setAlignment(Qt.AlignCenter)
        self._curve_label.setStyleSheet(
            f"color: {PALETTE['text_subtle']}; font-size: 14px; min-height: 200px; "
            f"border: 1px solid {PALETTE['line']}; border-radius: 6px;"
        )

        # Two stacked plots: Loss on top, task-specific quality on bottom.
        self._plot_widget = None
        self._loss_plot = None
        self._quality_plot = None
        self._train_loss_curve = None
        self._val_loss_curve = None
        self._quality_curves: list = []  # filled in _rebuild_quality_curves
        try:
            import pyqtgraph as pg
            pg.setConfigOptions(background=PALETTE["bg"], foreground=PALETTE["text"])

            self._loss_plot = pg.PlotWidget(title="Loss")
            self._loss_plot.setLabel("bottom", "Epoch")
            self._loss_plot.setLabel("left", "loss")
            self._loss_plot.addLegend(offset=(-10, 10))
            self._loss_plot.showGrid(x=True, y=True, alpha=0.2)
            self._train_loss_curve = self._loss_plot.plot([], [], pen=pg.mkPen(PALETTE["danger"], width=2), name="Train Loss")
            self._val_loss_curve = self._loss_plot.plot([], [], pen=pg.mkPen(PALETTE["primary"], width=2), name="Val Loss")

            self._quality_plot = pg.PlotWidget(title="mAP")
            self._quality_plot.setLabel("bottom", "Epoch")
            self._quality_plot.setLabel("left", "value")
            self._quality_plot.addLegend(offset=(-10, 10))
            self._quality_plot.showGrid(x=True, y=True, alpha=0.2)
            self._quality_plot.setYRange(0.0, 1.0, padding=0.05)

            chart_splitter = QSplitter(Qt.Vertical)
            chart_splitter.addWidget(self._loss_plot)
            chart_splitter.addWidget(self._quality_plot)
            chart_splitter.setStretchFactor(0, 1)
            chart_splitter.setStretchFactor(1, 1)
            chart_splitter.setSizes([220, 220])
            self._plot_widget = chart_splitter
            right_layout.addWidget(chart_splitter, 1)
            self._curve_label.hide()
        except ImportError:
            right_layout.addWidget(self._curve_label)

        # Epoch data for curves
        self._epoch_data: list[dict] = []

        # Set up quality curves for the initial task (deferred to _on_task_changed
        # at the end of __init__ as well, but call once now so the plot has curves).
        self._rebuild_quality_curves(self._task_combo.currentText())

        # Log
        log_label = QLabel("训练日志")
        log_label.setStyleSheet(text_style("section"))
        right_layout.addWidget(log_label)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(1000)
        right_layout.addWidget(self._log_text)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 600])
        layout.addWidget(splitter)

        # Disable scroll-wheel value changes on spinboxes and combos
        # to prevent accidental edits while scrolling the parameter list
        for w in self.findChildren(QSpinBox):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(self)
        for w in self.findChildren(QDoubleSpinBox):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(self)
        for w in self.findChildren(QComboBox):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(self)

        self._apply_elastic_form_layout()

    def _apply_elastic_form_layout(self) -> None:
        """Make the left-panel form rows resize fluidly with panel width:
        labels right-aligned, fields grow to fill, min widths so spinboxes
        don't clip. Also keeps group boxes from stretching vertically."""
        for form in self.findChildren(QFormLayout):
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.DontWrapRows)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(6)
            form.setContentsMargins(8, 6, 8, 6)
        for w in self.findChildren(QSpinBox):
            w.setMinimumWidth(70)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for w in self.findChildren(QDoubleSpinBox):
            w.setMinimumWidth(70)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for w in self.findChildren(QComboBox):
            w.setMinimumWidth(100)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for gb in self.findChildren(QGroupBox):
            gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def eventFilter(self, obj, event):
        """Ignore wheel events on spinboxes/combos that don't have focus."""
        if event.type() == event.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def _connect_signals(self) -> None:
        self._task_combo.currentTextChanged.connect(self._on_task_changed)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._freeze_default_check.toggled.connect(self._on_freeze_default_toggled)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_preview_aug.clicked.connect(self._on_preview_augmentation)
        self._btn_save_template.clicked.connect(self._on_save_template)
        self._btn_delete_template.clicked.connect(self._on_delete_template)
        self._preset_combo.currentTextChanged.connect(self._update_delete_template_button)

    def _on_freeze_default_toggled(self, use_default: bool) -> None:
        self._freeze_spin.setEnabled(not use_default)

    def _on_task_changed(self, task: str) -> None:
        self._update_augmentation_groups_for_task(task)
        self._update_model_combo_for_task(task)
        self._rebuild_quality_curves(task)
        self._repopulate_preset_combo()

    def _rebuild_quality_curves(self, task: str) -> None:
        """Recreate the bottom plot's curves for the given task. No-op if pyqtgraph
        is unavailable. Safe to call repeatedly — clears old curves first."""
        if self._quality_plot is None:
            return
        try:
            import pyqtgraph as pg
        except ImportError:
            return

        title, specs = _TASK_QUALITY_METRICS.get(task, _TASK_QUALITY_METRICS["detect"])
        self._quality_plot.setTitle(title)
        # Clear previous curves (both data and legend entries).
        self._quality_plot.clear()
        if hasattr(self._quality_plot, "plotItem") and self._quality_plot.plotItem.legend is not None:
            self._quality_plot.plotItem.legend.clear()
        palette = [PALETTE["success"], PALETTE["warning"], PALETTE["violet"], PALETTE["teal"]]
        self._quality_curves = []
        for i, (label, keys) in enumerate(specs):
            pen = pg.mkPen(palette[i % len(palette)], width=2)
            curve = self._quality_plot.plot([], [], pen=pen, name=label)
            self._quality_curves.append((curve, keys))

    def _update_augmentation_groups_for_task(self, task: str) -> None:
        self._common_geo_group.setVisible(True)
        self._detect_geo_group.setVisible(True)
        self._detect_mix_group.setVisible(True)
        self._classify_aug_group.setVisible(True)
        self._pose_group.setVisible(True)

    def _update_model_combo_for_task(self, task: str) -> None:
        """Replace pretrained model entries to match the selected task."""
        # Preserve registered models (after separator) and user edits
        separator_idx = -1
        registered_items: list[str] = []
        for i in range(self._model_combo.count()):
            text = self._model_combo.itemText(i)
            if text == "──────────":
                separator_idx = i
            elif separator_idx >= 0:
                registered_items.append(text)

        self._model_combo.clear()
        self._model_combo.addItems(_TASK_MODELS.get(task, _TASK_MODELS["detect"]))

        if registered_items:
            self._model_combo.addItem("──────────")
            idx = self._model_combo.count() - 1
            self._model_combo.model().item(idx).setEnabled(False)
            for item in registered_items:
                self._model_combo.addItem(item)

    def _on_preset_changed(self, preset_name: str) -> None:
        """Apply a preset/template to all UI fields."""
        if not preset_name:
            return
        if preset_name == "默认":
            self._apply_default_preset()
            logger.info("Applied training preset: 默认")
            return
        if self._template_registry is None:
            return
        current_task = self._task_combo.currentText()
        template = self._template_registry.get(preset_name, current_task)
        if template is None:
            logger.warning("Template not found: %s [%s]", preset_name, current_task)
            return
        self.apply_template_params(template.params)
        logger.info("Applied user template: %s [%s]", preset_name, current_task)

    def _apply_default_preset(self) -> None:
        """Reset all relevant UI fields to TrainConfig dataclass defaults."""
        defaults = TrainConfig(data_yaml="", model="", task=self._task_combo.currentText())
        snapshot = defaults.to_storage_dict()
        # Skip 'model' so we don't overwrite the user's basemodel selection on a defaults reset.
        snapshot.pop("model", None)
        snapshot.pop("task", None)
        self.apply_template_params(snapshot)

    def _on_save_template(self) -> None:
        """Prompt for a name and save the current params as a template for the current task."""
        if self._template_registry is None:
            QMessageBox.warning(self, "无法保存", "模板注册表未初始化。")
            return
        name, ok = QInputDialog.getText(self, "保存训练模板", "模板名称:", text="我的模板")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "名称无效", "模板名称不能为空。")
            return
        if name == "默认":
            QMessageBox.warning(self, "名称无效", "「默认」是保留名称，请改用其他名称。")
            return

        current_task = self._task_combo.currentText()
        if self._template_registry.get(name, current_task) is not None:
            reply = QMessageBox.question(
                self, "覆盖模板",
                f"已存在同名模板「{name}」，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        from datetime import datetime
        from src.core.train_templates import TrainTemplate

        params = self.get_train_template_params()
        template = TrainTemplate(
            name=name,
            task=current_task,
            params=params,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._template_registry.upsert(template)
        self._template_registry.save()
        logger.info("Saved train template: %s [%s]", name, current_task)

        self._repopulate_preset_combo()
        idx = self._preset_combo.findText(name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _on_delete_template(self) -> None:
        """Delete the currently selected user template after confirmation."""
        if self._template_registry is None:
            return
        name = self._preset_combo.currentText()
        current_task = self._task_combo.currentText()
        if name == "默认":
            return
        reply = QMessageBox.question(
            self, "删除模板",
            f"确认删除模板「{name}」？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if self._template_registry.remove(name, current_task):
            self._template_registry.save()
            logger.info("Deleted train template: %s [%s]", name, current_task)
        self._repopulate_preset_combo()
        self._preset_combo.setCurrentText("默认")

    def _update_delete_template_button(self, current_name: str) -> None:
        """Disable the delete button when the built-in '默认' is selected."""
        self._btn_delete_template.setEnabled(bool(current_name) and current_name != "默认")

    def _set_freeze_value(self, value: int | None) -> None:
        use_default = value is None
        self._freeze_default_check.blockSignals(True)
        self._freeze_default_check.setChecked(use_default)
        self._freeze_default_check.blockSignals(False)
        self._freeze_spin.setValue(0 if value is None else value)
        self._on_freeze_default_toggled(use_default)

    def apply_template_params(self, params: dict) -> None:
        """Set spin/check/combo values for keys present in `params`. Missing keys leave UI alone.
        Unknown keys are skipped with a warning log."""
        for key, value in params.items():
            if key in _NUMERIC_FIELD_MAP:
                getattr(self, _NUMERIC_FIELD_MAP[key]).setValue(value)
            elif key in _BOOL_FIELD_MAP:
                getattr(self, _BOOL_FIELD_MAP[key]).setChecked(bool(value))
            elif key in _COMBO_FIELD_MAP:
                combo = getattr(self, _COMBO_FIELD_MAP[key])
                text = str(value) if value is not None else ""
                if key == "device" and text == "":
                    text = "auto"
                combo.setCurrentText(text)
            elif key == "freeze":
                self._set_freeze_value(value)
            elif key == "auto_augment":
                text = value if value else "none"
                idx = self._auto_augment_combo.findText(text)
                if idx >= 0:
                    self._auto_augment_combo.setCurrentIndex(idx)
            elif key == "model":
                # Editable combo — accept arbitrary string
                self._model_combo.setCurrentText(str(value))
            elif key == "kpt_shape":
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    self._kpt_num_spin.setValue(int(value[0]))
                    self._kpt_dim_spin.setValue(int(value[1]))
            else:
                logger.warning("Skipping unknown template key: %s", key)

    def get_train_template_params(self) -> dict:
        """Read current UI state into a task-relevant params dict."""
        from src.core.train_templates import extract_task_params
        config = self.get_train_config(data_yaml="", model=None)
        return extract_task_params(config)

    def _on_start(self) -> None:
        self._btn_start.setEnabled(False)
        self._btn_start.setText("训练中")
        self._btn_stop.setEnabled(True)
        self._log_text.clear()
        self._epoch_data.clear()
        total_epochs = self._epochs_spin.value()
        self._epoch_progress.setRange(0, total_epochs)
        self._epoch_progress.setValue(0)
        self._epoch_progress.setVisible(True)
        # Reset curves so a previous run's data doesn't linger.
        if self._train_loss_curve is not None:
            self._train_loss_curve.setData([], [])
        if self._val_loss_curve is not None:
            self._val_loss_curve.setData([], [])
        for curve, _ in self._quality_curves:
            curve.setData([], [])

    def _on_stop(self) -> None:
        # Keep the start button disabled (still labelled "训练中") until the
        # worker actually emits cancelled — restoring it now would let a quick
        # second click race the cancellation.
        self._btn_stop.setEnabled(False)
        self.stop_requested.emit()

    def _on_preview_augmentation(self) -> None:
        """Emit augmentation params for preview."""
        params = self.get_effective_augmentation_params()
        self.preview_augmentation_requested.emit(params)

    def get_augmentation_params(self) -> dict:
        """Get current data augmentation parameters as dict."""
        return {
            "hsv_h": self._hsv_h_spin.value(),
            "hsv_s": self._hsv_s_spin.value(),
            "hsv_v": self._hsv_v_spin.value(),
            "degrees": self._degrees_spin.value(),
            "translate": self._translate_spin.value(),
            "scale": self._scale_spin.value(),
            "shear": self._shear_spin.value(),
            "perspective": self._perspective_spin.value(),
            "flipud": self._flipud_spin.value(),
            "fliplr": self._fliplr_spin.value(),
            "mosaic": self._mosaic_spin.value(),
            "mixup": self._mixup_spin.value(),
            "copy_paste": self._copy_paste_spin.value(),
            "erasing": self._erasing_spin.value(),
        }

    def get_effective_augmentation_params(self, task: str | None = None) -> dict:
        """Return the augmentation params that would actually be sent to training."""
        params = self.get_augmentation_params()
        keys = [
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "scale",
            "flipud",
            "fliplr",
        ]
        if self._include_detect_params_check.isChecked():
            keys.extend(("degrees", "translate", "shear", "perspective", "mosaic", "mixup", "copy_paste"))
        if self._include_classify_params_check.isChecked():
            keys.append("erasing")
        return {key: params[key] for key in keys}

    def get_train_config(self, data_yaml: str, model: str | None = None) -> TrainConfig:
        """Build TrainConfig from current UI values."""
        auto_augment = self._auto_augment_combo.currentText()
        if auto_augment == "none":
            auto_augment = ""

        config = TrainConfig(
            data_yaml=data_yaml,
            model=model or self._resolve_model_path(),
            task=self._task_combo.currentText(),
            epochs=self._epochs_spin.value(),
            batch=self._batch_spin.value(),
            imgsz=self._imgsz_spin.value(),
            device=self._device_combo.currentText() if self._device_combo.currentText() != "auto" else "",
            freeze=None if self._freeze_default_check.isChecked() else self._freeze_spin.value(),
            workers=self._workers_spin.value(),
            patience=self._patience_spin.value(),
            optimizer=self._optimizer_combo.currentText(),
            lr0=self._lr0_spin.value(),
            lrf=self._lrf_spin.value(),
            momentum=self._momentum_spin.value(),
            weight_decay=self._weight_decay_spin.value(),
            warmup_epochs=self._warmup_epochs_spin.value(),
            warmup_momentum=self._warmup_momentum_spin.value(),
            warmup_bias_lr=self._warmup_bias_lr_spin.value(),
            hsv_h=self._hsv_h_spin.value(),
            hsv_s=self._hsv_s_spin.value(),
            hsv_v=self._hsv_v_spin.value(),
            degrees=self._degrees_spin.value(),
            translate=self._translate_spin.value(),
            scale=self._scale_spin.value(),
            shear=self._shear_spin.value(),
            perspective=self._perspective_spin.value(),
            flipud=self._flipud_spin.value(),
            fliplr=self._fliplr_spin.value(),
            mosaic=self._mosaic_spin.value(),
            mixup=self._mixup_spin.value(),
            copy_paste=self._copy_paste_spin.value(),
            erasing=self._erasing_spin.value(),
            auto_augment=auto_augment,
            dropout=self._dropout_spin.value(),
            include_detect_params=self._include_detect_params_check.isChecked(),
            include_classify_params=self._include_classify_params_check.isChecked(),
            include_pose_params=self._include_pose_params_check.isChecked(),
            pose=self._pose_weight_spin.value(),
            kobj=self._kobj_spin.value(),
            resume=self._resume_check.isChecked(),
        )
        if self._task_combo.currentText() == "pose":
            config.kpt_shape = [self._kpt_num_spin.value(), self._kpt_dim_spin.value()]
        logger.info("Training config: epochs=%d, batch=%d, model=%s", config.epochs, config.batch, config.model)
        return config

    def get_val_ratio(self) -> float:
        """Get validation split ratio."""
        return self._val_ratio_spin.value()

    def get_tag_filter(self) -> TagFilter:
        """Current TagFilter selected in the dataset filter group.

        Returns an empty filter when the user has selected no tags — callers
        can simply check ``is_empty()`` or pass it through to
        ``DatasetPreparer.prepare(tag_filter=...)`` unconditionally.
        """
        return self._tag_filter_bar.current_filter()

    def set_filter_breakdown(self, counts: dict[str, int] | None) -> None:
        """Render the tag-filter diagnostic label.

        Passing ``None`` or an all-zero dict hides the label. Otherwise
        shows ``命中 N 张，排除 M 张`` plus a conflict suffix when any
        image landed in the ``conflict`` bucket.
        """
        label = self._filter_breakdown_label
        if not counts:
            label.setVisible(False)
            return
        match = counts.get("match", 0)
        dropped = (
            counts.get("excluded", 0)
            + counts.get("no_include", 0)
            + counts.get("conflict", 0)
        )
        if match == 0 and dropped == 0:
            label.setVisible(False)
            return
        text = f"命中 {match} 张，排除 {dropped} 张"
        conflict = counts.get("conflict", 0)
        if conflict > 0:
            text += f"（其中 {conflict} 张同时命中 include 和 exclude）"
        label.setText(text)
        label.setVisible(True)

    def set_available_tags(self, tags: list[str]) -> None:
        """Push the project's tag registry into the dataset filter dropdown."""
        self._tag_filter_bar.set_available_tags(tags)

    def append_log(self, text: str) -> None:
        """Append text to training log."""
        self._log_text.appendPlainText(text)

    def update_epoch(self, metrics: dict) -> None:
        """Update curves and log with epoch metrics."""
        self._epoch_data.append(metrics)
        epoch = metrics.get("epoch", len(self._epoch_data))

        # Update progress bar
        self._epoch_progress.setValue(epoch)

        # Log line
        parts = [f"Epoch {epoch}"]
        for k, v in metrics.items():
            if k != "epoch":
                parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        self.append_log(" | ".join(parts))

        # Update curves
        if self._loss_plot is None:
            return
        epochs = list(range(1, len(self._epoch_data) + 1))
        train_losses = [float(d.get("train_loss", 0) or 0) for d in self._epoch_data]
        val_losses_raw = [_compute_val_loss(d) for d in self._epoch_data]
        # Carry forward last known val_loss so the line doesn't dip to 0 on
        # epochs that have no val metrics (some tasks only validate periodically).
        last_val = 0.0
        val_losses: list[float] = []
        for v in val_losses_raw:
            if v is not None:
                last_val = v
            val_losses.append(last_val)
        self._train_loss_curve.setData(epochs, train_losses)
        self._val_loss_curve.setData(epochs, val_losses)

        for curve, keys in self._quality_curves:
            last = 0.0
            ys: list[float] = []
            for d in self._epoch_data:
                v = _pick_metric(d, keys)
                if v is not None:
                    last = v
                ys.append(last)
            curve.setData(epochs, ys)

    def on_training_finished(self, metrics: dict) -> None:
        """Handle training completion."""
        self._btn_start.setEnabled(True)
        self._btn_start.setText("开始训练")
        self._btn_stop.setEnabled(False)
        self._epoch_progress.setVisible(False)
        self.append_log("--- 训练完成 ---")
        if metrics:
            for k, v in metrics.items():
                self.append_log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def on_training_cancelled(self) -> None:
        """Handle training cancellation. Symmetric to ``on_training_finished``."""
        self._btn_start.setEnabled(True)
        self._btn_start.setText("开始训练")
        self._btn_stop.setEnabled(False)
        self._epoch_progress.setVisible(False)
        self.append_log("--- 训练已停止 ---")

    def reset_start_button_idle(self) -> None:
        """Restore the start/stop buttons to their pre-training idle state.

        ``_on_start`` flips the start button to the running state ("训练中",
        disabled) the instant it is clicked. When the training launch is then
        aborted *before* a worker exists — e.g. the user declines the
        LocateAnything-disable confirmation in ``MainWindow._on_start_training``
        — there is no ``finished`` / ``cancelled`` signal to restore the UI, so
        the caller invokes this. Side-effect-free (no log line) since no
        training actually ran. Public so callers don't poke the private
        ``_btn_start`` / ``_btn_stop`` widgets directly.
        """
        self._btn_start.setEnabled(True)
        self._btn_start.setText("开始训练")
        self._btn_stop.setEnabled(False)
        self._epoch_progress.setVisible(False)

    def on_training_error(self, error_msg: str) -> None:
        """Handle training error."""
        self._btn_start.setEnabled(True)
        self._btn_start.setText("开始训练")
        self._btn_stop.setEnabled(False)
        self._epoch_progress.setVisible(False)
        self.append_log(f"--- 训练失败: {error_msg} ---")

    def _resolve_model_path(self) -> str:
        """Resolve model combo text to actual path (handles registered models)."""
        text = self._model_combo.currentText()
        return self._registered_model_paths.get(text, text)

    def set_template_registry(self, registry: TemplateRegistry | None) -> None:
        """Inject the global template registry. Triggers an immediate combo repopulation."""
        self._template_registry = registry
        self._repopulate_preset_combo()

    def _repopulate_preset_combo(self) -> None:
        """Repopulate _preset_combo with builtin '默认' + user templates for current task."""
        current_task = self._task_combo.currentText()
        previous_selection = self._preset_combo.currentText()

        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        if self._template_registry is None:
            self._preset_combo.addItem("默认")
        else:
            for t in self._template_registry.list(task=current_task):
                self._preset_combo.addItem(t.name)
        self._preset_combo.blockSignals(False)

        idx = self._preset_combo.findText(previous_selection)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            # Previous selection no longer applies under the new task — reset to 默认
            # and apply defaults explicitly (setCurrentIndex(0) won't fire a signal
            # because addItem already made index 0 current while signals were blocked).
            self._preset_combo.setCurrentIndex(0)
            self._apply_default_preset()
        self._update_delete_template_button(self._preset_combo.currentText())

    def set_registered_models(self, models) -> None:
        """Update combo with registered models for finetune.

        Args:
            models: list of ModelInfo objects with .name and .path attributes.
        """
        # Remember current selection
        current = self._model_combo.currentText()

        # Remove old registered entries (after separator)
        separator_idx = -1
        for i in range(self._model_combo.count()):
            if self._model_combo.itemText(i) == "──────────":
                separator_idx = i
                break
        if separator_idx >= 0:
            while self._model_combo.count() > separator_idx:
                self._model_combo.removeItem(separator_idx)

        self._registered_model_paths.clear()

        if models:
            self._model_combo.addItem("──────────")
            # Make separator unselectable
            idx = self._model_combo.count() - 1
            self._model_combo.model().item(idx).setEnabled(False)
            for m in models:
                display = f"[已训练] {m.name}"
                self._registered_model_paths[display] = m.path
                self._model_combo.addItem(display)

        # Restore selection
        restore_idx = self._model_combo.findText(current)
        if restore_idx >= 0:
            self._model_combo.setCurrentIndex(restore_idx)
