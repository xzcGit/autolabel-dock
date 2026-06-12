"""Training engine — wraps ultralytics YOLO.train."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Callable

from src.engine.backends.base import DEFAULT_BACKEND_ID

logger = logging.getLogger(__name__)

DEFAULT_FREEZE = None
DEFAULT_WORKERS = 8
DEFAULT_PATIENCE = 100
DEFAULT_ERASING = 0.4


@dataclass
class TrainConfig:
    """Training configuration."""

    data_yaml: str
    model: str
    task: str  # "detect", "classify", "pose"

    # Basic hyperparameters
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = ""
    freeze: int | None = DEFAULT_FREEZE
    workers: int = DEFAULT_WORKERS
    patience: int = DEFAULT_PATIENCE
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Data augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = DEFAULT_ERASING
    auto_augment: str = "randaugment"
    dropout: float = 0.0

    include_detect_params: bool = False
    include_classify_params: bool = False
    include_pose_params: bool = False

    # Pose-specific
    pose: float = 12.0
    kobj: float = 1.0
    kpt_shape: list[int] | None = None  # [num_keypoints, dim] e.g. [17, 3]

    # Output
    project: str = ""
    name: str = ""
    resume: bool = False
    backend_id: str = DEFAULT_BACKEND_ID

    def to_train_args(self) -> dict:
        """Convert to kwargs dict for YOLO.train()."""
        args = {
            "task": self.task,
            "data": self.data_yaml,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "workers": self.workers,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
        }
        if self.freeze is not None:
            args["freeze"] = self.freeze
        if self.device:
            args["device"] = self.device
        if self.project:
            args["project"] = self.project
        if self.name:
            args["name"] = self.name
        if self.resume:
            args["resume"] = True

        common_aug_args = {
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "scale": self.scale,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
        }
        args.update(common_aug_args)

        if self.include_detect_params:
            detect_args = {
                "degrees": self.degrees,
                "translate": self.translate,
                "shear": self.shear,
                "perspective": self.perspective,
                "mosaic": self.mosaic,
                "mixup": self.mixup,
                "copy_paste": self.copy_paste,
            }
            args.update(detect_args)

        if self.include_classify_params:
            classify_args = {
                "erasing": self.erasing,
                "auto_augment": self.auto_augment,
                "dropout": self.dropout,
            }
            args.update(classify_args)

        if self.include_pose_params:
            pose_args = {
                "pose": self.pose,
                "kobj": self.kobj,
            }
            args.update(pose_args)

        return args

    def to_storage_dict(self) -> dict:
        """Snapshot user-facing training parameters for storage in ModelInfo.

        Excludes runtime-only fields (dataset path, output project/name, resume flag)
        but keeps all hyperparameters, augmentation knobs, and task-specific values
        so the trained model carries a faithful record of how it was produced.
        """
        snapshot = asdict(self)
        for k in ("data_yaml", "project", "name", "resume", "backend_id"):
            snapshot.pop(k, None)
        return snapshot


# ── Training presets ───────────────────────────────────────────
# Deprecated: kept for backward compatibility of external imports.
# Use src.core.train_templates.TemplateRegistry for user-facing templates.
TRAIN_PRESETS: dict[str, dict] = {
    "默认": {},  # Use TrainConfig defaults
}


class _TrainingCancelled(Exception):
    """Internal sentinel raised from an ultralytics callback to break out of the
    inner batch loop. Ultralytics has no try/except around its main training
    loop or `run_callbacks`, so this propagates cleanly up to `Trainer.train()`.
    """


class Trainer:
    """Wraps ultralytics YOLO training with callback support."""

    def __init__(self, yolo_cls=None):
        if yolo_cls is None:
            from ultralytics import YOLO
            yolo_cls = YOLO
        self._yolo_cls = yolo_cls
        self._model = None
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Request graceful cancellation of training."""
        self._cancel_requested = True
        logger.info("Training cancel requested")

    @property
    def cancelled(self) -> bool:
        return self._cancel_requested

    def train(
        self,
        config: TrainConfig,
        on_epoch_end: Callable[[dict], None] | None = None,
    ) -> None:
        """Start training."""
        logger.info("Starting training: model=%s, task=%s, epochs=%d", config.model, config.task, config.epochs)
        self._model = self._yolo_cls(config.model, task=config.task)

        def _epoch_callback(trainer_obj):
            if self._cancel_requested:
                # Fallback: if cancel arrives during validation/save (when no
                # batch callback fires), force the outer epoch loop to exit.
                trainer_obj.epoch = trainer_obj.epochs
                return
            if on_epoch_end:
                metrics = {}
                if hasattr(trainer_obj, "metrics") and trainer_obj.metrics:
                    metrics = dict(trainer_obj.metrics)
                if hasattr(trainer_obj, "loss") and trainer_obj.loss is not None:
                    metrics["train_loss"] = float(trainer_obj.loss.mean().item())
                metrics["epoch"] = trainer_obj.epoch
                on_epoch_end(metrics)

        def _batch_callback(_trainer_obj):
            if self._cancel_requested:
                raise _TrainingCancelled()

        self._model.add_callback("on_fit_epoch_end", _epoch_callback)
        self._model.add_callback("on_train_batch_end", _batch_callback)

        train_args = config.to_train_args()
        try:
            self._model.train(**train_args)
        except _TrainingCancelled:
            logger.info("Training cancelled mid-epoch")

    def get_best_metrics(self) -> dict[str, float]:
        """Extract best metrics after training completes."""
        if self._model is None or not hasattr(self._model, "trainer"):
            return {}

        raw = getattr(self._model.trainer, "metrics", {})
        if not raw:
            return {}

        result = {}
        for key, value in raw.items():
            clean_key = key.replace("metrics/", "").replace("(B)", "").replace("(P)", "")
            result[clean_key] = round(float(value), 4)
        return result
