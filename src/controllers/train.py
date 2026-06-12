"""Training controller — start, stop, register trained models."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QWidget, QMessageBox

from src.core.project import ProjectManager
from src.core.label_io import load_annotation
from src.core.tags import TagFilter
from src.engine.backends import get_backend
from src.engine.dataset import DatasetPreparer
from src.engine.model_manager import ModelRegistry, ModelInfo
from src.engine.trainer import TrainConfig
from src.utils.workers import TrainWorker

logger = logging.getLogger(__name__)


class TrainController:
    """Handles training lifecycle: validation, start, stop, model registration."""

    def __init__(self, parent_widget: QWidget):
        self._parent = parent_widget
        self._worker: TrainWorker | None = None
        self._run_name: str = ""
        self._dataset_size: int = 0
        self._prepared_classes: list[str] = []
        self._has_prepared_classes: bool = False
        self._train_config: TrainConfig | None = None
        # Snapshot of the training context captured at start(). Registration
        # reads from these instead of MainWindow's current state, so switching
        # projects mid-training does not misroute the registered model.
        self._project_at_start: ProjectManager | None = None
        self._task_at_start: str = ""
        self._base_model_at_start: str = ""

    @property
    def worker(self) -> TrainWorker | None:
        return self._worker

    @property
    def dataset_size(self) -> int:
        return self._dataset_size

    @property
    def project_at_start(self) -> ProjectManager | None:
        """Project that owned the currently running / just-finished training."""
        return self._project_at_start

    def validate_and_prepare(
        self, project: ProjectManager, task: str, val_ratio: float,
        kpt_shape: list[int] | None = None,
        tag_filter: TagFilter | None = None,
    ) -> str | None:
        """Validate dataset and prepare for training. Returns data_yaml path or None.

        ``tag_filter`` (optional) restricts both the validation counts AND the
        dataset preparation to images whose per-image ``tags`` match. Pass
        ``None`` or an empty filter for the original behavior.
        """
        self._prepared_classes = []
        self._has_prepared_classes = False
        confirmed_count = 0
        class_counts: dict[str, int] = {}

        for img_path in project.list_images():
            label_path = project.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            if tag_filter is not None and not tag_filter.matches(ia.tags):
                continue

            # Classification: count images with tags
            if task == "classify":
                if ia.image_tags and ia.image_tags_confirmed:
                    confirmed_count += 1
                    cls = ia.image_tags[0]
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            # Detection/Pose: count confirmed annotations
            else:
                for ann in ia.annotations:
                    if ann.confirmed:
                        confirmed_count += 1
                        class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1

        self._dataset_size = confirmed_count

        if confirmed_count < 10:
            reply = QMessageBox.question(
                self._parent, "标注数量不足",
                f"仅有 {confirmed_count} 个已确认标注，建议至少 10 个。\n是否仍然继续训练？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return None

        if class_counts:
            max_c = max(class_counts.values())
            min_c = min(class_counts.values())
            if min_c > 0 and max_c / min_c > 10:
                imbalanced = ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items()))
                reply = QMessageBox.question(
                    self._parent, "类别不均衡",
                    f"类别分布严重不均衡:\n{imbalanced}\n是否仍然继续训练？",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return None

        preparer = DatasetPreparer(project)
        output_dir = project.project_dir / "datasets" / "current"
        data_yaml = preparer.prepare(
            output_dir, task=task, val_ratio=val_ratio, kpt_shape=kpt_shape,
            tag_filter=tag_filter,
        )
        if task in {"detect", "pose"}:
            data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
            self._prepared_classes = list(data.get("names", []))
        else:
            # Ultralytics derives classify class names from train/<class>/ subdir order
            # (alphabetical). Match that here so the registered model lines up.
            self._prepared_classes = sorted(class_counts.keys())
        self._has_prepared_classes = True
        return str(data_yaml)

    def start(
        self, config: TrainConfig, project: ProjectManager, task: str,
        base_model: str = "",
    ) -> TrainWorker:
        """Create and start a training worker. Returns the worker.

        Raises ``RuntimeError`` if a previous training is still running — the
        TrainPanel button is the primary UX guard, this is the controller-level
        safety net.

        ``base_model`` is the user-visible name (combo text) at start time;
        captured so registration can record it even if the user switches
        projects (which rebinds the combo to another project's models).
        """
        if self._worker is not None and self._worker.isRunning():
            raise RuntimeError("已有训练任务在运行中")

        config.project = str(project.project_dir / "models")
        run_name = f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        config.name = run_name
        self._run_name = run_name
        self._train_config = config
        self._project_at_start = project
        self._task_at_start = task
        self._base_model_at_start = base_model or config.model

        self._worker = TrainWorker(config)
        self._worker.start()
        logger.info("Training started: %s | %s | %d epochs", task, config.model, config.epochs)
        return self._worker

    def stop(self) -> None:
        """Request graceful stop of current training."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    def register_model_after_training(self, metrics: dict) -> ModelInfo | None:
        """Register the just-trained model into its *original* project's registry.

        Uses the snapshot captured by ``start()`` — never reads MainWindow's
        current state — so a user switching projects mid-training does not
        misroute the registered model.

        Returns ``None`` if called without a prior ``start()``.
        """
        if self._project_at_start is None:
            logger.warning("register_model_after_training called without prior start()")
            return None

        project = self._project_at_start
        task = self._task_at_start
        base_model = self._base_model_at_start
        epochs = self._train_config.epochs if self._train_config else 0

        # Reload the snapshot project's registry from disk so any concurrent
        # additions (e.g. import) between start() and now aren't clobbered.
        registry = ModelRegistry(project.project_dir / "models")
        registry.load()

        classes = self._prepared_classes if self._has_prepared_classes else project.config.classes
        train_params = self._train_config.to_storage_dict() if self._train_config else {}
        backend = get_backend(self._train_config.backend_id) if self._train_config else get_backend()
        probe = backend.probe()
        model_path = f"models/{self._run_name}/weights/best.pt"
        model_info = ModelInfo(
            name=f"{task}-{len(registry.list_models()) + 1}",
            path=model_path,
            task=task,
            base_model=base_model,
            classes=classes,
            metrics=metrics,
            epochs=epochs,
            dataset_size=self._dataset_size,
            train_params=train_params,
            backend_id=backend.backend_id,
            model_format=backend.infer_model_format(model_path),
            backend_version=probe.version,
            backend_runtime=probe.runtime,
            backend_metadata=probe.metadata,
        )
        registry.register(model_info)
        registry.save()
        logger.info(
            "Registered trained model to project %s: %s",
            project.config.name, model_info.name,
        )
        return model_info
