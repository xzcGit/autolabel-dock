"""QThread workers for training and batch inference."""
from __future__ import annotations

import inspect
import logging
import threading
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QMutex

from src.core.annotation import Annotation
from src.engine.backends import get_backend
from src.engine.backends.base import TrainerProtocol
from src.engine.trainer import TrainConfig

logger = logging.getLogger(__name__)

_CLASSIFY_BATCH_SIZE = 16


class TrainWorker(QThread):
    """Runs backend model training in a background thread.

    Signals:
        epoch_update(dict): Emitted after each epoch with metrics dict.
        finished_ok(dict): Emitted on successful completion with best metrics.
        cancelled(): Emitted when training is cancelled by user.
        error(str): Emitted if training fails with error message.
    """

    epoch_update = pyqtSignal(dict)
    finished_ok = pyqtSignal(dict)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: TrainConfig, trainer_cls=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._trainer_cls = trainer_cls
        self._trainer: TrainerProtocol | None = None
        self._trainer_mutex = QMutex()

    def cancel(self) -> None:
        """Request graceful cancellation of training."""
        self._trainer_mutex.lock()
        try:
            if self._trainer:
                self._trainer.request_cancel()
        finally:
            self._trainer_mutex.unlock()

    def run(self) -> None:
        trainer = None
        try:
            if self._trainer_cls is not None:
                trainer = self._trainer_cls()
            else:
                trainer = get_backend(self._config.backend_id).create_trainer()
            self._trainer_mutex.lock()
            try:
                self._trainer = trainer
            finally:
                self._trainer_mutex.unlock()
            trainer.train(self._config, on_epoch_end=self._on_epoch)
            if trainer.cancelled:
                self.cancelled.emit()
            else:
                metrics = trainer.get_best_metrics()
                self.finished_ok.emit(metrics)
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("Training failed")
            self.error.emit(str(e))
        finally:
            # Release the YOLO model + nested ultralytics trainer (which owns the
            # PyTorch DataLoader and its worker subprocesses). Without this the
            # dataloader workers stay alive until the next training overwrites
            # self._worker, occupying RAM, file handles, and (for GPU runs) VRAM.
            self._release_trainer(trainer)

    def _release_trainer(self, trainer) -> None:
        """Drop references to the Trainer / YOLO model so DataLoader workers exit."""
        try:
            if trainer is not None:
                # Null out the nested ultralytics trainer's loaders explicitly so
                # any internal cycles don't keep the worker subprocesses alive
                # while waiting for the cycle collector.
                inner_model = getattr(trainer, "_model", None)
                if inner_model is not None:
                    inner_trainer = getattr(inner_model, "trainer", None)
                    if inner_trainer is not None:
                        for attr in ("train_loader", "test_loader", "validator"):
                            try:
                                setattr(inner_trainer, attr, None)
                            except Exception:
                                pass
                    try:
                        trainer._model = None
                    except Exception:
                        pass
        except Exception:
            logger.exception("Cleanup after training failed")
        finally:
            self._trainer_mutex.lock()
            try:
                self._trainer = None
            finally:
                self._trainer_mutex.unlock()
            # Encourage prompt collection of the freed objects (and their CUDA
            # tensors). Safe to call even if torch isn't on the GPU path.
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _on_epoch(self, metrics: dict) -> None:
        self.epoch_update.emit(metrics)


class BatchPredictWorker(QThread):
    """Runs batch inference in a background thread.

    Signals:
        progress(int, int): Emitted with (current, total) after each image.
        image_done(str, object, object): Emitted with (image_path, payload, image_size).
            payload is list[Annotation] for detect/pose, or tuple[str, float] | None for classify.
        finished_ok(): Emitted when all images are processed (not emitted on cancel).
        error(str): Emitted if inference fails with error message.
    """

    progress = pyqtSignal(int, int)
    image_done = pyqtSignal(str, object, object)  # (path, payload, image_size)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        predictor,
        image_paths: list[Path],
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
        task: str = "detect",
        parent=None,
    ):
        super().__init__(parent)
        self._predictor = predictor
        self._image_paths = image_paths
        self._conf = conf
        self._iou = iou
        self._project_classes = project_classes
        self._class_match_mode = class_match_mode
        self._kpt_labels = kpt_labels
        self._task = task
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of batch processing."""
        self._cancelled.set()

    def run(self) -> None:
        total = len(self._image_paths)
        try:
            if self._task == "classify":
                self._run_classify(total)
            else:
                self._run_detect_or_pose(total)
            if not self._cancelled.is_set():
                self.finished_ok.emit()
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("Batch inference failed")
            self.error.emit(str(e))

    def _run_classify(self, total: int) -> None:
        predict_batch = getattr(self._predictor, "predict_classify_batch", None)
        if inspect.ismethod(predict_batch):
            for batch_start in range(0, total, _CLASSIFY_BATCH_SIZE):
                if self._cancelled.is_set():
                    break
                batch_paths = self._image_paths[batch_start:batch_start + _CLASSIFY_BATCH_SIZE]
                payloads = predict_batch(
                    batch_paths,
                    project_classes=self._project_classes,
                    filter_to_project=False,
                )
                if len(payloads) != len(batch_paths):
                    raise ValueError(
                        "predict_classify_batch returned "
                        f"{len(payloads)} payloads for {len(batch_paths)} images"
                    )
                for offset, (img_path, payload) in enumerate(
                    zip(batch_paths, payloads), start=batch_start,
                ):
                    if self._cancelled.is_set():
                        break
                    self.image_done.emit(str(img_path), payload, (0, 0))
                    self.progress.emit(offset + 1, total)
            return

        for i, img_path in enumerate(self._image_paths):
            if self._cancelled.is_set():
                break
            payload = self._predictor.predict_classify(
                img_path,
                project_classes=self._project_classes,
                filter_to_project=False,
            )
            self.image_done.emit(str(img_path), payload, (0, 0))
            self.progress.emit(i + 1, total)

    def _run_detect_or_pose(self, total: int) -> None:
        for i, img_path in enumerate(self._image_paths):
            if self._cancelled.is_set():
                break
            payload, img_size = self._predictor.predict_with_size(
                img_path,
                conf=self._conf,
                iou=self._iou,
                project_classes=self._project_classes,
                class_match_mode=self._class_match_mode,
                kpt_labels=self._kpt_labels,
            )
            self.image_done.emit(str(img_path), payload, img_size)
            self.progress.emit(i + 1, total)


class SinglePredictWorker(QThread):
    """Runs a single-image detection/pose inference off the UI thread.

    Used for slow backends (e.g. LocateAnything) whose ``predict`` blocks for
    seconds — running that on the Qt/X event loop can stall the desktop and, on
    a single-GPU machine where the X server shares the same card, crash it.
    YOLO single-image inference stays synchronous (fast, and existing tests
    depend on it); this worker is only used when the controller knows a slow
    backend is active.

    Signals:
        done(object): emitted with the predicted ``list[Annotation]``.
        error(str): emitted with a readable message if inference fails.

    Mirrors ``BatchPredictWorker._run_detect_or_pose`` for one image but calls
    ``predict`` (not ``predict_with_size``) to match the synchronous
    single-image path in ``ModelController.predict_single``.
    """

    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        predictor,
        image_path: Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        parent=None,
    ):
        super().__init__(parent)
        self._predictor = predictor
        self._image_path = image_path
        self._conf = conf
        self._iou = iou
        self._project_classes = project_classes
        self._class_match_mode = class_match_mode

    def run(self) -> None:
        try:
            annotations = self._predictor.predict(
                self._image_path,
                conf=self._conf,
                iou=self._iou,
                project_classes=self._project_classes,
                class_match_mode=self._class_match_mode,
            )
            self.done.emit(annotations)
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently
            # kill the thread. Surface a readable message to the UI instead.
            logger.exception("Single-image inference failed")
            self.error.emit(str(e))


class LocateAnythingLoadWorker(QThread):
    """Loads the LocateAnything runtime (heavy 4-bit model) off the UI thread.

    Signals:
        progress(str): short status string for the UI during load.
        loaded(object): emitted with the ready LocateAnythingPredictor.
        error(str): emitted if loading fails.

    The heavy imports (torch/transformers) all happen inside
    ``backend.load_runtime()`` — this worker never imports them itself.
    """

    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend

    def run(self) -> None:
        try:
            predictor = self._backend.load_runtime(progress_cb=self._on_progress)
            self.loaded.emit(predictor)
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("LocateAnything load failed")
            self.error.emit(str(e))

    def _on_progress(self, message: str) -> None:
        self.progress.emit(message)
