"""Tests for QThread workers."""
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtCore import QCoreApplication, QMutex


def _process_events():
    """Process pending Qt events."""
    QCoreApplication.processEvents()


class TestTrainWorker:
    def test_emits_epoch_signal(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect", epochs=2)

        # Mock the Trainer
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Capture on_epoch_end callback and call it during train
        def fake_train(cfg, on_epoch_end=None):
            if on_epoch_end:
                on_epoch_end({"epoch": 0, "train_loss": 1.5})
                on_epoch_end({"epoch": 1, "train_loss": 0.8})

        mock_trainer_instance.train.side_effect = fake_train
        mock_trainer_instance.get_best_metrics.return_value = {"mAP50": 0.85}
        mock_trainer_instance.cancelled = False

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        epochs = []
        worker.epoch_update.connect(lambda d: epochs.append(d))
        finished_data = []
        worker.finished_ok.connect(lambda d: finished_data.append(d))

        worker.run()  # call run() directly, not start()
        assert len(epochs) == 2
        assert epochs[0]["epoch"] == 0
        assert len(finished_data) == 1
        assert finished_data[0]["mAP50"] == 0.85

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.train.side_effect = RuntimeError("CUDA OOM")

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))

        worker.run()
        assert len(errors) == 1
        assert "CUDA OOM" in errors[0]

    def test_releases_trainer_after_success(self, qapp):
        """After successful training, worker must drop its Trainer reference
        so the underlying YOLO model and DataLoader workers can be GC'd."""
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.cancelled = False
        mock_trainer_instance.get_best_metrics.return_value = {}
        # Simulate the YOLO + nested ultralytics trainer (with DataLoaders) attached
        mock_trainer_instance._model = MagicMock()
        mock_trainer_instance._model.trainer = MagicMock()

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        worker.run()

        assert worker._trainer is None, (
            "TrainWorker still holds Trainer after run(); "
            "DataLoader worker subprocesses won't be released"
        )
        # And the Trainer itself must drop its YOLO model reference
        assert mock_trainer_instance._model is None

    def test_releases_trainer_after_error(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.train.side_effect = RuntimeError("boom")
        mock_trainer_instance._model = MagicMock()

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        worker.run()

        assert worker._trainer is None
        assert mock_trainer_instance._model is None

    def test_releases_trainer_after_cancel(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.cancelled = True  # cancelled path
        mock_trainer_instance._model = MagicMock()

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        worker.run()

        assert worker._trainer is None
        assert mock_trainer_instance._model is None

    def test_default_trainer_created_from_config_backend(self, qapp, monkeypatch):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        mock_trainer = MagicMock()
        mock_trainer.cancelled = False
        mock_trainer.get_best_metrics.return_value = {"mAP50": 0.7}
        mock_backend = MagicMock()
        mock_backend.create_trainer.return_value = mock_trainer
        backend_ids = []

        def fake_get_backend(backend_id):
            backend_ids.append(backend_id)
            return mock_backend

        monkeypatch.setattr("src.utils.workers.get_backend", fake_get_backend)
        config = TrainConfig(
            data_yaml="data.yaml", model="custom.pt", task="detect",
            backend_id="custom-backend",
        )

        worker = TrainWorker(config)
        finished_data = []
        worker.finished_ok.connect(lambda d: finished_data.append(d))

        worker.run()

        assert backend_ids == ["custom-backend"]
        mock_backend.create_trainer.assert_called_once_with()
        mock_trainer.train.assert_called_once()
        assert finished_data == [{"mAP50": 0.7}]


class TestBatchPredictWorker:
    def test_emits_progress_and_results(self, qapp):
        from src.utils.workers import BatchPredictWorker
        from src.core.annotation import Annotation

        mock_predictor = MagicMock()
        ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4), confirmed=False, source="auto")
        mock_predictor.predict_with_size.return_value = ([ann], (640, 480))

        image_paths = [Path(f"/imgs/img{i}.jpg") for i in range(3)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            conf=0.5,
            iou=0.45,
        )

        progress_values = []
        worker.progress.connect(lambda cur, total: progress_values.append((cur, total)))
        results = []
        worker.image_done.connect(lambda path, anns, size: results.append((path, anns, size)))
        finished = []
        worker.finished_ok.connect(lambda: finished.append(True))

        worker.run()
        assert len(progress_values) == 3
        assert progress_values[-1] == (3, 3)
        assert len(results) == 3
        assert results[0][1][0].class_name == "cat"
        assert len(finished) == 1
        assert mock_predictor.predict_with_size.call_args.kwargs["class_match_mode"] == "class_id"

    def test_forwards_class_match_mode(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.return_value = ([], (640, 480))

        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/imgs/img0.jpg")],
            class_match_mode="class_name",
        )

        worker.run()
        assert mock_predictor.predict_with_size.call_args.kwargs["class_match_mode"] == "class_name"

    def test_cancel_stops_processing(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.return_value = ([], (640, 480))

        image_paths = [Path(f"/imgs/img{i}.jpg") for i in range(10)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            conf=0.5,
            iou=0.45,
        )

        results = []
        worker.image_done.connect(lambda path, anns, size: results.append(path))

        # Cancel after first call
        def cancel_after_first(*args, **kwargs):
            if mock_predictor.predict_with_size.call_count >= 2:
                worker.cancel()
            return ([], (640, 480))

        mock_predictor.predict_with_size.side_effect = cancel_after_first
        worker.run()
        assert len(results) < 10

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.side_effect = RuntimeError("model error")

        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
            conf=0.5,
            iou=0.45,
        )
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))

        worker.run()
        assert len(errors) == 1
        assert "model error" in errors[0]

    def test_classify_task_calls_predict_classify(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_classify.side_effect = [("cat", 0.9), None]

        image_paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg")]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
            project_classes=["cat", "dog"],
            task="classify",
        )

        results = []
        worker.image_done.connect(
            lambda path, payload, size: results.append((path, payload, size))
        )
        finished = []
        worker.finished_ok.connect(lambda: finished.append(True))

        worker.run()
        assert len(results) == 2
        assert results[0] == (str(Path("/imgs/a.jpg")), ("cat", 0.9), (0, 0))
        assert results[1] == (str(Path("/imgs/b.jpg")), None, (0, 0))
        assert mock_predictor.predict_with_size.call_count == 0
        assert finished == [True]

    def test_classify_task_forwards_project_classes(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_classify.return_value = ("cat", 0.5)

        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
            project_classes=["cat"],
            task="classify",
        )
        worker.run()
        kwargs = mock_predictor.predict_classify.call_args.kwargs
        assert kwargs["project_classes"] == ["cat"]

    def test_classify_task_prefers_predict_classify_batch_when_available(self, qapp):
        from src.utils.workers import BatchPredictWorker

        class BatchCapablePredictor:
            def __init__(self):
                self.batch_calls = []
                self.single_calls = []

            def predict_classify_batch(
                self, image_paths, project_classes=None, filter_to_project=True,
            ):
                self.batch_calls.append((
                    list(image_paths), project_classes, filter_to_project,
                ))
                return [("cat", 0.9), None]

            def predict_classify(
                self, image_path, project_classes=None, filter_to_project=True,
            ):
                self.single_calls.append((
                    image_path, project_classes, filter_to_project,
                ))
                return ("dog", 0.1)

        predictor = BatchCapablePredictor()
        image_paths = [Path("/imgs/a.jpg"), Path("/imgs/b.jpg")]
        worker = BatchPredictWorker(
            predictor=predictor,
            image_paths=image_paths,
            project_classes=["cat"],
            task="classify",
        )

        results = []
        worker.image_done.connect(
            lambda path, payload, size: results.append((path, payload, size))
        )

        worker.run()

        assert predictor.batch_calls == [
            (image_paths, ["cat"], False),
        ]
        assert predictor.single_calls == []
        assert results == [
            (str(image_paths[0]), ("cat", 0.9), (0, 0)),
            (str(image_paths[1]), None, (0, 0)),
        ]


class TestThreadSafety:
    def test_batch_worker_cancelled_is_event(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
        )
        assert isinstance(worker._cancelled, threading.Event)

    def test_batch_worker_cancel_sets_event(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
        )
        assert not worker._cancelled.is_set()
        worker.cancel()
        assert worker._cancelled.is_set()

    def test_train_worker_has_mutex(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        worker = TrainWorker(config)
        assert isinstance(worker._trainer_mutex, QMutex)

    def test_train_worker_cancel_before_run(self, qapp):
        """Calling cancel() before run() should not raise."""
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        worker = TrainWorker(config)
        worker.cancel()  # Should not raise when _trainer is None

    def test_train_worker_cancel_calls_request_cancel(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.cancelled = True

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)

        # In production cancel() is invoked from another thread while train() is
        # running and self._trainer is populated. Simulate that here.
        def fake_train(cfg, on_epoch_end=None):
            worker.cancel()

        mock_trainer.train.side_effect = fake_train

        worker.run()
        mock_trainer.request_cancel.assert_called_once()


class TestSinglePredictWorker:
    def test_emits_done_with_annotations(self, qapp):
        from src.utils.workers import SinglePredictWorker
        from src.core.annotation import Annotation

        mock_predictor = MagicMock()
        ann = Annotation(
            class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4),
            confirmed=False, source="auto",
        )
        mock_predictor.predict.return_value = [ann]

        worker = SinglePredictWorker(
            predictor=mock_predictor,
            image_path=Path("/imgs/x.jpg"),
            conf=0.5, iou=0.45,
            project_classes=["cat"],
            class_match_mode="class_name",
        )
        results = []
        worker.done.connect(results.append)
        errors = []
        worker.error.connect(errors.append)

        worker.run()

        assert errors == []
        assert results == [[ann]]
        # Uses predict() (single-image path), not predict_with_size.
        assert mock_predictor.predict.call_args.kwargs["class_match_mode"] == "class_name"

    def test_emits_error_on_exception(self, qapp):
        from src.utils.workers import SinglePredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = RuntimeError("推理显存不足 (CUDA OOM)")

        worker = SinglePredictWorker(
            predictor=mock_predictor,
            image_path=Path("/imgs/x.jpg"),
        )
        results = []
        worker.done.connect(results.append)
        errors = []
        worker.error.connect(errors.append)

        worker.run()

        assert results == []
        assert errors and "OOM" in errors[0]
