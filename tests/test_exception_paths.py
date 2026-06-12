"""Tests for exception paths and edge cases."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestLabelIOExceptionPaths:
    def test_load_corrupt_json(self, tmp_path):
        from src.core.label_io import load_annotation

        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{ invalid json !!!", encoding="utf-8")
        result = load_annotation(bad_file)
        assert result is None

    def test_load_missing_keys(self, tmp_path):
        from src.core.label_io import load_annotation

        bad_file = tmp_path / "missing.json"
        bad_file.write_text('{"foo": "bar"}', encoding="utf-8")
        result = load_annotation(bad_file)
        assert result is None

    def test_load_nonexistent(self, tmp_path):
        from src.core.label_io import load_annotation

        result = load_annotation(tmp_path / "nope.json")
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path):
        from src.core.label_io import save_annotation
        from src.core.annotation import ImageAnnotation

        ia = ImageAnnotation(
            image_path="test.jpg", image_size=(640, 480), image_tags=["x"],
        )
        deep_path = tmp_path / "a" / "b" / "c" / "test.json"
        save_annotation(ia, deep_path)
        assert deep_path.exists()


class TestModelRegistryExceptionPaths:
    def test_load_corrupt_registry(self, tmp_path):
        from src.engine.model_manager import ModelRegistry

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "registry.json").write_text("not json!", encoding="utf-8")
        registry = ModelRegistry(models_dir)
        registry.load()
        assert registry.list_models() == []

    def test_load_missing_registry(self, tmp_path):
        from src.engine.model_manager import ModelRegistry

        registry = ModelRegistry(tmp_path / "models")
        registry.load()
        assert registry.list_models() == []

    def test_get_nonexistent_model(self, tmp_path):
        from src.engine.model_manager import ModelRegistry

        registry = ModelRegistry(tmp_path / "models")
        registry.load()
        assert registry.get("fake-id") is None


class TestProjectExceptionPaths:
    def test_open_nonexistent_dir(self):
        from src.core.project import ProjectManager

        with pytest.raises(FileNotFoundError):
            ProjectManager.open(Path("/tmp/nonexistent_project_12345"))

    def test_open_corrupt_project_json(self, tmp_path):
        from src.core.project import ProjectManager

        proj = tmp_path / "corrupt_proj"
        proj.mkdir()
        (proj / "project.json").write_text("invalid{{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ProjectManager.open(proj)


class TestWorkerConcurrency:
    def test_train_worker_cancel_during_training(self, qapp):
        from src.utils.workers import TrainWorker
        from src.engine.trainer import TrainConfig

        config = TrainConfig(data_yaml="data.yaml", model="yolov8n.pt", task="detect")
        mock_trainer_cls = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.cancelled = True

        cancel_called = []

        def fake_train(cfg, on_epoch_end=None):
            # Simulate cancel being called during training
            worker.cancel()
            cancel_called.append(True)

        mock_trainer.train.side_effect = fake_train

        worker = TrainWorker(config, trainer_cls=mock_trainer_cls)
        cancelled_signals = []
        worker.cancelled.connect(lambda: cancelled_signals.append(True))
        worker.run()
        assert len(cancelled_signals) == 1
        assert mock_trainer.request_cancel.called

    def test_batch_worker_cancel_mid_batch(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        call_count = [0]

        def predict_and_cancel(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                worker.cancel()
            return ([], (640, 480))

        mock_predictor.predict_with_size.side_effect = predict_and_cancel

        image_paths = [Path(f"/img{i}.jpg") for i in range(10)]
        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=image_paths,
        )
        finished = []
        worker.finished_ok.connect(lambda: finished.append(True))
        worker.run()
        # Should NOT have emitted finished_ok since it was cancelled
        assert len(finished) == 0
        assert call_count[0] < 10

    def test_batch_worker_error_during_inference(self, qapp):
        from src.utils.workers import BatchPredictWorker

        mock_predictor = MagicMock()
        mock_predictor.predict_with_size.side_effect = RuntimeError("GPU OOM")

        worker = BatchPredictWorker(
            predictor=mock_predictor,
            image_paths=[Path("/img.jpg")],
        )
        errors = []
        worker.error.connect(lambda msg: errors.append(msg))
        worker.run()
        assert len(errors) == 1
        assert "GPU OOM" in errors[0]
