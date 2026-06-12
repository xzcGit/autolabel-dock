"""Tests for training engine."""
from unittest.mock import MagicMock

from src.engine.backends.base import DEFAULT_BACKEND_ID
from src.engine.trainer import TrainConfig, Trainer


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.device == ""
        assert cfg.optimizer == "auto"
        assert cfg.freeze is None
        assert cfg.workers == 8
        assert cfg.patience == 100
        assert cfg.erasing == 0.4
        assert cfg.auto_augment == "randaugment"
        assert cfg.dropout == 0.0
        assert cfg.include_detect_params is False
        assert cfg.include_classify_params is False
        assert cfg.include_pose_params is False
        assert cfg.backend_id == DEFAULT_BACKEND_ID

    def test_to_train_args_defaults_keep_specialist_groups_out(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )

        args = cfg.to_train_args()

        assert "degrees" not in args
        assert "erasing" not in args
        assert "pose" not in args

    def test_to_train_args(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=50,
            batch=8,
            freeze=3,
            workers=4,
            patience=20,
            project="/out",
            name="run1",
        )
        args = cfg.to_train_args()
        assert args["data"] == "/path/data.yaml"
        assert args["epochs"] == 50
        assert args["batch"] == 8
        assert args["freeze"] == 3
        assert args["workers"] == 4
        assert args["patience"] == 20
        assert args["project"] == "/out"
        assert args["name"] == "run1"

    def test_to_train_args_excludes_empty(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )
        args = cfg.to_train_args()
        assert "device" not in args
        assert "freeze" not in args

    def test_to_train_args_omits_detect_group_when_disabled(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
            include_detect_params=False,
            degrees=15.0,
            translate=0.2,
            shear=5.0,
            perspective=0.0005,
            mosaic=0.5,
            mixup=0.2,
            copy_paste=0.1,
        )

        args = cfg.to_train_args()

        assert "degrees" not in args
        assert "translate" not in args
        assert "shear" not in args
        assert "perspective" not in args
        assert "mosaic" not in args
        assert "mixup" not in args
        assert "copy_paste" not in args
        assert args["scale"] == cfg.scale

    def test_to_train_args_includes_detect_group_for_classify_when_enabled(self):
        cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n-cls.pt",
            task="classify",
            include_detect_params=True,
            degrees=15.0,
            translate=0.2,
            shear=5.0,
            perspective=0.0005,
            mosaic=0.5,
            mixup=0.2,
            copy_paste=0.1,
        )

        args = cfg.to_train_args()

        assert args["degrees"] == 15.0
        assert args["translate"] == 0.2
        assert args["shear"] == 5.0
        assert args["perspective"] == 0.0005
        assert args["mosaic"] == 0.5
        assert args["mixup"] == 0.2
        assert args["copy_paste"] == 0.1
        assert "erasing" not in args
        assert "auto_augment" not in args

    def test_to_train_args_omits_classify_group_when_disabled(self):
        disabled_cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n-cls.pt",
            task="classify",
            erasing=0.7,
            auto_augment="augmix",
            dropout=0.3,
        )
        disabled_args = disabled_cfg.to_train_args()
        assert "erasing" not in disabled_args
        assert "auto_augment" not in disabled_args
        assert "dropout" not in disabled_args

    def test_to_train_args_includes_classify_group_when_enabled(self):
        enabled_cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
            include_classify_params=True,
            erasing=0.7,
            auto_augment="augmix",
            dropout=0.3,
        )

        args = enabled_cfg.to_train_args()

        assert args["erasing"] == 0.7
        assert args["auto_augment"] == "augmix"
        assert args["dropout"] == 0.3

    def test_to_train_args_includes_pose_group_only_when_enabled(self):
        disabled_cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n-pose.pt",
            task="pose",
            pose=15.0,
            kobj=2.0,
        )
        disabled_args = disabled_cfg.to_train_args()
        assert "pose" not in disabled_args
        assert "kobj" not in disabled_args

        enabled_cfg = TrainConfig(
            data_yaml="/path/data.yaml",
            model="yolov8n.pt",
            task="detect",
            include_pose_params=True,
            pose=15.0,
            kobj=2.0,
        )

        args = enabled_cfg.to_train_args()

        assert args["pose"] == 15.0
        assert args["kobj"] == 2.0


class TestTrainConfigStorage:
    def test_to_storage_dict_excludes_runtime_fields(self):
        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            project="/out",
            name="run-1",
            resume=True,
        )

        snapshot = cfg.to_storage_dict()

        for k in ("data_yaml", "project", "name", "resume", "backend_id"):
            assert k not in snapshot

    def test_to_storage_dict_keeps_all_hyperparams(self):
        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=42,
            batch=8,
            lr0=0.005,
            freeze=3,
            include_detect_params=True,
            mosaic=0.7,
        )

        snapshot = cfg.to_storage_dict()

        assert snapshot["model"] == "yolov8n.pt"
        assert snapshot["task"] == "detect"
        assert snapshot["epochs"] == 42
        assert snapshot["batch"] == 8
        assert snapshot["lr0"] == 0.005
        assert snapshot["freeze"] == 3
        assert snapshot["include_detect_params"] is True
        assert snapshot["mosaic"] == 0.7

    def test_to_storage_dict_pose_includes_kpt_shape(self):
        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n-pose.pt",
            task="pose",
            include_pose_params=True,
            pose=15.0,
            kobj=2.0,
            kpt_shape=[17, 3],
        )

        snapshot = cfg.to_storage_dict()

        assert snapshot["task"] == "pose"
        assert snapshot["kpt_shape"] == [17, 3]
        assert snapshot["pose"] == 15.0
        assert snapshot["kobj"] == 2.0
        assert snapshot["include_pose_params"] is True


class TestTrainer:
    def test_train_calls_yolo(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=10,
            freeze=2,
            workers=2,
            patience=5,
            project="/out",
            name="test",
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)

        mock_yolo_cls.assert_called_once_with("yolov8n.pt", task="detect")
        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["data"] == "/data.yaml"
        assert train_kwargs["epochs"] == 10
        assert train_kwargs["task"] == "detect"
        assert train_kwargs["freeze"] == 2
        assert train_kwargs["workers"] == 2
        assert train_kwargs["patience"] == 5

    def test_train_with_callback(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        def on_epoch(metrics: dict):
            pass

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
            epochs=5,
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg, on_epoch_end=on_epoch)

        mock_model.add_callback.assert_called()

    def test_train_resume(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="/out/test/weights/last.pt",
            task="detect",
            resume=True,
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)

        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["resume"] is True

    def test_get_best_metrics(self):
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.trainer = MagicMock()
        mock_model.trainer.best_fitness = 0.85
        mock_model.trainer.metrics = {
            "metrics/mAP50(B)": 0.89,
            "metrics/mAP50-95(B)": 0.67,
        }
        mock_model.train.return_value = MagicMock()

        cfg = TrainConfig(
            data_yaml="/data.yaml",
            model="yolov8n.pt",
            task="detect",
        )

        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)
        metrics = trainer.get_best_metrics()

        assert metrics["mAP50"] == 0.89
        assert metrics["mAP50-95"] == 0.67

    def test_cancel_mid_epoch_via_batch_callback(self):
        """Simulates the inner ultralytics batch loop firing on_train_batch_end.

        After request_cancel(), the registered batch callback must raise so the
        inner loop breaks immediately; Trainer.train must swallow that signal
        and leave the trainer in the cancelled state.
        """
        mock_yolo_cls = MagicMock()
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        callbacks: dict[str, list] = {}

        def fake_add_callback(event, fn):
            callbacks.setdefault(event, []).append(fn)

        mock_model.add_callback.side_effect = fake_add_callback

        def fake_train(**_kwargs):
            # Simulate ultralytics' inner batch loop: a few batches, then a
            # cancel request lands and the next batch callback must abort.
            ult_trainer = MagicMock()
            for _ in range(3):
                for cb in callbacks.get("on_train_batch_end", []):
                    cb(ult_trainer)
            trainer.request_cancel()
            # Next batch callback after cancel — should raise _TrainingCancelled
            # and propagate up past `fake_train` if Trainer.train didn't catch
            # it. We let it propagate to verify Trainer.train swallows it.
            for cb in callbacks.get("on_train_batch_end", []):
                cb(ult_trainer)
            raise AssertionError("batch callback should have raised before this")

        mock_model.train.side_effect = fake_train

        cfg = TrainConfig(data_yaml="/data.yaml", model="yolov8n.pt", task="detect")
        trainer = Trainer(yolo_cls=mock_yolo_cls)
        trainer.train(cfg)  # must not raise

        assert trainer.cancelled is True
        events_registered = list(callbacks.keys())
        assert "on_train_batch_end" in events_registered
        assert "on_fit_epoch_end" in events_registered
