"""Tests for controllers — project, model, train."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PyQt5.QtWidgets import QWidget


def _create_test_project(tmp_path: Path) -> Path:
    """Create a minimal project for testing."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "images").mkdir()
    (proj / "labels").mkdir()
    config = {
        "name": "test", "image_dir": "images", "label_dir": "labels",
        "classes": ["cat", "dog"], "version": "1.0", "created_at": "2026-01-01",
        "class_colors": {}, "keypoint_templates": {},
        "default_model": "", "auto_label_conf": 0.5, "auto_label_iou": 0.45,
    }
    (proj / "project.json").write_text(json.dumps(config), encoding="utf-8")
    return proj


class TestProjectController:
    def test_open_project(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        pm = ctrl.open_project(proj)
        assert pm is not None
        assert pm.config.name == "test"

    def test_open_nonexistent_returns_none(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        with patch("src.controllers.project.QMessageBox"):
            pm = ctrl.open_project(tmp_path / "nonexistent")
        assert pm is None

    def test_backup_manager_initialized_on_open(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(proj)
        assert ctrl.backup_manager is not None

    def test_create_backup(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(proj)
        result = ctrl.create_backup()
        assert result is not None
        assert result.exists()

    def test_list_backups_empty(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        proj = _create_test_project(tmp_path)
        config = AppConfig()
        ctrl = ProjectController(config, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(proj)
        assert ctrl.list_backups() == []

    @pytest.mark.parametrize("fmt", ["ImageFolder", "CSV"])
    def test_export_classification_formats_use_project_exporters(
        self, qapp, tmp_path, monkeypatch, fmt
    ):
        from src.controllers.project import ProjectController
        from src.core.annotation import ImageAnnotation
        from src.core.config import AppConfig
        from src.core.label_io import save_annotation
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "classify_project",
            "cls",
            classes=["cat"],
            task_type="classify",
        )
        img = pm.project_dir / pm.config.image_dir / "cat_001.jpg"
        img.write_text("fake image", encoding="utf-8")
        save_annotation(
            ImageAnnotation(
                image_path=img.name,
                image_size=(100, 100),
                image_tags=["cat"],
            ),
            pm.label_path_for(img),
        )
        out_dir = tmp_path / "export"
        out_dir.mkdir()

        class FakeExportDialog:
            def __init__(self, parent=None):
                pass

            def exec_(self):
                return True

            def get_values(self):
                return fmt, str(out_dir), False

        monkeypatch.setattr("src.controllers.project.ExportDialog", FakeExportDialog)

        ctrl = ProjectController(AppConfig(), tmp_path / "cfg.json", QWidget())
        ctrl.open_project(pm.project_dir)

        ctrl.export(pm)

        if fmt == "ImageFolder":
            assert (out_dir / "cat" / "cat_001.jpg").exists()
        else:
            assert (out_dir / "labels.csv").read_text(encoding="utf-8").splitlines() == [
                "filename,class",
                "cat_001.jpg,cat",
            ]

    def test_yolo_import_prefers_external_data_yaml_over_project_classes(self, qapp, tmp_path):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig

        labels_dir = tmp_path / "yolo"
        labels_dir.mkdir()
        (labels_dir / "img_001.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n",
            encoding="utf-8",
        )
        (labels_dir / "data.yaml").write_text(
            yaml.safe_dump({"names": ["person", "bicycle"], "nc": 2}),
            encoding="utf-8",
        )
        ctrl = ProjectController(AppConfig(), tmp_path / "cfg.json", QWidget())

        imported = ctrl._invoke_importer("YOLO", str(labels_dir), ["person"])

        assert [ann.class_name for ann in imported[0].annotations] == [
            "person",
            "bicycle",
        ]

    def test_import_merge_preserves_existing_image_tags(self, qapp, tmp_path):
        """Merge mode must not silently drop existing image_tags (Bug #6)."""
        from src.controllers.project import ProjectController
        from src.core.annotation import ImageAnnotation
        from src.core.config import AppConfig
        from src.core.label_io import load_annotation, save_annotation
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "p", "p", classes=["cat", "dog"], task_type="classify",
        )
        img = pm.project_dir / pm.config.image_dir / "x.jpg"
        img.write_text("fake")
        save_annotation(
            ImageAnnotation(
                image_path="x.jpg", image_size=(100, 100),
                image_tags=["cat"], image_tags_confirmed=True, image_tags_source="manual",
            ),
            pm.label_path_for(img),
        )

        # Imported has no image_tags but a stray annotation
        imported = [
            ImageAnnotation(image_path="x.jpg", image_size=(100, 100), image_tags=[]),
        ]

        ctrl = ProjectController(AppConfig(), tmp_path / "cfg.json", QWidget())
        ctrl.open_project(pm.project_dir)

        with patch("src.controllers.project.ImportDialog") as DlgMock, \
                patch.object(ctrl, "_invoke_importer", return_value=imported), \
                patch.object(ctrl, "create_backup"), \
                patch("src.controllers.project.QMessageBox"):
            dlg = MagicMock()
            dlg.exec_.return_value = True
            dlg.get_values.return_value = ("YOLO", str(tmp_path), "merge")
            DlgMock.return_value = dlg
            ctrl.import_annotations(pm)

        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["cat"]
        assert ia.image_tags_confirmed is True
        assert ia.image_tags_source == "manual"

    def test_import_merge_adopts_image_tags_when_existing_empty(self, qapp, tmp_path):
        """When existing has no tags, merge should adopt imported image_tags (Bug #6)."""
        from src.controllers.project import ProjectController
        from src.core.annotation import ImageAnnotation
        from src.core.config import AppConfig
        from src.core.label_io import load_annotation, save_annotation
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "p", "p", classes=["cat", "dog"], task_type="classify",
        )
        img = pm.project_dir / pm.config.image_dir / "y.jpg"
        img.write_text("fake")
        save_annotation(
            ImageAnnotation(image_path="y.jpg", image_size=(100, 100), image_tags=[]),
            pm.label_path_for(img),
        )

        imported = [
            ImageAnnotation(
                image_path="y.jpg", image_size=(100, 100),
                image_tags=["dog"], image_tags_confirmed=True, image_tags_source="manual",
            ),
        ]

        ctrl = ProjectController(AppConfig(), tmp_path / "cfg.json", QWidget())
        ctrl.open_project(pm.project_dir)

        with patch("src.controllers.project.ImportDialog") as DlgMock, \
                patch.object(ctrl, "_invoke_importer", return_value=imported), \
                patch.object(ctrl, "create_backup"), \
                patch("src.controllers.project.QMessageBox"):
            dlg = MagicMock()
            dlg.exec_.return_value = True
            dlg.get_values.return_value = ("YOLO", str(tmp_path), "merge")
            DlgMock.return_value = dlg
            ctrl.import_annotations(pm)

        ia = load_annotation(pm.label_path_for(img))
        assert ia.image_tags == ["dog"]
        assert ia.image_tags_confirmed is True

    def test_import_imagefolder_via_controller(self, qapp, tmp_path):
        """ImageFolder import should work end-to-end via ImportRegistry (Bug #3)."""
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig
        from src.core.label_io import load_annotation
        from src.core.project import ProjectManager

        # Build source ImageFolder dataset
        src = tmp_path / "src"
        (src / "cat").mkdir(parents=True)
        (src / "dog").mkdir(parents=True)
        (src / "cat" / "a.jpg").write_text("a")
        (src / "dog" / "b.jpg").write_text("b")

        pm = ProjectManager.create(
            tmp_path / "p", "p", classes=[], task_type="classify",
        )
        ctrl = ProjectController(AppConfig(), tmp_path / "cfg.json", QWidget())
        ctrl.open_project(pm.project_dir)

        with patch("src.controllers.project.ImportDialog") as DlgMock, \
                patch.object(ctrl, "create_backup"), \
                patch("src.controllers.project.QMessageBox"):
            dlg = MagicMock()
            dlg.exec_.return_value = True
            dlg.get_values.return_value = ("ImageFolder", str(src), "skip")
            DlgMock.return_value = dlg
            ctrl.import_annotations(pm)

        images_dir = pm.project_dir / pm.config.image_dir
        assert (images_dir / "a.jpg").exists()
        assert (images_dir / "b.jpg").exists()

        ia_a = load_annotation(pm.label_path_for(images_dir / "a.jpg"))
        ia_b = load_annotation(pm.label_path_for(images_dir / "b.jpg"))
        assert ia_a.image_tags == ["cat"]
        assert ia_b.image_tags == ["dog"]
        assert "cat" in pm.config.classes and "dog" in pm.config.classes


class TestTrainController:
    def test_validate_returns_none_when_user_cancels(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._prepared_classes = ["stale"]
        ctrl._has_prepared_classes = True

        with patch("src.controllers.train.QMessageBox") as mock_mb:
            mock_mb.question.return_value = mock_mb.No
            mock_mb.Yes = 0x00004000
            mock_mb.No = 0x00010000
            result = ctrl.validate_and_prepare(pm, "detect", 0.2)
            assert result is None
        assert ctrl._prepared_classes == []
        assert ctrl._has_prepared_classes is False

    def test_validate_and_prepare_stores_effective_classes_from_generated_data_yaml(
        self, qapp, tmp_path
    ):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())

        fake_yaml = tmp_path / "data.yaml"
        fake_yaml.write_text(yaml.safe_dump({"names": ["dog"], "nc": 1}), encoding="utf-8")

        with patch("src.controllers.train.DatasetPreparer.prepare", return_value=fake_yaml):
            with patch("src.controllers.train.QMessageBox") as mock_mb:
                mock_mb.Yes = 0x00004000
                mock_mb.No = 0x00010000
                mock_mb.question.return_value = mock_mb.Yes
                result = ctrl.validate_and_prepare(pm, "detect", 0.2)

        assert result == str(fake_yaml)
        assert ctrl._prepared_classes == ["dog"]
        assert ctrl._has_prepared_classes is True

    def test_classify_validation_counts_only_confirmed_image_tags(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.annotation import ImageAnnotation
        from src.core.label_io import save_annotation
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "classify_project",
            "cls",
            classes=["cat"],
            task_type="classify",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        for i in range(10):
            img = img_dir / f"img_{i}.jpg"
            img.write_text("fake image", encoding="utf-8")
            save_annotation(
                ImageAnnotation(
                    image_path=img.name,
                    image_size=(100, 100),
                    image_tags=["cat"],
                    image_tags_confirmed=False,
                    image_tags_source="auto",
                ),
                pm.label_path_for(img),
            )

        ctrl = TrainController(QWidget())
        with patch("src.controllers.train.DatasetPreparer.prepare") as mock_prepare:
            with patch("src.controllers.train.QMessageBox") as mock_mb:
                mock_mb.Yes = 0x00004000
                mock_mb.No = 0x00010000
                mock_mb.question.return_value = mock_mb.No
                result = ctrl.validate_and_prepare(pm, "classify", 0.2)

        assert result is None
        assert ctrl.dataset_size == 0
        mock_prepare.assert_not_called()

    def test_register_model_uses_prepared_classes_instead_of_project_classes(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._run_name = "detect-20260427-000000"
        ctrl._prepared_classes = ["dog"]
        ctrl._has_prepared_classes = True
        ctrl._project_at_start = pm
        ctrl._task_at_start = "detect"
        ctrl._base_model_at_start = "yolov8n.pt"
        ctrl._train_config = TrainConfig(
            data_yaml="/tmp/data.yaml", model="yolov8n.pt", task="detect", epochs=10,
        )

        model = ctrl.register_model_after_training({"mAP50": 0.7})

        assert model.classes == ["dog"]

    def test_register_model_preserves_intentionally_empty_prepared_classes(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._run_name = "detect-20260427-000000"
        ctrl._prepared_classes = []
        ctrl._has_prepared_classes = True
        ctrl._project_at_start = pm
        ctrl._task_at_start = "detect"
        ctrl._base_model_at_start = "yolov8n.pt"
        ctrl._train_config = TrainConfig(
            data_yaml="/tmp/data.yaml", model="yolov8n.pt", task="detect", epochs=10,
        )

        model = ctrl.register_model_after_training({"mAP50": 0.7})

        assert model.classes == []

    def test_register_model_persists_train_params_from_started_config(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._run_name = "detect-20260508-000000"
        ctrl._project_at_start = pm
        ctrl._task_at_start = "detect"
        ctrl._base_model_at_start = "yolov8n.pt"
        ctrl._train_config = TrainConfig(
            data_yaml="/tmp/data.yaml", model="yolov8n.pt", task="detect",
            epochs=42, batch=8, lr0=0.005,
            include_detect_params=True, mosaic=0.7,
            project="/tmp/out", name="detect-20260508-000000",
        )

        model = ctrl.register_model_after_training({"mAP50": 0.8})

        assert model.train_params["epochs"] == 42
        assert model.train_params["lr0"] == 0.005
        assert model.train_params["mosaic"] == 0.7
        assert model.train_params["include_detect_params"] is True
        assert "data_yaml" not in model.train_params
        assert "project" not in model.train_params

    def test_register_model_persists_backend_metadata(self, qapp, tmp_path, monkeypatch):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        class Probe:
            version = "9.9.9"
            runtime = "external-python"
            metadata = {"env": "yolo-env"}

        backend = MagicMock()
        backend.backend_id = "custom-backend"
        backend.probe.return_value = Probe()
        backend.infer_model_format.return_value = "pt"
        backend_ids = []

        def fake_get_backend(backend_id):
            backend_ids.append(backend_id)
            return backend

        monkeypatch.setattr("src.controllers.train.get_backend", fake_get_backend)
        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._run_name = "detect-20260508-000000"
        ctrl._project_at_start = pm
        ctrl._task_at_start = "detect"
        ctrl._base_model_at_start = "base.pt"
        ctrl._train_config = TrainConfig(
            data_yaml="/tmp/data.yaml", model="base.pt", task="detect",
            backend_id="custom-backend",
        )

        model = ctrl.register_model_after_training({})

        assert backend_ids == ["custom-backend"]
        assert model.backend_id == "custom-backend"
        assert model.model_format == "pt"
        assert model.backend_version == "9.9.9"
        assert model.backend_runtime == "external-python"
        assert model.backend_metadata == {"env": "yolo-env"}
        backend.infer_model_format.assert_called_once_with("models/detect-20260508-000000/weights/best.pt")

    def test_register_model_train_params_empty_without_started_config(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._run_name = "detect-20260508-000000"
        ctrl._project_at_start = pm
        ctrl._task_at_start = "detect"
        ctrl._base_model_at_start = "yolov8n.pt"

        model = ctrl.register_model_after_training({})

        assert model.train_params == {}

    def test_register_model_pose_includes_kpt_shape(self, qapp, tmp_path):
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        ctrl._run_name = "pose-20260508-000000"
        ctrl._project_at_start = pm
        ctrl._task_at_start = "pose"
        ctrl._base_model_at_start = "yolov8n-pose.pt"
        ctrl._train_config = TrainConfig(
            data_yaml="/tmp/data.yaml", model="yolov8n-pose.pt", task="pose",
            include_pose_params=True, pose=15.0, kobj=2.0, kpt_shape=[17, 3],
        )

        model = ctrl.register_model_after_training({})

        assert model.train_params["kpt_shape"] == [17, 3]
        assert model.train_params["pose"] == 15.0
        assert model.train_params["kobj"] == 2.0

    def test_stop_without_worker(self, qapp):
        from src.controllers.train import TrainController

        ctrl = TrainController(QWidget())
        ctrl.stop()  # should not raise

    def test_start_rejects_when_worker_already_running(self, qapp, tmp_path):
        from unittest.mock import MagicMock
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        proj = _create_test_project(tmp_path)
        pm = ProjectManager.open(proj)
        ctrl = TrainController(QWidget())
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        ctrl._worker = fake_worker

        with pytest.raises(RuntimeError):
            ctrl.start(
                TrainConfig(data_yaml="d.yaml", model="yolov8n.pt", task="detect"),
                pm, "detect", base_model="yolov8n.pt",
            )

    def test_register_after_training_writes_to_snapshot_project_not_current(
        self, qapp, tmp_path
    ):
        """If user switches projects mid-training, the model goes to the project
        that started the training, not whichever project is currently open."""
        import json
        from src.controllers.train import TrainController
        from src.core.project import ProjectManager
        from src.engine.trainer import TrainConfig

        # Build two independent projects A and B.
        proj_a = tmp_path / "A"
        ProjectManager.create(proj_a, "A", classes=["cat"])
        proj_b = tmp_path / "B"
        ProjectManager.create(proj_b, "B", classes=["dog"])
        pm_a = ProjectManager.open(proj_a)

        ctrl = TrainController(QWidget())
        # Snapshot says training was started on A.
        ctrl._run_name = "detect-snapshot"
        ctrl._project_at_start = pm_a
        ctrl._task_at_start = "detect"
        ctrl._base_model_at_start = "yolov8n.pt"
        ctrl._train_config = TrainConfig(
            data_yaml="/tmp/data.yaml", model="yolov8n.pt", task="detect", epochs=5,
        )
        ctrl._prepared_classes = ["cat"]
        ctrl._has_prepared_classes = True

        model = ctrl.register_model_after_training({"mAP50": 0.9})

        assert model is not None
        # File landed in A, not B.
        a_registry = proj_a / "models" / "registry.json"
        b_registry = proj_b / "models" / "registry.json"
        assert a_registry.exists()
        assert not b_registry.exists()
        a_data = json.loads(a_registry.read_text(encoding="utf-8"))
        assert len(a_data["models"]) == 1
        assert a_data["models"][0]["classes"] == ["cat"]
        assert a_data["models"][0]["path"] == "models/detect-snapshot/weights/best.pt"

    def test_register_after_training_returns_none_without_start(self, qapp):
        from src.controllers.train import TrainController

        ctrl = TrainController(QWidget())
        assert ctrl.register_model_after_training({}) is None

    def test_classify_prepared_classes_sorted_to_match_ultralytics(self, qapp, tmp_path):
        """Bug #7: classify _prepared_classes must match ultralytics' alphabetical order."""
        from src.controllers.train import TrainController
        from src.core.annotation import ImageAnnotation
        from src.core.label_io import save_annotation
        from src.core.project import ProjectManager

        # Project classes in non-alphabetical order
        pm = ProjectManager.create(
            tmp_path / "p", "p",
            classes=["dog", "cat"], task_type="classify",
        )
        img_dir = pm.project_dir / pm.config.image_dir
        for cls in ("cat", "dog"):
            for i in range(2):
                img = img_dir / f"{cls}_{i}.jpg"
                img.write_text("fake")
                save_annotation(
                    ImageAnnotation(
                        image_path=img.name, image_size=(10, 10),
                        image_tags=[cls], image_tags_confirmed=True,
                    ),
                    pm.label_path_for(img),
                )

        ctrl = TrainController(QWidget())
        with patch("src.controllers.train.QMessageBox") as mock_mb:
            mock_mb.Yes = 0x00004000
            mock_mb.No = 0x00010000
            mock_mb.question.return_value = mock_mb.Yes
            result = ctrl.validate_and_prepare(pm, "classify", 0.2)

        assert result is not None
        # Must be sorted to match ultralytics' subdir-name iteration order
        assert ctrl._prepared_classes == ["cat", "dog"]


class TestModelController:
    def test_predict_single_without_model(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        with patch("src.controllers.model.QMessageBox"):
            result = ctrl.predict_single(Path("/fake.jpg"), ["cat"])
        assert result == []

    def test_create_single_predict_worker_without_model_returns_none(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        with patch("src.controllers.model.QMessageBox"):
            worker = ctrl.create_single_predict_worker(Path("/fake.jpg"), ["cat"])
        assert worker is None

    def test_create_single_predict_worker_builds_worker(self, qapp):
        from src.controllers.model import ModelController
        from src.utils.workers import SinglePredictWorker

        ctrl = ModelController(QWidget())
        ctrl.set_predictor(object())
        worker = ctrl.create_single_predict_worker(
            Path("/fake.jpg"), ["cat"], conf=0.4, iou=0.5,
            class_match_mode="class_name",
        )
        assert isinstance(worker, SinglePredictWorker)
        assert worker._class_match_mode == "class_name"

    def test_load_model_without_context(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        assert ctrl.load_model("fake-id") is False

    def test_delete_model_without_registry(self, qapp):
        from src.controllers.model import ModelController

        ctrl = ModelController(QWidget())
        assert ctrl.delete_model("fake-id") is False

    def test_load_model_uses_registered_backend(self, qapp, tmp_path, monkeypatch):
        from src.controllers.model import ModelController
        from src.core.project import ProjectManager
        from src.engine.model_manager import ModelInfo, ModelRegistry

        pm = ProjectManager.create(tmp_path / "p", "p", classes=["cat"])
        model_path = pm.project_dir / "model.fake"
        model_path.write_text("fake", encoding="utf-8")
        registry = ModelRegistry(pm.project_dir / "models")
        info = ModelInfo(
            name="fake-model", path="model.fake", task="detect",
            base_model="fake-base", classes=["cat"], backend_id="fake-backend",
        )
        registry.register(info)

        predictor = object()
        backend = MagicMock()
        backend.backend_id = "fake-backend"
        backend.load_predictor.return_value = predictor
        backend_ids = []

        def fake_get_backend(backend_id):
            backend_ids.append(backend_id)
            return backend

        monkeypatch.setattr("src.controllers.model.get_backend", fake_get_backend)
        ctrl = ModelController(QWidget())
        ctrl.set_context(pm, registry)

        assert ctrl.load_model(info.id) is True
        assert ctrl.predictor is predictor
        assert backend_ids == ["fake-backend"]
        backend.load_predictor.assert_called_once_with(model_path, info)

    def test_import_model_prefills_task_from_project(self, qapp, tmp_path, monkeypatch):
        from src.controllers.model import ModelController
        from src.core.project import ProjectManager
        from src.engine.model_manager import ModelRegistry
        from src.engine.backends.base import DEFAULT_BACKEND_ID, DEFAULT_BACKEND_RUNTIME

        captured: dict = {}

        def fake_get_open_filename(*args, **kwargs):
            return (str(tmp_path / "fake.pt"), "PyTorch")

        def fake_get_text(*args, **kwargs):
            return ("model-name", True)

        def fake_get_item(parent, title, prompt, items, current, editable):
            captured["items"] = list(items)
            captured["current"] = current
            return (items[current], True)

        monkeypatch.setattr(
            "src.controllers.model.QFileDialog.getOpenFileName", fake_get_open_filename
        )
        monkeypatch.setattr(
            "src.controllers.model.QInputDialog.getText", fake_get_text
        )
        monkeypatch.setattr(
            "src.controllers.model.QInputDialog.getItem", fake_get_item
        )

        for project_task, expected_idx in [("detect", 0), ("classify", 1), ("pose", 2)]:
            project_dir = tmp_path / f"proj_{project_task}"
            pm = ProjectManager.create(
                project_dir, f"p_{project_task}",
                classes=["a"],
                task_type=project_task,
            )
            registry = ModelRegistry(pm.project_dir / "models")
            ctrl = ModelController(QWidget())
            ctrl.set_context(pm, registry)

            captured.clear()
            imported = ctrl.import_model()

            assert imported is not None
            assert imported.backend_id == DEFAULT_BACKEND_ID
            assert imported.model_format == "pt"
            assert imported.backend_runtime == DEFAULT_BACKEND_RUNTIME
            assert captured["items"] == ["detect", "classify", "pose"]
            assert captured["current"] == expected_idx, (
                f"task={project_task}: expected default_idx={expected_idx}, got {captured['current']}"
            )


class TestRegisterAutoClass:
    """ProjectController.register_auto_class — 自动登记决策逻辑。"""

    def _make_controller(self, qapp, tmp_path: Path, *, classes=None, enabled=True):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "p", "p",
            classes=list(classes) if classes else [],
            task_type="classify",
        )
        pm.config.auto_register_classes = enabled
        pm.save()
        cfg = AppConfig()
        ctrl = ProjectController(cfg, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(pm.project_dir)
        return ctrl, ctrl.project

    def test_registers_new_class(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path, classes=["cat"])
        result = ctrl.register_auto_class("dog")
        assert result.action == "registered"
        assert result.applied_name == "dog"
        assert "dog" in pm.config.classes
        # Persisted to disk
        from src.core.project import ProjectManager
        reopened = ProjectManager.open(pm.project_dir)
        assert "dog" in reopened.config.classes

    def test_existing_class_no_change(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path, classes=["cat", "dog"])
        result = ctrl.register_auto_class("cat")
        assert result.action == "existing"
        assert result.applied_name == "cat"
        assert pm.config.classes == ["cat", "dog"]

    def test_blacklist_imagenet_id_rejected(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path)
        result = ctrl.register_auto_class("n01440764")
        assert result.action == "rejected_blacklist"
        assert result.applied_name is None
        assert pm.config.classes == []

    def test_invalid_empty_string_rejected(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path)
        result = ctrl.register_auto_class("   ")
        assert result.action == "rejected_invalid"
        assert pm.config.classes == []

    def test_invalid_too_long_rejected(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path)
        result = ctrl.register_auto_class("x" * 65)
        assert result.action == "rejected_invalid"

    def test_disabled_switch_rejects_new_class(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path, classes=["cat"], enabled=False)
        result = ctrl.register_auto_class("dog")
        assert result.action == "rejected_disabled"
        assert pm.config.classes == ["cat"]

    def test_disabled_switch_existing_class_still_existing(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path, classes=["cat"], enabled=False)
        result = ctrl.register_auto_class("cat")
        assert result.action == "existing"

    def test_force_bypasses_disabled_switch(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path, classes=[], enabled=False)
        result = ctrl.register_auto_class("dog", force=True)
        assert result.action == "registered"
        assert "dog" in pm.config.classes

    def test_force_does_not_bypass_blacklist(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path)
        result = ctrl.register_auto_class("n01440764", force=True)
        assert result.action == "rejected_blacklist"
        assert pm.config.classes == []

    def test_strips_whitespace(self, qapp, tmp_path):
        ctrl, pm = self._make_controller(qapp, tmp_path)
        result = ctrl.register_auto_class("  dog  ")
        assert result.action == "registered"
        assert result.applied_name == "dog"
        assert "dog" in pm.config.classes


class TestPreviewModelClasses:
    """ProjectController.preview_model_classes — diff model.names against project.classes."""

    def _make_controller(self, qapp, tmp_path: Path, *, classes=None):
        from src.controllers.project import ProjectController
        from src.core.config import AppConfig
        from src.core.project import ProjectManager

        pm = ProjectManager.create(
            tmp_path / "p", "p",
            classes=list(classes) if classes else [],
            task_type="classify",
        )
        cfg = AppConfig()
        ctrl = ProjectController(cfg, tmp_path / "cfg.json", QWidget())
        ctrl.open_project(pm.project_dir)
        return ctrl

    def _make_predictor(self, names):
        pred = MagicMock()
        pred.model.names = names
        return pred

    def test_returns_only_new_classes(self, qapp, tmp_path):
        ctrl = self._make_controller(qapp, tmp_path, classes=["cat"])
        pred = self._make_predictor({0: "cat", 1: "dog", 2: "bird"})
        items = ctrl.preview_model_classes(pred)
        names = [it.model_name for it in items]
        assert "cat" not in names
        assert set(names) == {"dog", "bird"}

    def test_marks_blacklisted_classes(self, qapp, tmp_path):
        ctrl = self._make_controller(qapp, tmp_path, classes=[])
        pred = self._make_predictor({0: "dog", 1: "n01440764"})
        items = ctrl.preview_model_classes(pred)
        by_name = {it.model_name: it for it in items}
        assert by_name["dog"].is_blacklisted is False
        assert by_name["dog"].default_checked is True
        assert by_name["n01440764"].is_blacklisted is True
        assert by_name["n01440764"].default_checked is False

    def test_handles_list_names(self, qapp, tmp_path):
        ctrl = self._make_controller(qapp, tmp_path, classes=[])
        pred = self._make_predictor(["cat", "dog"])
        items = ctrl.preview_model_classes(pred)
        assert {it.model_name for it in items} == {"cat", "dog"}

    def test_predictor_none_returns_empty(self, qapp, tmp_path):
        ctrl = self._make_controller(qapp, tmp_path)
        assert ctrl.preview_model_classes(None) == []

    def test_predictor_without_names_returns_empty(self, qapp, tmp_path):
        ctrl = self._make_controller(qapp, tmp_path)
        pred = MagicMock()
        pred.model = MagicMock(spec=[])  # no .names attribute
        assert ctrl.preview_model_classes(pred) == []

    def test_empty_model_names_returns_empty(self, qapp, tmp_path):
        ctrl = self._make_controller(qapp, tmp_path)
        pred = self._make_predictor({})
        assert ctrl.preview_model_classes(pred) == []

    def test_skips_invalid_names(self, qapp, tmp_path):
        """空字符串 / 过长的类名不进入预览。"""
        ctrl = self._make_controller(qapp, tmp_path)
        pred = self._make_predictor({0: "  ", 1: "x" * 65, 2: "valid"})
        items = ctrl.preview_model_classes(pred)
        assert {it.model_name for it in items} == {"valid"}
