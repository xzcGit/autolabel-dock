"""Tests for model registry and management."""
import json
from pathlib import Path

from src.engine.backends.base import DEFAULT_BACKEND_ID, DEFAULT_BACKEND_RUNTIME
from src.engine.model_manager import ModelInfo, ModelRegistry


class TestModelInfo:
    def test_create(self):
        info = ModelInfo(
            name="yolov8n-custom",
            path="models/yolov8n-custom/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["person", "car"],
        )
        assert info.name == "yolov8n-custom"
        assert info.task == "detect"
        assert info.id  # should have a UUID

    def test_to_dict_roundtrip(self):
        info = ModelInfo(
            name="test",
            path="models/test/best.pt",
            task="pose",
            base_model="yolov8n-pose.pt",
            classes=["person"],
            metrics={"mAP50": 0.89},
            epochs=100,
            dataset_size=500,
        )
        d = info.to_dict()
        restored = ModelInfo.from_dict(d)
        assert restored.name == "test"
        assert restored.metrics == {"mAP50": 0.89}
        assert restored.id == info.id

    def test_backend_metadata_roundtrip(self):
        info = ModelInfo(
            name="onnx-model",
            path="models/imported/model.onnx",
            task="detect",
            base_model="imported",
            classes=["person"],
            backend_id="ultralytics",
            model_format="onnx",
            backend_version="8.3.0",
            backend_runtime="in_process",
            backend_metadata={"python": "3.11"},
        )

        restored = ModelInfo.from_dict(info.to_dict())

        assert restored.backend_id == "ultralytics"
        assert restored.model_format == "onnx"
        assert restored.backend_version == "8.3.0"
        assert restored.backend_runtime == "in_process"
        assert restored.backend_metadata == {"python": "3.11"}

    def test_from_dict_legacy_defaults_to_ultralytics_backend(self):
        legacy = {
            "id": "legacy-id",
            "name": "legacy",
            "path": "models/legacy/weights/best.pt",
            "task": "detect",
            "base_model": "yolov8n.pt",
            "classes": ["cat"],
        }

        info = ModelInfo.from_dict(legacy)

        assert info.backend_id == DEFAULT_BACKEND_ID
        assert info.model_format == "pt"
        assert info.backend_version == ""
        assert info.backend_runtime == DEFAULT_BACKEND_RUNTIME
        assert info.backend_metadata == {}


class TestModelRegistry:
    def test_create_empty(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        assert registry.list_models() == []

    def test_register_and_list(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        models = registry.list_models()
        assert len(models) == 1
        assert models[0].name == "model-a"

    def test_save_and_load(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        registry.save()
        # Load fresh
        registry2 = ModelRegistry(tmp_path / "models")
        registry2.load()
        assert len(registry2.list_models()) == 1
        assert registry2.list_models()[0].name == "model-a"

    def test_remove_model(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        registry.remove(info.id)
        assert registry.list_models() == []

    def test_get_by_id(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(
            name="model-a",
            path="models/model-a/best.pt",
            task="detect",
            base_model="yolov8n.pt",
            classes=["cat"],
        )
        registry.register(info)
        found = registry.get(info.id)
        assert found is not None
        assert found.name == "model-a"
        assert registry.get("nonexistent") is None

    def test_load_nonexistent_is_empty(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        registry.load()  # no file yet
        assert registry.list_models() == []

    def test_list_by_task(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        registry.register(ModelInfo(name="det", path="a.pt", task="detect", base_model="yolov8n.pt", classes=["a"]))
        registry.register(ModelInfo(name="pose", path="b.pt", task="pose", base_model="yolov8n-pose.pt", classes=["a"]))
        assert len(registry.list_models(task="detect")) == 1
        assert len(registry.list_models(task="pose")) == 1
        assert len(registry.list_models()) == 2

    def test_rename(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        info = ModelInfo(name="old-name", path="a.pt", task="detect", base_model="yolov8n.pt", classes=["a"])
        registry.register(info)
        assert registry.rename(info.id, "new-name")
        assert registry.get(info.id).name == "new-name"
        # Persists after save/load
        registry.save()
        registry2 = ModelRegistry(tmp_path / "models")
        registry2.load()
        assert registry2.get(info.id).name == "new-name"

    def test_rename_nonexistent(self, tmp_path):
        registry = ModelRegistry(tmp_path / "models")
        assert not registry.rename("no-such-id", "whatever")


class TestModelInfoTrainParams:
    def test_train_params_default_empty(self):
        info = ModelInfo(
            name="m", path="m.pt", task="detect",
            base_model="yolov8n.pt", classes=["cat"],
        )
        assert info.train_params == {}

    def test_train_params_roundtrip(self):
        info = ModelInfo(
            name="m", path="m.pt", task="pose",
            base_model="yolov8n-pose.pt", classes=["person"],
            train_params={"epochs": 100, "lr0": 0.01, "kpt_shape": [17, 3]},
        )
        restored = ModelInfo.from_dict(info.to_dict())
        assert restored.train_params == {"epochs": 100, "lr0": 0.01, "kpt_shape": [17, 3]}

    def test_from_dict_legacy_without_train_params(self):
        # Older registry entries did not include train_params
        legacy = {
            "id": "abc", "name": "m", "path": "m.pt", "task": "detect",
            "base_model": "yolov8n.pt", "classes": ["cat"],
        }
        info = ModelInfo.from_dict(legacy)
        assert info.train_params == {}
