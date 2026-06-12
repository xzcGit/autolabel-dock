"""Tests for TemplateRegistry."""
from pathlib import Path

import pytest

from src.core.train_templates import TrainTemplate, TemplateRegistry


class TestTrainTemplate:
    def test_dataclass_defaults(self):
        t = TrainTemplate(name="my", task="detect", params={"epochs": 50}, created_at="2026-05-08T10:00:00")
        assert t.name == "my"
        assert t.task == "detect"
        assert t.params == {"epochs": 50}
        assert t.created_at == "2026-05-08T10:00:00"
        assert t.builtin is False

    def test_to_dict_round_trip(self):
        t = TrainTemplate(name="my", task="detect", params={"epochs": 50}, created_at="2026-05-08T10:00:00")
        d = t.to_dict()
        restored = TrainTemplate.from_dict(d)
        assert restored == t


class TestTemplateRegistryInMemory:
    def test_empty_registry_lists_no_user_templates(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        assert reg.list(task="detect") == [] or all(t.builtin for t in reg.list(task="detect"))

    def test_upsert_and_get(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        t = TrainTemplate(name="my", task="detect", params={"epochs": 50}, created_at="2026-05-08T10:00:00")
        reg.upsert(t)
        got = reg.get("my", "detect")
        assert got is not None
        assert got.params == {"epochs": 50}

    def test_upsert_same_name_task_overwrites(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="my", task="detect", params={"epochs": 50}, created_at="t1"))
        reg.upsert(TrainTemplate(name="my", task="detect", params={"epochs": 999}, created_at="t2"))
        got = reg.get("my", "detect")
        assert got.params == {"epochs": 999}
        assert got.created_at == "t2"

    def test_same_name_different_task_coexist(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="实验", task="detect", params={"epochs": 1}, created_at="t1"))
        reg.upsert(TrainTemplate(name="实验", task="classify", params={"epochs": 2}, created_at="t2"))
        assert reg.get("实验", "detect").params == {"epochs": 1}
        assert reg.get("实验", "classify").params == {"epochs": 2}

    def test_remove_returns_true_when_found(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="my", task="detect", params={}, created_at="t"))
        assert reg.remove("my", "detect") is True
        assert reg.get("my", "detect") is None

    def test_remove_returns_false_when_missing(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        assert reg.remove("nope", "detect") is False

    def test_list_filters_by_task(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="d1", task="detect", params={}, created_at="t"))
        reg.upsert(TrainTemplate(name="c1", task="classify", params={}, created_at="t"))
        detect_names = [t.name for t in reg.list(task="detect") if not t.builtin]
        classify_names = [t.name for t in reg.list(task="classify") if not t.builtin]
        assert detect_names == ["d1"]
        assert classify_names == ["c1"]


class TestTemplateRegistryBuiltin:
    def test_list_includes_builtin_default_first(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        reg.upsert(TrainTemplate(name="z_user", task="detect", params={}, created_at="t"))
        items = reg.list(task="detect")
        assert items[0].name == "默认"
        assert items[0].builtin is True

    def test_builtin_default_visible_under_every_task(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        for task in ("detect", "classify", "pose"):
            names = [t.name for t in reg.list(task=task)]
            assert "默认" in names

    def test_remove_builtin_default_returns_false(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        # Built-in is not in self._templates, so remove returns False naturally
        assert reg.remove("默认", "detect") is False

    def test_upsert_builtin_template_raises(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        bad = TrainTemplate(name="x", task="detect", params={}, created_at="t", builtin=True)
        with pytest.raises(ValueError):
            reg.upsert(bad)

    def test_get_default_returns_builtin_regardless_of_task(self, tmp_path):
        reg = TemplateRegistry(tmp_path / "tpl.json")
        for task in ("detect", "classify", "pose"):
            t = reg.get("默认", task)
            assert t is not None
            assert t.builtin is True


class TestTemplateRegistryPersistence:
    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "tpl.json"
        reg = TemplateRegistry(path)
        reg.upsert(TrainTemplate(name="A", task="detect", params={"epochs": 50}, created_at="t1"))
        reg.upsert(TrainTemplate(name="B", task="classify", params={"erasing": 0.3}, created_at="t2"))
        reg.save()

        reg2 = TemplateRegistry(path)
        reg2.load()
        a = reg2.get("A", "detect")
        b = reg2.get("B", "classify")
        assert a is not None and a.params == {"epochs": 50}
        assert b is not None and b.params == {"erasing": 0.3}

    def test_save_does_not_persist_builtin(self, tmp_path):
        path = tmp_path / "tpl.json"
        reg = TemplateRegistry(path)
        reg.upsert(TrainTemplate(name="A", task="detect", params={}, created_at="t1"))
        reg.save()
        raw = path.read_text(encoding="utf-8")
        assert "默认" not in raw

    def test_load_missing_file_yields_empty_registry(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        reg = TemplateRegistry(path)
        reg.load()
        assert [t for t in reg.list() if not t.builtin] == []

    def test_load_corrupt_json_yields_empty_registry_with_warning(self, tmp_path, caplog):
        path = tmp_path / "tpl.json"
        path.write_text("{ not valid json", encoding="utf-8")
        reg = TemplateRegistry(path)
        with caplog.at_level("WARNING"):
            reg.load()
        assert [t for t in reg.list() if not t.builtin] == []
        assert any("template" in rec.getMessage().lower() for rec in caplog.records)

    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "subdir" / "tpl.json"
        reg = TemplateRegistry(path)
        reg.upsert(TrainTemplate(name="A", task="detect", params={}, created_at="t"))
        reg.save()
        assert path.exists()


class TestExtractTaskParams:
    def _base_config(self, task: str, **overrides):
        from src.engine.trainer import TrainConfig
        kwargs = dict(data_yaml="/d.yaml", model="m.pt", task=task)
        kwargs.update(overrides)
        return TrainConfig(**kwargs)

    def test_detect_keeps_common_and_detect_aug(self):
        from src.core.train_templates import extract_task_params

        cfg = self._base_config(
            "detect",
            epochs=50, batch=8, lr0=0.005,
            include_detect_params=True, mosaic=0.7,
            erasing=0.9, pose=15.0,
        )
        params = extract_task_params(cfg)
        # Common params kept
        assert params["epochs"] == 50
        assert params["batch"] == 8
        assert params["lr0"] == 0.005
        assert params["model"] == "m.pt"
        # Detect-group params kept
        assert params["mosaic"] == 0.7
        assert params["include_detect_params"] is True
        # Classify- and pose-only fields dropped
        assert "erasing" not in params
        assert "include_classify_params" not in params
        assert "pose" not in params
        assert "kobj" not in params
        assert "kpt_shape" not in params
        assert "include_pose_params" not in params
        # Task itself not stored in params (lives at template level)
        assert "task" not in params
        # Runtime-only fields not stored
        assert "data_yaml" not in params
        assert "project" not in params
        assert "name" not in params
        assert "resume" not in params

    def test_classify_keeps_classify_only_params(self):
        from src.core.train_templates import extract_task_params

        cfg = self._base_config(
            "classify",
            epochs=20, erasing=0.6, auto_augment="augmix",
            dropout=0.25,
            include_classify_params=True,
            mosaic=0.5, include_detect_params=True,
            pose=10.0,
        )
        params = extract_task_params(cfg)
        assert params["erasing"] == 0.6
        assert params["dropout"] == 0.25
        assert params["include_classify_params"] is True
        # Detect-group dropped for classify
        assert "mosaic" not in params
        assert "include_detect_params" not in params
        # Pose-group dropped
        assert "pose" not in params

    def test_pose_keeps_detect_aug_and_pose_specific(self):
        from src.core.train_templates import extract_task_params

        cfg = self._base_config(
            "pose",
            epochs=80, mosaic=0.8, include_detect_params=True,
            pose=15.0, kobj=2.0, kpt_shape=[17, 3],
            include_pose_params=True,
            erasing=0.5,
        )
        params = extract_task_params(cfg)
        # Detect-group available for pose
        assert params["mosaic"] == 0.8
        assert params["include_detect_params"] is True
        # Pose-specific kept
        assert params["pose"] == 15.0
        assert params["kobj"] == 2.0
        assert params["kpt_shape"] == [17, 3]
        assert params["include_pose_params"] is True
        # Classify-only dropped
        assert "erasing" not in params
        assert "include_classify_params" not in params

    def test_unknown_task_falls_back_to_common_only(self):
        from src.core.train_templates import extract_task_params

        cfg = self._base_config("segment", mosaic=0.5, erasing=0.5, pose=10.0)
        params = extract_task_params(cfg)
        assert params["epochs"] == cfg.epochs
        assert "mosaic" not in params
        assert "erasing" not in params
        assert "pose" not in params
