"""Tests for global app configuration."""
from pathlib import Path

from src.core.config import AppConfig


class TestAppConfig:
    def test_default_values(self):
        cfg = AppConfig()
        assert cfg.recent_projects == []
        assert cfg.theme == "dark"
        assert cfg.auto_save is True
        assert cfg.default_conf_threshold == 0.5
        assert cfg.script_tools == {}

    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig(recent_projects=["/path/a", "/path/b"])
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.recent_projects == ["/path/a", "/path/b"]

    def test_load_missing_returns_default(self, tmp_path):
        cfg = AppConfig.load(tmp_path / "missing.json")
        assert cfg.recent_projects == []

    def test_add_recent_project(self):
        cfg = AppConfig()
        cfg.add_recent_project("/proj/a")
        cfg.add_recent_project("/proj/b")
        cfg.add_recent_project("/proj/a")  # move to front
        assert cfg.recent_projects[0] == "/proj/a"
        assert cfg.recent_projects[1] == "/proj/b"
        assert len(cfg.recent_projects) == 2

    def test_recent_projects_max_10(self):
        cfg = AppConfig()
        for i in range(15):
            cfg.add_recent_project(f"/proj/{i}")
        assert len(cfg.recent_projects) == 10
        assert cfg.recent_projects[0] == "/proj/14"

    def test_script_tools_roundtrip(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig(script_tools={"crop": "print('crop')\n"})
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.script_tools == {"crop": "print('crop')\n"}

    def test_classify_keys_default(self):
        cfg = AppConfig()
        assert cfg.classify_grid_density == 96
        assert cfg.classify_grid_sort == "filename"
        assert cfg.classify_preview_width == 320
        assert cfg.classify_preview_visible is True

    def test_classify_keys_roundtrip(self, tmp_path):
        config_path = tmp_path / "config.json"
        cfg = AppConfig(
            classify_grid_density=128,
            classify_grid_sort="class",
            classify_preview_width=400,
            classify_preview_visible=False,
        )
        cfg.save(config_path)
        loaded = AppConfig.load(config_path)
        assert loaded.classify_grid_density == 128
        assert loaded.classify_grid_sort == "class"
        assert loaded.classify_preview_width == 400
        assert loaded.classify_preview_visible is False

    def test_classify_keys_legacy_compat(self, tmp_path):
        """旧 config.json 没这 4 个键 → 应使用默认值"""
        import json
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"theme": "dark", "auto_save": True}))
        cfg = AppConfig.load(config_path)
        assert cfg.classify_grid_density == 96
        assert cfg.classify_grid_sort == "filename"
        assert cfg.classify_preview_width == 320
        assert cfg.classify_preview_visible is True


class TestAnnotationPanelPersistence:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.annotation_panel_splitter_sizes == []
        assert cfg.annotation_panel_collapsed == {}

    def test_round_trip(self):
        cfg = AppConfig()
        cfg.annotation_panel_splitter_sizes = [120, 220, 90, 80, 160]
        cfg.annotation_panel_collapsed = {"属性": True, "Tag": False}

        restored = AppConfig.from_dict(cfg.to_dict())

        assert restored.annotation_panel_splitter_sizes == [120, 220, 90, 80, 160]
        assert restored.annotation_panel_collapsed == {"属性": True, "Tag": False}

    def test_backward_compat_missing_keys(self):
        # Older config.json without the new keys must load with defaults.
        legacy = {"recent_projects": ["/x"], "theme": "dark"}
        cfg = AppConfig.from_dict(legacy)
        assert cfg.annotation_panel_splitter_sizes == []
        assert cfg.annotation_panel_collapsed == {}
        assert cfg.recent_projects == ["/x"]

    def test_invalid_types_fall_back_to_defaults(self):
        # Defensive: corrupted entries should not crash from_dict.
        bad = {
            "annotation_panel_splitter_sizes": "not a list",
            "annotation_panel_collapsed": ["not", "a", "dict"],
        }
        cfg = AppConfig.from_dict(bad)
        assert cfg.annotation_panel_splitter_sizes == []
        assert cfg.annotation_panel_collapsed == {}
