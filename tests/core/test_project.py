"""Tests for project configuration and management."""
import json
from pathlib import Path

from src.core.project import ProjectConfig, ProjectManager


class TestProjectConfig:
    def test_create_minimal(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["person", "car"],
        )
        assert cfg.name == "test"
        assert cfg.classes == ["person", "car"]
        assert cfg.class_colors == {}
        assert cfg.keypoint_templates == {}
        assert cfg.version == "1.0"

    def test_to_dict_roundtrip(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["a", "b"],
            class_colors={"a": "#ff0000"},
            keypoint_templates={
                "pose": {
                    "labels": ["nose", "eye"],
                    "skeleton": [[0, 1]],
                }
            },
        )
        d = cfg.to_dict()
        restored = ProjectConfig.from_dict(d)
        assert restored.name == cfg.name
        assert restored.classes == cfg.classes
        assert restored.class_colors == {"a": "#ff0000"}
        assert restored.keypoint_templates["pose"]["labels"] == ["nose", "eye"]

    def test_get_class_color_assigned(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["a", "b", "c"],
            class_colors={"a": "#ff0000"},
        )
        assert cfg.get_class_color("a") == "#ff0000"
        # b and c should get auto-assigned from palette
        color_b = cfg.get_class_color("b")
        assert color_b.startswith("#")

    def test_get_class_id(self):
        cfg = ProjectConfig(
            name="test",
            image_dir="images",
            label_dir="labels",
            classes=["person", "car", "dog"],
        )
        assert cfg.get_class_id("person") == 0
        assert cfg.get_class_id("car") == 1
        assert cfg.get_class_id("dog") == 2
        assert cfg.get_class_id("unknown") == -1

    def test_auto_register_classes_default_true(self):
        cfg = ProjectConfig(
            name="t", image_dir="images", label_dir="labels", classes=[],
        )
        assert cfg.auto_register_classes is True

    def test_auto_register_classes_roundtrip(self):
        cfg = ProjectConfig(
            name="t", image_dir="images", label_dir="labels",
            classes=[], auto_register_classes=False,
        )
        d = cfg.to_dict()
        assert d["auto_register_classes"] is False
        restored = ProjectConfig.from_dict(d)
        assert restored.auto_register_classes is False

    def test_auto_register_classes_legacy_json_defaults_true(self):
        legacy = {
            "name": "t", "image_dir": "images", "label_dir": "labels",
            "classes": [], "class_colors": {}, "keypoint_templates": {},
            "default_model": "", "auto_label_conf": 0.5, "auto_label_iou": 0.45,
            "created_at": "2026-01-01", "version": "1.0",
        }
        cfg = ProjectConfig.from_dict(legacy)
        assert cfg.auto_register_classes is True


class TestProjectManager:
    def test_create_project(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "my_project",
            name="my_project",
            image_dir="images",
            classes=["person", "car"],
        )
        assert (tmp_path / "my_project" / "project.json").exists()
        assert (tmp_path / "my_project" / "images").is_dir()
        assert (tmp_path / "my_project" / "labels").is_dir()
        assert pm.config.name == "my_project"

    def test_open_project(self, tmp_path):
        # Create first
        ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        # Open
        pm = ProjectManager.open(tmp_path / "proj")
        assert pm.config.name == "proj"
        assert pm.config.classes == ["a"]

    def test_open_nonexistent_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            ProjectManager.open(tmp_path / "nope")

    def test_save_config(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        pm.config.classes.append("b")
        pm.save()
        # Re-open and verify
        pm2 = ProjectManager.open(tmp_path / "proj")
        assert "b" in pm2.config.classes

    def test_list_images(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        (img_dir / "a.jpg").write_bytes(b"fake")
        (img_dir / "b.png").write_bytes(b"fake")
        (img_dir / "c.txt").write_bytes(b"fake")  # not an image
        images = pm.list_images()
        names = [p.name for p in images]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_label_path_for_image(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img = tmp_path / "proj" / "images" / "photo.jpg"
        label_path = pm.label_path_for(img)
        assert label_path == tmp_path / "proj" / "labels" / "photo.json"

    def test_add_class(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        pm.add_class("b")
        assert "b" in pm.config.classes
        # Duplicate should not add
        pm.add_class("b")
        assert pm.config.classes.count("b") == 1

    def test_remove_class(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a", "b"],
        )
        pm.remove_class("a")
        assert "a" not in pm.config.classes

    def test_delete_images_removes_image_and_label(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img = tmp_path / "proj" / "images" / "a.jpg"
        img.write_bytes(b"fake")
        label = pm.label_path_for(img)
        label.write_text("{}", encoding="utf-8")

        img_n, lbl_n = pm.delete_images([img])

        assert (img_n, lbl_n) == (1, 1)
        assert not img.exists()
        assert not label.exists()

    def test_delete_images_when_no_label(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img = tmp_path / "proj" / "images" / "a.jpg"
        img.write_bytes(b"fake")

        img_n, lbl_n = pm.delete_images([img])

        assert (img_n, lbl_n) == (1, 0)
        assert not img.exists()

    def test_delete_images_missing_image_is_silent(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        ghost = tmp_path / "proj" / "images" / "missing.jpg"

        img_n, lbl_n = pm.delete_images([ghost])

        assert (img_n, lbl_n) == (0, 0)

    def test_delete_images_handles_multiple_mixed(self, tmp_path):
        pm = ProjectManager.create(
            project_dir=tmp_path / "proj",
            name="proj",
            image_dir="images",
            classes=["a"],
        )
        img_dir = tmp_path / "proj" / "images"
        a = img_dir / "a.jpg"
        b = img_dir / "b.jpg"
        c = img_dir / "c.jpg"
        for p in (a, b, c):
            p.write_bytes(b"fake")
        pm.label_path_for(a).write_text("{}", encoding="utf-8")
        pm.label_path_for(c).write_text("{}", encoding="utf-8")

        img_n, lbl_n = pm.delete_images([a, b, c])

        assert (img_n, lbl_n) == (3, 2)
        assert not a.exists() and not b.exists() and not c.exists()
