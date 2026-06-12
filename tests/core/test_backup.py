"""Tests for backup manager."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _create_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for testing."""
    proj = tmp_path / "test_project"
    proj.mkdir()
    labels = proj / "labels"
    labels.mkdir()

    # project.json
    config = {"name": "test", "image_dir": "images", "label_dir": "labels",
              "classes": ["cat", "dog"], "version": "1.0"}
    (proj / "project.json").write_text(json.dumps(config), encoding="utf-8")

    # A few label files
    for i in range(3):
        data = {"image_path": f"img{i}.jpg", "image_size": [640, 480], "annotations": []}
        (labels / f"img{i}.json").write_text(json.dumps(data), encoding="utf-8")

    return proj


class TestBackupManager:
    def test_create_backup(self, tmp_path):
        from src.core.backup import BackupManager

        proj = _create_project(tmp_path)
        mgr = BackupManager(proj)
        result = mgr.create_backup()
        assert result is not None
        assert result.exists()
        assert (result / "project.json").exists()
        assert (result / "labels").exists()
        assert len(list((result / "labels").glob("*.json"))) == 3

    def test_list_backups(self, tmp_path):
        from src.core.backup import BackupManager

        proj = _create_project(tmp_path)
        mgr = BackupManager(proj)
        mgr.create_backup()
        backups = mgr.list_backups()
        assert len(backups) == 1
        assert backups[0]["label_count"] == 3

    def test_restore_backup(self, tmp_path):
        from src.core.backup import BackupManager

        proj = _create_project(tmp_path)
        mgr = BackupManager(proj)

        # Create backup
        mgr.create_backup()
        backup_name = mgr.list_backups()[0]["name"]

        # Modify project: delete a label and change config
        (proj / "labels" / "img0.json").unlink()
        config = json.loads((proj / "project.json").read_text())
        config["classes"] = ["bird"]
        (proj / "project.json").write_text(json.dumps(config))

        # Verify modification
        assert not (proj / "labels" / "img0.json").exists()

        # Restore
        result = mgr.restore_backup(backup_name)
        assert result is True
        assert (proj / "labels" / "img0.json").exists()
        restored_config = json.loads((proj / "project.json").read_text())
        assert restored_config["classes"] == ["cat", "dog"]

    def test_restore_nonexistent_returns_false(self, tmp_path):
        from src.core.backup import BackupManager

        proj = _create_project(tmp_path)
        mgr = BackupManager(proj)
        assert mgr.restore_backup("nonexistent") is False

    def test_prune_old_backups(self, tmp_path):
        from src.core.backup import BackupManager
        import time

        proj = _create_project(tmp_path)
        mgr = BackupManager(proj, max_backups=3)

        for _ in range(5):
            mgr.create_backup()
            time.sleep(0.01)  # ensure unique timestamps

        backups = mgr.list_backups()
        assert len(backups) <= 3

    def test_no_project_json_returns_none(self, tmp_path):
        from src.core.backup import BackupManager

        empty = tmp_path / "empty"
        empty.mkdir()
        mgr = BackupManager(empty)
        assert mgr.create_backup() is None

    def test_list_backups_empty(self, tmp_path):
        from src.core.backup import BackupManager

        proj = _create_project(tmp_path)
        mgr = BackupManager(proj)
        assert mgr.list_backups() == []
