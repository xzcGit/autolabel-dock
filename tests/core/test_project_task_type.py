"""Tests for ProjectConfig task_type field."""
import json
from pathlib import Path

import pytest

from src.core.project import ProjectConfig, ProjectManager


def test_project_config_default_task_type():
    """Default task_type should be 'detect'."""
    config = ProjectConfig(
        name="test",
        image_dir="images",
        label_dir="labels",
        classes=["cat", "dog"],
    )
    assert config.task_type == "detect"


def test_project_config_to_dict_includes_task_type():
    """Serialization should include task_type."""
    config = ProjectConfig(
        name="test",
        image_dir="images",
        label_dir="labels",
        classes=["cat", "dog"],
        task_type="classify",
    )
    data = config.to_dict()
    assert "task_type" in data
    assert data["task_type"] == "classify"


def test_project_config_from_dict_backward_compat():
    """Old project.json without task_type should default to 'detect'."""
    data = {
        "name": "old_project",
        "image_dir": "images",
        "label_dir": "labels",
        "classes": ["cat", "dog"],
        "created_at": "2024-01-01T00:00:00",
        "version": "1.0",
    }
    config = ProjectConfig.from_dict(data)
    assert config.task_type == "detect"


def test_project_config_from_dict_with_task_type():
    """New project.json with task_type should preserve it."""
    data = {
        "name": "new_project",
        "image_dir": "images",
        "label_dir": "labels",
        "classes": ["cat", "dog"],
        "task_type": "classify",
        "created_at": "2024-01-01T00:00:00",
        "version": "1.0",
    }
    config = ProjectConfig.from_dict(data)
    assert config.task_type == "classify"


def test_create_classify_project(tmp_path):
    """ProjectManager.create should support task_type parameter."""
    project_dir = tmp_path / "classify_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Classify Test",
        classes=["cat", "dog", "bird"],
        task_type="classify",
    )
    assert pm.config.task_type == "classify"

    # Verify persisted to disk
    config_path = project_dir / "project.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["task_type"] == "classify"
