"""Shared test fixtures."""
import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory structure."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_project_config():
    """Return a minimal project config dict."""
    return {
        "name": "test_project",
        "image_dir": "images",
        "label_dir": "labels",
        "classes": ["person", "car", "dog"],
        "class_colors": {},
        "keypoint_templates": {},
        "default_model": "",
        "auto_label_conf": 0.5,
        "auto_label_iou": 0.45,
        "created_at": "2026-03-23T10:00:00",
        "version": "1.0",
    }


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for the entire test session."""
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
