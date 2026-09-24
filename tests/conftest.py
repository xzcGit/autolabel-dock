"""Shared test fixtures."""
import json
import os
import shutil
from pathlib import Path

# Run project-open label scans synchronously in tests: the async
# LabelScanWorker finishes non-deterministically under qapp.processEvents()
# and would make view assertions racy. The production path stays threaded.
os.environ.setdefault("AUTOLABEL_SYNC_SCAN", "1")

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


@pytest.fixture
def make_video(tmp_path):
    """Factory: synthesize a tiny mp4 for video-frame-import tests.

    Skips the test when cv2 or a usable mp4 codec is unavailable. Imports
    cv2 through ``src.core.video_frames._lazy_cv2`` so opencv's Qt
    plugin-path hijack is stripped even when this fixture runs before the
    session ``qapp`` exists.
    """
    def _make(name="clip.mp4", frames=30, fps=10.0, size=(64, 48), directory=None):
        try:
            from src.core.video_frames import _lazy_cv2
            cv2 = _lazy_cv2()
            import numpy as np
        except ImportError:
            pytest.skip("cv2/numpy unavailable")
        out_dir = Path(directory) if directory else tmp_path
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
        )
        if not writer.isOpened():
            pytest.skip("cv2 VideoWriter cannot encode mp4v on this machine")
        for i in range(frames):
            frame = np.full((size[1], size[0], 3), (i * 8) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _make
