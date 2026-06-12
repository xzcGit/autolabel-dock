import builtins

import pytest

from src.engine.backends import get_backend, registered_backend_ids
from src.engine.backends.base import (
    BackendUnavailableError,
    DEFAULT_BACKEND_ID,
    UnknownBackendError,
)
from src.engine.backends.ultralytics import UltralyticsBackend


def test_builtin_ultralytics_backend_registered():
    assert DEFAULT_BACKEND_ID in registered_backend_ids()
    backend = get_backend(DEFAULT_BACKEND_ID)
    assert isinstance(backend, UltralyticsBackend)


def test_unknown_backend_raises():
    try:
        get_backend("missing-backend")
    except UnknownBackendError as exc:
        assert "missing-backend" in str(exc)
    else:
        raise AssertionError("expected UnknownBackendError")


def test_ultralytics_backend_infers_model_format():
    backend = UltralyticsBackend(yolo_cls=object)
    assert backend.infer_model_format("/tmp/model.pt") == "pt"
    assert backend.infer_model_format("/tmp/model.onnx") == "onnx"
    assert backend.infer_model_format("/tmp/model") == "unknown"


def test_ultralytics_backend_reports_missing_dependency_at_runtime(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise ImportError("missing ultralytics")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(BackendUnavailableError):
        UltralyticsBackend().create_trainer()


def test_ultralytics_backend_probe_warns_below_yolo_api_baseline(monkeypatch):
    """Probe should warn when ultralytics predates the YOLO API baseline."""
    def fake_version(package):
        if package == "ultralytics":
            return "7.9.0"
        raise Exception("unexpected package")

    monkeypatch.setattr("src.engine.backends.ultralytics.version", fake_version)

    backend = UltralyticsBackend(yolo_cls=object)
    probe = backend.probe()

    assert probe.available is True
    assert probe.version == "7.9.0"
    assert "低于支持版本 8.0" in probe.message


def test_ultralytics_backend_probe_accepts_future_versions(monkeypatch):
    """Probe should not block newer ultralytics versions that keep the YOLO API."""
    def fake_version(package):
        if package == "ultralytics":
            return "9.0.0"
        raise Exception("unexpected package")

    monkeypatch.setattr("src.engine.backends.ultralytics.version", fake_version)

    backend = UltralyticsBackend(yolo_cls=object)
    probe = backend.probe()

    assert probe.available is True
    assert probe.version == "9.0.0"
    assert probe.message == ""


def test_ultralytics_backend_probe_accepts_compatible_version(monkeypatch):
    """Probe should not warn for versions at or above the YOLO API baseline."""
    def fake_version(package):
        if package == "ultralytics":
            return "8.0.0"
        raise Exception("unexpected package")

    monkeypatch.setattr("src.engine.backends.ultralytics.version", fake_version)

    backend = UltralyticsBackend(yolo_cls=object)
    probe = backend.probe()

    assert probe.available is True
    assert probe.version == "8.0.0"
    assert probe.message == ""  # No warning
