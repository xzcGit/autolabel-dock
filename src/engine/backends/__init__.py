"""Model runtime backend registry."""
from __future__ import annotations

from src.engine.backends.base import (
    BackendError,
    BackendProbe,
    BackendUnavailableError,
    DEFAULT_BACKEND_ID,
    DEFAULT_BACKEND_RUNTIME,
    ModelBackend,
    PredictorProtocol,
    TrainerProtocol,
    UnknownBackendError,
)


def get_backend(*args, **kwargs):
    from src.engine.backends.registry import get_backend as _get_backend

    return _get_backend(*args, **kwargs)


def list_backends(*args, **kwargs):
    from src.engine.backends.registry import list_backends as _list_backends

    return _list_backends(*args, **kwargs)


def register_backend(*args, **kwargs):
    from src.engine.backends.registry import register_backend as _register_backend

    return _register_backend(*args, **kwargs)


def registered_backend_ids(*args, **kwargs):
    from src.engine.backends.registry import registered_backend_ids as _registered_backend_ids

    return _registered_backend_ids(*args, **kwargs)


def __getattr__(name: str):
    if name == "UltralyticsBackend":
        from src.engine.backends.ultralytics import UltralyticsBackend

        return UltralyticsBackend
    raise AttributeError(name)

__all__ = [
    "BackendError",
    "BackendProbe",
    "BackendUnavailableError",
    "DEFAULT_BACKEND_ID",
    "DEFAULT_BACKEND_RUNTIME",
    "ModelBackend",
    "PredictorProtocol",
    "TrainerProtocol",
    "UnknownBackendError",
    "UltralyticsBackend",
    "get_backend",
    "list_backends",
    "register_backend",
    "registered_backend_ids",
]
