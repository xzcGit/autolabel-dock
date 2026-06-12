"""In-process registry for model runtime backends."""
from __future__ import annotations

import logging
from collections.abc import Iterable

from src.engine.backends.base import (
    DEFAULT_BACKEND_ID,
    ModelBackend,
    UnknownBackendError,
)
from src.engine.backends.ultralytics import UltralyticsBackend

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, ModelBackend] = {}


def register_backend(backend: ModelBackend) -> None:
    """Register or replace a model backend."""
    _BACKENDS[backend.backend_id] = backend


def get_backend(backend_id: str = DEFAULT_BACKEND_ID) -> ModelBackend:
    """Return a registered backend by id."""
    try:
        return _BACKENDS[backend_id]
    except KeyError as exc:
        raise UnknownBackendError(f"未注册的模型后端: {backend_id}") from exc


def list_backends() -> list[ModelBackend]:
    """Return all registered backends."""
    return list(_BACKENDS.values())


def registered_backend_ids() -> Iterable[str]:
    """Return backend ids for diagnostics/tests."""
    return tuple(_BACKENDS.keys())


def _register_builtin_backends() -> None:
    register_backend(UltralyticsBackend())
    # Optional, lazily-loaded backend. Registration is pure-stdlib (the module
    # top level imports no heavy deps). Wrapped in try/except so that any
    # failure here can never take down the Ultralytics backend or app startup.
    try:
        from src.engine.backends.locateanything import LocateAnythingBackend

        register_backend(LocateAnythingBackend())
    except Exception:  # noqa: BLE001 - never let an optional backend break startup
        logger.warning("Failed to register LocateAnything backend", exc_info=True)


_register_builtin_backends()
