"""Tests for LocateAnythingController enable/disable flow.

Uses a fake backend + a fake load worker so no torch/transformers/model load
ever happens. Verifies probe→preflight gating, predictor installation, query
forwarding, and unload-on-disable.
"""
from __future__ import annotations

import pytest
from PyQt5.QtCore import QObject, pyqtSignal

from src.engine.backends.base import BackendProbe


# ── Fakes ─────────────────────────────────────────────────────────────────


class _FakePredictor:
    def __init__(self):
        self.released = False
        self.query = None

    def set_query(self, prompt, target):
        self.query = (prompt, target)

    def release(self):
        self.released = True


class _FakeBackend:
    backend_id = "locateanything"
    display_name = "LocateAnything"

    def __init__(self, probe_available=True, preflight_ok=True):
        self._probe_available = probe_available
        self._preflight_ok = preflight_ok
        self.predictor = _FakePredictor()

    def probe(self):
        return BackendProbe(
            backend_id=self.backend_id,
            display_name=self.display_name,
            available=self._probe_available,
            message="" if self._probe_available else "缺少依赖",
        )

    def preflight(self):
        return {
            "ok": self._preflight_ok,
            "message": "ok" if self._preflight_ok else "显存不足",
            "total_gb": 8.0,
            "free_gb": 6.0,
        }

    def load_runtime(self, progress_cb=None):
        if progress_cb:
            progress_cb("loading")
        return self.predictor


class _FakeModelCtrl:
    def __init__(self):
        self._predictor = None
        self.unload_calls = 0

    @property
    def predictor(self):
        return self._predictor

    def unload(self):
        self.unload_calls += 1
        if self._predictor is not None and hasattr(self._predictor, "release"):
            self._predictor.release()
        self._predictor = None

    def set_predictor(self, predictor):
        self._predictor = predictor


class _SyncLoadWorker(QObject):
    """Stand-in for the QThread worker that runs synchronously on start()."""

    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend

    def start(self):
        try:
            predictor = self._backend.load_runtime(progress_cb=self.progress.emit)
            self.loaded.emit(predictor)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))
        finally:
            self.finished.emit()


def _make_controller(monkeypatch, backend):
    import src.controllers.locateanything as la_mod

    monkeypatch.setattr(la_mod, "get_backend", lambda _id: backend)
    monkeypatch.setattr(la_mod, "LocateAnythingLoadWorker", _SyncLoadWorker)

    model_ctrl = _FakeModelCtrl()
    ctrl = la_mod.LocateAnythingController(model_ctrl)
    return ctrl, model_ctrl


# ── Tests ─────────────────────────────────────────────────────────────────


def test_enable_success_installs_predictor(qapp, monkeypatch):
    backend = _FakeBackend()
    ctrl, model_ctrl = _make_controller(monkeypatch, backend)

    enabled_events = []
    ctrl.enabled.connect(lambda: enabled_events.append(True))

    ctrl.begin_enable()

    assert enabled_events == [True]
    assert ctrl.is_active is True
    assert model_ctrl.predictor is backend.predictor
    # YOLO unload happened before LA load.
    assert model_ctrl.unload_calls == 1


def test_enable_blocked_when_probe_unavailable(qapp, monkeypatch):
    backend = _FakeBackend(probe_available=False)
    ctrl, model_ctrl = _make_controller(monkeypatch, backend)

    blocked = []
    ctrl.preflight_blocked.connect(blocked.append)

    ctrl.begin_enable()

    assert blocked and "依赖" in blocked[0]
    assert ctrl.is_active is False
    assert model_ctrl.predictor is None
    # Unload not called — we never got past probe.
    assert model_ctrl.unload_calls == 0


def test_enable_blocked_when_preflight_fails(qapp, monkeypatch):
    backend = _FakeBackend(preflight_ok=False)
    ctrl, model_ctrl = _make_controller(monkeypatch, backend)

    blocked = []
    ctrl.preflight_blocked.connect(blocked.append)

    ctrl.begin_enable()

    assert blocked and "显存" in blocked[0]
    assert ctrl.is_active is False
    # Unload IS called before preflight (to free VRAM for the check).
    assert model_ctrl.unload_calls == 1


def test_set_query_forwards_to_predictor(qapp, monkeypatch):
    backend = _FakeBackend()
    ctrl, model_ctrl = _make_controller(monkeypatch, backend)
    ctrl.begin_enable()

    ctrl.set_query("a dog", "dog")
    assert backend.predictor.query == ("a dog", "dog")


def test_disable_unloads_and_emits(qapp, monkeypatch):
    backend = _FakeBackend()
    ctrl, model_ctrl = _make_controller(monkeypatch, backend)
    ctrl.begin_enable()

    disabled = []
    ctrl.disabled.connect(lambda: disabled.append(True))

    ctrl.disable()

    assert disabled == [True]
    assert ctrl.is_active is False
    assert model_ctrl.predictor is None
    assert backend.predictor.released is True
