"""Tests for SamAssistController — load/encode state machine + data signals.

Drives the controller with a fake SamAssistant (pure Python, no ultralytics,
no weights). The real SamPrepareWorker QThread is exercised — its run() only
calls the fake assistant, so runs are instant.
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import QCoreApplication

from src.controllers.sam_assist import SamAssistController
from src.engine.sam_assist import BoxPrompt, MaskResult, PointPrompt, SamAssistError


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QCoreApplication.processEvents()
    return predicate()


def _record(signal, into):
    signal.connect(lambda *args: into.append(args))


_POLY = [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]
_BBOX = (0.3, 0.3, 0.2, 0.2)


class _FakeAssistant:
    """Scripted stand-in for engine.sam_assist.SamAssistant."""

    def __init__(self):
        self.loaded = False
        self.encoded: list[Path] = []
        self.released = 0
        self.load_error: str | None = None
        self.encode_error: str | None = None
        self.predict_result: MaskResult | None = MaskResult(polygon=_POLY, bbox=_BBOX)
        self.predict_error: str | None = None
        self.prompts: list = []

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        if self.load_error:
            raise SamAssistError(self.load_error)
        self.loaded = True

    def encode(self, path, size) -> None:
        if self.encode_error:
            raise SamAssistError(self.encode_error)
        self.encoded.append(Path(path))

    def predict(self, prompt):
        if self.predict_error:
            raise SamAssistError(self.predict_error)
        self.prompts.append(prompt)
        return self.predict_result

    def release(self) -> None:
        self.released += 1
        self.loaded = False


def _make(qapp):
    fake = _FakeAssistant()
    ctrl = SamAssistController(assistant=fake)
    return ctrl, fake


class TestActivationStateMachine:
    def test_activate_loads_and_encodes(self, qapp):
        ctrl, fake = _make(qapp)
        loads, enc_started, enc_ready = [], [], []
        _record(ctrl.load_started, loads)
        _record(ctrl.encode_started, enc_started)
        _record(ctrl.encode_ready, enc_ready)

        img = Path("/tmp/img0.png")
        ctrl.activate(img, (100, 80))
        assert ctrl.is_busy
        assert _wait_until(lambda: not ctrl.is_busy)
        assert fake.loaded
        assert fake.encoded == [img]
        assert loads and enc_started == [(img,)] and enc_ready == [(img,)]
        assert ctrl.ready_path == img
        ctrl.shutdown()

    def test_activate_without_image_loads_only(self, qapp):
        ctrl, fake = _make(qapp)
        ctrl.activate(None, None)
        assert _wait_until(lambda: not ctrl.is_busy)
        assert fake.loaded
        assert fake.encoded == []
        assert ctrl.ready_path is None
        ctrl.shutdown()

    def test_reactivate_same_image_is_noop(self, qapp):
        ctrl, fake = _make(qapp)
        img = Path("/tmp/img0.png")
        ctrl.activate(img, (100, 80))
        assert _wait_until(lambda: not ctrl.is_busy)
        ctrl.activate(img, (100, 80))
        assert not ctrl.is_busy
        assert fake.encoded == [img]
        ctrl.shutdown()

    def test_switch_during_encode_coalesces_to_latest(self, qapp):
        ctrl, fake = _make(qapp)

        # Slow the first encode so the pending queue is exercised.
        orig_encode = fake.encode

        def slow_encode(path, size):
            time.sleep(0.05)
            orig_encode(path, size)

        fake.encode = slow_encode
        a, b, c = Path("/tmp/a.png"), Path("/tmp/b.png"), Path("/tmp/c.png")
        ctrl.activate(a, (10, 10))
        ctrl.set_image(b, (10, 10))
        ctrl.set_image(c, (10, 10))  # b is superseded before its worker starts
        assert _wait_until(lambda: not ctrl.is_busy and ctrl.ready_path == c)
        assert fake.encoded == [a, c]
        ctrl.shutdown()

    def test_load_failure_emits_failed_and_status(self, qapp):
        ctrl, fake = _make(qapp)
        fake.load_error = "SAM 模型加载失败：no net"
        failed, status = [], []
        _record(ctrl.load_failed, failed)
        _record(ctrl.status_message, status)
        ctrl.activate(Path("/tmp/a.png"), (10, 10))
        assert _wait_until(lambda: bool(failed))
        assert _wait_until(lambda: not ctrl.is_busy)
        assert "SAM 模型加载失败" in failed[0][0]
        assert any("SAM 模型加载失败" in s[0] for s in status)
        assert ctrl.ready_path is None
        ctrl.shutdown()


class TestPrompts:
    def _ready(self, qapp):
        ctrl, fake = _make(qapp)
        img = Path("/tmp/img0.png")
        ctrl.activate(img, (100, 80))
        assert _wait_until(lambda: not ctrl.is_busy)
        return ctrl, fake

    def test_point_prompt_emits_mask_ready(self, qapp):
        ctrl, fake = self._ready(qapp)
        masks = []
        _record(ctrl.mask_ready, masks)
        ctrl.request_point(0.3, 0.3)
        assert masks == [(_POLY, _BBOX)]
        assert isinstance(fake.prompts[0], PointPrompt)
        ctrl.shutdown()

    def test_box_prompt_emits_mask_ready(self, qapp):
        ctrl, fake = self._ready(qapp)
        masks = []
        _record(ctrl.mask_ready, masks)
        ctrl.request_box(0.1, 0.1, 0.5, 0.5)
        assert masks == [(_POLY, _BBOX)]
        assert isinstance(fake.prompts[0], BoxPrompt)
        ctrl.shutdown()

    def test_prompt_before_ready_is_dropped_with_status(self, qapp):
        ctrl, fake = _make(qapp)
        masks, status = [], []
        _record(ctrl.mask_ready, masks)
        _record(ctrl.status_message, status)
        # Never activated: no predictor call may happen (segment-everything guard).
        ctrl.request_point(0.5, 0.5)
        assert masks == []
        assert fake.prompts == []
        assert any("请稍候" in s[0] for s in status)
        ctrl.shutdown()

    def test_prompt_during_encode_is_dropped(self, qapp):
        ctrl, fake = _make(qapp)
        orig_encode = fake.encode

        def slow_encode(path, size):
            time.sleep(0.05)
            orig_encode(path, size)

        fake.encode = slow_encode
        ctrl.activate(Path("/tmp/a.png"), (10, 10))
        assert ctrl.is_busy
        masks = []
        _record(ctrl.mask_ready, masks)
        ctrl.request_point(0.5, 0.5)
        assert masks == [] and fake.prompts == []
        assert _wait_until(lambda: not ctrl.is_busy)
        ctrl.shutdown()

    def test_empty_mask_emits_status_not_mask_ready(self, qapp):
        ctrl, fake = self._ready(qapp)
        fake.predict_result = None
        masks, status = [], []
        _record(ctrl.mask_ready, masks)
        _record(ctrl.status_message, status)
        ctrl.request_point(0.9, 0.9)
        assert masks == []
        assert any("未能生成有效多边形" in s[0] for s in status)
        ctrl.shutdown()

    def test_predict_error_surfaces_as_status(self, qapp):
        ctrl, fake = self._ready(qapp)
        fake.predict_error = "SAM 推理失败：boom"
        masks, status = [], []
        _record(ctrl.mask_ready, masks)
        _record(ctrl.status_message, status)
        ctrl.request_box(0.1, 0.1, 0.6, 0.6)
        assert masks == []
        assert any("SAM 推理失败" in s[0] for s in status)
        ctrl.shutdown()


class TestShutdown:
    def test_shutdown_releases_and_is_idempotent(self, qapp):
        ctrl, fake = _make(qapp)
        ctrl.activate(Path("/tmp/a.png"), (10, 10))
        ctrl.shutdown()
        assert not ctrl.is_busy
        assert fake.released >= 1
        assert ctrl.ready_path is None
        ctrl.shutdown()  # safe to repeat
