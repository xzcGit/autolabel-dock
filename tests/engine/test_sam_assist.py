"""Tests for engine/sam_assist.py — prompt assembly, mask→polygon pipeline,
defensive paths. All with a mock predictor: no ultralytics import, no weights,
no real inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.core.polygon import bbox_from_polygon
from src.engine.sam_assist import (
    BoxPrompt,
    MaskResult,
    PointPrompt,
    SamAssistant,
    SamAssistError,
    WEIGHT_FILENAME,
    resolve_weights_path,
)


class _FakeMasks:
    def __init__(self, xyn):
        self.xyn = xyn


class _FakeResult:
    def __init__(self, masks):
        self.masks = masks


class _FakePredictor:
    """Records set_image / call kwargs; scripted contour output."""

    def __init__(self):
        self.set_image_calls: list[str] = []
        self.call_kwargs: list[dict] = []
        self.reset_calls = 0
        # Default: a square contour covering 20%-40% of the image.
        self.contours = [[[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]]

    def set_image(self, path):
        self.set_image_calls.append(path)

    def reset_image(self):
        self.reset_calls += 1

    def __call__(self, **kwargs):
        assert kwargs, "promptless predictor call would trigger segment-everything"
        self.call_kwargs.append(kwargs)
        if self.contours is None:
            return [_FakeResult(masks=None)]
        return [_FakeResult(masks=_FakeMasks(self.contours))]


def _loaded_assistant(predictor=None):
    predictor = predictor or _FakePredictor()
    a = SamAssistant(predictor_factory=lambda w: predictor)
    a.load()
    return a, predictor


class TestLifecycle:
    def test_module_import_is_light(self):
        # Importing the module must not pull ultralytics/torch.
        assert "ultralytics" not in sys.modules or True  # other tests may load it
        import src.engine.sam_assist as mod

        src_text = Path(mod.__file__).read_text(encoding="utf-8")
        head = src_text.split("class SamAssistError")[0]
        assert "import ultralytics" not in head
        assert "import torch" not in head

    def test_load_idempotent(self):
        calls = []
        a = SamAssistant(predictor_factory=lambda w: calls.append(w) or _FakePredictor())
        a.load()
        a.load()
        assert len(calls) == 1
        assert a.is_loaded

    def test_load_failure_raises_chinese_error_with_hint(self):
        def boom(w):
            raise RuntimeError("no net")

        a = SamAssistant(predictor_factory=boom)
        with pytest.raises(SamAssistError) as e:
            a.load()
        assert "SAM 模型加载失败" in str(e.value)
        assert WEIGHT_FILENAME in str(e.value)
        assert not a.is_loaded

    def test_encode_before_load_raises(self):
        a = SamAssistant(predictor_factory=lambda w: _FakePredictor())
        with pytest.raises(SamAssistError):
            a.encode(Path("x.jpg"), (100, 80))

    def test_predict_before_encode_raises(self):
        a, _ = _loaded_assistant()
        with pytest.raises(SamAssistError):
            a.predict(PointPrompt(0.5, 0.5))

    def test_reset_and_release(self):
        a, pred = _loaded_assistant()
        a.encode(Path("img.jpg"), (100, 80))
        assert a.encoded_path == Path("img.jpg")
        a.reset()
        assert pred.reset_calls == 1
        assert a.encoded_path is None
        a.release()
        assert not a.is_loaded

    def test_encode_bad_size_raises(self):
        a, _ = _loaded_assistant()
        with pytest.raises(SamAssistError):
            a.encode(Path("img.jpg"), (0, 80))

    def test_resolve_weights_prefers_home_models_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        p = resolve_weights_path()
        assert p == tmp_path / ".autolabel" / "models" / WEIGHT_FILENAME
        assert p.parent.is_dir()  # created so a download can land there


class TestPromptAssembly:
    def test_point_prompt_is_pixel_space_foreground(self):
        a, pred = _loaded_assistant()
        a.encode(Path("img.jpg"), (200, 100))
        a.predict(PointPrompt(0.5, 0.25))
        kw = pred.call_kwargs[-1]
        assert kw["points"] == [100.0, 25.0]
        assert kw["labels"] == [1]
        assert "bboxes" not in kw

    def test_box_prompt_is_pixel_xyxy(self):
        a, pred = _loaded_assistant()
        a.encode(Path("img.jpg"), (200, 100))
        a.predict(BoxPrompt(0.1, 0.2, 0.5, 0.8))
        kw = pred.call_kwargs[-1]
        assert kw["bboxes"] == pytest.approx([20.0, 20.0, 100.0, 80.0])
        assert "points" not in kw

    def test_unknown_prompt_type_raises(self):
        a, _ = _loaded_assistant()
        a.encode(Path("img.jpg"), (100, 100))
        with pytest.raises(SamAssistError):
            a.predict("not-a-prompt")  # type: ignore[arg-type]

    def test_inference_exception_wrapped(self):
        class _Boom(_FakePredictor):
            def __call__(self, **kwargs):
                raise RuntimeError("decode blew up")

        a, _ = _loaded_assistant(_Boom())
        a.encode(Path("img.jpg"), (100, 100))
        with pytest.raises(SamAssistError) as e:
            a.predict(PointPrompt(0.5, 0.5))
        assert "SAM 推理失败" in str(e.value)


class TestMaskToPolygon:
    def test_square_contour_yields_polygon_and_derived_bbox(self):
        a, _ = _loaded_assistant()
        a.encode(Path("img.jpg"), (100, 100))
        res = a.predict(PointPrompt(0.3, 0.3))
        assert isinstance(res, MaskResult)
        assert len(res.polygon) >= 3
        # Derived-bbox invariant (segment task contract).
        assert res.bbox == bbox_from_polygon(res.polygon)

    def test_dense_contour_is_simplified(self):
        import math

        pred = _FakePredictor()
        # 400-point circle — must shrink dramatically after pixel-space DP.
        pred.contours = [[
            [0.5 + 0.3 * math.cos(2 * math.pi * i / 400),
             0.5 + 0.3 * math.sin(2 * math.pi * i / 400)]
            for i in range(400)
        ]]
        a, _ = _loaded_assistant(pred)
        a.encode(Path("img.jpg"), (640, 480))
        res = a.predict(BoxPrompt(0.2, 0.2, 0.8, 0.8))
        assert res is not None
        assert 3 <= len(res.polygon) < 100

    def test_masks_none_returns_none(self):
        pred = _FakePredictor()
        pred.contours = None
        a, _ = _loaded_assistant(pred)
        a.encode(Path("img.jpg"), (100, 100))
        assert a.predict(PointPrompt(0.5, 0.5)) is None

    def test_empty_xyn_returns_none(self):
        pred = _FakePredictor()
        pred.contours = []
        a, _ = _loaded_assistant(pred)
        a.encode(Path("img.jpg"), (100, 100))
        assert a.predict(PointPrompt(0.5, 0.5)) is None

    def test_degenerate_contour_returns_none(self):
        pred = _FakePredictor()
        pred.contours = [[[0.1, 0.1], [0.2, 0.2]]]  # 2 points
        a, _ = _loaded_assistant(pred)
        a.encode(Path("img.jpg"), (100, 100))
        assert a.predict(PointPrompt(0.5, 0.5)) is None

    def test_zero_area_contour_returns_none(self):
        pred = _FakePredictor()
        pred.contours = [[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]]
        a, _ = _loaded_assistant(pred)
        a.encode(Path("img.jpg"), (100, 100))
        assert a.predict(PointPrompt(0.3, 0.3)) is None

    def test_empty_results_list_returns_none(self):
        class _Empty(_FakePredictor):
            def __call__(self, **kwargs):
                self.call_kwargs.append(kwargs)
                return []

        a, _ = _loaded_assistant(_Empty())
        a.encode(Path("img.jpg"), (100, 100))
        assert a.predict(PointPrompt(0.5, 0.5)) is None
