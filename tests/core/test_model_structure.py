"""Tests for core model-structure parsing.

No real .pt is committed, so these tests build a minimal fake object mimicking
the ultralytics structure (a top-level ``.model`` whose ``.model`` is an
iterable of modules carrying ``.i`` / ``.type`` / ``.np``) and exercise the
pure-parsing helper ``_layers_from_model``. The load path
(``parse_model_structure``) is covered for the file-not-found error case.
"""
import pytest

from src.core.model_structure import (
    LayerInfo,
    ModelStructureError,
    _layers_from_model,
    parse_model_structure,
)


class _FakeModule:
    """Mimics a parsed ultralytics top-level module (no torch)."""

    def __init__(self, i, type_str, np):
        self.i = i
        self.type = type_str
        self.np = np

    # No register_forward_hook / parameters — so the forward pass is skipped
    # and output_shape falls back to "-".


class _FakeInner:
    """Mimics the inner nn.Module: has a ``.model`` Sequential-like list."""

    def __init__(self, modules):
        self.model = modules

    def parameters(self):
        # Return empty so _layers_from_model falls back to per-module .np sum.
        return iter([])


class _FakeYOLO:
    """Mimics the ultralytics.YOLO wrapper: has a ``.model`` inner module."""

    def __init__(self, modules):
        self.model = _FakeInner(modules)


def _make_model():
    return _FakeYOLO([
        _FakeModule(0, "ultralytics.nn.modules.conv.Conv", 100),
        _FakeModule(1, "ultralytics.nn.modules.conv.Conv", 200),
        _FakeModule(2, "ultralytics.nn.modules.block.C2f", 700),
        _FakeModule(3, "ultralytics.nn.modules.head.Detect", 0),
    ])


class TestLayersFromModel:
    def test_indices_match_positions(self):
        layers = _layers_from_model(_make_model())
        assert [ly.index for ly in layers] == [0, 1, 2, 3]

    def test_short_type_names(self):
        layers = _layers_from_model(_make_model())
        assert [ly.module_type for ly in layers] == ["Conv", "Conv", "C2f", "Detect"]

    def test_params_captured(self):
        layers = _layers_from_model(_make_model())
        assert [ly.params for ly in layers] == [100, 200, 700, 0]

    def test_cumulative_ratio_monotonic_and_ends_at_one(self):
        layers = _layers_from_model(_make_model())
        ratios = [ly.params_ratio for ly in layers]
        # Monotonic non-decreasing.
        assert all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))
        # Ends at 1.0 (total params = 1000, cumulative reaches 1000).
        assert ratios[-1] == pytest.approx(1.0)
        # First layer 100/1000 = 0.1.
        assert ratios[0] == pytest.approx(0.1)

    def test_output_shape_dash_when_no_forward(self):
        # Fake modules have no register_forward_hook and inner has no real
        # forward, so the best-effort forward fails and shapes are "-".
        layers = _layers_from_model(_make_model())
        assert all(ly.output_shape == "-" for ly in layers)

    def test_returns_layerinfo_instances(self):
        layers = _layers_from_model(_make_model())
        assert all(isinstance(ly, LayerInfo) for ly in layers)

    def test_not_a_yolo_model_raises(self):
        class _NotYolo:
            model = None

        with pytest.raises(ModelStructureError):
            _layers_from_model(_NotYolo())

    def test_empty_sequential_raises(self):
        with pytest.raises(ModelStructureError):
            _layers_from_model(_FakeYOLO([]))

    def test_explicit_i_attribute_used_over_position(self):
        # If .i disagrees with position, .i (freeze-aligned) wins.
        model = _FakeYOLO([
            _FakeModule(5, "a.b.Conv", 10),
            _FakeModule(6, "a.b.Conv", 20),
        ])
        layers = _layers_from_model(model)
        assert [ly.index for ly in layers] == [5, 6]


class TestParseModelStructure:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nope.pt"
        with pytest.raises(ModelStructureError):
            parse_model_structure(missing)

    def test_missing_file_message_is_chinese(self, tmp_path):
        missing = tmp_path / "nope.pt"
        with pytest.raises(ModelStructureError) as exc:
            parse_model_structure(missing)
        assert "模型文件不存在" in str(exc.value)


class TestLayerInfo:
    def test_to_dict(self):
        ly = LayerInfo(index=0, module_type="Conv", params=100, params_ratio=0.5, output_shape="-")
        d = ly.to_dict()
        assert d == {
            "index": 0,
            "module_type": "Conv",
            "params": 100,
            "params_ratio": 0.5,
            "output_shape": "-",
        }
