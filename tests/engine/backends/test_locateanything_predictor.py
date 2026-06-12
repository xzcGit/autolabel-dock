"""Tests for LocateAnythingPredictor parsing + hybrid class mapping (sidecar).

No torch/transformers required: the predictor is a client/proxy over a sidecar
worker, so we install a fake ``_WorkerProcess`` (or feed raw text directly into
``_parse_and_map``) to exercise parsing, coordinate conversion, the hybrid
natural-language → class mapping rules, and the IPC round-trip — all without
any heavy ML dependency or model load.
"""
import pytest

from src.engine.backends.locateanything import LocateAnythingPredictor


class _FakeWorker:
    """Stand-in for _WorkerProcess: records the prompt, returns canned text."""

    def __init__(self, raw_text="", width=640, height=480, error=None):
        self.raw_text = raw_text
        self.width = width
        self.height = height
        self.error = error
        self.terminated = False
        self.last_request = None

    def infer(self, image_path, prompt, timeout=None):
        self.last_request = (image_path, prompt)
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.raw_text, self.width, self.height

    def terminate(self):
        self.terminated = True


def _make_predictor(raw_text, width=640, height=480):
    worker = _FakeWorker(raw_text=raw_text, width=width, height=height)
    return LocateAnythingPredictor(worker=worker), worker


# ── Coordinate conversion ─────────────────────────────────────────────────


def test_parse_box_converts_to_normalized_center():
    raw = "<ref>cat</ref><box><100><200><300><400></box>"
    pred, _ = _make_predictor(raw)
    pred.set_query("", None)
    anns = pred._parse_and_map(raw, ["cat"])
    assert len(anns) == 1
    ann = anns[0]
    # [0,1000] corners -> center format normalized.
    assert ann.bbox == pytest.approx((0.2, 0.3, 0.2, 0.2))
    assert ann.class_name == "cat"
    assert ann.class_id == 0
    assert ann.confidence == 1.0
    assert ann.confirmed is False
    assert ann.source == "auto"


def test_parse_multiple_boxes():
    raw = (
        "<ref>cat</ref><box><100><100><200><200></box>"
        "<ref>dog</ref><box><500><500><600><700></box>"
    )
    pred, _ = _make_predictor(raw)
    pred.set_query("", None)
    anns = pred._parse_and_map(raw, ["cat", "dog"])
    assert [a.class_name for a in anns] == ["cat", "dog"]


# ── Hybrid mapping: target_class forces all boxes into one class ──────────


def test_target_class_forces_all_into_one_class():
    raw = (
        "<ref>kitten</ref><box><100><100><200><200></box>"
        "<ref>puppy</ref><box><300><300><400><400></box>"
    )
    pred, _ = _make_predictor(raw)
    pred.set_query("a cute animal", target_class="cat")
    anns = pred._parse_and_map(raw, ["cat", "dog"])
    assert len(anns) == 2
    assert all(a.class_name == "cat" for a in anns)
    assert all(a.class_id == 0 for a in anns)
    assert pred.last_dropped == 0


def test_target_class_not_in_project_uses_zero_id():
    raw = "<ref>x</ref><box><0><0><100><100></box>"
    pred, _ = _make_predictor(raw)
    pred.set_query("x", target_class="not_a_project_class")
    anns = pred._parse_and_map(raw, ["cat", "dog"])
    assert len(anns) == 1
    assert anns[0].class_name == "not_a_project_class"
    assert anns[0].class_id == 0


# ── Hybrid mapping: no target — name match + drop unmatched ───────────────


def test_no_target_name_match_drops_unmatched_and_counts():
    raw = (
        "<ref>cat</ref><box><100><100><200><200></box>"
        "<ref>elephant</ref><box><300><300><400><400></box>"
    )
    pred, _ = _make_predictor(raw)
    pred.set_query("cat, elephant", target_class=None)
    anns = pred._parse_and_map(raw, ["cat", "dog"])
    assert len(anns) == 1
    assert anns[0].class_name == "cat"
    assert pred.last_dropped == 1


def test_no_target_name_match_is_case_insensitive():
    raw = "<ref>  CAT </ref><box><100><100><200><200></box>"
    pred, _ = _make_predictor(raw)
    pred.set_query("cat", target_class=None)
    anns = pred._parse_and_map(raw, ["Cat"])
    assert len(anns) == 1
    assert anns[0].class_name == "Cat"


# ── Prompt assembly ───────────────────────────────────────────────────────


def test_build_prompt_uses_user_prompt_when_present():
    pred, _ = _make_predictor("")
    pred.set_query("a red car", None)
    prompt = pred._build_prompt(["cat", "dog"])
    assert "a red car" in prompt


def test_build_prompt_falls_back_to_project_classes_when_empty():
    pred, _ = _make_predictor("")
    pred.set_query("", None)
    prompt = pred._build_prompt(["cat", "dog"])
    assert "cat</c>dog" in prompt


# ── predict / predict_with_size via fake sidecar worker ────────────────────


def test_predict_with_size_returns_annotations_and_worker_size():
    raw = "<ref>cat</ref><box><100><200><300><400></box>"
    pred, worker = _make_predictor(raw, width=640, height=480)
    pred.set_query("", None)

    anns, size = pred.predict_with_size("fake.jpg", project_classes=["cat"])
    # Size comes from the worker (client never opens the image).
    assert size == (640, 480)
    assert len(anns) == 1
    assert anns[0].class_name == "cat"
    # The prompt was assembled and shipped to the worker.
    assert worker.last_request[0] == "fake.jpg"
    assert "cat" in worker.last_request[1]


def test_predict_ships_prompt_built_from_query():
    raw = ""
    pred, worker = _make_predictor(raw)
    pred.set_query("a red car", target_class=None)
    pred.predict("img.png", project_classes=["car"])
    assert "a red car" in worker.last_request[1]


def test_predict_classify_not_supported():
    from src.engine.backends.base import BackendUnavailableError

    pred, _ = _make_predictor("")
    with pytest.raises(BackendUnavailableError):
        pred.predict_classify("fake.jpg")


def test_empty_output_produces_no_annotations():
    pred, _ = _make_predictor("no boxes here")
    pred.set_query("", None)
    anns = pred._parse_and_map("no boxes here", ["cat"])
    assert anns == []


def test_predict_without_worker_raises():
    pred = LocateAnythingPredictor(worker=None)
    pred.set_query("", None)
    with pytest.raises(RuntimeError):
        pred.predict("fake.jpg", project_classes=["cat"])


# ── Worker error surfaces as a RuntimeError (does NOT crash) ───────────────


def test_worker_error_surfaces_as_runtime_error():
    pred = LocateAnythingPredictor(
        worker=_FakeWorker(error="推理显存不足 (CUDA OOM)")
    )
    pred.set_query("", None)
    with pytest.raises(RuntimeError) as exc:
        pred.predict("fake.jpg", project_classes=["cat"])
    assert "显存不足" in str(exc.value)


# ── release() terminates the sidecar (no torch import) ─────────────────────


def test_release_terminates_worker():
    pred, worker = _make_predictor("")
    pred.release()
    assert worker.terminated is True
    # Idempotent: second release is a no-op (worker already dropped).
    pred.release()


def test_release_without_worker_is_noop():
    pred = LocateAnythingPredictor(worker=None)
    pred.release()  # must not raise
