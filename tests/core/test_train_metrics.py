"""Tests for src/core/train_metrics.py — Qt-free curve/metric engine."""
from src.core.train_metrics import (
    TASK_QUALITY_METRICS,
    TrainSeries,
    build_quality_series,
    build_series,
    compute_val_loss,
    pick_metric,
)


class TestPickMetric:
    def test_returns_first_present_candidate(self):
        metrics = {"metrics/mAP50(B)": 0.8, "mAP50": 0.3}
        assert pick_metric(metrics, ["metrics/mAP50(B)", "mAP50"]) == 0.8

    def test_falls_through_to_later_candidate(self):
        metrics = {"mAP50": 0.42}
        assert pick_metric(metrics, ["metrics/mAP50(B)", "mAP50"]) == 0.42

    def test_skips_non_numeric_candidate_and_continues(self):
        metrics = {"a": "not-a-number", "b": 0.5}
        assert pick_metric(metrics, ["a", "b"]) == 0.5

    def test_returns_none_when_no_candidate_present(self):
        assert pick_metric({"other": 1.0}, ["a", "b"]) is None

    def test_returns_none_when_only_candidate_is_non_numeric(self):
        assert pick_metric({"a": None}, ["a"]) is None

    def test_coerces_int_to_float(self):
        assert pick_metric({"a": 3}, ["a"]) == 3.0
        assert isinstance(pick_metric({"a": 3}, ["a"]), float)


class TestComputeValLoss:
    def test_legacy_val_loss_key_wins(self):
        assert compute_val_loss({"val_loss": 1.25}) == 1.25

    def test_legacy_val_loss_non_numeric_falls_back_to_sum(self):
        m = {"val_loss": "bad", "val/box_loss": 0.5, "val/cls_loss": 0.25}
        assert compute_val_loss(m) == 0.75

    def test_sums_all_val_loss_like_keys(self):
        m = {"val/box_loss": 0.5, "val/cls_loss": 0.25, "val/dfl_loss": 0.25}
        assert compute_val_loss(m) == 1.0

    def test_ignores_non_loss_val_keys(self):
        m = {"val/box_loss": 0.5, "val/accuracy": 0.9}
        assert compute_val_loss(m) == 0.5

    def test_skips_non_numeric_val_loss_entries(self):
        m = {"val/box_loss": 0.5, "val/cls_loss": None}
        assert compute_val_loss(m) == 0.5

    def test_returns_none_when_no_val_metrics(self):
        assert compute_val_loss({"train_loss": 1.0}) is None

    def test_case_insensitive_loss_match(self):
        assert compute_val_loss({"val/Box_Loss": 0.4}) == 0.4


class TestBuildSeries:
    def test_empty_history(self):
        s = build_series([])
        assert isinstance(s, TrainSeries)
        assert s.epochs == []
        assert s.train_losses == []
        assert s.val_losses == []

    def test_positional_epochs_ignore_epoch_key(self):
        history = [{"epoch": 5, "train_loss": 1.0}, {"epoch": 6, "train_loss": 0.9}]
        s = build_series(history)
        assert s.epochs == [1, 2]

    def test_train_loss_missing_or_falsy_becomes_zero(self):
        history = [{}, {"train_loss": 0}, {"train_loss": 0.5}]
        s = build_series(history)
        assert s.train_losses == [0.0, 0.0, 0.5]

    def test_val_loss_carry_forward_over_gaps(self):
        history = [
            {"train_loss": 1.0, "val/box_loss": 0.8},  # val=0.8
            {"train_loss": 0.9},                        # no val → carry 0.8
            {"train_loss": 0.8, "val/box_loss": 0.5},  # val=0.5
        ]
        s = build_series(history)
        assert s.val_losses == [0.8, 0.8, 0.5]

    def test_val_loss_initial_zero_until_first_value(self):
        history = [{"train_loss": 1.0}, {"train_loss": 0.9, "val/box_loss": 0.5}]
        s = build_series(history)
        assert s.val_losses == [0.0, 0.5]

    def test_no_val_metrics_at_all_stays_zero(self):
        history = [{"train_loss": 1.0}, {"train_loss": 0.9}]
        s = build_series(history)
        assert s.val_losses == [0.0, 0.0]


class TestBuildQualitySeries:
    def test_carry_forward_over_gaps(self):
        history = [
            {"metrics/mAP50(B)": 0.3},
            {},                          # carry 0.3
            {"metrics/mAP50(B)": 0.6},
        ]
        assert build_quality_series(history, ["metrics/mAP50(B)"]) == [0.3, 0.3, 0.6]

    def test_initial_zero_until_first_value(self):
        history = [{}, {"mAP50": 0.4}]
        assert build_quality_series(history, ["metrics/mAP50(B)", "mAP50"]) == [0.0, 0.4]

    def test_empty_history(self):
        assert build_quality_series([], ["metrics/mAP50(B)"]) == []

    def test_no_matching_key_ever_stays_zero(self):
        history = [{"other": 1.0}, {"other": 2.0}]
        assert build_quality_series(history, ["metrics/mAP50(B)"]) == [0.0, 0.0]

    def test_candidate_priority_within_epoch(self):
        history = [{"metrics/mAP50(B)": 0.7, "mAP50": 0.1}]
        assert build_quality_series(history, ["metrics/mAP50(B)", "mAP50"]) == [0.7]


class TestTaskQualityMetrics:
    def test_has_the_four_project_tasks(self):
        assert set(TASK_QUALITY_METRICS) == {"detect", "pose", "classify", "segment"}

    def test_entry_shape_is_title_and_specs(self):
        title, specs = TASK_QUALITY_METRICS["detect"]
        assert title == "mAP"
        assert specs[0][0] == "mAP50"
        assert "metrics/mAP50(B)" in specs[0][1]

    def test_segment_quality_prefers_mask_variant(self):
        title, specs = TASK_QUALITY_METRICS["segment"]
        assert title == "mAP (Segment)"
        # The primary segment curve reads the mask (M) key, not the box (B) key.
        assert specs[0][1][0] == "metrics/mAP50(M)"

    def test_segment_pick_metric_reads_mask_over_box(self):
        _, specs = TASK_QUALITY_METRICS["segment"]
        metrics = {"metrics/mAP50(M)": 0.8, "metrics/mAP50(B)": 0.3}
        assert pick_metric(metrics, specs[0][1]) == 0.8
