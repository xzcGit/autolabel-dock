"""Qt-free training-curve engine: metric picking and curve-series building.

Extracted from ``src/ui/train_panel.py`` so the numeric behaviour of the
loss / quality curves can be unit-tested without a QApplication. The panel
keeps only the presentation side (``setData`` calls, log lines, pens/colors).

Carry-forward semantics are preserved bit-for-bit from the old inline loops
in ``TrainPanel.update_epoch``:

* series start at 0.0 until the first real value appears;
* epochs with no matching metric repeat the last known value (some tasks
  only validate periodically — the line must not dip back to 0);
* values that fail ``float()`` coercion are skipped as if absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Quality metric per task: (display title, list of (curve label, metric-key candidates))
# The first key that appears in the emitted metrics dict wins.
TASK_QUALITY_METRICS: dict[str, tuple[str, list[tuple[str, list[str]]]]] = {
    "detect": (
        "mAP",
        [
            ("mAP50", ["metrics/mAP50(B)", "mAP50"]),
            ("mAP50-95", ["metrics/mAP50-95(B)", "mAP50-95"]),
        ],
    ),
    "pose": (
        "mAP (Pose)",
        [
            ("Pose mAP50", ["metrics/mAP50(P)", "metrics/mAP50(B)"]),
            ("Pose mAP50-95", ["metrics/mAP50-95(P)", "metrics/mAP50-95(B)"]),
        ],
    ),
    "segment": (
        "mAP (Segment)",
        [
            ("Mask mAP50", ["metrics/mAP50(M)", "metrics/mAP50(B)"]),
            ("Mask mAP50-95", ["metrics/mAP50-95(M)", "metrics/mAP50-95(B)"]),
            ("Box mAP50", ["metrics/mAP50(B)"]),
        ],
    ),
    "classify": (
        "Accuracy",
        [
            ("Top-1", ["metrics/accuracy_top1", "accuracy_top1"]),
            ("Top-5", ["metrics/accuracy_top5", "accuracy_top5"]),
        ],
    ),
}


def pick_metric(metrics: dict, candidates: list[str]) -> float | None:
    """Return the first matching numeric value among candidate keys, else None."""
    for key in candidates:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def compute_val_loss(metrics: dict) -> float | None:
    """Sum all val/* loss-like keys. Falls back to legacy 'val_loss' if present."""
    if "val_loss" in metrics:
        try:
            return float(metrics["val_loss"])
        except (TypeError, ValueError):
            pass
    total = 0.0
    found = False
    for k, v in metrics.items():
        if k.startswith("val/") and "loss" in k.lower():
            try:
                total += float(v)
                found = True
            except (TypeError, ValueError):
                continue
    return total if found else None


@dataclass
class TrainSeries:
    """Loss-curve series derived from an epoch-metrics history."""

    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def build_series(history: list[dict]) -> TrainSeries:
    """Build the loss series from the accumulated per-epoch metrics dicts.

    ``epochs`` is positional (1..N) regardless of any 'epoch' key in the
    dicts. Train loss falls back to 0.0 for missing/falsy values. Val loss
    carries the last known value forward (initial 0.0) so the line doesn't
    dip to 0 on epochs that have no val metrics.
    """
    epochs = list(range(1, len(history) + 1))
    train_losses = [float(d.get("train_loss", 0) or 0) for d in history]
    val_losses_raw = [compute_val_loss(d) for d in history]
    last_val = 0.0
    val_losses: list[float] = []
    for v in val_losses_raw:
        if v is not None:
            last_val = v
        val_losses.append(last_val)
    return TrainSeries(epochs=epochs, train_losses=train_losses, val_losses=val_losses)


def build_quality_series(history: list[dict], candidates: list[str]) -> list[float]:
    """Build one quality curve's y-values with the same carry-forward rule.

    Each epoch takes the first numeric value among ``candidates`` present in
    its metrics dict; epochs with no match repeat the last value (initial 0.0).
    """
    last = 0.0
    ys: list[float] = []
    for d in history:
        v = pick_metric(d, candidates)
        if v is not None:
            last = v
        ys.append(last)
    return ys
