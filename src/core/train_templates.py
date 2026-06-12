"""Training parameter templates — task-aware, user-savable presets persisted globally."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainTemplate:
    """A named, task-bound snapshot of training parameters."""

    name: str
    task: str  # "detect" | "classify" | "pose"
    params: dict
    created_at: str
    builtin: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "params": dict(self.params),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainTemplate":
        return cls(
            name=d["name"],
            task=d["task"],
            params=dict(d.get("params", {})),
            created_at=d.get("created_at", ""),
            builtin=False,
        )


def _builtin_default() -> TrainTemplate:
    """Code-injected '默认' template — applies TrainConfig dataclass defaults."""
    return TrainTemplate(
        name="默认",
        task="*",
        params={},
        created_at="",
        builtin=True,
    )


# ── Task-relevant parameter sets ───────────────────────────────

_COMMON_KEYS = (
    "model",
    "epochs", "batch", "imgsz", "device", "freeze", "workers", "patience",
    "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "hsv_h", "hsv_s", "hsv_v", "scale", "flipud", "fliplr", "auto_augment",
)
_DETECT_KEYS = (
    "degrees", "translate", "shear", "perspective",
    "mosaic", "mixup", "copy_paste",
    "include_detect_params",
)
_CLASSIFY_KEYS = ("erasing", "dropout", "include_classify_params")
_POSE_KEYS = ("pose", "kobj", "kpt_shape", "include_pose_params")


def extract_task_params(config) -> dict:
    """Build the saveable subset for the config's task.

    Returns a dict with:
      - common params (always)
      - detect/pose aug params (when task in {"detect", "pose"})
      - classify-only params (when task == "classify")
      - pose-specific params (when task == "pose")
    Excludes runtime-only fields (data_yaml/project/name/resume) and `task`
    itself (`task` lives on TrainTemplate, not in params).
    """
    snapshot = config.to_storage_dict()
    keys = list(_COMMON_KEYS)
    task = config.task
    if task in ("detect", "pose"):
        keys.extend(_DETECT_KEYS)
    if task == "classify":
        keys.extend(_CLASSIFY_KEYS)
    if task == "pose":
        keys.extend(_POSE_KEYS)
    return {k: snapshot[k] for k in keys if k in snapshot}


class TemplateRegistry:
    """Persists user training templates to a JSON file. Built-in '默认' is code-injected."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._templates: list[TrainTemplate] = []

    def upsert(self, template: TrainTemplate) -> None:
        if template.builtin:
            raise ValueError("Cannot upsert a builtin template")
        for i, existing in enumerate(self._templates):
            if existing.name == template.name and existing.task == template.task:
                self._templates[i] = template
                return
        self._templates.append(template)

    def remove(self, name: str, task: str) -> bool:
        for i, t in enumerate(self._templates):
            if t.name == name and t.task == task:
                if t.builtin:
                    return False
                del self._templates[i]
                return True
        return False

    def get(self, name: str, task: str) -> TrainTemplate | None:
        if name == "默认":
            return _builtin_default()
        for t in self._templates:
            if t.name == name and t.task == task:
                return t
        return None

    def list(self, task: str | None = None) -> list[TrainTemplate]:
        result: list[TrainTemplate] = [_builtin_default()]
        for t in self._templates:
            if task is None or t.task == task:
                result.append(t)
        return result

    def load(self) -> None:
        """Load registry from disk. Missing/corrupt files degrade to empty list with warning."""
        if not self._path.exists():
            self._templates = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._templates = [TrainTemplate.from_dict(t) for t in data.get("templates", [])]
            logger.info("Templates loaded: %d from %s", len(self._templates), self._path)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to load train templates from %s: %s", self._path, e)
            self._templates = []

    def save(self) -> None:
        """Save user templates to disk. Built-in '默认' is never persisted."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "templates": [t.to_dict() for t in self._templates],
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
