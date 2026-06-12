"""Model registry and management."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.engine.backends.base import DEFAULT_BACKEND_ID, DEFAULT_BACKEND_RUNTIME

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model."""

    name: str
    path: str  # relative to project dir
    task: str  # "detect", "classify", "pose"
    base_model: str
    classes: list[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    epochs: int = 0
    dataset_size: int = 0
    train_params: dict = field(default_factory=dict)
    backend_id: str = DEFAULT_BACKEND_ID
    model_format: str = "pt"
    backend_version: str = ""
    backend_runtime: str = DEFAULT_BACKEND_RUNTIME
    backend_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "task": self.task,
            "base_model": self.base_model,
            "classes": self.classes,
            "metrics": self.metrics,
            "trained_at": self.trained_at,
            "epochs": self.epochs,
            "dataset_size": self.dataset_size,
            "train_params": self.train_params,
            "backend_id": self.backend_id,
            "model_format": self.model_format,
            "backend_version": self.backend_version,
            "backend_runtime": self.backend_runtime,
            "backend_metadata": self.backend_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelInfo:
        return cls(
            id=d["id"],
            name=d["name"],
            path=d["path"],
            task=d["task"],
            base_model=d["base_model"],
            classes=d["classes"],
            metrics=d.get("metrics", {}),
            trained_at=d.get("trained_at", ""),
            epochs=d.get("epochs", 0),
            dataset_size=d.get("dataset_size", 0),
            train_params=d.get("train_params", {}),
            backend_id=d.get("backend_id", DEFAULT_BACKEND_ID),
            model_format=d.get("model_format", "pt"),
            backend_version=d.get("backend_version", ""),
            backend_runtime=d.get("backend_runtime", DEFAULT_BACKEND_RUNTIME),
            backend_metadata=d.get("backend_metadata", {}),
        )


class ModelRegistry:
    """Manages the model registry (models/registry.json)."""

    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self._models: list[ModelInfo] = []

    @property
    def _registry_path(self) -> Path:
        return self.models_dir / "registry.json"

    def load(self) -> None:
        """Load registry from disk."""
        if not self._registry_path.exists():
            self._models = []
            return
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            self._models = [ModelInfo.from_dict(m) for m in data.get("models", [])]
            logger.info("Registry loaded: %d models from %s", len(self._models), self._registry_path)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load model registry from %s: %s", self._registry_path, e)
            self._models = []

    def save(self) -> None:
        """Save registry to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        data = {"models": [m.to_dict() for m in self._models]}
        self._registry_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def register(self, model_info: ModelInfo) -> None:
        """Register a model."""
        self._models.append(model_info)

    def remove(self, model_id: str) -> None:
        """Remove a model by ID."""
        self._models = [m for m in self._models if m.id != model_id]

    def get(self, model_id: str) -> ModelInfo | None:
        """Get a model by ID."""
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    def rename(self, model_id: str, new_name: str) -> bool:
        """Rename a model's display name. Returns True if found and renamed."""
        model = self.get(model_id)
        if model is None:
            return False
        model.name = new_name
        return True

    def list_models(self, task: str | None = None) -> list[ModelInfo]:
        """List models, optionally filtered by task."""
        if task:
            return [m for m in self._models if m.task == task]
        return list(self._models)
