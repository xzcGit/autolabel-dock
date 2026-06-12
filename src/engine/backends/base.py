"""Backend contracts for model training and inference runtimes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from src.core.annotation import Annotation
    from src.engine.trainer import TrainConfig

DEFAULT_BACKEND_ID = "ultralytics"
DEFAULT_BACKEND_RUNTIME = "in_process"


class BackendError(RuntimeError):
    """Base error raised by model backends."""


class BackendUnavailableError(BackendError):
    """Raised when a backend dependency is not installed or cannot be used."""


class UnknownBackendError(BackendError):
    """Raised when a model references a backend that is not registered."""


@dataclass(frozen=True)
class BackendProbe:
    """Runtime information about an installed backend."""

    backend_id: str
    display_name: str
    available: bool
    version: str = ""
    runtime: str = DEFAULT_BACKEND_RUNTIME
    message: str = ""
    metadata: dict = field(default_factory=dict)


class TrainerProtocol(Protocol):
    """Minimal training object contract used by TrainWorker."""

    @property
    def cancelled(self) -> bool:
        ...

    def request_cancel(self) -> None:
        ...

    def train(
        self,
        config: TrainConfig,
        on_epoch_end: Callable[[dict], None] | None = None,
    ) -> None:
        ...

    def get_best_metrics(self) -> dict[str, float]:
        ...


class PredictorProtocol(Protocol):
    """Minimal inference contract used by controllers and workers."""

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
    ) -> list[Annotation]:
        ...

    def predict_with_size(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        ...

    def predict_classify(
        self,
        image_path: str | Path,
        project_classes: list[str] | None = None,
        filter_to_project: bool = True,
    ) -> tuple[str, float] | None:
        ...

    def release(self) -> None:
        """Optional: free model + GPU memory. No-op for in-process backends
        that don't hold scarce GPU resources (e.g. Ultralytics)."""
        ...


class ModelBackend(Protocol):
    """Pluggable model runtime provider."""

    backend_id: str
    display_name: str

    def probe(self) -> BackendProbe:
        ...

    def infer_model_format(self, model_path: str | Path) -> str:
        ...

    def load_predictor(self, model_path: str | Path, model_info) -> PredictorProtocol:
        ...

    def create_trainer(self) -> TrainerProtocol:
        ...
