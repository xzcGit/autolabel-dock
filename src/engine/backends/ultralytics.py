"""Ultralytics model backend."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.engine.backends.base import (
    BackendProbe,
    BackendUnavailableError,
    DEFAULT_BACKEND_ID,
    DEFAULT_BACKEND_RUNTIME,
)
from src.engine.predictor import Predictor
from src.engine.trainer import Trainer


class UltralyticsBackend:
    """Default backend backed by the Ultralytics YOLO package."""

    backend_id = DEFAULT_BACKEND_ID
    display_name = "Ultralytics"

    def __init__(self, yolo_cls=None):
        self._yolo_cls = yolo_cls

    def probe(self) -> BackendProbe:
        """Inspect the installed Ultralytics package without importing YOLO."""
        try:
            backend_version = version("ultralytics")
        except PackageNotFoundError:
            return BackendProbe(
                backend_id=self.backend_id,
                display_name=self.display_name,
                available=False,
                runtime=DEFAULT_BACKEND_RUNTIME,
                message="未安装 ultralytics",
            )

        # Check the modern Ultralytics YOLO API baseline without capping future versions.
        try:
            from packaging.version import Version
            ver = Version(backend_version)
            if ver < Version("8.0.0"):
                return BackendProbe(
                    backend_id=self.backend_id,
                    display_name=self.display_name,
                    available=True,
                    version=backend_version,
                    runtime=DEFAULT_BACKEND_RUNTIME,
                    message=f"版本 {backend_version} 低于支持版本 8.0，无法保证提供当前项目所需的 YOLO 接口",
                )
        except Exception:
            # If packaging is not available or version parsing fails, proceed without check.
            pass

        return BackendProbe(
            backend_id=self.backend_id,
            display_name=self.display_name,
            available=True,
            version=backend_version,
            runtime=DEFAULT_BACKEND_RUNTIME,
        )

    def infer_model_format(self, model_path: str | Path) -> str:
        """Infer a simple model format label from the file extension."""
        suffix = Path(model_path).suffix.lower().lstrip(".")
        return suffix or "unknown"

    def load_predictor(self, model_path: str | Path, model_info) -> Predictor:
        """Load a YOLO-compatible model and wrap it in the project predictor."""
        yolo_cls = self._resolve_yolo_cls()
        yolo_model = yolo_cls(str(model_path))
        return Predictor(yolo_model)

    def create_trainer(self) -> Trainer:
        """Create a trainer for Ultralytics training jobs."""
        return Trainer(yolo_cls=self._resolve_yolo_cls())

    def _resolve_yolo_cls(self):
        if self._yolo_cls is not None:
            return self._yolo_cls
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError(
                "未安装 ultralytics，无法使用 Ultralytics 后端加载或训练模型"
            ) from exc
        return YOLO
