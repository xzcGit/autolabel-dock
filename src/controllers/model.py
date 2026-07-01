"""Model controller — load, delete, import, inference."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog, QInputDialog, QMessageBox

from src.core.project import ProjectManager
from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.core.model_structure import LayerInfo, ModelStructureError, parse_model_structure
from src.engine.backends import get_backend
from src.engine.backends.base import BackendError, PredictorProtocol
from src.engine.model_manager import ModelRegistry, ModelInfo
from src.utils.workers import BatchPredictWorker, SinglePredictWorker

logger = logging.getLogger(__name__)


class ModelController:
    """Handles model lifecycle: load, delete, import, auto-label."""

    def __init__(self, parent_widget: QWidget):
        self._parent = parent_widget
        self._predictor: PredictorProtocol | None = None
        self._registry: ModelRegistry | None = None
        self._project: ProjectManager | None = None
        self._batch_worker: BatchPredictWorker | None = None

    @property
    def predictor(self) -> PredictorProtocol | None:
        return self._predictor

    @property
    def registry(self) -> ModelRegistry | None:
        return self._registry

    def set_context(self, project: ProjectManager, registry: ModelRegistry) -> None:
        self._project = project
        self._registry = registry

    def unload(self) -> None:
        """Drop the current predictor and free GPU memory.

        Called before enabling a VRAM-heavy backend (e.g. LocateAnything) so a
        resident YOLO model doesn't coexist with it. Invokes the predictor's
        optional ``release()`` hook, drops the reference, then runs gc.

        IMPORTANT (out-of-process LA): ``torch.cuda.empty_cache()`` is only
        invoked **when torch is already imported** in this process. A pure-LA
        session never imports torch in the GUI process (the model lives in a
        sidecar subprocess; ``LocateAnythingPredictor.release()`` just kills that
        subprocess, which frees its own CUDA context). Importing torch here would
        re-introduce the very CUDA-in-the-GUI-process conflict that the sidecar
        architecture exists to avoid, so we must skip it on the LA-only path.
        For the YOLO path torch is already resident, so the cache release still
        happens as before.
        """
        import sys

        predictor = self._predictor
        self._predictor = None
        if predictor is not None:
            release = getattr(predictor, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:  # noqa: BLE001 - teardown must be best-effort
                    logger.debug("Predictor release() failed", exc_info=True)
        import gc

        gc.collect()
        # Only touch CUDA if torch is ALREADY loaded — never import it here.
        if "torch" in sys.modules:
            try:
                torch = sys.modules["torch"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort
                pass

    def set_predictor(self, predictor: PredictorProtocol | None) -> None:
        """Inject a predictor directly (e.g. a LocateAnything runtime that is
        not backed by the ModelRegistry / file path). Existing auto-label flows
        read ``self._predictor`` and work unchanged."""
        self._predictor = predictor

    def load_model(self, model_id: str) -> bool:
        """Load a model for inference. Returns True on success."""
        if not self._registry or not self._project:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        try:
            model_path = Path(model_info.path)
            if not model_path.is_absolute():
                model_path = self._project.project_dir / model_path
            if not model_path.exists():
                QMessageBox.warning(self._parent, "错误", f"模型文件不存在: {model_path}")
                return False
            backend = get_backend(model_info.backend_id)
            self._predictor = backend.load_predictor(model_path, model_info)
            logger.info(
                "Loaded model: %s from %s via %s",
                model_info.name, model_path, backend.backend_id,
            )
            return True
        except (BackendError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.error("Failed to load model: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "加载失败", f"模型加载失败: {e}")
            return False

    def inspect_model_structure(self, model_path: str | Path) -> list[LayerInfo]:
        """Parse a .pt file's layer hierarchy on CPU. Thin passthrough to
        ``core.model_structure.parse_model_structure``. Raises
        ``ModelStructureError`` on bad/corrupt/non-YOLO files — callers show it
        in a QMessageBox."""
        logger.info("Inspecting model structure: %s", model_path)
        return parse_model_structure(model_path)

    def inspect_registered_model(self, model_id: str) -> list[LayerInfo]:
        """Resolve a registered model's path (mirroring ``load_model``'s
        relative→absolute logic) then parse its structure."""
        if not self._registry or not self._project:
            raise ModelStructureError("当前没有打开的项目或模型注册表")
        model_info = self._registry.get(model_id)
        if not model_info:
            raise ModelStructureError("找不到指定的模型")
        model_path = Path(model_info.path)
        if not model_path.is_absolute():
            model_path = self._project.project_dir / model_path
        return self.inspect_model_structure(model_path)

    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry (not file). Returns True if deleted."""
        if not self._registry:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        reply = QMessageBox.question(
            self._parent, "确认删除",
            f"确定要删除模型 \"{model_info.name}\" 吗？\n（模型文件不会被删除）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._registry.remove(model_id)
            self._registry.save()
            return True
        return False

    def rename_model(self, model_id: str) -> bool:
        """Rename a model's display name via dialog. Returns True if renamed."""
        if not self._registry:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        new_name, ok = QInputDialog.getText(
            self._parent, "重命名模型", "请输入新的模型名称:",
            text=model_info.name,
        )
        if not ok or not new_name.strip() or new_name.strip() == model_info.name:
            return False
        self._registry.rename(model_id, new_name.strip())
        self._registry.save()
        logger.info("Renamed model %s -> %s", model_id, new_name.strip())
        return True

    def import_model(self) -> ModelInfo | None:
        """Import an external model file. Returns ModelInfo or None."""
        if not self._registry or not self._project:
            return None
        file_path, _ = QFileDialog.getOpenFileName(
            self._parent, "选择模型文件", "", "模型文件 (*.pt *.onnx);;PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if not file_path:
            return None
        name, ok = QInputDialog.getText(self._parent, "模型名称", "请输入模型名称:")
        if not ok or not name.strip():
            return None
        tasks = ["detect", "classify", "pose"]
        current_task = self._project.config.task_type if self._project else "detect"
        default_idx = tasks.index(current_task) if current_task in tasks else 0
        task, ok = QInputDialog.getItem(
            self._parent, "任务类型", "选择任务类型:", tasks, default_idx, False
        )
        if not ok:
            return None
        p = Path(file_path)
        try:
            rel = p.relative_to(self._project.project_dir)
            model_path = str(rel)
        except ValueError:
            model_path = str(p)
        backend = get_backend()
        probe = backend.probe()
        model_info = ModelInfo(
            name=name.strip(),
            path=model_path,
            task=task,
            base_model="imported",
            classes=self._project.config.classes,
            backend_id=backend.backend_id,
            model_format=backend.infer_model_format(p),
            backend_version=probe.version,
            backend_runtime=probe.runtime,
            backend_metadata=probe.metadata,
        )
        self._registry.register(model_info)
        self._registry.save()
        logger.info("Imported model: %s", name.strip())
        return model_info

    def predict_single(
        self,
        img_path: Path,
        classes: list[str],
        conf: float = 0.5,
        iou: float = 0.45,
        class_match_mode: str = "class_id",
    ) -> list:
        """Run single-image prediction. Returns annotations list."""
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return []
        try:
            return self._predictor.predict(
                img_path,
                conf=conf,
                iou=iou,
                project_classes=classes,
                class_match_mode=class_match_mode,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Auto-label failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "自动标注失败", str(e))
            return []

    def create_single_predict_worker(
        self,
        img_path: Path,
        classes: list[str],
        conf: float = 0.5,
        iou: float = 0.45,
        class_match_mode: str = "class_id",
    ) -> SinglePredictWorker | None:
        """Build a background worker for single-image detection/pose inference.

        Used for slow backends (e.g. LocateAnything) so ``predict`` does not
        block the Qt/X event loop. Returns ``None`` (and warns the user) when no
        predictor is loaded — mirroring ``predict_single``'s guard so the
        caller can bail without special-casing. The caller owns connecting the
        worker's ``done`` / ``error`` signals and starting it.
        """
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return None
        return SinglePredictWorker(
            predictor=self._predictor,
            image_path=img_path,
            conf=conf,
            iou=iou,
            project_classes=classes,
            class_match_mode=class_match_mode,
        )

    def predict_single_classify(
        self,
        img_path: Path,
        classes: list[str],
    ) -> tuple[str, float] | None:
        """Run classify inference, returning (class_name, conf) or None.

        Returns the raw model class name without filtering — the caller
        (MainWindow) routes the result through ProjectController.register_auto_class.
        """
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return None
        try:
            return self._predictor.predict_classify(
                img_path, project_classes=classes, filter_to_project=False,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Classify failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "自动标注失败", str(e))
            return None
