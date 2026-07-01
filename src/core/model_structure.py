"""Model structure inspection — parse a YOLO model into top-level layer info.

Qt-free core layer. Loads a YOLO model on CPU only and walks the top-level
modules of ``model.model`` (the ``nn.Sequential``) so the reported layer index
matches Ultralytics ``freeze=N`` semantics (``freeze=N`` freezes
``model.0.``..``model.{N-1}.``).

torch / ultralytics are imported lazily *inside* the loader function so this
module stays importable without a torch install (mirrors the zero-torch import
discipline used elsewhere in the project).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelStructureError(Exception):
    """Raised when a model file cannot be parsed into layer structure.

    Carries a friendly Chinese message suitable for display in a QMessageBox.
    """


@dataclass
class LayerInfo:
    """One top-level module of a YOLO model's ``model.model`` Sequential."""

    index: int           # 层索引（对应 freeze 参数）
    module_type: str     # 短类型名，如 "Conv" / "C2f" / "SPPF" / "Detect"
    params: int          # 该层参数量
    params_ratio: float  # 累计参数占比 (0.0 ~ 1.0)
    output_shape: str    # 如 "[1, 64, 160, 160]"，无法确定时为 "-"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "module_type": self.module_type,
            "params": self.params,
            "params_ratio": self.params_ratio,
            "output_shape": self.output_shape,
        }


def _short_type(module) -> str:
    """Derive a short display type name for a top-level module.

    Prefers the parsed ``.type`` attribute (a full class path string like
    ``ultralytics.nn.modules.conv.Conv``) written by ultralytics' parse_model;
    falls back to the Python class name.
    """
    type_str = getattr(module, "type", None)
    if isinstance(type_str, str) and type_str:
        return type_str.rsplit(".", 1)[-1]
    return type(module).__name__


def _module_params(module) -> int:
    """Parameter count for a module.

    Uses the parsed ``.np`` attribute when present (set by parse_model),
    otherwise sums ``parameters()``.
    """
    np_val = getattr(module, "np", None)
    if isinstance(np_val, int):
        return np_val
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:  # noqa: BLE001 - defensive; fall back to 0
        return 0


def _format_shape(output) -> str:
    """Render a module's forward output as a shape string, or '-' if unknown."""
    shape = getattr(output, "shape", None)
    if shape is not None:
        return "[" + ", ".join(str(int(d)) for d in shape) + "]"
    # Detect / multi-branch heads return a list/tuple of tensors.
    if isinstance(output, (list, tuple)):
        for item in output:
            item_shape = getattr(item, "shape", None)
            if item_shape is not None:
                return "[" + ", ".join(str(int(d)) for d in item_shape) + "] (多输出)"
    return "-"


def _layers_from_model(model, imgsz: int = 640) -> list[LayerInfo]:
    """Build ``LayerInfo`` list from an already-loaded ultralytics model object.

    ``model`` is the ``ultralytics.YOLO`` wrapper (has a ``.model`` nn.Module
    whose ``.model`` attribute is the top-level ``nn.Sequential``). Separated
    from :func:`parse_model_structure` so the pure-parsing logic is unit-testable
    without a real .pt file or GPU.

    The forward pass (for output shapes) is best-effort: any failure sets every
    ``output_shape`` to "-" while still returning index/type/params data.
    """
    inner = getattr(model, "model", None)
    seq = getattr(inner, "model", None)
    if seq is None or not hasattr(seq, "__iter__"):
        raise ModelStructureError(
            "无法解析模型结构：这不是一个标准的 YOLO 模型（缺少 model.model 层级）。"
        )

    top_modules = list(seq)
    if not top_modules:
        raise ModelStructureError("无法解析模型结构：模型层级为空。")

    total_params = 0
    try:
        total_params = int(sum(p.numel() for p in inner.parameters()))
    except Exception:  # noqa: BLE001 - fall back to summing per-module below
        total_params = 0
    if total_params <= 0:
        total_params = sum(_module_params(m) for m in top_modules)

    # ── Best-effort forward pass for output shapes ──
    shapes: dict[int, str] = {}
    _capture_output_shapes(model, inner, top_modules, imgsz, shapes)

    layers: list[LayerInfo] = []
    cumulative = 0
    for i, module in enumerate(top_modules):
        # Prefer the parsed .i attribute; fall back to positional index.
        idx = getattr(module, "i", None)
        if not isinstance(idx, int):
            idx = i
        params = _module_params(module)
        cumulative += params
        ratio = (cumulative / total_params) if total_params > 0 else 0.0
        layers.append(
            LayerInfo(
                index=idx,
                module_type=_short_type(module),
                params=params,
                params_ratio=ratio,
                output_shape=shapes.get(i, "-"),
            )
        )
    return layers


def _capture_output_shapes(model, inner, top_modules, imgsz, shapes) -> None:
    """Run one CPU forward pass with hooks to capture per-module output shapes.

    YOLO has skip connections (routed via each module's ``.f`` attribute), so a
    naive sequential forward is wrong — we run the model's real forward with
    hooks attached to each top-level module. Fully best-effort: on ANY failure
    ``shapes`` is left partially/entirely empty and callers render "-". Hooks are
    always removed in the finally block.
    """
    handles = []
    try:
        import torch  # lazy — keep this module importable without torch

        def _make_hook(pos: int):
            def _hook(_mod, _inp, output):
                shapes[pos] = _format_shape(output)
            return _hook

        for pos, module in enumerate(top_modules):
            if hasattr(module, "register_forward_hook"):
                handles.append(module.register_forward_hook(_make_hook(pos)))

        inner.eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            inner(dummy)
    except Exception:  # noqa: BLE001 - shapes are optional per PRD
        logger.debug("Forward pass for output shapes failed; shapes unavailable", exc_info=True)
        shapes.clear()
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass


def parse_model_structure(model_path: str | Path, imgsz: int = 640) -> list[LayerInfo]:
    """Load a YOLO model on CPU and return its top-level layer structure.

    Args:
        model_path: Path to a ``.pt`` YOLO weights file.
        imgsz: Square input size used for the best-effort forward pass.

    Returns:
        List of :class:`LayerInfo`, one per top-level module of ``model.model``.

    Raises:
        ModelStructureError: file missing, unloadable/corrupt, or not a YOLO
            model. The message is a friendly Chinese string.
    """
    path = Path(model_path)
    # A bare filename with no directory component (e.g. "yolov8n.pt") is a
    # pretrained weight name that ultralytics resolves from its cache or
    # downloads — don't reject it on a local-existence check. Only paths that
    # explicitly point at a location must exist.
    if path.parent != Path(".") and not path.exists():
        raise ModelStructureError(f"模型文件不存在：{path}")

    try:
        from ultralytics import YOLO  # lazy import (torch/ultralytics)
    except Exception as e:  # noqa: BLE001
        raise ModelStructureError(f"无法导入 ultralytics，无法解析模型：{e}") from e

    try:
        model = YOLO(str(path))
    except Exception as e:  # noqa: BLE001 - corrupt / non-YOLO file
        logger.error("Failed to load model %s: %s", path, e, exc_info=True)
        raise ModelStructureError(
            f"无法加载模型文件（可能已损坏或不是有效的 YOLO 模型）：{e}"
        ) from e

    # Force CPU — never touch GPU VRAM.
    try:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "cpu"):
            inner.cpu()
    except Exception:  # noqa: BLE001 - best-effort; parsing still proceeds on CPU
        logger.debug("Failed to move model to CPU explicitly", exc_info=True)

    return _layers_from_model(model, imgsz=imgsz)
