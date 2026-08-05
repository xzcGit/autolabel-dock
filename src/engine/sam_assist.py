"""Qt-free prompted-segmentation assistant (MobileSAM via ultralytics 8.2.69).

The module is written against a "prompted segmentation helper" abstraction,
not MobileSAM specifics: prompts are plain data objects (``PointPrompt`` /
``BoxPrompt``), the output is always a normalized polygon + derived bbox
(``MaskResult``). Upgrading ultralytics (multi-point refinement, SAM2) or
swapping the backend (FastSAM, …) replaces only the internals — the
controller/UI faces stay unchanged.

Runtime facts pinned by ``.trellis/tasks/07-12-sam-assisted-polygon/research/
ultralytics-sam-api.md`` (verified against the installed 8.2.69):

- ``ultralytics.models.sam.Predictor`` is the interactive API: ``set_image``
  encodes once (~1 s CPU) and caches; each subsequent ``predictor(points=…)``
  / ``predictor(bboxes=…)`` call only decodes (~65 ms CPU).
- Prompt coordinates are PIXELS in the ORIGINAL image space; ``labels`` 1 =
  foreground. One prompt → one mask (8.2.69 cannot combine clicks).
- Prompted ``Results`` have ``boxes=None`` and no quality scores; the polygon
  comes from ``masks.xyn`` (normalized, one largest external contour per
  mask, typically hundreds of points → MUST be simplified before storing).
- Calling the predictor with NO prompt triggers ``generate()`` — a 32×32
  grid segment-everything sweep. ``predict`` therefore requires a prompt
  object; there is no promptless path.
- Device policy (CPU-only) is an implementation detail of this layer
  (decision 3 in the task PRD): zero VRAM coordination with YOLO/LA.

Heavy imports (ultralytics → torch) happen lazily inside ``load()`` so
importing this module costs nothing; a mock predictor factory is injectable
for tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.core.polygon import bbox_from_polygon, simplify_polygon

logger = logging.getLogger(__name__)

# Preferred pre-placed weight location (fully offline). The filename must be
# exactly "mobile_sam.pt" — ultralytics' build_sam matches by name suffix.
WEIGHT_FILENAME = "mobile_sam.pt"
DEFAULT_WEIGHTS_DIR = Path.home() / ".autolabel" / "models"

_DOWNLOAD_HINT = (
    f"请将 {WEIGHT_FILENAME}（约 39MB，"
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt）"
    f"放置到 {DEFAULT_WEIGHTS_DIR}/ 后重试。"
)


class SamAssistError(Exception):
    """Raised with a user-facing Chinese message when SAM assist fails."""


@dataclass(frozen=True)
class PointPrompt:
    """Single positive click, normalized [0,1] image coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class BoxPrompt:
    """Box prompt, normalized [0,1] corner coordinates (x1<x2, y1<y2)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class MaskResult:
    """One accepted mask: normalized open polygon + derived bbox.

    Invariant: ``bbox == bbox_from_polygon(polygon)`` (the segment task's
    derived-bbox contract — the caller stores both verbatim).
    """

    polygon: list
    bbox: tuple


def resolve_weights_path() -> Path:
    """Return the weight path to hand to ultralytics.

    The pre-placed ``~/.autolabel/models/mobile_sam.pt`` is preferred (fully
    offline). When missing, the same absolute path is returned anyway —
    ultralytics' ``attempt_download_asset`` downloads the GitHub asset to the
    given path; a failed download surfaces as ``SamAssistError`` in
    ``load()`` with a manual-placement hint.
    """
    weights_dir = Path.home() / ".autolabel" / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir / WEIGHT_FILENAME


def _default_predictor_factory(weights_path: Path):
    """Build the real ultralytics SAM predictor (heavy import lives here)."""
    from ultralytics.models.sam import Predictor as SAMPredictor

    return SAMPredictor(
        overrides=dict(
            task="segment",
            mode="predict",
            imgsz=1024,
            model=str(weights_path),
            device="cpu",  # decision 3: CPU-only, no VRAM coordination
            save=False,
            verbose=False,
        )
    )


class SamAssistant:
    """Long-lived prompted-segmentation helper around one SAM predictor.

    Lifecycle: ``load()`` once (builds the predictor; idempotent), then per
    image ``encode(path, size)``; each ``predict(prompt)`` decodes against
    the cached embedding. ``release()`` drops everything.

    All methods are synchronous — threading lives in the controller.
    """

    def __init__(self, predictor_factory=None, weights_path: Path | None = None):
        self._predictor_factory = predictor_factory or _default_predictor_factory
        self._weights_path = weights_path
        self._predictor = None
        self._image_size: tuple[int, int] | None = None  # (w, h) of encoded image
        self._encoded_path: Path | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def encoded_path(self) -> Path | None:
        """Path of the currently-encoded image (None before any encode)."""
        return self._encoded_path

    def load(self) -> None:
        """Build the predictor (no-op when already loaded).

        Raises SamAssistError with an actionable Chinese message when the
        weights are missing and cannot be downloaded, or construction fails.
        """
        if self._predictor is not None:
            return
        weights = self._weights_path or resolve_weights_path()
        try:
            self._predictor = self._predictor_factory(weights)
        except Exception as e:  # noqa: BLE001 — surface any load failure readably
            logger.exception("SAM predictor load failed")
            raise SamAssistError(
                f"SAM 模型加载失败：{e}\n{_DOWNLOAD_HINT}"
            ) from e
        logger.info("SAM assistant loaded (weights=%s, device=cpu)", weights)

    def encode(self, image_path: Path | str, image_size: tuple[int, int]) -> None:
        """Encode one image (≈1 s CPU); subsequent predicts reuse the embedding.

        ``image_size`` is the original (w, h) — the caller reads it (the
        engine stays free of Qt/PIL); it converts normalized prompts to the
        pixel space SAM expects and drives pixel-space simplification.
        """
        if self._predictor is None:
            raise SamAssistError("SAM 模型尚未加载")
        w, h = int(image_size[0]), int(image_size[1])
        if w <= 0 or h <= 0:
            raise SamAssistError(f"无法读取图像尺寸: {image_path}")
        path = Path(image_path)
        try:
            # The 8.2.69 predictor lazily builds its model inside set_image on
            # first use, then caches self.im + self.features.
            self._predictor.set_image(str(path))
        except Exception as e:  # noqa: BLE001
            logger.exception("SAM encode failed: %s", path)
            self._encoded_path = None
            self._image_size = None
            raise SamAssistError(f"SAM 图像编码失败：{e}") from e
        self._encoded_path = path
        self._image_size = (w, h)

    def reset(self) -> None:
        """Clear the cached image embedding (image switched away)."""
        if self._predictor is not None:
            try:
                self._predictor.reset_image()
            except Exception:  # noqa: BLE001 — best-effort cache clear
                logger.exception("SAM reset_image failed")
        self._encoded_path = None
        self._image_size = None

    def release(self) -> None:
        """Drop the predictor (project close / shutdown); gc frees the model."""
        self._predictor = None
        self._encoded_path = None
        self._image_size = None

    # ── Prompted prediction ───────────────────────────────────

    def predict(self, prompt: PointPrompt | BoxPrompt) -> MaskResult | None:
        """Run one prompt against the encoded image.

        Returns ``None`` when the model produced no usable mask (empty mask,
        degenerate contour, simplification below 3 vertices) — a soft "no
        object here", not an error. Raises SamAssistError on real failures
        (not loaded / not encoded / inference exception / bad prompt type).

        A prompt object is REQUIRED: calling the underlying predictor with no
        prompt would trigger the segment-everything grid sweep (research §6).
        """
        if self._predictor is None:
            raise SamAssistError("SAM 模型尚未加载")
        if self._encoded_path is None or self._image_size is None:
            raise SamAssistError("SAM 尚未完成当前图像编码，请稍候")

        w, h = self._image_size
        if isinstance(prompt, PointPrompt):
            # Pixel coords in original image space; label 1 = foreground.
            kwargs = dict(points=[prompt.x * w, prompt.y * h], labels=[1])
        elif isinstance(prompt, BoxPrompt):
            kwargs = dict(
                bboxes=[prompt.x1 * w, prompt.y1 * h, prompt.x2 * w, prompt.y2 * h]
            )
        else:
            raise SamAssistError(f"未知的 SAM prompt 类型: {type(prompt).__name__}")

        try:
            results = self._predictor(**kwargs)
        except Exception as e:  # noqa: BLE001 — inference failure → readable error
            logger.exception("SAM inference failed")
            raise SamAssistError(f"SAM 推理失败：{e}") from e

        return self._result_to_mask(results, (w, h))

    @staticmethod
    def _result_to_mask(results, image_size: tuple[int, int]) -> MaskResult | None:
        """Extract masks.xyn[0] → simplify (pixel space) → polygon + bbox.

        Mirrors the YOLO-seg ingest path in ``engine/predictor.py``: raw
        contours carry hundreds of points, so simplification runs BEFORE the
        polygon is stored anywhere. Prompted Results have ``boxes=None`` —
        the bbox is always derived from the polygon (segment invariant).
        """
        if not results:
            return None
        masks = getattr(results[0], "masks", None)
        if masks is None:
            return None
        xyn = getattr(masks, "xyn", None)
        if not xyn or len(xyn) == 0:
            return None
        contour = xyn[0]
        points = [[float(p[0]), float(p[1])] for p in contour]
        if len(points) < 3:
            return None
        polygon = simplify_polygon(points, image_size=image_size)
        if len(polygon) < 3:
            return None
        bbox = bbox_from_polygon(polygon)
        if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
            return None
        return MaskResult(polygon=polygon, bbox=bbox)
