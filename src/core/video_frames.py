"""Qt-free video → frame extraction for the video-import feature.

Videos are never project data themselves: this module samples frames out of
a video file and writes them as plain JPEG images (the caller points it at
the project image dir), after which the frames flow through the ordinary
image pipeline (auto-label / filters / training) with no special casing.

cv2 is a hard ultralytics dependency but is lazy-imported here so a core
import never pays for OpenCV — and, mandatorily, every lazy import is
followed by ``_strip_cv2_qt_plugin_hijack()`` (reused from
``src.core.polygon``): opencv-python's import hijacks
``QT_QPA_PLATFORM_PLUGIN_PATH`` and any QApplication constructed afterwards
aborts (see .trellis/spec/backend/quality-guidelines.md, which names video
frame extraction as a consumer of this exact guard). Unlike
``simplify_polygon`` there is no pure-Python fallback for ``VideoCapture``,
so an ImportError propagates to the caller as a real error.

Decode strategy: sequential ``grab()``/``retrieve()`` — ``grab()`` advances
the decoder cheaply for skipped frames and ``retrieve()`` only decodes the
sampled ones. ``CAP_PROP_POS_FRAMES`` random seeking is deliberately NOT
used (unreliable on some codecs).

Collision policy: a frame file that already exists is skipped (no decode of
that frame, no rewrite) and counted — frame names are deterministic
(``<prefix>_f<original frame index>.jpg``), so same name = same video frame
= same content, and existing frames may already carry annotations. This
matches the silent-skip precedent of image drag-drop import
(``LabelPanel._on_images_dropped``) and makes a cancelled run resumable:
re-importing skips what exists and only fills the gap.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.core.polygon import _strip_cv2_qt_plugin_hijack

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95

# Sampling modes for SamplingParams.mode.
MODE_INTERVAL = "interval"  # take one frame every N frames
MODE_FPS = "fps"            # target output fps, converted via the video's fps


def _lazy_cv2():
    """Import cv2 on demand and immediately undo its Qt plugin-path hijack.

    Mirrors the established guard pattern in ``src.core.polygon`` (spec:
    .trellis/spec/backend/quality-guidelines.md). Raises ImportError when
    opencv is unavailable — video decoding has no pure-Python fallback.
    """
    import cv2

    _strip_cv2_qt_plugin_hijack()
    return cv2


@dataclass(frozen=True)
class SamplingParams:
    """Frame-sampling parameters shared by every video in one import run.

    ``max_frames`` caps the number of SAMPLED frames per video (written +
    skipped-existing). Counting skips against the cap keeps re-runs with the
    same params idempotent: a completed run re-imported skips exactly its own
    frames and writes nothing new, instead of marching deeper into the video.
    """

    mode: str = MODE_INTERVAL
    interval: int = 30          # every N frames (mode == MODE_INTERVAL)
    target_fps: float = 1.0     # output fps (mode == MODE_FPS)
    max_frames: int = 500       # per-video cap on sampled frames

    def effective_interval(self, video_fps: float | None) -> int:
        """Resolve the concrete grab-every-N interval for one video.

        MODE_FPS converts the target fps into an interval using the video's
        own fps. Unknown/invalid video fps degrades to 1 (every frame — the
        ``max_frames`` cap still bounds the run; metadata problems must not
        block import).
        """
        if self.mode == MODE_FPS:
            if not video_fps or video_fps <= 0 or not math.isfinite(video_fps):
                return 1
            if self.target_fps <= 0:
                return 1
            return max(1, round(video_fps / self.target_fps))
        return max(1, int(self.interval))


@dataclass(frozen=True)
class VideoInfo:
    """Probe metadata for one video; ``None`` fields mean "unknown"."""

    path: Path
    ok: bool = False                 # container opened at all
    fps: float | None = None
    frame_count: int | None = None
    duration_s: float | None = None


@dataclass
class ExtractResult:
    """Outcome of extracting one video (never raised — returned)."""

    video: Path
    prefix: str = ""
    written: int = 0
    skipped: int = 0        # frame files that already existed (not rewritten)
    cancelled: bool = False
    failed: bool = False    # could not open / could not decode any frame / write error
    error: str = ""         # readable Chinese message when failed

    @property
    def sampled(self) -> int:
        return self.written + self.skipped


def frame_filename(prefix: str, frame_index: int) -> str:
    """Deterministic frame name: ``<prefix>_f%06d.jpg``.

    ``frame_index`` is the ORIGINAL video frame index (not the output
    ordinal) so the name keeps the source lineage — and stays stable across
    re-runs with different sampling intervals (skip-existing depends on it).
    """
    return f"{prefix}_f{frame_index:06d}.jpg"


def assign_output_prefixes(paths: Sequence[Path | str]) -> list[str]:
    """Assign a unique output prefix per queued video.

    First occurrence of a stem keeps it; later videos with the same stem get
    ``-2``, ``-3``, … suffixes (different files must not share frame-name
    prefixes — same name means same content under the skip policy). Suffixed
    candidates also dodge real stems already in the queue.
    """
    taken: set[str] = set()
    prefixes: list[str] = []
    for p in paths:
        stem = Path(p).stem
        candidate = stem
        n = 1
        while candidate in taken:
            n += 1
            candidate = f"{stem}-{n}"
        taken.add(candidate)
        prefixes.append(candidate)
    return prefixes


def probe_video(path: Path | str) -> VideoInfo:
    """Read container metadata (fps / frame count / duration) for estimates.

    Never raises and never blocks import: unopenable or metadata-less files
    (corrupt, variable frame rate) come back with ``ok=False`` / ``None``
    fields and the dialog shows「未知」.
    """
    path = Path(path)
    try:
        cv2 = _lazy_cv2()
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return VideoInfo(path=path, ok=False)
            fps: float | None = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                fps = None
            raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = raw_count if raw_count > 0 else None
            duration = (
                frame_count / fps if (fps and frame_count) else None
            )
            return VideoInfo(
                path=path, ok=True, fps=fps,
                frame_count=frame_count, duration_s=duration,
            )
        finally:
            cap.release()
    except Exception:
        # Metadata is advisory only — degrade to unknown, never block.
        logger.warning("probe_video failed for %s", path, exc_info=True)
        return VideoInfo(path=path, ok=False)


def estimate_sampled(info: VideoInfo, params: SamplingParams) -> int | None:
    """Uncapped estimate of how many frames sampling would visit.

    ``None`` when the needed metadata is unknown (no frame count; or MODE_FPS
    without a video fps to convert with). The dialog compares this against
    ``max_frames`` for the over-cap warning.
    """
    if not info.frame_count or info.frame_count <= 0:
        return None
    if params.mode == MODE_FPS and not info.fps:
        return None
    interval = params.effective_interval(info.fps)
    return math.ceil(info.frame_count / interval)


def estimate_frames(info: VideoInfo, params: SamplingParams) -> int | None:
    """Estimated frames one extraction run takes (capped at ``max_frames``)."""
    raw = estimate_sampled(info, params)
    if raw is None:
        return None
    return min(raw, params.max_frames)


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    params: SamplingParams,
    prefix: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> ExtractResult:
    """Extract sampled frames from one video into ``output_dir``.

    - Sequential decode: ``grab()`` every frame, ``retrieve()`` + JPEG-write
      (quality ``JPEG_QUALITY``) only for sampled indices whose target file
      does not already exist (existing → counted as skipped, not rewritten).
    - ``progress_cb(written, skipped)`` fires after each sampled frame.
    - ``cancel_event`` is checked every loop iteration; a cancelled run keeps
      the frames already written (``cancelled=True`` in the result).
    - Per-video failures (unopenable file, zero decodable frames, write
      error) are reported via ``failed``/``error`` on the result — the
      caller's queue keeps going. Only ImportError (cv2 missing) propagates.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    prefix = prefix if prefix else video_path.stem
    result = ExtractResult(video=video_path, prefix=prefix)

    cv2 = _lazy_cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            result.failed = True
            result.error = f"无法打开视频: {video_path.name}"
            return result

        fps: float | None = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = None
        interval = params.effective_interval(fps)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        any_grabbed = False
        while True:
            if cancel_event is not None and cancel_event.is_set():
                result.cancelled = True
                break
            if result.sampled >= params.max_frames:
                break
            if not cap.grab():
                break  # end of stream (or decoder gave up on a corrupt tail)
            any_grabbed = True
            if frame_index % interval == 0:
                out_path = output_dir / frame_filename(prefix, frame_index)
                if out_path.exists():
                    result.skipped += 1
                else:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    try:
                        # imencode + write_bytes instead of imwrite: robust
                        # for non-ASCII (Chinese) file names on every OS.
                        encoded, buf = cv2.imencode(
                            ".jpg", frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                        )
                        if not encoded:
                            raise OSError("JPEG 编码失败")
                        out_path.write_bytes(buf.tobytes())
                    except OSError as e:
                        result.failed = True
                        result.error = f"写入帧失败: {out_path.name} ({e})"
                        return result
                    result.written += 1
                if progress_cb is not None:
                    progress_cb(result.written, result.skipped)
            frame_index += 1

        if not any_grabbed and not result.cancelled:
            result.failed = True
            result.error = f"无法读取视频帧: {video_path.name}"
        return result
    finally:
        cap.release()
