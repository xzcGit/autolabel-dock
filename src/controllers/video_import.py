"""Video-import controller — QThread worker lifecycle + data-only signal face.

``VideoImportController`` owns the background extraction worker for the
video → frames import flow, modeled on ``AutoLabelController``'s batch face
(``batch_started/progress/finished`` + terminal-once guarantee in the
QThread-``finished`` slot). MainWindow keeps only the ``BatchProgressDialog``
shell: it builds the dialog on ``started``, updates it on
``progress``/``video_progress``, and closes it on the single terminal
``finished(summary)`` — which fires exactly once for every outcome
(completed / failed / cancelled).

The worker mirrors ``BatchPredictWorker`` (src/utils/workers.py):
``threading.Event`` cancellation checked per frame inside the core extractor,
broad exception catch → ``error(str)`` (uncaught exceptions in QThread
silently kill the thread), and ``finished_ok`` NOT emitted on cancel. Unlike
batch auto-label there is no shared mutable label state, so the worker writes
its frame files itself (plain new images); per-video failures (unopenable /
corrupt) become failed ``ExtractResult`` entries and the queue continues.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from src.core.video_frames import (
    ExtractResult,
    SamplingParams,
    assign_output_prefixes,
    extract_frames,
)

logger = logging.getLogger(__name__)

# Sampled frames between throttled video_progress emissions (queued signals;
# an unthrottled interval=1 run over a long video would flood the event loop).
_PROGRESS_EVERY = 25


def summarize_results(results: list[ExtractResult], cancelled: bool = False) -> str:
    """Build the terminal Chinese summary (written / skipped / failed counts)."""
    written = sum(r.written for r in results)
    skipped = sum(r.skipped for r in results)
    failed = sum(1 for r in results if r.failed)
    parts = [f"已写入 {written} 帧", f"跳过 {skipped} 帧"]
    if failed:
        parts.append(f"{failed} 个视频失败")
    head = "视频抽帧已取消" if cancelled else "视频抽帧完成"
    return f"{head}（{'，'.join(parts)}）"


class VideoFrameExtractWorker(QThread):
    """Extracts frames from a queue of videos in a background thread.

    Signals:
        progress(int, int): (videos_done, total_videos) after each video.
        video_progress(str, int, int): (video_name, written, skipped) — live
            counts for the in-flight video, throttled to every
            ``_PROGRESS_EVERY`` sampled frames plus one final emission per
            video.
        finished_ok(object): list[ExtractResult] when the queue completed
            un-cancelled (NOT emitted on cancel — BatchPredictWorker
            semantics; QThread's built-in ``finished`` is the always-fires
            signal the controller uses for cancel detection).
        error(str): a failure that aborts the whole queue (e.g. cv2 missing).

    Partial results survive cancellation via ``self.results`` (safe to read
    once QThread ``finished`` has been delivered on the main thread); frames
    already written stay on disk — re-importing skips them and resumes.
    """

    progress = pyqtSignal(int, int)
    video_progress = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        videos: list[Path],
        output_dir: Path,
        params: SamplingParams,
        parent=None,
    ):
        super().__init__(parent)
        self._videos = [Path(v) for v in videos]
        self._output_dir = Path(output_dir)
        self._params = params
        self._cancelled = threading.Event()
        self.results: list[ExtractResult] = []

    def cancel(self) -> None:
        """Request cancellation (checked per frame inside the extractor)."""
        self._cancelled.set()

    def run(self) -> None:
        try:
            prefixes = assign_output_prefixes(self._videos)
            total = len(self._videos)
            for i, (video, prefix) in enumerate(zip(self._videos, prefixes)):
                if self._cancelled.is_set():
                    break
                res = self._extract_one(video, prefix)
                self.results.append(res)
                # Final per-video emission so the last counts always land.
                self.video_progress.emit(video.name, res.written, res.skipped)
                if res.cancelled:
                    break
                self.progress.emit(i + 1, total)
            if not self._cancelled.is_set():
                self.finished_ok.emit(list(self.results))
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("Video frame extraction worker failed")
            self.error.emit(str(e))

    def _extract_one(self, video: Path, prefix: str) -> ExtractResult:
        """Run one video through the core extractor with throttled progress.

        Per-video fault tolerance: an unexpected exception becomes a failed
        ``ExtractResult`` so the queue continues (mirrors batch predict's
        per-image tolerance). ImportError (cv2 missing) is re-raised — no
        video in the queue could succeed, so it aborts via ``error``.
        """
        sampled_count = 0

        def on_frame(written: int, skipped: int) -> None:
            nonlocal sampled_count
            sampled_count += 1
            if sampled_count % _PROGRESS_EVERY == 0:
                self.video_progress.emit(video.name, written, skipped)

        try:
            return extract_frames(
                video, self._output_dir, self._params,
                prefix=prefix, progress_cb=on_frame,
                cancel_event=self._cancelled,
            )
        except ImportError:
            raise
        except Exception as e:
            logger.exception("Frame extraction failed: %s", video)
            return ExtractResult(
                video=video, prefix=prefix, failed=True, error=str(e),
            )


class VideoImportController(QObject):
    """Orchestrates video-frame extraction runs (one at a time).

    Signals (data-only face — MainWindow renders them):
        started(int): a run started with ``total`` videos — build the
            progress dialog.
        progress(int, int): (videos_done, total_videos).
        video_progress(str, int, int): (video_name, written, skipped) — live
            detail for the in-flight video.
        finished(str): terminal Chinese summary, exactly once per run for
            every outcome (completed / failed / cancelled) — close the
            dialog, show the text, then rescan the image list.
        error(str): failure detail accompanying a failed run (emitted after
            ``finished``, mirroring AutoLabelController's error ordering).
    """

    started = pyqtSignal(int)
    progress = pyqtSignal(int, int)
    video_progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: VideoFrameExtractWorker | None = None
        # True once a terminal summary (completed / failed) was emitted for
        # the current run; the worker-finished slot uses it to detect the
        # cancel path (neither finished_ok nor error fired).
        self._summary_emitted = False

    @property
    def is_running(self) -> bool:
        """True while a run's worker exists (cleared in the finished slot)."""
        return self._worker is not None

    def start_import(
        self,
        videos: list[Path],
        output_dir: Path | str,
        params: SamplingParams,
    ) -> bool:
        """Start extracting ``videos`` into ``output_dir`` (the project image
        dir — resolved by the caller, mirroring ``ProjectManager.list_images``).

        Returns False without side effects when a run is already in flight
        (re-entrancy guard) or the queue is empty.
        """
        if self._worker is not None:
            return False
        video_paths = [Path(v) for v in videos]
        if not video_paths:
            return False
        self._summary_emitted = False
        # No Qt parent — matching the batch auto-label pattern: the controller
        # holds the only reference while the run is in flight and drops it in
        # the worker-finished slot, letting the finished QThread be collected.
        worker = VideoFrameExtractWorker(video_paths, Path(output_dir), params)
        worker.progress.connect(self.progress)
        worker.video_progress.connect(self.video_progress)
        worker.finished_ok.connect(self._on_finished_ok)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()
        # Emitted after start() to mirror the batch auto-label choreography
        # (worker signals are queued, so the dialog exists before the first).
        self.started.emit(len(video_paths))
        return True

    def cancel(self) -> None:
        """Request cancellation of the running extraction (dialog button)."""
        if self._worker is not None:
            self._worker.cancel()

    def shutdown(self, timeout_ms: int = 30000) -> None:
        """Cancel and wait out any in-flight run (app close). Cancellation is
        checked per frame, so this returns quickly."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(timeout_ms)

    # ── Worker slots (queued → main thread) ──────────────────────

    def _on_finished_ok(self, results) -> None:
        self._summary_emitted = True
        self.finished.emit(summarize_results(list(results), cancelled=False))

    def _on_error(self, message: str) -> None:
        self._summary_emitted = True
        # Terminal first (dialog closes), then the detail for the warning box
        # — same ordering as AutoLabelController._on_batch_error.
        self.finished.emit("视频抽帧失败")
        self.error.emit(message)

    def _on_worker_finished(self) -> None:
        """Worker ``finished`` (always fires): drop the ref + cancel cleanup."""
        worker = self._worker
        self._worker = None
        # Finishing without finished_ok/error means the run was cancelled —
        # surface the terminal summary (with the partial counts the worker
        # accumulated) so MainWindow closes the progress dialog.
        if not self._summary_emitted:
            self._summary_emitted = True
            results = list(worker.results) if worker is not None else []
            self.finished.emit(summarize_results(results, cancelled=True))
