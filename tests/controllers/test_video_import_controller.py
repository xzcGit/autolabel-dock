"""Tests for VideoImportController — worker lifecycle + terminal-once face.

Drives the controller with a fake worker (via monkeypatch) so no cv2 /
QThread is needed for the signal-face assertions, plus one real end-to-end
run over a synthesized video. Mirrors the AutoLabelController batch-face
contract: started-after-start ordering, single terminal ``finished`` for
every outcome (completed / failed / cancelled), and per-video fault
tolerance folded into the summary.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QCoreApplication, QObject, pyqtSignal

from src.controllers.video_import import (
    VideoImportController,
    summarize_results,
)
from src.core.video_frames import ExtractResult, MODE_INTERVAL, SamplingParams


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QCoreApplication.processEvents()
    return predicate()


def _record(signal, into):
    signal.connect(lambda *args: into.append(args))


class _FakeWorker(QObject):
    """Stand-in for VideoFrameExtractWorker; emits scripted signals on start."""

    progress = pyqtSignal(int, int)
    video_progress = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    script = None  # class attr set per test: "ok" | "error" | "cancel"
    results_payload: list = []

    def __init__(self, videos, output_dir, params, parent=None):
        super().__init__(parent)
        self.videos = list(videos)
        self.output_dir = output_dir
        self.params = params
        self.results = list(type(self).results_payload)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def isRunning(self):
        return False

    def wait(self, timeout):
        return True

    def start(self):
        if type(self).script == "ok":
            self.finished_ok.emit(list(self.results))
        elif type(self).script == "error":
            self.error.emit("boom")
        # "cancel": emit neither → finished-only (cancel detection)
        self.finished.emit()


@pytest.fixture
def patch_worker(monkeypatch):
    def _install(script="ok", results_payload=None):
        _FakeWorker.script = script
        _FakeWorker.results_payload = results_payload or []
        monkeypatch.setattr(
            "src.controllers.video_import.VideoFrameExtractWorker", _FakeWorker,
        )
    return _install


class TestSummarize:
    def test_counts_written_skipped_failed(self):
        results = [
            ExtractResult(video=Path("a.mp4"), written=5, skipped=1),
            ExtractResult(video=Path("b.mp4"), written=0, skipped=0, failed=True),
        ]
        summary = summarize_results(results)
        assert "已写入 5 帧" in summary
        assert "跳过 1 帧" in summary
        assert "1 个视频失败" in summary
        assert summary.startswith("视频抽帧完成")

    def test_cancelled_header(self):
        results = [ExtractResult(video=Path("a.mp4"), written=2)]
        assert summarize_results(results, cancelled=True).startswith("视频抽帧已取消")

    def test_no_failures_omits_failed_clause(self):
        summary = summarize_results([ExtractResult(video=Path("a.mp4"), written=3)])
        assert "失败" not in summary


class TestControllerLifecycle:
    def test_ok_run_emits_started_then_single_finished(self, qapp, patch_worker):
        patch_worker(
            script="ok",
            results_payload=[ExtractResult(video=Path("a.mp4"), written=4, skipped=1)],
        )
        ctrl = VideoImportController()
        started, finished, errors = [], [], []
        _record(ctrl.started, started)
        _record(ctrl.finished, finished)
        _record(ctrl.error, errors)

        ok = ctrl.start_import(
            [Path("a.mp4")], Path("/out"),
            SamplingParams(mode=MODE_INTERVAL, interval=10),
        )
        assert ok is True
        assert _wait_until(lambda: len(finished) == 1)

        assert started == [(1,)]
        assert finished[0][0].startswith("视频抽帧完成")
        assert "已写入 4 帧" in finished[0][0]
        assert errors == []
        assert ctrl.is_running is False  # ref cleared in finished slot

    def test_empty_queue_is_noop(self, qapp, patch_worker):
        patch_worker(script="ok")
        ctrl = VideoImportController()
        started = []
        _record(ctrl.started, started)
        assert ctrl.start_import([], Path("/out"), SamplingParams()) is False
        assert started == []

    def test_reentrancy_guard_blocks_second_run(self, qapp, monkeypatch):
        # A worker that stays "running" (never emits finished) so the ref lingers.
        class _StuckWorker(_FakeWorker):
            def start(self):
                pass  # no terminal signal

        monkeypatch.setattr(
            "src.controllers.video_import.VideoFrameExtractWorker", _StuckWorker,
        )
        ctrl = VideoImportController()
        assert ctrl.start_import([Path("a.mp4")], Path("/o"), SamplingParams()) is True
        assert ctrl.is_running is True
        # Second call refused while the first worker is still referenced.
        assert ctrl.start_import([Path("b.mp4")], Path("/o"), SamplingParams()) is False

    def test_error_run_emits_finished_then_error(self, qapp, patch_worker):
        patch_worker(script="error")
        ctrl = VideoImportController()
        finished, errors = [], []
        _record(ctrl.finished, finished)
        _record(ctrl.error, errors)

        ctrl.start_import([Path("a.mp4")], Path("/out"), SamplingParams())
        assert _wait_until(lambda: len(errors) == 1)

        assert finished == [("视频抽帧失败",)]
        assert errors == [("boom",)]
        assert ctrl.is_running is False

    def test_cancel_path_emits_cancelled_summary_once(self, qapp, patch_worker):
        # Worker emits finished only (no finished_ok/error) with partial results.
        patch_worker(
            script="cancel",
            results_payload=[ExtractResult(video=Path("a.mp4"), written=2, skipped=0)],
        )
        ctrl = VideoImportController()
        finished = []
        _record(ctrl.finished, finished)

        ctrl.start_import([Path("a.mp4")], Path("/out"), SamplingParams())
        assert _wait_until(lambda: len(finished) == 1)

        assert len(finished) == 1
        assert finished[0][0].startswith("视频抽帧已取消")
        assert "已写入 2 帧" in finished[0][0]

    def test_cancel_forwards_to_worker(self, qapp, monkeypatch):
        class _StuckWorker(_FakeWorker):
            def start(self):
                pass

        monkeypatch.setattr(
            "src.controllers.video_import.VideoFrameExtractWorker", _StuckWorker,
        )
        ctrl = VideoImportController()
        ctrl.start_import([Path("a.mp4")], Path("/o"), SamplingParams())
        ctrl.cancel()
        assert ctrl._worker.cancelled is True


class TestControllerEndToEnd:
    def test_real_extraction_writes_frames_and_summarizes(
        self, qapp, make_video, tmp_path,
    ):
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "images"
        ctrl = VideoImportController()
        finished = []
        _record(ctrl.finished, finished)

        ctrl.start_import(
            [video], out,
            SamplingParams(mode=MODE_INTERVAL, interval=10, max_frames=100),
        )
        assert _wait_until(lambda: len(finished) == 1)

        assert finished[0][0].startswith("视频抽帧完成")
        assert "已写入 3 帧" in finished[0][0]
        assert sorted(p.name for p in out.iterdir()) == [
            "clip_f000000.jpg", "clip_f000010.jpg", "clip_f000020.jpg",
        ]

    def test_bad_video_in_queue_counted_and_queue_continues(
        self, qapp, make_video, tmp_path,
    ):
        good = make_video("good.mp4", frames=10)
        bad = tmp_path / "bad.mp4"
        bad.write_text("not a video")
        out = tmp_path / "images"
        ctrl = VideoImportController()
        finished, progress = [], []
        _record(ctrl.finished, finished)
        _record(ctrl.progress, progress)

        ctrl.start_import(
            [bad, good], out,
            SamplingParams(mode=MODE_INTERVAL, interval=5, max_frames=100),
        )
        assert _wait_until(lambda: len(finished) == 1)

        # Bad video is counted as failed; good one still produced frames.
        assert "1 个视频失败" in finished[0][0]
        assert "已写入 2 帧" in finished[0][0]
        assert len(list(out.iterdir())) == 2  # good.mp4 frames f0 / f5
        # Both videos advanced the (videos_done, total) progress counter.
        assert progress[-1] == (2, 2)

    def test_same_stem_videos_get_distinct_prefixes(
        self, qapp, make_video, tmp_path,
    ):
        v1 = make_video("v.mp4", frames=4, directory=tmp_path / "d1")
        v2 = make_video("v.mp4", frames=4, directory=tmp_path / "d2")
        out = tmp_path / "images"
        ctrl = VideoImportController()
        finished = []
        _record(ctrl.finished, finished)

        ctrl.start_import(
            [v1, v2], out,
            SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=2),
        )
        assert _wait_until(lambda: len(finished) == 1)

        assert sorted(p.name for p in out.iterdir()) == [
            "v-2_f000000.jpg", "v-2_f000001.jpg",
            "v_f000000.jpg", "v_f000001.jpg",
        ]

    def test_cancel_mid_run_keeps_partial_frames(
        self, qapp, tmp_path, monkeypatch,
    ):
        """Deterministic cancel: a fake extractor blocks on the shared cancel
        event, so ``cancel()`` always lands mid-video."""
        import threading

        from src.core.video_frames import ExtractResult as _ER

        extraction_started = threading.Event()

        def fake_extract(video, out, params, prefix=None,
                         progress_cb=None, cancel_event=None):
            extraction_started.set()
            cancel_event.wait(10)  # park until the test cancels
            return _ER(
                video=Path(video), prefix=prefix or "",
                written=3, skipped=1, cancelled=True,
            )

        monkeypatch.setattr(
            "src.controllers.video_import.extract_frames", fake_extract,
        )
        ctrl = VideoImportController()
        finished = []
        _record(ctrl.finished, finished)

        ctrl.start_import(
            [tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path, SamplingParams(),
        )
        assert extraction_started.wait(5)
        ctrl.cancel()
        assert _wait_until(lambda: len(finished) == 1)

        # Cancelled summary carries the partial counts; second video untouched.
        assert finished == [("视频抽帧已取消（已写入 3 帧，跳过 1 帧）",)]
        assert ctrl.is_running is False

    def test_unexpected_per_video_exception_becomes_failed_result(
        self, qapp, tmp_path, monkeypatch,
    ):
        """A non-ImportError exception inside one video's extraction must not
        abort the queue — it folds into a failed result (per-video tolerance,
        mirroring batch predict)."""
        from src.core.video_frames import ExtractResult as _ER

        calls = []

        def fake_extract(video, out, params, prefix=None,
                         progress_cb=None, cancel_event=None):
            video = Path(video)
            calls.append(video.name)
            if video.name == "boom.mp4":
                raise RuntimeError("kaboom")
            return _ER(video=video, prefix=prefix or "", written=1)

        monkeypatch.setattr(
            "src.controllers.video_import.extract_frames", fake_extract,
        )
        ctrl = VideoImportController()
        finished, errors = [], []
        _record(ctrl.finished, finished)
        _record(ctrl.error, errors)

        ctrl.start_import(
            [tmp_path / "a.mp4", tmp_path / "boom.mp4", tmp_path / "c.mp4"],
            tmp_path, SamplingParams(),
        )
        assert _wait_until(lambda: len(finished) == 1)

        assert calls == ["a.mp4", "boom.mp4", "c.mp4"]  # queue continued
        assert finished == [("视频抽帧完成（已写入 2 帧，跳过 0 帧，1 个视频失败）",)]
        assert errors == []

    def test_import_error_aborts_queue_via_error_signal(
        self, qapp, tmp_path, monkeypatch,
    ):
        """cv2 missing → nothing can succeed: whole run fails through the
        error path (terminal finished first, then the error detail)."""
        def fake_extract(*args, **kwargs):
            raise ImportError("No module named 'cv2'")

        monkeypatch.setattr(
            "src.controllers.video_import.extract_frames", fake_extract,
        )
        ctrl = VideoImportController()
        finished, errors = [], []
        _record(ctrl.finished, finished)
        _record(ctrl.error, errors)

        ctrl.start_import([tmp_path / "a.mp4"], tmp_path, SamplingParams())
        assert _wait_until(lambda: len(errors) == 1)

        assert finished == [("视频抽帧失败",)]
        assert errors == [("No module named 'cv2'",)]


class TestShutdown:
    def test_shutdown_cancels_and_waits_running_worker(self, qapp):
        from unittest.mock import MagicMock

        ctrl = VideoImportController()
        fake = MagicMock()
        fake.isRunning.return_value = True
        ctrl._worker = fake

        ctrl.shutdown(1234)

        fake.cancel.assert_called_once()
        fake.wait.assert_called_once_with(1234)

    def test_shutdown_noop_without_worker(self, qapp):
        VideoImportController().shutdown(100)  # must not raise
