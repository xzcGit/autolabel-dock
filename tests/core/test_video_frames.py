"""Tests for src/core/video_frames.py — Qt-free video → frame extraction.

Runs against real tiny synthesized videos (``make_video`` fixture, skipped
when no mp4 codec is available); the pure param/naming/prefix logic needs no
cv2 at all.
"""
import os
import threading
from pathlib import Path

import pytest

from src.core.video_frames import (
    MODE_FPS,
    MODE_INTERVAL,
    SamplingParams,
    VideoInfo,
    assign_output_prefixes,
    estimate_frames,
    estimate_sampled,
    extract_frames,
    frame_filename,
    probe_video,
)


def test_module_is_qt_free():
    """video_frames.py must not drag in Qt (unit-testable without QApplication)."""
    import src.core.video_frames as mod

    src_text = open(mod.__file__, encoding="utf-8").read()
    assert "PyQt5" not in src_text
    assert "src.ui" not in src_text
    assert not any(
        name.startswith("PyQt5") for name in getattr(mod, "__dict__", {})
    )


def test_cv2_import_does_not_hijack_qt_plugin_path(make_video, tmp_path):
    """Import-order regression (spec: backend/quality-guidelines.md): the cv2
    path must strip QT_QPA_PLATFORM_PLUGIN_PATH, or a QApplication constructed
    after an extraction (e.g. a pytest subset running this file before the
    first qapp test) aborts with 'Fatal Python error: Aborted'."""
    video = make_video("hijack.mp4", frames=5)
    extract_frames(
        video, tmp_path / "out",
        SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=2),
    )
    leftover = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    assert "cv2" not in leftover and "opencv" not in leftover


class TestSamplingParams:
    def test_interval_mode_returns_interval(self):
        p = SamplingParams(mode=MODE_INTERVAL, interval=12)
        assert p.effective_interval(30.0) == 12
        assert p.effective_interval(None) == 12  # fps irrelevant in this mode

    def test_interval_clamped_to_one(self):
        assert SamplingParams(mode=MODE_INTERVAL, interval=0).effective_interval(30.0) == 1

    def test_fps_mode_converts_via_video_fps(self):
        p = SamplingParams(mode=MODE_FPS, target_fps=2.0)
        assert p.effective_interval(30.0) == 15

    def test_fps_mode_unknown_video_fps_degrades_to_every_frame(self):
        """Unreliable metadata must not block import — degrade, cap protects."""
        p = SamplingParams(mode=MODE_FPS, target_fps=2.0)
        assert p.effective_interval(None) == 1
        assert p.effective_interval(0.0) == 1

    def test_fps_target_above_video_fps_clamps_to_one(self):
        p = SamplingParams(mode=MODE_FPS, target_fps=120.0)
        assert p.effective_interval(30.0) == 1


class TestFrameFilename:
    def test_format_uses_six_digit_original_index(self):
        assert frame_filename("clip", 0) == "clip_f000000.jpg"
        assert frame_filename("clip", 12345) == "clip_f012345.jpg"

    def test_wider_indices_do_not_truncate(self):
        assert frame_filename("clip", 1234567) == "clip_f1234567.jpg"


class TestAssignOutputPrefixes:
    def test_unique_stems_untouched(self):
        assert assign_output_prefixes(
            [Path("/a/x.mp4"), Path("/a/y.mp4")]
        ) == ["x", "y"]

    def test_duplicate_stems_get_numeric_suffix(self):
        prefixes = assign_output_prefixes(
            [Path("/a/v.mp4"), Path("/b/v.mp4"), Path("/c/v.avi")]
        )
        assert prefixes == ["v", "v-2", "v-3"]

    def test_suffix_dodges_real_stem_already_in_queue(self):
        prefixes = assign_output_prefixes(
            [Path("/a/v.mp4"), Path("/b/v-2.mp4"), Path("/c/v.mp4")]
        )
        assert prefixes == ["v", "v-2", "v-3"]


class TestEstimates:
    def _info(self, fps=30.0, frames=300):
        return VideoInfo(
            path=Path("v.mp4"), ok=True, fps=fps,
            frame_count=frames, duration_s=None,
        )

    def test_uncapped_sampled_count(self):
        params = SamplingParams(mode=MODE_INTERVAL, interval=30)
        assert estimate_sampled(self._info(), params) == 10

    def test_ceil_semantics(self):
        params = SamplingParams(mode=MODE_INTERVAL, interval=30)
        assert estimate_sampled(self._info(frames=301), params) == 11

    def test_estimate_frames_capped_by_max_frames(self):
        params = SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=50)
        assert estimate_sampled(self._info(), params) == 300
        assert estimate_frames(self._info(), params) == 50

    def test_unknown_frame_count_returns_none(self):
        info = VideoInfo(path=Path("v.mp4"), ok=True, fps=30.0)
        assert estimate_sampled(info, SamplingParams()) is None
        assert estimate_frames(info, SamplingParams()) is None

    def test_fps_mode_with_unknown_video_fps_returns_none(self):
        info = VideoInfo(path=Path("v.mp4"), ok=True, frame_count=100)
        params = SamplingParams(mode=MODE_FPS, target_fps=1.0)
        assert estimate_sampled(info, params) is None


class TestProbeVideo:
    def test_real_video_metadata(self, make_video):
        video = make_video("probe.mp4", frames=30, fps=10.0)
        info = probe_video(video)
        assert info.ok
        assert info.frame_count == 30
        assert info.fps == pytest.approx(10.0, rel=0.01)
        assert info.duration_s == pytest.approx(3.0, rel=0.01)

    def test_corrupt_file_degrades_to_unknown_without_raise(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_text("this is not a video")
        info = probe_video(bad)
        assert not info.ok
        assert info.fps is None and info.frame_count is None

    def test_missing_file_never_raises(self, tmp_path):
        info = probe_video(tmp_path / "missing.mp4")
        assert not info.ok


class TestExtractFrames:
    def test_interval_sampling_names_use_original_frame_index(
        self, make_video, tmp_path,
    ):
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "out"
        res = extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=10, max_frames=100),
        )
        assert not res.failed and not res.cancelled
        assert res.written == 3 and res.skipped == 0
        assert sorted(p.name for p in out.iterdir()) == [
            "clip_f000000.jpg", "clip_f000010.jpg", "clip_f000020.jpg",
        ]

    def test_written_files_are_jpegs(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=5)
        out = tmp_path / "out"
        extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=1),
        )
        data = (out / "clip_f000000.jpg").read_bytes()
        assert data[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_fps_mode_end_to_end(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=30, fps=10.0)
        res = extract_frames(
            video, tmp_path / "out",
            SamplingParams(mode=MODE_FPS, target_fps=2.0, max_frames=100),
        )
        # 10fps @ target 2fps → every 5th frame → indices 0,5,10,15,20,25.
        assert res.written == 6

    def test_skip_existing_counts_and_never_rewrites(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "out"
        params = SamplingParams(mode=MODE_INTERVAL, interval=10, max_frames=100)
        extract_frames(video, out, params)
        sentinel = out / "clip_f000010.jpg"
        sentinel.write_bytes(b"sentinel")  # must NOT be overwritten

        res2 = extract_frames(video, out, params)

        assert res2.written == 0
        assert res2.skipped == 3
        assert sentinel.read_bytes() == b"sentinel"

    def test_denser_rerun_only_fills_new_frames(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "out"
        extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=10, max_frames=100),
        )
        res = extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=5, max_frames=100),
        )
        assert res.skipped == 3   # f0 / f10 / f20 already on disk
        assert res.written == 3   # f5 / f15 / f25 filled in
        assert len(list(out.iterdir())) == 6

    def test_max_frames_cap_stops_early(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "out"
        res = extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=5),
        )
        assert res.written == 5
        assert len(list(out.iterdir())) == 5

    def test_cap_counts_skipped_frames_for_idempotent_reruns(
        self, make_video, tmp_path,
    ):
        """Re-running a completed extraction with the same params must write
        nothing more: skipped frames count toward the cap (otherwise a re-run
        would march deeper into the video past the first run's frames)."""
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "out"
        params = SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=5)
        extract_frames(video, out, params)

        res2 = extract_frames(video, out, params)

        assert res2.written == 0 and res2.skipped == 5
        assert len(list(out.iterdir())) == 5

    def test_cancel_preserves_partial_frames(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=30)
        out = tmp_path / "out"
        cancel = threading.Event()

        def cb(written, skipped):
            if written + skipped >= 2:
                cancel.set()

        res = extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=100),
            progress_cb=cb, cancel_event=cancel,
        )
        assert res.cancelled and not res.failed
        assert res.written == 2
        assert len(list(out.iterdir())) == 2  # written frames stay on disk

    def test_precancelled_event_writes_nothing(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=10)
        cancel = threading.Event()
        cancel.set()
        res = extract_frames(
            video, tmp_path / "out", SamplingParams(), cancel_event=cancel,
        )
        assert res.cancelled and res.written == 0 and not res.failed

    def test_unopenable_video_fails_without_raise(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_text("junk")
        res = extract_frames(bad, tmp_path / "out", SamplingParams())
        assert res.failed
        assert res.written == 0
        assert res.error  # readable Chinese message

    def test_missing_video_fails_without_raise(self, tmp_path):
        res = extract_frames(
            tmp_path / "nope.mp4", tmp_path / "out", SamplingParams(),
        )
        assert res.failed

    def test_custom_prefix_overrides_stem(self, make_video, tmp_path):
        video = make_video("clip.mp4", frames=3)
        out = tmp_path / "out"
        res = extract_frames(
            video, out,
            SamplingParams(mode=MODE_INTERVAL, interval=1, max_frames=1),
            prefix="clip-2",
        )
        assert res.prefix == "clip-2"
        assert (out / "clip-2_f000000.jpg").exists()
