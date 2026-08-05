"""Tests for VideoImportDialog — key paths with an injected fake prober.

The prober injection (keyword-only, script_tools convention) keeps cv2 out
of these tests entirely: estimates are computed from crafted VideoInfo
metadata.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.video_frames import (
    MODE_FPS,
    MODE_INTERVAL,
    VideoInfo,
)
from src.ui.video_import_dialog import VideoImportDialog


def _prober(fps=10.0, frames=100):
    def probe(path):
        return VideoInfo(
            path=Path(path), ok=True, fps=fps, frame_count=frames,
            duration_s=(frames / fps) if fps else None,
        )
    return probe


def _unknown_prober(path):
    return VideoInfo(path=Path(path), ok=False)


class TestQueueManagement:
    def test_prefill_lists_videos_in_order(self, qapp, tmp_path):
        v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mkv"
        dlg = VideoImportDialog(initial_videos=[v1, v2], prober=_prober())
        assert dlg.get_videos() == [v1, v2]

    def test_duplicate_add_ignored(self, qapp, tmp_path):
        v = tmp_path / "a.mp4"
        dlg = VideoImportDialog(initial_videos=[v], prober=_prober())
        dlg.add_videos([v])
        assert dlg.get_videos() == [v]

    def test_remove_selected(self, qapp, tmp_path):
        v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
        dlg = VideoImportDialog(initial_videos=[v1, v2], prober=_prober())
        dlg._list.item(0).setSelected(True)
        dlg._on_remove_clicked()
        assert dlg.get_videos() == [v2]

    def test_ok_disabled_when_empty_enabled_after_add(self, qapp, tmp_path):
        dlg = VideoImportDialog(prober=_prober())
        assert not dlg._btn_ok.isEnabled()
        dlg.add_videos([tmp_path / "a.mp4"])
        assert dlg._btn_ok.isEnabled()
        dlg._list.item(0).setSelected(True)
        dlg._on_remove_clicked()
        assert not dlg._btn_ok.isEnabled()


class TestParams:
    def test_default_params_interval_mode(self, qapp):
        dlg = VideoImportDialog(prober=_prober())
        params = dlg.get_params()
        assert params.mode == MODE_INTERVAL
        assert params.interval == 30
        assert params.max_frames == 500

    def test_fps_mode_reflected_in_params(self, qapp):
        dlg = VideoImportDialog(prober=_prober())
        dlg._mode_combo.setCurrentIndex(1)  # 目标帧率
        dlg._fps_spin.setValue(2.5)
        dlg._max_spin.setValue(120)
        params = dlg.get_params()
        assert params.mode == MODE_FPS
        assert params.target_fps == pytest.approx(2.5)
        assert params.max_frames == 120

    def test_mode_switch_toggles_value_widgets(self, qapp):
        dlg = VideoImportDialog(prober=_prober())
        assert dlg._interval_spin.isVisibleTo(dlg)
        assert not dlg._fps_spin.isVisibleTo(dlg)
        dlg._mode_combo.setCurrentIndex(1)
        assert not dlg._interval_spin.isVisibleTo(dlg)
        assert dlg._fps_spin.isVisibleTo(dlg)


class TestEstimates:
    def test_known_metadata_shows_estimate(self, qapp, tmp_path):
        # 100 frames @ default interval 30 → ceil(100/30) = 4
        dlg = VideoImportDialog(
            initial_videos=[tmp_path / "a.mp4"], prober=_prober(frames=100),
        )
        text = dlg._list.item(0).text()
        assert "a.mp4" in text
        assert "预计 4 帧" in text

    def test_unknown_metadata_shows_unknown_and_does_not_block(
        self, qapp, tmp_path,
    ):
        dlg = VideoImportDialog(
            initial_videos=[tmp_path / "x.mp4"], prober=_unknown_prober,
        )
        assert "未知" in dlg._list.item(0).text()
        assert dlg._btn_ok.isEnabled()  # unknown metadata never blocks import

    def test_estimates_update_when_interval_changes(self, qapp, tmp_path):
        dlg = VideoImportDialog(
            initial_videos=[tmp_path / "a.mp4"], prober=_prober(frames=100),
        )
        dlg._interval_spin.setValue(10)
        assert "预计 10 帧" in dlg._list.item(0).text()

    def test_over_cap_warning_toggles_with_params(self, qapp, tmp_path):
        dlg = VideoImportDialog(
            initial_videos=[tmp_path / "a.mp4"], prober=_prober(frames=10000),
        )
        dlg._max_spin.setValue(100)
        dlg._interval_spin.setValue(1)  # raw estimate 10000 > cap 100
        assert dlg._warning_label.isVisibleTo(dlg)
        # Displayed estimate is clamped to the cap.
        assert "预计 100 帧" in dlg._list.item(0).text()

        dlg._interval_spin.setValue(200)  # raw estimate 50 ≤ cap
        assert not dlg._warning_label.isVisibleTo(dlg)
        assert "预计 50 帧" in dlg._list.item(0).text()

    def test_summary_counts_videos_and_total(self, qapp, tmp_path):
        dlg = VideoImportDialog(
            initial_videos=[tmp_path / "a.mp4", tmp_path / "b.mp4"],
            prober=_prober(frames=100),
        )
        dlg._interval_spin.setValue(10)
        assert "共 2 个视频" in dlg._summary_label.text()
        assert "约 20 帧" in dlg._summary_label.text()
