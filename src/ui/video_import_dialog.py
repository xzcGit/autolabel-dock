"""Video-import dialog — pick videos + sampling params, with frame estimates.

Pure parameter-collection view for the video → frames import flow: the user
queues video files (file picker or pre-filled from a drag-drop), picks one
global sampling setting (每 N 帧 | 目标帧率) plus a per-video max-frames cap,
and sees a metadata-based estimate per video (「未知」 when the container
metadata is unreliable — estimates never block the import). Extraction
itself runs in ``VideoImportController``; MainWindow reads
``get_videos()`` / ``get_params()`` after ``exec_()`` accepts.

``prober`` is a keyword-only injection for tests (defaults to
``core.video_frames.probe_video``), following the script_tools panel
convention for panel-local collaborators.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.core.project import VIDEO_EXTENSIONS
from src.core.video_frames import (
    MODE_FPS,
    MODE_INTERVAL,
    SamplingParams,
    VideoInfo,
    estimate_frames,
    estimate_sampled,
    probe_video,
)
from src.ui.theme import text_style

_OVER_CAP_WARNING = (
    "部分视频预计帧数超过最大帧数上限：抽帧到上限即停（偏向视频开头）。\n"
    "如需覆盖完整视频，请调大采样间隔或提高上限。"
)


class VideoImportDialog(QDialog):
    """Multi-video queue + sampling params for frame extraction."""

    def __init__(
        self,
        parent=None,
        *,
        initial_videos: Iterable[Path] | None = None,
        prober: Callable[[Path], VideoInfo] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("导入视频帧")
        self.setMinimumWidth(560)
        self._prober: Callable[[Path], VideoInfo] = prober or probe_video
        self._infos: dict[str, VideoInfo] = {}  # str(path) -> VideoInfo
        self._init_ui()
        if initial_videos:
            self.add_videos([Path(p) for p in initial_videos])
        self._refresh_estimates()

    # ── UI construction ────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("视频文件（按顺序抽帧到项目图片目录）:"))

        list_row = QHBoxLayout()
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.setMinimumHeight(140)
        list_row.addWidget(self._list, 1)

        btn_col = QVBoxLayout()
        self._btn_add = QPushButton("添加视频...")
        self._btn_add.clicked.connect(self._on_add_clicked)
        btn_col.addWidget(self._btn_add)
        self._btn_remove = QPushButton("移除选中")
        self._btn_remove.clicked.connect(self._on_remove_clicked)
        btn_col.addWidget(self._btn_remove)
        btn_col.addStretch(1)
        list_row.addLayout(btn_col)
        layout.addLayout(list_row)

        form = QFormLayout()

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("每 N 帧取一帧", MODE_INTERVAL)
        self._mode_combo.addItem("目标帧率 (fps)", MODE_FPS)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("采样方式:", self._mode_combo)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 100000)
        self._interval_spin.setValue(30)
        self._interval_spin.setSuffix(" 帧")
        self._interval_spin.valueChanged.connect(self._refresh_estimates)
        self._interval_row_label = QLabel("帧间隔 N:")
        form.addRow(self._interval_row_label, self._interval_spin)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.1, 240.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setSingleStep(0.5)
        self._fps_spin.setValue(1.0)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.valueChanged.connect(self._refresh_estimates)
        self._fps_row_label = QLabel("目标帧率:")
        form.addRow(self._fps_row_label, self._fps_spin)

        self._max_spin = QSpinBox()
        self._max_spin.setRange(1, 1000000)
        self._max_spin.setValue(500)
        self._max_spin.setSuffix(" 帧")
        self._max_spin.valueChanged.connect(self._refresh_estimates)
        form.addRow("每视频最大帧数:", self._max_spin)

        layout.addLayout(form)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet(text_style("hint"))
        layout.addWidget(self._summary_label)

        self._warning_label = QLabel(_OVER_CAP_WARNING)
        self._warning_label.setStyleSheet(text_style("error"))
        self._warning_label.setWordWrap(True)
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        buttons = QDialogButtonBox()
        self._btn_ok = buttons.addButton("开始导入", QDialogButtonBox.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_mode_changed()

    # ── Public face ────────────────────────────────────────────

    def add_videos(self, paths: Iterable[Path]) -> None:
        """Append videos to the queue (duplicates ignored), probing each for
        the estimate line. Used by the 添加 button and drag-drop pre-fill."""
        existing = {str(p) for p in self.get_videos()}
        for path in paths:
            path = Path(path)
            key = str(path)
            if key in existing:
                continue
            existing.add(key)
            self._infos[key] = self._prober(path)
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, key)
            item.setToolTip(key)
            self._list.addItem(item)
        self._refresh_estimates()

    def get_videos(self) -> list[Path]:
        """Queued video paths, in list order."""
        return [
            Path(self._list.item(i).data(Qt.UserRole))
            for i in range(self._list.count())
        ]

    def get_params(self) -> SamplingParams:
        """Sampling params reflecting the current widget state."""
        return SamplingParams(
            mode=self._mode_combo.currentData(),
            interval=self._interval_spin.value(),
            target_fps=self._fps_spin.value(),
            max_frames=self._max_spin.value(),
        )

    # ── Internal slots ─────────────────────────────────────────

    def _on_add_clicked(self) -> None:
        pattern = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "",
            f"视频文件 ({pattern});;所有文件 (*)",
        )
        if files:
            self.add_videos([Path(f) for f in files])

    def _on_remove_clicked(self) -> None:
        for item in self._list.selectedItems():
            self._infos.pop(item.data(Qt.UserRole), None)
            self._list.takeItem(self._list.row(item))
        self._refresh_estimates()

    def _on_mode_changed(self) -> None:
        interval_mode = self._mode_combo.currentData() == MODE_INTERVAL
        self._interval_spin.setVisible(interval_mode)
        self._interval_row_label.setVisible(interval_mode)
        self._fps_spin.setVisible(not interval_mode)
        self._fps_row_label.setVisible(not interval_mode)
        self._refresh_estimates()

    def _refresh_estimates(self) -> None:
        """Recompute per-video estimate texts + the total / over-cap hint."""
        params = self.get_params()
        total_known = 0
        any_unknown = False
        any_over_cap = False
        for i in range(self._list.count()):
            item = self._list.item(i)
            key = item.data(Qt.UserRole)
            path = Path(key)
            info = self._infos.get(key) or VideoInfo(path=path, ok=False)
            capped = estimate_frames(info, params)
            raw = estimate_sampled(info, params)
            if capped is None:
                any_unknown = True
                item.setText(f"{path.name} — 预计 未知")
            else:
                if raw is not None and raw > params.max_frames:
                    any_over_cap = True
                item.setText(f"{path.name} — 预计 {capped} 帧")
                total_known += capped

        count = self._list.count()
        if count == 0:
            self._summary_label.setText("尚未选择视频")
        else:
            text = f"共 {count} 个视频，预计写入约 {total_known} 帧"
            if any_unknown:
                text += "（部分视频元数据未知，实际以抽帧为准）"
            self._summary_label.setText(text)
        self._warning_label.setVisible(any_over_cap)
        self._btn_ok.setEnabled(count > 0)
