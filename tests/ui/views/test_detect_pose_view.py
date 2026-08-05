"""DetectPoseView tests — keypoint-attach draw-state handling + seam guards.

The structural guards pin the 07-11 seam cleanup: the shell must drive views
through the TaskView contract (no hasattr sniffing), and DetectPoseView must
drive the canvas through public methods only (no `_canvas._*` private reach).
"""
import re
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QColor, QKeyEvent

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"


def _make_view(qapp):
    from src.ui.views.detect_pose import DetectPoseView
    from src.utils.image import ImageCache

    return DetectPoseView(ImageCache(max_count=2, max_memory_mb=16.0), OrderedDict())


def _make_segment_project(tmp_path):
    from src.core.project import ProjectManager

    pm = ProjectManager.create(
        tmp_path / "proj", "seg", classes=["leaf", "petal"], task_type="segment",
    )
    img_dir = pm.project_dir / pm.config.image_dir
    for i in range(2):
        img = QImage(100, 80, QImage.Format_RGB32)
        img.fill(QColor(Qt.blue))
        img.save(str(img_dir / f"img{i}.png"), "PNG")
    return pm


def _key(view, key):
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))



class TestKeypointAttachDrawState:
    def test_empty_draw_start_early_exits_and_clears_draw_state(self, qapp):
        from src.core.annotation import Annotation

        view = _make_view(qapp)
        try:
            ann = Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.3, 0.4))
            view._canvas.set_annotations([ann])
            # No pending draw-origin, but stale in-progress state exists.
            assert view._canvas.consume_draw_start() is None
            view._canvas._draw_current = (0.2, 0.2)

            view._on_keypoint_attach_requested(ann.id, 10.0, 10.0)

            assert ann.keypoints == []  # early exit — picker never opened
            assert view._canvas._draw_current is None  # draw state cleared
        finally:
            view.deleteLater()

    def test_unknown_annotation_consumes_draw_start(self, qapp):
        """Consume-at-entry: even on the ann-missing early exit, the pending
        draw-origin is taken and cleared so later events can't reuse it."""
        view = _make_view(qapp)
        try:
            view._canvas._draw_start = (0.4, 0.6)
            view._canvas._draw_current = (0.4, 0.6)

            view._on_keypoint_attach_requested("no-such-id", 10.0, 10.0)

            assert view._canvas.consume_draw_start() is None
            assert view._canvas._draw_current is None
        finally:
            view.deleteLater()


class TestSeamStructuralGuards:
    def test_label_panel_has_no_hasattr_sniffing(self):
        """The shell drives views through the TaskView contract (default
        no-op members) — hasattr view-type sniffing must not come back."""
        pattern = re.compile(r"hasattr\s*\(")
        text = (SRC_ROOT / "ui" / "label_panel.py").read_text(encoding="utf-8")
        offenders = [
            f"src/ui/label_panel.py:{lineno}: {line.strip()}"
            for lineno, line in enumerate(text.splitlines(), start=1)
            if pattern.search(line)
        ]
        assert not offenders, (
            "hasattr sniffing in the LabelPanel shell — add the member to the "
            "TaskView contract (default no-op) instead:\n" + "\n".join(offenders)
        )

    def test_detect_pose_has_no_canvas_private_reach(self):
        """DetectPoseView uses AnnotationCanvas public methods only —
        `_canvas._*` attribute touches must not come back."""
        pattern = re.compile(r"_canvas\._")
        text = (SRC_ROOT / "ui" / "views" / "detect_pose.py").read_text(encoding="utf-8")
        offenders = [
            f"src/ui/views/detect_pose.py:{lineno}: {line.strip()}"
            for lineno, line in enumerate(text.splitlines(), start=1)
            if pattern.search(line)
        ]
        assert not offenders, (
            "private canvas state reached from DetectPoseView — add/extend a "
            "public AnnotationCanvas method instead:\n" + "\n".join(offenders)
        )


class TestSegmentPolygonIntegration:
    def test_polygon_button_visible_only_for_segment(self, qapp, tmp_path):
        from src.core.project import ProjectManager

        seg = _make_segment_project(tmp_path / "seg")
        view = _make_view(qapp)
        try:
            view.set_project(seg)
            # isHidden() reflects the explicit setVisible() flag regardless of
            # whether the top-level window is shown.
            assert view._btn_polygon.isHidden() is False
            assert view._btn_keypoint.isHidden() is True

            det = ProjectManager.create(
                tmp_path / "det", "det", classes=["a"], task_type="detect",
            )
            view.set_project(det)
            assert view._btn_polygon.isHidden() is True
        finally:
            view.deleteLater()

    def test_p_key_selects_polygon_tool(self, qapp, tmp_path):
        seg = _make_segment_project(tmp_path)
        view = _make_view(qapp)
        try:
            view.set_project(seg)
            _key(view, Qt.Key_P)
            assert view._canvas.tool_mode == "draw_polygon"
            assert view._btn_polygon.isChecked() is True
        finally:
            view.deleteLater()

    def test_class_request_creates_polygon_annotation(self, qapp, tmp_path, monkeypatch):
        seg = _make_segment_project(tmp_path)
        view = _make_view(qapp)
        try:
            view.set_project(seg)
            view._set_tool("draw_polygon")
            # Seed a 3-vertex draft on the canvas.
            view._canvas._polygon_points = [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
            # Stub the class picker to return "leaf" without a dialog.
            monkeypatch.setattr(view, "_show_class_picker", lambda *a, **k: "leaf")

            created = []
            view._canvas.annotation_created.connect(lambda a: created.append(a))
            view._on_class_requested(70.0, 60.0)

            assert len(created) == 1
            assert created[0].polygon == [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
            assert created[0].bbox is not None
        finally:
            view.deleteLater()

    def test_undo_restores_polygon(self, qapp, tmp_path, monkeypatch):
        seg = _make_segment_project(tmp_path)
        view = _make_view(qapp)
        try:
            view.set_project(seg)
            view._file_list.setCurrentRow(0)
            view._set_tool("draw_polygon")
            view._canvas._polygon_points = [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
            monkeypatch.setattr(view, "_show_class_picker", lambda *a, **k: "leaf")
            view._on_class_requested(70.0, 60.0)
            assert len(view._canvas.annotations) == 1

            view.undo()
            assert len(view._canvas.annotations) == 0

            view.redo()
            assert len(view._canvas.annotations) == 1
            assert view._canvas.annotations[0].polygon == [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
        finally:
            view.deleteLater()

    def test_escape_cancels_polygon_draft(self, qapp, tmp_path):
        seg = _make_segment_project(tmp_path)
        view = _make_view(qapp)
        try:
            view.set_project(seg)
            view._set_tool("draw_polygon")
            view._canvas._polygon_points = [[0.1, 0.1], [0.6, 0.1]]
            _key(view, Qt.Key_Escape)
            assert view._canvas.has_polygon_draft is False
        finally:
            view.deleteLater()

    def test_polygon_edit_persists_and_reloads(self, qapp, tmp_path, monkeypatch):
        seg = _make_segment_project(tmp_path)
        view = _make_view(qapp)
        try:
            view.set_project(seg)
            view._file_list.setCurrentRow(0)
            view._set_tool("draw_polygon")
            view._canvas._polygon_points = [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
            monkeypatch.setattr(view, "_show_class_picker", lambda *a, **k: "leaf")
            view._on_class_requested(70.0, 60.0)

            view.commit_pending_save()
            # Reload from disk; polygon must survive the round-trip.
            view.reload_current()
            anns = view._canvas.annotations
            assert len(anns) == 1
            assert anns[0].polygon == [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]
        finally:
            view.deleteLater()



class TestProjectLoadSinglePass:
    """set_project reads each label JSON once, feeding file-list metadata AND
    stats from the same scan; the stats must match a full recompute."""

    def _write_record(self, pm, name, annotations, tags=()):
        from src.core.annotation import ImageAnnotation
        from src.core.label_io import save_annotation

        ia = ImageAnnotation(
            image_path=name, image_size=(100, 80),
            annotations=annotations, tags=list(tags),
        )
        save_annotation(ia, pm.label_path_for(pm.project_dir / pm.config.image_dir / name))

    def test_stats_match_full_recompute(self, qapp, tmp_path):
        from src.core.annotation import Annotation

        pm = _make_segment_project(tmp_path)
        self._write_record(pm, "img0.png", [
            Annotation("leaf", 0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
            Annotation("petal", 1, bbox=(0.3, 0.3, 0.1, 0.1), confirmed=True),
        ], tags=["night"])
        self._write_record(pm, "img1.png", [
            Annotation("leaf", 0, bbox=(0.4, 0.4, 0.2, 0.2), confirmed=False),
        ])

        view = _make_view(qapp)
        try:
            view.set_project(pm)
            expected = view._compute_project_stats()
            assert view._stats_cache == expected
            assert expected["total_images"] == 2
            assert expected["labeled_images"] == 2
            assert expected["confirmed_images"] == 1
            assert expected["total_annotations"] == 3
            assert expected["class_counts"] == {"leaf": 2, "petal": 1}
        finally:
            view.cleanup()
            view.deleteLater()

    def test_file_list_metadata_populated_from_single_scan(self, qapp, tmp_path):
        from src.core.annotation import Annotation

        pm = _make_segment_project(tmp_path)
        self._write_record(pm, "img0.png", [
            Annotation("leaf", 0, bbox=(0.5, 0.5, 0.2, 0.2), confirmed=True),
        ], tags=["night"])

        view = _make_view(qapp)
        try:
            view.set_project(pm)
            fl = view._file_list
            img0 = str(pm.project_dir / pm.config.image_dir / "img0.png")
            img1 = str(pm.project_dir / pm.config.image_dir / "img1.png")
            assert fl._statuses[img0] == "confirmed"
            assert img1 not in fl._statuses
            assert fl._image_classes[img0] == {"leaf"}
            assert fl._image_tags[img0] == {"night"}
            assert fl.item(0).text().startswith("✓")
        finally:
            view.cleanup()
            view.deleteLater()
