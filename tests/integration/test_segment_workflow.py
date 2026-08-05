"""End-to-end integration test for the segment (instance segmentation) workflow.

Mirrors ``test_classification_workflow.py``: minimal tmp_path projects, no real
model / GPU (predictions are constructed the way ``Predictor._run`` would emit
them: dense mask contour → ``simplify_polygon`` → derived bbox). The four
chains covered here are all Qt-free — canvas polygon interaction has its own
UI suite (``tests/ui/test_canvas_polygon.py``).

Raw ``label_io`` use is fine here: tests are outside the LabelStore guard
allowlist scope (which pins ``src/`` only), same precedent as the classify
workflow test.
"""
import json
import math
from dataclasses import replace

import pytest
import yaml

from src.core.annotation import Annotation, ImageAnnotation
from src.core.autolabel import merge_predictions
from src.core.formats import get_export_registry, get_import_registry
from src.core.formats.import_merge import merge_imported_records
from src.core.label_io import load_annotation, save_annotation
from src.core.label_store import LabelStore
from src.core.polygon import bbox_from_polygon, bbox_to_polygon, simplify_polygon
from src.core.project import ProjectManager


def _polygon_ann(
    class_name,
    class_id,
    polygon,
    confirmed=True,
    source="manual",
    confidence=1.0,
):
    """Annotation with the polygon↔derived-bbox invariant every producer keeps."""
    return Annotation(
        class_name=class_name,
        class_id=class_id,
        bbox=bbox_from_polygon(polygon),
        polygon=polygon,
        confirmed=confirmed,
        source=source,
        confidence=confidence,
    )


def _circle_contour(cx, cy, r, n=72):
    """Dense mask-like contour (what ``result.masks.xyn`` delivers)."""
    return [
        [cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)]
        for i in range(n)
    ]


def _assert_polygons_close(actual, expected, abs_tol=1e-6):
    """Vertex-wise approximate polygon equality (export rounds at 1e-6)."""
    assert len(actual) == len(expected)
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert ax == pytest.approx(ex, abs=abs_tol)
        assert ay == pytest.approx(ey, abs=abs_tol)


def _rows_are_pure_polygons(txt_path):
    """Every YOLO-seg row must be class + ≥3 (x,y) pairs — never a 5-field bbox
    row (ultralytics reshapes any >6-field file as (-1, 2), so one stray bbox
    row corrupts the whole file)."""
    for line in txt_path.read_text().strip().splitlines():
        n = len(line.split())
        assert n >= 7, f"{txt_path.name}: row has {n} fields (bbox/pose row in a seg file)"
        assert (n - 1) % 2 == 0, f"{txt_path.name}: odd coordinate count"


def test_segment_project_label_roundtrip(tmp_path):
    """Create segment project → save polygon label → reopen → polygon and
    derived bbox restored; legacy label JSON without 'polygon' loads unchanged."""

    project_dir = tmp_path / "segment_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Leaf Segmentation",
        classes=["leaf", "stem"],
        task_type="segment",
    )
    assert pm.config.task_type == "segment"

    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"
    (images_dir / "seg_000.jpg").write_text("fake image")

    poly = [[0.3, 0.2], [0.7, 0.25], [0.65, 0.8], [0.35, 0.75]]
    derived = bbox_from_polygon(poly)
    small_poly = [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]

    ia = ImageAnnotation(
        image_path="seg_000.jpg",
        image_size=(640, 480),
        annotations=[
            _polygon_ann("leaf", 0, poly),
            _polygon_ann("stem", 1, small_poly, confirmed=False, source="auto", confidence=0.8),
        ],
    )
    # Confirm-lifecycle matches detect: one unconfirmed annotation → pending.
    assert ia.status == "pending"
    save_annotation(ia, labels_dir / "seg_000.json")

    # Reopen the project and reload the record.
    pm2 = ProjectManager.open(project_dir)
    assert pm2.config.task_type == "segment"

    loaded = load_annotation(pm2.label_path_for(images_dir / "seg_000.jpg"))
    assert loaded is not None
    assert loaded.status == "pending"
    first, second = loaded.annotations

    assert first.polygon == poly
    assert first.bbox == pytest.approx(derived)
    assert second.polygon == small_poly
    assert second.confirmed is False
    assert second.source == "auto"
    # polygon ↔ derived-bbox invariant survives the disk round-trip.
    for ann in loaded.annotations:
        assert bbox_from_polygon(ann.polygon) == pytest.approx(ann.bbox)

    # Legacy (pre-segment) label JSON without a "polygon" key loads unchanged.
    legacy = {
        "image_path": "legacy.jpg",
        "image_size": [640, 480],
        "annotations": [
            {
                "id": "legacy-1",
                "class_name": "leaf",
                "class_id": 0,
                "bbox": [0.5, 0.5, 0.2, 0.2],
                "keypoints": [],
            }
        ],
    }
    legacy_path = labels_dir / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    old = load_annotation(legacy_path)
    assert old is not None
    assert old.annotations[0].polygon is None
    assert old.annotations[0].bbox == (0.5, 0.5, 0.2, 0.2)
    assert old.annotations[0].confirmed is True


def test_segment_auto_label_merge_flow(tmp_path):
    """Polygon predictions (mask contour → simplify → derived bbox) merge via
    ``merge_predictions`` exactly like detect: same-class overlap on the
    derived bbox conflicts, everything else is accepted; the batch surface
    drops conflicts and persists the accepted predictions."""

    project_dir = tmp_path / "segment_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Seg AutoLabel",
        classes=["leaf", "stem"],
        task_type="segment",
    )
    images_dir = project_dir / "images"
    (images_dir / "img.jpg").write_text("fake image")
    label_path = pm.label_path_for(images_dir / "img.jpg")

    # User-confirmed polygon (square around center → derived bbox 0.4×0.4).
    existing = _polygon_ann("leaf", 0, [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])
    save_annotation(
        ImageAnnotation(image_path="img.jpg", image_size=(640, 480), annotations=[existing]),
        label_path,
    )

    # Build "predictions" the way Predictor._run ingests a mask: dense contour
    # → simplify_polygon → derived bbox, source="auto", confirmed=False.
    dense = _circle_contour(0.5, 0.5, 0.2)
    simplified = simplify_polygon(dense)
    assert 3 <= len(simplified) < len(dense)  # simplification actually ran

    pred_overlap = _polygon_ann(  # same class, IoU≈1.0 vs existing → conflict
        "leaf", 0, simplified, confirmed=False, source="auto", confidence=0.9,
    )
    pred_clean = _polygon_ann(  # same class, elsewhere → accepted
        "leaf", 0, [[0.05, 0.05], [0.25, 0.05], [0.15, 0.25]],
        confirmed=False, source="auto", confidence=0.85,
    )
    pred_other_class = _polygon_ann(  # overlaps (IoU 0.5625) but class differs → accepted
        "stem", 1, [[0.35, 0.35], [0.65, 0.35], [0.5, 0.65]],
        confirmed=False, source="auto", confidence=0.7,
    )
    preds = [pred_overlap, pred_clean, pred_other_class]

    record = load_annotation(label_path)
    outcome = merge_predictions(record.annotations, preds, iou_threshold=0.5)

    assert [p.id for p in outcome.accepted] == [pred_clean.id, pred_other_class.id]
    assert [(e.id, p.id) for e, p in outcome.conflict_pairs] == [
        (existing.id, pred_overlap.id)
    ]

    # Detect parity: strip the polygons (bbox-only twins keep their ids) —
    # the partition must be identical, because conflict detection runs on the
    # derived bbox and nothing else.
    detect_outcome = merge_predictions(
        [replace(a, polygon=None) for a in record.annotations],
        [replace(p, polygon=None) for p in preds],
        iou_threshold=0.5,
    )
    assert [a.id for a in detect_outcome.accepted] == [a.id for a in outcome.accepted]
    assert [(e.id, p.id) for e, p in detect_outcome.conflict_pairs] == [
        (e.id, p.id) for e, p in outcome.conflict_pairs
    ]

    # Batch surface: drop conflicts, persist accepted predictions.
    record.annotations.extend(outcome.accepted)
    save_annotation(record, label_path)

    final = load_annotation(label_path)
    assert len(final.annotations) == 3
    ids = {a.id for a in final.annotations}
    assert pred_overlap.id not in ids  # conflict was dropped, not saved

    by_id = {a.id: a for a in final.annotations}
    _assert_polygons_close(by_id[pred_clean.id].polygon, pred_clean.polygon)
    _assert_polygons_close(by_id[pred_other_class.id].polygon, pred_other_class.polygon)
    assert by_id[pred_clean.id].confirmed is False
    assert by_id[pred_clean.id].source == "auto"
    for ann in final.annotations:
        assert bbox_from_polygon(ann.polygon) == pytest.approx(ann.bbox)
    # 1 confirmed + 2 pending auto predictions → image is pending review.
    assert final.status == "pending"


def test_segment_dataset_preparation(tmp_path):
    """DatasetPreparer(task='segment') writes pure-polygon rows (cid x1 y1 ...),
    folds bbox-only annotations to 4-corner polygons, excludes unconfirmed
    images, and emits a detect-shaped data.yaml (no kpt_shape)."""
    from src.engine.dataset import DatasetPreparer

    project_dir = tmp_path / "segment_project"
    pm = ProjectManager.create(
        project_dir=project_dir,
        name="Seg Dataset",
        classes=["leaf"],
        task_type="segment",
    )
    images_dir = project_dir / "images"
    labels_dir = project_dir / "labels"

    triangle = [[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]]
    quad = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
    box = (0.5, 0.5, 0.4, 0.2)

    specs = {
        "poly": [_polygon_ann("leaf", 0, triangle)],
        "poly2": [_polygon_ann("leaf", 0, quad)],
        "boxonly": [Annotation(class_name="leaf", class_id=0, bbox=box, confirmed=True)],
        "mixed": [
            _polygon_ann("leaf", 0, triangle),
            Annotation(class_name="leaf", class_id=0, bbox=box, confirmed=True),
        ],
        # Unconfirmed only → whole image excluded from the dataset.
        "pending": [_polygon_ann("leaf", 0, triangle, confirmed=False, source="auto")],
    }
    for stem, anns in specs.items():
        (images_dir / f"{stem}.jpg").write_text("fake image")
        save_annotation(
            ImageAnnotation(image_path=f"{stem}.jpg", image_size=(640, 480), annotations=anns),
            labels_dir / f"{stem}.json",
        )

    dataset_dir = project_dir / "datasets" / "current"
    data_yaml = DatasetPreparer(pm).prepare(
        dataset_dir, task="segment", val_ratio=0.25, seed=42,
    )

    # data.yaml is detect-shaped: names/nc/train/val, NO kpt_shape.
    data = yaml.safe_load(data_yaml.read_text())
    assert data["names"] == ["leaf"]
    assert data["nc"] == 1
    assert data["train"] == "train/images"
    assert data["val"] == "val/images"
    assert "kpt_shape" not in data

    # 4 confirmed images split 3/1 (single class, val_ratio 0.25); the
    # pending-only image never enters the dataset.
    txts = {
        p.stem: p
        for split in ("train", "val")
        for p in (dataset_dir / split / "labels").glob("*.txt")
    }
    assert set(txts) == {"poly", "poly2", "boxonly", "mixed"}
    assert len(list((dataset_dir / "train" / "images").iterdir())) == 3
    assert len(list((dataset_dir / "val" / "images").iterdir())) == 1

    # Every row in every split is a pure polygon row.
    for txt in txts.values():
        _rows_are_pure_polygons(txt)

    # Polygon rows carry the vertices verbatim (class + 3 points = 7 fields).
    poly_parts = txts["poly"].read_text().strip().split()
    assert len(poly_parts) == 7
    assert poly_parts[0] == "0"
    coords = [float(v) for v in poly_parts[1:]]
    _assert_polygons_close(
        [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)], triangle,
    )

    # bbox-only annotation folded to its 4-corner rectangle (TL TR BR BL).
    box_parts = txts["boxonly"].read_text().strip().split()
    assert len(box_parts) == 9
    coords = [float(v) for v in box_parts[1:]]
    _assert_polygons_close(
        [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)],
        bbox_to_polygon(box),
    )

    # Mixed image: one 7-field polygon row + one 9-field folded-bbox row —
    # never a 5-field bbox row alongside polygons.
    mixed_lines = txts["mixed"].read_text().strip().splitlines()
    assert sorted(len(line.split()) for line in mixed_lines) == [7, 9]


def test_segment_yolo_export_import_roundtrip(tmp_path):
    """Export through the registry ('YOLO-seg' really registered — PRD decision),
    re-import through the registered YOLO importer (auto-detects segment from
    variable-length rows) and the merge pipeline: polygons and derived bboxes
    survive the round-trip; a bbox-only annotation returns as its 4-corner
    polygon."""

    # ── Source project: polygon + bbox-only annotations ──
    src_dir = tmp_path / "source_project"
    pm = ProjectManager.create(
        project_dir=src_dir,
        name="Seg Source",
        classes=["leaf", "stem"],
        task_type="segment",
    )
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"

    triangle = [[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]]
    quad = [[0.1, 0.1], [0.4, 0.15], [0.35, 0.45], [0.05, 0.4]]
    box = (0.25, 0.35, 0.3, 0.2)

    for stem in ("seg1", "seg2"):
        (images_dir / f"{stem}.jpg").write_text("fake image")
    save_annotation(
        ImageAnnotation(
            image_path="seg1.jpg",
            image_size=(640, 480),
            annotations=[_polygon_ann("leaf", 0, triangle), _polygon_ann("stem", 1, quad)],
        ),
        labels_dir / "seg1.json",
    )
    save_annotation(
        ImageAnnotation(
            image_path="seg2.jpg",
            image_size=(640, 480),
            annotations=[Annotation(class_name="leaf", class_id=0, bbox=box, confirmed=True)],
        ),
        labels_dir / "seg2.json",
    )

    # ── Export via ExportRegistry (the controller path) ──
    registry = get_export_registry()
    assert "YOLO-seg" in registry.list_names()  # really registered, not an orphan fn

    export_dir = tmp_path / "export_yoloseg"
    annotations = [
        load_annotation(labels_dir / "seg1.json"),
        load_annotation(labels_dir / "seg2.json"),
    ]
    registry.export("YOLO-seg", annotations, export_dir, classes=pm.config.classes)

    data = yaml.safe_load((export_dir / "data.yaml").read_text())
    assert data["names"] == ["leaf", "stem"]
    assert data["nc"] == 2

    # ultralytics-trainable shape: pure polygon rows in every file.
    txt_files = sorted((export_dir / "labels").glob("*.txt"))
    assert [p.stem for p in txt_files] == ["seg1", "seg2"]
    for txt in txt_files:
        _rows_are_pure_polygons(txt)

    # ── Import back into a fresh project via registry + merge pipeline ──
    dst_dir = tmp_path / "reimport_project"
    pm2 = ProjectManager.create(
        project_dir=dst_dir,
        name="Seg Reimport",
        classes=[],
        task_type="segment",
    )
    for stem in ("seg1", "seg2"):
        (dst_dir / "images" / f"{stem}.jpg").write_text("fake image")

    # The generic "YOLO" record importer must auto-detect segment (rows are
    # variable-length: 7- and 9-field), not mis-parse them as pose.
    records = get_import_registry().import_records(
        "YOLO", export_dir / "labels", pm2.config.classes,
    )
    assert len(records) == 2

    store = LabelStore()
    result = merge_imported_records(
        pm2, store, records,
        conflict_mode="overwrite",
        read_image_size=lambda p: (640, 480),
    )
    assert result.imported == 2
    assert result.skipped == 0
    assert result.new_classes == ["leaf", "stem"]  # discovered from data.yaml
    assert pm2.config.classes == ["leaf", "stem"]

    # seg1: both polygons restored vertex-for-vertex with consistent bboxes.
    merged1 = store.load(pm2.label_path_for(dst_dir / "images" / "seg1.jpg"))
    assert merged1 is not None
    by_class = {a.class_name: a for a in merged1.annotations}
    assert set(by_class) == {"leaf", "stem"}
    _assert_polygons_close(by_class["leaf"].polygon, triangle)
    _assert_polygons_close(by_class["stem"].polygon, quad)
    for ann in merged1.annotations:
        assert ann.bbox == pytest.approx(bbox_from_polygon(ann.polygon))
        assert ann.confirmed is True
    assert by_class["leaf"].class_id == 0
    assert by_class["stem"].class_id == 1
    # Derived bbox matches the pre-export original within export rounding.
    assert by_class["leaf"].bbox == pytest.approx(bbox_from_polygon(triangle), abs=1e-5)

    # seg2: bbox-only annotation came back as its 4-corner polygon whose
    # derived bbox equals the original bbox.
    merged2 = store.load(pm2.label_path_for(dst_dir / "images" / "seg2.jpg"))
    assert merged2 is not None
    ann2 = merged2.annotations[0]
    assert ann2.class_name == "leaf"
    assert len(ann2.polygon) == 4
    _assert_polygons_close(ann2.polygon, bbox_to_polygon(box))
    assert ann2.bbox == pytest.approx(box, abs=1e-5)
