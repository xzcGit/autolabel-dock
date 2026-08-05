"""YOLO format import/export (detection + pose)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from src.core.annotation import Annotation, ImageAnnotation, Keypoint
from src.core.class_mapping import resolve_detection_class_map
from src.core.polygon import bbox_from_polygon, bbox_to_polygon


def _iter_label_files(labels_dir: Path) -> list[Path]:
    """Return YOLO label files, excluding auxiliary metadata files."""
    return [
        txt_path
        for txt_path in sorted(labels_dir.glob("*.txt"))
        if txt_path.name.lower() != "classes.txt"
    ]


def _find_classes_txt(labels_dir: Path) -> Path | None:
    """Search for classes.txt in the given dir or its parent."""
    for candidate in [
        labels_dir / "classes.txt",
        labels_dir.parent / "classes.txt",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_classes_txt(classes_txt: Path | str) -> list[str]:
    """Read class names from a YOLO classes.txt file."""
    return [
        line.strip()
        for line in Path(classes_txt).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_detection_fields(line: str, txt_path: Path, line_no: int) -> tuple[int, float, float, float, float]:
    """Parse the leading YOLO detection fields with context-rich errors."""
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"{txt_path.name}:{line_no}: expected at least 5 fields, got {len(parts)}")
    try:
        return (
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
    except ValueError as exc:
        raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc


def _normalize_detection_export_annotations(
    image_annotations: list[ImageAnnotation],
    classes: list[str],
) -> list[ImageAnnotation]:
    normalized: list[ImageAnnotation] = []

    for image_annotation in image_annotations:
        normalized_annotations = []
        for annotation in image_annotation.annotations:
            class_name = annotation.class_name
            if not class_name:
                if 0 <= annotation.class_id < len(classes):
                    class_name = classes[annotation.class_id]
                elif annotation.class_id >= 0:
                    class_name = str(annotation.class_id)
            normalized_annotations.append(replace(annotation, class_name=class_name))
        normalized.append(replace(image_annotation, annotations=normalized_annotations))

    return normalized


def export_yolo_detection(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO detection format (txt + data.yaml)."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = class_map.id_by_name[ann.class_name]
            cx, cy, w, h = ann.bbox
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    # data.yaml
    data = {
        "nc": len(class_map.names),
        "names": class_map.names,
    }
    (output_dir / "data.yaml").write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def import_yolo_detection(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
) -> list[ImageAnnotation]:
    """Import YOLO detection format. Provide classes or data_yaml."""
    labels_dir = Path(labels_dir)

    if classes is None and data_yaml:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        classes = data["names"]

    if classes is None:
        raise ValueError("Must provide classes or data_yaml")

    results = []
    for txt_path in _iter_label_files(labels_dir):
        annotations = []
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            cid, cx, cy, w, h = _parse_detection_fields(line.strip(), txt_path, line_no)
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),  # unknown without actual image
            annotations=annotations,
        ))
    return results


def export_yolo_pose(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    kpt_dim: int = 3,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO pose format."""
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue
            cid = class_map.id_by_name[ann.class_name]
            cx, cy, w, h = ann.bbox
            parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kp in ann.keypoints:
                parts.append(f"{kp.x:.6f}")
                parts.append(f"{kp.y:.6f}")
                if kpt_dim == 3:
                    parts.append(f"{kp.visible}")
            lines.append(" ".join(parts))
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    data = {
        "nc": len(class_map.names),
        "names": class_map.names,
    }
    (output_dir / "data.yaml").write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def import_yolo_pose(
    labels_dir: Path | str,
    classes: list[str],
    kpt_labels: list[str],
    kpt_dim: int = 3,
) -> list[ImageAnnotation]:
    """Import YOLO pose format."""
    labels_dir = Path(labels_dir)
    num_kpts = len(kpt_labels)

    results = []
    for txt_path in _iter_label_files(labels_dir):
        annotations = []
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.strip().split()
            cid, cx, cy, w, h = _parse_detection_fields(line.strip(), txt_path, line_no)
            keypoints = []
            kp_start = 5
            try:
                for i in range(num_kpts):
                    offset = kp_start + i * kpt_dim
                    kx = float(parts[offset])
                    ky = float(parts[offset + 1])
                    vis = int(float(parts[offset + 2])) if kpt_dim == 3 else 2
                    keypoints.append(Keypoint(x=kx, y=ky, visible=vis, label=kpt_labels[i]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=(cx, cy, w, h),
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),
            annotations=annotations,
        ))
    return results


def export_yolo_segment(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to YOLO-seg format (polygon txt + data.yaml).

    Each row is a *pure polygon*: ``cid x1 y1 x2 y2 ... xn yn`` (normalized,
    NO bbox fields — unlike pose). Ultralytics reshapes any file whose rows
    exceed 6 fields as (-1, 2), so a bbox-only annotation is folded to a
    4-corner rectangle rather than written as a 5-field row (which would
    corrupt the whole file). Degenerate polygons (<3 points) are dropped.
    """
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = _normalize_detection_export_annotations(image_annotations, classes)
    class_map = resolve_detection_class_map(
        normalized_annotations,
        classes,
        only_confirmed=only_confirmed,
    )

    for ia in normalized_annotations:
        stem = Path(ia.image_path).stem
        lines = []
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            points = ann.polygon or (bbox_to_polygon(ann.bbox) if ann.bbox else None)
            if not points or len(points) < 3:
                continue
            cid = class_map.id_by_name[ann.class_name]
            parts = [f"{cid}"]
            for px, py in points:
                parts.append(f"{px:.6f}")
                parts.append(f"{py:.6f}")
            lines.append(" ".join(parts))
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    data = {
        "nc": len(class_map.names),
        "names": class_map.names,
    }
    (output_dir / "data.yaml").write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def import_yolo_segment(
    labels_dir: Path | str,
    classes: list[str],
) -> list[ImageAnnotation]:
    """Import YOLO-seg format (variable-length polygon rows).

    Each row is ``cid x1 y1 ... xn yn`` (≥3 points). The polygon is stored
    verbatim and a derived axis-aligned bbox is computed so bbox-based logic
    (conflict detection, viewport culling, stats) works unchanged.
    """
    labels_dir = Path(labels_dir)

    results = []
    for txt_path in _iter_label_files(labels_dir):
        annotations = []
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                raise ValueError(
                    f"{txt_path.name}:{line_no}: segment row needs class + at least 3 points, "
                    f"got {len(parts)} fields"
                )
            try:
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc
            if len(coords) % 2 != 0:
                raise ValueError(
                    f"{txt_path.name}:{line_no}: odd coordinate count {len(coords)} "
                    "(polygon needs x,y pairs)"
                )
            polygon = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            annotations.append(Annotation(
                class_name=classes[cid] if cid < len(classes) else str(cid),
                class_id=cid,
                bbox=bbox_from_polygon(polygon),
                polygon=polygon,
                confirmed=True,
                source="manual",
            ))
        results.append(ImageAnnotation(
            image_path=txt_path.stem,
            image_size=(0, 0),
            annotations=annotations,
        ))
    return results


def _find_data_yaml(labels_dir: Path) -> Path | None:
    """Search for data.yaml in the given dir, parent dir, or sibling paths."""
    for candidate in [
        labels_dir / "data.yaml",
        labels_dir.parent / "data.yaml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _read_kpt_shape(data_yaml: Path | str | None) -> tuple[int, int] | None:
    """Read ``kpt_shape: [num_kpts, dim]`` from a data.yaml, else None.

    Never raises: a missing/corrupt yaml or a malformed kpt_shape just means
    "no authoritative pose signal" and format detection falls back to the
    row-length heuristics.
    """
    if not data_yaml:
        return None
    try:
        meta = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    raw = meta.get("kpt_shape") if isinstance(meta, dict) else None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return None
    return None


def _detect_yolo_format(
    labels_dir: Path,
    kpt_shape: tuple[int, int] | None = None,
    has_data_yaml: bool = False,
) -> tuple[str, int]:
    """Detect whether YOLO labels are detection, pose, or segment.

    Returns ("detection", 0), ("pose", num_keypoints), or ("segment", 0).

    Discriminators, in order:

    1. **Variable row lengths (any > 5 fields) ⇒ segment** — pose rows are
       fixed-length (``5 + k*dim``, keypoint count is constant for a model)
       while segment polygons vary per instance. This is the fix for the old
       code, which saw a >5-field polygon row and always guessed pose.
    2. Uniform rows ≤ 5 fields ⇒ detection.
    3. Uniform rows > 5 fields with ``kpt_shape`` in data.yaml ⇒ **pose**
       (authoritative — a trainable ultralytics pose data.yaml must declare
       kpt_shape; it also fixes the keypoint count and dim, which the old
       %3/%2 heuristic could get wrong for dim-2 datasets).
    4. Uniform rows > 5 fields next to a data.yaml WITHOUT kpt_shape ⇒
       **segment** — this is what our own YOLO-seg export looks like when all
       polygons share one vertex count (e.g. bbox-folded 4-corner rectangles).
    5. No data.yaml at all: legacy dim-3-first heuristic (a bare pose labels
       dir keeps importing as pose; a bare uniform-length segment dir is
       genuinely ambiguous and documented as a known limitation).
    """
    lengths: set[int] = set()
    for txt_path in _iter_label_files(labels_dir):
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lengths.add(len(line.split()))
    if not lengths:
        return "detection", 0

    # Variable-length rows longer than a bbox ⇒ segment polygons.
    if len(lengths) > 1 and any(n > 5 for n in lengths):
        return "segment", 0

    n = max(lengths)
    if n <= 5:
        return "detection", 0
    if kpt_shape is not None and kpt_shape[0] > 0:
        return "pose", kpt_shape[0]
    if has_data_yaml:
        return "segment", 0
    # Assume kpt_dim=3: extra columns = num_keypoints * 3
    extra = n - 5
    if extra % 3 == 0:
        return "pose", extra // 3
    elif extra % 2 == 0:
        return "pose", extra // 2
    # Fallback: treat as detection
    return "detection", 0


def import_yolo_auto(
    labels_dir: Path | str,
    classes: list[str] | None = None,
    data_yaml: Path | str | None = None,
    kpt_labels: list[str] | None = None,
    kpt_dim: int = 3,
) -> list[ImageAnnotation]:
    """Auto-detect YOLO format (detection, pose, or segment) and import accordingly.

    Searches for data.yaml in the directory and its parent.
    Falls back to numeric class names if no classes are available.
    """
    labels_dir = Path(labels_dir)

    # Resolve data.yaml once: names feed the class fallback, and kpt_shape is
    # the authoritative pose-vs-segment signal for uniform-length rows.
    if data_yaml is None:
        data_yaml = _find_data_yaml(labels_dir)

    if classes is None and data_yaml is not None:
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
        classes = data.get("names")

    if classes is None:
        classes_txt = _find_classes_txt(labels_dir)
        if classes_txt is not None:
            classes = _load_classes_txt(classes_txt)

    # Detect format
    kpt_shape = _read_kpt_shape(data_yaml)
    fmt, num_kpts = _detect_yolo_format(
        labels_dir, kpt_shape=kpt_shape, has_data_yaml=data_yaml is not None,
    )

    if fmt == "segment":
        if classes is None:
            classes = _infer_classes_from_files(labels_dir)
        return import_yolo_segment(labels_dir, classes)
    elif fmt == "pose" and num_kpts > 0:
        if classes is None:
            # Infer max class id from files to generate fallback names
            classes = _infer_classes_from_files(labels_dir)
        if kpt_labels is None:
            kpt_labels = [f"kp_{i}" for i in range(num_kpts)]
        if kpt_shape is not None:
            kpt_dim = kpt_shape[1]  # authoritative dim from data.yaml
        return import_yolo_pose(labels_dir, classes, kpt_labels, kpt_dim)
    else:
        if classes is None:
            classes = _infer_classes_from_files(labels_dir)
        return import_yolo_detection(labels_dir, classes)


def import_yolo_for_project(
    source: Path | str,
    project_classes: list[str] | None = None,
) -> list[ImageAnnotation]:
    """Record-importer adapter (registry shape: ``(source, project_classes)``).

    "External metadata wins": when a ``data.yaml`` or ``classes.txt`` exists
    in the source dir or its parent, ``project_classes`` is ignored so
    ``import_yolo_auto`` discovers the class names itself. This pre-check
    decides *whether to pass* project classes and must probe exactly these
    four locations (moved verbatim from the controller) — do NOT replace it
    with ``_find_data_yaml``/``_find_classes_txt``, whose search scope serves
    the auto-discovery fallback, not this decision.
    """
    p = Path(source)
    has_external_metadata = any(
        candidate.exists()
        for candidate in [
            p / "data.yaml",
            p.parent / "data.yaml",
            p / "classes.txt",
            p.parent / "classes.txt",
        ]
    )
    return import_yolo_auto(
        p,
        classes=None if has_external_metadata else project_classes,
    )


def _infer_classes_from_files(labels_dir: Path) -> list[str]:
    """Scan all txt files to find max class_id and generate numeric class names."""
    max_id = -1
    for txt_path in _iter_label_files(labels_dir):
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.strip().split()
            try:
                cid = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"{txt_path.name}:{line_no}: {exc}") from exc
            if cid > max_id:
                max_id = cid
    if max_id < 0:
        return []
    return [str(i) for i in range(max_id + 1)]
