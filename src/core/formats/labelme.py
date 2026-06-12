"""labelme JSON format import/export."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_labelme(
    image_annotations: list[ImageAnnotation],
    output_dir: Path | str,
    only_confirmed: bool = False,
) -> None:
    """Export annotations to labelme JSON format (one JSON per image).

    Pose annotations (bbox + keypoints) are linked via ``group_id`` so that
    importers can reconstruct the bbox <-> keypoints relationship.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ia in image_annotations:
        w_img, h_img = ia.image_size
        shapes = []
        next_group_id = 1

        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue

            # Assign a group_id only when bbox and keypoints need to be linked.
            group_id = next_group_id if (ann.bbox is not None and ann.keypoints) else None
            if group_id is not None:
                next_group_id += 1

            if ann.bbox is not None:
                cx, cy, bw, bh = ann.bbox
                x1 = (cx - bw / 2) * w_img
                y1 = (cy - bh / 2) * h_img
                x2 = (cx + bw / 2) * w_img
                y2 = (cy + bh / 2) * h_img
                shapes.append({
                    "label": ann.class_name,
                    "shape_type": "rectangle",
                    "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                    "group_id": group_id,
                    "flags": {},
                })

            for kp in ann.keypoints:
                shapes.append({
                    "label": kp.label,
                    "shape_type": "point",
                    "points": [[round(kp.x * w_img, 2), round(kp.y * h_img, 2)]],
                    "group_id": group_id,
                    "flags": {},
                })

        labelme_data = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": ia.image_path,
            "imageData": None,
            "imageWidth": w_img,
            "imageHeight": h_img,
        }

        stem = Path(ia.image_path).stem
        out_path = output_dir / f"{stem}.json"
        out_path.write_text(json.dumps(labelme_data, indent=2, ensure_ascii=False), encoding="utf-8")


def _shape_to_bbox(shape: dict, w_img: int, h_img: int) -> tuple[float, float, float, float] | None:
    points = shape.get("points", [])
    if len(points) != 2 or not w_img or not h_img:
        return None
    x1, y1 = points[0]
    x2, y2 = points[1]
    cx = ((x1 + x2) / 2) / w_img
    cy = ((y1 + y2) / 2) / h_img
    bw = abs(x2 - x1) / w_img
    bh = abs(y2 - y1) / h_img
    return (cx, cy, bw, bh)


def _shape_to_keypoint(shape: dict, w_img: int, h_img: int) -> Keypoint | None:
    points = shape.get("points", [])
    if len(points) != 1:
        return None
    px, py = points[0]
    return Keypoint(
        x=px / w_img if w_img else 0,
        y=py / h_img if h_img else 0,
        visible=2,
        label=shape.get("label", "unknown"),
    )


def import_labelme(input_dir: Path | str) -> list[ImageAnnotation]:
    """Import annotations from labelme JSON files in a directory.

    Shapes sharing a ``group_id`` are reconstructed into a single ``Annotation``
    (rectangle -> bbox + class_name, points -> keypoints). Shapes without a
    ``group_id`` keep the legacy per-shape behavior for backward compatibility.
    """
    input_dir = Path(input_dir)
    results = []

    for json_path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if "shapes" not in data:
            continue

        w_img = data.get("imageWidth", 0)
        h_img = data.get("imageHeight", 0)
        raw_image_path = data.get("imagePath") or (json_path.stem + ".jpg")
        # labelme on Windows writes paths like "..\\images\\foo.jpg"; on Linux
        # Path() will not split those backslashes, so the downstream stem-match
        # against project images breaks. Normalize to just the basename.
        image_path = raw_image_path.replace("\\", "/").rsplit("/", 1)[-1]

        # Bucket shapes: grouped (linked by group_id) vs ungrouped (legacy).
        groups: dict[int, list[dict]] = {}
        ungrouped: list[dict] = []
        for shape in data["shapes"]:
            gid = shape.get("group_id")
            if gid is None:
                ungrouped.append(shape)
            else:
                groups.setdefault(gid, []).append(shape)

        # Auto-merge heuristic: ungrouped single-instance pose.
        # When shapes have no group_id and there is exactly one rectangle plus
        # one or more points, treat them as a single pose Annotation. This is
        # the common case when labelme files come from external tools that
        # don't write group_id (e.g. converted from other annotation formats).
        ungrouped_rects = [s for s in ungrouped if s.get("shape_type") == "rectangle"]
        ungrouped_points = [s for s in ungrouped if s.get("shape_type") == "point"]
        auto_merge = (
            not groups
            and len(ungrouped_rects) == 1
            and len(ungrouped_points) >= 1
        )

        annotations: list[Annotation] = []

        if auto_merge:
            rect = ungrouped_rects[0]
            bb = _shape_to_bbox(rect, w_img, h_img)
            keypoints: list[Keypoint] = []
            for shape in ungrouped_points:
                kp = _shape_to_keypoint(shape, w_img, h_img)
                if kp is not None:
                    keypoints.append(kp)
            annotations.append(Annotation(
                class_name=rect.get("label", "unknown"),
                class_id=0,
                bbox=bb,
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))
            # Skip the per-shape legacy fall-through below; everything is merged.
            ungrouped = []

        # Reconstruct grouped shapes into single Annotations.
        for _gid, members in sorted(groups.items()):
            bbox: tuple[float, float, float, float] | None = None
            class_name = "unknown"
            keypoints: list[Keypoint] = []
            for shape in members:
                stype = shape.get("shape_type", "")
                if stype == "rectangle" and bbox is None:
                    bb = _shape_to_bbox(shape, w_img, h_img)
                    if bb is not None:
                        bbox = bb
                        class_name = shape.get("label", "unknown")
                elif stype == "point":
                    kp = _shape_to_keypoint(shape, w_img, h_img)
                    if kp is not None:
                        keypoints.append(kp)
            # Group with no rectangle: use first keypoint label as class_name fallback.
            if bbox is None and keypoints:
                class_name = members[0].get("label", "unknown")
            annotations.append(Annotation(
                class_name=class_name,
                class_id=0,
                bbox=bbox,
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))

        # Ungrouped: legacy per-shape behavior (each shape -> its own Annotation).
        for shape in ungrouped:
            stype = shape.get("shape_type", "")
            label = shape.get("label", "unknown")
            if stype == "rectangle":
                bb = _shape_to_bbox(shape, w_img, h_img)
                if bb is None:
                    continue
                annotations.append(Annotation(
                    class_name=label,
                    class_id=0,
                    bbox=bb,
                    confirmed=True,
                    source="manual",
                ))
            elif stype == "point":
                kp = _shape_to_keypoint(shape, w_img, h_img)
                if kp is None:
                    continue
                annotations.append(Annotation(
                    class_name=label,
                    class_id=0,
                    bbox=None,
                    keypoints=[kp],
                    confirmed=True,
                    source="manual",
                ))

        results.append(ImageAnnotation(
            image_path=image_path,
            image_size=(w_img, h_img),
            annotations=annotations,
        ))

    return results
