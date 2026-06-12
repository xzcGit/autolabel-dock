"""COCO format import/export."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation, Keypoint


def export_coco(
    image_annotations: list[ImageAnnotation],
    output_path: Path | str,
    classes: list[str],
    only_confirmed: bool = False,
) -> None:
    """Export annotations to COCO JSON format.

    For pose annotations, the keypoint label list (taken from the first
    annotation per category that has keypoints) is written into the matching
    ``categories[*].keypoints`` field so that importers can reconstruct names.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1
    # First-seen keypoint label list per class name (preserves order of definition).
    kpt_names_by_class: dict[str, list[str]] = {}

    for img_id, ia in enumerate(image_annotations, start=1):
        w_img, h_img = ia.image_size
        images.append({
            "id": img_id,
            "file_name": ia.image_path,
            "width": w_img,
            "height": h_img,
        })

        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if ann.bbox is None:
                continue

            cx, cy, bw, bh = ann.bbox
            # Convert normalized center to COCO pixel [x_tl, y_tl, w, h]
            x_tl = (cx - bw / 2) * w_img
            y_tl = (cy - bh / 2) * h_img
            pw = bw * w_img
            ph = bh * h_img

            cat_id = (classes.index(ann.class_name) + 1) if ann.class_name in classes else ann.class_id + 1

            coco_ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(x_tl, 2), round(y_tl, 2), round(pw, 2), round(ph, 2)],
                "area": round(pw * ph, 2),
                "iscrowd": 0,
            }

            if ann.keypoints:
                kps = []
                for kp in ann.keypoints:
                    kps.extend([round(kp.x * w_img, 2), round(kp.y * h_img, 2), kp.visible])
                coco_ann["keypoints"] = kps
                coco_ann["num_keypoints"] = len(ann.keypoints)
                kpt_names_by_class.setdefault(
                    ann.class_name, [kp.label for kp in ann.keypoints]
                )

            annotations.append(coco_ann)
            ann_id += 1

    categories = []
    for i, name in enumerate(classes):
        cat = {"id": i + 1, "name": name}
        if name in kpt_names_by_class:
            cat["keypoints"] = kpt_names_by_class[name]
        categories.append(cat)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")


def import_coco(coco_path: Path | str, classes: list[str] | None = None) -> list[ImageAnnotation]:
    """Import annotations from COCO JSON format."""
    coco_path = Path(coco_path)
    data = json.loads(coco_path.read_text(encoding="utf-8"))

    # Build lookups
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    cat_kpt_names: dict[int, list[str]] = {
        c["id"]: list(c["keypoints"])
        for c in data["categories"]
        if isinstance(c.get("keypoints"), list) and c["keypoints"]
    }
    img_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    results = []
    for img_id, img_info in sorted(img_map.items()):
        w_img = img_info["width"]
        h_img = img_info["height"]
        annotations = []

        for coco_ann in anns_by_image.get(img_id, []):
            x_tl, y_tl, pw, ph = coco_ann["bbox"]
            # Convert pixel [x_tl, y_tl, w, h] to normalized center
            cx = (x_tl + pw / 2) / w_img
            cy = (y_tl + ph / 2) / h_img
            bw = pw / w_img
            bh = ph / h_img

            cat_id = coco_ann["category_id"]
            class_name = cat_map.get(cat_id, str(cat_id))

            # Resolve class_id from provided classes list, or fallback
            if classes and class_name in classes:
                class_idx = classes.index(class_name)
            else:
                class_idx = cat_id - 1  # COCO is 1-indexed

            keypoints = []
            if "keypoints" in coco_ann:
                kps = coco_ann["keypoints"]
                kpt_names = cat_kpt_names.get(cat_id)
                for i in range(0, len(kps), 3):
                    kx = kps[i] / w_img if w_img else 0
                    ky = kps[i + 1] / h_img if h_img else 0
                    vis = int(kps[i + 2])
                    idx = i // 3
                    label = kpt_names[idx] if kpt_names and idx < len(kpt_names) else f"kp_{idx}"
                    keypoints.append(Keypoint(x=kx, y=ky, visible=vis, label=label))

            annotations.append(Annotation(
                class_name=class_name,
                class_id=class_idx,
                bbox=(cx, cy, bw, bh),
                keypoints=keypoints,
                confirmed=True,
                source="manual",
            ))

        results.append(ImageAnnotation(
            image_path=img_info["file_name"],
            image_size=(w_img, h_img),
            annotations=annotations,
        ))

    return results
