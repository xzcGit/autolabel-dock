"""Annotation data models for AutoLabel Dock."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Keypoint:
    """A single keypoint with normalized coordinates."""

    x: float
    y: float
    visible: int  # 0=invisible, 1=occluded, 2=visible
    label: str

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "visible": self.visible, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> Keypoint:
        return cls(x=d["x"], y=d["y"], visible=d["visible"], label=d["label"])

    def clamp(self) -> None:
        """Clamp coordinates to [0, 1]."""
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))


@dataclass
class Annotation:
    """A single annotation (bbox, keypoints, or both)."""

    class_name: str
    class_id: int
    bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 1.0
    confirmed: bool = True
    source: str = "manual"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Annotation:
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        keypoints = [Keypoint.from_dict(kp) for kp in d.get("keypoints", [])]
        return cls(
            id=d["id"],
            class_name=d["class_name"],
            class_id=d["class_id"],
            bbox=bbox,
            keypoints=keypoints,
            confidence=d.get("confidence", 1.0),
            confirmed=d.get("confirmed", True),
            source=d.get("source", "manual"),
        )

    def clamp(self) -> None:
        """Clamp bbox and keypoints to [0, 1] image bounds."""
        if self.bbox:
            cx, cy, w, h = self.bbox
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            self.bbox = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        for kp in self.keypoints:
            kp.clamp()


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""

    image_path: str
    image_size: tuple[int, int]  # (width, height)
    annotations: list[Annotation] = field(default_factory=list)
    image_tags: list[str] = field(default_factory=list)
    image_tags_confirmed: bool = True
    image_tags_source: str = "manual"
    # Free-form user tags for dataset organization / filtering. Distinct from
    # image_tags (which stores the classify task's class label).
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "image_tags": self.image_tags,
            "image_tags_confirmed": self.image_tags_confirmed,
            "image_tags_source": self.image_tags_source,
            "tags": self.tags,
            "annotations": [ann.to_dict() for ann in self.annotations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ImageAnnotation:
        return cls(
            image_path=d["image_path"],
            image_size=tuple(d["image_size"]),
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
            image_tags=d.get("image_tags", []),
            image_tags_confirmed=d.get("image_tags_confirmed", True),
            image_tags_source=d.get("image_tags_source", "manual"),
            tags=list(d.get("tags", [])),
        )

    @property
    def confirmed_count(self) -> int:

        return sum(1 for a in self.annotations if a.confirmed)

    @property
    def unconfirmed_count(self) -> int:
        return sum(1 for a in self.annotations if not a.confirmed)

    @property
    def status(self) -> str:
        """Return 'unlabeled', 'confirmed', or 'pending'."""
        if self.image_tags:
            return "confirmed" if self.image_tags_confirmed else "pending"
        if not self.annotations:
            return "unlabeled"
        if all(a.confirmed for a in self.annotations):
            return "confirmed"
        return "pending"


def compute_iou(bbox1: tuple[float, float, float, float],
                bbox2: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (cx, cy, w, h) normalized bboxes."""
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2
    # Convert to x1, y1, x2, y2
    ax1, ay1, ax2, ay2 = cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2
    bx1, by1, bx2, by2 = cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2
    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def find_conflicts(
    existing: list[Annotation],
    predictions: list[Annotation],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[Annotation, Annotation]], list[Annotation]]:
    """Match predictions against confirmed same-class existing annotations by IoU.

    Returns (conflict_pairs, non_conflict_predictions).
    Each existing annotation is matched to at most one prediction (greedy, highest IoU first).
    """
    confirmed = [a for a in existing if a.confirmed and a.bbox]
    matched_existing: set[str] = set()
    conflicts: list[tuple[Annotation, Annotation]] = []
    non_conflicts: list[Annotation] = []

    for pred in predictions:
        if not pred.bbox:
            non_conflicts.append(pred)
            continue
        best_iou = 0.0
        best_match: Annotation | None = None
        for ex in confirmed:
            if ex.id in matched_existing:
                continue
            if ex.class_name != pred.class_name:
                continue
            iou = compute_iou(ex.bbox, pred.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = ex
        if best_match is not None and best_iou >= iou_threshold:
            matched_existing.add(best_match.id)
            conflicts.append((best_match, pred))
        else:
            non_conflicts.append(pred)

    return conflicts, non_conflicts
