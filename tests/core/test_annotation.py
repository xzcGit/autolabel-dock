"""Tests for annotation data models."""
import uuid

from src.core.annotation import Annotation, ImageAnnotation, Keypoint, compute_iou, find_conflicts


class TestKeypoint:
    def test_create_keypoint(self):
        kp = Keypoint(x=0.5, y=0.3, visible=2, label="nose")
        assert kp.x == 0.5
        assert kp.y == 0.3
        assert kp.visible == 2
        assert kp.label == "nose"

    def test_to_dict(self):
        kp = Keypoint(x=0.25, y=0.15, visible=2, label="left_eye")
        d = kp.to_dict()
        assert d == {"x": 0.25, "y": 0.15, "visible": 2, "label": "left_eye"}

    def test_from_dict(self):
        d = {"x": 0.25, "y": 0.15, "visible": 2, "label": "left_eye"}
        kp = Keypoint.from_dict(d)
        assert kp.x == 0.25
        assert kp.label == "left_eye"

    def test_clamp_coordinates(self):
        kp = Keypoint(x=1.2, y=-0.1, visible=2, label="test")
        kp.clamp()
        assert kp.x == 1.0
        assert kp.y == 0.0


class TestAnnotation:
    def test_create_manual_bbox(self):
        ann = Annotation(
            class_name="person",
            class_id=0,
            bbox=(0.5, 0.4, 0.3, 0.6),
        )
        assert ann.class_name == "person"
        assert ann.bbox == (0.5, 0.4, 0.3, 0.6)
        assert ann.keypoints == []
        assert ann.confidence == 1.0
        assert ann.confirmed is True
        assert ann.source == "manual"
        # id should be a valid UUID
        uuid.UUID(ann.id)

    def test_create_auto_bbox(self):
        ann = Annotation(
            class_name="car",
            class_id=1,
            bbox=(0.6, 0.3, 0.25, 0.35),
            confidence=0.87,
            confirmed=False,
            source="auto",
        )
        assert ann.confirmed is False
        assert ann.source == "auto"
        assert ann.confidence == 0.87

    def test_to_dict_roundtrip(self):
        ann = Annotation(
            class_name="person",
            class_id=0,
            bbox=(0.5, 0.4, 0.3, 0.6),
            keypoints=[Keypoint(0.25, 0.15, 2, "nose")],
        )
        d = ann.to_dict()
        restored = Annotation.from_dict(d)
        assert restored.class_name == ann.class_name
        assert restored.bbox == ann.bbox
        assert len(restored.keypoints) == 1
        assert restored.keypoints[0].label == "nose"
        assert restored.id == ann.id

    def test_annotation_without_bbox(self):
        ann = Annotation(
            class_name="point",
            class_id=0,
            bbox=None,
            keypoints=[Keypoint(0.5, 0.5, 2, "center")],
        )
        assert ann.bbox is None
        d = ann.to_dict()
        assert d["bbox"] is None
        restored = Annotation.from_dict(d)
        assert restored.bbox is None

    def test_clamp_bbox(self):
        ann = Annotation(
            class_name="test",
            class_id=0,
            bbox=(0.5, 0.5, 1.2, 0.8),  # width exceeds
        )
        ann.clamp()
        x, y, w, h = ann.bbox
        # After clamp, bbox should not extend beyond [0,1]
        assert x - w / 2 >= 0.0
        assert x + w / 2 <= 1.0
        assert y - h / 2 >= 0.0
        assert y + h / 2 <= 1.0


class TestImageAnnotation:
    def test_create_empty(self):
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
        )
        assert ia.annotations == []
        assert ia.image_tags == []

    def test_to_dict_roundtrip(self):
        ann = Annotation(class_name="person", class_id=0, bbox=(0.5, 0.4, 0.3, 0.6))
        ia = ImageAnnotation(
            image_path="img_001.jpg",
            image_size=(1920, 1080),
            annotations=[ann],
            image_tags=["outdoor"],
        )
        d = ia.to_dict()
        restored = ImageAnnotation.from_dict(d)
        assert restored.image_path == "img_001.jpg"
        assert restored.image_size == (1920, 1080)
        assert len(restored.annotations) == 1
        assert restored.image_tags == ["outdoor"]

    def test_confirmed_count(self):
        a1 = Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True)
        a2 = Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8)
        ia = ImageAnnotation(
            image_path="test.jpg",
            image_size=(100, 100),
            annotations=[a1, a2],
        )
        assert ia.confirmed_count == 1
        assert ia.unconfirmed_count == 1

    def test_status(self):
        # Empty
        ia = ImageAnnotation(image_path="a.jpg", image_size=(100, 100))
        assert ia.status == "unlabeled"

        # All confirmed
        ia.annotations = [
            Annotation(class_name="a", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1), confirmed=True),
        ]
        assert ia.status == "confirmed"

        # Has unconfirmed
        ia.annotations.append(
            Annotation(class_name="b", class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), confirmed=False, source="auto", confidence=0.8),
        )
        assert ia.status == "pending"

    def test_status_classify_confirmed(self):
        """Classification: image_tags non-empty -> confirmed."""
        ia = ImageAnnotation(
            image_path="cat.jpg",
            image_size=(100, 100),
            image_tags=["cat"],
        )
        assert ia.status == "confirmed"

    def test_status_classify_unlabeled(self):
        """Classification: image_tags empty -> unlabeled."""
        ia = ImageAnnotation(
            image_path="unlabeled.jpg",
            image_size=(100, 100),
            image_tags=[],
        )
        assert ia.status == "unlabeled"


class TestComputeIou:
    def test_identical_boxes(self):
        bbox = (0.5, 0.5, 0.2, 0.2)
        assert abs(compute_iou(bbox, bbox) - 1.0) < 1e-9

    def test_no_overlap(self):
        a = (0.1, 0.1, 0.1, 0.1)  # [0.05, 0.05, 0.15, 0.15]
        b = (0.9, 0.9, 0.1, 0.1)  # [0.85, 0.85, 0.95, 0.95]
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0.5, 0.5, 0.4, 0.4)  # [0.3, 0.3, 0.7, 0.7]
        b = (0.6, 0.6, 0.4, 0.4)  # [0.4, 0.4, 0.8, 0.8]
        # Intersection: [0.4, 0.4, 0.7, 0.7] = 0.3 * 0.3 = 0.09
        # Union: 0.16 + 0.16 - 0.09 = 0.23
        iou = compute_iou(a, b)
        assert abs(iou - 0.09 / 0.23) < 1e-6

    def test_one_inside_other(self):
        outer = (0.5, 0.5, 0.6, 0.6)
        inner = (0.5, 0.5, 0.2, 0.2)
        iou = compute_iou(outer, inner)
        # Intersection = inner area = 0.04, union = 0.36
        assert abs(iou - 0.04 / 0.36) < 1e-6


class TestFindConflicts:
    def _make_ann(self, cx, cy, w, h, cls="person", confirmed=True, source="manual"):
        return Annotation(
            class_name=cls, class_id=0, bbox=(cx, cy, w, h),
            confirmed=confirmed, source=source,
            confidence=0.9 if source == "auto" else 1.0,
        )

    def test_no_existing(self):
        preds = [self._make_ann(0.5, 0.5, 0.2, 0.2, confirmed=False, source="auto")]
        conflicts, clean = find_conflicts([], preds)
        assert len(conflicts) == 0
        assert len(clean) == 1

    def test_no_overlap(self):
        existing = [self._make_ann(0.1, 0.1, 0.1, 0.1)]
        preds = [self._make_ann(0.9, 0.9, 0.1, 0.1, confirmed=False, source="auto")]
        conflicts, clean = find_conflicts(existing, preds)
        assert len(conflicts) == 0
        assert len(clean) == 1

    def test_overlap_same_class(self):
        existing = [self._make_ann(0.5, 0.5, 0.3, 0.3)]
        preds = [self._make_ann(0.52, 0.52, 0.3, 0.3, confirmed=False, source="auto")]
        conflicts, clean = find_conflicts(existing, preds, iou_threshold=0.5)
        assert len(conflicts) == 1
        assert len(clean) == 0
        assert conflicts[0][0] is existing[0]
        assert conflicts[0][1] is preds[0]

    def test_overlap_different_class_no_conflict(self):
        existing = [self._make_ann(0.5, 0.5, 0.3, 0.3, cls="person")]
        preds = [self._make_ann(0.52, 0.52, 0.3, 0.3, cls="car", confirmed=False, source="auto")]
        conflicts, clean = find_conflicts(existing, preds, iou_threshold=0.5)
        assert len(conflicts) == 0
        assert len(clean) == 1

    def test_only_confirmed_match(self):
        # Unconfirmed existing should not trigger conflict
        existing = [self._make_ann(0.5, 0.5, 0.3, 0.3, confirmed=False)]
        preds = [self._make_ann(0.52, 0.52, 0.3, 0.3, confirmed=False, source="auto")]
        conflicts, clean = find_conflicts(existing, preds, iou_threshold=0.5)
        assert len(conflicts) == 0
        assert len(clean) == 1

    def test_below_threshold(self):
        existing = [self._make_ann(0.3, 0.3, 0.2, 0.2)]
        preds = [self._make_ann(0.5, 0.5, 0.2, 0.2, confirmed=False, source="auto")]
        conflicts, clean = find_conflicts(existing, preds, iou_threshold=0.5)
        assert len(conflicts) == 0
        assert len(clean) == 1

    def test_pred_without_bbox(self):
        existing = [self._make_ann(0.5, 0.5, 0.3, 0.3)]
        pred = Annotation(class_name="person", class_id=0, bbox=None,
                          keypoints=[Keypoint(0.5, 0.5, 2, "pt")],
                          confirmed=False, source="auto")
        conflicts, clean = find_conflicts(existing, [pred])
        assert len(conflicts) == 0
        assert len(clean) == 1

    def test_greedy_one_existing_one_pred(self):
        # Two predictions overlap the same existing — only the best match wins
        existing = [self._make_ann(0.5, 0.5, 0.3, 0.3)]
        pred1 = self._make_ann(0.51, 0.51, 0.3, 0.3, confirmed=False, source="auto")
        pred2 = self._make_ann(0.55, 0.55, 0.3, 0.3, confirmed=False, source="auto")
        conflicts, clean = find_conflicts(existing, [pred1, pred2], iou_threshold=0.3)
        assert len(conflicts) == 1
        # pred1 has higher IoU with existing, so it should be the conflict
        assert conflicts[0][1] is pred1
        assert len(clean) == 1


def test_image_tags_confirmed_default_true():
    ia = ImageAnnotation(image_path="x.jpg", image_size=(10, 10))
    assert ia.image_tags_confirmed is True
    assert ia.image_tags_source == "manual"


def test_image_tags_to_dict_roundtrip_with_new_fields():
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10),
        image_tags=["cat"], image_tags_confirmed=False, image_tags_source="auto",
    )
    d = ia.to_dict()
    assert d["image_tags_confirmed"] is False
    assert d["image_tags_source"] == "auto"
    ia2 = ImageAnnotation.from_dict(d)
    assert ia2.image_tags_confirmed is False
    assert ia2.image_tags_source == "auto"


def test_image_tags_from_dict_legacy_defaults_to_confirmed_manual():
    """Legacy JSON without the new fields → treated as confirmed/manual."""
    legacy = {
        "image_path": "x.jpg",
        "image_size": [10, 10],
        "image_tags": ["cat"],
        "annotations": [],
    }
    ia = ImageAnnotation.from_dict(legacy)
    assert ia.image_tags_confirmed is True
    assert ia.image_tags_source == "manual"


def test_status_classify_pending_when_unconfirmed():
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10),
        image_tags=["cat"], image_tags_confirmed=False,
    )
    assert ia.status == "pending"


def test_status_classify_confirmed_when_confirmed():
    ia = ImageAnnotation(
        image_path="x.jpg", image_size=(10, 10),
        image_tags=["cat"], image_tags_confirmed=True,
    )
    assert ia.status == "confirmed"
