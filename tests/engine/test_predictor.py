"""Tests for inference engine (predictor)."""
from unittest.mock import MagicMock

import torch

from src.engine.predictor import Predictor


class TestPredictor:
    def test_predict_defaults_to_class_id_matching(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.91])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "totally-different-name"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            project_classes=["person"],
        )

        assert len(annotations) == 1
        assert annotations[0].class_name == "person"
        assert annotations[0].class_id == 0

    def test_predict_returns_annotations(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0, 1])
        boxes.conf = torch.tensor([0.95, 0.80])
        boxes.xywhn = torch.tensor([[0.5, 0.4, 0.3, 0.6], [0.2, 0.3, 0.1, 0.2]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person", 1: "car"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict("test.jpg", conf=0.5, iou=0.45)

        assert len(annotations) == 2
        assert annotations[0].class_name == "person"
        assert annotations[0].confidence == 0.95
        assert annotations[0].confirmed is False
        assert annotations[0].source == "auto"
        assert annotations[0].bbox[0] == 0.5

    def test_predict_filters_by_project_classes(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0, 1])
        boxes.conf = torch.tensor([0.90, 0.85])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person", 1: "car"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            project_classes=["person"],
        )

        assert len(annotations) == 1
        assert annotations[0].class_name == "person"

    def test_predict_matches_project_classes_case_insensitively(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.91])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: " Person "}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            project_classes=["person"],
            class_match_mode="class_name",
        )

        assert len(annotations) == 1
        assert annotations[0].class_name == "person"
        assert annotations[0].class_id == 0

    def test_predict_class_id_mode_falls_back_when_project_classes_too_short(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([2])
        boxes.conf = torch.tensor([0.88])
        boxes.xywhn = torch.tensor([[0.4, 0.4, 0.2, 0.2]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {2: "car"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            project_classes=["person"],
        )

        assert len(annotations) == 1
        assert annotations[0].class_name == "car"
        assert annotations[0].class_id == 2

    def test_predict_keeps_raw_detection_when_project_filter_removes_everything(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.88])
        boxes.xywhn = torch.tensor([[0.4, 0.4, 0.2, 0.2]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "car"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            project_classes=["person"],
            class_match_mode="class_name",
        )

        assert len(annotations) == 1
        assert annotations[0].class_name == "car"
        assert annotations[0].class_id == 0

    def test_predict_with_keypoints(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.92])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.6]])

        kpts = MagicMock()
        kpts.xyn = torch.tensor([[[0.45, 0.3], [0.50, 0.35]]])
        kpts.conf = torch.tensor([[0.9, 0.8]])

        result = MagicMock()
        result.boxes = boxes
        result.keypoints = kpts
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict(
            "test.jpg", conf=0.5, iou=0.45,
            kpt_labels=["nose", "left_eye"],
        )

        assert len(annotations) == 1
        assert len(annotations[0].keypoints) == 2
        assert annotations[0].keypoints[0].label == "nose"
        assert annotations[0].keypoints[0].x == 0.45

    def test_predict_empty_result(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([])
        boxes.conf = torch.tensor([])
        boxes.xywhn = torch.zeros((0, 4))
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (480, 640)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations = predictor.predict("test.jpg", conf=0.5, iou=0.45)

        assert annotations == []

    def test_image_size_from_result(self):
        mock_model = MagicMock()
        boxes = MagicMock()
        boxes.cls = torch.tensor([0])
        boxes.conf = torch.tensor([0.9])
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.orig_shape = (1080, 1920)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        predictor = Predictor(mock_model)
        annotations, img_size = predictor.predict_with_size("test.jpg", conf=0.5, iou=0.45)

        assert img_size == (1920, 1080)


class TestClassNames:
    """Predictor.class_names() — PredictorProtocol capability seam."""

    def test_dict_names_sorted_by_key(self):
        mock_model = MagicMock()
        mock_model.names = {1: "bike", 0: "person", 2: "car"}
        assert Predictor(mock_model).class_names() == ["person", "bike", "car"]

    def test_list_names_preserve_order(self):
        mock_model = MagicMock()
        mock_model.names = ["person", "bike", "car"]
        assert Predictor(mock_model).class_names() == ["person", "bike", "car"]

    def test_tuple_names_preserve_order(self):
        mock_model = MagicMock()
        mock_model.names = ("cat", "dog")
        assert Predictor(mock_model).class_names() == ["cat", "dog"]

    def test_empty_names_returns_empty(self):
        mock_model = MagicMock()
        mock_model.names = {}
        assert Predictor(mock_model).class_names() == []

    def test_missing_names_returns_empty(self):
        mock_model = MagicMock(spec=[])  # no .names attribute
        assert Predictor(mock_model).class_names() == []

    def test_last_dropped_is_zero(self):
        """YOLO is fixed-vocabulary: it never drops on a name mismatch."""
        assert Predictor(MagicMock()).last_dropped == 0


class TestPredictorSegment:
    """Predictor._run masks.xyn → polygon + derived bbox (symmetric to keypoints)."""

    def _make_result(self, masks, n_boxes=1):
        boxes = MagicMock()
        boxes.cls = torch.tensor([0] * n_boxes)
        boxes.conf = torch.tensor([0.9] * n_boxes)
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.3, 0.3]] * n_boxes)
        result = MagicMock()
        result.boxes = boxes
        result.keypoints = None
        result.masks = masks
        result.orig_shape = (480, 640)
        return result

    def test_parses_polygon_from_masks_xyn(self):
        import numpy as np

        mock_model = MagicMock()
        masks = MagicMock()
        # One contour with 4 points (already sparse — simplify keeps it).
        masks.xyn = [np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]], dtype=np.float32)]
        result = self._make_result(masks)
        mock_model.names = {0: "leaf"}
        mock_model.predict.return_value = [result]

        anns = Predictor(mock_model).predict("t.jpg")
        assert len(anns) == 1
        assert anns[0].polygon is not None
        assert len(anns[0].polygon) >= 3
        # bbox is the model's box (not derived here), still present
        assert anns[0].bbox is not None

    def test_masks_none_leaves_polygon_none(self):
        """Non-seg model / no masks: behavior identical to detect (polygon=None)."""
        mock_model = MagicMock()
        result = self._make_result(None)
        mock_model.names = {0: "person"}
        mock_model.predict.return_value = [result]

        anns = Predictor(mock_model).predict("t.jpg")
        assert len(anns) == 1
        assert anns[0].polygon is None

    def test_empty_segment_placeholder_yields_no_polygon(self):
        """A degenerate mask contributes an empty (0, 2) array — guarded to None,
        the box still survives (list length is not shrunk)."""
        import numpy as np

        mock_model = MagicMock()
        masks = MagicMock()
        masks.xyn = [np.zeros((0, 2), dtype=np.float32)]
        result = self._make_result(masks)
        mock_model.names = {0: "leaf"}
        mock_model.predict.return_value = [result]

        anns = Predictor(mock_model).predict("t.jpg")
        assert len(anns) == 1
        assert anns[0].polygon is None
        assert anns[0].bbox is not None

    def test_boxes_and_masks_one_to_one(self):
        import numpy as np

        mock_model = MagicMock()
        masks = MagicMock()
        masks.xyn = [
            np.array([[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]], dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),  # second box has degenerate mask
        ]
        result = self._make_result(masks, n_boxes=2)
        mock_model.names = {0: "leaf"}
        mock_model.predict.return_value = [result]

        anns = Predictor(mock_model).predict("t.jpg")
        assert len(anns) == 2
        assert anns[0].polygon is not None  # first box got its contour
        assert anns[1].polygon is None       # second box's empty mask → None

    def test_simplification_runs_in_pixel_space_of_orig_shape(self):
        """_run must hand the image size to simplify_polygon so DP works in
        pixel space: on a 2000×100 image a 0.01-normalized vertical zigzag is
        ~1 px of noise and must be simplified away (normalized-space DP would
        keep it — dropping the img_size plumbing regresses this test)."""
        import numpy as np

        top = [[0.1 + 0.8 * i / 21, 0.2 + (0.01 if i % 2 else 0.0)] for i in range(22)]
        contour = np.array(top + [[0.9, 0.8], [0.1, 0.8]], dtype=np.float32)

        mock_model = MagicMock()
        masks = MagicMock()
        masks.xyn = [contour]
        result = self._make_result(masks)
        result.orig_shape = (100, 2000)  # (h, w) → img_size (2000, 100)
        mock_model.names = {0: "leaf"}
        mock_model.predict.return_value = [result]

        anns = Predictor(mock_model).predict("t.jpg")
        assert anns[0].polygon is not None
        assert len(anns[0].polygon) <= 8
        # Output stays normalized after the pixel-space round trip.
        for x, y in anns[0].polygon:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

