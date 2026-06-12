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
