"""Inference adapter for Ultralytics-style prediction results."""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.annotation import Annotation, Keypoint

logger = logging.getLogger(__name__)


class Predictor:
    """Wraps a YOLO model for inference, converting results to Annotations."""

    def __init__(self, model):
        """Initialize with a loaded ultralytics YOLO model instance."""
        self.model = model

    def release(self) -> None:
        """No-op release. Ultralytics models share the in-process runtime and
        don't require explicit GPU teardown; provided so the predictor satisfies
        the optional ``PredictorProtocol.release`` hook used by ModelController."""

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        """Normalize class names for tolerant matching across model/project metadata."""
        return " ".join(str(class_name).split()).casefold()

    @staticmethod
    def _resolve_model_class_name(names, cls_id: int) -> str:
        """Resolve a class name from ultralytics model.names, tolerating list/dict variants."""
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
    ) -> list[Annotation]:
        """Run inference and return list of Annotations."""
        annotations, _ = self._run(
            image_path, conf, iou, project_classes, class_match_mode, kpt_labels,
        )
        return annotations

    def predict_classify(
        self,
        image_path: str | Path,
        project_classes: list[str] | None = None,
        filter_to_project: bool = True,
    ) -> tuple[str, float] | None:
        """Run classify inference. Return (class_name, confidence) or None.

        When ``filter_to_project`` is True (default) and ``project_classes`` is
        non-empty, predictions whose class is not in ``project_classes`` return
        None (legacy behavior). When False, the raw model class name is
        returned even if it is not in the project — caller handles registration.
        """
        results = self.model.predict(source=str(image_path), verbose=False)
        if not results:
            return None
        return self._parse_classify_result(
            results[0],
            image_path=image_path,
            project_classes=project_classes,
            filter_to_project=filter_to_project,
        )

    def predict_classify_batch(
        self,
        image_paths: list[str | Path],
        project_classes: list[str] | None = None,
        filter_to_project: bool = True,
    ) -> list[tuple[str, float] | None]:
        """Run classify inference for multiple images in one model call."""
        if not image_paths:
            return []
        sources = [str(path) for path in image_paths]
        results = self.model.predict(source=sources, verbose=False)
        if not results:
            return [None] * len(sources)
        if len(results) != len(sources):
            logger.warning(
                "Classify batch returned %d results for %d sources; padding missing results with None",
                len(results),
                len(sources),
            )
        payloads = [
            self._parse_classify_result(
                result,
                image_path=image_path,
                project_classes=project_classes,
                filter_to_project=filter_to_project,
            )
            for image_path, result in zip(image_paths, results)
        ]
        if len(payloads) < len(sources):
            payloads.extend([None] * (len(sources) - len(payloads)))
        return payloads

    def _parse_classify_result(
        self,
        result,
        *,
        image_path: str | Path,
        project_classes: list[str] | None,
        filter_to_project: bool,
    ) -> tuple[str, float] | None:
        probs = getattr(result, "probs", None)
        if probs is None:
            return None
        cls_id = int(probs.top1)
        confidence = round(float(probs.top1conf.item()), 4)
        raw_name = self._resolve_model_class_name(self.model.names, cls_id)
        if not project_classes or not filter_to_project:
            return (raw_name, confidence)
        norm = self._normalize_class_name(raw_name)
        for cls in project_classes:
            if self._normalize_class_name(cls) == norm:
                return (cls, confidence)
        logger.warning(
            "Classify prediction '%s' not in project classes %s for %s",
            raw_name, project_classes, image_path,
        )
        return None

    def predict_with_size(
        self,
        image_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        """Run inference and return annotations + image size (w, h)."""
        return self._run(
            image_path, conf, iou, project_classes, class_match_mode, kpt_labels,
        )

    def _run(
        self,
        image_path: str | Path,
        conf: float,
        iou: float,
        project_classes: list[str] | None,
        class_match_mode: str,
        kpt_labels: list[str] | None,
    ) -> tuple[list[Annotation], tuple[int, int]]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            verbose=False,
        )
        logger.debug("Predict: %s (conf=%.2f, iou=%.2f)", image_path, conf, iou)
        if not results:
            return [], (0, 0)

        result = results[0]
        h, w = result.orig_shape
        img_size = (w, h)
        names = self.model.names
        matched_annotations = []
        raw_annotations = []
        project_class_lookup: dict[str, tuple[str, int]] = {}
        if project_classes and class_match_mode == "class_name":
            project_class_lookup = {
                self._normalize_class_name(class_name): (class_name, idx)
                for idx, class_name in enumerate(project_classes)
            }

        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            return [], img_size

        has_kpts = result.keypoints is not None

        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i].item())
            confidence = round(float(boxes.conf[i].item()), 4)
            raw_class_name = self._resolve_model_class_name(names, cls_id)
            project_match = None
            if project_classes:
                if class_match_mode == "class_id":
                    if 0 <= cls_id < len(project_classes):
                        project_match = (project_classes[cls_id], cls_id)
                elif class_match_mode == "class_name":
                    project_match = project_class_lookup.get(self._normalize_class_name(raw_class_name))
                else:
                    raise ValueError(f"Unsupported class_match_mode: {class_match_mode}")
            if project_match is not None:
                class_name, resolved_id = project_match
            else:
                class_name = raw_class_name
                resolved_id = cls_id

            cx = round(float(boxes.xywhn[i][0].item()), 6)
            cy = round(float(boxes.xywhn[i][1].item()), 6)
            bw = round(float(boxes.xywhn[i][2].item()), 6)
            bh = round(float(boxes.xywhn[i][3].item()), 6)

            keypoints = []
            if has_kpts and result.keypoints.xyn is not None:
                kpts_xy = result.keypoints.xyn[i]
                kpts_conf = result.keypoints.conf[i] if result.keypoints.conf is not None else None
                for j in range(len(kpts_xy)):
                    kx = round(float(kpts_xy[j][0].item()), 6)
                    ky = round(float(kpts_xy[j][1].item()), 6)
                    kc = float(kpts_conf[j].item()) if kpts_conf is not None else 1.0
                    visible = 2 if kc > 0.5 else (1 if kc > 0 else 0)
                    label = kpt_labels[j] if kpt_labels and j < len(kpt_labels) else f"kp_{j}"
                    keypoints.append(Keypoint(x=kx, y=ky, visible=visible, label=label))

            annotation = Annotation(
                class_name=class_name,
                class_id=resolved_id,
                bbox=(cx, cy, bw, bh),
                keypoints=keypoints,
                confidence=confidence,
                confirmed=False,
                source="auto",
            )
            raw_annotations.append(annotation)
            if not project_classes or project_match is not None:
                matched_annotations.append(annotation)

        if project_classes and not matched_annotations and raw_annotations:
            # Avoid reporting "no objects" when detections exist but class metadata does not align.
            logger.warning(
                "Predict filtered out all %d detections for %s because model classes did not match "
                "project classes %s; returning raw detections instead",
                len(raw_annotations),
                image_path,
                project_classes,
            )
            annotations = raw_annotations
        else:
            annotations = matched_annotations if project_classes else raw_annotations

        logger.debug(
            "Predict result: %d annotations (raw=%d, matched=%d)",
            len(annotations),
            len(raw_annotations),
            len(matched_annotations),
        )
        return annotations, img_size
