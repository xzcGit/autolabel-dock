"""Dataset preparation for YOLO training (train/val split, symlinks, data.yaml)."""
from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import yaml

from src.core.annotation import ImageAnnotation
from src.core.class_mapping import ResolvedClassMap, resolve_detection_class_map
from src.core.label_io import load_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.utils.fs import link_or_copy

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares a YOLO-compatible dataset from a project."""

    def __init__(self, project_manager: ProjectManager):
        self.pm = project_manager

    def prepare(
        self,
        output_dir: Path | str,
        task: str = "detect",
        val_ratio: float = 0.2,
        seed: int = 42,
        kpt_shape: list[int] | None = None,
        tag_filter: TagFilter | None = None,
    ) -> Path:
        """Prepare dataset and return path to data.yaml.

        When ``tag_filter`` is provided and non-empty, only images whose
        per-image ``tags`` match the filter are considered (see
        ``src.core.tags.TagFilter``). Passing ``None`` or an empty filter
        is a no-op — original behavior is preserved.
        """
        output_dir = Path(output_dir)

        # Clean previous dataset to avoid stale symlinks / ultralytics .cache files
        if output_dir.exists():
            shutil.rmtree(output_dir)

        classes = self.pm.config.classes

        # Collect labeled images
        labeled: list[tuple[Path, ImageAnnotation]] = []
        for img_path in self.pm.list_images():
            label_path = self.pm.label_path_for(img_path)
            ia = load_annotation(label_path)
            if ia is None:
                continue
            if tag_filter is not None and not tag_filter.matches(ia.tags):
                continue

            # Classification: check image_tags
            if task == "classify":
                if ia.image_tags and ia.image_tags_confirmed:
                    labeled.append((img_path, ia))
            # Detection/Pose: check confirmed annotations
            else:
                confirmed = [a for a in ia.annotations if a.confirmed]
                if not confirmed:
                    continue
                normalized = self._normalize_detection_or_pose_annotations(
                    ImageAnnotation(
                        image_path=ia.image_path,
                        image_size=ia.image_size,
                        annotations=confirmed,
                        image_tags=list(ia.image_tags),
                    ),
                    classes,
                )
                labeled.append((
                    img_path,
                    normalized,
                ))

        if not labeled:
            raise ValueError("没有找到已确认标注的图片，无法准备数据集")

        # Stratified split by primary class
        train_set, val_set = self._stratified_split(labeled, val_ratio, seed)

        if not train_set:
            raise ValueError("训练集为空，请减小验证集比例或增加标注数据")

        logger.info(
            "Dataset prepared: %d train, %d val (task=%s)",
            len(train_set), len(val_set), task,
        )

        if task == "classify":
            self._export_classify(output_dir, train_set, val_set)
            # Ultralytics' check_cls_dataset expects the dataset root directory
            # (it does `data_dir / "train"` directly). A data.yaml file is never
            # consulted for classify, so don't write one and return the root dir.
            return output_dir

        class_map = resolve_detection_class_map(
            [ia for _, ia in train_set + val_set],
            classes,
            only_confirmed=False,
        )
        self._export_detection_or_pose(output_dir, train_set, val_set, class_map, task)

        # Generate data.yaml for detect/pose
        data_yaml_path = output_dir / "data.yaml"
        data = self._build_data_yaml(
            output_dir,
            class_map.names,
            task,
            kpt_shape,
            has_val=bool(val_set),
        )
        data_yaml_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        return data_yaml_path

    def _stratified_split(
        self,
        items: list[tuple[Path, ImageAnnotation]],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[tuple[Path, ImageAnnotation]], list[tuple[Path, ImageAnnotation]]]:
        """Split items into train/val using stratified sampling by primary class."""
        if val_ratio <= 0:
            return items, []
        if val_ratio >= 1:
            return [], items

        by_class: dict[str, list[tuple[Path, ImageAnnotation]]] = defaultdict(list)
        for item in items:
            ia = item[1]
            # Classification: use image_tags[0]
            if ia.image_tags:
                primary_class = ia.image_tags[0]
            # Detection/Pose: use first annotation's class
            elif ia.annotations:
                primary_class = ia.annotations[0].class_name
            else:
                continue  # Skip items without class info
            by_class[primary_class].append(item)

        rng = random.Random(seed)
        train, val = [], []
        for cls_items in by_class.values():
            rng.shuffle(cls_items)
            n_val = max(1, round(len(cls_items) * val_ratio))
            if n_val >= len(cls_items):
                n_val = max(0, len(cls_items) - 1)
            val.extend(cls_items[:n_val])
            train.extend(cls_items[n_val:])

        return train, val

    def _normalize_detection_or_pose_annotations(
        self,
        image_annotation: ImageAnnotation,
        classes: list[str],
    ) -> ImageAnnotation:
        normalized_annotations = []
        for annotation in image_annotation.annotations:
            class_name = annotation.class_name
            if not class_name:
                if 0 <= annotation.class_id < len(classes):
                    class_name = classes[annotation.class_id]
                elif annotation.class_id >= 0:
                    class_name = str(annotation.class_id)
            normalized_annotations.append(replace(annotation, class_name=class_name))
        return replace(image_annotation, annotations=normalized_annotations)

    def _export_detection_or_pose(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
        class_map: ResolvedClassMap,
        task: str,
    ) -> None:
        """Export to YOLO detection/pose directory structure with symlinks."""
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            img_dir = output_dir / split_name / "images"
            lbl_dir = output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path, ia in split_data:
                link = img_dir / img_path.name
                if not link.exists():
                    link_or_copy(img_path, link)

                lines = []
                for ann in ia.annotations:
                    if ann.bbox is None:
                        continue
                    cid = class_map.id_by_name[ann.class_name]
                    cx, cy, w, h = ann.bbox
                    parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
                    if task == "pose" and ann.keypoints:
                        for kp in ann.keypoints:
                            parts.extend([f"{kp.x:.6f}", f"{kp.y:.6f}", f"{kp.visible}"])
                    lines.append(" ".join(parts))
                (lbl_dir / (img_path.stem + ".txt")).write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
                )

    def _export_classify(
        self,
        output_dir: Path,
        train_set: list[tuple[Path, ImageAnnotation]],
        val_set: list[tuple[Path, ImageAnnotation]],
    ) -> None:
        """Export to YOLO classification directory structure.

        All project classes get a subdirectory in every non-empty split — even
        classes with no images — so the alphabetical class index ultralytics
        derives from the folder list stays stable across runs. Empty folders
        are tolerated via ``ImageFolder(allow_empty=True)`` on torchvision >=0.18.
        """
        all_classes = list(self.pm.config.classes)
        for split_name, split_data in [("train", train_set), ("val", val_set)]:
            if not split_data:
                continue
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for cls_name in all_classes:
                (split_dir / cls_name).mkdir(parents=True, exist_ok=True)
            for img_path, ia in split_data:
                if ia.image_tags:
                    cls_name = ia.image_tags[0]
                else:
                    cls_name = ia.annotations[0].class_name
                cls_dir = split_dir / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                link = cls_dir / img_path.name
                if not link.exists():
                    link_or_copy(img_path, link)

    def _build_data_yaml(
        self,
        output_dir: Path,
        classes: list[str],
        task: str,
        kpt_shape: list[int] | None,
        has_val: bool = True,
    ) -> dict:
        """Build data.yaml content dict for detect/pose tasks.

        Classification does not use data.yaml — ultralytics reads classify datasets
        directly from the directory structure.
        """
        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "nc": len(classes),
            "names": classes,
        }
        if has_val:
            data["val"] = "val/images"
        if task == "pose" and kpt_shape:
            data["kpt_shape"] = kpt_shape
        return data
