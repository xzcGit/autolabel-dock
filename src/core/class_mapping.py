"""Shared helpers for resolving detection class mappings."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.annotation import ImageAnnotation


@dataclass(frozen=True)
class ResolvedClassMap:
    names: list[str]
    id_by_name: dict[str, int]


def resolve_detection_class_map(
    image_annotations: list[ImageAnnotation],
    project_classes: list[str],
    *,
    only_confirmed: bool,
) -> ResolvedClassMap:
    seen: set[str] = set()
    extras: list[str] = []

    for image_annotation in image_annotations:
        for annotation in image_annotation.annotations:
            if annotation.bbox is None:
                continue
            if only_confirmed and not annotation.confirmed:
                continue
            if annotation.class_name:
                seen.add(annotation.class_name)

    ordered = [name for name in project_classes if name in seen]

    for image_annotation in image_annotations:
        for annotation in image_annotation.annotations:
            if annotation.bbox is None:
                continue
            if only_confirmed and not annotation.confirmed:
                continue
            if (
                annotation.class_name
                and annotation.class_name in seen
                and annotation.class_name not in ordered
                and annotation.class_name not in extras
            ):
                extras.append(annotation.class_name)

    names = ordered + extras
    return ResolvedClassMap(
        names=names,
        id_by_name={name: idx for idx, name in enumerate(names)},
    )
