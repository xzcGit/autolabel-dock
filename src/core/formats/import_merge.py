"""Qt-free merge pipeline for imported label records.

Takes the records a record-importer produced (``ImportRegistry.import_records``)
and folds them into the project's label records: stem-matching against project
images, new-class collection, class_id re-mapping, image-size resolution and
the three conflict modes (skip / overwrite / merge).

Dependency direction (both injected by the caller — the controller keeps the
Qt side):
- Record IO goes through the injected ``LabelStore`` (flush-first reads); do
  not swap it back to raw ``label_io``.
- The image-size fallback reader is an injected callable so this module never
  imports ``src.utils.image`` (``get_image_size`` is a Qt dependency — same
  precedent as LabelStore never computing sizes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.core.annotation import ImageAnnotation
from src.core.label_store import LabelStore
from src.core.project import ProjectManager

logger = logging.getLogger(__name__)


@dataclass
class ImportMergeResult:
    """Outcome of ``merge_imported_records``."""

    imported: int
    skipped: int
    new_classes: list[str]


def merge_imported_records(
    project: ProjectManager,
    store: LabelStore,
    records: list[ImageAnnotation],
    conflict_mode: str,
    read_image_size: Callable[[Path], tuple[int, int]],
) -> ImportMergeResult:
    """Merge imported records into the project's label records.

    - New class names (annotation ``class_name`` AND ``image_tags``) are
      appended to ``project.config.classes`` in first-seen order (deduped);
      ``project.save()`` runs only when there is something new.
    - Records whose image stem has no match in the project are skipped.
    - ``class_id`` is re-resolved against the (possibly updated) class list.
    - ``image_size`` resolution: existing record > imported record >
      ``read_image_size(matched)`` fallback > ``(0, 0)`` (reader exceptions
      collapse to ``(0, 0)``).
    - ``conflict_mode``: ``"skip"`` keeps an existing annotated record,
      ``"overwrite"`` replaces it, ``"merge"`` appends annotations and adopts
      imported ``image_tags`` (+ confirmed/source companions) only when the
      existing record has none.
    """
    # Build lookup of existing project images by stem
    image_by_stem: dict[str, Path] = {
        p.stem: p for p in project.list_images()
    }

    # Collect new classes (preserve order, dedupe)
    existing_classes = set(project.config.classes)
    new_classes: list[str] = []
    for ia in records:
        for ann in ia.annotations:
            if ann.class_name and ann.class_name not in existing_classes:
                existing_classes.add(ann.class_name)
                new_classes.append(ann.class_name)
        for tag in ia.image_tags:
            if tag and tag not in existing_classes:
                existing_classes.add(tag)
                new_classes.append(tag)
    if new_classes:
        project.config.classes.extend(new_classes)
        project.save()

    # Re-resolve class_id against (possibly updated) project classes
    imported_count = 0
    skipped_count = 0
    for ia in records:
        stem = Path(ia.image_path).stem
        matched = image_by_stem.get(stem)
        if matched is None:
            skipped_count += 1
            continue

        label_path = project.label_path_for(matched)
        existing = store.load(label_path)

        # Re-map class_id to current project classes
        for ann in ia.annotations:
            cid = project.config.get_class_id(ann.class_name)
            if cid >= 0:
                ann.class_id = cid

        # Determine image_size (prefer existing, else imported, else read from disk)
        img_size = (0, 0)
        if existing and existing.image_size != (0, 0):
            img_size = existing.image_size
        elif ia.image_size != (0, 0):
            img_size = ia.image_size
        else:
            try:
                img_size = read_image_size(matched)
            except (OSError, ValueError):
                img_size = (0, 0)

        # Apply conflict resolution
        if conflict_mode == "skip" and existing and existing.annotations:
            skipped_count += 1
            continue
        elif conflict_mode == "overwrite" or existing is None:
            new_ia = ia
            new_ia.image_path = matched.name
            new_ia.image_size = img_size
        elif conflict_mode == "merge":
            existing.annotations.extend(ia.annotations)
            if existing.image_size == (0, 0):
                existing.image_size = img_size
            if ia.image_tags and not existing.image_tags:
                existing.image_tags = list(ia.image_tags)
                existing.image_tags_confirmed = ia.image_tags_confirmed
                existing.image_tags_source = ia.image_tags_source
            new_ia = existing
        else:
            # Fallback: treat as overwrite when no existing
            new_ia = ia
            new_ia.image_path = matched.name
            new_ia.image_size = img_size

        store.save(new_ia, label_path)
        imported_count += 1

    return ImportMergeResult(
        imported=imported_count,
        skipped=skipped_count,
        new_classes=new_classes,
    )
