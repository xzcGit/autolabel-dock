"""Read/write the internal JSON annotation format."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.core.annotation import ImageAnnotation

logger = logging.getLogger(__name__)


def save_annotation(image_annotation: ImageAnnotation, path: Path | str) -> None:
    """Save an ImageAnnotation to a JSON file.

    If the annotation carries no annotations, no image_tags, and no user tags,
    any existing file at ``path`` is removed instead — an empty record is
    equivalent to no file on disk and keeps the labels directory clean.
    """
    path = Path(path)
    if (
        not image_annotation.annotations
        and not image_annotation.image_tags
        and not image_annotation.tags
    ):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to delete empty label: %s", path, exc_info=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = image_annotation.to_dict()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_annotation(path: Path | str) -> ImageAnnotation | None:
    """Load an ImageAnnotation from a JSON file. Returns None if file missing or corrupted."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ImageAnnotation.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to load annotation %s: %s", path, e)
        return None
