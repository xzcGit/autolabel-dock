"""ImageFolder format import/export for classification tasks."""
from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path

from src.core.annotation import ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager, IMAGE_EXTENSIONS
from src.utils.fs import link_or_copy

logger = logging.getLogger(__name__)


class ImageFolderImporter:
    """Import ImageFolder format classification datasets."""

    def scan_structure(
        self, source_dir: Path, merge_splits: bool = True
    ) -> dict[str, list[Path]]:
        """Scan ImageFolder directory and return class->images mapping."""
        source_dir = Path(source_dir)
        structure: dict[str, list[Path]] = defaultdict(list)

        has_splits = (source_dir / "train").is_dir() or (source_dir / "val").is_dir()

        if has_splits and merge_splits:
            for split_dir in ["train", "val", "test"]:
                split_path = source_dir / split_dir
                if not split_path.is_dir():
                    continue
                for class_dir in split_path.iterdir():
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                            structure[class_name].append(img_path)
        else:
            for class_dir in source_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        structure[class_name].append(img_path)

        return dict(structure)

    def import_to_project(
        self,
        source_dir: Path,
        project: ProjectManager,
        copy_mode: bool = True,
        merge_splits: bool = True,
    ) -> dict:
        """Import ImageFolder dataset to project."""
        source_dir = Path(source_dir)
        structure = self.scan_structure(source_dir, merge_splits)

        if not structure:
            logger.warning("No images found in %s", source_dir)
            return {"imported": 0, "skipped": 0, "classes": []}

        all_classes = sorted(structure.keys())
        for cls in all_classes:
            if cls not in project.config.classes:
                project.config.classes.append(cls)
        project.save()

        images_dir = project.project_dir / project.config.image_dir
        labels_dir = project.project_dir / project.config.label_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        skipped = 0
        used_names: set[str] = set()

        for class_name, img_paths in structure.items():
            for src_path in img_paths:
                dest_name = src_path.name
                if dest_name in used_names:
                    dest_name = f"{class_name}_{src_path.name}"

                dest_path = images_dir / dest_name
                used_names.add(dest_name)

                try:
                    if copy_mode:
                        shutil.copy2(src_path, dest_path)
                    else:
                        link_or_copy(src_path, dest_path)
                except Exception as e:
                    logger.error("Failed to import %s: %s", src_path, e)
                    skipped += 1
                    continue

                img_size = (640, 480)  # Placeholder, updated on first load
                ia = ImageAnnotation(
                    image_path=str(dest_path.name),
                    image_size=img_size,
                    image_tags=[class_name],
                    image_tags_confirmed=True,
                    image_tags_source="manual",
                    annotations=[],
                )
                label_path = labels_dir / (dest_path.stem + ".json")
                save_annotation(ia, label_path)

                imported += 1

        logger.info(
            "Imported %d images from ImageFolder (%d skipped), classes: %s",
            imported, skipped, all_classes,
        )

        return {"imported": imported, "skipped": skipped, "classes": all_classes}


# ── Export ───────────────────────────────────────────────────────


def _resolve_source(ia: ImageAnnotation, source_image_dir: Path | None) -> Path | None:
    """Resolve image_path to an existing absolute path. Returns None if unresolvable."""
    p = Path(ia.image_path)
    if p.is_absolute():
        return p if p.exists() else None
    if source_image_dir is not None:
        candidate = Path(source_image_dir) / p.name
        if candidate.exists():
            return candidate
    return None


def export_imagefolder(
    arg1,
    output_dir: Path | str,
    only_confirmed: bool = False,
    source_image_dir: Path | str | None = None,
) -> None:
    """Export to ImageFolder format (folder-per-class).

    Two call modes (auto-detected by first argument type):
    - export_imagefolder(project: ProjectManager, output_dir, only_confirmed=False)
    - export_imagefolder(annotations: list[ImageAnnotation], output_dir,
                         only_confirmed=False, source_image_dir=...)
      In annotations mode, ia.image_path may be relative; pass source_image_dir
      so the exporter can locate the source file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(arg1, ProjectManager):
        project = arg1
        from src.core.label_io import load_annotation
        annotations = []
        for img_path in project.list_images():
            ia = load_annotation(project.label_path_for(img_path))
            if ia is None:
                continue
            annotations.append(ia)
        source = project.project_dir / project.config.image_dir
    else:
        annotations = arg1
        source = Path(source_image_dir) if source_image_dir is not None else None

    exported = 0
    skipped = 0
    for ia in annotations:
        if not ia.image_tags:
            skipped += 1
            continue
        if only_confirmed and not ia.image_tags_confirmed:
            skipped += 1
            continue
        src = _resolve_source(ia, source)
        if src is None:
            logger.warning("Skipping %s: source image not found", ia.image_path)
            skipped += 1
            continue
        cls_name = ia.image_tags[0]
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(exist_ok=True)
        dst = cls_dir / src.name
        try:
            shutil.copy2(src, dst)
            exported += 1
        except OSError as e:
            logger.error("Failed to export %s: %s", src, e)
            skipped += 1

    logger.info("Exported %d images to ImageFolder (%d skipped)", exported, skipped)


def export_csv(
    arg1,
    output_dir: Path | str,
    only_confirmed: bool = False,
    source_image_dir: Path | str | None = None,
) -> None:
    """Export project labels to CSV (filename,class).

    Same dispatch contract as ``export_imagefolder``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(arg1, ProjectManager):
        project = arg1
        from src.core.label_io import load_annotation
        annotations = []
        for img_path in project.list_images():
            ia = load_annotation(project.label_path_for(img_path))
            if ia is None:
                continue
            annotations.append(ia)
        source = project.project_dir / project.config.image_dir
    else:
        annotations = arg1
        source = Path(source_image_dir) if source_image_dir is not None else None

    rows = ["filename,class"]
    exported = 0
    skipped = 0

    for ia in annotations:
        if not ia.image_tags:
            skipped += 1
            continue
        if only_confirmed and not ia.image_tags_confirmed:
            skipped += 1
            continue
        src = _resolve_source(ia, source)
        if src is None:
            logger.warning("Skipping %s: source image not found", ia.image_path)
            skipped += 1
            continue
        rows.append(f"{src.name},{ia.image_tags[0]}")
        exported += 1

    (output_dir / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    logger.info("Exported %d labels to CSV (%d skipped)", exported, skipped)
