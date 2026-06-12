"""Project configuration and management."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.colors import assign_color

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class ProjectConfig:
    """Project configuration stored in project.json."""

    name: str
    image_dir: str  # relative to project dir
    label_dir: str  # relative to project dir
    classes: list[str]
    class_colors: dict[str, str] = field(default_factory=dict)
    keypoint_templates: dict[str, dict] = field(default_factory=dict)
    default_model: str = ""
    auto_label_conf: float = 0.5
    auto_label_iou: float = 0.45
    created_at: str = ""
    version: str = "1.0"
    task_type: str = "detect"  # "detect" | "pose" | "classify"
    auto_register_classes: bool = True
    # Project-level registry of known user tags. Per-image tag selections
    # live on ImageAnnotation.tags; this list is just the autocomplete source.
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "classes": self.classes,
            "class_colors": self.class_colors,
            "keypoint_templates": self.keypoint_templates,
            "default_model": self.default_model,
            "auto_label_conf": self.auto_label_conf,
            "auto_label_iou": self.auto_label_iou,
            "created_at": self.created_at,
            "version": self.version,
            "task_type": self.task_type,
            "auto_register_classes": self.auto_register_classes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectConfig:
        return cls(
            name=d["name"],
            image_dir=d["image_dir"],
            label_dir=d["label_dir"],
            classes=d["classes"],
            class_colors=d.get("class_colors", {}),
            keypoint_templates=d.get("keypoint_templates", {}),
            default_model=d.get("default_model", ""),
            auto_label_conf=d.get("auto_label_conf", 0.5),
            auto_label_iou=d.get("auto_label_iou", 0.45),
            created_at=d.get("created_at", ""),
            version=d.get("version", "1.0"),
            task_type=d.get("task_type", "detect"),
            auto_register_classes=d.get("auto_register_classes", True),
            tags=list(d.get("tags", [])),
        )

    def get_class_color(self, class_name: str) -> str:
        """Get color for a class. Uses custom color if set, otherwise auto-assigns from palette."""
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        idx = self.classes.index(class_name) if class_name in self.classes else 0
        return assign_color(idx)

    def get_class_id(self, class_name: str) -> int:
        """Get class index. Returns -1 if not found."""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1


class ProjectManager:
    """Manages a project directory and its configuration."""

    def __init__(self, project_dir: Path, config: ProjectConfig):
        self.project_dir = project_dir
        self.config = config

    @classmethod
    def create(
        cls,
        project_dir: Path | str,
        name: str,
        image_dir: str = "images",
        classes: list[str] | None = None,
        task_type: str = "detect",
    ) -> ProjectManager:
        """Create a new project.

        Args:
            project_dir: Path where the project will be created.
            name: Project name.
            image_dir: Either a relative subdir name (created inside project_dir)
                       or an absolute path to an existing image directory.
            classes: Initial class list.
            task_type: Task type - "detect", "pose", or "classify".
        """
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(image_dir)
        if image_path.is_absolute():
            # External image directory — store relative to project or absolute
            if not image_path.exists():
                image_path.mkdir(parents=True, exist_ok=True)
            try:
                rel = image_path.relative_to(project_dir)
                image_dir_str = str(rel)
            except ValueError:
                # Outside project dir — store absolute path
                image_dir_str = str(image_path)
        else:
            image_dir_str = image_dir
            (project_dir / image_dir_str).mkdir(exist_ok=True)

        label_dir = "labels"
        (project_dir / label_dir).mkdir(exist_ok=True)

        config = ProjectConfig(
            name=name,
            image_dir=image_dir_str,
            label_dir=label_dir,
            classes=classes or [],
            created_at=datetime.now().isoformat(timespec="seconds"),
            task_type=task_type,
        )
        pm = cls(project_dir, config)
        pm.save()
        return pm

    @classmethod
    def open(cls, project_dir: Path | str) -> ProjectManager:
        """Open an existing project."""
        project_dir = Path(project_dir)
        config_path = project_dir / "project.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No project.json found in {project_dir}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ProjectConfig.from_dict(data)
        return cls(project_dir, config)

    def save(self) -> None:
        """Save project config to project.json."""
        path = self.project_dir / "project.json"
        path.write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_images(self) -> list[Path]:
        """List all image files in the image directory, sorted by name."""
        img_dir = Path(self.config.image_dir)
        if not img_dir.is_absolute():
            img_dir = self.project_dir / img_dir
        if not img_dir.exists():
            return []
        return sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def label_path_for(self, image_path: Path | str) -> Path:
        """Get the label JSON path for a given image."""
        image_path = Path(image_path)
        return self.project_dir / self.config.label_dir / (image_path.stem + ".json")

    def delete_images(self, paths) -> tuple[int, int]:
        """Delete image files and their corresponding label JSONs from disk.

        Args:
            paths: Iterable of image paths (absolute) to delete.

        Returns:
            (images_deleted, labels_deleted) — count of files that actually existed
            and were removed.
        """
        img_deleted = 0
        lbl_deleted = 0
        for img_path in paths:
            img_path = Path(img_path)
            try:
                if img_path.exists():
                    img_path.unlink()
                    img_deleted += 1
            except OSError:
                logger.warning("Failed to delete image: %s", img_path, exc_info=True)
            label_path = self.label_path_for(img_path)
            try:
                if label_path.exists():
                    label_path.unlink()
                    lbl_deleted += 1
            except OSError:
                logger.warning("Failed to delete label: %s", label_path, exc_info=True)
        return img_deleted, lbl_deleted

    def add_class(self, class_name: str) -> None:
        """Add a class if it doesn't exist."""
        if class_name not in self.config.classes:
            self.config.classes.append(class_name)

    def remove_class(self, class_name: str) -> None:
        """Remove a class."""
        if class_name in self.config.classes:
            self.config.classes.remove(class_name)
            self.config.class_colors.pop(class_name, None)
