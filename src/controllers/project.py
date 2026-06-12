"""Project controller — create, open, export, class management."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QListWidgetItem

from src.core.config import AppConfig
from src.core.project import ProjectManager
from src.core.label_io import load_annotation, save_annotation
from src.core.backup import BackupManager
from src.ui.dialogs import NewProjectDialog, ExportDialog, ClassManagerDialog, ImportDialog

logger = logging.getLogger(__name__)


_IMAGENET_ID_RE = re.compile(r"^n\d{8}$")
_MAX_CLASS_NAME_LEN = 64


@dataclass
class RegistrationResult:
    """Outcome of ProjectController.register_auto_class."""
    action: Literal[
        "registered",
        "existing",
        "rejected_blacklist",
        "rejected_disabled",
        "rejected_invalid",
    ]
    applied_name: str | None
    reason: str


@dataclass
class ClassPreviewItem:
    """A model class diffed against the current project's class list."""
    model_name: str
    is_blacklisted: bool
    default_checked: bool


class ProjectController:
    """Handles project lifecycle: create, open, export, class management."""

    def __init__(self, app_config: AppConfig, config_path: Path, parent_widget: QWidget):
        self._app_config = app_config
        self._config_path = config_path
        self._parent = parent_widget
        self._project: ProjectManager | None = None
        self._backup_mgr: BackupManager | None = None

    @property
    def project(self) -> ProjectManager | None:
        return self._project

    @property
    def backup_manager(self) -> BackupManager | None:
        return self._backup_mgr

    def create_project(self) -> ProjectManager | None:
        """Show new project dialog and create. Returns ProjectManager or None."""
        dlg = NewProjectDialog(self._parent)
        if not dlg.exec_():
            return None
        name, proj_dir, image_dir, classes, task_type = dlg.get_values()
        if not name or not proj_dir:
            return None
        try:
            pm = ProjectManager.create(
                proj_dir, name,
                image_dir=image_dir or "images",
                classes=classes or None,
                task_type=task_type,
            )
            self._project = pm
            self._add_recent(pm)
            return pm
        except (OSError, ValueError) as e:
            logger.error("Failed to create project: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "错误", f"创建项目失败: {e}")
            return None

    def open_project_dialog(self) -> ProjectManager | None:
        """Show file dialog and open project. Returns ProjectManager or None."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "打开项目", "", "项目文件 (project.json)"
        )
        if not path:
            return None
        return self.open_project(Path(path).parent)

    def open_project(self, project_dir: Path) -> ProjectManager | None:
        """Open a project from directory. Returns ProjectManager or None."""
        try:
            pm = ProjectManager.open(project_dir)
            self._project = pm
            self._add_recent(pm)
            return pm
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to open project: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "打开失败", f"无法打开项目: {e}")
            return None

    def open_recent(self, item: QListWidgetItem) -> ProjectManager | None:
        """Open a project from the recent list. Returns ProjectManager or None."""
        project_dir = Path(item.text())
        pm = self.open_project(project_dir)
        if pm is None:
            # Remove invalid entry
            self._app_config.recent_projects = [
                p for p in self._app_config.recent_projects if p != item.text()
            ]
            self._app_config.save(self._config_path)
        return pm

    def export(self, project: ProjectManager) -> None:
        """Show export dialog and run export."""
        dlg = ExportDialog(self._parent)
        if not dlg.exec_():
            return
        fmt, out_dir, only_confirmed = dlg.get_values()
        if not out_dir:
            return
        try:
            # Auto-backup before export
            self.create_backup()
            annotations = []
            for img_path in project.list_images():
                label_path = project.label_path_for(img_path)
                ia = load_annotation(label_path)
                if ia:
                    annotations.append(ia)

            from src.core.formats import get_export_registry
            registry = get_export_registry()
            registry.export(
                fmt, annotations, out_dir,
                classes=project.config.classes,
                only_confirmed=only_confirmed,
                source_image_dir=project.project_dir / project.config.image_dir,
            )
            logger.info("Exported %s to %s", fmt, out_dir)
        except (OSError, ValueError, KeyError) as e:
            logger.error("Export failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导出失败", str(e))
            raise

    def import_annotations(self, project: ProjectManager) -> int | None:
        """Show import dialog and import annotations.

        Returns number of images imported, or None if cancelled/failed.
        New class names found in imported data are auto-added to project classes.
        """
        dlg = ImportDialog(self._parent)
        if not dlg.exec_():
            return None
        fmt, path, conflict_mode = dlg.get_values()
        if not fmt or not path:
            return None

        try:
            # Auto-backup before import
            self.create_backup()

            from src.core.formats import get_import_registry
            info = get_import_registry().get(fmt)
            if info is None:
                QMessageBox.warning(self._parent, "导入失败", f"未知的导入格式: {fmt}")
                return None

            if info.is_full_import:
                # Importer manages images + labels directly (e.g. ImageFolder)
                result = info.import_fn(Path(path), project)
                imported_count = result.get("imported", 0)
                skipped_count = result.get("skipped", 0)
                new_classes = result.get("classes", [])
                msg = f"成功导入 {imported_count} 张图片"
                if skipped_count:
                    msg += f"，跳过 {skipped_count} 张"
                if new_classes:
                    msg += f"\n类别: {', '.join(new_classes)}"
                QMessageBox.information(self._parent, "导入完成", msg)
                logger.info("Imported %s from %s: %d ok, %d skipped",
                            fmt, path, imported_count, skipped_count)
                return imported_count

            imported = self._invoke_importer(fmt, path, project.config.classes)
            if not imported:
                QMessageBox.information(self._parent, "提示", "未找到可导入的标注")
                return 0

            # Build lookup of existing project images by stem
            image_by_stem: dict[str, Path] = {
                p.stem: p for p in project.list_images()
            }

            # Collect new classes (preserve order, dedupe)
            existing_classes = set(project.config.classes)
            new_classes: list[str] = []
            for ia in imported:
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
            for ia in imported:
                stem = Path(ia.image_path).stem
                matched = image_by_stem.get(stem)
                if matched is None:
                    skipped_count += 1
                    continue

                label_path = project.label_path_for(matched)
                existing = load_annotation(label_path)

                # Re-map class_id to current project classes
                for ann in ia.annotations:
                    cid = project.config.get_class_id(ann.class_name)
                    if cid >= 0:
                        ann.class_id = cid

                # Determine image_size (prefer existing, else imported, else load from disk)
                img_size = (0, 0)
                if existing and existing.image_size != (0, 0):
                    img_size = existing.image_size
                elif ia.image_size != (0, 0):
                    img_size = ia.image_size
                else:
                    from src.utils.image import get_image_size
                    try:
                        img_size = get_image_size(matched)
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

                save_annotation(new_ia, label_path)
                imported_count += 1

            msg = f"成功导入 {imported_count} 个图片的标注"
            if skipped_count:
                msg += f"，跳过 {skipped_count} 个"
            if new_classes:
                msg += f"\n自动添加了 {len(new_classes)} 个新类别: {', '.join(new_classes)}"
            QMessageBox.information(self._parent, "导入完成", msg)
            logger.info("Imported %s from %s: %d ok, %d skipped", fmt, path, imported_count, skipped_count)
            return imported_count

        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            logger.error("Import failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导入失败", str(e))
            return None

    def _invoke_importer(self, fmt: str, path: str, classes: list[str]) -> list:
        """Invoke the appropriate importer. Each importer has a different signature."""
        from src.core.formats import get_import_registry
        from src.core.formats.yolo import import_yolo_auto
        from src.core.formats.coco import import_coco
        from src.core.formats.labelme import import_labelme

        registry = get_import_registry()
        info = registry.get(fmt)
        if info is None:
            raise ValueError(f"未知的导入格式: {fmt}")

        p = Path(path)
        if fmt == "YOLO":
            has_external_metadata = any(
                candidate.exists()
                for candidate in [
                    p / "data.yaml",
                    p.parent / "data.yaml",
                    p / "classes.txt",
                    p.parent / "classes.txt",
                ]
            )
            return import_yolo_auto(
                p,
                classes=None if has_external_metadata else (classes or None),
            )
        elif fmt == "COCO":
            return import_coco(p, classes=classes or None)
        elif fmt == "labelme":
            return import_labelme(p)
        else:
            raise ValueError(f"未实现的导入格式: {fmt}")

    def manage_classes(self, project: ProjectManager) -> bool:
        """Show class manager dialog. Returns True if classes were changed."""
        dlg = ClassManagerDialog(
            project.config.classes,
            project.config.class_colors,
            self._parent,
        )
        if dlg.exec_():
            self.create_backup()  # Auto-backup before class changes
            project.config.classes = dlg.get_classes()
            project.save()
            return True
        return False

    def register_auto_class(
        self, raw_name: str, *, force: bool = False,
    ) -> RegistrationResult:
        """Register a class produced by an auto-label prediction.

        - ``force=True`` skips the project-level ``auto_register_classes`` gate
          (used by the batch pre-dialog where the user has explicitly approved
          a list of new classes). It still applies name validation.
        - Idempotent: returns ``"existing"`` when the class is already known.
        - On success, persists ``project.json`` immediately so worker threads
          and subsequent calls see the updated class list.
        """
        if self._project is None:
            return RegistrationResult(
                action="rejected_invalid",
                applied_name=None,
                reason="no project loaded",
            )
        ok, name, reason_kind = self._validate_class_name(raw_name)
        if not ok:
            if reason_kind == "rejected_blacklist":
                return RegistrationResult(
                    action="rejected_blacklist",
                    applied_name=None,
                    reason=f"模型类名 '{name}' 疑似 ImageNet ID，已跳过",
                )
            return RegistrationResult(
                action="rejected_invalid",
                applied_name=None,
                reason="模型类名为空或过长",
            )
        if name in self._project.config.classes:
            return RegistrationResult(
                action="existing",
                applied_name=name,
                reason=f"类别 '{name}' 已存在",
            )
        if not force and not self._project.config.auto_register_classes:
            return RegistrationResult(
                action="rejected_disabled",
                applied_name=None,
                reason="未开启自动登记",
            )
        self._project.add_class(name)
        self._project.save()
        logger.info("Auto-registered class: %s", name)
        return RegistrationResult(
            action="registered",
            applied_name=name,
            reason=f"已新增类别 '{name}'",
        )

    @staticmethod
    def _validate_class_name(raw: str) -> tuple[bool, str, str | None]:
        name = (raw or "").strip()
        if not name or len(name) > _MAX_CLASS_NAME_LEN:
            return False, name, "rejected_invalid"
        if _IMAGENET_ID_RE.match(name):
            return False, name, "rejected_blacklist"
        return True, name, None

    def preview_model_classes(self, predictor) -> list[ClassPreviewItem]:
        """Diff predictor.model.names against project.classes.

        Returns one ClassPreviewItem per model class that is *not* in
        ``project.classes``. Already-registered classes are excluded — the
        dialog should only ask the user about new ones.
        """
        if predictor is None or self._project is None:
            return []
        model = getattr(predictor, "model", None)
        names = getattr(model, "names", None) if model is not None else None
        if not names:
            return []
        if isinstance(names, dict):
            iterable = names.values()
        elif isinstance(names, (list, tuple)):
            iterable = names
        else:
            return []
        existing = set(self._project.config.classes)
        items: list[ClassPreviewItem] = []
        seen: set[str] = set()
        for raw in iterable:
            ok, name, reason = self._validate_class_name(raw)
            if not ok and reason == "rejected_invalid":
                continue
            if name in existing or name in seen:
                continue
            seen.add(name)
            is_black = reason == "rejected_blacklist"
            items.append(
                ClassPreviewItem(
                    model_name=name,
                    is_blacklisted=is_black,
                    default_checked=not is_black,
                )
            )
        return items

    def _add_recent(self, pm: ProjectManager) -> None:
        self._app_config.add_recent_project(str(pm.project_dir))
        self._app_config.save(self._config_path)
        self._backup_mgr = BackupManager(pm.project_dir)

    def create_backup(self) -> Path | None:
        """Create a manual backup of the current project. Returns backup path."""
        if self._backup_mgr and self._project:
            return self._backup_mgr.create_backup(self._project.config.label_dir)
        return None

    def list_backups(self) -> list[dict]:
        """List available backups for the current project."""
        if self._backup_mgr:
            return self._backup_mgr.list_backups()
        return []

    def restore_backup(self, backup_name: str) -> bool:
        """Restore a backup by name. Returns True on success."""
        if self._backup_mgr and self._project:
            return self._backup_mgr.restore_backup(backup_name, self._project.config.label_dir)
        return False
