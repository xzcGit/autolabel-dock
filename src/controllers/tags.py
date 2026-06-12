"""Controller for user-defined image tags.

Bridges the UI (TagChipBar / TagFilterBar / TagManagerDialog) with the core
data layer. The actual filter object (``TagFilter``) and CRUD helpers live in
``src.core.tags`` — this class only adds Qt signaling, backup integration,
and per-image disk IO.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from src.core.annotation import ImageAnnotation
from src.core.backup import BackupManager
from src.core.label_io import load_annotation, save_annotation
from src.core.project import ProjectManager
from src.core.tags import (
    TagError, TagFilter, TagService, dedupe_preserving_order, normalize,
)
from src.utils.image import get_image_size

logger = logging.getLogger(__name__)


class TagController(QObject):
    """Manages project-level tag registry and per-image tag assignments."""

    # Emitted whenever the project's tag registry changes (add/remove/rename).
    # Listeners refresh their TagFilterBar / TagChipBar options.
    tags_changed = pyqtSignal()
    # Emitted when a single image's tag list changes (path, new_tags).
    image_tags_changed = pyqtSignal(object, list)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._project: ProjectManager | None = None
        self._backup_mgr: BackupManager | None = None
        self._tag_cache: dict[str, set[str]] | None = None
        # Any per-image or registry mutation invalidates the cache.
        self.image_tags_changed.connect(self._invalidate_tag_cache)
        self.tags_changed.connect(self._invalidate_tag_cache)

    def set_project(
        self,
        project: ProjectManager | None,
        backup_mgr: BackupManager | None = None,
    ) -> None:
        self._project = project
        self._backup_mgr = backup_mgr
        self._tag_cache = None
        self.tags_changed.emit()

    @property
    def project(self) -> ProjectManager | None:
        return self._project

    def project_tags(self) -> list[str]:
        if self._project is None:
            return []
        return list(self._project.config.tags)

    # ── Project-level CRUD ────────────────────────────────────────

    def add_tag(self, tag: str) -> bool:
        """Append ``tag`` to project registry. Returns True if added.

        Raises ``TagError`` on validation failure (caller shows the error).
        """
        if self._project is None:
            return False
        if TagService.add_project_tag(self._project.config, tag):
            self._project.save()
            self.tags_changed.emit()
            return True
        return False

    def remove_tag(self, tag: str, *, cascade: bool = True) -> bool:
        """Remove ``tag`` from project registry. If ``cascade``, also strip
        the tag from every image's label JSON. Returns True if removed."""
        if self._project is None:
            return False
        if tag not in self._project.config.tags:
            return False
        self._snapshot_before_destructive(f"remove_tag_{tag}")
        TagService.remove_project_tag(self._project.config, tag)
        self._project.save()
        if cascade:
            self._strip_tag_from_all_images(tag)
        self.tags_changed.emit()
        return True

    def rename_tag(self, old: str, new: str) -> str:
        """Rename a tag both in the registry and in every image's label JSON.

        Returns the normalized new name. Raises ``TagError`` on conflict.
        """
        if self._project is None:
            raise TagError("没有打开的项目")
        self._snapshot_before_destructive(f"rename_tag_{old}")
        normalized_new = TagService.rename_project_tag(
            self._project.config, old, new,
        )
        self._project.save()
        if normalized_new != old:
            self._rename_tag_in_all_images(old, normalized_new)
        self.tags_changed.emit()
        return normalized_new

    # ── Per-image CRUD ────────────────────────────────────────────

    def set_image_tags(self, image_path: Path, tags: list[str]) -> list[str]:
        """Persist ``tags`` as the new tag list for ``image_path``.

        Also auto-registers any new tags into the project registry. Returns
        the normalized, deduped list actually written.
        """
        if self._project is None:
            return []
        normalized: list[str] = []
        for t in tags:
            try:
                normalized.append(normalize(t))
            except TagError:
                continue
        normalized = dedupe_preserving_order(normalized)

        added = TagService.ensure_registered(self._project.config, normalized)
        if added:
            self._project.save()

        label_path = self._project.label_path_for(image_path)
        ia = load_annotation(label_path)
        if ia is None:
            w, h = get_image_size(image_path)
            ia = ImageAnnotation(image_path=Path(image_path).name, image_size=(w, h))
        ia.tags = normalized
        save_annotation(ia, label_path)

        self.image_tags_changed.emit(Path(image_path), list(normalized))
        if added:
            self.tags_changed.emit()
        return normalized

    def get_image_tags(self, image_path: Path) -> list[str]:
        if self._project is None:
            return []
        ia = load_annotation(self._project.label_path_for(image_path))
        return list(ia.tags) if ia else []

    def register_new_tags(self, tags: list[str]) -> list[str]:
        """Add any of ``tags`` that aren't already in the project registry.

        Use this when the per-image label JSON has *already* been written by
        another writer (e.g. a view) and we just need to keep the project
        registry / autocomplete in sync. Emits ``tags_changed`` when the
        registry actually grows. Returns the list of newly-added tags.
        """
        if self._project is None:
            return []
        added = TagService.ensure_registered(self._project.config, tags)
        if added:
            self._project.save()
            self.tags_changed.emit()
        return added

    def apply_tag_to_images(self, tag: str, paths: list[Path]) -> int:
        """Add ``tag`` to the user tags of every image at ``paths`` (union).

        Idempotent — images already carrying ``tag`` are skipped (no write,
        no signal). Auto-registers ``tag`` into the project tag registry if
        it isn't there yet. Per-image read/write failures are caught and
        logged; the final count reflects only successful writes.
        Returns the number of images actually modified.
        """
        if self._project is None or not paths:
            return 0

        try:
            tag_norm = normalize(tag)
        except TagError:
            return 0

        # Auto-register so future autocomplete / TagApplyBar refreshes see it.
        self.register_new_tags([tag_norm])

        modified = 0
        for p in paths:
            try:
                label_path = self._project.label_path_for(p)
                ia = load_annotation(label_path)
                if ia is None:
                    w, h = get_image_size(p)
                    ia = ImageAnnotation(image_path=Path(p).name, image_size=(w, h))
                if tag_norm in ia.tags:
                    continue
                ia.tags = list(ia.tags) + [tag_norm]
                save_annotation(ia, label_path)
                self.image_tags_changed.emit(Path(p), list(ia.tags))
                modified += 1
            except (OSError, ValueError, KeyError):
                logger.warning("apply_tag_to_images failed for %s", p, exc_info=True)
                continue
        return modified

    # ── Bulk loading helpers (for UI cache initialization) ────────

    def load_all_image_tags(self) -> dict[str, set[str]]:
        """Return {image_path_str: set(tags)} for every labeled image.

        Used by FileListWidget to populate its tag filter cache on project
        open. Images without labels map to an empty set.
        """
        result: dict[str, set[str]] = {}
        if self._project is None:
            return result
        for img in self._project.list_images():
            ia = load_annotation(self._project.label_path_for(img))
            result[str(img)] = set(ia.tags) if ia else set()
        return result

    # ── Internals ─────────────────────────────────────────────────

    def _snapshot_before_destructive(self, label: str) -> None:
        if self._backup_mgr is None:
            return
        try:
            self._backup_mgr.create_backup()
        except Exception:  # pragma: no cover — best-effort
            logger.warning("Backup before %s failed", label, exc_info=True)

    def _strip_tag_from_all_images(self, tag: str) -> None:
        if self._project is None:
            return
        for img in self._project.list_images():
            label_path = self._project.label_path_for(img)
            ia = load_annotation(label_path)
            if ia is None or tag not in ia.tags:
                continue
            ia.tags = [t for t in ia.tags if t != tag]
            save_annotation(ia, label_path)
            self.image_tags_changed.emit(Path(img), list(ia.tags))

    def _rename_tag_in_all_images(self, old: str, new: str) -> None:
        if self._project is None:
            return
        for img in self._project.list_images():
            label_path = self._project.label_path_for(img)
            ia = load_annotation(label_path)
            if ia is None or old not in ia.tags:
                continue
            ia.tags = dedupe_preserving_order(
                new if t == old else t for t in ia.tags
            )
            save_annotation(ia, label_path)
            self.image_tags_changed.emit(Path(img), list(ia.tags))

    # ── Filter breakdown ──────────────────────────────────────────

    def compute_filter_breakdown(self, filt: TagFilter) -> dict[str, int]:
        """Aggregate ``filt.classify`` over every image in the open project.

        Returns ``{"match": int, "excluded": int, "no_include": int,
        "conflict": int}``. Uses an internal cache populated from
        ``load_all_image_tags()`` on first access; the cache is
        invalidated automatically whenever the registry or any per-image
        tag list changes (see ``__init__`` signal wiring).
        """
        counts = {"match": 0, "excluded": 0, "no_include": 0, "conflict": 0}
        if self._project is None:
            return counts
        cache = self._get_tag_cache()
        for img in self._project.list_images():
            tags = cache.get(str(img), set())
            counts[filt.classify(tags)] += 1
        return counts

    def _get_tag_cache(self) -> dict[str, set[str]]:
        if self._tag_cache is None:
            self._tag_cache = self.load_all_image_tags()
        return self._tag_cache

    def _invalidate_tag_cache(self, *_args) -> None:
        self._tag_cache = None
