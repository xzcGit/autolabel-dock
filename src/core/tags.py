"""User-defined image tags — pluggable subsystem.

This module is intentionally Qt-free and stateless so it can be reused by:

  - UI widgets (FileList tag filter, AnnotationPanel chip bar, TrainPanel
    dataset filter)
  - Engine code (DatasetPreparer filters images by ``TagFilter``)
  - Controllers (TagController orchestrates project-level CRUD)

The data model lives on the existing dataclasses:

  - ``ProjectConfig.tags`` — list[str], the project's known-tag registry
    (autocomplete source; new tags are appended here when first used)
  - ``ImageAnnotation.tags`` — list[str], the per-image tag assignment

This module never touches ``ImageAnnotation.image_tags`` — that field stores
classification labels for the classify task and has different semantics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from src.core.project import ProjectConfig


MAX_TAG_LEN = 64
_INVALID_TAG_CHARS = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")


class TagError(ValueError):
    """Raised when a tag fails validation."""


def normalize(tag: str) -> str:
    """Strip + validate. Raises ``TagError`` on invalid input."""
    if tag is None:
        raise TagError("Tag 不能为空")
    t = tag.strip()
    if not t:
        raise TagError("Tag 不能为空")
    if len(t) > MAX_TAG_LEN:
        raise TagError(f"Tag 长度不能超过 {MAX_TAG_LEN} 字符")
    if _INVALID_TAG_CHARS.search(t):
        raise TagError("Tag 含有非法字符")
    return t


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@dataclass(frozen=True)
class TagFilter:
    """Reusable filter applied wherever tag-based image selection is needed.

    ``includes`` and ``excludes`` are tuples so the dataclass is hashable
    (e.g. usable as a dict key or signal payload). Runtime semantics:

    - Image is **excluded** if it has any tag in ``excludes``.
    - Otherwise, image **matches** if it satisfies the include condition
      (OR-of-includes or AND-of-includes, per ``mode``). Empty ``includes``
      means the include condition is trivially satisfied.

    See ``classify()`` for the 4-way breakdown used by diagnostic UI.
    The UI guarantees ``set(includes) & set(excludes) == set()`` via
    tri-state chip cycling — this class does not defend against violation.
    """

    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    mode: Literal["or", "and"] = "or"  # applies only to includes

    def is_empty(self) -> bool:
        return not (self.includes or self.excludes)

    def matches(self, image_tags: Iterable[str]) -> bool:
        return self.classify(image_tags) == "match"

    def classify(
        self, image_tags: Iterable[str]
    ) -> Literal["match", "excluded", "no_include", "conflict"]:
        present = set(image_tags or ())
        include_hit = (not self.includes) or (
            all(t in present for t in self.includes) if self.mode == "and"
            else any(t in present for t in self.includes)
        )
        exclude_hit = bool(self.excludes and present.intersection(self.excludes))
        if exclude_hit and include_hit and self.includes:
            return "conflict"
        if exclude_hit:
            return "excluded"
        if not include_hit:
            return "no_include"
        return "match"


class TagService:
    """Stateless helpers operating on ProjectConfig.tags.

    Per-image tag mutations are trivial list ops on ``ImageAnnotation.tags``
    and don't need a wrapper here — controllers handle the IO side.
    """

    @staticmethod
    def add_project_tag(cfg: ProjectConfig, tag: str) -> bool:
        """Append ``tag`` to the project registry if not present.

        Returns True if the registry was modified.
        """
        t = normalize(tag)
        if t in cfg.tags:
            return False
        cfg.tags.append(t)
        return True

    @staticmethod
    def remove_project_tag(cfg: ProjectConfig, tag: str) -> bool:
        """Remove ``tag`` from the project registry. Returns True if removed."""
        if tag in cfg.tags:
            cfg.tags.remove(tag)
            return True
        return False

    @staticmethod
    def rename_project_tag(cfg: ProjectConfig, old: str, new: str) -> str:
        """Rename a registry tag in-place. Returns the normalized new name.

        Raises ``TagError`` if ``old`` is missing or ``new`` collides with
        another existing tag.
        """
        n = normalize(new)
        if old not in cfg.tags:
            raise TagError(f"Tag \"{old}\" 不存在")
        if n != old and n in cfg.tags:
            raise TagError(f"Tag \"{n}\" 已存在")
        idx = cfg.tags.index(old)
        cfg.tags[idx] = n
        return n

    @staticmethod
    def ensure_registered(cfg: ProjectConfig, tags: Iterable[str]) -> list[str]:
        """Append any of ``tags`` not already in the registry.

        Returns the list of newly-added tags. Invalid tags are skipped
        silently — callers that care should ``normalize`` first.
        """
        added: list[str] = []
        for raw in tags:
            try:
                t = normalize(raw)
            except TagError:
                continue
            if t not in cfg.tags:
                cfg.tags.append(t)
                added.append(t)
        return added
