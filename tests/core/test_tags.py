"""Tests for the user-tag subsystem (TagFilter + TagService)."""
from __future__ import annotations

import pytest

from src.core.project import ProjectConfig
from src.core.tags import (
    TagError,
    TagFilter,
    TagService,
    dedupe_preserving_order,
    normalize,
)


# ── normalize / validation ────────────────────────────────────


class TestNormalize:
    def test_strips_whitespace(self):
        assert normalize("  foo  ") == "foo"

    def test_rejects_empty(self):
        with pytest.raises(TagError):
            normalize("")
        with pytest.raises(TagError):
            normalize("   ")

    def test_rejects_too_long(self):
        with pytest.raises(TagError):
            normalize("a" * 65)

    def test_rejects_invalid_chars(self):
        for bad in ["foo/bar", "a:b", "x*", "?", "<", ">", "|"]:
            with pytest.raises(TagError):
                normalize(bad)

    def test_accepts_unicode(self):
        assert normalize("夜间") == "夜间"


# ── TagFilter ─────────────────────────────────────────────────


class TestTagFilter:
    def test_empty_filter_matches_anything(self):
        f = TagFilter()
        assert f.is_empty()
        assert f.matches([])
        assert f.matches(["a", "b"])

    def test_or_mode(self):
        f = TagFilter(includes=("a", "b"), mode="or")
        assert f.matches(["a"])
        assert f.matches(["b", "c"])
        assert not f.matches(["c"])
        assert not f.matches([])

    def test_and_mode(self):
        f = TagFilter(includes=("a", "b"), mode="and")
        assert f.matches(["a", "b"])
        assert f.matches(["a", "b", "c"])
        assert not f.matches(["a"])
        assert not f.matches([])

    def test_is_hashable(self):
        # Must be usable as a dict key / signal payload.
        d = {TagFilter(includes=("a",)): 1}
        assert d[TagFilter(includes=("a",))] == 1


class TestTagFilterShape:
    def test_default_is_empty(self):
        f = TagFilter()
        assert f.includes == ()
        assert f.excludes == ()
        assert f.mode == "or"
        assert f.is_empty()

    def test_is_empty_requires_both_empty(self):
        assert not TagFilter(includes=("a",)).is_empty()
        assert not TagFilter(excludes=("b",)).is_empty()
        assert not TagFilter(includes=("a",), excludes=("b",)).is_empty()

    def test_hashable_with_both_sets(self):
        d = {TagFilter(includes=("a",), excludes=("b",)): 1}
        assert d[TagFilter(includes=("a",), excludes=("b",))] == 1


class TestTagFilterClassify:
    def test_empty_filter_classifies_match(self):
        assert TagFilter().classify([]) == "match"
        assert TagFilter().classify(["a"]) == "match"

    def test_includes_only_match_or_no_include(self):
        f = TagFilter(includes=("a",))
        assert f.classify(["a"]) == "match"
        assert f.classify(["a", "b"]) == "match"
        assert f.classify([]) == "no_include"
        assert f.classify(["b"]) == "no_include"

    def test_excludes_only_match_or_excluded(self):
        f = TagFilter(excludes=("bad",))
        assert f.classify([]) == "match"
        assert f.classify(["good"]) == "match"
        assert f.classify(["bad"]) == "excluded"
        assert f.classify(["bad", "good"]) == "excluded"

    def test_excludes_only_does_not_produce_conflict(self):
        """With no includes, an exclude hit is 'excluded', not 'conflict'."""
        f = TagFilter(excludes=("bad",))
        assert f.classify(["bad"]) == "excluded"

    def test_conflict_when_image_has_both_include_and_exclude_tags(self):
        f = TagFilter(includes=("good",), excludes=("blurry",))
        assert f.classify(["good", "blurry"]) == "conflict"
        assert f.classify(["good"]) == "match"
        assert f.classify(["blurry"]) == "excluded"
        assert f.classify(["other"]) == "no_include"

    def test_and_mode_requires_all_includes(self):
        f = TagFilter(includes=("a", "b"), mode="and")
        assert f.classify(["a", "b"]) == "match"
        assert f.classify(["a", "b", "c"]) == "match"
        assert f.classify(["a"]) == "no_include"
        assert f.classify(["b"]) == "no_include"

    def test_and_mode_partial_include_with_exclude_is_excluded(self):
        """AND mode + partial include match + exclude hit = 'excluded' (no
        include passed, so it's not a conflict)."""
        f = TagFilter(includes=("a", "b"), excludes=("bad",), mode="and")
        assert f.classify(["a", "bad"]) == "excluded"

    def test_matches_is_classify_equals_match(self):
        f = TagFilter(includes=("a",), excludes=("b",))
        assert f.matches(["a"]) is True
        assert f.matches(["a", "b"]) is False
        assert f.matches(["b"]) is False
        assert f.matches([]) is False
        assert TagFilter().matches([]) is True


# ── TagService ────────────────────────────────────────────────


def _empty_cfg() -> ProjectConfig:
    return ProjectConfig(name="p", image_dir="i", label_dir="l", classes=[])


class TestTagService:
    def test_add_normalizes_and_dedupes(self):
        cfg = _empty_cfg()
        assert TagService.add_project_tag(cfg, "  foo  ")
        assert cfg.tags == ["foo"]
        assert not TagService.add_project_tag(cfg, "foo")  # already present
        assert cfg.tags == ["foo"]

    def test_add_rejects_invalid(self):
        cfg = _empty_cfg()
        with pytest.raises(TagError):
            TagService.add_project_tag(cfg, "")

    def test_remove(self):
        cfg = _empty_cfg()
        cfg.tags = ["a", "b"]
        assert TagService.remove_project_tag(cfg, "a")
        assert cfg.tags == ["b"]
        assert not TagService.remove_project_tag(cfg, "missing")

    def test_rename_in_place_preserves_order(self):
        cfg = _empty_cfg()
        cfg.tags = ["a", "b", "c"]
        TagService.rename_project_tag(cfg, "b", "B")
        assert cfg.tags == ["a", "B", "c"]

    def test_rename_rejects_collision(self):
        cfg = _empty_cfg()
        cfg.tags = ["a", "b"]
        with pytest.raises(TagError):
            TagService.rename_project_tag(cfg, "a", "b")

    def test_rename_missing(self):
        cfg = _empty_cfg()
        with pytest.raises(TagError):
            TagService.rename_project_tag(cfg, "x", "y")

    def test_ensure_registered_appends_only_new(self):
        cfg = _empty_cfg()
        cfg.tags = ["x"]
        added = TagService.ensure_registered(cfg, ["x", "y", "z"])
        assert added == ["y", "z"]
        assert cfg.tags == ["x", "y", "z"]

    def test_ensure_registered_skips_invalid(self):
        cfg = _empty_cfg()
        added = TagService.ensure_registered(cfg, ["good", "", "  ", "ok"])
        assert added == ["good", "ok"]


def test_dedupe_preserves_order():
    assert dedupe_preserving_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
