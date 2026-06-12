"""Tests for the TagController (project-level CRUD + per-image writes)."""
from __future__ import annotations

import pytest

from src.controllers.tags import TagController
from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter


def _make_proj(tmp_path) -> ProjectManager:
    pm = ProjectManager.create(tmp_path / "p", "p", classes=["a"])
    img = pm.project_dir / pm.config.image_dir / "x.jpg"
    img.write_bytes(b"fake")
    return pm


def test_add_remove_rename_emits_and_persists(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)

    fired = []
    ctrl.tags_changed.connect(lambda: fired.append("x"))

    assert ctrl.add_tag("foo")
    assert pm.config.tags == ["foo"]
    assert len(fired) >= 1

    assert ctrl.rename_tag("foo", "bar") == "bar"
    # Reload from disk to confirm persistence.
    pm2 = ProjectManager.open(pm.project_dir)
    assert pm2.config.tags == ["bar"]

    assert ctrl.remove_tag("bar")
    assert pm.config.tags == []


def test_remove_cascades_to_images(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    ctrl.add_tag("alpha")

    img = pm.list_images()[0]
    save_annotation(
        ImageAnnotation(
            image_path=img.name, image_size=(10, 10), tags=["alpha", "beta"],
        ),
        pm.label_path_for(img),
    )

    ctrl.remove_tag("alpha", cascade=True)

    reloaded = load_annotation(pm.label_path_for(img))
    assert reloaded.tags == ["beta"]


def test_rename_propagates_to_images(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    ctrl.add_tag("old")

    img = pm.list_images()[0]
    save_annotation(
        ImageAnnotation(
            image_path=img.name, image_size=(10, 10), tags=["old", "keep"],
        ),
        pm.label_path_for(img),
    )

    ctrl.rename_tag("old", "new")

    reloaded = load_annotation(pm.label_path_for(img))
    assert reloaded.tags == ["new", "keep"]


def test_set_image_tags_auto_registers(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)

    img = pm.list_images()[0]
    written = ctrl.set_image_tags(img, ["fresh", "fresh", "  fresh "])
    # Dedupe + normalize
    assert written == ["fresh"]
    # Project registry now contains "fresh"
    assert "fresh" in pm.config.tags

    reloaded = load_annotation(pm.label_path_for(img))
    assert reloaded.tags == ["fresh"]


def test_register_new_tags_only_grows_registry(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    ctrl.add_tag("already_here")

    added = ctrl.register_new_tags(["already_here", "new1", "new2"])
    assert added == ["new1", "new2"]
    assert pm.config.tags == ["already_here", "new1", "new2"]


def test_load_all_image_tags(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)

    img = pm.list_images()[0]
    save_annotation(
        ImageAnnotation(image_path=img.name, image_size=(10, 10), tags=["t1"]),
        pm.label_path_for(img),
    )
    snapshot = ctrl.load_all_image_tags()
    assert snapshot[str(img)] == {"t1"}


def _make_proj_with_images(tmp_path, names):
    pm = ProjectManager.create(tmp_path / "p", "p", classes=["a"])
    img_dir = pm.project_dir / pm.config.image_dir
    for name in names:
        (img_dir / name).write_bytes(b"fake")
    return pm


def test_compute_filter_breakdown_classifies_every_image(qapp, tmp_path):
    """4 images: a-only / b-only / a+b / untagged → expect one of each
    classify() outcome for filter include=a, exclude=b."""
    pm = _make_proj_with_images(
        tmp_path, ["a.jpg", "b.jpg", "ab.jpg", "none.jpg"]
    )
    img_dir = pm.project_dir / pm.config.image_dir

    save_annotation(
        ImageAnnotation(image_path="a.jpg", image_size=(10, 10), tags=["a"]),
        pm.label_path_for(img_dir / "a.jpg"),
    )
    save_annotation(
        ImageAnnotation(image_path="b.jpg", image_size=(10, 10), tags=["b"]),
        pm.label_path_for(img_dir / "b.jpg"),
    )
    save_annotation(
        ImageAnnotation(
            image_path="ab.jpg", image_size=(10, 10), tags=["a", "b"],
        ),
        pm.label_path_for(img_dir / "ab.jpg"),
    )
    save_annotation(
        ImageAnnotation(image_path="none.jpg", image_size=(10, 10), tags=[]),
        pm.label_path_for(img_dir / "none.jpg"),
    )

    ctrl = TagController()
    ctrl.set_project(pm)
    counts = ctrl.compute_filter_breakdown(
        TagFilter(includes=("a",), excludes=("b",))
    )
    assert counts == {
        "match": 1,        # a.jpg
        "excluded": 1,     # b.jpg
        "conflict": 1,     # ab.jpg
        "no_include": 1,   # none.jpg
    }


def test_compute_filter_breakdown_empty_filter_all_match(qapp, tmp_path):
    pm = _make_proj_with_images(tmp_path, ["x.jpg", "y.jpg"])
    img_dir = pm.project_dir / pm.config.image_dir
    save_annotation(
        ImageAnnotation(image_path="x.jpg", image_size=(10, 10), tags=[]),
        pm.label_path_for(img_dir / "x.jpg"),
    )

    ctrl = TagController()
    ctrl.set_project(pm)
    counts = ctrl.compute_filter_breakdown(TagFilter())
    assert counts == {"match": 2, "excluded": 0, "conflict": 0, "no_include": 0}


def test_compute_filter_breakdown_invalidates_on_image_tags_changed(
    qapp, tmp_path,
):
    pm = _make_proj_with_images(tmp_path, ["x.jpg"])
    img_dir = pm.project_dir / pm.config.image_dir

    ctrl = TagController()
    ctrl.set_project(pm)
    filt = TagFilter(includes=("a",))
    assert ctrl.compute_filter_breakdown(filt) == {
        "match": 0, "excluded": 0, "no_include": 1, "conflict": 0,
    }

    ctrl.set_image_tags(img_dir / "x.jpg", ["a"])

    assert ctrl.compute_filter_breakdown(filt) == {
        "match": 1, "excluded": 0, "no_include": 0, "conflict": 0,
    }


def test_compute_filter_breakdown_no_project(qapp):
    ctrl = TagController()
    counts = ctrl.compute_filter_breakdown(TagFilter(includes=("a",)))
    assert counts == {"match": 0, "excluded": 0, "no_include": 0, "conflict": 0}


def test_apply_tag_to_images_adds_to_all_paths(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    # Add two more images so we have three total
    for name in ("y.jpg", "z.jpg"):
        (pm.project_dir / pm.config.image_dir / name).write_bytes(b"fake")
    ctrl = TagController()
    ctrl.set_project(pm)

    paths = sorted(pm.list_images())
    emitted: list = []
    ctrl.image_tags_changed.connect(lambda p, t: emitted.append((p, list(t))))

    n = ctrl.apply_tag_to_images("blur", paths)
    assert n == 3
    assert len(emitted) == 3
    for p in paths:
        ia = load_annotation(pm.label_path_for(p))
        assert ia is not None and "blur" in ia.tags
    # Tag was auto-registered
    assert "blur" in pm.config.tags


def test_apply_tag_to_images_idempotent(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    paths = list(pm.list_images())

    ctrl.apply_tag_to_images("blur", paths)

    emitted: list = []
    ctrl.image_tags_changed.connect(lambda p, t: emitted.append(p))
    n2 = ctrl.apply_tag_to_images("blur", paths)
    assert n2 == 0
    assert emitted == []


def test_apply_tag_to_images_mixed_returns_partial(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    (pm.project_dir / pm.config.image_dir / "y.jpg").write_bytes(b"fake")
    ctrl = TagController()
    ctrl.set_project(pm)
    paths = sorted(pm.list_images())

    # Pre-tag the first image only
    ctrl.set_image_tags(paths[0], ["blur"])

    emitted: list = []
    ctrl.image_tags_changed.connect(lambda p, t: emitted.append(p))
    n = ctrl.apply_tag_to_images("blur", paths)
    assert n == 1
    assert emitted == [paths[1]]


def test_apply_tag_to_images_empty_paths(qapp, tmp_path):
    pm = _make_proj(tmp_path)
    ctrl = TagController()
    ctrl.set_project(pm)
    emitted: list = []
    ctrl.image_tags_changed.connect(lambda p, t: emitted.append(p))
    assert ctrl.apply_tag_to_images("blur", []) == 0
    assert emitted == []


def test_apply_tag_to_images_no_project(qapp):
    ctrl = TagController()
    # No set_project called
    assert ctrl.apply_tag_to_images("blur", []) == 0
