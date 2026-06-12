"""Integration: train data preparation respects per-image user tags."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_io import save_annotation
from src.core.project import ProjectManager
from src.core.tags import TagFilter
from src.engine.dataset import DatasetPreparer


def _make_proj(tmp_path: Path) -> ProjectManager:
    pm = ProjectManager.create(tmp_path / "p", "p", classes=["cat"])
    img_dir = pm.project_dir / pm.config.image_dir
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (img_dir / name).write_bytes(b"fake")
    return pm


def _save_label(pm: ProjectManager, name: str, *, tags: list[str]) -> None:
    img = pm.project_dir / pm.config.image_dir / name
    save_annotation(
        ImageAnnotation(
            image_path=name,
            image_size=(100, 100),
            annotations=[
                Annotation(class_name="cat", class_id=0,
                           bbox=(0.5, 0.5, 0.3, 0.3), confirmed=True),
            ],
            tags=list(tags),
        ),
        pm.label_path_for(img),
    )


def test_no_tag_filter_includes_all(tmp_path):
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=["x"])
    _save_label(pm, "b.jpg", tags=[])
    _save_label(pm, "c.jpg", tags=["y"])

    out = tmp_path / "ds"
    DatasetPreparer(pm).prepare(out, task="detect", val_ratio=0.0)
    train_imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert train_imgs == ["a.jpg", "b.jpg", "c.jpg"]


def test_tag_filter_or_mode(tmp_path):
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=["x"])
    _save_label(pm, "b.jpg", tags=["y"])
    _save_label(pm, "c.jpg", tags=["z"])

    out = tmp_path / "ds"
    DatasetPreparer(pm).prepare(
        out, task="detect", val_ratio=0.0,
        tag_filter=TagFilter(includes=("x", "z"), mode="or"),
    )
    train_imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert train_imgs == ["a.jpg", "c.jpg"]


def test_tag_filter_and_mode(tmp_path):
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=["x", "y"])
    _save_label(pm, "b.jpg", tags=["x"])
    _save_label(pm, "c.jpg", tags=["x", "y"])

    out = tmp_path / "ds"
    DatasetPreparer(pm).prepare(
        out, task="detect", val_ratio=0.0,
        tag_filter=TagFilter(includes=("x", "y"), mode="and"),
    )
    train_imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert train_imgs == ["a.jpg", "c.jpg"]


def test_empty_tag_filter_is_passthrough(tmp_path):
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=[])
    _save_label(pm, "b.jpg", tags=[])

    out = tmp_path / "ds"
    DatasetPreparer(pm).prepare(
        out, task="detect", val_ratio=0.0, tag_filter=TagFilter(),
    )
    train_imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert train_imgs == ["a.jpg", "b.jpg"]


def test_tag_filter_excluding_all_raises(tmp_path):
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=["x"])
    _save_label(pm, "b.jpg", tags=["x"])

    out = tmp_path / "ds"
    with pytest.raises(ValueError):
        DatasetPreparer(pm).prepare(
            out, task="detect", val_ratio=0.0,
            tag_filter=TagFilter(includes=("nonexistent",)),
        )


def test_prepare_drops_images_matching_excludes(tmp_path):
    """Excludes-only filter removes tagged images from the prepared dataset."""
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=["good"])
    _save_label(pm, "b.jpg", tags=["bad"])
    _save_label(pm, "c.jpg", tags=[])

    out = tmp_path / "ds"
    DatasetPreparer(pm).prepare(
        out, task="detect", val_ratio=0.0,
        tag_filter=TagFilter(excludes=("bad",)),
    )

    linked = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert "a.jpg" in linked
    assert "c.jpg" in linked
    assert "b.jpg" not in linked


def test_prepare_drops_conflict_images(tmp_path):
    """Image with both an include tag and an exclude tag is dropped (exclude wins)."""
    pm = _make_proj(tmp_path)
    _save_label(pm, "a.jpg", tags=["good"])
    _save_label(pm, "b.jpg", tags=["good", "blurry"])  # conflict
    _save_label(pm, "c.jpg", tags=["blurry"])

    out = tmp_path / "ds"
    DatasetPreparer(pm).prepare(
        out, task="detect", val_ratio=0.0,
        tag_filter=TagFilter(includes=("good",), excludes=("blurry",)),
    )

    linked = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert linked == ["a.jpg"]
