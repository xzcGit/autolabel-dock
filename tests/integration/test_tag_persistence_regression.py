"""Regression test for tag persistence bug (快捷键添加的tag无法被保存).

When a tag is added via the T shortcut (which writes directly to disk via
TagController), switching images should preserve that tag rather than
overwriting it with stale in-memory data.
"""
from pathlib import Path

import pytest
from PyQt5.QtGui import QImage

from src.controllers.tags import TagController
from src.core.label_io import load_annotation, save_annotation
from src.core.annotation import ImageAnnotation
from src.core.project import ProjectManager
from src.ui.label_panel import LabelPanel


@pytest.fixture
def project_with_images(tmp_path):
    """Create a detect project with two images."""
    proj_dir = tmp_path / "test_proj"
    images_dir = proj_dir / "images"
    images_dir.mkdir(parents=True)

    # Create two valid test images
    img1 = images_dir / "img1.jpg"
    img2 = images_dir / "img2.jpg"
    for img in [img1, img2]:
        qimg = QImage(100, 80, QImage.Format_RGB32)
        qimg.fill(0xFF000000)  # black
        qimg.save(str(img))

    pm = ProjectManager.create(proj_dir, "test", "images", ["cat", "dog"], task_type="detect")

    # Create initial label files (empty annotations)
    for img in [img1, img2]:
        ia = ImageAnnotation(image_path=img.name, image_size=(100, 80), tags=[])
        save_annotation(ia, pm.label_path_for(img))

    return pm, img1, img2


def test_tag_added_via_shortcut_survives_image_switch(qapp, project_with_images):
    """Tags added via T shortcut should survive image switching.

    Regression test for: 通过快捷键添加的tag无法被保存，切换图片再切回去，tag就没了

    The bug: TagController.apply_tag_to_images() writes directly to disk,
    but DetectPoseView._save_current() overwrites the file with stale
    in-memory data, losing the tag. The fix: refresh_image_tags() syncs
    the new tag into self._current_annotation.tags.
    """
    pm, img1, img2 = project_with_images

    tag_ctrl = TagController()
    tag_ctrl.set_project(pm)

    panel = LabelPanel(tag_controller=tag_ctrl)
    panel.set_project(pm)

    # 1. Load img1 (simulate user clicking on it)
    view = panel._view
    assert view is not None
    view._on_image_selected(img1)

    # 2. Simulate the T-shortcut tag apply pipeline: controller writes disk,
    #    then emits image_tags_changed, which LabelPanel routes to view.refresh_image_tags
    tag_ctrl.apply_tag_to_images("urgent", [img1])
    disk_ia = load_annotation(pm.label_path_for(img1))
    assert disk_ia is not None
    assert "urgent" in disk_ia.tags, "TagController should have written to disk"

    # The signal wiring in MainWindow triggers this call:
    view.refresh_image_tags(img1, disk_ia.tags)

    # 3. Switch away (triggers _save_current) — the bug would overwrite tags here
    view._on_image_selected(img2)

    # 4. Verify img1's tags survived on disk
    reloaded = load_annotation(pm.label_path_for(img1))
    assert reloaded is not None
    assert "urgent" in reloaded.tags, "Tag should survive image switch"


def test_multiple_tags_via_shortcut_accumulate(qapp, project_with_images):
    """Multiple T-shortcut tag additions should all survive."""
    pm, img1, img2 = project_with_images

    tag_ctrl = TagController()
    tag_ctrl.set_project(pm)

    panel = LabelPanel(tag_controller=tag_ctrl)
    panel.set_project(pm)

    view = panel._view
    view._on_image_selected(img1)

    # Add first tag
    tag_ctrl.apply_tag_to_images("tag1", [img1])
    ia = load_annotation(pm.label_path_for(img1))
    view.refresh_image_tags(img1, ia.tags)

    # Add second tag
    tag_ctrl.apply_tag_to_images("tag2", [img1])
    ia = load_annotation(pm.label_path_for(img1))
    view.refresh_image_tags(img1, ia.tags)

    # Switch away
    view._on_image_selected(img2)

    # Both tags should survive
    reloaded = load_annotation(pm.label_path_for(img1))
    assert set(reloaded.tags) == {"tag1", "tag2"}
