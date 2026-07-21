"""Tests for the Qt-free import merge pipeline (core/formats/import_merge.py).

No qapp fixture: the pipeline must be testable without QMessageBox/dialogs.
Record IO is asserted to go through the injected LabelStore; the image-size
fallback reader is an injected callable.
"""
from pathlib import Path
from unittest.mock import patch

from src.core.annotation import Annotation, ImageAnnotation
from src.core.formats.import_merge import ImportMergeResult, merge_imported_records
from src.core.label_io import load_annotation, save_annotation
from src.core.label_store import LabelStore
from src.core.project import ProjectManager


def _create_project(tmp_path, classes=None, images=None):
    """Create a minimal project with fake image files."""
    if classes is None:
        classes = ["person", "car"]
    pm = ProjectManager.create(tmp_path / "proj", "test", classes=classes)
    img_dir = pm.project_dir / pm.config.image_dir
    for name in (images or ["img_001.jpg", "img_002.jpg"]):
        (img_dir / name).write_bytes(b"\xff\xd8")  # minimal JPEG header
    return pm


def _fail_reader(path):
    raise AssertionError(f"read_image_size must not be called (got {path})")


class RecordingStore(LabelStore):
    """LabelStore spy — records every load/save path."""

    def __init__(self, flush_cb=None):
        super().__init__(flush_cb)
        self.loads: list[Path] = []
        self.saves: list[Path] = []

    def load(self, label_path):
        self.loads.append(Path(label_path))
        return super().load(label_path)

    def save(self, ia, label_path):
        self.saves.append(Path(label_path))
        super().save(ia, label_path)


class TestConflictModes:
    def _existing_person(self, pm):
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        save_annotation(
            ImageAnnotation(
                image_path=img.name,
                image_size=(640, 480),
                annotations=[Annotation(class_name="person", class_id=0,
                                        bbox=(0.5, 0.5, 0.2, 0.2))],
            ),
            label_path,
        )
        return img, label_path

    def _imported_car(self, img):
        return [ImageAnnotation(
            image_path=img.name,
            image_size=(640, 480),
            annotations=[Annotation(class_name="car", class_id=1,
                                    bbox=(0.2, 0.2, 0.1, 0.1))],
        )]

    def test_skip_keeps_existing_annotated_record(self, tmp_path):
        pm = _create_project(tmp_path)
        img, label_path = self._existing_person(pm)

        result = merge_imported_records(
            pm, LabelStore(), self._imported_car(img), "skip",
            read_image_size=_fail_reader,
        )

        assert isinstance(result, ImportMergeResult)
        assert result.imported == 0
        assert result.skipped == 1
        final = load_annotation(label_path)
        assert [a.class_name for a in final.annotations] == ["person"]

    def test_overwrite_replaces_existing(self, tmp_path):
        pm = _create_project(tmp_path)
        img, label_path = self._existing_person(pm)

        result = merge_imported_records(
            pm, LabelStore(), self._imported_car(img), "overwrite",
            read_image_size=_fail_reader,
        )

        assert result.imported == 1
        assert result.skipped == 0
        final = load_annotation(label_path)
        assert [a.class_name for a in final.annotations] == ["car"]

    def test_merge_appends_annotations(self, tmp_path):
        pm = _create_project(tmp_path)
        img, label_path = self._existing_person(pm)

        result = merge_imported_records(
            pm, LabelStore(), self._imported_car(img), "merge",
            read_image_size=_fail_reader,
        )

        assert result.imported == 1
        final = load_annotation(label_path)
        assert {a.class_name for a in final.annotations} == {"person", "car"}

    def test_merge_preserves_existing_image_tags(self, tmp_path):
        pm = _create_project(tmp_path, classes=["cat", "dog"])
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        save_annotation(
            ImageAnnotation(
                image_path=img.name, image_size=(100, 100),
                image_tags=["cat"], image_tags_confirmed=True,
                image_tags_source="manual",
            ),
            label_path,
        )
        imported = [ImageAnnotation(
            image_path=img.name, image_size=(100, 100), image_tags=[],
        )]

        merge_imported_records(
            pm, LabelStore(), imported, "merge", read_image_size=_fail_reader,
        )

        final = load_annotation(label_path)
        assert final.image_tags == ["cat"]
        assert final.image_tags_confirmed is True
        assert final.image_tags_source == "manual"

    def test_merge_adopts_image_tags_when_existing_empty(self, tmp_path):
        pm = _create_project(tmp_path, classes=["cat", "dog"])
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        save_annotation(
            ImageAnnotation(image_path=img.name, image_size=(100, 100),
                            image_tags=[]),
            label_path,
        )
        imported = [ImageAnnotation(
            image_path=img.name, image_size=(100, 100),
            image_tags=["dog"], image_tags_confirmed=True,
            image_tags_source="manual",
        )]

        merge_imported_records(
            pm, LabelStore(), imported, "merge", read_image_size=_fail_reader,
        )

        final = load_annotation(label_path)
        assert final.image_tags == ["dog"]
        assert final.image_tags_confirmed is True
        assert final.image_tags_source == "manual"


class TestNewClassCollection:
    def test_collects_from_annotations_and_image_tags_ordered_deduped(self, tmp_path):
        pm = _create_project(tmp_path, classes=["person"])
        img1, img2 = pm.list_images()
        records = [
            ImageAnnotation(
                image_path=img1.name, image_size=(100, 100),
                annotations=[
                    Annotation(class_name="bicycle", class_id=0, bbox=(0.5, 0.5, 0.1, 0.1)),
                    Annotation(class_name="bicycle", class_id=0, bbox=(0.2, 0.2, 0.1, 0.1)),
                    Annotation(class_name="person", class_id=0, bbox=(0.3, 0.3, 0.1, 0.1)),
                ],
            ),
            ImageAnnotation(
                image_path=img2.name, image_size=(100, 100),
                image_tags=["cat", "person"],
            ),
        ]

        result = merge_imported_records(
            pm, LabelStore(), records, "overwrite", read_image_size=_fail_reader,
        )

        assert result.new_classes == ["bicycle", "cat"]
        # project.json persisted with the extended class list
        reopened = ProjectManager.open(pm.project_dir)
        assert reopened.config.classes == ["person", "bicycle", "cat"]

    def test_project_save_not_called_without_new_classes(self, tmp_path):
        pm = _create_project(tmp_path, classes=["person"])
        img = pm.list_images()[0]
        records = [ImageAnnotation(
            image_path=img.name, image_size=(100, 100),
            annotations=[Annotation(class_name="person", class_id=0,
                                    bbox=(0.5, 0.5, 0.1, 0.1))],
        )]

        with patch.object(pm, "save", wraps=pm.save) as save_spy:
            merge_imported_records(
                pm, LabelStore(), records, "overwrite", read_image_size=_fail_reader,
            )

        assert not save_spy.called


class TestClassIdRemap:
    def test_class_id_remapped_to_project_order(self, tmp_path):
        pm = _create_project(tmp_path, classes=["person", "car"])
        img = pm.list_images()[0]
        # Foreign dataset had "car" at id 0
        records = [ImageAnnotation(
            image_path=img.name, image_size=(100, 100),
            annotations=[Annotation(class_name="car", class_id=0,
                                    bbox=(0.5, 0.5, 0.1, 0.1))],
        )]

        merge_imported_records(
            pm, LabelStore(), records, "overwrite", read_image_size=_fail_reader,
        )

        final = load_annotation(pm.label_path_for(img))
        assert final.annotations[0].class_id == 1

    def test_new_class_gets_appended_id(self, tmp_path):
        pm = _create_project(tmp_path, classes=["person", "car"])
        img = pm.list_images()[0]
        records = [ImageAnnotation(
            image_path=img.name, image_size=(100, 100),
            annotations=[Annotation(class_name="bicycle", class_id=0,
                                    bbox=(0.5, 0.5, 0.1, 0.1))],
        )]

        merge_imported_records(
            pm, LabelStore(), records, "overwrite", read_image_size=_fail_reader,
        )

        final = load_annotation(pm.label_path_for(img))
        assert final.annotations[0].class_id == 2


class TestImageSizeResolution:
    def _record(self, img, size):
        return [ImageAnnotation(
            image_path=img.name, image_size=size,
            annotations=[Annotation(class_name="person", class_id=0,
                                    bbox=(0.5, 0.5, 0.1, 0.1))],
        )]

    def test_existing_size_wins_over_imported(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        save_annotation(
            ImageAnnotation(
                image_path=img.name, image_size=(640, 480),
                annotations=[Annotation(class_name="person", class_id=0,
                                        bbox=(0.1, 0.1, 0.1, 0.1))],
            ),
            label_path,
        )

        merge_imported_records(
            pm, LabelStore(), self._record(img, (100, 100)), "overwrite",
            read_image_size=_fail_reader,
        )

        assert load_annotation(label_path).image_size == (640, 480)

    def test_imported_size_used_when_no_existing(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]

        merge_imported_records(
            pm, LabelStore(), self._record(img, (100, 100)), "skip",
            read_image_size=_fail_reader,
        )

        assert load_annotation(pm.label_path_for(img)).image_size == (100, 100)

    def test_injected_reader_used_as_fallback(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]
        seen = []

        def reader(path):
            seen.append(Path(path))
            return (123, 456)

        merge_imported_records(
            pm, LabelStore(), self._record(img, (0, 0)), "skip",
            read_image_size=reader,
        )

        assert seen == [img]
        assert load_annotation(pm.label_path_for(img)).image_size == (123, 456)

    def test_reader_exception_falls_back_to_zero(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]

        def broken_reader(path):
            raise OSError("boom")

        merge_imported_records(
            pm, LabelStore(), self._record(img, (0, 0)), "skip",
            read_image_size=broken_reader,
        )

        assert load_annotation(pm.label_path_for(img)).image_size == (0, 0)


class TestStemMatching:
    def test_unmatched_stem_counts_skipped(self, tmp_path):
        pm = _create_project(tmp_path)
        records = [ImageAnnotation(
            image_path="img_999.jpg", image_size=(100, 100),
            annotations=[Annotation(class_name="person", class_id=0,
                                    bbox=(0.5, 0.5, 0.1, 0.1))],
        )]

        result = merge_imported_records(
            pm, LabelStore(), records, "overwrite", read_image_size=_fail_reader,
        )

        assert result.imported == 0
        assert result.skipped == 1
        assert not any((pm.project_dir / pm.config.label_dir).glob("*.json"))


class TestStoreInjection:
    def test_record_io_goes_through_injected_store(self, tmp_path):
        pm = _create_project(tmp_path)
        img = pm.list_images()[0]
        label_path = pm.label_path_for(img)
        flushes = []
        store = RecordingStore(flush_cb=lambda: flushes.append(True))
        records = [ImageAnnotation(
            image_path=img.name, image_size=(100, 100),
            annotations=[Annotation(class_name="person", class_id=0,
                                    bbox=(0.5, 0.5, 0.1, 0.1))],
        )]

        merge_imported_records(
            pm, store, records, "overwrite", read_image_size=_fail_reader,
        )

        # Reads flush first (LabelStore semantics) and writes land via the store.
        assert store.loads == [label_path]
        assert store.saves == [label_path]
        assert flushes  # load triggered the injected flush callback
