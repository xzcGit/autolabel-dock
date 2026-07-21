"""Tests for LabelStore — the flush-first read/write face for label records."""
import json

import pytest

from src.core import label_io
from src.core.annotation import Annotation, ImageAnnotation
from src.core.label_store import LabelStore


def _record(name="img.png", tags=None, with_ann=False):
    ia = ImageAnnotation(image_path=name, image_size=(100, 80), tags=list(tags or []))
    if with_ann:
        ia.annotations.append(
            Annotation(class_name="cat", class_id=0, bbox=(0.5, 0.5, 0.2, 0.2))
        )
    return ia


class TestFlushBeforeRead:
    def test_load_invokes_flush_before_reading(self, tmp_path):
        """The read must observe data that only the flush callback produced."""
        label_path = tmp_path / "img.json"
        events = []

        def flush():
            events.append("flush")
            label_io.save_annotation(_record(tags=["flushed"]), label_path)

        store = LabelStore(flush_cb=flush)
        loaded = store.load(label_path)

        assert events == ["flush"]
        assert loaded is not None
        assert loaded.tags == ["flushed"]

    def test_flush_invoked_on_every_read(self, tmp_path):
        events = []
        store = LabelStore(flush_cb=lambda: events.append("flush"))

        store.load(tmp_path / "a.json")
        store.load(tmp_path / "b.json")
        store.load_or_empty(tmp_path / "c.json", "c.png")

        assert events == ["flush", "flush", "flush"]

    def test_save_does_not_flush(self, tmp_path):
        """Writes are thin delegation — a save frequently IS the flush."""
        events = []
        store = LabelStore(flush_cb=lambda: events.append("flush"))

        store.save(_record(with_ann=True), tmp_path / "img.json")

        assert events == []

    def test_works_without_flush_callback(self, tmp_path):
        store = LabelStore()
        assert store.load(tmp_path / "missing.json") is None
        store.save(_record(with_ann=True), tmp_path / "img.json")
        loaded = store.load(tmp_path / "img.json")
        assert loaded is not None and len(loaded.annotations) == 1

    def test_set_flush_callback_replaces_and_clears(self, tmp_path):
        events = []
        store = LabelStore(flush_cb=lambda: events.append("first"))
        store.load(tmp_path / "a.json")
        store.set_flush_callback(lambda: events.append("second"))
        store.load(tmp_path / "a.json")
        store.set_flush_callback(None)
        store.load(tmp_path / "a.json")
        assert events == ["first", "second"]


class TestReentrancyGuard:
    def test_store_calls_inside_flush_do_not_reflush(self, tmp_path):
        calls = []
        store = LabelStore()

        def flush():
            calls.append("flush")
            # A flush implementation may itself read through the store
            # (e.g. status refresh) — this must not recurse.
            store.load(tmp_path / "inner.json")

        store.set_flush_callback(flush)
        store.load(tmp_path / "outer.json")

        assert calls == ["flush"]

    def test_flush_exception_propagates_and_guard_resets(self, tmp_path):
        calls = []

        def boom():
            calls.append("flush")
            raise RuntimeError("disk full")

        store = LabelStore(flush_cb=boom)
        with pytest.raises(RuntimeError):
            store.load(tmp_path / "a.json")
        # A second read must attempt the flush again (guard was reset in
        # finally) — if _flushing stuck True, this would silently skip.
        with pytest.raises(RuntimeError):
            store.load(tmp_path / "a.json")
        assert calls == ["flush", "flush"]


class TestLoadSemantics:
    def test_load_missing_returns_none(self, tmp_path):
        assert LabelStore().load(tmp_path / "missing.json") is None

    def test_load_corrupt_returns_none(self, tmp_path):
        p = tmp_path / "corrupt.json"
        p.write_text("{not valid json", encoding="utf-8")
        assert LabelStore().load(p) is None

    def test_load_existing_record(self, tmp_path):
        p = tmp_path / "img.json"
        label_io.save_annotation(_record(with_ann=True, tags=["t"]), p)
        loaded = LabelStore().load(p)
        assert loaded is not None
        assert len(loaded.annotations) == 1
        assert loaded.tags == ["t"]


class TestLoadOrEmpty:
    def test_missing_without_size_fabricates_placeholder(self, tmp_path):
        ia = LabelStore().load_or_empty(tmp_path / "missing.json", "img.png")
        assert ia.image_path == "img.png"
        assert tuple(ia.image_size) == (1, 1)
        assert ia.annotations == [] and ia.image_tags == [] and ia.tags == []

    def test_missing_with_size_uses_given_size(self, tmp_path):
        ia = LabelStore().load_or_empty(
            tmp_path / "missing.json", "img.png", image_size=(640, 480),
        )
        assert tuple(ia.image_size) == (640, 480)

    def test_corrupt_fabricates_empty(self, tmp_path):
        p = tmp_path / "corrupt.json"
        p.write_text("]]]", encoding="utf-8")
        ia = LabelStore().load_or_empty(p, "img.png", image_size=(10, 10))
        assert ia.annotations == []
        assert tuple(ia.image_size) == (10, 10)

    def test_existing_record_wins_over_fabrication(self, tmp_path):
        p = tmp_path / "img.json"
        label_io.save_annotation(_record(with_ann=True), p)
        ia = LabelStore().load_or_empty(p, "img.png", image_size=(999, 999))
        assert len(ia.annotations) == 1
        assert tuple(ia.image_size) == (100, 80)  # disk record, not the arg

    def test_flushes_first(self, tmp_path):
        """load_or_empty must see the record the flush just wrote."""
        p = tmp_path / "img.json"
        store = LabelStore()
        store.set_flush_callback(
            lambda: label_io.save_annotation(_record(tags=["pending"]), p)
        )
        ia = store.load_or_empty(p, "img.png")
        assert ia.tags == ["pending"]


class TestSaveSemantics:
    def test_save_roundtrip(self, tmp_path):
        p = tmp_path / "img.json"
        store = LabelStore()
        store.save(_record(with_ann=True, tags=["a"]), p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["tags"] == ["a"]
        assert len(data["annotations"]) == 1

    def test_save_empty_record_deletes_file(self, tmp_path):
        """label_io's delete-empty-record semantics must be preserved."""
        p = tmp_path / "img.json"
        label_io.save_annotation(_record(with_ann=True), p)
        assert p.exists()
        LabelStore().save(_record(), p)  # no annotations / image_tags / tags
        assert not p.exists()
