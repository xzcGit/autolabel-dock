"""Tests for export registry."""
from pathlib import Path

import pytest


class TestExportRegistry:
    def test_builtin_formats_registered(self):
        from src.core.formats import get_export_registry

        registry = get_export_registry()
        names = registry.list_names()
        assert "YOLO" in names
        assert "COCO" in names
        assert "labelme" in names

    def test_get_returns_exporter_info(self):
        from src.core.formats import get_export_registry

        registry = get_export_registry()
        info = registry.get("YOLO")
        assert info is not None
        assert info.name == "YOLO"
        assert info.needs_classes is True

    def test_get_unknown_returns_none(self):
        from src.core.formats import get_export_registry

        registry = get_export_registry()
        assert registry.get("unknown_format") is None

    def test_coco_is_file_output(self):
        from src.core.formats import get_export_registry

        registry = get_export_registry()
        info = registry.get("COCO")
        assert info.output_is_file is True

    def test_labelme_no_classes(self):
        from src.core.formats import get_export_registry

        registry = get_export_registry()
        info = registry.get("labelme")
        assert info.needs_classes is False

    def test_register_custom_format(self):
        from src.core.formats import ExportRegistry

        registry = ExportRegistry()
        called = []
        registry.register("custom", "Custom Format", lambda anns, out, **kw: called.append(True))
        assert "custom" in registry.list_names()
        registry.export("custom", [], "/tmp/out")
        assert len(called) == 1


class TestImportRegistryDispatch:
    """ImportRegistry.import_records — real dispatch, mirror of ExportRegistry.export."""

    def test_unknown_format_raises(self, tmp_path):
        from src.core.formats import get_import_registry

        with pytest.raises(ValueError, match="未知的导入格式"):
            get_import_registry().import_records("nope", tmp_path)

    def test_full_import_format_raises(self, tmp_path):
        from src.core.formats import get_import_registry

        with pytest.raises(ValueError, match="完整导入器"):
            get_import_registry().import_records("ImageFolder", tmp_path)

    def test_registered_fn_is_the_called_fn(self, monkeypatch, tmp_path):
        """Pin the dead-metadata bug class: import_records must call the
        function stored in the registry entry — not a controller-side twin."""
        from src.core.formats import get_import_registry

        registry = get_import_registry()
        info = registry.get("YOLO")
        calls = []

        def fake_import(source, project_classes=None):
            calls.append((source, project_classes))
            return []

        monkeypatch.setattr(info, "import_fn", fake_import)

        out = registry.import_records("YOLO", str(tmp_path), ["person"])

        assert out == []
        assert calls == [(Path(tmp_path), ["person"])]

    def test_registry_entries_point_at_real_adapters(self):
        from src.core.formats import get_import_registry
        from src.core.formats.yolo import import_yolo_for_project
        from src.core.formats.coco import import_coco
        from src.core.formats.labelme import import_labelme_records

        registry = get_import_registry()
        assert registry.get("YOLO").import_fn is import_yolo_for_project
        assert registry.get("COCO").import_fn is import_coco
        assert registry.get("labelme").import_fn is import_labelme_records

    def test_empty_class_list_normalized_to_none(self, monkeypatch, tmp_path):
        """Single normalization point: adapters receive a non-empty list or None."""
        from src.core.formats import get_import_registry

        registry = get_import_registry()
        info = registry.get("YOLO")
        received = []
        monkeypatch.setattr(
            info, "import_fn",
            lambda src, classes=None: received.append(classes) or [],
        )

        registry.import_records("YOLO", tmp_path, [])

        assert received == [None]

    def test_csv_is_export_only(self):
        from src.core.formats import get_export_registry, get_import_registry

        assert get_export_registry().get("CSV") is not None
        assert get_import_registry().get("CSV") is None
