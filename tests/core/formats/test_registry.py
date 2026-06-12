"""Tests for export registry."""
from pathlib import Path


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
