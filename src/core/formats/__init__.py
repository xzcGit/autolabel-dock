"""Format registry — plugin-style exporter/importer management."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.core.annotation import ImageAnnotation


class ExporterInfo:
    """Metadata for a registered exporter."""

    def __init__(
        self,
        name: str,
        label: str,
        export_fn: Callable,
        needs_classes: bool = True,
        output_is_file: bool = False,
        needs_source_dir: bool = False,
    ):
        self.name = name
        self.label = label
        self.export_fn = export_fn
        self.needs_classes = needs_classes
        self.output_is_file = output_is_file
        self.needs_source_dir = needs_source_dir


class ImporterInfo:
    """Metadata for a registered importer.

    Two honest importer kinds, selected by ``is_full_import``:
    - record importer (default): ``import_fn(source: Path,
      project_classes: list[str] | None) -> list[ImageAnnotation]``,
      dispatched via ``ImportRegistry.import_records`` and fed into the
      merge pipeline (``import_merge.merge_imported_records``);
    - full importer: ``import_fn(source: Path, project) -> dict`` manages
      images + labels itself (caller invokes ``info.import_fn`` directly).
    """

    def __init__(
        self,
        name: str,
        label: str,
        import_fn: Callable,
        input_is_file: bool = False,
        file_filter: str = "",
        is_full_import: bool = False,
    ):
        self.name = name
        self.label = label
        self.import_fn = import_fn
        self.input_is_file = input_is_file  # True=select file, False=select directory
        self.file_filter = file_filter  # e.g. "JSON files (*.json)" for file dialog
        # is_full_import: importer takes (source_dir, project) and writes images+labels
        # directly. Skips the controller's read-then-merge pipeline.
        self.is_full_import = is_full_import


class ExportRegistry:
    """Registry of available export formats."""

    def __init__(self):
        self._exporters: dict[str, ExporterInfo] = {}

    def register(
        self,
        name: str,
        label: str,
        export_fn: Callable,
        needs_classes: bool = True,
        output_is_file: bool = False,
        needs_source_dir: bool = False,
    ) -> None:
        self._exporters[name] = ExporterInfo(
            name=name, label=label, export_fn=export_fn,
            needs_classes=needs_classes, output_is_file=output_is_file,
            needs_source_dir=needs_source_dir,
        )

    def get(self, name: str) -> ExporterInfo | None:
        return self._exporters.get(name)

    def list_names(self) -> list[str]:
        return list(self._exporters.keys())

    def list_labels(self) -> list[str]:
        return [e.label for e in self._exporters.values()]

    def export(
        self,
        name: str,
        annotations: list[ImageAnnotation],
        output_dir: Path | str,
        classes: list[str] | None = None,
        only_confirmed: bool = False,
        source_image_dir: Path | str | None = None,
    ) -> None:
        info = self._exporters.get(name)
        if info is None:
            raise ValueError(f"Unknown export format: {name}")
        output = Path(output_dir)
        if info.output_is_file:
            output = output / f"{name.lower()}.json"
        kwargs = {"only_confirmed": only_confirmed}
        if info.needs_classes and classes is not None:
            kwargs["classes"] = classes
        if info.needs_source_dir:
            kwargs["source_image_dir"] = source_image_dir
        info.export_fn(annotations, output, **kwargs)


class ImportRegistry:
    """Registry of available import formats."""

    def __init__(self):
        self._importers: dict[str, ImporterInfo] = {}

    def register(
        self,
        name: str,
        label: str,
        import_fn: Callable,
        input_is_file: bool = False,
        file_filter: str = "",
        is_full_import: bool = False,
    ) -> None:
        self._importers[name] = ImporterInfo(
            name=name, label=label, import_fn=import_fn,
            input_is_file=input_is_file, file_filter=file_filter,
            is_full_import=is_full_import,
        )

    def get(self, name: str) -> ImporterInfo | None:
        return self._importers.get(name)

    def list_names(self) -> list[str]:
        return list(self._importers.keys())

    def list_info(self) -> list[ImporterInfo]:
        return list(self._importers.values())

    def import_records(
        self,
        name: str,
        source: Path | str,
        project_classes: list[str] | None = None,
    ) -> list[ImageAnnotation]:
        """Dispatch a record-importer (mirror of ``ExportRegistry.export``).

        The registered ``import_fn`` IS the function called here — no
        side-table of per-format call conventions. Raises ``ValueError`` on
        an unknown format or when the format is a full importer
        (``is_full_import=True``; call ``info.import_fn(source, project)``
        for those instead).
        """
        info = self._importers.get(name)
        if info is None:
            raise ValueError(f"未知的导入格式: {name}")
        if info.is_full_import:
            raise ValueError(
                f"格式 {name} 是完整导入器（is_full_import），"
                "应通过 info.import_fn(source, project) 调用"
            )
        # Single normalization point: adapters always receive a non-empty
        # class list or None (empty list means "no project classes").
        return info.import_fn(Path(source), project_classes or None)


# Global registry instances
_registry = ExportRegistry()
_import_registry = ImportRegistry()


def get_export_registry() -> ExportRegistry:
    """Get the global export format registry."""
    return _registry


def get_import_registry() -> ImportRegistry:
    """Get the global import format registry."""
    return _import_registry


def _register_builtin_formats() -> None:
    """Register all built-in export and import formats."""
    from src.core.formats.yolo import export_yolo_detection, import_yolo_for_project
    from src.core.formats.coco import export_coco, import_coco
    from src.core.formats.labelme import export_labelme, import_labelme_records
    from src.core.formats.imagefolder import (
        ImageFolderImporter,
        export_imagefolder,
        export_csv,
    )

    _registry.register("YOLO", "YOLO (txt)", export_yolo_detection, needs_classes=True)
    _registry.register("COCO", "COCO (json)", export_coco, needs_classes=True, output_is_file=True)
    _registry.register("labelme", "labelme (json)", export_labelme, needs_classes=False)
    _registry.register(
        "ImageFolder", "ImageFolder (分类)", export_imagefolder,
        needs_classes=False, needs_source_dir=True,
    )
    _registry.register(
        "CSV", "CSV (分类)", export_csv,
        needs_classes=False, needs_source_dir=True,
    )

    # Record importers: the registered import_fn is exactly what
    # ImportRegistry.import_records calls at runtime.
    _import_registry.register(
        "YOLO", "YOLO (txt)", import_yolo_for_project,
        input_is_file=False,
    )
    _import_registry.register(
        "COCO", "COCO (json)", import_coco,
        input_is_file=True, file_filter="JSON 文件 (*.json)",
    )
    _import_registry.register(
        "labelme", "labelme (json)", import_labelme_records,
        input_is_file=False,
    )
    _import_registry.register(
        "ImageFolder", "ImageFolder (分类)", ImageFolderImporter().import_to_project,
        input_is_file=False, is_full_import=True,
    )
    # CSV is export-only by design: no importer is registered on purpose
    # (a CSV importer would be a new feature, not a missing registration).


_register_builtin_formats()
