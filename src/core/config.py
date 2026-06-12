"""Global application configuration (~/.autolabel/config.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Global app settings persisted across sessions."""

    recent_projects: list[str] = field(default_factory=list)
    theme: str = "dark"
    auto_save: bool = True
    default_conf_threshold: float = 0.5
    default_iou_threshold: float = 0.45
    overlap_iou_threshold: float = 0.5
    script_tools: dict[str, str] = field(default_factory=dict)
    window_geometry: dict[str, int] = field(
        default_factory=lambda: {"x": 100, "y": 100, "width": 1400, "height": 900}
    )
    classify_grid_density: int = 96
    classify_grid_sort: str = "filename"  # "filename" | "class"
    classify_preview_width: int = 320
    classify_preview_visible: bool = True
    annotation_panel_splitter_sizes: list[int] = field(default_factory=list)
    annotation_panel_collapsed: dict[str, bool] = field(default_factory=dict)
    # Experimental: master switch to fully hide the optional LocateAnything
    # text-labeling backend. Defaults True (visible); set False to hide it
    # entirely. Old config.json files without the key load as True.
    enable_locateanything: bool = True

    def to_dict(self) -> dict:
        return {
            "recent_projects": self.recent_projects,
            "theme": self.theme,
            "auto_save": self.auto_save,
            "default_conf_threshold": self.default_conf_threshold,
            "default_iou_threshold": self.default_iou_threshold,
            "overlap_iou_threshold": self.overlap_iou_threshold,
            "script_tools": self.script_tools,
            "window_geometry": self.window_geometry,
            "classify_grid_density": self.classify_grid_density,
            "classify_grid_sort": self.classify_grid_sort,
            "classify_preview_width": self.classify_preview_width,
            "classify_preview_visible": self.classify_preview_visible,
            "annotation_panel_splitter_sizes": self.annotation_panel_splitter_sizes,
            "annotation_panel_collapsed": self.annotation_panel_collapsed,
            "enable_locateanything": self.enable_locateanything,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AppConfig:
        raw_script_tools = d.get("script_tools", {})
        script_tools: dict[str, str] = {}
        if isinstance(raw_script_tools, dict):
            script_tools = {
                str(k): v for k, v in raw_script_tools.items() if isinstance(v, str)
            }

        raw_sizes = d.get("annotation_panel_splitter_sizes", [])
        splitter_sizes: list[int] = []
        if isinstance(raw_sizes, list):
            splitter_sizes = [int(x) for x in raw_sizes if isinstance(x, int)]

        raw_collapsed = d.get("annotation_panel_collapsed", {})
        collapsed: dict[str, bool] = {}
        if isinstance(raw_collapsed, dict):
            collapsed = {
                str(k): bool(v) for k, v in raw_collapsed.items()
                if isinstance(v, bool)
            }

        return cls(
            recent_projects=d.get("recent_projects", []),
            theme=d.get("theme", "dark"),
            auto_save=d.get("auto_save", True),
            default_conf_threshold=d.get("default_conf_threshold", 0.5),
            default_iou_threshold=d.get("default_iou_threshold", 0.45),
            overlap_iou_threshold=d.get("overlap_iou_threshold", 0.5),
            script_tools=script_tools,
            window_geometry=d.get("window_geometry", {"x": 100, "y": 100, "width": 1400, "height": 900}),
            classify_grid_density=d.get("classify_grid_density", 96),
            classify_grid_sort=d.get("classify_grid_sort", "filename"),
            classify_preview_width=d.get("classify_preview_width", 320),
            classify_preview_visible=d.get("classify_preview_visible", True),
            annotation_panel_splitter_sizes=splitter_sizes,
            annotation_panel_collapsed=collapsed,
            enable_locateanything=bool(d.get("enable_locateanything", True)),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> AppConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return cls()

    def add_recent_project(self, project_path: str) -> None:
        """Add a project to recent list (most recent first, max 10)."""
        if project_path in self.recent_projects:
            self.recent_projects.remove(project_path)
        self.recent_projects.insert(0, project_path)
        self.recent_projects = self.recent_projects[:10]
