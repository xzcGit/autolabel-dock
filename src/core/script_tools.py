"""Script tool repository — Qt-free file layer for the 小工具 tab.

Owns the ``~/.autolabel/tools`` directory: builtin tool installation, legacy
``AppConfig.script_tools`` migration, tool-name sanitization and tool file
CRUD. Script templates live here as module constants. No Qt imports — the
ScriptToolPanel (ui layer) is a pure view delegating to this repository; the
process lifecycle lives in the ScriptRunner controller
(controllers layer).
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_TOOLS_DIR = Path.home() / ".autolabel" / "tools"
BUILTIN_CROP_FILENAME = "内置_按标注框裁剪图片.py"

DEFAULT_SCRIPT = """# 在这里编写你的 Python 脚本\nprint('Hello AutoLabel!')\n"""

CROP_BY_BBOX_SCRIPT = '''"""按标注框裁剪图片（AutoLabel Dock 内置脚本）

使用方式:
1) 将工作目录设置为项目根目录（包含 project.json）
2) 直接运行本脚本
3) 结果默认输出到项目目录下 crops/
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(".")
OUTPUT_DIR = PROJECT_DIR / "crops"
ONLY_CONFIRMED = False
KEEP_CLASS_SUBDIR = True
LIST_LIMIT = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def bbox_to_xyxy(bbox: list[float] | tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized (cx, cy, w, h) to pixel box (x1, y1, x2, y2)."""
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2.0) * width)
    y1 = int((cy - bh / 2.0) * height)
    x2 = int((cx + bw / 2.0) * width)
    y2 = int((cy + bh / 2.0) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    return x1, y1, x2, y2


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_image_for_json(
    label_file: Path,
    doc: dict,
    image_by_name: dict[str, Path],
    images_by_stem: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    declared = str(doc.get("image_path", "")).strip()
    if declared:
        declared_name = Path(declared).name
        if declared_name in image_by_name:
            return image_by_name[declared_name], ""

    stem = Path(declared).stem if declared else label_file.stem
    candidates = images_by_stem.get(stem, [])
    if len(candidates) == 1:
        return candidates[0], ""
    if len(candidates) > 1:
        return None, f"同名 stem 对应多张图片: {stem}"
    return None, "没有匹配到图片"


def print_preview(title: str, rows: list[str]) -> None:
    print(f"{title}: {len(rows)}")
    for row in rows[:LIST_LIMIT]:
        print(f"  - {row}")
    if len(rows) > LIST_LIMIT:
        print(f"  ... 其余 {len(rows) - LIST_LIMIT} 条省略")


def main() -> None:
    project_json = PROJECT_DIR / "project.json"
    if not project_json.exists():
        print("未找到 project.json，请将工作目录切到项目根目录")
        return

    try:
        config = json.loads(project_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"project.json 读取失败: {exc}")
        return

    image_dir = Path(config.get("image_dir", "images"))
    if not image_dir.is_absolute():
        image_dir = PROJECT_DIR / image_dir

    label_dir = Path(config.get("label_dir", "labels"))
    if not label_dir.is_absolute():
        label_dir = PROJECT_DIR / label_dir

    if not image_dir.exists():
        print(f"图片目录不存在: {image_dir}")
        return
    if not label_dir.exists():
        print(f"标签目录不存在: {label_dir}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = collect_images(image_dir)
    image_by_name = {p.name: p for p in images}
    images_by_stem: dict[str, list[Path]] = {}
    for image_path in images:
        images_by_stem.setdefault(image_path.stem, []).append(image_path)

    label_files = sorted(label_dir.glob("*.json"))

    print(f"图片数量: {len(images)}")
    print(f"标签文件数量: {len(label_files)}")
    print(f"图片目录: {image_dir}")
    print(f"输出目录: {OUTPUT_DIR}")

    broken_json: list[str] = []
    unmatched_json: list[str] = []
    matched_records: list[tuple[Path, Path, dict]] = []
    matched_image_paths: set[Path] = set()

    for label_file in label_files:
        try:
            doc = json.loads(label_file.read_text(encoding="utf-8"))
        except Exception as exc:
            broken_json.append(f"{label_file.name} | 解析失败: {exc}")
            continue

        image_path, reason = resolve_image_for_json(label_file, doc, image_by_name, images_by_stem)
        if image_path is None:
            unmatched_json.append(f"{label_file.name} | {reason}")
            continue

        matched_records.append((label_file, image_path, doc))
        matched_image_paths.add(image_path)

    unmatched_images = [p.name for p in images if p not in matched_image_paths]

    saved_count = 0
    invalid_bbox = 0
    open_failed = 0
    matched_without_bbox = 0

    for label_file, image_path, doc in matched_records:
        annotations = doc.get("annotations", [])
        if not annotations:
            matched_without_bbox += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                for i, ann in enumerate(annotations):
                    bbox = ann.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    if ONLY_CONFIRMED and not ann.get("confirmed", False):
                        continue

                    try:
                        bbox_vals = [float(v) for v in bbox]
                    except (TypeError, ValueError):
                        invalid_bbox += 1
                        continue

                    x1, y1, x2, y2 = bbox_to_xyxy(bbox_vals, width, height)
                    if x2 <= x1 or y2 <= y1:
                        invalid_bbox += 1
                        continue

                    crop = img.crop((x1, y1, x2, y2))
                    class_name = str(ann.get("class_name", "unknown"))
                    out_dir = OUTPUT_DIR / class_name if KEEP_CLASS_SUBDIR else OUTPUT_DIR
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"{image_path.stem}_{i:03d}.jpg"
                    crop.save(out_dir / out_name, quality=95)
                    saved_count += 1
        except Exception:
            open_failed += 1

    print("\\n--- 匹配统计汇总 ---")
    print(f"匹配成功的图像-JSON对: {len(matched_records)}")
    print(f"未匹配JSON: {len(unmatched_json)}")
    print(f"未匹配图片: {len(unmatched_images)}")
    print(f"损坏JSON: {len(broken_json)}")
    print(f"匹配但无标注的JSON: {matched_without_bbox}")
    print(f"无效标注框数量: {invalid_bbox}")
    print(f"图片打开失败数量: {open_failed}")
    print(f"裁剪完成: 保存 {saved_count} 个目标框")

    if unmatched_json:
        print_preview("未匹配JSON列表", unmatched_json)
    if unmatched_images:
        print_preview("未匹配图片列表", unmatched_images)
    if broken_json:
        print_preview("损坏JSON列表", broken_json)


if __name__ == "__main__":
    main()
'''


def filename_from_tool_name(name: str) -> str:
    """Sanitize a user-supplied tool name into a ``*.py`` filename.

    Returns ``""`` when nothing usable remains after sanitization.
    """
    cleaned = str(name).strip().replace("\n", " ").replace("\r", " ")
    if cleaned.lower().endswith(".py"):
        cleaned = cleaned[:-3]
    for ch in '/\\:*?"<>|':
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned.strip().strip(".")
    if not cleaned:
        return ""
    return f"{cleaned}.py"


class ToolRepository:
    """File-backed repository for user script tools (one ``.py`` per tool)."""

    def __init__(self, tools_dir: Path | str | None = None):
        self._tools_dir = Path(tools_dir) if tools_dir else DEFAULT_TOOLS_DIR
        self._tools_dir.mkdir(parents=True, exist_ok=True)

    @property
    def tools_dir(self) -> Path:
        return self._tools_dir

    def ensure_builtin_tools(self) -> None:
        """Install builtin tools that are missing. Existing files are never overwritten."""
        crop_path = self._tools_dir / BUILTIN_CROP_FILENAME
        if not crop_path.exists():
            crop_path.write_text(CROP_BY_BBOX_SCRIPT, encoding="utf-8")

    def migrate_legacy(
        self,
        app_config: AppConfig | None,
        config_path: Path | str | None,
    ) -> bool:
        """Migrate legacy ``AppConfig.script_tools`` entries to tool files.

        Only when at least one tool was actually migrated is
        ``app_config.script_tools`` cleared (and saved when ``config_path``
        is given). Returns whether a migration happened.
        """
        if app_config is None or not app_config.script_tools:
            return False

        migrated = False
        for name, script in app_config.script_tools.items():
            if not isinstance(script, str):
                continue
            fname = filename_from_tool_name(name)
            if not fname:
                continue
            path = self._tools_dir / fname
            if path.exists():
                continue
            path.write_text(script, encoding="utf-8")
            migrated = True

        if migrated:
            app_config.script_tools = {}
            if config_path is not None:
                app_config.save(Path(config_path))
        return migrated

    def list_tools(self) -> list[Path]:
        """Return all tool files sorted by name (case-insensitive)."""
        return sorted(self._tools_dir.glob("*.py"), key=lambda p: p.name.lower())

    def find_tool(self, name: str) -> Path | None:
        """Return the existing tool file for ``name``, or None."""
        filename = filename_from_tool_name(name)
        if not filename:
            return None
        path = self._tools_dir / filename
        return path if path.exists() else None

    def create_tool(self, name: str, content: str = DEFAULT_SCRIPT) -> Path | None:
        """Create a new tool file from a user-supplied name.

        Returns None when the sanitized name is empty. When a tool with the
        same filename already exists it is returned untouched (never
        overwritten).
        """
        filename = filename_from_tool_name(name)
        if not filename:
            return None
        path = self._tools_dir / filename
        if path.exists():
            return path
        path.write_text(content, encoding="utf-8")
        return path

    def load_tool(self, path: Path) -> str:
        """Read a tool's script text. OSError propagates to the caller."""
        return Path(path).read_text(encoding="utf-8")

    def save_tool(self, path: Path, text: str) -> None:
        """Write a tool's script text. OSError propagates to the caller."""
        Path(path).write_text(text, encoding="utf-8")
