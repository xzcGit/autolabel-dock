"""Script tool panel — editable Python script runner with live output."""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QProcess, pyqtSignal
from PyQt5.QtGui import QFontDatabase, QKeySequence, QTextCursor
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPlainTextEdit,
    QPushButton,
    QLabel,
    QMessageBox,
    QScrollArea,
    QShortcut,
    QInputDialog,
)

from src.core.config import AppConfig
from src.ui.theme import set_button_role, text_style

_DEFAULT_SCRIPT = """# 在这里编写你的 Python 脚本\nprint('Hello AutoLabel!')\n"""
_DEFAULT_TOOLS_DIR = Path.home() / ".autolabel" / "tools"
_BUILTIN_CROP_FILENAME = "内置_按标注框裁剪图片.py"

_CROP_BY_BBOX_SCRIPT = '''"""按标注框裁剪图片（AutoLabel Dock 内置脚本）

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


class ScriptToolPanel(QWidget):
    """Python script editor and runner panel."""

    status_changed = pyqtSignal(str)

    def __init__(
        self,
        app_config: AppConfig | None = None,
        config_path: Path | str | None = None,
        tools_dir: Path | str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._app_config = app_config
        self._config_path = Path(config_path) if config_path else None
        self._tools_dir = Path(tools_dir) if tools_dir else _DEFAULT_TOOLS_DIR
        self._tools_dir.mkdir(parents=True, exist_ok=True)

        self._process: QProcess | None = None
        self._temp_script_path: Path | None = None
        self._working_dir = Path.cwd()

        self._current_tool_path: Path | None = None
        self._tool_buttons: dict[Path, QPushButton] = {}
        self._tool_files: list[Path] = []
        self._is_dirty = False
        self._loading_script = False

        self._ensure_builtin_tools()
        self._migrate_legacy_script_tools()

        self._init_ui()
        self._connect_signals()
        self._refresh_tool_buttons()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        top_layout = QHBoxLayout()

        left_group = QGroupBox("快捷小工具")
        left_group.setMaximumWidth(260)
        left_layout = QVBoxLayout(left_group)

        self._tools_scroll = QScrollArea()
        self._tools_scroll.setWidgetResizable(True)

        self._tool_list_widget = QWidget()
        self._tool_list_layout = QVBoxLayout(self._tool_list_widget)
        self._tool_list_layout.setContentsMargins(4, 4, 4, 4)
        self._tool_list_layout.setSpacing(6)
        self._tool_list_layout.addStretch()
        self._tools_scroll.setWidget(self._tool_list_widget)

        self._btn_add_tool = QPushButton("+ 添加工具")
        set_button_role(self._btn_add_tool, "secondary")

        left_layout.addWidget(self._tools_scroll)
        left_layout.addWidget(self._btn_add_tool)

        top_layout.addWidget(left_group)

        editor_group = QGroupBox("Python 脚本")
        editor_layout = QVBoxLayout(editor_group)

        info_layout = QHBoxLayout()
        self._python_label = QLabel(f"解释器: {sys.executable}")
        self._cwd_label = QLabel()
        self._cwd_label.setToolTip(str(self._working_dir))
        self._save_state_label = QLabel("状态: 已保存")
        self._save_state_label.setStyleSheet(text_style("success"))
        info_layout.addWidget(self._python_label)
        info_layout.addStretch()
        info_layout.addWidget(self._cwd_label)
        info_layout.addSpacing(12)
        info_layout.addWidget(self._save_state_label)
        editor_layout.addLayout(info_layout)

        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText("在这里输入 Python 脚本，然后点击“执行脚本”。")
        mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._editor.setFont(mono_font)
        editor_layout.addWidget(self._editor)

        btn_layout = QHBoxLayout()
        self._btn_run = QPushButton("执行脚本")
        set_button_role(self._btn_run, "primary")
        self._btn_stop = QPushButton("停止执行")
        set_button_role(self._btn_stop, "danger")
        self._btn_stop.setEnabled(False)
        self._btn_clear = QPushButton("清空输出")
        set_button_role(self._btn_clear, "secondary")
        self._save_hint_label = QLabel("快捷键: Ctrl+S 保存")
        self._save_hint_label.setStyleSheet(text_style("hint"))

        btn_layout.addWidget(self._btn_run)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self._save_hint_label)
        editor_layout.addLayout(btn_layout)

        top_layout.addWidget(editor_group, 1)

        layout.addLayout(top_layout, 2)

        output_group = QGroupBox("执行输出")
        output_layout = QVBoxLayout(output_group)
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(mono_font)
        output_layout.addWidget(self._output)
        layout.addWidget(output_group, 1)

        self._save_shortcut = QShortcut(QKeySequence.Save, self)
        self.set_working_directory(self._working_dir)

    def _connect_signals(self) -> None:
        self._btn_add_tool.clicked.connect(self._on_add_tool_clicked)
        self._btn_run.clicked.connect(self._on_run_clicked)
        self._btn_stop.clicked.connect(self.stop_script)
        self._btn_clear.clicked.connect(self._output.clear)
        self._editor.textChanged.connect(self._on_editor_text_changed)
        self._save_shortcut.activated.connect(self._on_save_shortcut)

    def _ensure_builtin_tools(self) -> None:
        crop_path = self._tools_dir / _BUILTIN_CROP_FILENAME
        if not crop_path.exists():
            crop_path.write_text(_CROP_BY_BBOX_SCRIPT, encoding="utf-8")

    def _migrate_legacy_script_tools(self) -> None:
        if self._app_config is None or not self._app_config.script_tools:
            return

        migrated = False
        for name, script in self._app_config.script_tools.items():
            if not isinstance(script, str):
                continue
            fname = self._filename_from_tool_name(name)
            if not fname:
                continue
            path = self._tools_dir / fname
            if path.exists():
                continue
            path.write_text(script, encoding="utf-8")
            migrated = True

        if migrated:
            self._app_config.script_tools = {}
            if self._config_path is not None:
                self._app_config.save(self._config_path)

    def _filename_from_tool_name(self, name: str) -> str:
        cleaned = str(name).strip().replace("\n", " ").replace("\r", " ")
        if cleaned.lower().endswith(".py"):
            cleaned = cleaned[:-3]
        for ch in '/\\:*?"<>|':
            cleaned = cleaned.replace(ch, "_")
        cleaned = cleaned.strip().strip(".")
        if not cleaned:
            return ""
        return f"{cleaned}.py"

    def _clear_tool_buttons(self) -> None:
        while self._tool_list_layout.count():
            item = self._tool_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_tool_buttons(self, select_path: Path | None = None) -> None:
        self._tool_files = sorted(self._tools_dir.glob("*.py"), key=lambda p: p.name.lower())
        self._clear_tool_buttons()
        self._tool_buttons = {}

        if not self._tool_files:
            default_path = self._tools_dir / "新建工具.py"
            default_path.write_text(_DEFAULT_SCRIPT, encoding="utf-8")
            self._tool_files = [default_path]

        for tool_path in self._tool_files:
            btn = QPushButton(tool_path.stem)
            btn.setCheckable(True)
            set_button_role(btn, "list-item")
            btn.setToolTip(str(tool_path))
            btn.clicked.connect(lambda _checked=False, p=tool_path: self._on_tool_button_clicked(p))
            self._tool_buttons[tool_path] = btn
            self._tool_list_layout.addWidget(btn)

        self._tool_list_layout.addStretch()

        target = select_path
        if target is None or target not in self._tool_buttons:
            if self._current_tool_path in self._tool_buttons:
                target = self._current_tool_path
            else:
                target = self._tool_files[0]

        self._load_tool(target)

    def _set_selected_tool_button(self) -> None:
        for path, btn in self._tool_buttons.items():
            is_current = path == self._current_tool_path
            btn.setChecked(is_current)

    def _update_button_labels(self) -> None:
        for path, btn in self._tool_buttons.items():
            text = path.stem
            if self._is_dirty and path == self._current_tool_path:
                text += " *"
            btn.setText(text)

    def _set_dirty(self, dirty: bool) -> None:
        self._is_dirty = dirty
        if dirty:
            self._save_state_label.setText("状态: 未保存*")
            self._save_state_label.setStyleSheet(text_style("warning"))
        else:
            self._save_state_label.setText("状态: 已保存")
            self._save_state_label.setStyleSheet(text_style("success"))
        self._update_button_labels()

    def _load_tool(self, tool_path: Path) -> None:
        try:
            script = tool_path.read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "读取失败", f"无法读取脚本: {tool_path.name}\n{exc}")
            return

        self._loading_script = True
        try:
            self._editor.setPlainText(script)
        finally:
            self._loading_script = False

        self._current_tool_path = tool_path
        self._set_dirty(False)
        self._set_selected_tool_button()
        self.status_changed.emit(f"已加载工具: {tool_path.stem}")

    def _maybe_resolve_unsaved(self) -> bool:
        if not self._is_dirty:
            return True

        reply = QMessageBox.question(
            self,
            "未保存更改",
            "当前脚本有未保存修改，是否先保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if reply == QMessageBox.Save:
            return self._save_current_script()
        if reply == QMessageBox.Discard:
            self._set_dirty(False)
            return True
        return False

    def _on_tool_button_clicked(self, tool_path: Path) -> None:
        if tool_path == self._current_tool_path:
            self._set_selected_tool_button()
            return

        if not self._maybe_resolve_unsaved():
            self._set_selected_tool_button()
            return

        self._load_tool(tool_path)

    def _on_add_tool_clicked(self) -> None:
        if self._process is not None:
            QMessageBox.information(self, "提示", "脚本执行中，停止后再添加工具")
            return

        if not self._maybe_resolve_unsaved():
            return

        name, ok = QInputDialog.getText(self, "添加工具", "请输入工具名称:")
        if not ok:
            return

        filename = self._filename_from_tool_name(name)
        if not filename:
            QMessageBox.warning(self, "名称无效", "工具名称不能为空")
            return

        tool_path = self._tools_dir / filename
        if tool_path.exists():
            QMessageBox.information(self, "工具已存在", "工具已存在，已切换到该工具")
            self._refresh_tool_buttons(select_path=tool_path)
            return

        tool_path.write_text(_DEFAULT_SCRIPT, encoding="utf-8")
        self._refresh_tool_buttons(select_path=tool_path)
        self.status_changed.emit(f"已添加工具: {tool_path.stem}")

    def _on_editor_text_changed(self) -> None:
        if self._loading_script:
            return
        if self._current_tool_path is None:
            return
        if not self._is_dirty:
            self._set_dirty(True)

    def _on_save_shortcut(self) -> None:
        self._save_current_script()

    def _save_current_script(self) -> bool:
        if self._current_tool_path is None:
            QMessageBox.warning(self, "保存失败", "当前没有可保存的工具")
            return False

        script = self._editor.toPlainText()
        try:
            self._current_tool_path.write_text(script, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存脚本: {self._current_tool_path.name}\n{exc}")
            return False

        self._set_dirty(False)
        self.status_changed.emit(f"已保存工具: {self._current_tool_path.stem}")
        return True

    def set_working_directory(self, path: Path | str | None) -> None:
        """Set script working directory shown in UI and used for execution."""
        if path is None:
            self._working_dir = Path.cwd()
        else:
            candidate = Path(path)
            self._working_dir = candidate if candidate.exists() else Path.cwd()
        cwd_text = f"工作目录: {self._working_dir}"
        self._cwd_label.setText(cwd_text)
        self._cwd_label.setToolTip(str(self._working_dir))

    def _on_run_clicked(self) -> None:
        if self._process is not None:
            return

        script = self._editor.toPlainText().strip()
        if not script:
            QMessageBox.warning(self, "脚本为空", "请先输入 Python 脚本内容")
            return

        self._cleanup_temp_script()
        fd, script_path = tempfile.mkstemp(prefix="autolabel_script_", suffix=".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(script)
                if not script.endswith("\n"):
                    f.write("\n")
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            Path(script_path).unlink(missing_ok=True)
            raise

        self._temp_script_path = Path(script_path)

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-u", str(self._temp_script_path)])
        process.setWorkingDirectory(str(self._working_dir))

        process.readyReadStandardOutput.connect(self._on_stdout)
        process.readyReadStandardError.connect(self._on_stderr)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_finished)

        self._process = process
        self._set_running_state(True)

        self._append_line(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行脚本")
        self._append_line(f"工作目录: {self._working_dir}")
        self._append_line("-" * 50)

        process.start()
        if not process.waitForStarted(1000):
            self._append_line("脚本启动失败")
            self._set_running_state(False)
            process.deleteLater()
            self._process = None
            self._cleanup_temp_script()
            self.status_changed.emit("脚本启动失败")
            return
        self.status_changed.emit("脚本执行中...")

    def stop_script(self) -> None:
        """Stop currently running script if needed."""
        if self._process is None:
            return

        self._append_line("\n[系统] 正在停止脚本...")
        self._process.terminate()
        if not self._process.waitForFinished(1500):
            self._process.kill()
            self._process.waitForFinished(1000)

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput())
        self._append_text(data.decode("utf-8", errors="replace"))

    def _on_stderr(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardError())
        self._append_text(data.decode("utf-8", errors="replace"))

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        error_names = {
            QProcess.FailedToStart: "启动失败",
            QProcess.Crashed: "进程崩溃",
            QProcess.Timedout: "超时",
            QProcess.WriteError: "写入错误",
            QProcess.ReadError: "读取错误",
            QProcess.UnknownError: "未知错误",
        }
        self._append_line(f"\n[系统] 进程错误: {error_names.get(error, str(int(error)))}")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status = "正常退出" if exit_status == QProcess.NormalExit else "异常终止"
        self._append_line(f"\n[系统] 脚本结束: {status} | 退出码: {exit_code}")

        if self._process is not None:
            self._process.deleteLater()
            self._process = None

        self._set_running_state(False)
        self._cleanup_temp_script()
        self.status_changed.emit(f"脚本执行完成 (退出码: {exit_code})")

    def _set_running_state(self, running: bool) -> None:
        self._btn_run.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._editor.setReadOnly(running)
        self._btn_add_tool.setEnabled(not running)
        for btn in self._tool_buttons.values():
            btn.setEnabled(not running)

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self._output.moveCursor(QTextCursor.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(QTextCursor.End)

    def _append_line(self, text: str) -> None:
        self._output.appendPlainText(text)
        self._output.moveCursor(QTextCursor.End)

    def _cleanup_temp_script(self) -> None:
        if self._temp_script_path is None:
            return
        try:
            self._temp_script_path.unlink(missing_ok=True)
        finally:
            self._temp_script_path = None

    def prepare_close(self) -> bool:
        """Handle save/cancel flow before app close. Return True if close can continue."""
        if self._process is not None:
            self.stop_script()
        return self._maybe_resolve_unsaved()

    def closeEvent(self, event) -> None:
        if not self.prepare_close():
            event.ignore()
            return
        self._cleanup_temp_script()
        super().closeEvent(event)
