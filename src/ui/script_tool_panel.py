"""Script tool panel — editable Python script runner view with live output.

Pure view: tool-file CRUD delegates to ``core.script_tools.ToolRepository``
and the process lifecycle to ``controllers.script_tools.ScriptRunner``. The
panel keeps only UI construction, dirty/unsaved-confirm flow, running-state
visuals and output presentation (timestamp lines, separators, ``[系统]``
prefixes — the runner emits data only).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
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

from src.controllers.script_tools import ScriptRunner
from src.core.config import AppConfig
from src.core.script_tools import ToolRepository
from src.ui.theme import set_button_role, text_style


class ScriptToolPanel(QWidget):
    """Python script editor and runner panel."""

    status_changed = pyqtSignal(str)

    def __init__(
        self,
        app_config: AppConfig | None = None,
        config_path: Path | str | None = None,
        tools_dir: Path | str | None = None,
        parent=None,
        *,
        repository: ToolRepository | None = None,
        runner: ScriptRunner | None = None,
    ):
        super().__init__(parent)
        self._repo = repository if repository is not None else ToolRepository(tools_dir)
        self._runner = runner if runner is not None else ScriptRunner(parent=self)

        self._working_dir = Path.cwd()

        self._current_tool_path: Path | None = None
        self._tool_buttons: dict[Path, QPushButton] = {}
        self._tool_files: list[Path] = []
        self._is_dirty = False
        self._loading_script = False

        self._repo.ensure_builtin_tools()
        self._repo.migrate_legacy(app_config, Path(config_path) if config_path else None)

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

        self._runner.output.connect(self._append_text)
        self._runner.process_error.connect(self._on_process_error)
        self._runner.finished.connect(self._on_script_finished)

    def _clear_tool_buttons(self) -> None:
        while self._tool_list_layout.count():
            item = self._tool_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_tool_buttons(self, select_path: Path | None = None) -> None:
        self._tool_files = self._repo.list_tools()
        self._clear_tool_buttons()
        self._tool_buttons = {}

        if not self._tool_files:
            default_path = self._repo.create_tool("新建工具")
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
            script = self._repo.load_tool(tool_path)
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
        if self._runner.is_running:
            QMessageBox.information(self, "提示", "脚本执行中，停止后再添加工具")
            return

        if not self._maybe_resolve_unsaved():
            return

        name, ok = QInputDialog.getText(self, "添加工具", "请输入工具名称:")
        if not ok:
            return

        existing = self._repo.find_tool(name)
        if existing is not None:
            QMessageBox.information(self, "工具已存在", "工具已存在，已切换到该工具")
            self._refresh_tool_buttons(select_path=existing)
            return

        tool_path = self._repo.create_tool(name)
        if tool_path is None:
            QMessageBox.warning(self, "名称无效", "工具名称不能为空")
            return

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
            self._repo.save_tool(self._current_tool_path, script)
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
        if self._runner.is_running:
            return

        script = self._editor.toPlainText().strip()
        if not script:
            QMessageBox.warning(self, "脚本为空", "请先输入 Python 脚本内容")
            return

        started = self._runner.run(script, self._working_dir)

        self._append_line(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行脚本")
        self._append_line(f"工作目录: {self._working_dir}")
        self._append_line("-" * 50)

        if not started:
            self._append_line("脚本启动失败")
            self._set_running_state(False)
            self.status_changed.emit("脚本启动失败")
            return

        self._set_running_state(True)
        self.status_changed.emit("脚本执行中...")

    def stop_script(self) -> None:
        """Stop currently running script if needed."""
        if not self._runner.is_running:
            return

        self._append_line("\n[系统] 正在停止脚本...")
        self._runner.stop()

    def _on_process_error(self, error_name: str) -> None:
        self._append_line(f"\n[系统] 进程错误: {error_name}")

    def _on_script_finished(self, exit_code: int, normal_exit: bool) -> None:
        status = "正常退出" if normal_exit else "异常终止"
        self._append_line(f"\n[系统] 脚本结束: {status} | 退出码: {exit_code}")

        self._set_running_state(False)
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

    def prepare_close(self) -> bool:
        """Handle save/cancel flow before app close. Return True if close can continue."""
        if self._runner.is_running:
            self.stop_script()
        return self._maybe_resolve_unsaved()

    def closeEvent(self, event) -> None:
        if not self.prepare_close():
            event.ignore()
            return
        self._runner.cleanup_temp()
        super().closeEvent(event)
