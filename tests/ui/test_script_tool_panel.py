"""Tests for ScriptToolPanel."""
from __future__ import annotations

import time

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QMessageBox


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QCoreApplication.processEvents()
    return predicate()


class TestScriptToolPanel:
    def test_creates_and_initializes_builtin_tool(self, qapp, tmp_path):
        from src.core.script_tools import BUILTIN_CROP_FILENAME
        from src.ui.script_tool_panel import ScriptToolPanel

        tools_dir = tmp_path / "tools"
        panel = ScriptToolPanel(tools_dir=tools_dir)

        assert panel._editor is not None
        assert panel._output is not None
        assert panel._btn_run is not None
        assert panel._btn_stop is not None
        assert panel._btn_add_tool is not None
        assert panel._save_state_label.text() == "状态: 已保存"
        assert (tools_dir / BUILTIN_CROP_FILENAME).exists()
        assert len(panel._tool_buttons) >= 1

    def test_builtin_crop_tool_can_load_and_edit(self, qapp, tmp_path):
        from src.core.script_tools import BUILTIN_CROP_FILENAME
        from src.ui.script_tool_panel import ScriptToolPanel

        tools_dir = tmp_path / "tools"
        panel = ScriptToolPanel(tools_dir=tools_dir)

        crop_path = tools_dir / BUILTIN_CROP_FILENAME
        panel._on_tool_button_clicked(crop_path)

        script = panel._editor.toPlainText()
        assert "bbox_to_xyxy" in script
        assert "from PIL import Image" in script
        assert "匹配统计汇总" in script
        assert "未匹配JSON列表" in script

        panel._editor.appendPlainText("# edited")
        assert panel._is_dirty is True
        assert panel._save_state_label.text() == "状态: 未保存*"

    def test_add_tool_creates_file_and_selects_it(self, qapp, tmp_path, monkeypatch):
        from src.ui.script_tool_panel import ScriptToolPanel

        tools_dir = tmp_path / "tools"
        panel = ScriptToolPanel(tools_dir=tools_dir)

        monkeypatch.setattr(
            "src.ui.script_tool_panel.QInputDialog.getText",
            lambda *args, **kwargs: ("demo_tool", True),
        )

        panel._on_add_tool_clicked()

        tool_path = tools_dir / "demo_tool.py"
        assert tool_path.exists()
        assert panel._current_tool_path == tool_path
        assert tool_path in panel._tool_buttons

    def test_ctrl_s_saves_current_tool_and_clears_dirty(self, qapp, tmp_path):
        from src.ui.script_tool_panel import ScriptToolPanel

        tools_dir = tmp_path / "tools"
        panel = ScriptToolPanel(tools_dir=tools_dir)

        assert panel._current_tool_path is not None
        current_path = panel._current_tool_path
        panel._editor.setPlainText("print('saved by ctrl+s')")
        assert panel._is_dirty is True

        panel._save_shortcut.activated.emit()

        assert current_path.read_text(encoding="utf-8") == "print('saved by ctrl+s')"
        assert panel._is_dirty is False
        assert panel._save_state_label.text() == "状态: 已保存"

    def test_switch_tool_prompts_when_unsaved(self, qapp, tmp_path, monkeypatch):
        from src.ui.script_tool_panel import ScriptToolPanel

        tools_dir = tmp_path / "tools"
        tool_a = tools_dir / "a_tool.py"
        tool_b = tools_dir / "b_tool.py"
        tools_dir.mkdir(parents=True, exist_ok=True)
        tool_a.write_text("print('a')\n", encoding="utf-8")
        tool_b.write_text("print('b')\n", encoding="utf-8")

        panel = ScriptToolPanel(tools_dir=tools_dir)
        panel._refresh_tool_buttons(select_path=tool_a)
        panel._editor.appendPlainText("# dirty")
        assert panel._is_dirty is True

        monkeypatch.setattr(
            "src.ui.script_tool_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Cancel,
        )
        panel._on_tool_button_clicked(tool_b)
        assert panel._current_tool_path == tool_a
        assert panel._is_dirty is True

        monkeypatch.setattr(
            "src.ui.script_tool_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Discard,
        )
        panel._on_tool_button_clicked(tool_b)
        assert panel._current_tool_path == tool_b
        assert panel._is_dirty is False

    def test_prepare_close_can_cancel(self, qapp, tmp_path, monkeypatch):
        from src.ui.script_tool_panel import ScriptToolPanel

        panel = ScriptToolPanel(tools_dir=tmp_path / "tools")
        panel._editor.appendPlainText("# dirty")

        monkeypatch.setattr(
            "src.ui.script_tool_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Cancel,
        )
        assert panel.prepare_close() is False

        monkeypatch.setattr(
            "src.ui.script_tool_panel.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.Discard,
        )
        assert panel.prepare_close() is True

    def test_run_script_appends_output(self, qapp, tmp_path):
        from src.ui.script_tool_panel import ScriptToolPanel

        panel = ScriptToolPanel(tools_dir=tmp_path / "tools")
        panel.set_working_directory(tmp_path)
        panel._editor.setPlainText("print('hello script panel')")

        panel._on_run_clicked()

        assert panel._runner.is_running
        assert _wait_until(lambda: not panel._runner.is_running)
        output = panel._output.toPlainText()
        assert "hello script panel" in output
        assert "退出码: 0" in output

    def test_failed_start_renders_message_and_stays_idle(self, qapp, tmp_path):
        from src.controllers.script_tools import ScriptRunner
        from src.ui.script_tool_panel import ScriptToolPanel

        runner = ScriptRunner(program="/nonexistent/interpreter-xyz")
        panel = ScriptToolPanel(tools_dir=tmp_path / "tools", runner=runner)
        statuses = []
        panel.status_changed.connect(statuses.append)
        panel._editor.setPlainText("print('never runs')")

        panel._on_run_clicked()

        assert not panel._runner.is_running
        output = panel._output.toPlainText()
        assert "开始执行脚本" in output
        assert "脚本启动失败" in output
        # single report — FailedToStart is deduped in the runner
        assert "进程错误" not in output
        assert statuses[-1] == "脚本启动失败"
        assert panel._btn_run.isEnabled()
        assert not panel._btn_stop.isEnabled()
