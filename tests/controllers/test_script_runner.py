"""Tests for src/controllers/script_tools.py — ScriptRunner QProcess lifecycle.

Runs real python subprocesses (same pattern as the panel's end-to-end test).
"""
from __future__ import annotations

import time

from PyQt5.QtCore import QCoreApplication

from src.controllers.script_tools import ScriptRunner


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QCoreApplication.processEvents()
    return predicate()


class _Recorder:
    """Collects all four runner signals."""

    def __init__(self, runner: ScriptRunner):
        self.outputs: list[str] = []
        self.errors: list[str] = []
        self.finished: list[tuple[int, bool]] = []
        self.failed_starts = 0
        runner.output.connect(self.outputs.append)
        runner.process_error.connect(self.errors.append)
        runner.finished.connect(lambda code, normal: self.finished.append((code, normal)))
        runner.failed_to_start.connect(self._on_failed)

    def _on_failed(self):
        self.failed_starts += 1

    @property
    def text(self) -> str:
        return "".join(self.outputs)


class TestScriptRunner:
    def test_run_emits_output_and_finished(self, qapp, tmp_path):
        runner = ScriptRunner()
        rec = _Recorder(runner)

        assert runner.run("print('hello runner')", tmp_path) is True
        assert runner.is_running

        assert _wait_until(lambda: not runner.is_running)
        assert "hello runner" in rec.text
        assert rec.finished == [(0, True)]
        assert rec.failed_starts == 0

    def test_stderr_flows_to_output(self, qapp, tmp_path):
        runner = ScriptRunner()
        rec = _Recorder(runner)

        script = "import sys\nsys.stderr.write('boom-err\\n')"
        assert runner.run(script, tmp_path) is True
        assert _wait_until(lambda: not runner.is_running)
        assert "boom-err" in rec.text

    def test_exit_code_forwarded(self, qapp, tmp_path):
        runner = ScriptRunner()
        rec = _Recorder(runner)

        assert runner.run("import sys\nsys.exit(3)", tmp_path) is True
        assert _wait_until(lambda: not runner.is_running)
        assert rec.finished == [(3, True)]

    def test_working_directory_respected(self, qapp, tmp_path):
        runner = ScriptRunner()
        _Recorder(runner)

        assert runner.run("open('marker.txt', 'w').write('x')", tmp_path) is True
        assert _wait_until(lambda: not runner.is_running)
        assert (tmp_path / "marker.txt").read_text() == "x"

    def test_second_run_while_running_returns_false(self, qapp, tmp_path):
        runner = ScriptRunner()
        rec = _Recorder(runner)

        assert runner.run("import time\ntime.sleep(30)", tmp_path) is True
        # double-run guard: silent refusal, no failed_to_start
        assert runner.run("print('nope')", tmp_path) is False
        assert rec.failed_starts == 0
        assert runner.is_running

        runner.stop()
        assert not runner.is_running

    def test_stop_is_synchronous_and_reports_crash_exit(self, qapp, tmp_path):
        runner = ScriptRunner()
        rec = _Recorder(runner)

        assert runner.run("import time\ntime.sleep(30)", tmp_path) is True
        runner.stop()

        # §6.2 pin: when stop() returns the process has terminated, the
        # finished signal has already been delivered, and temp is cleaned.
        assert not runner.is_running
        assert len(rec.finished) == 1
        assert rec.finished[0][1] is False  # CrashExit (terminated)
        assert "进程崩溃" in rec.errors
        assert runner._temp_script_path is None

    def test_temp_script_created_then_cleaned(self, qapp, tmp_path):
        runner = ScriptRunner()
        _Recorder(runner)

        assert runner.run("print('temp check')", tmp_path) is True
        temp_path = runner._temp_script_path
        assert temp_path is not None
        assert temp_path.name.startswith("autolabel_script_")
        assert temp_path.suffix == ".py"
        assert temp_path.exists()

        assert _wait_until(lambda: not runner.is_running)
        assert runner._temp_script_path is None
        assert not temp_path.exists()

    def test_failed_start_returns_false_and_dedupes_error(self, qapp, tmp_path):
        runner = ScriptRunner(program="/nonexistent/interpreter-xyz")
        rec = _Recorder(runner)

        assert runner.run("print('x')", tmp_path) is False

        assert rec.failed_starts == 1
        assert not runner.is_running
        assert runner._temp_script_path is None

        # Dedup pin: FailedToStart must not surface through process_error,
        # and finished must not fire for a process that never started.
        for _ in range(20):
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert rec.errors == []
        assert rec.finished == []
