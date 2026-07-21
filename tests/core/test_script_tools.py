"""Tests for src/core/script_tools.py — Qt-free tool repository.

Runs without a QApplication: the module under test must never import PyQt5.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.core.config import AppConfig
from src.core.script_tools import (
    BUILTIN_CROP_FILENAME,
    CROP_BY_BBOX_SCRIPT,
    DEFAULT_SCRIPT,
    ToolRepository,
    filename_from_tool_name,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestModuleIsQtFree:
    def test_source_has_no_qt_references(self):
        import src.core.script_tools as mod

        src_text = open(mod.__file__, encoding="utf-8").read()
        # The crop template legitimately mentions PIL, but never Qt.
        assert "PyQt5" not in src_text
        assert "pyqtgraph" not in src_text
        assert "src.ui" not in src_text

    def test_import_in_fresh_interpreter_pulls_no_qt(self):
        code = (
            "import sys; import src.core.script_tools; "
            "bad = [m for m in sys.modules if m == 'PyQt5' or m.startswith('PyQt5.') "
            "or m == 'pyqtgraph' or m.startswith('pyqtgraph.')]; "
            "assert not bad, bad"
        )
        subprocess.run(
            [sys.executable, "-c", code], check=True, cwd=str(_REPO_ROOT)
        )


class TestFilenameFromToolName:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("demo", "demo.py"),
            ("demo.py", "demo.py"),          # .py suffix stripped then re-added
            ("DEMO.PY", "DEMO.py"),          # suffix strip is case-insensitive
            ('a/b\\c:d*e?f"g<h>i|j', "a_b_c_d_e_f_g_h_i_j.py"),
            ("  spaced  ", "spaced.py"),
            ("line1\nline2", "line1 line2.py"),
            ("a\rb", "a b.py"),
            ("name.", "name.py"),            # trailing dots stripped
            ("...", ""),
            ("", ""),
            ("   ", ""),
            (".py", ""),
            ("..py", ""),
        ],
    )
    def test_sanitize_table(self, name, expected):
        assert filename_from_tool_name(name) == expected


class TestToolRepository:
    def test_init_creates_tools_dir(self, tmp_path):
        tools_dir = tmp_path / "nested" / "tools"
        repo = ToolRepository(tools_dir)
        assert tools_dir.is_dir()
        assert repo.tools_dir == tools_dir

    def test_init_accepts_str_path(self, tmp_path):
        repo = ToolRepository(str(tmp_path / "tools"))
        assert repo.tools_dir == tmp_path / "tools"
        assert repo.tools_dir.is_dir()

    def test_ensure_builtin_tools_writes_crop_script(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        repo.ensure_builtin_tools()
        crop = tmp_path / "tools" / BUILTIN_CROP_FILENAME
        assert crop.read_text(encoding="utf-8") == CROP_BY_BBOX_SCRIPT

    def test_ensure_builtin_tools_never_overwrites(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        crop = tmp_path / "tools" / BUILTIN_CROP_FILENAME
        crop.write_text("# user edited\n", encoding="utf-8")
        repo.ensure_builtin_tools()
        assert crop.read_text(encoding="utf-8") == "# user edited\n"

    def test_list_tools_sorted_case_insensitive_py_only(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        (tmp_path / "tools" / "B.py").write_text("b", encoding="utf-8")
        (tmp_path / "tools" / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "tools" / "C.py").write_text("c", encoding="utf-8")
        (tmp_path / "tools" / "notes.txt").write_text("x", encoding="utf-8")
        assert [p.name for p in repo.list_tools()] == ["a.py", "B.py", "C.py"]

    def test_create_tool_writes_default_script(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        path = repo.create_tool("demo")
        assert path == tmp_path / "tools" / "demo.py"
        assert path.read_text(encoding="utf-8") == DEFAULT_SCRIPT

    def test_create_tool_custom_content(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        path = repo.create_tool("demo", content="print('custom')\n")
        assert path.read_text(encoding="utf-8") == "print('custom')\n"

    def test_create_tool_invalid_name_returns_none_writes_nothing(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        assert repo.create_tool("...") is None
        assert list((tmp_path / "tools").iterdir()) == []

    def test_create_tool_existing_returned_untouched(self, tmp_path):
        # Pins the conflict behavior: an existing tool is never overwritten.
        repo = ToolRepository(tmp_path / "tools")
        existing = tmp_path / "tools" / "demo.py"
        existing.write_text("print('keep me')\n", encoding="utf-8")
        path = repo.create_tool("demo")
        assert path == existing
        assert existing.read_text(encoding="utf-8") == "print('keep me')\n"

    def test_find_tool_hit_and_miss(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        (tmp_path / "tools" / "demo.py").write_text("x", encoding="utf-8")
        assert repo.find_tool("demo") == tmp_path / "tools" / "demo.py"
        assert repo.find_tool("demo.py") == tmp_path / "tools" / "demo.py"
        assert repo.find_tool("missing") is None

    def test_find_tool_invalid_name_is_none_not_tools_dir(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        assert repo.find_tool("") is None
        assert repo.find_tool("...") is None

    def test_save_and_load_tool_roundtrip(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        path = tmp_path / "tools" / "demo.py"
        repo.save_tool(path, "print('saved')\n")
        assert repo.load_tool(path) == "print('saved')\n"

    def test_save_tool_oserror_propagates(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        missing_parent = tmp_path / "tools" / "no-such-dir" / "demo.py"
        with pytest.raises(OSError):
            repo.save_tool(missing_parent, "x")

    def test_load_tool_oserror_propagates(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        with pytest.raises(OSError):
            repo.load_tool(tmp_path / "tools" / "missing.py")


class TestMigrateLegacy:
    def test_none_app_config_is_noop(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        assert repo.migrate_legacy(None, tmp_path / "config.json") is False

    def test_empty_script_tools_is_noop(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        cfg = AppConfig()
        assert repo.migrate_legacy(cfg, tmp_path / "config.json") is False
        assert not (tmp_path / "config.json").exists()

    def test_migrates_tools_clears_config_and_saves(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        cfg = AppConfig(script_tools={"工具A": "print('a')\n", "b": "print('b')\n"})
        config_path = tmp_path / "config.json"

        assert repo.migrate_legacy(cfg, config_path) is True

        assert (tmp_path / "tools" / "工具A.py").read_text(encoding="utf-8") == "print('a')\n"
        assert (tmp_path / "tools" / "b.py").read_text(encoding="utf-8") == "print('b')\n"
        assert cfg.script_tools == {}
        assert config_path.exists()
        assert AppConfig.load(config_path).script_tools == {}

    def test_skips_non_str_values(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        cfg = AppConfig(script_tools={"bad": 123})
        assert repo.migrate_legacy(cfg, tmp_path / "config.json") is False
        assert cfg.script_tools == {"bad": 123}
        assert list((tmp_path / "tools").iterdir()) == []
        assert not (tmp_path / "config.json").exists()

    def test_skips_names_that_sanitize_to_empty(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        cfg = AppConfig(script_tools={"...": "print('x')\n"})
        assert repo.migrate_legacy(cfg, tmp_path / "config.json") is False
        assert cfg.script_tools == {"...": "print('x')\n"}
        assert list((tmp_path / "tools").iterdir()) == []

    def test_skips_existing_files_and_does_not_clear_when_nothing_migrated(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        existing = tmp_path / "tools" / "demo.py"
        existing.write_text("print('on disk')\n", encoding="utf-8")
        cfg = AppConfig(script_tools={"demo": "print('legacy')\n"})

        assert repo.migrate_legacy(cfg, tmp_path / "config.json") is False

        assert existing.read_text(encoding="utf-8") == "print('on disk')\n"
        assert cfg.script_tools == {"demo": "print('legacy')\n"}
        assert not (tmp_path / "config.json").exists()

    def test_partial_migration_still_clears_and_saves(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        existing = tmp_path / "tools" / "old.py"
        existing.write_text("print('on disk')\n", encoding="utf-8")
        cfg = AppConfig(script_tools={"old": "print('legacy')\n", "new": "print('new')\n"})
        config_path = tmp_path / "config.json"

        assert repo.migrate_legacy(cfg, config_path) is True

        assert existing.read_text(encoding="utf-8") == "print('on disk')\n"
        assert (tmp_path / "tools" / "new.py").read_text(encoding="utf-8") == "print('new')\n"
        assert cfg.script_tools == {}
        assert config_path.exists()

    def test_config_path_none_clears_without_saving(self, tmp_path):
        repo = ToolRepository(tmp_path / "tools")
        cfg = AppConfig(script_tools={"demo": "print('x')\n"})

        assert repo.migrate_legacy(cfg, None) is True

        assert cfg.script_tools == {}
        assert (tmp_path / "tools" / "demo.py").exists()
        # nothing written anywhere else in tmp_path
        assert [p.name for p in tmp_path.iterdir()] == ["tools"]
