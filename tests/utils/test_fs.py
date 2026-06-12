"""Tests for filesystem helpers (link_or_copy three-tier fallback)."""
from pathlib import Path
from unittest.mock import patch


class TestLinkOrCopy:
    def test_uses_symlink_when_supported(self, tmp_path):
        from src.utils.fs import link_or_copy

        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"

        mode = link_or_copy(src, dst)

        assert mode == "symlink"
        assert dst.is_symlink()
        assert dst.read_text() == "hello"

    def test_falls_back_to_hardlink_when_symlink_fails(self, tmp_path):
        from src.utils.fs import link_or_copy

        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"

        def fake_symlink(self, target, *args, **kwargs):
            raise OSError("[WinError 1314] privilege not held")

        with patch.object(Path, "symlink_to", fake_symlink):
            mode = link_or_copy(src, dst)

        assert mode == "hardlink"
        assert not dst.is_symlink()
        assert dst.exists()
        assert dst.read_text() == "hello"
        # Hard link shares inode with source
        assert dst.stat().st_ino == src.stat().st_ino

    def test_falls_back_to_copy_when_link_methods_fail(self, tmp_path):
        from src.utils.fs import link_or_copy

        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"

        def fail_symlink(self, target, *args, **kwargs):
            raise OSError("symlink not supported")

        def fail_hardlink(src_p, dst_p):
            raise OSError("[Errno 18] Invalid cross-device link")

        with patch.object(Path, "symlink_to", fail_symlink), \
             patch("os.link", fail_hardlink):
            mode = link_or_copy(src, dst)

        assert mode == "copy"
        assert not dst.is_symlink()
        assert dst.exists()
        assert dst.read_text() == "hello"
        # Copy creates a distinct inode
        assert dst.stat().st_ino != src.stat().st_ino
