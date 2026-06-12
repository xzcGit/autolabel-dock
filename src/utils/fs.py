"""Filesystem helpers."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def link_or_copy(src: Path, dst: Path) -> str:
    """Materialize *dst* pointing at *src* using the cheapest viable method.

    Tries in order:
      1. symbolic link (Linux/Mac default; Windows requires developer mode)
      2. hard link (works on Windows NTFS without privilege; same volume only)
      3. file copy (last-resort, always works)

    Returns the strategy used: ``"symlink"``, ``"hardlink"``, or ``"copy"``.
    Caller is responsible for ensuring *dst* does not already exist.
    """
    src_resolved = Path(src).resolve()
    dst = Path(dst)

    try:
        dst.symlink_to(src_resolved)
        return "symlink"
    except OSError as e:
        logger.debug("symlink failed (%s); trying hardlink", e)

    try:
        os.link(src_resolved, dst)
        return "hardlink"
    except OSError as e:
        logger.debug("hardlink failed (%s); falling back to copy", e)

    shutil.copy2(src_resolved, dst)
    return "copy"
