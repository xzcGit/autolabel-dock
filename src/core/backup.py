"""Project backup and restore — automatic snapshots of project.json and labels."""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_BACKUPS = 20


class BackupManager:
    """Manages automatic backups of project config and label files.

    Backups are stored in ``<project_dir>/.backups/`` with timestamped
    subdirectories.
    """

    def __init__(self, project_dir: Path | str, max_backups: int = _MAX_BACKUPS):
        self._project_dir = Path(project_dir)
        self._backup_dir = self._project_dir / ".backups"
        self._max_backups = max_backups

    @property
    def backup_dir(self) -> Path:
        return self._backup_dir

    def create_backup(self, label_dir: str = "labels") -> Path | None:
        """Create a timestamped backup of project.json and all label files.

        Returns the backup directory path, or None if nothing to back up.
        """
        project_json = self._project_dir / "project.json"
        labels_dir = self._project_dir / label_dir

        if not project_json.exists():
            return None

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = self._backup_dir / ts
        dest.mkdir(parents=True, exist_ok=True)

        # Back up project.json
        shutil.copy2(project_json, dest / "project.json")

        # Back up label files
        if labels_dir.exists():
            dest_labels = dest / label_dir
            dest_labels.mkdir(exist_ok=True)
            for f in labels_dir.glob("*.json"):
                shutil.copy2(f, dest_labels / f.name)

        self._prune_old_backups()
        logger.info("Backup created: %s", dest.name)
        return dest

    def list_backups(self) -> list[dict]:
        """List all available backups, newest first.

        Returns list of dicts with keys: name, path, timestamp, label_count.
        """
        if not self._backup_dir.exists():
            return []
        backups = []
        for d in sorted(self._backup_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            pj = d / "project.json"
            if not pj.exists():
                continue
            labels_dir = d / "labels"
            label_count = len(list(labels_dir.glob("*.json"))) if labels_dir.exists() else 0
            backups.append({
                "name": d.name,
                "path": d,
                "timestamp": d.name,  # YYYYMMDD-HHMMSS
                "label_count": label_count,
            })
        return backups

    def restore_backup(self, backup_name: str, label_dir: str = "labels") -> bool:
        """Restore project.json and labels from a named backup.

        A safety backup of the current state is created first (with
        ``pre-restore-`` prefix) so the user can undo the restore.
        """
        src = self._backup_dir / backup_name
        if not src.is_dir() or not (src / "project.json").exists():
            logger.warning("Backup not found: %s", backup_name)
            return False

        # Create a safety backup with unique prefix before overwriting
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safety = self._backup_dir / f"pre-restore-{ts}"
        safety.mkdir(parents=True, exist_ok=True)
        pj = self._project_dir / "project.json"
        if pj.exists():
            shutil.copy2(pj, safety / "project.json")
        cur_labels = self._project_dir / label_dir
        if cur_labels.exists():
            safety_labels = safety / label_dir
            safety_labels.mkdir(exist_ok=True)
            for f in cur_labels.glob("*.json"):
                shutil.copy2(f, safety_labels / f.name)

        # Restore project.json
        shutil.copy2(src / "project.json", self._project_dir / "project.json")

        # Restore labels
        src_labels = src / label_dir
        if src_labels.exists():
            dest_labels = self._project_dir / label_dir
            dest_labels.mkdir(exist_ok=True)
            # Clear current labels
            for f in dest_labels.glob("*.json"):
                f.unlink()
            # Copy backup labels
            for f in src_labels.glob("*.json"):
                shutil.copy2(f, dest_labels / f.name)

        logger.info("Restored backup: %s", backup_name)
        return True

    def _prune_old_backups(self) -> None:
        """Remove oldest backups to stay within max_backups limit."""
        if not self._backup_dir.exists():
            return
        dirs = sorted(
            (d for d in self._backup_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
        )
        while len(dirs) > self._max_backups:
            oldest = dirs.pop(0)
            shutil.rmtree(oldest)
            logger.debug("Pruned old backup: %s", oldest.name)
