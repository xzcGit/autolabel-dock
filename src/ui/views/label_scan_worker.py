"""Background project-label scan worker.

Opening a project needs every image's label metadata (status / classes /
user tags / stats) but each record is a small JSON on disk — scanning
thousands of them synchronously on the GUI thread froze the window on
project open. Both task views now populate their lists immediately with
paths only, then hand the label JSON reads to this worker and apply the
result dicts once the scan completes.

The worker reads via the injected ``read_record`` callback. Views pass
``LabelStore.load_unflushed`` — the flush callback touches Qt widgets and
must never run here; project open already flushed on the GUI thread
(``LabelPanel.set_project``) before the view is built.
Importing ``label_io`` here would trip the direct-IO guard
(``tests/core/test_label_io_guard.py``).
"""
from __future__ import annotations

import logging
from typing import Callable
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.project import ProjectManager

logger = logging.getLogger(__name__)


class LabelScanWorker(QThread):
    """Scan every label record of one project off the GUI thread.

    Emits ``scan_done`` once with ``(project, statuses, classes, tags,
    records)`` — plain dicts keyed by ``str(image_path)``; one signal so
    the receiving view applies a single consistent snapshot and mutating
    signals stay in the correct slot order.
    """

    # NOT named ``finished`` — that would shadow QThread's built-in signal.
    scan_done = pyqtSignal(object, dict, dict, dict, dict)

    def __init__(
        self,
        project: ProjectManager,
        read_record: Callable[[Path], object],
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._read_record = read_record
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self._scan_impl()
        if self._stop:
            return
        logger.info("Label scan done: %s (%d records)", self._project.config.name, len(self._records))
        self.scan_done.emit(self._project, self._statuses, self._classes, self._tags, self._records)

    def _scan_impl(self) -> None:
        (_, self._statuses, self._classes, self._tags,
         self._records) = scan_labels(self._project, self._read_record, lambda: self._stop)


def scan_labels(
    project: ProjectManager,
    read_record: Callable[[Path], object],
    should_stop: Callable[[], bool] = lambda: False,
) -> tuple:
    """Scan every label record; returns ``(project, statuses, classes, tags,
    records)``. Plain function so tests can scan inline without a QThread."""
    statuses: dict[str, str] = {}
    classes: dict[str, set] = {}
    tags: dict[str, set] = {}
    records: dict[str, object] = {}
    for img in project.list_images():
        if should_stop():
            break
        try:
            ia = read_record(project.label_path_for(img))
        except Exception as e:  # one corrupt record must not kill the scan
            logger.warning("Label scan skipped %s: %s", img, e)
            continue
        if ia is None:
            continue
        key = str(img)
        statuses[key] = ia.status
        classes[key] = {a.class_name for a in ia.annotations}
        tags[key] = set(ia.tags)
        records[key] = ia
    return project, statuses, classes, tags, records
