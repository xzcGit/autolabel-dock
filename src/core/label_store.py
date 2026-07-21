"""LabelStore — the single read/write face for label records in session code.

The "flush before reading a label record" ordering constraint used to be a
convention spread across 8+ call sites in two layers (and it failed twice:
commit 8d769b4 and the tag-manager cascade). This module owns that invariant
structurally: every read through the store first invokes an injected flush
callback, so callers *cannot* forget it at the interface level.

Design (see docs/adr/0001-labelstore-flush-coordinator.md):
- The store is a stateless **flush coordinator**, NOT a source of truth.
  Pending edits stay in the views (canvas); the store holds no annotation
  state and no cache — do not add one.
- Qt-free by contract: session code (views/controllers) injects the flush
  callback (``LabelPanel.save_and_cleanup``); the store never imports UI.
- Writes also go through the store (thin delegation to ``label_io``) so the
  contract reads "all session label IO goes through the store". Saves do NOT
  trigger a flush — a save frequently *is* the flush.
- The flush callback must be idempotent and cheap when clean (scan loops call
  ``load`` once per image); flush-callback exceptions propagate — data
  integrity beats silent continuation.

Independent Qt-free consumers that run strictly after an explicit flush
(``engine/dataset.py``, ``core/formats/*``) keep using ``label_io`` directly
by design; a guard test pins the allowlist.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.core import label_io
from src.core.annotation import ImageAnnotation

logger = logging.getLogger(__name__)


class LabelStore:
    """Flush-first gateway for reading/writing label records.

    A read through the store always sees flushed truth: ``load`` /
    ``load_or_empty`` invoke the injected flush callback before touching
    disk. A reentrancy guard makes store calls issued from *inside* the
    flush callback skip re-flushing.
    """

    def __init__(self, flush_cb: Callable[[], None] | None = None):
        self._flush_cb = flush_cb
        self._flushing = False

    def set_flush_callback(self, cb: Callable[[], None] | None) -> None:
        """Install (or clear) the callback invoked before every read."""
        self._flush_cb = cb

    # ── Reads (flush-first) ───────────────────────────────────────

    def load(self, label_path: Path | str) -> ImageAnnotation | None:
        """Flush pending edits, then load the record.

        Semantics past the flush are identical to
        ``label_io.load_annotation`` — returns ``None`` on missing or
        corrupt files (never raises for those).
        """
        self._flush()
        return label_io.load_annotation(label_path)

    def load_or_empty(
        self,
        label_path: Path | str,
        image_name: str,
        image_size: tuple[int, int] | None = None,
    ) -> ImageAnnotation:
        """Flush, then load the record — fabricating an empty one if absent.

        ``image_size=None`` fabricates with a ``(1, 1)`` placeholder and is
        reserved for read-only display paths. Callers that will save the
        record back MUST pass the real size (``get_image_size`` is a Qt
        dependency and stays at the caller — the store never computes sizes).
        """
        ia = self.load(label_path)
        if ia is not None:
            return ia
        size = image_size if image_size is not None else (1, 1)
        return ImageAnnotation(image_path=image_name, image_size=size)

    # ── Writes (thin delegation, no flush) ────────────────────────

    def save(self, ia: ImageAnnotation, label_path: Path | str) -> None:
        """Persist a record. Thin delegate to ``label_io.save_annotation``,
        including its delete-empty-record semantics (an empty record removes
        the file instead of writing it)."""
        label_io.save_annotation(ia, label_path)

    # ── Internals ─────────────────────────────────────────────────

    def _flush(self) -> None:
        if self._flush_cb is None or self._flushing:
            return
        self._flushing = True
        try:
            self._flush_cb()
        finally:
            self._flushing = False
