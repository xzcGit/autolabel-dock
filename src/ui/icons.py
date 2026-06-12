"""Built-in SVG icon provider — no external icon files needed."""
from __future__ import annotations

from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtCore import QByteArray, QRectF, Qt

# SVG templates (Lucide-inspired, 24x24 viewBox)
_SVGS: dict[str, str] = {
    # Tool modes
    "cursor": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M5 3l14 14-5 0-4 6-2-8-6-2z"/></svg>'
    ),
    "bbox": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="18" height="18" rx="2"/></svg>'
    ),
    "keypoint": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="3"/>'
        '<line x1="12" y1="2" x2="12" y2="9"/>'
        '<line x1="12" y1="15" x2="12" y2="22"/>'
        '<line x1="2" y1="12" x2="9" y2="12"/>'
        '<line x1="15" y1="12" x2="22" y2="12"/></svg>'
    ),
    # Actions
    "check_all": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M18 6L7 17l-4-4"/><path d="M22 10L11 21"/></svg>'
    ),
    "auto_label": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="4"/>'
        '<line x1="12" y1="2" x2="12" y2="6"/>'
        '<line x1="12" y1="18" x2="12" y2="22"/>'
        '<line x1="2" y1="12" x2="6" y2="12"/>'
        '<line x1="18" y1="12" x2="22" y2="12"/></svg>'
    ),
    "batch": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="2" y="7" width="15" height="14" rx="2"/>'
        '<path d="M7 7V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-2"/></svg>'
    ),
    # File menu
    "new_project": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="12" y1="5" x2="12" y2="19"/>'
        '<line x1="5" y1="12" x2="19" y2="12"/></svg>'
    ),
    "open_project": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9'
        'a2 2 0 0 1 2 2z"/></svg>'
    ),
    "export": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/></svg>'
    ),
    "exit": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>'
        '<polyline points="16 17 21 12 16 7"/>'
        '<line x1="21" y1="12" x2="9" y2="12"/></svg>'
    ),
    "classes": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="8" y1="6" x2="21" y2="6"/>'
        '<line x1="8" y1="12" x2="21" y2="12"/>'
        '<line x1="8" y1="18" x2="21" y2="18"/>'
        '<circle cx="4" cy="6" r="1" fill="{color}"/>'
        '<circle cx="4" cy="12" r="1" fill="{color}"/>'
        '<circle cx="4" cy="18" r="1" fill="{color}"/></svg>'
    ),
    # Model panel
    "load_model": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="16 16 12 12 8 16"/>'
        '<line x1="12" y1="12" x2="12" y2="21"/>'
        '<path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>'
    ),
    "delete": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="3 6 5 6 21 6"/>'
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4'
        'a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'
    ),
    "import": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/></svg>'
    ),
    # Train panel
    "start": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="5 3 19 12 5 21 5 3"/></svg>'
    ),
    "stop": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="4" y="4" width="16" height="16" rx="2"/></svg>'
    ),
    # Undo / Redo
    "undo": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="1 4 1 10 7 10"/>'
        '<path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>'
    ),
    "redo": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="23 4 23 10 17 10"/>'
        '<path d="M20.49 15a9 9 0 1 1-2.13-9.36L23 10"/></svg>'
    ),
    # Save
    "save": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11'
        'a2 2 0 0 1-2 2z"/>'
        '<polyline points="17 21 17 13 7 13 7 21"/>'
        '<polyline points="7 3 7 8 15 8"/></svg>'
    ),
    # Refresh
    "refresh": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="23 4 23 10 17 10"/>'
        '<polyline points="1 20 1 14 7 14"/>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"/>'
        '<path d="M20.49 15a9 9 0 0 1-14.85 3.36L1 14"/></svg>'
    ),
    # Tabs
    "welcome": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<polyline points="9 22 9 12 15 12 15 22"/></svg>'
    ),
    "label_tab": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<line x1="9" y1="3" x2="9" y2="21"/></svg>'
    ),
    "train_tab": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
    ),
    "model_tab": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4'
        'A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4'
        'A2 2 0 0 0 21 16z"/>'
        '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/>'
        '<line x1="12" y1="22.08" x2="12" y2="12"/></svg>'
    ),
    "script_tab": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="16 18 22 12 16 6"/>'
        '<polyline points="8 6 2 12 8 18"/>'
        '<line x1="14" y1="4" x2="10" y2="20"/></svg>'
    ),
    # Zoom
    "zoom_in": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        '<line x1="11" y1="8" x2="11" y2="14"/>'
        '<line x1="8" y1="11" x2="14" y2="11"/></svg>'
    ),
    "zoom_out": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        '<line x1="8" y1="11" x2="14" y2="11"/></svg>'
    ),
    "zoom_fit": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3'
        'm0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>'
    ),
    # Batch visible operations
    "confirm_visible": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M16 6L9 17l-4-4"/>'
        '<rect x="1" y="1" width="22" height="22" rx="3" stroke-dasharray="4 2"/></svg>'
    ),
    "revert_visible": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/>'
        '<rect x="1" y="1" width="22" height="22" rx="3" stroke-dasharray="4 2"/></svg>'
    ),
    # Cancel
    "cancel": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/></svg>'
    ),
    "eye": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/>'
        '<circle cx="12" cy="12" r="3"/></svg>'
    ),
}

_icon_cache: dict[tuple[str, str, int], QIcon] = {}


def icon(name: str, color: str = "#d8dee9", size: int = 20) -> QIcon:
    """Return a cached QIcon rendered from an inline SVG."""
    key = (name, color, size)
    if key in _icon_cache:
        return _icon_cache[key]

    svg_template = _SVGS.get(name)
    if svg_template is None:
        return QIcon()

    svg_data = svg_template.replace("{color}", color).encode("utf-8")
    renderer = QSvgRenderer(QByteArray(svg_data))

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()

    qi = QIcon(pixmap)
    _icon_cache[key] = qi
    return qi
