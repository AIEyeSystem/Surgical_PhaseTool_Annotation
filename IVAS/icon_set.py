# -*- coding: utf-8 -*-
"""Theme-aware line icons for the workbench chrome.

Every glyph is a 24×24 stroke drawing with round caps.  ``C`` inside a body is
replaced by the requested colour so filled details (vertex dots, handles)
follow the same theme role as the outline.  Icons are rendered on demand per
theme, which keeps toolbar, menu and transport glyphs visually consistent and
avoids mixing coloured raster assets with monochrome chrome.
"""

from functools import lru_cache

from PyQt5 import QtCore, QtGui, QtSvg

_STROKE_WIDTH = 1.75

ICON_BODIES = {
    # files and projects
    "folder-image": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"/>'
                    '<path d="M7 16l3-3 2 2 2-2.5 3 3.5"/>',
    "folder-label": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"/>'
                    '<path d="M8 12h8M8 15.5h5"/>',
    "folder-video": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"/>'
                    '<path d="M10.5 10.5v5l4-2.5z" fill="C"/>',
    "save": '<path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z"/><path d="M8 4v4.5h7V4"/><path d="M8 20v-6h8v6"/>',
    "export": '<path d="M12 15V3.5"/><path d="M7.5 8L12 3.5 16.5 8"/><path d="M4 13.5V19a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5.5"/>',
    "import": '<path d="M12 3.5V15"/><path d="M7.5 10.5L12 15l4.5-4.5"/><path d="M4 13.5V19a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5.5"/>',
    "convert": '<path d="M4 8h14"/><path d="M14.5 4.5L18 8l-3.5 3.5"/><path d="M20 16H6"/><path d="M9.5 12.5L6 16l3.5 3.5"/>',
    "link": '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1.4 1.4"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1.4-1.4"/>',
    "power": '<path d="M12 3.5v8"/><path d="M6.6 7a7.5 7.5 0 1 0 10.8 0"/>',
    "files": '<path d="M14 3.5H6.5a1 1 0 0 0-1 1v15a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1V8z"/><path d="M14 3.5V8h4.5"/>',
    "image": '<rect x="3" y="4.5" width="18" height="15" rx="2"/><circle cx="8.5" cy="9.5" r="1.6"/><path d="M21 15.5l-5-5L7 19.5"/>',
    "video": '<rect x="3" y="6" width="12.5" height="12" rx="2"/><path d="M15.5 10.5L21 7.5v9l-5.5-3"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7.5 4v16M16.5 4v16M3 9h4.5M3 15h4.5M16.5 9H21M16.5 15H21"/>',
    # navigation
    "chevron-left": '<path d="M15 5.5L8.5 12l6.5 6.5"/>',
    "chevron-down": '<path d="M6 9.5l6 6 6-6"/>',
    "chevron-up": '<path d="M6 14.5l6-6 6 6"/>',
    "chevron-right": '<path d="M9 5.5l6.5 6.5L9 18.5"/>',
    "chevrons-left": '<path d="M11.5 6.5L6 12l5.5 5.5"/><path d="M18 6.5L12.5 12l5.5 5.5"/>',
    "chevrons-right": '<path d="M6 6.5l5.5 5.5L6 17.5"/><path d="M12.5 6.5L18 12l-5.5 5.5"/>',
    # drawing tools
    "polygon": '<path d="M5.5 8.5L11 4.5l8 4.5-2 10.5H7.5z"/>'
               '<circle cx="5.5" cy="8.5" r="1.9" fill="C" stroke="none"/><circle cx="11" cy="4.5" r="1.9" fill="C" stroke="none"/>'
               '<circle cx="19" cy="9" r="1.9" fill="C" stroke="none"/><circle cx="17" cy="19.5" r="1.9" fill="C" stroke="none"/>'
               '<circle cx="7.5" cy="19.5" r="1.9" fill="C" stroke="none"/>',
    "bbox": '<rect x="5" y="5" width="14" height="14"/>'
            '<rect x="3" y="3" width="4" height="4" fill="C" stroke="none"/><rect x="17" y="3" width="4" height="4" fill="C" stroke="none"/>'
            '<rect x="3" y="17" width="4" height="4" fill="C" stroke="none"/><rect x="17" y="17" width="4" height="4" fill="C" stroke="none"/>',
    "obb": '<path d="M8.5 3.5l12 7-5 8.5-12-7z"/><circle cx="8.5" cy="3.5" r="1.8" fill="C" stroke="none"/>'
           '<circle cx="15.5" cy="19" r="1.8" fill="C" stroke="none"/>',
    "sam-point": '<circle cx="10.5" cy="13.5" r="6.5"/><circle cx="10.5" cy="13.5" r="2.2" fill="C" stroke="none"/>'
                 '<path d="M19.5 2.5v5M17 5h5"/>',
    "sam-box": '<rect x="3" y="7" width="14" height="13.5" rx="1.5" stroke-dasharray="3.2 2.4"/>'
               '<path d="M19.5 2.5v5M17 5h5"/><circle cx="10" cy="13.75" r="1.8" fill="C" stroke="none"/>',
    "repaint": '<path d="M14 5l5 5-9.5 9.5H4.5v-5z"/><path d="M12 7l5 5"/><path d="M15 20.5h5.5"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "check-circle": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.3l2.7 2.7L16.2 9.5"/>',
    "x-circle": '<circle cx="12" cy="12" r="8.5"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "backspace": '<path d="M9 5h11a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H9l-6-7z"/><path d="M12 9.5l5 5M17 9.5l-5 5"/>',
    "undo": '<path d="M9 13.5L4.5 9 9 4.5"/><path d="M4.5 9H15a5 5 0 0 1 0 10h-4"/>',
    "redo": '<path d="M15 13.5L19.5 9 15 4.5"/><path d="M19.5 9H9a5 5 0 0 0 0 10h4"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
    "copy": '<rect x="8.5" y="8.5" width="12" height="12" rx="2"/><path d="M15.5 8.5V6a2 2 0 0 0-2-2h-7.5a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5"/>',
    "trash": '<path d="M4 7h16"/><path d="M9.5 7V4.5h5V7"/><path d="M6.5 7l1 13h9l1-13"/><path d="M10 11v5.5M14 11v5.5"/>',
    "to-top": '<path d="M5 3.5h14"/><path d="M12 20.5V8"/><path d="M7.5 12.5L12 8l4.5 4.5"/>',
    "to-bottom": '<path d="M5 20.5h14"/><path d="M12 3.5V16"/><path d="M7.5 11.5L12 16l4.5-4.5"/>',
    "union": '<path d="M4 4h10v6h6v10H10v-6H4z"/>',
    "subtract": '<rect x="4" y="4" width="10" height="10"/><path d="M14 10h6v10H10v-6" stroke-dasharray="2.2 2"/>',
    "intersect": '<rect x="4" y="4" width="10" height="10" stroke-dasharray="2.2 2"/><rect x="10" y="10" width="10" height="10" stroke-dasharray="2.2 2"/>'
                 '<rect x="10" y="10" width="4" height="4" fill="C"/>',
    "exclude": '<path d="M4 4h10v6h-4v4H4z" fill="C" fill-opacity="0.35"/><path d="M10 14h4v-4h6v10H10z" fill="C" fill-opacity="0.35"/>'
               '<rect x="4" y="4" width="10" height="10"/><rect x="10" y="10" width="10" height="10"/>',
    # view
    "fit": '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/>'
           '<rect x="8.5" y="8.5" width="7" height="7" rx="1"/>',
    "zoom-in": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.6-4.6"/><path d="M10.5 7.5v6M7.5 10.5h6"/>',
    "zoom-out": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.6-4.6"/><path d="M7.5 10.5h6"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    "layers": '<path d="M12 3.5l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "camera": '<path d="M4 8h3l1.5-2.5h7L17 8h3a1 1 0 0 1 1 1v9.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
    "layout": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    "palette": '<path d="M12 3a9 9 0 1 0 0 18c1.2 0 1.8-.8 1.8-1.7 0-1.3-1-1.6-1-2.8 0-.9.7-1.5 1.6-1.5H17a4 4 0 0 0 4-4C21 6.6 17 3 12 3z"/>'
               '<circle cx="7.5" cy="11.5" r="1.3" fill="C" stroke="none"/><circle cx="9.5" cy="7.3" r="1.3" fill="C" stroke="none"/>'
               '<circle cx="14.5" cy="7.3" r="1.3" fill="C" stroke="none"/>',
    "grid": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/>'
            '<rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
    "more": '<circle cx="5.5" cy="12" r="1.7" fill="C" stroke="none"/><circle cx="12" cy="12" r="1.7" fill="C" stroke="none"/>'
            '<circle cx="18.5" cy="12" r="1.7" fill="C" stroke="none"/>',
    # models and tools
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="0.8"/>'
            '<path d="M9.5 3v3M14.5 3v3M9.5 18v3M14.5 18v3M3 9.5h3M3 14.5h3M18 9.5h3M18 14.5h3"/>',
    "cloud": '<path d="M7 18.5a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 17.9 9.6 4.5 4.5 0 0 1 17.5 18.5z"/>',
    "puzzle": '<path d="M5 8.5h3.5a2 2 0 1 1 4 0H16v3.5a2 2 0 1 1 0 4V19.5H5z"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7"/>'
                '<path d="M12 2.5V5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/>',
    "keyboard": '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6.5 10h.01M10 10h.01M14 10h.01M17.5 10h.01M7.5 14h9"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18 14 14 0 0 1 0-18z"/>',
    "book": '<path d="M5 5.5A2.5 2.5 0 0 1 7.5 3H19v15H7.5A2.5 2.5 0 0 0 5 20.5z"/><path d="M5 20.5A2.5 2.5 0 0 0 7.5 21H19"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5"/><path d="M12 7.6h.01"/>',
    "wand": '<path d="M4 20L15 9"/><path d="M13.5 7.5l3 3"/><path d="M18 3v3M16.5 4.5h3M20 11v2M19 12h2M10 3.5v2M9 4.5h2"/>',
    "shield-check": '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M9 12l2.2 2.2L15.5 10"/>',
    "rotate": '<path d="M4 12a8 8 0 1 0 2.3-5.7"/><path d="M4 3.5V8h4.5"/>',
    "sparkles": '<path d="M11 3.5l1.8 4.7 4.7 1.8-4.7 1.8L11 16.5l-1.8-4.7L4.5 10l4.7-1.8z"/><path d="M18.5 15v5M16 17.5h5"/>',
    "text": '<path d="M5 6.5V5h14v1.5"/><path d="M12 5v14"/><path d="M9.5 19h5"/>',
    "tag": '<path d="M3.5 4.5v7l9 9 8-8-9-9h-7a1 1 0 0 0-1 1z"/><circle cx="8" cy="9" r="1.4" fill="C" stroke="none"/>',
    "list": '<path d="M9 6.5h11M9 12h11M9 17.5h11"/><circle cx="4.5" cy="6.5" r="1.3" fill="C" stroke="none"/>'
            '<circle cx="4.5" cy="12" r="1.3" fill="C" stroke="none"/><circle cx="4.5" cy="17.5" r="1.3" fill="C" stroke="none"/>',
    # video annotation and transport
    "mark-start": '<path d="M9 4H6v16h3"/><path d="M11.5 12h8.5"/><path d="M16.5 8.5L20 12l-3.5 3.5"/>',
    "mark-end": '<path d="M15 4h3v16h-3"/><path d="M4 12h8.5"/><path d="M9 8.5l3.5 3.5L9 15.5"/>',
    "keyframe": '<path d="M12 3.5l7 8.5-7 8.5-7-8.5z"/>',
    "star": '<path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"/>',
    "crosshair": '<circle cx="12" cy="12" r="8"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><circle cx="12" cy="12" r="1.6" fill="C" stroke="none"/>',
    "timeline": '<path d="M3 7.5h18M3 16.5h18"/><rect x="5" y="5" width="8" height="5" rx="1.2" fill="C" fill-opacity="0.3"/>'
                '<rect x="10" y="14" width="9" height="5" rx="1.2" fill="C" fill-opacity="0.3"/>',
    "play": '<path d="M7.5 4.5v15l12-7.5z" fill="C"/>',
    "pause": '<path d="M8 5v14M16 5v14" stroke-width="3"/>',
    "step-back": '<path d="M18 6l-8.5 6 8.5 6z" fill="C"/><path d="M6.5 6v12"/>',
    "step-forward": '<path d="M6 6l8.5 6L6 18z" fill="C"/><path d="M17.5 6v12"/>',
}


def available_icons():
    return tuple(ICON_BODIES)


def svg_markup(name: str, color: str, stroke_width: float = _STROKE_WIDTH) -> str:
    body = ICON_BODIES[name].replace('"C"', '"{}"'.format(color))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        'stroke-linejoin="round">{body}</svg>'
    ).format(color=color, width=stroke_width, body=body)


@lru_cache(maxsize=4096)
def _render(name: str, color: str, size: int, ratio: float) -> QtGui.QPixmap:
    pixels = max(1, int(round(size * ratio)))
    pixmap = QtGui.QPixmap(pixels, pixels)
    pixmap.fill(QtCore.Qt.transparent)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_markup(name, color).encode("utf-8")))
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def themed_icon(name: str, normal: str, disabled: str, checked: str = None,
                active: str = None) -> QtGui.QIcon:
    """Build a multi-state icon.  Colours are hex strings from theme roles."""

    icon = QtGui.QIcon()
    checked = checked or normal
    active = active or normal
    ratio = 1.0
    app = QtGui.QGuiApplication.instance()
    if app is not None:
        ratio = max(1.0, float(app.devicePixelRatio()))
    for size in (16, 18, 20, 22, 24, 32):
        for mode, colour in (
            (QtGui.QIcon.Normal, normal),
            (QtGui.QIcon.Active, active),
            (QtGui.QIcon.Selected, active),
            (QtGui.QIcon.Disabled, disabled),
        ):
            icon.addPixmap(_render(name, colour, size, ratio), mode, QtGui.QIcon.Off)
        icon.addPixmap(_render(name, checked, size, ratio), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon.addPixmap(_render(name, checked, size, ratio), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon.addPixmap(_render(name, disabled, size, ratio), QtGui.QIcon.Disabled, QtGui.QIcon.On)
    return icon


def write_svg(path, name: str, color: str) -> None:
    """Persist one glyph for stylesheet sub-controls (combo arrows, checks)."""

    with open(path, "w", encoding="utf-8") as stream:
        stream.write(svg_markup(name, color))
