# -*- coding: utf-8 -*-
"""Toolbar chrome shared by the workbench: the Mode switch and themed icons."""

from functools import lru_cache

from PyQt5 import QtCore, QtGui, QtWidgets

from IVAS.icon_set import themed_icon
from IVAS.theme import active_theme, theme_hex


# QAction attribute name on MainWindow -> icon_set glyph.
ACTION_ICONS = {
    "actionImages_dir": "folder-image",
    "actionLabel_dir": "folder-label",
    "actionSave": "save",
    "actionExport": "export",
    "actionImport": "import",
    "actionExit": "power",
    "actionSetting": "settings",
    "actionPrev_image": "chevron-left",
    "actionNext_image": "chevron-right",
    "actionPrev_group": "chevrons-left",
    "actionNext_group": "chevrons-right",
    "actionPolygon": "polygon",
    "actionBBox": "bbox",
    "actionOBB": "obb",
    "actionSegment_anything_point": "sam-point",
    "actionSegment_anything_box": "sam-box",
    "actionRepaint": "repaint",
    "actionBackspace": "backspace",
    "actionFinish": "check-circle",
    "actionCancel": "x-circle",
    "actionUndo": "undo",
    "actionRedo": "redo",
    "actionEdit": "edit",
    "actionCopy": "copy",
    "actionDelete": "trash",
    "actionTo_top": "to-top",
    "actionTo_bottom": "to-bottom",
    "actionUnion": "union",
    "actionSubtract": "subtract",
    "actionIntersect": "intersect",
    "actionExclude": "exclude",
    "actionZoom_in": "zoom-in",
    "actionZoom_out": "zoom-out",
    "actionFit_window": "fit",
    "actionBit_map": "layers",
    "actionVisible": "eye",
    "actionScene_shot": "camera",
    "actionWindow_shot": "image",
    "actionModel_manage": "chip",
    "actionRemote_SAM": "cloud",
    "actionPlugins": "puzzle",
    "actionShortcut": "keyboard",
    "actionLanguage": "globe",
    "actionDocs": "book",
    "actionAbout": "info",
    "actionConverter": "convert",
    "actionVideo_to_frames": "film",
    "actionAuto_segment_with_bounding_box": "wand",
    "actionAnno_validator": "shield-check",
    "actionProcess_EXIF_tag": "rotate",
    "actionCreate_a_desktop_shortcut": "link",
    "actionReset_layout": "layout",
    "actionOpen_video_workspace": "video",
    "actionOpen_video_project_manifest": "folder-video",
    "actionVideo_output_root": "folder-video",
    "actionVideo_relink": "link",
    "actionAnnotate_current_video_frame": "image",
    "actionReturn_to_video": "video",
    "actionShow_video_workspace": "video",
    "actionShow_image_workspace": "image",
    "actionTemporal_mark_start": "mark-start",
    "actionTemporal_mark_end": "mark-end",
    "actionTemporal_discard_pending": "x-circle",
    "actionTemporal_add_point": "keyframe",
    "actionTemporal_capture_keyframe": "camera",
    "actionTemporal_add_rating": "star",
    "actionTemporal_templates": "list",
    "actionTemporal_import_templates": "import",
    "actionTemporal_edit": "edit",
    "actionTemporal_delete": "trash",
    "actionTemporal_undo": "undo",
    "actionTemporal_redo": "redo",
    "actionVideo_seek_backward_5s": "chevrons-left",
    "actionVideo_seek_backward_1s": "chevron-left",
    "actionVideo_seek_forward_1s": "chevron-right",
    "actionVideo_seek_forward_5s": "chevrons-right",
    "actionTimeline_zoom_in": "zoom-in",
    "actionTimeline_zoom_out": "zoom-out",
    "actionTimeline_reset_zoom": "fit",
    "actionTemporal_validate": "shield-check",
    "actionTemporal_import": "import",
    "actionTemporal_export": "export",
    "actionVideo_rebuild_index": "rotate",
    "actionVideo_cancel_task": "x-circle",
    "actionTracking_start": "crosshair",
    "actionTracking_review": "check-circle",
    "actionTracking_rollback": "undo",
}


@lru_cache(maxsize=1024)
def _cached_icon(name, theme, normal, disabled, checked):
    return themed_icon(name, normal, disabled, checked)


def icon_for(name: str, *, on_primary: bool = False, filled: bool = False) -> QtGui.QIcon:
    """Return a glyph in the active theme's icon colours.

    ``on_primary`` is for controls whose checked state fills with the primary
    colour (the Mode switch); ``filled`` is for controls that are always
    primary-filled (the Export button).  Other checked controls sit on the
    selection colour and use its foreground.
    """

    normal = theme_hex("on-primary") if filled else theme_hex("icon")
    checked = theme_hex("on-primary") if (on_primary or filled) else theme_hex("on-selection")
    return _cached_icon(
        name,
        active_theme(),
        normal,
        theme_hex("disabled-text"),
        checked,
    )


class ModeSegment(QtWidgets.QFrame):
    """Two-segment ``Video | Frame`` switch bound to the workspace actions.

    The actions stay the single source of truth: a click triggers the action
    (Frame within Video first extracts the playhead frame, which may fail or
    be cancelled) and the buttons then mirror the actions' real state.
    """

    def __init__(self, video_action, frame_action, parent=None):
        super().__init__(parent)
        self.setObjectName("modeSegment")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self._mode = "Frame"
        self._buttons = {}
        for key, action, glyph in (
            ("Video", video_action, "video"),
            ("Frame", frame_action, "image"),
        ):
            button = QtWidgets.QToolButton(self)
            button.setObjectName("modeSegment{}".format(key))
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            button.setFocusPolicy(QtCore.Qt.StrongFocus)
            button.setIconSize(QtCore.QSize(16, 16))
            button.setText(key)
            button.clicked.connect(
                lambda _checked=False, action=action: self._activate(action)
            )
            action.changed.connect(self.sync)
            layout.addWidget(button)
            self._buttons[key] = (button, action, glyph)
        self.apply_icons()
        self.sync()

    def button(self, key):
        return self._buttons[key][0]

    def _activate(self, action):
        if action.isEnabled() and not action.isChecked():
            action.trigger()
        self.sync()

    def sync(self):
        for button, action, _glyph in self._buttons.values():
            button.setEnabled(action.isEnabled())
            button.setChecked(action.isChecked())
            button.setToolTip(action.toolTip() or action.text())

    def apply_icons(self):
        for button, _action, glyph in self._buttons.values():
            button.setIcon(icon_for(glyph, on_primary=True))

    def set_labels(self, video_text, frame_text):
        self._buttons["Video"][0].setText(video_text)
        self._buttons["Frame"][0].setText(frame_text)
        self._fix_size()

    def _fix_size(self):
        # Toolbars squeeze embedded widgets; the switch must never elide.
        for button, _action, _glyph in self._buttons.values():
            button.setMinimumWidth(button.sizeHint().width())
        self.setMinimumWidth(self.sizeHint().width())

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self._fix_size()

    def set_mode(self, mode):
        self._mode = mode
        self.setAccessibleName("Mode: {}".format(mode))
        self.sync()

    def mode(self):
        return self._mode

    def text(self):
        """Compatibility with the former ``Mode: …`` dropdown button."""
        return "Mode: {}".format(self._mode)


def apply_action_icons(owner) -> None:
    """Set themed icons on every known action of ``owner`` (the main window)."""

    for attribute, glyph in ACTION_ICONS.items():
        action = getattr(owner, attribute, None)
        if isinstance(action, QtWidgets.QAction):
            action.setIcon(icon_for(glyph))
