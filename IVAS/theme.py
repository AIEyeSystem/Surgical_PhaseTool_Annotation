# -*- coding: utf-8 -*-
"""Semantic colour themes and the Qt stylesheet for the unified workbench."""

import os
import tempfile
from collections import OrderedDict
from types import MappingProxyType

from PyQt5 import QtGui, QtWidgets


_UI_FONT = '"Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif'
_MONO_FONT = '"Cascadia Code", "JetBrains Mono", Consolas, "Microsoft YaHei UI", monospace'

# Keep these palettes synchronized with DESIGN.md and theme-matrix.json.
# Feature widgets consume roles through theme_hex/theme_color, never raw hex.
THEMES = OrderedDict(
    (
        (
            "medical",
            {
                "zh": "医学蓝白",
                "en": "Medical Blue",
                "dark": False,
                "font": _UI_FONT,
                "roles": {
                    "canvas": "#0B1220",
                    "canvas-raised": "#16233A",
                    "canvas-text": "#C9D6EA",
                    "chrome": "#FFFFFF",
                    "surface": "#F3F7FC",
                    "surface-raised": "#FFFFFF",
                    "surface-subtle": "#E7EEF8",
                    "input": "#FFFFFF",
                    "hover": "#EAF2FE",
                    "pressed": "#D6E6FD",
                    "text": "#0F1E36",
                    "text-muted": "#50607A",
                    "icon": "#2D4C7C",
                    "border": "#7D8DA6",
                    "border-subtle": "#D5DEEA",
                    "primary": "#1463D8",
                    "primary-hover": "#0F55BE",
                    "primary-active": "#0B469F",
                    "on-primary": "#FFFFFF",
                    "danger": "#C42B22",
                    "danger-hover": "#A3231C",
                    "danger-active": "#831C16",
                    "on-danger": "#FFFFFF",
                    "focus": "#1463D8",
                    "selection": "#DCEAFE",
                    "on-selection": "#0A3D8F",
                    "disabled-bg": "#EEF2F7",
                    "disabled-text": "#97A3B5",
                    "timeline-phase": "#3B8BEB",
                    "timeline-task": "#F2A93B",
                    "success": "#1C7C54",
                    "warning": "#A15F00",
                },
            },
        ),
        (
            "geek",
            {
                "zh": "极客终端",
                "en": "Geek Terminal",
                "dark": True,
                "font": _MONO_FONT,
                "roles": {
                    "canvas": "#040607",
                    "canvas-raised": "#0D1316",
                    "canvas-text": "#8FD9B0",
                    "chrome": "#0A0E10",
                    "surface": "#0C1013",
                    "surface-raised": "#11171B",
                    "surface-subtle": "#162026",
                    "input": "#070A0C",
                    "hover": "#132A20",
                    "pressed": "#174A33",
                    "text": "#C8F7DC",
                    "text-muted": "#73A68A",
                    "icon": "#5FE3A1",
                    "border": "#3A6B52",
                    "border-subtle": "#1C2C25",
                    "primary": "#00E38C",
                    "primary-hover": "#3BF0A8",
                    "primary-active": "#00B870",
                    "on-primary": "#02160C",
                    "danger": "#FF5C7A",
                    "danger-hover": "#FF7A93",
                    "danger-active": "#E0405E",
                    "on-danger": "#1A0308",
                    "focus": "#00E38C",
                    "selection": "#0F3B29",
                    "on-selection": "#A6FFD4",
                    "disabled-bg": "#0E1417",
                    "disabled-text": "#3E5C4C",
                    "timeline-phase": "#00C2FF",
                    "timeline-task": "#FF9F1C",
                    "success": "#00E38C",
                    "warning": "#FFC857",
                },
            },
        ),
        (
            "graphite",
            {
                "zh": "午夜石墨",
                "en": "Midnight Graphite",
                "dark": True,
                "font": _UI_FONT,
                "roles": {
                    "canvas": "#0C0D10",
                    "canvas-raised": "#1A1C21",
                    "canvas-text": "#A9B0BC",
                    "chrome": "#191B1F",
                    "surface": "#1E2025",
                    "surface-raised": "#25282E",
                    "surface-subtle": "#2D3037",
                    "input": "#17191D",
                    "hover": "#30343C",
                    "pressed": "#3A3F49",
                    "text": "#E7E9EE",
                    "text-muted": "#9DA4B0",
                    "icon": "#C5CAD3",
                    "border": "#6E7684",
                    "border-subtle": "#353941",
                    "primary": "#2F6FDF",
                    "primary-hover": "#2A66CF",
                    "primary-active": "#2560C4",
                    "on-primary": "#FFFFFF",
                    "danger": "#FF6B6B",
                    "danger-hover": "#FF8787",
                    "danger-active": "#E05252",
                    "on-danger": "#1B0707",
                    "focus": "#6BA1FF",
                    "selection": "#2B3F66",
                    "on-selection": "#E1EAFF",
                    "disabled-bg": "#222429",
                    "disabled-text": "#5C626C",
                    "timeline-phase": "#4C8DFF",
                    "timeline-task": "#F5A524",
                    "success": "#4CC38A",
                    "warning": "#F5B841",
                },
            },
        ),
        (
            "surgical",
            {
                "zh": "手术室青",
                "en": "Surgical Teal",
                "dark": False,
                "font": _UI_FONT,
                "roles": {
                    "canvas": "#0F1A1C",
                    "canvas-raised": "#1B2B2E",
                    "canvas-text": "#C3D7D8",
                    "chrome": "#FBFDFD",
                    "surface": "#F2F7F7",
                    "surface-raised": "#FFFFFF",
                    "surface-subtle": "#E2EEEE",
                    "input": "#FFFFFF",
                    "hover": "#E3F3F3",
                    "pressed": "#CBE9EA",
                    "text": "#122426",
                    "text-muted": "#4D6467",
                    "icon": "#1F5A5F",
                    "border": "#76908F",
                    "border-subtle": "#CFE0E0",
                    "primary": "#08808B",
                    "primary-hover": "#066C76",
                    "primary-active": "#055A62",
                    "on-primary": "#FFFFFF",
                    "danger": "#B42318",
                    "danger-hover": "#8F1C14",
                    "danger-active": "#731610",
                    "on-danger": "#FFFFFF",
                    "focus": "#08808B",
                    "selection": "#CDECEE",
                    "on-selection": "#0B3E45",
                    "disabled-bg": "#EAF1F1",
                    "disabled-text": "#93A8A9",
                    "timeline-phase": "#21A6B3",
                    "timeline-task": "#F4A340",
                    "success": "#207A4B",
                    "warning": "#9A6700",
                },
            },
        ),
        (
            "aurora",
            {
                "zh": "北极光",
                "en": "Nordic Aurora",
                "dark": True,
                "font": _UI_FONT,
                "roles": {
                    "canvas": "#191D25",
                    "canvas-raised": "#262C38",
                    "canvas-text": "#C9D1DE",
                    "chrome": "#2A303C",
                    "surface": "#2E3440",
                    "surface-raised": "#353C4A",
                    "surface-subtle": "#3B4252",
                    "input": "#262B35",
                    "hover": "#434C5E",
                    "pressed": "#4C566A",
                    "text": "#ECEFF4",
                    "text-muted": "#AEB7C8",
                    "icon": "#D8DEE9",
                    "border": "#7A88A1",
                    "border-subtle": "#434C5E",
                    "primary": "#88C0D0",
                    "primary-hover": "#9FCDDA",
                    "primary-active": "#6FAEC0",
                    "on-primary": "#1D2330",
                    "danger": "#E07A84",
                    "danger-hover": "#E8939B",
                    "danger-active": "#BF616A",
                    "on-danger": "#1D1013",
                    "focus": "#8FBCBB",
                    "selection": "#3E5A6E",
                    "on-selection": "#EAF6FA",
                    "disabled-bg": "#323846",
                    "disabled-text": "#6B7488",
                    "timeline-phase": "#81A1C1",
                    "timeline-task": "#D08770",
                    "success": "#A3BE8C",
                    "warning": "#EBCB8B",
                },
            },
        ),
    )
)

DEFAULT_THEME = "medical"

_active_key = DEFAULT_THEME
_active_roles = dict(THEMES[DEFAULT_THEME]["roles"])
# A live read-only view: switching themes updates it in place, so painters
# that read roles at paint time (timeline, transport) follow the new theme.
THEME_TOKENS = MappingProxyType(_active_roles)


def available_themes():
    """Return ``(key, Chinese name, English name)`` for every bundled theme."""

    return tuple((key, theme["zh"], theme["en"]) for key, theme in THEMES.items())


def normalize_theme(key) -> str:
    return key if key in THEMES else DEFAULT_THEME


def active_theme() -> str:
    return _active_key


def theme_is_dark(key=None) -> bool:
    return bool(THEMES[normalize_theme(key or _active_key)]["dark"])


def theme_hex(role: str, key=None) -> str:
    """Return one declared semantic colour role of the active (or given) theme."""

    roles = _active_roles if key is None else THEMES[normalize_theme(key)]["roles"]
    try:
        return roles[role]
    except KeyError as error:
        raise KeyError("Unknown IVAS theme role: {}".format(role)) from error


def theme_color(role: str) -> QtGui.QColor:
    return QtGui.QColor(theme_hex(role))


def set_active_theme(key) -> str:
    global _active_key
    _active_key = normalize_theme(key)
    _active_roles.clear()
    _active_roles.update(THEMES[_active_key]["roles"])
    return _active_key


def _asset_dir(key: str) -> str:
    """Stylesheet sub-control glyphs, regenerated per theme in a cache folder."""

    from IVAS.icon_set import write_svg

    roles = THEMES[key]["roles"]
    directory = os.path.join(tempfile.gettempdir(), "ivas-theme", key)
    os.makedirs(directory, exist_ok=True)
    for name, glyph, role in (
        ("arrow-down", "chevron-down", "text-muted"),
        ("arrow-down-disabled", "chevron-down", "disabled-text"),
        ("arrow-up", "chevron-up", "text-muted"),
        ("check", "check", "on-primary"),
        ("menu-arrow", "chevron-right", "text-muted"),
    ):
        path = os.path.join(directory, name + ".svg")
        write_svg(path, glyph, roles[role])
    return directory.replace("\\", "/")


_STYLESHEET_TEMPLATE = r"""
QWidget {{
    color: {text};
    font-family: {font};
    font-size: 10pt;
}}

QMainWindow, QDialog, QWidget#mediaWorkspaceStack {{
    background: {surface};
}}

QMainWindow::separator {{
    background: {border-subtle};
    width: 1px;
    height: 1px;
}}

QMainWindow::separator:hover {{
    background: {primary};
}}

QGraphicsView#annotationCanvas, QFrame#videoAnnotationStage {{
    background: {canvas};
    border: 0;
}}

/* ---------- menus ---------- */
QMenuBar {{
    background: {chrome};
    border-bottom: 1px solid {border-subtle};
    padding: 2px 6px;
}}

QMenuBar::item {{
    background: transparent;
    border-radius: 6px;
    padding: 5px 10px;
}}

QMenuBar::item:selected, QMenuBar::item:pressed {{
    color: {text};
    background: {hover};
}}

QMenu {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border-subtle};
    border-radius: 8px;
    padding: 6px;
}}

QMenu::item {{
    border-radius: 6px;
    padding: 6px 30px 6px 10px;
    margin: 1px 0;
}}

QMenu::icon {{
    padding-left: 10px;
}}

QMenu::item:selected {{
    color: {on-selection};
    background: {selection};
}}

QMenu::item:disabled {{
    color: {disabled-text};
    background: transparent;
}}

QMenu::item:checked {{
    font-weight: 600;
}}

QMenu::right-arrow {{
    image: url({assets}/menu-arrow.svg);
    width: 12px;
    height: 12px;
    padding-right: 6px;
}}

QMenu::separator {{
    background: {border-subtle};
    height: 1px;
    margin: 5px 8px;
}}

QMenu#menuFile::separator {{
    background: transparent;
    border-top: 1px dashed {border-subtle};
    height: 1px;
    margin: 6px 8px;
}}

QLabel#fileMenuHeading {{
    color: {text-muted};
    font-size: 9pt;
    font-weight: 600;
    padding: 6px 12px 3px 12px;
}}

/* ---------- toolbars ---------- */
QToolBar {{
    background: {chrome};
    border: 0;
    border-bottom: 1px solid {border-subtle};
    spacing: 2px;
    padding: 4px 6px;
}}

QToolBar::separator {{
    background: {border-subtle};
    margin: 7px 6px;
    width: 1px;
    height: 1px;
}}

QToolBar QLabel#temporalToolbarLabel, QToolBar QLabel#toolbarSectionLabel {{
    color: {text-muted};
    font-weight: 600;
    padding: 0 4px 0 2px;
}}

QToolButton {{
    color: {text};
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    min-width: 28px;
    min-height: 28px;
    padding: 3px 5px;
}}

QToolButton:hover {{
    background: {hover};
    border-color: {border-subtle};
}}

QToolButton:pressed {{
    background: {pressed};
}}

QToolButton:checked {{
    color: {on-selection};
    background: {selection};
    border-color: {primary};
}}

QToolButton:focus {{
    border: 1px solid {focus};
}}

QToolButton:disabled {{
    color: {disabled-text};
    background: transparent;
    border-color: transparent;
}}

QToolButton[popupMode="2"] {{
    padding-right: 20px;
}}

QToolButton::menu-indicator {{
    image: url({assets}/arrow-down.svg);
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 11px;
    height: 11px;
    right: 6px;
}}

QToolButton::menu-indicator:disabled {{
    image: url({assets}/arrow-down-disabled.svg);
}}

QToolButton#primaryToolMenu {{
    font-weight: 600;
}}

QToolButton#modelStatusButton {{
    color: {text-muted};
    border: 1px solid {border-subtle};
    border-radius: 9px;
    min-width: 0;
    min-height: 0;
    padding: 1px 8px;
}}

QToolButton#modelStatusButton:hover {{
    color: {text};
    background: {hover};
}}

QToolButton#modelStatusButton[modelState="ready"] {{
    color: {text};
}}

QToolButton#modelStatusButton[modelState="loading"] {{
    color: {warning};
    border-color: {warning};
}}

QToolButton#modelStatusButton[modelState="failed"] {{
    color: {danger};
    border-color: {danger};
}}

QFrame#modeSegment {{
    background: {surface-subtle};
    border: 1px solid {border-subtle};
    border-radius: 9px;
}}

QFrame#modeSegment QToolButton {{
    color: {text-muted};
    border: 0;
    border-radius: 7px;
    min-width: 0;
    min-height: 24px;
    padding: 2px 8px;
    font-weight: 600;
}}

QFrame#modeSegment QToolButton:hover {{
    color: {text};
    background: {hover};
}}

QFrame#modeSegment QToolButton:checked {{
    color: {on-primary};
    background: {primary};
}}

QFrame#modeSegment QToolButton:disabled {{
    color: {disabled-text};
    background: transparent;
}}

QFrame#modeSegment QToolButton:checked:disabled {{
    color: {on-primary};
    background: {primary-active};
}}

/* ---------- docks and tabs ---------- */
QDockWidget {{
    color: {text-muted};
    font-weight: 600;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background: {chrome};
    border-bottom: 1px solid {border-subtle};
    padding: 6px 10px;
    text-align: left;
}}

QDockWidget > QWidget {{
    background: {surface-raised};
}}

QTabBar#workbenchDockTabs {{
    background: {chrome};
}}

QTabBar#workbenchDockTabs::tab {{
    color: {text-muted};
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 6px 10px;
    min-height: 22px;
}}

QTabBar#workbenchDockTabs::tab:hover {{
    color: {text};
    background: {hover};
}}

QTabBar#workbenchDockTabs::tab:selected {{
    color: {text};
    border-bottom: 2px solid {primary};
}}

QToolButton#workbenchDockTabScrollButton {{
    background: {chrome};
    border: 1px solid {border-subtle};
    border-radius: 0;
    width: 28px;
    min-width: 28px;
    padding: 0;
}}

QTabBar#workbenchDockTabs::scroller {{
    width: 56px;
}}

QTabWidget::pane {{
    border: 1px solid {border-subtle};
    border-radius: 6px;
    top: -1px;
}}

QTabBar::tab {{
    color: {text-muted};
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 6px 12px;
}}

QTabBar::tab:selected {{
    color: {text};
    border-bottom: 2px solid {primary};
}}

QGroupBox {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border-subtle};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {text-muted};
}}

/* ---------- inputs ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    color: {text};
    background: {input};
    border: 1px solid {border-subtle};
    border-radius: 6px;
    selection-color: {on-selection};
    selection-background-color: {selection};
    padding: 4px 8px;
}}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover,
QTextEdit:hover, QPlainTextEdit:hover {{
    border-color: {border};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {focus};
    padding: 3px 7px;
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled,
QTextEdit:disabled, QPlainTextEdit:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
    border-color: {border-subtle};
}}

QComboBox {{
    padding-right: 24px;
}}

QComboBox:focus {{
    padding-right: 23px;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: 0;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url({assets}/arrow-down.svg);
    width: 12px;
    height: 12px;
}}

QComboBox::down-arrow:disabled {{
    image: url({assets}/arrow-down-disabled.svg);
}}

QComboBox QAbstractItemView {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border-subtle};
    border-radius: 6px;
    padding: 4px;
    outline: 0;
    selection-color: {on-selection};
    selection-background-color: {selection};
}}

QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px;
    border: 0;
    background: transparent;
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({assets}/arrow-up.svg);
    width: 10px;
    height: 10px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({assets}/arrow-down.svg);
    width: 10px;
    height: 10px;
}}

QCheckBox, QRadioButton {{
    spacing: 7px;
    background: transparent;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {border};
    background: {input};
}}

QCheckBox::indicator {{
    border-radius: 4px;
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {primary};
}}

QCheckBox::indicator:checked {{
    background: {primary};
    border-color: {primary};
    image: url({assets}/check.svg);
}}

QRadioButton::indicator:checked {{
    border: 1px solid {primary};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {primary}, stop:0.45 {primary}, stop:0.55 {input}, stop:1 {input});
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {border-subtle};
    background: {disabled-bg};
}}

/* ---------- buttons ---------- */
QPushButton {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border-subtle};
    border-radius: 6px;
    min-height: 26px;
    padding: 4px 12px;
}}

QPushButton:hover {{
    background: {hover};
    border-color: {border};
}}

QPushButton:pressed, QPushButton:checked {{
    color: {on-selection};
    background: {selection};
    border-color: {primary};
}}

QPushButton:focus {{
    border: 2px solid {focus};
    padding: 3px 11px;
}}

QPushButton:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
    border-color: {disabled-bg};
}}

QPushButton#primaryAction {{
    color: {on-primary};
    background: {primary};
    border-color: {primary};
    font-weight: 600;
}}

QPushButton#primaryAction:hover {{
    background: {primary-hover};
    border-color: {primary-hover};
}}

QPushButton#primaryAction:pressed {{
    background: {primary-active};
    border-color: {primary-active};
}}

QPushButton#primaryAction:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
    border-color: {disabled-bg};
}}

QPushButton#dangerAction {{
    color: {on-danger};
    background: {danger};
    border-color: {danger};
}}

QPushButton#dangerAction:hover {{
    background: {danger-hover};
    border-color: {danger-hover};
}}

QPushButton#dangerAction:pressed {{
    background: {danger-active};
    border-color: {danger-active};
}}

/* ---------- item views ---------- */
QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView {{
    color: {text};
    background: {surface-raised};
    alternate-background-color: {surface};
    border: 1px solid {border-subtle};
    border-radius: 8px;
    outline: 0;
}}

QListWidget::item, QTreeWidget::item, QTableWidget::item {{
    min-height: 27px;
    padding: 3px 5px;
    border-radius: 5px;
}}

QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{
    background: {hover};
}}

QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    color: {on-selection};
    background: {selection};
}}

QHeaderView::section {{
    color: {text-muted};
    background: {surface-subtle};
    border: 0;
    border-right: 1px solid {border-subtle};
    border-bottom: 1px solid {border-subtle};
    padding: 6px;
    font-weight: 600;
}}

QTableCornerButton::section {{
    background: {surface-subtle};
    border: 0;
}}

/* ---------- sliders, scrollbars, progress ---------- */
QSlider::groove:horizontal {{
    background: {border-subtle};
    height: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {primary};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {surface-raised};
    border: 2px solid {primary};
    width: 12px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {primary};
}}

QSlider:focus::handle:horizontal {{
    border: 3px solid {focus};
}}

QSplitter::handle {{
    background: {border-subtle};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {border-subtle};
    border-radius: 3px;
    min-height: 28px;
}}

QScrollBar::handle:horizontal {{
    background: {border-subtle};
    border-radius: 3px;
    min-width: 28px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {border};
}}

QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: 0;
    width: 0;
    height: 0;
}}

QAbstractScrollArea::corner {{
    background: transparent;
}}

QProgressBar {{
    color: {text};
    background: {disabled-bg};
    border: 1px solid {border-subtle};
    border-radius: 5px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: {primary};
    border-radius: 4px;
}}

/* ---------- status ---------- */
QStatusBar {{
    color: {text-muted};
    background: {chrome};
    border-top: 1px solid {border-subtle};
}}

QStatusBar::item {{
    border: 0;
}}

QStatusBar QLabel {{
    padding: 2px 7px;
}}

QLabel#selectedCategory {{
    color: {on-selection};
    background: {selection};
    border: 1px solid {primary};
    border-radius: 9px;
    padding: 2px 10px;
    font-weight: 600;
}}

QLabel#selectedCategory[canvasState="create"] {{
    color: {on-primary};
    background: {success};
    border-color: {success};
}}

QLabel#selectedCategory[canvasState="edit"] {{
    color: {on-primary};
    background: {primary};
    border-color: {primary};
}}

QLabel#selectedCategory[canvasState="repaint"] {{
    color: {on-primary};
    background: {warning};
    border-color: {warning};
}}

QLabel#workspaceEyebrow, QLabel#workspaceMeta {{
    color: {text-muted};
}}

QToolTip {{
    color: {surface-raised};
    background: {text};
    border: 0;
    border-radius: 4px;
    padding: 5px 7px;
}}
"""


def build_stylesheet(key=None) -> str:
    key = normalize_theme(key or _active_key)
    values = dict(THEMES[key]["roles"])
    values["font"] = THEMES[key]["font"]
    values["assets"] = _asset_dir(key)
    return _STYLESHEET_TEMPLATE.format_map(values)


def _palette_for(key) -> QtGui.QPalette:
    roles = THEMES[key]["roles"]
    color = lambda role: QtGui.QColor(roles[role])
    palette = QtGui.QPalette()
    for group in (QtGui.QPalette.Active, QtGui.QPalette.Inactive):
        palette.setColor(group, QtGui.QPalette.Window, color("surface"))
        palette.setColor(group, QtGui.QPalette.WindowText, color("text"))
        palette.setColor(group, QtGui.QPalette.Base, color("input"))
        palette.setColor(group, QtGui.QPalette.AlternateBase, color("surface"))
        palette.setColor(group, QtGui.QPalette.Text, color("text"))
        palette.setColor(group, QtGui.QPalette.Button, color("surface-raised"))
        palette.setColor(group, QtGui.QPalette.ButtonText, color("text"))
        palette.setColor(group, QtGui.QPalette.Highlight, color("selection"))
        palette.setColor(group, QtGui.QPalette.HighlightedText, color("on-selection"))
        palette.setColor(group, QtGui.QPalette.ToolTipBase, color("text"))
        palette.setColor(group, QtGui.QPalette.ToolTipText, color("surface-raised"))
        palette.setColor(group, QtGui.QPalette.PlaceholderText, color("text-muted"))
        palette.setColor(group, QtGui.QPalette.Link, color("primary"))
        palette.setColor(group, QtGui.QPalette.BrightText, color("danger"))
        palette.setColor(group, QtGui.QPalette.Light, color("surface-raised"))
        palette.setColor(group, QtGui.QPalette.Midlight, color("surface-subtle"))
        palette.setColor(group, QtGui.QPalette.Mid, color("border-subtle"))
        palette.setColor(group, QtGui.QPalette.Dark, color("border"))
        palette.setColor(group, QtGui.QPalette.Shadow, color("canvas"))
    disabled = QtGui.QPalette.Disabled
    palette.setColor(disabled, QtGui.QPalette.Window, color("surface"))
    palette.setColor(disabled, QtGui.QPalette.Base, color("disabled-bg"))
    palette.setColor(disabled, QtGui.QPalette.Button, color("disabled-bg"))
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text,
                 QtGui.QPalette.ButtonText, QtGui.QPalette.PlaceholderText):
        palette.setColor(disabled, role, color("disabled-text"))
    palette.setColor(disabled, QtGui.QPalette.Highlight, color("disabled-bg"))
    palette.setColor(disabled, QtGui.QPalette.HighlightedText, color("disabled-text"))
    return palette


# Backwards-compatible constant for callers that captured the default sheet.
WORKBENCH_STYLESHEET = None


def apply_workbench_theme(app: QtWidgets.QApplication, theme=None) -> str:
    """Apply one bundled theme to the application and return its key."""

    global WORKBENCH_STYLESHEET
    key = set_active_theme(theme if theme is not None else _active_key)
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    font = QtGui.QFont("Segoe UI", 10)
    if THEMES[key]["font"] is _MONO_FONT:
        font = QtGui.QFont("Cascadia Code", 10)
        font.setStyleHint(QtGui.QFont.Monospace)
    app.setFont(font)
    app.setPalette(_palette_for(key))
    WORKBENCH_STYLESHEET = build_stylesheet(key)
    app.setStyleSheet(WORKBENCH_STYLESHEET)
    return key
