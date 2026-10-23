# -*- coding: utf-8 -*-
"""Semantic colour system and Qt stylesheet for the unified workbench."""

from types import MappingProxyType

from PyQt5 import QtGui, QtWidgets


# Keep these roles synchronized with DESIGN.md and theme-matrix.json. Feature
# widgets consume roles through theme_hex/theme_color instead of raw colours.
THEME_TOKENS = MappingProxyType(
    {
        "canvas": "#111827",
        "canvas-raised": "#1F2937",
        "surface": "#F6F8FB",
        "surface-raised": "#FFFFFF",
        "surface-subtle": "#E8EEF4",
        "text": "#172033",
        "text-muted": "#526174",
        "border": "#718096",
        "border-subtle": "#CBD5E1",
        "primary": "#087F8C",
        "primary-hover": "#066B76",
        "primary-active": "#075B66",
        "on-primary": "#FFFFFF",
        "danger": "#B42318",
        "danger-hover": "#8F1C14",
        "danger-active": "#731610",
        "on-danger": "#FFFFFF",
        "focus": "#087F8C",
        "selection": "#C9E9ED",
        "on-selection": "#0B3E45",
        "disabled-bg": "#E2E8EF",
        "disabled-text": "#526174",
        "timeline-phase": "#21A6B3",
        "timeline-task": "#F4A340",
        "success": "#207A4B",
        "warning": "#9A6700",
    }
)


def theme_hex(role: str) -> str:
    """Return one declared semantic colour role."""

    try:
        return THEME_TOKENS[role]
    except KeyError as error:
        raise KeyError("Unknown IVAS theme role: {}".format(role)) from error


def theme_color(role: str) -> QtGui.QColor:
    return QtGui.QColor(theme_hex(role))


_STYLESHEET_TEMPLATE = r"""
QWidget {{
    color: {text};
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 10pt;
}}

QMainWindow, QDialog, QWidget#mediaWorkspaceStack {{
    background: {surface};
}}

QGraphicsView#annotationCanvas, QFrame#videoAnnotationStage {{
    background: {canvas};
    border: 1px solid {border};
}}

QMenuBar {{
    background: {surface-raised};
    border-bottom: 1px solid {border-subtle};
    padding: 2px 6px;
}}

QMenuBar::item {{
    background: transparent;
    border-radius: 4px;
    padding: 6px 10px;
}}

QMenuBar::item:selected, QMenuBar::item:pressed {{
    color: {on-selection};
    background: {selection};
}}

QMenu {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border};
    padding: 5px;
}}

QMenu::item {{
    border-radius: 4px;
    padding: 7px 30px 7px 24px;
}}

QMenu::item:selected {{
    color: {on-primary};
    background: {primary};
}}

QMenu::separator {{
    background: {border-subtle};
    height: 1px;
    margin: 5px 8px;
}}

QMenu#menuFile::separator {{
    background: transparent;
    border-top: 1px dashed {border};
    height: 1px;
    margin: 6px 8px;
}}

QLabel#fileMenuHeading {{
    color: {text-muted};
    font-weight: 600;
    padding: 5px 24px;
}}

QToolBar {{
    background: {surface-raised};
    border: 0;
    border-bottom: 1px solid {border-subtle};
    spacing: 4px;
    padding: 5px 7px;
}}

QToolBar#toolBar_2 {{
    border-bottom: 0;
    border-left: 1px solid {border-subtle};
}}

QToolBar::separator {{
    background: {border-subtle};
    margin: 5px 5px;
    width: 1px;
    height: 1px;
}}

QToolButton {{
    color: {text};
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    min-width: 30px;
    min-height: 30px;
    padding: 3px 7px;
}}

QToolButton:hover {{
    color: {on-selection};
    background: {selection};
    border-color: {border};
}}

QToolButton:pressed, QToolButton:checked {{
    color: {on-primary};
    background: {primary-active};
    border-color: {primary-active};
}}

QToolButton:focus {{
    border: 2px solid {focus};
    padding: 2px 6px;
}}

QToolButton:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
    border-color: {border-subtle};
}}

QToolButton#primaryToolMenu {{
    color: {on-primary};
    background: {primary};
    border-color: {primary};
    font-weight: 600;
    padding-left: 10px;
    padding-right: 10px;
}}

QToolButton#primaryToolMenu:hover {{
    background: {primary-hover};
    border-color: {primary-hover};
}}

QToolButton#primaryToolMenu:pressed {{
    background: {primary-active};
    border-color: {primary-active};
}}

QToolButton#primaryToolMenu:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
    border-color: {border-subtle};
}}

QDockWidget {{
    color: {text};
    font-weight: 600;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background: {surface-subtle};
    border-top: 1px solid {border-subtle};
    border-bottom: 1px solid {border-subtle};
    padding: 7px 9px;
    text-align: left;
}}

QDockWidget > QWidget {{
    background: {surface-raised};
}}

QGroupBox {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border-subtle};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 4px;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border};
    border-radius: 5px;
    selection-color: {on-selection};
    selection-background-color: {selection};
    padding: 5px 7px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {focus};
    padding: 4px 6px;
}}

QListWidget:focus, QTreeWidget:focus, QTableWidget:focus {{
    border: 2px solid {focus};
}}

QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled,
QTextEdit:disabled, QPlainTextEdit:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
}}

QPushButton {{
    color: {text};
    background: {surface-raised};
    border: 1px solid {border};
    border-radius: 5px;
    min-height: 28px;
    padding: 4px 11px;
}}

QPushButton:hover {{
    color: {on-selection};
    background: {selection};
    border-color: {focus};
}}

QPushButton:pressed, QPushButton:checked {{
    color: {on-primary};
    background: {primary-active};
    border-color: {primary-active};
}}

QPushButton:focus {{
    border: 2px solid {focus};
    padding: 3px 10px;
}}

QPushButton:disabled {{
    color: {disabled-text};
    background: {disabled-bg};
    border-color: {border-subtle};
}}

QPushButton#primaryAction {{
    color: {on-primary};
    background: {primary};
    border-color: {primary};
    font-weight: 600;
}}

QPushButton#primaryAction:hover {{
    color: {on-primary};
    background: {primary-hover};
    border-color: {primary-hover};
}}

QPushButton#primaryAction:pressed {{
    background: {primary-active};
    border-color: {primary-active};
}}

QPushButton#dangerAction {{
    color: {on-danger};
    background: {danger};
    border-color: {danger};
}}

QPushButton#dangerAction:hover {{
    color: {on-danger};
    background: {danger-hover};
    border-color: {danger-hover};
}}

QListWidget, QTreeWidget, QTableWidget {{
    color: {text};
    background: {surface-raised};
    alternate-background-color: {surface};
    border: 1px solid {border};
    border-radius: 5px;
    outline: 0;
}}

QListWidget::item, QTreeWidget::item, QTableWidget::item {{
    min-height: 27px;
    padding: 3px 5px;
}}

QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{
    background: {surface-subtle};
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
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

QSlider:focus::handle:horizontal {{
    border: 3px solid {focus};
}}

QSplitter::handle {{
    background: {border-subtle};
}}

QTabBar#workbenchDockTabs::tab {{
    color: {text};
    background: {surface-subtle};
    border: 1px solid {border-subtle};
    padding: 6px 12px;
    min-height: 22px;
}}

QTabBar#workbenchDockTabs::tab:selected {{
    color: {on-selection};
    background: {surface-raised};
    border-bottom: 3px solid {primary};
}}

QToolButton#workbenchDockTabScrollButton {{
    background: {surface-raised};
    border: 1px solid {border-subtle};
    border-radius: 0;
    width: 28px;
    min-width: 28px;
    padding: 0;
}}

QTabBar#workbenchDockTabs::scroller {{
    width: 56px;
}}

QProgressBar {{
    color: {text};
    background: {disabled-bg};
    border: 1px solid {border};
    border-radius: 4px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: {primary};
    border-radius: 3px;
}}

QStatusBar {{
    color: {text-muted};
    background: {surface-raised};
    border-top: 1px solid {border-subtle};
}}

QStatusBar QLabel {{
    padding: 2px 7px;
}}

QLabel#selectedCategory {{
    color: {on-selection};
    background: {selection};
    border: 1px solid {focus};
    border-radius: 5px;
    padding: 4px 7px;
    font-weight: 600;
}}

QLabel#workspaceEyebrow, QLabel#workspaceMeta {{
    color: {text-muted};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 1px;
}}

QScrollBar::handle:vertical {{
    background: {border};
    border-radius: 4px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: {text-muted};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
    border: 0;
    height: 0;
}}

QToolTip {{
    color: {surface-raised};
    background: {text};
    border: 1px solid {border};
    padding: 5px;
}}
"""


WORKBENCH_STYLESHEET = _STYLESHEET_TEMPLATE.format_map(THEME_TOKENS)


def apply_workbench_theme(app: QtWidgets.QApplication) -> None:
    """Apply the single supported clinical-light theme before widget creation."""

    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    palette = app.palette()
    palette.setColor(QtGui.QPalette.Window, theme_color("surface"))
    palette.setColor(QtGui.QPalette.WindowText, theme_color("text"))
    palette.setColor(QtGui.QPalette.Base, theme_color("surface-raised"))
    palette.setColor(QtGui.QPalette.AlternateBase, theme_color("surface"))
    palette.setColor(QtGui.QPalette.Text, theme_color("text"))
    palette.setColor(QtGui.QPalette.Button, theme_color("surface-raised"))
    palette.setColor(QtGui.QPalette.ButtonText, theme_color("text"))
    palette.setColor(QtGui.QPalette.Highlight, theme_color("selection"))
    palette.setColor(QtGui.QPalette.HighlightedText, theme_color("on-selection"))
    app.setPalette(palette)
    app.setStyleSheet(WORKBENCH_STYLESHEET)
