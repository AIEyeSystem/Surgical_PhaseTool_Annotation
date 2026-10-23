# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import sys
from importlib.metadata import entry_points

from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog

from IVAS.branding import (
    LEGACY_PLUGIN_ENTRYPOINT_GROUPS,
    PLUGIN_ENTRYPOINT_GROUP,
)
from IVAS.ui.plugin_manager_dialog import Ui_Dialog


class PluginManagerDialog(ResponsiveDialog, Ui_Dialog):
    """Plugin manager interface, also include most of all functions of plugin."""

    def __init__(self, parent, mainwindow):
        super(PluginManagerDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setSectionResizeMode(
            4, QtWidgets.QHeaderView.Stretch
        )
        self.tableWidget.setColumnWidth(0, 100)
        self.tableWidget.setColumnWidth(1, 250)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 150)

        self.plugins = []

        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_reload.clicked.connect(self.load_plugins)
        self.load_plugins()

    def load_plugins(self):
        self.tableWidget.setRowCount(0)
        for plugin in self.plugins:
            try:
                plugin.disable_plugin()
            except Exception as error:
                self._report_plugin_error(plugin, "disable_plugin", error)
        self.plugins = []

        print("loading IVAS plugins")
        discovered = entry_points()
        eps = []
        seen = set()
        for group in (
            PLUGIN_ENTRYPOINT_GROUP,
            *LEGACY_PLUGIN_ENTRYPOINT_GROUPS,
        ):
            if sys.version_info >= (3, 10):
                selected = discovered.select(group=group)
            else:
                selected = discovered.get(group, [])
            for ep in selected:
                identity = (ep.name, ep.value)
                if identity not in seen:
                    seen.add(identity)
                    eps.append(ep)
        for ep in eps:
            try:
                plugin_class = ep.load()
                plugin_instance = plugin_class()
                plugin_instance.init_plugin(self.mainwindow)
                self.plugins.append(plugin_instance)
                print("loaded plugin: ", plugin_instance.get_plugin_name())
            except Exception as error:
                print("failed to load plugin [{}]: {}".format(ep, error))

        self.update_gui()

    def update_gui(self):
        self.tableWidget.setRowCount(0)
        row = 0
        for plugin_instance in self.plugins:
            activate_checkbox = QtWidgets.QCheckBox()
            activate_checkbox.stateChanged.connect(
                plugin_instance.activate_state_changed
            )
            plugin_name_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_name()
            )
            plugin_author_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_author()
            )
            plugin_author_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plugin_version_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_version()
            )
            plugin_version_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plugin_description_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_description()
            )

            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.tableWidget.setCellWidget(row, 0, activate_checkbox)
            self.tableWidget.setItem(row, 1, plugin_name_item)
            self.tableWidget.setItem(
                row, 2, QtWidgets.QTableWidgetItem(plugin_author_item)
            )
            self.tableWidget.setItem(
                row, 3, QtWidgets.QTableWidgetItem(plugin_version_item)
            )
            self.tableWidget.setItem(
                row, 4, QtWidgets.QTableWidgetItem(plugin_description_item)
            )

            row += 1

    def _report_plugin_error(self, plugin, event, error):
        message = "Plugin {} / {}: {}".format(type(plugin).__name__, event, error)
        print(message)
        self.mainwindow.statusbar.showMessage(message, 10000)

    def _dispatch(self, event, *args, critical=False):
        for plugin in tuple(self.plugins):
            if not getattr(plugin, "enabled", False):
                continue
            try:
                getattr(plugin, event)(*args)
            except Exception as error:
                self._report_plugin_error(plugin, event, error)
                if critical:
                    raise RuntimeError("Plugin rejected annotation save: {}".format(error)) from error

    def trigger_before_image_open(self, image_path):
        self._dispatch("before_image_open_event", image_path)

    def trigger_after_image_open(self):
        self._dispatch("after_image_open_event")

    def trigger_before_annotation_start(self):
        self._dispatch("before_annotation_start_event")

    def trigger_after_annotation_created(self):
        self._dispatch("after_annotation_created_event")

    def trigger_after_annotation_changed(self):
        self._dispatch("after_annotation_changed_event")

    def trigger_before_annotations_save(self):
        self._dispatch("before_annotations_save_event", critical=True)

    def trigger_after_annotations_saved(self):
        self._dispatch("after_annotations_saved_event")

    def trigger_after_sam_encode_finished(self, index: int):
        self._dispatch("after_sam_encode_finished_event", index)

    def trigger_on_mouse_move(self, scene_pos: QtCore.QPointF):
        self._dispatch("on_mouse_move_event", scene_pos)

    def trigger_on_mouse_release(self, scene_pos: QtCore.QPointF):
        self._dispatch("on_mouse_release_event", scene_pos)

    def trigger_on_mouse_press(self, scene_pos: QtCore.QPointF):
        self._dispatch("on_mouse_press_event", scene_pos)

    def trigger_on_mouse_pressed_and_mouse_move(self, scene_pos: QtCore.QPointF):
        self._dispatch("on_mouse_pressed_and_mouse_move_event", scene_pos)

    def trigger_application_start(self):
        self._dispatch("application_start_event")

    def trigger_application_shutdown(self):
        self._dispatch("application_shutdown_event")
