# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os

from PyQt5 import QtCore, QtGui, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog

from IVAS.configs import *
from IVAS.ui.category_setting_dialog import Ui_Dialog
from IVAS.widgets.list_item import fit_list_item


class CategorySettingDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(CategorySettingDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.color_button.ensurePolished()
        self.color_button.setMinimumSize(self.color_button.minimumSizeHint())

        self.init_connect()

    def get_item_and_widget(
        self, category: str, color: str
    ) -> (QtWidgets.QListWidgetItem, QtWidgets.QWidget):
        """
        Return item and item widget for given category and color.

        Arguments:
            category: category name
            color: color name

        Returns:
            The item of category list widget and the widget of the item.
        """
        item = QtWidgets.QListWidgetItem()
        labels = self.mainwindow.cfg.get("label", [])
        original_index = next((i for i, label in enumerate(labels)
                               if label.get("name") == category), None)
        if original_index is not None:
            source = dict(labels[original_index])
            source.setdefault("id", original_index)
        else:
            existing = labels + [
                self.category_list_widget.item(i).data(QtCore.Qt.UserRole)
                for i in range(self.category_list_widget.count())
            ]
            ids = [label.get("id", i) for i, label in enumerate(existing)]
            category_id = max((value for value in ids if isinstance(value, int)), default=-1) + 1
            source = {"name": category, "id": category_id}
        item.setData(QtCore.Qt.UserRole, source)

        widget = QtWidgets.QWidget(self.category_list_widget)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(9, 4, 9, 4)
        category_label = QtWidgets.QLabel()
        category_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        display = self.mainwindow.category_display_name(category)
        if category not in self.mainwindow.category_aliases:
            display = "{} [{}]".format(category, source["id"])
        category_label.setText(display)
        category_label.setToolTip(category_label.text())
        category_label.setObjectName("category")
        # 颜色
        color_button = QtWidgets.QPushButton()
        color_button.setStyleSheet("QWidget {background-color: %s}" % color)
        color_button.setMinimumWidth(50)
        color_button.clicked.connect(self.choice_color)
        color_button.setObjectName("color")
        # 删除
        delete_button = QtWidgets.QPushButton()
        delete_button.setText("delete")
        delete_button.setMinimumWidth(80)
        delete_button.clicked.connect(self.remove_category_item)

        if category == "__background__":
            color_button.setEnabled(False)
            delete_button.setEnabled(False)

        layout.addWidget(category_label, 1)
        layout.addWidget(color_button)
        layout.addWidget(delete_button)
        widget.setLayout(layout)
        fit_list_item(item, widget)
        return item, widget

    def remove_category_item(self):
        """Remove category item."""
        button = self.sender()
        row = self.category_list_widget.indexAt(button.parent().pos()).row()
        self.category_list_widget.takeItem(row)

    def load_cfg(self):
        """Load the cfg and update the interface."""
        self.label_config_file.setText(self.mainwindow.config_file)
        self.category_list_widget.clear()

        for name, color in self.mainwindow.category_color_dict.items():
            item, item_widget = self.get_item_and_widget(name, color=color)
            self.category_list_widget.addItem(item)
            self.category_list_widget.setItemWidget(item, item_widget)

    def add_new_category(self):
        """Add a new category item."""
        category = self.category_input.text().strip()
        if any(self.category_list_widget.item(index).data(QtCore.Qt.UserRole)["name"] == category
               for index in range(self.category_list_widget.count())):
            return
        color = self.color_button.palette().button().color().name()
        if category:
            item, item_widget = self.get_item_and_widget(category, color)
            self.category_list_widget.addItem(item)
            self.category_list_widget.setItemWidget(item, item_widget)
        self.category_input.clear()

    def choice_color(self):
        """Triggered by color button. Select a new color for the category."""
        button = self.sender()
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet("QWidget {background-color: %s}" % color.name())

    def import_cfg(self):
        """Import cfg from yaml file."""
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, filter="Yaml File(*.yaml)"
        )
        if file:
            self.mainwindow.config_file = file
            self.mainwindow.actionSetting.setStatusTip("Config yaml: {}".format(file))
            self.mainwindow.reload_cfg()
        self.load_cfg()

    def export_cfg(self):
        """Export cfg to yaml file."""
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, filter="Yaml File(*.yaml)"
        )
        if not file:
            return
        if not file.endswith(".yaml"):
            file += ".yaml"
        self.mainwindow.save_cfg(file)
        self.load_cfg()

    def apply(self):
        """Apply setting of categories."""
        cfg = load_config(self.mainwindow.config_file) or {}
        cfg["label"] = []
        for index in range(self.category_list_widget.count()):
            item = self.category_list_widget.item(index)
            widget = self.category_list_widget.itemWidget(item)
            color_button = widget.findChild(QtWidgets.QPushButton, "color")
            label = dict(item.data(QtCore.Qt.UserRole))
            label["color"] = color_button.palette().button().color().name()
            cfg["label"].append(label)

        save_config(cfg, self.mainwindow.config_file)
        self.mainwindow.reload_cfg()
        self.close()

    def cancel(self):
        self.close()

    def init_connect(self):
        self.pushButton_add.clicked.connect(self.add_new_category)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_close.clicked.connect(self.cancel)
        self.color_button.clicked.connect(self.choice_color)
        self.pushButton_import.clicked.connect(self.import_cfg)
        self.pushButton_export.clicked.connect(self.export_cfg)
