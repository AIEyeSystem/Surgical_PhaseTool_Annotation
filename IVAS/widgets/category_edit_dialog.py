# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

from PyQt5 import QtCore, QtGui, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog

from IVAS.ui.category_edit import Ui_Dialog
from IVAS.widgets.list_item import fit_list_item


class CategoryEditDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow, scene):
        super(CategoryEditDialog, self).__init__(parent)

        self.setupUi(self)
        self.mainwindow = mainwindow
        self.scene = scene
        self.polygons = []

        self.listWidget.itemClicked.connect(self.get_category)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)

        #
        self.checkBox_category_enabled.stateChanged.connect(self.check_category_enabled)
        self.checkBox_group_enabled.stateChanged.connect(self.check_group_enabled)
        self.checkBox_note_enabled.stateChanged.connect(self.check_note_enabled)
        self.checkBox_iscrowded_enabled.stateChanged.connect(self.check_crowded_enabled)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

    def check_category_enabled(self, checked):
        self.lineEdit_category.setEnabled(checked)

    def check_group_enabled(self, checked):
        self.spinBox_group.setEnabled(checked)

    def check_note_enabled(self, checked):
        self.lineEdit_note.setEnabled(checked)

    def check_crowded_enabled(self, checked):
        self.checkBox_iscrowded.setEnabled(checked)

    def load_cfg(self):
        """Load the cfg and update the interface."""
        self.listWidget.clear()

        labels = self.mainwindow.cfg.get("label", [])

        for label in labels:
            name = label.get("name", "UNKNOW")
            color = label.get("color", "#000000")
            # item = QtWidgets.QListWidgetItem()
            # item.setText(name)
            # item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            # self.listWidget.addItem(item)

            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, name)
            widget = QtWidgets.QWidget(self.listWidget)

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)
            label_category = QtWidgets.QLabel()
            label_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label_category.setText(self.mainwindow.category_display_name(name))
            label_category.setToolTip(label_category.text())
            label_category.setObjectName("label_category")

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName("label_color")

            layout.addWidget(label_color)
            layout.addWidget(label_category)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)
            fit_list_item(item, widget)

            if len(self.polygons) == 1 and self.polygons[0].category == name:
                self.listWidget.setCurrentItem(item)

        if len(self.polygons) != 1:
            self.spinBox_group.clear()
            self.lineEdit_category.clear()
            self.checkBox_iscrowded.setCheckState(False)
            self.lineEdit_note.clear()
            self.label_layer.setText("{}".format(""))
            self.label_area.setText("{}".format(""))

            self.checkBox_category_enabled.setChecked(False)
            self.checkBox_group_enabled.setChecked(False)
            self.checkBox_note_enabled.setChecked(False)
            self.checkBox_iscrowded_enabled.setChecked(False)

        elif len(self.polygons) == 1:
            self.lineEdit_category.setText("{}".format(self.polygons[0].category))
            self.lineEdit_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.spinBox_group.setValue(self.polygons[0].group)
            iscrowd = (
                QtCore.Qt.CheckState.Checked
                if self.polygons[0].iscrowd
                else QtCore.Qt.CheckState.Unchecked
            )
            self.checkBox_iscrowded.setCheckState(iscrowd)
            self.lineEdit_note.setText("{}".format(self.polygons[0].note))
            self.label_layer.setText("{}".format(self.polygons[0].zValue()))
            self.label_area.setText(
                "{:.0f}{}".format(
                    self.polygons[0].area,
                    (
                        ""
                        if self.mainwindow.cfg["software"]["real_time_area"]
                        else "(no real time)"
                    ),
                )
            )

            self.checkBox_category_enabled.setChecked(True)
            self.checkBox_group_enabled.setChecked(True)
            self.checkBox_note_enabled.setChecked(True)
            self.checkBox_iscrowded_enabled.setChecked(True)

        if self.listWidget.count() == 0:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please set categorys before tagging."
            )

    def get_category(self, item: QtWidgets.QListWidgetItem):
        """
        Triggered when category item selected.

        Arguments:
            item: category item.
        """
        self.lineEdit_category.setText(item.data(QtCore.Qt.UserRole))
        self.lineEdit_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def apply(self):
        """Set attributes of polygon."""
        for polygon in self.polygons:
            category = self.lineEdit_category.text() if self.checkBox_category_enabled.isChecked() else polygon.category
            group = self.spinBox_group.value() if self.checkBox_group_enabled.isChecked() else polygon.group
            is_crowd = self.checkBox_iscrowded.isChecked() if self.checkBox_iscrowded_enabled.isChecked() else polygon.iscrowd
            note = self.lineEdit_note.text() if self.checkBox_note_enabled.isChecked() else polygon.note

            if not category:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please select one category before submitting."
                )
                return

            # 设置polygon 属性
            polygon.set_drawed(
                category,
                group,
                is_crowd,
                note,
                QtGui.QColor(self.mainwindow.category_color_dict.get(category, "#6F737A")),
            )
            self.mainwindow.annos_dock_widget.update_listwidget()

        self.polygons = []
        self.scene.change_mode_to_view()
        self.close()

    def cancel(self):
        self.scene.cancel_draw()
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.cancel()

    def reject(self):
        self.cancel()
