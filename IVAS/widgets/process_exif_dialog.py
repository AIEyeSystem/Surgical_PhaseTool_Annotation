# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os

from PIL import Image, ImageOps
from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from IVAS.widgets.background import TaskDialogMixin
from IVAS.utils.batch import process_exif

from IVAS.ui.process_exif_dialog import Ui_Dialog


class ProcessExifDialog(TaskDialogMixin, ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.mainwindow = mainwindow

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.radioButton_apply_exif.setChecked(True)
        self.pushButton_image_root.clicked.connect(self.open_dir)
        self.pushButton_save_root.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_close.clicked.connect(self.close)

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open dir")
        if self.sender() == self.pushButton_image_root:
            lineEdit = self.lineEdit_image_root
        elif self.sender() == self.pushButton_save_root:
            lineEdit = self.lineEdit_save_root
        else:
            return
        if dir:
            lineEdit.setText(dir)
        else:
            lineEdit.clear()

    def start(self):
        image_root = self.lineEdit_image_root.text().strip()
        save_root = self.lineEdit_save_root.text().strip()
        if not image_root or not save_root or not os.path.isdir(image_root) or not os.path.isdir(save_root):
            self.textBrowser.append("Error: select valid input and output directories.")
            return
        apply_exif = self.radioButton_apply_exif.isChecked()
        self.start_task(lambda cancelled, progress: process_exif(image_root, save_root, apply_exif, cancelled, progress))
