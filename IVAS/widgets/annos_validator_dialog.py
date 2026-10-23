# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import json
import os

from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from IVAS.widgets.background import TaskDialogMixin
from IVAS.utils.batch import validate_annotations
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from IVAS.ui.annos_validator import Ui_Dialog


class AnnosValidatorDialog(TaskDialogMixin, ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(AnnosValidatorDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.pushButton_json_root.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.validate)
        self.pushButton_start.setEnabled(False)
        self.pushButton_close.clicked.connect(self.close)

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open dir")
        if dir:
            self.lineEdit_json_root.setText(dir)
            self.pushButton_start.setEnabled(True)
        else:
            self.lineEdit_json_root.clear()
            self.pushButton_start.setEnabled(False)

    def validate(self):
        root = self.lineEdit_json_root.text().strip()
        if not root or not os.path.isdir(root):
            self.textBrowser.append("Error: select an existing annotation directory.")
            return
        self.start_task(lambda cancelled, progress: validate_annotations(root, cancelled, progress))
