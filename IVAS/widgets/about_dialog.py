# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog

from IVAS.ui.about_dialog import Ui_Dialog


class AboutDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent):
        super(AboutDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        from IVAS import __version__

        version_text = "V{}".format(__version__)
        self.label_version.setText(version_text)
        self.pushButton_close.clicked.connect(self.close)
