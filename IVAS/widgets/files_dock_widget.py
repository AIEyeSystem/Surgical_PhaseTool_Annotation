# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from IVAS.ui.file_dock import Ui_Form


class FilesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(FilesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.listWidget.clicked.connect(self.listwidget_doubleclick)
        self.lineEdit_jump.returnPressed.connect(self.mainwindow.jump_to)

        self.setAcceptDrops(True)

    def update_widget(self):
        """Update the file list widget."""
        self.listWidget.clear()
        if self.mainwindow.files_list is None:
            return

        for idx, file_name in enumerate(self.mainwindow.files_list):
            # _, file_name = os.path.split(file_path)
            item = QtWidgets.QListWidgetItem()
            # item, item_widget = self.generate_item_and_itemwidget(file_name)

            item.setText(f"[{idx + 1}] {file_name}")
            self.listWidget.addItem(item)
            # self.listWidget.setItemWidget(item, item_widget)

        self.label_all.setText("{}".format(len(self.mainwindow.files_list)))

    def set_select(self, row):
        """Set the current file highlight."""
        self.listWidget.setCurrentRow(row)

    def listwidget_doubleclick(self):
        """Double-click the file list item to open the image."""
        row = self.listWidget.currentRow()
        self.mainwindow.show_image(row)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Support drop event for open a image file or image directory."""
        if len(event.mimeData().urls()) != 1:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Only support one path or dir."
            )
            return
        path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(path):
            loaded = self.mainwindow.load_image_directory(path)
        elif os.path.isfile(path):
            suffixes = tuple('.' + bytes(fmt).decode('ascii').lower()
                             for fmt in QtGui.QImageReader.supportedImageFormats())
            if not path.lower().endswith(suffixes + ('.dcm',)):
                event.ignore()
                return
            directory, filename = os.path.split(path)
            loaded = self.mainwindow.load_image_directory(
                directory, selected_filename=filename
            )
        else:
            event.ignore()
            return
        if loaded:
            event.acceptProposedAction()
        else:
            event.ignore()
