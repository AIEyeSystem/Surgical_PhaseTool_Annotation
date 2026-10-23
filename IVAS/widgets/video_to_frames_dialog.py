# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
from pathlib import Path

import cv2
from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from IVAS.widgets.background import TaskDialogMixin
from IVAS.utils.batch import video_to_frames

from IVAS.ui.video_to_frames import Ui_Dialog
from IVAS.utils.check import has_chinese


class Video2FramesDialog(TaskDialogMixin, ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.video_path = None
        self.save_root = None
        self.pushButton_video_path.clicked.connect(self.open_file)
        self.pushButton_frames_root.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_close.clicked.connect(self.close)

    def open_file(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open video", filter="mp4 (*.mp4 *.MP4);;All Files (*)"
        )
        if file:
            self.video_path = file
            self.lineEdit_video_path.setText(file)

            try:
                camera = cv2.VideoCapture(self.video_path)
                fps = int(camera.get(cv2.CAP_PROP_FPS))
                width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

                self.spinBox_step.setValue(max(1, fps))

                self.textBrowser.append("Frames num   : {}".format(frame_count))
                self.textBrowser.append("Frames width : {}".format(width))
                self.textBrowser.append("Frames height: {}".format(height))
                self.textBrowser.append("Frames fps   : {}".format(fps))
            except Exception as e:
                self.textBrowser.append("ERROR | {}".format(e))

            finally:
                if "camera" in locals():
                    camera.release()

        else:
            self.video_path = None
            self.lineEdit_video_path.clear()

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open dir")
        if dir:
            self.save_root = dir
            self.lineEdit_frames_root.setText(dir)
        else:
            self.save_root = None
            self.lineEdit_frames_root.clear()

    def start(self):
        video_path = self.lineEdit_video_path.text().strip()
        save_root = self.lineEdit_frames_root.text().strip()
        step = self.spinBox_step.value()
        if not video_path or not save_root or not os.path.isfile(video_path) or not os.path.isdir(save_root) or step <= 0:
            self.textBrowser.append("Error: select valid paths and a positive frame step.")
            return
        self.start_task(lambda cancelled, progress: video_to_frames(video_path, save_root, step, cancelled, progress))
