# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
import json
from contextlib import nullcontext
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import imgviz
import numpy as np
import yaml
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from PyQt5.QtCore import QThread, pyqtSignal

from IVAS.branding import ANNOTATION_FORMAT, CATEGORY_CONFIG_NAME
from IVAS.configs import CONTOURMode, CONTOURMethod, save_config
from IVAS.annotation import Annotation, Object
from IVAS.utils.masks import mask_to_polygons
from IVAS.ui.auto_segment import Ui_Dialog


class AutoSegmentThread(QThread):
    message = pyqtSignal(int, int, str)

    def __init__(self, mainwindow):
        super(AutoSegmentThread, self).__init__()
        self.mainwindow = mainwindow
        self.cancel = False
        self.image_dir = None
        self.label_dir = None
        self.save_dir = None

    def cancelled(self):
        return self.cancel or self.isInterruptionRequested()

    def run(self):
        try:
            self._run_batch()
        except Exception as error:
            self.message.emit(-1, -1, "Error: {}".format(error))

    def _run_batch(self):
        model = self.model
        image_names = sorted(path for path in Path(self.image_dir).iterdir()
                             if path.suffix.lower() in Image.registered_extensions())
        categories = set()
        errors = 0
        for index, image_path in enumerate(image_names, 1):
            if self.cancelled():
                self.message.emit(-1, -1, "Cancelled")
                return
            self.message.emit(index, len(image_names), image_path.name)
            label = Path(self.label_dir) / (image_path.stem + ".xml")
            try:
                objects = ET.parse(label).getroot().findall("object")
                annotation = Annotation(str(image_path), str(Path(self.save_dir) / (image_path.stem + ".json")))
                with Image.open(image_path) as image:
                    pixels = np.asarray(image.convert("RGB"))
                # Encoding and all predictions for an image share one state lease.
                object_failed = False
                with getattr(model, "inference_lock", nullcontext()):
                    if self.cancelled():
                        return
                    model.set_image(pixels)
                    for group, obj in enumerate(objects, 1):
                        if self.cancelled():
                            return
                        try:
                            name = obj.findtext("name")
                            box = [float(obj.findtext("bndbox/" + key))
                                   for key in ("xmin", "ymin", "xmax", "ymax")]
                            if not name or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                                raise ValueError("Invalid category or bounding box")
                            mask = model.predict_with_box_prompt(box=np.asarray(box))
                            contours, hierarchy = mask_to_polygons(mask, *self.contour_options)
                            for number, contour in enumerate(contours):
                                if len(contour) < 3:
                                    continue
                                inner = (self.contour_options[0] == CONTOURMode.SAVE_ALL
                                         and hierarchy is not None and hierarchy[0][number][3] != -1)
                                category = "__background__" if inner else name
                                points = contour.reshape(-1, 2).tolist()
                                annotation.objects.append(Object(category, 0 if inner else group, points,
                                    float(cv2.contourArea(contour)), len(annotation.objects) + 1, box,
                                    source="sam", review_status="draft"))
                                categories.add(category)
                        except Exception as error:
                            object_failed = True
                            errors += 1
                            self.message.emit(-1, -1, "Error: {} object {}: {}".format(image_path.name, group, error))
                if self.cancelled():
                    return
                if not object_failed:
                    annotation.save_annotation()
            except Exception as error:
                errors += 1
                self.message.emit(-1, -1, "Error: {}: {}".format(image_path.name, error))
        if self.cancelled():
            return
        cmap = imgviz.label_colormap()
        labels = [{"name": name, "color": "#{:02x}{:02x}{:02x}".format(*cmap[(i + 1) % len(cmap)])}
                  for i, name in enumerate(sorted(categories))]
        save_config({"label": labels}, str(Path(self.save_dir) / CATEGORY_CONFIG_NAME))
        self.message.emit(-1, -1, "Finished: {} error(s)".format(errors))


class AutoSegmentDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(AutoSegmentDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.image_dir = None
        self.label_dir = None
        self.save_dir = None

        self.auto_segment_thread = AutoSegmentThread(self.mainwindow)
        self.auto_segment_thread.message.connect(self.print_message)
        self.auto_segment_thread.finished.connect(self._finished)
        self._close_when_finished = False
        self.pushButton_close.clicked.connect(self.close)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.init_connect()

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open dir")
        if self.sender() == self.pushButton_image_dir:
            lineEdit = self.lineEdit_image_dir
        elif self.sender() == self.pushButton_label_dir:
            lineEdit = self.lineEdit_label_dir
        elif self.sender() == self.pushButton_save_dir:
            lineEdit = self.lineEdit_save_dir
        else:
            return
        if dir:
            if lineEdit is not None:
                lineEdit.setText(dir)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def start(self):
        if self.auto_segment_thread.isRunning():
            return
        self.auto_segment_thread.cancel = False
        self._close_when_finished = False

        self.image_dir = self.lineEdit_image_dir.text()
        self.label_dir = self.lineEdit_label_dir.text()
        self.save_dir = self.lineEdit_save_dir.text()

        if self.image_dir == "" or self.label_dir == "" or self.save_dir == "":
            self.textBrowser.append(
                "Error: image root / xml root / save root is None. "
            )
            return

        if not all(os.path.isdir(path) for path in (self.image_dir, self.label_dir, self.save_dir)):
            self.textBrowser.append("Error: select existing directories")
            return
        model = getattr(self.mainwindow, "segany", None)
        if model is None:
            self.textBrowser.append("Error: select a SAM model")
            return
        self.auto_segment_thread.model = model
        self.auto_segment_thread.contour_options = (
            self.mainwindow.scene.contour_mode, self.mainwindow.scene.contour_method,
            self.mainwindow.cfg["software"]["use_polydp"])
        self.pushButton_start.setEnabled(False)
        self.auto_segment_thread.image_dir = self.image_dir
        self.auto_segment_thread.label_dir = self.label_dir
        self.auto_segment_thread.save_dir = self.save_dir
        self.auto_segment_thread.start()

    def cancel(self):
        self.auto_segment_thread.cancel = True
        self.auto_segment_thread.requestInterruption()

    def _finished(self):
        self.pushButton_start.setEnabled(True)
        if self._close_when_finished:
            self.close()

    def closeEvent(self, event):
        if self.auto_segment_thread.isRunning():
            self._close_when_finished = True
            self.cancel()
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        self.close()

    def print_message(self, index, all, message):
        if index > 0:
            self.progressBar.setValue(index)
        if all > 0:
            self.progressBar.setMaximum(all)
        if message:
            self.textBrowser.append(
                "{} | {}".format(
                    (
                        "{:>8s}/{:<8s}".format(str(index), str(all))
                        if (index > 0 and all > 0)
                        else "{:>8s} {:<8s}".format("", "")
                    ),
                    message,
                )
            )
            print(
                "{} | {}".format(
                    (
                        "{:>8s}/{:<8s}".format(str(index), str(all))
                        if (index > 0 and all > 0)
                        else "{:>8s} {:<8s}".format("", "")
                    ),
                    message,
                )
            )

    def init_connect(self):
        self.pushButton_image_dir.clicked.connect(self.open_dir)
        self.pushButton_label_dir.clicked.connect(self.open_dir)
        self.pushButton_save_dir.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_cancel.clicked.connect(self.cancel)
