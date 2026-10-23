# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

from functools import partial
import os
from PIL import Image
from PyQt5 import QtWidgets
from IVAS.widgets.background import TaskWorker

from IVAS.ui.visual_prompt_dock import Ui_Form


class VisualPromptDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(VisualPromptDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self._worker = None
        self._context = None
        for button in (self.pushButton_positive_box, self.pushButton_negative_box,
                       self.pushButton_predict):
            button.setMinimumWidth(0)
            button.setMaximumWidth(16777215)

        self.pushButton_positive_box.clicked.connect(partial(self.mainwindow.scene.start_segment_anything_visual, True))
        self.pushButton_negative_box.clicked.connect(partial(self.mainwindow.scene.start_segment_anything_visual, False))
        self.pushButton_predict.clicked.connect(self._predict)

    def _predict(self):
        scene = self.mainwindow.scene
        if self._worker is not None:
            return
        if not scene._frame_is_editable():
            return
        if (not self.mainwindow.use_segment_anything) or self.mainwindow.segany.model_source not in ("sam3", "sam3.1"):
            QtWidgets.QMessageBox.warning(self, "warning", "Only SAM3 is supported!")
            self.mainwindow.scene.cancel_draw()
            return

        if len(self.mainwindow.scene.prompt_visual_items) < 1:
            self.mainwindow.scene.cancel_draw()
            return

        self.setEnabled(False)
        self.label_num_objects.setText("......")
        self.label_num_objects.repaint()

        prompt_visual_positions = [[visual_item.points[0].x(),
                                    visual_item.points[0].y(),
                                    visual_item.points[1].x(),
                                    visual_item.points[1].y()]
                                   for visual_item in self.mainwindow.scene.prompt_visual_items]

        window = self.mainwindow
        category = window.current_category
        path = os.path.abspath(os.path.join(window.image_root, window.files_list[window.current_index]))
        model = window.segany
        cached = window.seganythread.results_dict.get(window.current_index, {}).get("features")
        labels = list(scene.prompt_visual_labels)
        context = (object(), window.sam_model_generation, path, category)
        self._context = context

        def predict(cancelled, progress):
            with model.inference_lock:
                if cancelled.is_set():
                    return None
                if model.model_source == "sam3.1":
                    if cached is None:
                        raise ValueError("Wait for image encoding to finish")
                    model.predictor.set_cached_state(cached)
                    image = None
                else:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                return model.predictor.predict_with_visual_prompt(image, prompt_visual_positions, labels)

        self._worker = TaskWorker(predict)
        self._worker.result.connect(self._prediction_finished)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def invalidate_prediction(self):
        self._context = None
        if self._worker is not None:
            self._worker.cancel()

    def _prediction_finished(self, result, error):
        window = self.mainwindow
        context = self._context
        if (context is None or getattr(window, "_closing_requested", False)
                or context[1] != window.sam_model_generation or window.current_index is None
                or not window.scene._frame_is_editable()
                or context[2] != os.path.abspath(os.path.join(window.image_root, window.files_list[window.current_index]))):
            return
        if error:
            self.label_num_objects.setText("0")
            QtWidgets.QMessageBox.warning(self, "Visual segmentation failed", error)
            return
        if result is not None:
            masks, scores = result
            count = window._process_sam_masks(masks, scores, context[3])
            self.label_num_objects.setText(str(count))
            window.scene.finish_draw()

    def _worker_finished(self):
        self._worker = None
        self._context = None
        self.setEnabled(True)
