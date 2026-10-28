# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets

from IVAS.ui.visual_prompt_dock import Ui_Form
from IVAS.widgets.background import TaskWorker
from IVAS.widgets.mask_prediction import restore_cached_image


_TEXT = {
    "positive_point": ("+ 点", "+ Point"),
    "negative_point": ("− 点", "− Point"),
    "clear": ("清除", "Clear"),
    "positive_point_tip": (
        "添加正样本点：点击目标作为示例，查找所有同类目标。左键按所选极性添加，右键添加相反极性。",
        "Add positive example points: click an object to find all objects like it. "
        "Left click uses the chosen polarity, right click the opposite.",
    ),
    "negative_point_tip": (
        "添加负样本点：点击不需要的目标，排除与其相似的结果。左键按所选极性添加，右键添加相反极性。",
        "Add negative example points: click an unwanted object to exclude results like it. "
        "Left click uses the chosen polarity, right click the opposite.",
    ),
    "clear_tip": ("清除所有视觉提示（框和点）。", "Clear all visual prompts (boxes and points)."),
    "unsupported_tip": ("当前画布不支持点提示。", "This canvas does not support point prompts."),
    "no_prompt": ("请先添加正/负样本框或点。", "Add a positive/negative box or point first."),
    "encoding": ("正在编码当前图像，完成后自动预测…",
                 "Encoding the current image; prediction starts when it is ready…"),
    "no_result": ("视觉提示未找到匹配的目标。", "No objects matched the visual prompt in this image."),
    "found": ("找到 {} 个目标。", "Found {} object(s)."),
}


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

        # Point exemplars sit next to the box buttons; Clear drops every prompt.
        self.widget_points = QtWidgets.QWidget(self)
        self.widget_points.setObjectName("visualPointPrompts")
        self.horizontalLayout_points = QtWidgets.QHBoxLayout(self.widget_points)
        self.horizontalLayout_points.setContentsMargins(0, 0, 0, 0)
        self.pushButton_positive_point = QtWidgets.QPushButton(self.widget_points)
        self.pushButton_positive_point.setObjectName("pushButton_positive_point")
        self.pushButton_negative_point = QtWidgets.QPushButton(self.widget_points)
        self.pushButton_negative_point.setObjectName("pushButton_negative_point")
        self.pushButton_clear = QtWidgets.QPushButton(self.widget_points)
        self.pushButton_clear.setObjectName("pushButton_clear_visual_prompts")
        for button in (self.pushButton_positive_point, self.pushButton_negative_point,
                       self.pushButton_clear):
            button.setMinimumWidth(0)
            self.horizontalLayout_points.addWidget(button)
        self.verticalLayout.insertWidget(1, self.widget_points)
        self.label_result_note = QtWidgets.QLabel(self)
        self.label_result_note.setObjectName("visualPromptResultNote")
        self.label_result_note.setWordWrap(True)
        self.label_result_note.setVisible(False)
        self.verticalLayout.addWidget(self.label_result_note)
        # Keep the controls packed at the top instead of spread over the dock.
        self.verticalLayout.addStretch(1)

        scene = self.mainwindow.scene
        self.pushButton_positive_box.clicked.connect(
            lambda _checked=False: self._start_box(True))
        self.pushButton_negative_box.clicked.connect(
            lambda _checked=False: self._start_box(False))
        self.pushButton_positive_point.clicked.connect(
            lambda _checked=False: self._start_point(True))
        self.pushButton_negative_point.clicked.connect(
            lambda _checked=False: self._start_point(False))
        self.pushButton_clear.clicked.connect(self.clear_prompts)
        self.pushButton_predict.clicked.connect(self._predict)
        self._points_supported = callable(getattr(scene, "start_segment_anything_visual_point", None))
        for button in (self.pushButton_positive_point, self.pushButton_negative_point):
            button.setEnabled(self._points_supported)
        self.retranslateUi(self)

    # ------------------------------------------------------------------ text
    def _chinese(self):
        mainwindow = getattr(self, "mainwindow", None)
        language = getattr(mainwindow, "_ui_language", None)
        if language is None:
            cfg = getattr(mainwindow, "cfg", None) or {}
            language = (cfg.get("software") or {}).get("language")
        return language == "zh"

    def _text(self, key):
        chinese, english = _TEXT[key]
        return chinese if self._chinese() else english

    def retranslateUi(self, Form):
        super().retranslateUi(Form)
        if not hasattr(self, "pushButton_clear"):
            return  # Called by setupUi before the point controls exist.
        self.pushButton_positive_point.setText(self._text("positive_point"))
        self.pushButton_negative_point.setText(self._text("negative_point"))
        self.pushButton_clear.setText(self._text("clear"))
        self.pushButton_clear.setToolTip(self._text("clear_tip"))
        if self._points_supported:
            self.pushButton_positive_point.setToolTip(self._text("positive_point_tip"))
            self.pushButton_negative_point.setToolTip(self._text("negative_point_tip"))
        else:
            for button in (self.pushButton_positive_point, self.pushButton_negative_point):
                button.setToolTip(self._text("unsupported_tip"))

    def show_result_note(self, text):
        self.label_result_note.setText(text)
        self.label_result_note.setVisible(bool(text))

    # --------------------------------------------------------------- prompts
    def _start_box(self, positive):
        self.show_result_note("")
        self.mainwindow.scene.start_segment_anything_visual(positive)

    def _start_point(self, positive):
        self.show_result_note("")
        start = getattr(self.mainwindow.scene, "start_segment_anything_visual_point", None)
        if callable(start):
            start(positive)

    def clear_prompts(self):
        self.show_result_note("")
        if self._worker is None:
            self.mainwindow.scene.cancel_draw()

    def prompts(self):
        """Capture the canvas prompts as plain lists (boxes in xyxy, points in xy)."""
        scene = self.mainwindow.scene
        boxes = []
        for item in scene.prompt_visual_items:
            (x1, y1), (x2, y2) = ((p.x(), p.y()) for p in item.points[:2])
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        box_labels = [bool(label) for label in scene.prompt_visual_labels]
        points = [[float(x), float(y)]
                  for x, y in getattr(scene, "prompt_visual_point_positions", [])]
        point_labels = [bool(label) for label in getattr(scene, "prompt_visual_point_labels", [])]
        return boxes, box_labels, points, point_labels

    # ------------------------------------------------------------ prediction
    def _predict(self):
        scene = self.mainwindow.scene
        if self._worker is not None:
            return
        if not scene._frame_is_editable():
            return
        window = self.mainwindow
        if (not window.use_segment_anything) or window.segany.model_source not in ("sam3", "sam3.1"):
            QtWidgets.QMessageBox.warning(self, "warning", "Only SAM3 is supported!")
            scene.cancel_draw()
            return

        boxes, box_labels, points, point_labels = self.prompts()
        if not boxes and not points:
            scene.cancel_draw()
            self.show_result_note(self._text("no_prompt"))
            return
        if (window.segany.model_source == "sam3.1"
                and window.current_index not in window.seganythread.results_dict):
            self.show_result_note(self._text("encoding"))
            window.run_when_encoded("visual", self._predict)
            return

        self.setEnabled(False)
        self.show_result_note("")
        self.label_num_objects.setText("......")
        self.label_num_objects.repaint()

        category = window.current_category
        path = os.path.abspath(os.path.join(window.image_root, window.files_list[window.current_index]))
        model = window.segany
        cached = window.seganythread.results_dict.get(window.current_index)
        if cached is not None and cached.get("image_path") not in (None, path) and \
                os.path.normcase(os.path.abspath(cached["image_path"])) != os.path.normcase(path):
            cached = None
        context = (object(), window.sam_model_generation, path, category)
        self._context = context
        facade = getattr(model, "predict_with_visual_prompt", None)

        def predict(cancelled, progress):
            with model.inference_lock:
                if cancelled.is_set():
                    return None
                if model.model_source == "sam3.1":
                    if cached is None or cached.get("features") is None:
                        raise ValueError("Wait for image encoding to finish")
                    model.predictor.set_cached_state(cached["features"])
                    image = None
                else:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                    if points:
                        # Points resolve instances with the interactive head,
                        # which needs this image's features.
                        if cached is not None and cached.get("features") is not None:
                            restore_cached_image(model, cached)
                        else:
                            model.set_image(np.asarray(image))
                if cancelled.is_set():
                    return None
                if points:
                    if not callable(facade):
                        raise ValueError("This SAM model does not support point prompts")
                    return facade(image, boxes, box_labels, points, point_labels)
                return model.predictor.predict_with_visual_prompt(image, boxes, box_labels)

        self._worker = TaskWorker(predict)
        self._worker.result.connect(self._prediction_finished)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def invalidate_prediction(self):
        self._context = None
        self.show_result_note("")
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
            message = self._text("found").format(count) if count else self._text("no_result")
            self.show_result_note("" if count else message)
            window.statusbar.showMessage(message, 6000)

    def _worker_finished(self):
        self._worker = None
        self._context = None
        self.setEnabled(True)
