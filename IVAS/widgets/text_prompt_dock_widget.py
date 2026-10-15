# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
from contextlib import nullcontext

from PIL import Image
from PyQt5 import QtCore, QtWidgets

from IVAS.ui.text_prompt_dock import Ui_Form
from IVAS.configs import MAPMode, STATUSMode


class TextPromptThread(QtCore.QThread):
    """Run text grounding without blocking or mutating the Qt scene."""

    resultReady = QtCore.pyqtSignal(int, object, str)

    def __init__(self, job_id, segany, image_path, requests, parent=None, cached_state=None):
        super(TextPromptThread, self).__init__(parent)
        self.job_id = job_id
        self.segany = segany
        self.image_path = image_path
        self.requests = list(requests)
        self.cached_state = cached_state

    def run(self):
        try:
            prompts = [prompt for _, prompt in self.requests]
            # Restore the captured frame under the same lock as prediction:
            # navigation may have requested another image since this job started.
            with getattr(self.segany, "inference_lock", nullcontext()):
                if self.segany.model_source == "sam3.1":
                    self.segany.predictor.set_cached_state(self.cached_state)
                    outputs = self.segany.predict_with_text_prompts(prompts)
                else:
                    with Image.open(self.image_path) as source:
                        image = source.convert("RGB")
                    outputs = [
                        self.segany.predictor.predict_with_text_prompt(image, prompt)
                        for prompt in prompts
                    ]
            results = [
                (category, masks, scores)
                for (category, _), (masks, scores) in zip(
                    self.requests, outputs
                )
            ]
            self.resultReady.emit(self.job_id, results, "")
        except Exception as error:
            self.resultReady.emit(self.job_id, [], str(error))


class TextPromptDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(TextPromptDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self._worker = None
        self._job_id = 0
        self._job_context = None
        # Stack the controls so full category names and prompts fit a narrow dock.
        self.horizontalLayout.removeWidget(self.lineEdit_prompt_text_for_single_category)
        self.horizontalLayout.removeWidget(self.pushButton_predict_for_single_category)
        self.single_category_layout = QtWidgets.QVBoxLayout()
        self.horizontalLayout.addLayout(self.single_category_layout)
        self.label_selected_category = QtWidgets.QLabel()
        self.label_selected_category.setObjectName("selectedCategory")
        self.label_selected_category.setWordWrap(True)
        self.label_selected_category.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        self.single_category_layout.addWidget(self.label_selected_category)

        self.checkBox_use_selected_category = QtWidgets.QCheckBox("Use selected category")
        self.checkBox_use_selected_category.setChecked(True)
        self.single_category_layout.addWidget(self.checkBox_use_selected_category)
        self.comboBox_prompt = QtWidgets.QComboBox()
        self.comboBox_prompt.setEditable(True)
        self.comboBox_prompt.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.comboBox_prompt.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.comboBox_prompt.setMinimumContentsLength(12)
        self.comboBox_prompt.setLineEdit(self.lineEdit_prompt_text_for_single_category)
        self.single_category_layout.addWidget(self.comboBox_prompt)
        self.comboBox_prompt.activated[int].connect(self._choose_prompt)
        self.checkBox_use_selected_category.toggled.connect(self._prompt_source_changed)
        self.comboBox_prompt.setEnabled(False)
        self.lineEdit_prompt_text_for_single_category.setPlaceholderText("Choose a category or enter an English prompt.")
        self.single_category_layout.addWidget(self.pushButton_predict_for_single_category)
        self.pushButton_predict_for_single_category.setText("Segment")
        self.pushButton_predict_for_single_category.setMinimumWidth(0)
        self.pushButton_predict_for_single_category.setMaximumWidth(16777215)
        self.pushButton_predict_for_single_category.setToolTip(
            "Confirm text-guided segmentation for the selected category."
        )
        self.pushButton_predict_for_multi_categoriies.setMaximumWidth(16777215)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addStretch(1)

        self.pushButton_predict_for_single_category.clicked.connect(
            self.predict_for_single_category
        )
        self.pushButton_catgories_setting.clicked.connect(
            self.mainwindow.category_setting
        )
        self.pushButton_predict_for_multi_categoriies.clicked.connect(
            self.predict_for_multi_categories
        )
        self.mainwindow.categories_dock_widget.categoryChanged.connect(
            self.set_selected_category
        )
        self.set_selected_category(self.mainwindow.current_category)

    def refresh_categories(self):
        prompt = self.lineEdit_prompt_text_for_single_category.text()
        with QtCore.QSignalBlocker(self.comboBox_prompt):
            self.comboBox_prompt.clear()
            for category in self.mainwindow.category_color_dict:
                if category != "__background__":
                    self.comboBox_prompt.addItem(
                        self.mainwindow.category_display_name(category), category
                    )
            self.comboBox_prompt.setEditText(prompt)
        self.set_selected_category(self.mainwindow.current_category)

    def _choose_prompt(self, index):
        self.comboBox_prompt.setEditText(self.comboBox_prompt.itemData(index) or "")

    def _prompt_source_changed(self, automatic):
        self.comboBox_prompt.setEnabled(not automatic and self._worker is None)
        self.set_selected_category(self.mainwindow.current_category)

    def set_selected_category(self, category):
        category = category or "__background__"
        display = self.mainwindow.category_display_name(category)
        self.label_selected_category.setText(display)
        self.label_selected_category.setToolTip("Annotation category: {}".format(category))
        if self.checkBox_use_selected_category.isChecked():
            self.comboBox_prompt.setEditText("" if category == "__background__" else category)

    def invalidate_prediction(self):
        """A result belongs to one frame visit, even after navigating away and back."""
        self._job_context = None

    def predict_for_selected_category(self):
        category = self.mainwindow.current_category
        if category == "__background__":
            QtWidgets.QMessageBox.warning(self, "Warning", "Select a foreground category first.")
            return
        self._start_prediction([(category, category)])

    def _validate_ready(self):
        if self.mainwindow.scene.mode != STATUSMode.VIEW:
            self.mainwindow.statusbar.showMessage("请先完成或取消当前图形，再执行文本分割。", 4000)
            return False
        if (getattr(self.mainwindow, "annotation_mode", "Frame") != "Frame"
                or self.mainwindow.map_mode != MAPMode.LABEL
                or not self.mainwindow.can_be_annotated):
            return False
        if self._worker is not None:
            return False
        segany = getattr(self.mainwindow, "segany", None)
        if (
            not self.mainwindow.use_segment_anything
            or segany is None
            or segany.model_source not in ("sam3", "sam3.1")
        ):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Only SAM3/SAM3.1 text models are supported."
            )
            return False
        if self.mainwindow.current_index is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image is open.")
            return False
        if (
            segany.model_source == "sam3.1"
            and self.mainwindow.current_index
            not in self.mainwindow.seganythread.results_dict
        ):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Wait for SAM image encoding to finish."
            )
            return False
        return True

    def predict_for_single_category(self):
        category = self.mainwindow.current_category
        prompt = self.lineEdit_prompt_text_for_single_category.text().strip()
        if category == "__background__":
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Select a foreground category first."
            )
            return
        if not prompt:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Enter a text prompt first."
            )
            return
        self._start_prediction([(category, prompt)])

    def predict_for_multi_categories(self):
        requests = [
            (category, category)
            for category in self.mainwindow.category_color_dict
            if category != "__background__"
        ]
        self._start_prediction(requests)

    def _start_prediction(self, requests):
        if not requests or not self._validate_ready():
            return
        index = self.mainwindow.current_index
        filename = self.mainwindow.files_list[index]
        image_path = os.path.join(self.mainwindow.image_root, filename)
        self._job_id += 1
        job_id = self._job_id
        self._job_context = (
            job_id,
            getattr(self.mainwindow, "sam_model_generation", 0),
            index,
            filename,
            os.path.normcase(os.path.abspath(image_path)),
        )
        self._set_busy(True, len(requests))
        self._worker = TextPromptThread(
            job_id,
            self.mainwindow.segany,
            image_path,
            requests,
            self,
            cached_state=(
                self.mainwindow.seganythread.results_dict.get(index, {}).get("features")
                if self.mainwindow.segany.model_source == "sam3.1" else None
            ),
        )
        self._worker.resultReady.connect(self._prediction_finished)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _context_is_current(self, job_id):
        if self._job_context is None or self._job_context[0] != job_id:
            return False
        _, generation, index, filename, image_path = self._job_context
        return (
            generation == getattr(self.mainwindow, "sam_model_generation", 0)
            and index == self.mainwindow.current_index
            and index is not None
            and 0 <= index < len(self.mainwindow.files_list)
            and filename == self.mainwindow.files_list[index]
            and image_path == os.path.normcase(os.path.abspath(os.path.join(
                self.mainwindow.image_root, self.mainwindow.files_list[index]
            )))
            and getattr(self.mainwindow, "annotation_mode", "Frame") == "Frame"
            and self.mainwindow.scene.mode == STATUSMode.VIEW
            and self.mainwindow.map_mode == MAPMode.LABEL
            and self.mainwindow.can_be_annotated
        )

    @QtCore.pyqtSlot(int, object, str)
    def _prediction_finished(self, job_id, results, error):
        if not self._context_is_current(job_id):
            return
        if error:
            QtWidgets.QMessageBox.warning(self, "Text segmentation failed", error)
            self.label_num_objects.setText("0")
            return
        created = 0
        for position, (category, masks, scores) in enumerate(results, 1):
            created += self.mainwindow._process_sam_masks(
                masks, scores, category, notify=False
            )
            self.progressBar.setValue(position)
        if created:
            self.mainwindow.set_saved_state(False)
            self.mainwindow.plugin_manager_dialog.trigger_after_annotation_created()
        self.label_num_objects.setText(str(created))

    @QtCore.pyqtSlot()
    def _worker_finished(self):
        worker = self._worker
        self._worker = None
        self._job_context = None
        self._set_busy(False, 0)
        self.mainwindow.SeganyEnabled()
        if worker is not None:
            worker.deleteLater()

    def _set_busy(self, busy, total):
        self.pushButton_predict_for_single_category.setEnabled(not busy)
        self.pushButton_predict_for_multi_categoriies.setEnabled(not busy)
        self.pushButton_catgories_setting.setEnabled(not busy)
        self.mainwindow.categories_dock_widget.pushButton_segment.setEnabled(not busy)
        self.checkBox_use_selected_category.setEnabled(not busy)
        self.comboBox_prompt.setEnabled(not busy and not self.checkBox_use_selected_category.isChecked())
        if busy:
            self.mainwindow.actionSegment_anything_point.setEnabled(False)
            self.mainwindow.actionSegment_anything_box.setEnabled(False)
        self.progressBar.setMaximum(max(total, 1))
        self.progressBar.setValue(0)
        if busy:
            self.label_num_objects.setText("......")
