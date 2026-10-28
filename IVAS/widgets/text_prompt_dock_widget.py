# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
from contextlib import nullcontext
from functools import partial

from PIL import Image
from PyQt5 import QtCore, QtWidgets

from IVAS.ui.text_prompt_dock import Ui_Form
from IVAS.branding import CATEGORY_CONFIG_NAME, LEGACY_CATEGORY_CONFIG_NAMES
from IVAS.configs import MAPMode, STATUSMode
from IVAS.widgets.responsive import ResponsiveDialog


def _quoted_prompts(prompts, chinese):
    if chinese:
        return "".join("「{}」".format(prompt) for prompt in prompts)
    return ", ".join("‘{}’".format(prompt) for prompt in prompts)


def empty_result_message(prompts, chinese, created=0):
    """Non-blocking note for prompts that produced no mask on the current image."""
    names = _quoted_prompts(prompts, chinese)
    if chinese:
        message = "未在当前图像中找到{}".format(names)
        return "新增 {} 个目标；{}".format(created, message) if created else message
    message = "No {} objects found in this image".format(names)
    return "Added {} object(s). {}".format(created, message) if created else message


class TextPromptBatchDialog(ResponsiveDialog):
    cancelRequested = QtCore.pyqtSignal()

    # Long folders can list many images without detections; keep the dialog light.
    MAX_LISTED_EMPTY = 200

    def __init__(self, parent, category, total, missing_only, chinese):
        super().__init__(parent)
        self.chinese = chinese
        self.running = True
        self.setWindowTitle("SegMiss" if missing_only else "SegAll")
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.resize(480, 260)
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(category, self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        # Per-image work can take long (wait for the encoder, full SAM3 encode).
        # A live position and elapsed clock shows the job is alive meanwhile.
        self.status = QtWidgets.QLabel(self)
        self.status.setObjectName("batchStatus")
        layout.addWidget(self.status)
        self.details = QtWidgets.QPlainTextEdit(self)
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.buttons = QtWidgets.QDialogButtonBox(self)
        self.cancel_button = self.buttons.addButton(
            "取消" if chinese else "Cancel", QtWidgets.QDialogButtonBox.RejectRole
        )
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.finished.connect(self.deleteLater)
        self._current, self._total = 0, total
        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()
        self._clock = QtCore.QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._update_status)
        self._clock.start()
        self._update_status()

    def _update_status(self):
        seconds = self._elapsed.elapsed() // 1000
        clock = "{}:{:02d}".format(seconds // 60, seconds % 60)
        template = "第 {} / {} 张 · 已用时 {}" if self.chinese else "Image {} / {} · elapsed {}"
        shown = min(self._current + (1 if self.running else 0), self._total)
        self.status.setText(template.format(shown, self._total, clock))

    def update_progress(self, current, total, message):
        self._current, self._total = current, total
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(current)
        if self.cancel_button.isEnabled():
            self.details.setPlainText(message)
        self._update_status()

    def complete(self, summary, errors, empty_files=()):
        self.running = False
        self._clock.stop()
        self._update_status()
        self.label.setText(summary)
        lines = list(errors)
        empty_files = list(empty_files)
        if empty_files:
            if lines:
                lines.append("")
            lines.append("无检测结果：" if self.chinese else "No detections:")
            lines.extend(empty_files[:self.MAX_LISTED_EMPTY])
            if len(empty_files) > self.MAX_LISTED_EMPTY:
                lines.append("…")
        self.details.setPlainText("\n".join(lines))
        self.cancel_button.setText("关闭" if self.chinese else "Close")
        self.cancel_button.setEnabled(True)

    def reject(self):
        if self.running:
            self.cancelRequested.emit()
            self.cancel_button.setEnabled(False)
            self.details.setPlainText(
                "正在取消；等待当前推理结束，已保存的结果会保留。" if self.chinese
                else "Cancelling; waiting for inference to finish. Saved results are kept."
            )
        else:
            super().reject()

    def closeEvent(self, event):
        if self.running:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


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
            if self.isInterruptionRequested():
                return
            prompts = [prompt for _, prompt in self.requests]
            # Restore the captured frame under the same lock as prediction:
            # navigation may have requested another image since this job started.
            with getattr(self.segany, "inference_lock", nullcontext()):
                if self.isInterruptionRequested():
                    return
                if self.segany.model_source == "sam3.1":
                    self.segany.predictor.set_cached_state(self.cached_state)
                    outputs = self.segany.predict_with_text_prompts(prompts)
                else:
                    with Image.open(self.image_path) as source:
                        image = source.convert("RGB")
                    outputs = []
                    for prompt in prompts:
                        if self.isInterruptionRequested():
                            return
                        outputs.append(self.segany.predictor.predict_with_text_prompt(image, prompt))
            results = [
                (category, masks, scores)
                for (category, _), (masks, scores) in zip(
                    self.requests, outputs
                )
            ]
            if not self.isInterruptionRequested():
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
        self._batch_dialog = None
        self._batch_report = None
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
        self.pushButton_predict_for_multi_categoriies.setMinimumWidth(0)
        self.pushButton_predict_for_multi_categoriies.setMaximumWidth(16777215)
        self.pushButton_predict_for_single_category.setToolTip(
            "Confirm text-guided segmentation for the selected category."
        )
        self.pushButton_predict_for_multi_categoriies.setMaximumWidth(16777215)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # Non-blocking outcome note, e.g. when a prompt found nothing.
        self.label_result_note = QtWidgets.QLabel()
        self.label_result_note.setObjectName("promptResultNote")
        self.label_result_note.setWordWrap(True)
        self.label_result_note.setVisible(False)
        self.verticalLayout.addWidget(self.label_result_note)
        self.verticalLayout.addStretch(1)
        self._job_prompts = ()

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
        self.show_result_note("")

    def _chinese(self):
        language = getattr(self.mainwindow, "_ui_language", None)
        if language is None:
            cfg = getattr(self.mainwindow, "cfg", None) or {}
            language = (cfg.get("software") or {}).get("language")
        return language == "zh"

    def show_result_note(self, text):
        self.label_result_note.setText(text)
        self.label_result_note.setToolTip(text)
        self.label_result_note.setVisible(bool(text))

    def predict_for_selected_category(self):
        category = self.mainwindow.current_category
        if category == "__background__":
            QtWidgets.QMessageBox.warning(self, "Warning", "Select a foreground category first.")
            return
        self._start_prediction([(category, category)])

    def _refuse(self, chinese, english):
        # Refusals used to be silent, which read as "the button does nothing".
        self.mainwindow.statusbar.showMessage(chinese if self._chinese() else english, 5000)
        return False

    def _validate_ready(self, require_cached_state=True):
        if self.mainwindow.scene.mode != STATUSMode.VIEW:
            return self._refuse("请先完成或取消当前图形，再执行文本分割。",
                                "Finish or cancel the current shape before text segmentation.")
        if (getattr(self.mainwindow, "annotation_mode", "Frame") != "Frame"
                or self.mainwindow.map_mode != MAPMode.LABEL
                or not self.mainwindow.can_be_annotated):
            return self._refuse("文本分割仅适用于帧模式下可标注的图像（标注视图）。",
                                "Text segmentation needs an annotatable image in Frame mode (label view).")
        if self._worker is not None:
            return self._refuse("文本分割任务正在运行。", "A text segmentation job is already running.")
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
            require_cached_state
            and segany.model_source == "sam3.1"
            and self.mainwindow.current_index
            not in self.mainwindow.seganythread.results_dict
        ):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Wait for SAM image encoding to finish."
            )
            return False
        return True

    def predict_for_folder(self, missing_only=False):
        """Capture a folder/category and keep its writes off the GUI thread."""
        window = self.mainwindow
        category = window.current_category
        if not category or category == "__background__":
            QtWidgets.QMessageBox.warning(self, "Warning", "Select a foreground category first.")
            return
        if not self._validate_ready(require_cached_state=False):
            return
        window._drop_encode_waiters()  # a queued single-image Segment yields to the batch
        workspace = window.video_annotation_workspace
        competing = (
            getattr(window.visual_prompt_dock_widget, "_worker", None),
            getattr(workspace, "_tracking_worker", None),
            getattr(getattr(window, "auto_segment_dialog", None), "auto_segment_thread", None),
        )
        if any(worker is not None and worker.isRunning() for worker in competing):
            window.statusbar.showMessage("Wait for the current segmentation or tracking job to finish.", 5000)
            return
        # Include unsaved edits in SegMiss, and seed the label directory's config.
        try:
            if not window.saved:
                window.save()
            if not any(os.path.isfile(os.path.join(window.label_root, name))
                       for name in (CATEGORY_CONFIG_NAME, *LEGACY_CATEGORY_CONFIG_NAMES)):
                window.save_cfg(os.path.join(window.label_root, CATEGORY_CONFIG_NAME))
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(error))
            return

        from IVAS.utils.text_segmentation import segment_folder_with_text_prompt
        from IVAS.widgets.background import TaskWorker

        store = workspace.store if window._video_frame_belongs_to_active_project() else None
        index = window.current_index
        filename = window.files_list[index]
        self._job_id += 1
        self._job_context = (
            self._job_id, window.sam_model_generation, index, filename,
            os.path.normcase(os.path.abspath(os.path.join(window.image_root, filename))),
        )
        self._batch_label_root = window.label_root
        self._batch_store = store
        self._batch_report = None
        self.show_result_note("")
        filenames = tuple(window.files_list)
        chinese = self._chinese()
        self._worker = TaskWorker(partial(
            segment_folder_with_text_prompt, window.segany, window.image_root,
            window.label_root, filenames, category, category, bool(missing_only),
            (window.scene.contour_mode, window.scene.contour_method,
             window.cfg["software"]["use_polydp"]),
            project_path=store.path if store is not None else None,
            language="zh" if chinese else "en",
        ))
        self._set_busy(True, len(filenames))
        dialog = self._batch_dialog = TextPromptBatchDialog(
            window, window.category_display_name(category), len(filenames),
            missing_only, chinese,
        )
        dialog.cancelRequested.connect(self._worker.cancel)
        self._worker.progress.connect(dialog.update_progress)
        self._worker.progress.connect(self._batch_progress)
        self._worker.result.connect(self._batch_prediction_finished)
        self._worker.finished.connect(self._batch_worker_finished)
        dialog.show()
        self._worker.start()

    @staticmethod
    def _empty_report(cancelled=False):
        return dict(processed=0, skipped=0, empty=0, objects=0, cancelled=cancelled,
                    written_paths=[], errors=[], empty_files=[])

    @QtCore.pyqtSlot(int, int, str)
    def _batch_progress(self, current, total, _message):
        self.progressBar.setMaximum(max(total, 1))
        self.progressBar.setValue(current)

    @QtCore.pyqtSlot(object, str)
    def _batch_prediction_finished(self, report, error):
        self._batch_report = report or self._empty_report()
        self._batch_report.setdefault("empty_files", [])
        if error:
            self._batch_report["errors"].append(error)

    @QtCore.pyqtSlot()
    def _batch_worker_finished(self):
        """Refresh what the batch wrote, then always release the modal dialog.

        A failure while refreshing must never leave the WindowModal dialog in
        its running state: it refuses to close then and blocks the window.
        """
        window = self.mainwindow
        worker = self._worker
        if self._batch_report is None:
            cancelled = worker is not None and worker.cancel_event.is_set()
            self._batch_report = self._empty_report(cancelled)
            self._batch_report["errors"].append("The batch stopped without a report.")
        report = self._batch_report
        try:
            current = (self._context_is_current(self._job_id)
                       and window.label_root == self._batch_label_root and window.saved)
            workspace = window.video_annotation_workspace
            if self._batch_store is not None and workspace.store is self._batch_store:
                for path in report["written_paths"]:
                    if not workspace.mark_frame_annotation_saved(path):
                        report["errors"].append("Could not update the video project link: " + path)
            if current and window.current_label is not None:
                label_path = os.path.normcase(os.path.abspath(window.current_label.label_path))
                if any(os.path.normcase(os.path.abspath(path)) == label_path
                       for path in report["written_paths"]):
                    if (window.show_image(window.current_index, zoomfit=False) is False
                            or not window.load_finished):
                        report["errors"].append("Could not refresh the current image annotations.")
        except Exception as error:
            report["errors"].append(str(error))
        finally:
            try:
                self._worker_finished()
            except Exception as error:
                report["errors"].append(str(error))
            self._complete_batch(report)

    def _complete_batch(self, report):
        dialog = self._batch_dialog
        chinese = dialog.chinese if dialog is not None else self._chinese()
        state = ("已取消" if report["cancelled"] else "完成") if chinese else (
            "Cancelled" if report["cancelled"] else "Finished")
        template = ("{}：处理 {} 张，跳过 {} 张，无检测结果 {} 张，新增 {} 个轮廓，失败 {} 项。"
                    if chinese else
                    "{}: processed {}, skipped {}, no detections in {}, added {} objects, {} error(s).")
        summary = template.format(state, report["processed"], report["skipped"],
                                  report["empty"], report["objects"], len(report["errors"]))
        self.label_num_objects.setText(str(report["objects"]))
        if dialog is not None:
            try:
                dialog.complete(summary, report["errors"], report.get("empty_files", ()))
            except RuntimeError:
                pass  # The dialog was already destroyed.
        self.mainwindow.statusbar.showMessage(summary, 10000)

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

    def _needs_encoding(self):
        segany = self.mainwindow.segany
        return (segany.model_source == "sam3.1"
                and self.mainwindow.current_index not in self.mainwindow.seganythread.results_dict)

    def _start_prediction(self, requests):
        if not requests or not self._validate_ready(require_cached_state=False):
            return
        if self._needs_encoding():
            # SAM3.1 predicts from the cached encoding; wait for it and go on.
            self.show_result_note(
                "正在编码当前图像，完成后自动分割…" if self._chinese()
                else "Encoding the current image; segmentation starts when it is ready…")
            self.mainwindow.run_when_encoded("text", lambda: self._start_prediction(requests))
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
        self._job_prompts = tuple(prompt for _, prompt in requests)
        self.show_result_note("")
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
            not getattr(self.mainwindow, "_closing_requested", False)
            and generation == getattr(self.mainwindow, "sam_model_generation", 0)
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
        empty_prompts = []
        prompts = self._job_prompts if len(self._job_prompts) == len(results) else (
            tuple(category for category, _, _ in results))
        for position, ((category, masks, scores), prompt) in enumerate(zip(results, prompts), 1):
            count = self.mainwindow._process_sam_masks(
                masks, scores, category, notify=False
            )
            created += count
            if not count:
                empty_prompts.append(prompt)
            self.progressBar.setValue(position)
        if created:
            self.mainwindow.set_saved_state(False)
            self.mainwindow.plugin_manager_dialog.trigger_after_annotation_created()
        self.label_num_objects.setText(str(created))
        if empty_prompts:
            # Zero masks is a valid answer; say so instead of silently showing "0".
            message = empty_result_message(empty_prompts, self._chinese(), created)
            self.show_result_note(message)
            self.mainwindow.statusbar.showMessage(message, 8000)

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
        self.mainwindow.categories_dock_widget.pushButton_segall.setEnabled(not busy)
        self.mainwindow.categories_dock_widget.pushButton_segmiss.setEnabled(not busy)
        self.checkBox_use_selected_category.setEnabled(not busy)
        self.comboBox_prompt.setEnabled(not busy and not self.checkBox_use_selected_category.isChecked())
        if busy:
            self.mainwindow.actionSegment_anything_point.setEnabled(False)
            self.mainwindow.actionSegment_anything_box.setEnabled(False)
        self.progressBar.setMaximum(max(total, 1))
        self.progressBar.setValue(0)
        if busy:
            self.label_num_objects.setText("......")
