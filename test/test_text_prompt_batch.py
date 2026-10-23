"""Folder prompt segmentation: selection, persistence, cancellation and real Qt wiring."""

import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS.annotation import Annotation, Object
from IVAS.configs import CONTOURMethod, CONTOURMode
from IVAS.utils.text_segmentation import segment_folder_with_text_prompt
from test import test_category_navigation as category_fixtures
from test import test_annotation_modes as mode_fixtures


class FakeTextModel:
    model_source = "sam3.1"

    def __init__(self):
        self.inference_lock = threading.RLock()
        self.calls = []
        self.thread_ids = []

    def set_image(self, image):
        assert self.inference_lock._is_owned()
        self.image = image

    def predict_with_text_prompt(self, prompt):
        assert self.inference_lock._is_owned()
        self.calls.append((int(self.image[0, 0, 0]), prompt))
        self.thread_ids.append(threading.get_ident())
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        mask[3:14, 4:16] = True
        return mask[None, None, ...], [0.9]


class TextPromptBatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        self.filenames = ["{}.png".format(i) for i in range(4)]
        for index, name in enumerate(self.filenames):
            Image.new("RGB", (32, 24), (index, index, index)).save(self.images / name)
        self.model = FakeTextModel()
        self.cancelled = threading.Event()
        self.progress = []

    def annotation(self, index, category=None):
        annotation = Annotation(str(self.images / self.filenames[index]),
                                str(self.labels / "{}.json".format(index)))
        if category:
            annotation.objects.append(Object(
                category, 7, [[1, 1], [4, 1], [4, 4]], 4.5, 10, [1, 1, 4, 4],
                note="keep object", object_id="object-1", track_id="track-1",
                source="manual", review_status="reviewed",
            ))
        annotation.note = "keep image note"
        annotation.save_annotation()
        return annotation

    def run_batch(self, missing_only=False, progress=None, contour_mode=CONTOURMode.SAVE_MAX_ONLY):
        return segment_folder_with_text_prompt(
            self.model, str(self.images), str(self.labels), self.filenames,
            "Forceps", "Forceps", missing_only,
            (contour_mode, CONTOURMethod.SIMPLE, True), self.cancelled,
            progress or (lambda *args: self.progress.append(args)),
        )

    def read(self, index):
        return json.loads((self.labels / "{}.json".format(index)).read_text(encoding="utf-8"))

    def test_segall_encodes_every_image_and_preserves_existing_objects_and_metadata(self):
        original = self.annotation(0, "Forceps")
        report = self.run_batch()
        self.assertEqual(self.model.calls, [(i, "Forceps") for i in range(4)])
        self.assertEqual((report["processed"], report["skipped"], report["objects"]), (4, 0, 4))
        data = self.read(0)
        self.assertEqual(data["objects"][0], original.objects[0].to_dict())
        self.assertEqual(data["info"]["note"], "keep image note")
        added = data["objects"][1]
        self.assertEqual((added["category"], added["group"], added["layer"]), ("Forceps", 8, 11))
        self.assertEqual((added["source"], added["review_status"]), ("sam", "draft"))
        self.assertFalse(any(self.images.glob("*.json")))
        self.assertEqual(self.progress[-1][:2], (4, 4))

    def test_segmiss_checks_the_selected_category_instead_of_label_file_existence(self):
        self.annotation(0, "Forceps")
        self.annotation(1, "Cornea")
        self.annotation(2)
        original = (self.labels / "0.json").read_bytes()
        report = self.run_batch(missing_only=True)
        self.assertEqual(self.model.calls, [(i, "Forceps") for i in (1, 2, 3)])
        self.assertEqual((report["processed"], report["skipped"]), (3, 1))
        self.assertEqual((self.labels / "0.json").read_bytes(), original)
        self.assertEqual([obj["category"] for obj in self.read(1)["objects"]], ["Cornea", "Forceps"])
        self.model.calls.clear()
        self.assertEqual(self.run_batch(missing_only=True)["skipped"], 4)
        self.assertEqual(self.model.calls, [])

    def test_corrupt_labels_and_prediction_errors_do_not_stop_later_images(self):
        broken = self.labels / "0.json"
        broken.write_text("broken json", encoding="utf-8")
        original = self.model.predict_with_text_prompt

        def predict(prompt):
            if int(self.model.image[0, 0, 0]) == 1:
                raise RuntimeError("prediction failed")
            return original(prompt)

        self.model.predict_with_text_prompt = predict
        report = self.run_batch(missing_only=True)
        self.assertEqual(len(report["errors"]), 2)
        self.assertEqual(report["processed"], 2)
        self.assertEqual(broken.read_text(), "broken json")
        self.assertFalse((self.labels / "1.json").exists())
        self.assertEqual(self.read(3)["objects"][0]["category"], "Forceps")

    def test_save_failure_preserves_label_and_continues(self):
        self.annotation(0, "Cornea")
        original = (self.labels / "0.json").read_bytes()
        replace = os.replace

        def fail_first(source, destination):
            if Path(destination).name == "0.json":
                raise OSError("disk full")
            return replace(source, destination)

        with mock.patch("IVAS.annotation.os.replace", side_effect=fail_first):
            report = self.run_batch()
        self.assertEqual((len(report["errors"]), report["processed"]), (1, 3))
        self.assertEqual((self.labels / "0.json").read_bytes(), original)
        self.assertFalse(list(self.labels.glob("*.tmp")))

    def test_cancel_during_prediction_discards_inflight_results(self):
        self.annotation(0, "Cornea")
        original = (self.labels / "0.json").read_bytes()
        predict = self.model.predict_with_text_prompt

        def cancel(prompt):
            result = predict(prompt)
            self.cancelled.set()
            return result

        self.model.predict_with_text_prompt = cancel
        report = self.run_batch()
        self.assertTrue(report["cancelled"])
        self.assertEqual(report["written_paths"], [])
        self.assertEqual((self.labels / "0.json").read_bytes(), original)
        self.assertFalse((self.labels / "1.json").exists())

    def test_cancel_between_images_keeps_completed_results(self):
        def progress(current, total, filename):
            if current == 1:
                self.cancelled.set()

        report = self.run_batch(progress=progress)
        self.assertTrue(report["cancelled"])
        self.assertEqual(report["processed"], 1)
        self.assertEqual(len(self.read(0)["objects"]), 1)
        self.assertFalse((self.labels / "1.json").exists())

    def test_merge_reloads_annotations_changed_during_inference(self):
        predict = self.model.predict_with_text_prompt

        def external_edit(prompt):
            self.annotation(int(self.model.image[0, 0, 0]), "Cornea")
            return predict(prompt)

        self.model.predict_with_text_prompt = external_edit
        self.run_batch()
        self.assertEqual([obj["category"] for obj in self.read(0)["objects"]], ["Cornea", "Forceps"])

    def test_segmiss_rechecks_category_after_inference(self):
        predict = self.model.predict_with_text_prompt

        def external_edit(prompt):
            self.annotation(int(self.model.image[0, 0, 0]), "Forceps")
            return predict(prompt)

        self.model.predict_with_text_prompt = external_edit
        report = self.run_batch(missing_only=True)
        self.assertEqual((report["processed"], report["skipped"]), (0, 4))
        self.assertEqual(len(self.read(0)["objects"]), 1)

    def test_empty_predictions_remain_eligible_for_segmiss(self):
        self.model.predict_with_text_prompt = lambda prompt: ([], [])
        report = self.run_batch(missing_only=True)
        self.assertEqual((report["processed"], report["empty"]), (4, 4))
        self.assertEqual(self.read(0)["objects"], [])
        self.assertEqual(self.run_batch(missing_only=True)["processed"], 4)

    def test_duplicate_masks_and_holes_follow_contour_settings(self):
        mask = np.zeros((24, 32), dtype=bool)
        mask[2:20, 2:28] = True
        mask[7:14, 9:18] = False
        self.model.predict_with_text_prompt = lambda prompt: (np.stack([mask, mask]), [1, 1])
        self.run_batch(contour_mode=CONTOURMode.SAVE_ALL)
        objects = self.read(0)["objects"]
        self.assertEqual([obj["category"] for obj in objects], ["Forceps", "__background__"])
        self.assertEqual([obj["group"] for obj in objects], [1, 0])

    def test_wrong_mask_dimensions_never_overwrite_existing_annotation(self):
        self.annotation(0, "Cornea")
        original = (self.labels / "0.json").read_bytes()
        self.model.predict_with_text_prompt = lambda prompt: (np.ones((8, 8)), [1])
        self.assertEqual(len(self.run_batch()["errors"]), 4)
        self.assertEqual((self.labels / "0.json").read_bytes(), original)

    def test_sam3_uses_image_prompt_api(self):
        self.model.model_source = "sam3"
        predict = mock.Mock(return_value=(np.ones((24, 32), dtype=bool), [1]))
        self.model.predictor = SimpleNamespace(predict_with_text_prompt=predict)
        self.assertEqual(self.run_batch()["processed"], 4)
        self.assertEqual(predict.call_count, 4)
        self.assertEqual(predict.call_args.args[0].mode, "RGB")
        self.assertEqual(predict.call_args.args[1], "Forceps")


class TextPromptBatchGuiTests(unittest.TestCase):
    setUpClass = classmethod(category_fixtures.CategoryNavigationTests.setUpClass.__func__)
    tearDownClass = classmethod(category_fixtures.CategoryNavigationTests.tearDownClass.__func__)
    setUp = category_fixtures.CategoryNavigationTests.setUp
    _close_window = category_fixtures.CategoryNavigationTests._close_window
    _wait = category_fixtures.CategoryNavigationTests._wait
    _select_category = category_fixtures.CategoryNavigationTests._select_category
    _add_polygon = category_fixtures.CategoryNavigationTests._add_polygon

    def prepare(self):
        window = self.window
        self.assertTrue(window.load_image_directory(str(self.root)))
        return self.install_model()

    def install_model(self):
        window = self.window
        self._select_category("Forceps")
        self.model = window.segany = FakeTextModel()
        window.seganythread = SimpleNamespace(
            results_dict={}, request_image=mock.Mock(), invalidate=mock.Mock(),
            requestInterruption=mock.Mock(), wait=mock.Mock(),
        )
        window.use_segment_anything = True
        dock = window.text_prompt_dock_widget
        self.addCleanup(self.finish_batch)
        return dock

    def finish_batch(self):
        dock = self.window.text_prompt_dock_widget
        if dock._worker is not None:
            dock._worker.cancel()
            self.assertTrue(dock._worker.wait(5000))
            self.app.processEvents()
        if dock._batch_dialog is not None:
            try:
                dock._batch_dialog.reject()
            except RuntimeError:
                pass

    def test_buttons_are_below_segment_and_segall_runs_without_cached_encoding(self):
        dock = self.prepare()
        categories = self.window.categories_dock_widget
        self.assertEqual(categories.verticalLayout.indexOf(categories.batch_actions),
                         categories.verticalLayout.indexOf(categories.category_actions) + 1)
        categories.pushButton_segall.click()
        self._wait(lambda: dock._worker is None)
        self.assertEqual(self.model.calls, [(i, "Forceps") for i in range(4)])
        self.assertTrue(all(ident != threading.get_ident() for ident in self.model.thread_ids))
        self.assertEqual(len(self.window.polygons), 1)
        self.assertTrue(self.window.saved)
        self.assertTrue(categories.pushButton_segall.isEnabled())
        self.assertTrue(categories.pushButton_segmiss.isEnabled())
        self.assertEqual(dock._batch_report["errors"], [])

    def test_segmiss_includes_unsaved_category_and_uses_canonical_name(self):
        dock = self.prepare()
        self._add_polygon("Forceps")
        self.window.saved = False
        dock.checkBox_use_selected_category.setChecked(False)
        dock.lineEdit_prompt_text_for_single_category.setText("a different manual prompt")
        self.window.categories_dock_widget.pushButton_segmiss.click()
        self._wait(lambda: dock._worker is None)
        self.assertEqual(self.model.calls, [(i, "Forceps") for i in (1, 2, 3)])
        self.assertEqual(dock._batch_report["skipped"], 1)
        data = json.loads((self.root / "0.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["objects"]), 1)
        self.assertEqual(data["objects"][0]["track_id"], "track-7")

    def test_cancel_keeps_dialog_modal_until_worker_stops_and_ui_ticks(self):
        dock = self.prepare()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        predict = self.model.predict_with_text_prompt

        def slow(prompt):
            entered.set()
            release.wait(5)
            return predict(prompt)

        self.model.predict_with_text_prompt = slow
        dock.predict_for_folder()
        self.assertTrue(entered.wait(2))
        ticks = []
        QtCore.QTimer.singleShot(0, lambda: ticks.append(True))
        QtTest.QTest.qWait(30)
        self.assertEqual(ticks, [True])
        self.assertFalse(self.window.categories_dock_widget.pushButton_segmiss.isEnabled())
        dock._batch_dialog.reject()
        self.assertTrue(dock._batch_dialog.isVisible())
        self.assertEqual(dock._batch_dialog.windowModality(), QtCore.Qt.WindowModal)
        self.assertFalse(dock._batch_dialog.cancel_button.isEnabled())
        release.set()
        self._wait(lambda: dock._worker is None)
        self.assertTrue(dock._batch_report["cancelled"])
        self.assertEqual(dock._batch_report["written_paths"], [])

    def test_failed_presave_aborts_batch(self):
        dock = self.prepare()
        self.window.saved = False
        with mock.patch.object(self.window, "save", side_effect=OSError("disk full")), \
             mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            dock.predict_for_folder()
        self.assertTrue(warning.called)
        self.assertIsNone(dock._worker)
        self.assertEqual(self.model.calls, [])

    def test_segmiss_does_not_resave_current_image_that_already_has_category(self):
        dock = self.prepare()
        self._add_polygon("Forceps")
        self.window.save()
        label = self.root / "0.json"
        original = label.read_bytes()
        with mock.patch.object(self.window, "save") as save:
            dock.predict_for_folder(missing_only=True)
            self._wait(lambda: dock._worker is None)
        save.assert_not_called()
        self.assertEqual(label.read_bytes(), original)
        self.assertEqual(dock._batch_report["skipped"], 1)

    def test_batch_marks_video_frame_annotation_saved_in_project(self):
        from IVAS.video.project import UnifiedVideoProjectStore

        workspace = mode_fixtures.AnnotationModeTests._video(self)
        self.window.request_frame_mode()
        self._wait(lambda: self.window.annotation_mode == "Frame" and not workspace._busy)
        self.assertEqual(workspace.store.project["spatial_frames"][0]["annotation_status"], "pending")
        dock = self.install_model()
        dock.predict_for_folder()
        self._wait(lambda: dock._worker is None)
        self.assertEqual(dock._batch_report["errors"], [])
        self.assertEqual(len(self.window.polygons), 1)
        saved = UnifiedVideoProjectStore.load(workspace.store.path)
        self.assertEqual(saved.project["spatial_frames"][0]["annotation_status"], "saved")


if __name__ == "__main__":
    unittest.main()
