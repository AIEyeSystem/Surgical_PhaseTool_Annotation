"""SegAll/SegMiss liveness, empty-result reporting and visual point prompts.

The SAM doubles mirror the real call path: ``SegAny`` methods and the SAM3.1
predictor share one ``RLock`` (``SegAny.inference_lock``), and the image
encoder is the real ``SegAnyThread`` holding that lock while it encodes.
"""

import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PIL import Image
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from IVAS.configs import CONTOURMethod, CONTOURMode, STATUSMode
from IVAS.segment_any.segment_any import SegAny
from IVAS.segment_any import visual_prompt
from IVAS.utils import text_segmentation
from IVAS.widgets import mainwindow
from IVAS.widgets.text_prompt_dock_widget import empty_result_message
from test import test_category_navigation as category_fixtures


H, W = 24, 32


class LockedSam31Predictor:
    """Mimics Sam31Predictor: every entry point takes the shared model lock."""

    def __init__(self, lock):
        self._lock = lock
        self._cached_state = None
        self.encode_entered = threading.Event()
        self.encode_release = threading.Event()
        self.encode_release.set()
        self.text_results = None  # None: one 11x12 box per image
        self.calls = []

    def encode(self, image):
        with self._lock:
            self.encode_entered.set()
            assert self.encode_release.wait(10), "test never released the encoder"
            state = {"backbone_out": {}, "original_height": image.shape[0],
                     "original_width": image.shape[1], "pixel": int(image[0, 0, 0])}
            self._cached_state = state
            return state, tuple(image.shape[:2])

    def set_image(self, image):
        self.encode(image)

    def set_cached_state(self, state):
        with self._lock:
            self._cached_state = state

    def reset_image(self):
        with self._lock:
            self._cached_state = None

    def predict_with_text_prompt(self, image=None, prompt=None):
        with self._lock:
            assert self._lock._is_owned()
            state = self._cached_state
            self.calls.append((state["pixel"], prompt, threading.get_ident()))
            if self.text_results is not None:
                return self.text_results
            mask = np.zeros((1, state["original_height"], state["original_width"]), bool)
            mask[0, 3:14, 4:16] = True
            return mask, np.array([0.9], np.float32)

    def predict_with_text_prompts(self, prompts):
        with self._lock:
            return [self.predict_with_text_prompt(prompt=prompt) for prompt in prompts]


def make_sam31(predictor_factory=LockedSam31Predictor):
    return _make_sam31(predictor_factory)


def _make_sam31(predictor_factory=LockedSam31Predictor):
    """A real SegAny facade around a lock-sharing fake predictor."""
    segany = SegAny.__new__(SegAny)
    segany.inference_lock = threading.RLock()
    segany.model_source = segany.model_type = "sam3.1"
    segany.device = "cpu"
    segany.model_dtype = torch.float32
    segany.checkpoint = "sam3.1_fake.pt"
    segany.image = None
    segany.predictor = predictor_factory(segany.inference_lock)
    return segany


class GuiCase(unittest.TestCase):
    setUpClass = classmethod(category_fixtures.CategoryNavigationTests.setUpClass.__func__)
    tearDownClass = classmethod(category_fixtures.CategoryNavigationTests.tearDownClass.__func__)
    _wait = category_fixtures.CategoryNavigationTests._wait
    _select_category = category_fixtures.CategoryNavigationTests._select_category

    def setUp(self):
        category_fixtures.CategoryNavigationTests.setUp(self)

    def _close_window(self):
        dock = self.window.text_prompt_dock_widget
        model = getattr(self.window, "segany", None)
        predictor = getattr(model, "predictor", None)
        if hasattr(predictor, "encode_release"):
            predictor.encode_release.set()
        if dock._worker is not None:
            dock._worker.cancel()
            dock._worker.wait(5000)
            self.app.processEvents()
        if dock._batch_dialog is not None:
            try:
                dock._batch_dialog.reject()
            except RuntimeError:
                pass
        category_fixtures.CategoryNavigationTests._close_window(self)


class BatchLivenessTests(GuiCase):
    """SegMiss/SegAll used to sit at 0/N, uncancellable, behind a busy model."""

    def start_window(self, encoder_busy, labelled=()):
        window = self.window
        window.cfg["software"]["language"] = "en"
        for index in labelled:
            from IVAS.annotation import Annotation, Object
            annotation = Annotation(str(self.root / "{}.png".format(index)),
                                    str(self.root / "{}.json".format(index)))
            annotation.objects.append(Object("Forceps", 1, [[1, 1], [5, 1], [5, 5]], 8, 1, [1, 1, 5, 5]))
            annotation.save_annotation()
        window.segany = make_sam31()
        self.predictor = window.segany.predictor
        if encoder_busy:
            self.predictor.encode_release.clear()
        # The real encoder thread, wired like MainWindow.init_sam_finish.
        window.seganythread = mainwindow.SegAnyThread(window)
        window.seganythread.tag.connect(window.sam_encoder_finish)
        window.seganythread.finished.connect(window.SeganyEnabled)
        window.use_segment_anything = True
        self.assertTrue(window.load_image_directory(str(self.root)))
        self._select_category("Forceps")
        if encoder_busy:
            # Opening the folder queued the first frame: the encoder owns the model.
            self.assertTrue(self.predictor.encode_entered.wait(3))
            self.assertTrue(window.segany.inference_lock.acquire(blocking=False) is False)
        return window.text_prompt_dock_widget

    def test_segmiss_waiting_for_busy_encoder_is_visible_and_cancel_stops_it(self):
        dock = self.start_window(encoder_busy=True)
        self.window.categories_dock_widget.pushButton_segmiss.click()
        dialog = dock._batch_dialog
        self._wait(lambda: "waiting for the SAM model" in dialog.details.toPlainText())
        self.assertEqual(dialog.progress.value(), 0)
        dialog.reject()  # Cancel while the encoder still owns the model.
        self._wait(lambda: dock._worker is None)
        self.assertFalse(self.predictor.encode_release.is_set(), "encoder must still be busy")
        self.assertTrue(dock._batch_report["cancelled"])
        self.assertEqual(dock._batch_report["written_paths"], [])
        self.assertEqual(self.predictor.calls, [])
        self.assertFalse(dialog.running)
        self.assertTrue(dialog.cancel_button.isEnabled())
        self.assertTrue(self.window.categories_dock_widget.pushButton_segmiss.isEnabled())
        dialog.reject()
        self.assertTrue(dialog.isHidden())

    def test_segall_proceeds_as_soon_as_the_encoder_releases_the_model(self):
        dock = self.start_window(encoder_busy=True)
        self.window.categories_dock_widget.pushButton_segall.click()
        dialog = dock._batch_dialog
        self._wait(lambda: "waiting for the SAM model" in dialog.details.toPlainText())
        self.predictor.encode_release.set()
        self._wait(lambda: dock._worker is None)
        report = dock._batch_report
        self.assertEqual((report["processed"], report["objects"], report["errors"]), (4, 4, []))
        self.assertEqual([call[:2] for call in self.predictor.calls], [(i, "Forceps") for i in range(4)])
        self.assertTrue(all(call[2] != threading.get_ident() for call in self.predictor.calls))
        self.assertIn("no detections in 0", dialog.label.text())
        self.assertFalse(dialog.running)

    def test_segmiss_progress_reaches_dock_and_counts_images_without_detections(self):
        dock = self.start_window(encoder_busy=False)
        self.predictor.text_results = (np.zeros((0, H, W), bool), np.zeros((0,), np.float32))
        values = []
        dock.progressBar.valueChanged.connect(values.append)
        self.window.categories_dock_widget.pushButton_segmiss.click()
        self._wait(lambda: dock._worker is None)
        dialog = dock._batch_dialog
        self.assertEqual((dock._batch_report["empty"], dock._batch_report["empty_files"]),
                         (4, ["0.png", "1.png", "2.png", "3.png"]))
        self.assertIn("no detections in 4", dialog.label.text())
        self.assertIn("No detections:\n0.png", dialog.details.toPlainText())
        self.assertEqual(max(values), 4)  # the dock bar follows the batch too

    def test_dialog_is_released_even_if_post_batch_refresh_fails(self):
        # The current image already has the category, so SegMiss leaves it alone.
        dock = self.start_window(encoder_busy=False, labelled=(0,))
        self._wait(lambda: self.window.current_index in self.window.seganythread.results_dict)
        with mock.patch.object(self.window, "SeganyEnabled", side_effect=RuntimeError("refresh failed")),              mock.patch.object(QtWidgets.QMessageBox, "warning"):
            self.window.categories_dock_widget.pushButton_segmiss.click()
            self._wait(lambda: dock._worker is None)
        dialog = dock._batch_dialog
        self.assertFalse(dialog.running)
        self.assertEqual((dock._batch_report["processed"], dock._batch_report["skipped"]), (3, 1))
        self.assertIn("refresh failed", dock._batch_report["errors"])
        self.assertTrue(self.window.categories_dock_widget.pushButton_segall.isEnabled())
        dialog.reject()
        self.assertTrue(dialog.isHidden())


class SlowSam31Predictor(LockedSam31Predictor):
    """Encoding takes real time, like SAM3.1 on a GPU."""

    def __init__(self, lock):
        super().__init__(lock)
        self.encodes = []

    def encode(self, image):
        with self._lock:
            self.encodes.append(int(image[0, 0, 0]))
            time.sleep(0.25)
            return super().encode(image)


class DeferredSegmentTests(GuiCase):
    """Segment pressed while the image is still encoding runs when it is ready.

    User report: after cancelling SegMiss, Segment on another image (and on
    images after a SegAll) answered "Wait for SAM image encoding to finish".
    """

    start_window = BatchLivenessTests.start_window

    def start_slow_window(self):
        for index in range(4, 9):
            Image.new("RGB", (W, H), (index, index, index)).save(self.root / f"{index}.png")
        with mock.patch(__name__ + ".make_sam31",
                        lambda factory=SlowSam31Predictor: _make_sam31(factory)):
            dock = self.start_window(encoder_busy=False)
        self._wait(lambda: 0 in self.window.seganythread.results_dict)
        return dock

    def press_segment(self):
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.window.categories_dock_widget.pushButton_segment.click()
        self.assertFalse(warning.called, warning.call_args)

    def test_segment_after_cancelled_segmiss_runs_once_encoded(self):
        dock = self.start_slow_window()
        window = self.window
        window.categories_dock_widget.pushButton_segmiss.click()
        dialog = dock._batch_dialog
        self._wait(lambda: dialog.progress.value() >= 2)
        dialog.reject()
        self._wait(lambda: dock._worker is None)
        dialog.reject()

        window.show_image(6)
        self.assertNotIn(6, window.seganythread.results_dict)
        before = len(self.predictor.calls)
        self.press_segment()  # immediately, while 6 is still encoding
        self.assertIn("Encoding", dock.label_result_note.text())
        self._wait(lambda: len(self.predictor.calls) > before and dock._worker is None)
        self.assertEqual(self.predictor.calls[-1][:2], (6, "Forceps"))
        self.assertEqual(self.predictor.encodes.count(6), 1)
        self.assertGreater(len(window.polygons), 0)
        self.assertFalse(window.has_encode_waiter("text"))

    def test_navigating_away_drops_the_waiting_segment(self):
        dock = self.start_slow_window()
        window = self.window
        window.show_image(5)
        self.press_segment()
        self.assertTrue(window.has_encode_waiter("text"))
        window.show_image(7)
        self.assertFalse(window.has_encode_waiter("text"))
        before = len(self.predictor.calls)
        self._wait(lambda: 7 in window.seganythread.results_dict)
        QtTest.QTest.qWait(100)
        self.assertEqual(len(self.predictor.calls), before)
        self.assertEqual(dock.label_result_note.text(), "")

    def test_pressing_again_does_not_restart_the_running_encode(self):
        self.start_slow_window()
        window = self.window
        window.show_image(8)
        self._wait(lambda: 8 in self.predictor.encodes)
        for _ in range(3):
            self.press_segment()
        self._wait(lambda: self.predictor.calls and self.predictor.calls[-1][0] == 8)
        self.assertEqual(self.predictor.encodes.count(8), 1)


class SegMissScanTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp" / "sam-prompts"
        root.mkdir(parents=True, exist_ok=True)
        import tempfile
        self.directory = tempfile.TemporaryDirectory(dir=str(root))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.names = ["{}.png".format(i) for i in range(3)]
        for index, name in enumerate(self.names):
            Image.new("RGB", (W, H), (index, index, index)).save(self.root / name)
        self.model = make_sam31()

    def label(self, index, category):
        from IVAS.annotation import Annotation, Object
        annotation = Annotation(str(self.root / self.names[index]), str(self.root / "{}.json".format(index)))
        annotation.objects.append(Object(category, 1, [[1, 1], [5, 1], [5, 5]], 8, 1, [1, 1, 5, 5]))
        annotation.save_annotation()

    def run_batch(self, missing_only=True, cancelled=None):
        return text_segmentation.segment_folder_with_text_prompt(
            self.model, str(self.root), str(self.root), self.names, "Forceps", "Forceps",
            missing_only, (CONTOURMode.SAVE_EXTERNAL, CONTOURMethod.SIMPLE, True),
            cancelled or threading.Event(), lambda *args: None)

    def test_segmiss_skip_decision_never_decodes_the_image(self):
        for index in range(3):
            self.label(index, "Forceps")
        with mock.patch.object(text_segmentation, "Annotation",
                               side_effect=AssertionError("decoded a skipped image")), \
             mock.patch.object(text_segmentation, "_load_rgb",
                               side_effect=AssertionError("decoded a skipped image")):
            report = self.run_batch()
        self.assertEqual((report["skipped"], report["errors"]), (3, []))

    def test_label_has_category_matches_annotation_parsing(self):
        self.label(0, "Forceps")
        self.label(1, "Cornea")
        (self.root / "2.json").write_text(json.dumps({"info": {"description": "other"}}), encoding="utf-8")
        self.assertTrue(text_segmentation.label_has_category(self.root / "0.json", "Forceps"))
        self.assertFalse(text_segmentation.label_has_category(self.root / "1.json", "Forceps"))
        self.assertFalse(text_segmentation.label_has_category(self.root / "missing.json", "Forceps"))
        with self.assertRaises(ValueError):
            text_segmentation.label_has_category(self.root / "2.json", "Forceps")

    def test_model_lease_gives_up_promptly_when_cancelled(self):
        lock = self.model.inference_lock
        held, release = threading.Event(), threading.Event()

        def holder():
            with lock:
                held.set()
                release.wait(5)

        thread = threading.Thread(target=holder)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(2))
        cancelled, waits = threading.Event(), []
        threading.Timer(0.15, cancelled.set).start()
        started = time.monotonic()
        with text_segmentation.model_lease(self.model, cancelled, waits.append) as leased:
            self.assertFalse(leased)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(waits, [0])
        release.set()
        thread.join(2)
        with text_segmentation.model_lease(self.model, threading.Event()) as leased:
            self.assertTrue(leased)
            self.assertTrue(lock._is_owned())
        self.assertFalse(lock._is_owned())


class EmptyResultMessageTests(GuiCase):
    def prepare(self, language, results):
        window = self.window
        self.assertTrue(window.load_image_directory(str(self.root)))
        self._select_category("Forceps")
        # change_language() updates both the saved and the live UI language.
        window.cfg["software"]["language"] = language
        window._ui_language = language
        window.segany = make_sam31()
        window.segany.predictor.text_results = results
        window.seganythread = SimpleNamespace(
            results_dict={0: {"features": {"backbone_out": {}, "original_height": H,
                                           "original_width": W, "pixel": 0},
                              "original_size": (H, W), "input_size": (H, W),
                              "image_path": str(self.root / "0.png")}},
            request_image=mock.Mock(), invalidate=mock.Mock(),
            requestInterruption=mock.Mock(), wait=mock.Mock(), isRunning=lambda: False)
        window.use_segment_anything = True
        return window.text_prompt_dock_widget

    def segment(self, dock):
        self.window.categories_dock_widget.pushButton_segment.click()
        self._wait(lambda: dock._worker is None)

    def test_segment_without_masks_reports_the_prompt(self):
        dock = self.prepare("en", (np.zeros((0, H, W), bool), np.zeros((0,), np.float32)))
        self.segment(dock)
        expected = "No ‘Forceps’ objects found in this image"
        self.assertEqual(dock.label_num_objects.text(), "0")
        self.assertTrue(dock.label_result_note.isVisibleTo(dock))
        self.assertEqual(dock.label_result_note.text(), expected)
        self.assertEqual(self.window.statusbar.currentMessage(), expected)
        self.assertEqual(self.window.polygons, [])
        dock.invalidate_prediction()  # navigation clears the note
        self.assertFalse(dock.label_result_note.isVisibleTo(dock))

    def test_segment_without_masks_reports_in_chinese(self):
        dock = self.prepare("zh", (np.zeros((0, H, W), bool), np.zeros((0,), np.float32)))
        self.segment(dock)
        self.assertEqual(dock.label_result_note.text(), "未在当前图像中找到「Forceps」")

    def test_found_objects_do_not_show_the_note(self):
        dock = self.prepare("en", None)
        self.segment(dock)
        self.assertEqual(dock.label_num_objects.text(), "1")
        self.assertFalse(dock.label_result_note.isVisibleTo(dock))

    def test_message_lists_empty_prompts_after_partial_success(self):
        self.assertEqual(empty_result_message(["a", "b"], False, 2),
                         "Added 2 object(s). No ‘a’, ‘b’ objects found in this image")
        self.assertEqual(empty_result_message(["a"], True, 1), "新增 1 个目标；未在当前图像中找到「a」")

    def test_refused_segmentation_is_announced(self):
        dock = self.prepare("en", None)
        self.window.scene.mode = STATUSMode.CREATE
        try:
            self.window.categories_dock_widget.pushButton_segmiss.click()
            self.assertIsNone(dock._worker)
            self.assertIn("Finish or cancel", self.window.statusbar.currentMessage())
        finally:
            self.window.scene.mode = STATUSMode.VIEW


# --------------------------------------------------------------- visual points

OBJECTS = [(4, 4, 12, 12), (18, 4, 26, 12), (4, 14, 12, 22)]  # three similar squares


def rect_mask(rect, height=H, width=W):
    x0, y0, x1, y1 = rect
    mask = np.zeros((height, width), bool)
    mask[y0:y1, x0:x1] = True
    return mask


class ConceptPredictor(LockedSam31Predictor):
    """Fake SAM3.1: clicks select the square under them; exemplars find squares.

    The concept detector returns every square except those matching a
    negative exemplar box, and nothing without a positive exemplar.
    """

    def __init__(self, lock):
        super().__init__(lock)
        self.objects = list(OBJECTS)
        self.point_calls, self.visual_calls = [], []
        self._cached_state = {"backbone_out": {}, "original_height": H, "original_width": W}

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None,
                multimask_output=True, **kwargs):
        with self._lock:
            assert self._lock._is_owned()
            (x, y), label = np.asarray(point_coords)[0], int(np.asarray(point_labels)[0])
            self.point_calls.append(((float(x), float(y)), label))
            mask = np.zeros((H, W), bool)
            for rect in self.objects:
                if rect[0] <= x < rect[2] and rect[1] <= y < rect[3]:
                    mask = rect_mask(rect)
            return mask[None], np.array([0.8], np.float32), mask[None].astype(np.float32)

    def predict_with_visual_prompt(self, image, boxes, labels):
        with self._lock:
            assert self._lock._is_owned()
            self.visual_calls.append(([list(map(float, b)) for b in boxes], list(labels)))
            if not any(labels):
                return np.zeros((0, H, W), bool), np.zeros((0,), np.float32)
            negatives = [tuple(map(float, b)) for b, label in zip(boxes, labels) if not label]
            found = [rect for rect in self.objects if tuple(map(float, rect)) not in negatives]
            return (np.stack([rect_mask(r) for r in found]) if found else np.zeros((0, H, W), bool),
                    torch.full((len(found),), 0.9))


class VisualPromptHelperTests(unittest.TestCase):
    def test_mask_bbox_uses_pixel_edges(self):
        self.assertEqual(visual_prompt.mask_bbox(rect_mask((4, 5, 12, 9))), [4.0, 5.0, 12.0, 9.0])
        self.assertIsNone(visual_prompt.mask_bbox(np.zeros((H, W), bool)))

    def test_point_exemplars_query_positive_clicks_and_keep_polarity(self):
        queried = []

        def instance(x, y):
            queried.append((x, y))
            return (rect_mask(OBJECTS[0]) if x < 13 else np.zeros((H, W), bool)), 0.7

        boxes, labels, instances = visual_prompt.point_exemplars(
            instance, [[6, 6], [30, 2]], [False, True])
        self.assertEqual(queried, [(6.0, 6.0), (30.0, 2.0)])
        self.assertEqual((boxes, labels), ([[4.0, 4.0, 12.0, 12.0]], [False]))
        self.assertIsNone(instances[1][0])

    def test_merge_drops_negative_hits_and_adds_missed_positive_instances(self):
        detected = np.stack([rect_mask(OBJECTS[0]), rect_mask(OBJECTS[1])])
        instances = [(rect_mask(OBJECTS[2]), 0.8, (6.0, 16.0), True),   # missed object
                     (rect_mask(OBJECTS[1]), 0.8, (20.0, 6.0), False)]  # excluded object
        masks, scores = visual_prompt.merge_point_results(detected, [0.9, 0.9], instances)
        self.assertEqual([visual_prompt.mask_bbox(mask) for mask in masks],
                         [[4.0, 4.0, 12.0, 12.0], [4.0, 14.0, 12.0, 22.0]])
        self.assertEqual(scores.tolist(), [pytest_approx(0.9), pytest_approx(0.8)])


def pytest_approx(value):
    return float(np.float32(value))


class SegAnyVisualPromptTests(unittest.TestCase):
    def setUp(self):
        self.segany = make_sam31(ConceptPredictor)
        self.predictor = self.segany.predictor

    def test_positive_point_becomes_an_exemplar_box_and_finds_all_similar_objects(self):
        masks, scores = self.segany.predict_with_visual_prompt(None, [], [], [[6, 6]], [True])
        self.assertEqual(self.predictor.point_calls, [((6.0, 6.0), 1)])
        self.assertEqual(self.predictor.visual_calls, [([[4.0, 4.0, 12.0, 12.0]], [True])])
        self.assertEqual(masks.shape, (3, H, W))
        self.assertEqual(scores.dtype, np.float32)

    def test_negative_point_is_a_negative_exemplar_and_removes_its_object(self):
        masks, _ = self.segany.predict_with_visual_prompt(
            None, [[4, 14, 12, 22]], [True], [[20, 6]], [False])
        # The click is resolved as a *positive* instance query, used as a negative exemplar.
        self.assertEqual(self.predictor.point_calls, [((20.0, 6.0), 1)])
        self.assertEqual(self.predictor.visual_calls,
                         [([[4.0, 14.0, 12.0, 22.0], [18.0, 4.0, 26.0, 12.0]], [True, False])])
        self.assertEqual(sorted(visual_prompt.mask_bbox(m) for m in masks),
                         [[4.0, 4.0, 12.0, 12.0], [4.0, 14.0, 12.0, 22.0]])

    def test_positive_click_is_kept_even_if_the_detector_misses_it(self):
        self.predictor.predict_with_visual_prompt = lambda image, boxes, labels: (
            np.zeros((0, H, W), bool), np.zeros((0,), np.float32))
        masks, scores = self.segany.predict_with_visual_prompt(None, [], [], [[6, 6]], [True])
        self.assertEqual([visual_prompt.mask_bbox(m) for m in masks], [[4.0, 4.0, 12.0, 12.0]])

    def test_points_on_background_skip_the_detector(self):
        masks, _ = self.segany.predict_with_visual_prompt(None, [], [], [[30, 22]], [True])
        self.assertEqual(self.predictor.visual_calls, [])
        self.assertEqual(masks.shape, (0, H, W))

    def test_boxes_only_keep_the_detector_contract(self):
        masks, scores = self.segany.predict_with_visual_prompt(None, [[4, 4, 12, 12]], [True])
        self.assertEqual(self.predictor.point_calls, [])
        self.assertEqual((masks.shape, scores.shape), ((3, H, W), (3,)))

    def test_sam3_resolves_clicks_like_the_point_tool_and_uses_the_image(self):
        self.segany.model_source = self.segany.model_type = "sam3"
        masks, _ = self.segany.predict_with_visual_prompt(
            Image.new("RGB", (W, H)), [], [], [[6, 6]], [True])
        # Multimask pass + refinement pass, exactly as SegAny.predict_with_point_prompt.
        self.assertEqual(self.predictor.point_calls, [((6.0, 6.0), 1), ((6.0, 6.0), 1)])
        self.assertEqual(self.predictor.visual_calls, [([[4.0, 4.0, 12.0, 12.0]], [True])])
        self.assertEqual(masks.shape, (3, H, W))

    def test_sam3_requires_the_image_and_rejects_unlabelled_prompts(self):
        with self.assertRaises(ValueError):
            self.segany.predict_with_visual_prompt(None, [[1, 1, 2, 2]], [])
        self.segany.model_source = "sam3"
        with self.assertRaises(ValueError):
            self.segany.predict_with_visual_prompt(None, [], [], [[1, 1]], [True])


class VisualPromptDockTests(GuiCase):
    def prepare(self):
        window = self.window
        self.assertTrue(window.load_image_directory(str(self.root)))
        self._select_category("Forceps")
        window.cfg["software"]["language"] = "en"
        window.segany = make_sam31(ConceptPredictor)
        state = window.segany.predictor._cached_state
        window.seganythread = SimpleNamespace(
            results_dict={0: {"features": state, "original_size": (H, W), "input_size": (H, W),
                              "image_path": str(self.root / "0.png")}},
            request_image=mock.Mock(), invalidate=mock.Mock(),
            requestInterruption=mock.Mock(), wait=mock.Mock(), isRunning=lambda: False)
        window.use_segment_anything = True
        dock = window.visual_prompt_dock_widget
        dock.retranslateUi(dock)
        return dock

    def set_points(self, points, labels):
        scene = self.window.scene
        if not hasattr(scene, "start_segment_anything_visual_point"):
            # Without the canvas patch the dock still reads these lists.
            scene.prompt_visual_point_positions = [list(p) for p in points]
            scene.prompt_visual_point_labels = list(labels)
            scene.prompt_visual_point_items = []
            return
        scene.start_segment_anything_visual_point(True)
        for (x, y), label in zip(points, labels):
            scene._add_visual_point(QtCore.QPointF(x, y), label)

    def predict(self, dock):
        dock.pushButton_predict.click()
        self._wait(lambda: dock._worker is None)

    def test_dock_has_point_controls_next_to_box_controls(self):
        dock = self.prepare()
        layout = dock.verticalLayout
        self.assertEqual(layout.indexOf(dock.widget_points), layout.indexOf(dock.widget) + 1)
        self.assertEqual([dock.pushButton_positive_point.text(), dock.pushButton_negative_point.text(),
                          dock.pushButton_clear.text()], ["+ Point", "− Point", "Clear"])

    def test_points_predict_all_similar_objects_with_category(self):
        dock = self.prepare()
        self.set_points([[6, 6]], [True])
        self.predict(dock)
        self.assertEqual(dock.label_num_objects.text(), "3")
        self.assertEqual([polygon.category for polygon in self.window.polygons], ["Forceps"] * 3)
        self.assertEqual(self.window.segany.predictor.point_calls, [((6.0, 6.0), 1)])
        if hasattr(self.window.scene, "start_segment_anything_visual_point"):
            self.assertEqual(self.window.scene.prompt_visual_point_positions, [])

    def test_boxes_and_points_are_combined(self):
        dock = self.prepare()
        scene = self.window.scene
        scene.start_segment_anything_visual(True)
        item = scene.prompt_visual_current_item
        item.addPoint(QtCore.QPointF(12, 22))
        item.addPoint(QtCore.QPointF(4, 14))  # dragged up-left: normalised to xyxy
        scene.prompt_visual_items.append(item)
        scene.prompt_visual_labels.append(True)
        scene.prompt_visual_current_item = None
        scene.change_mode_to_view()
        self.set_points([[20, 6]], [False])
        self.predict(dock)
        self.assertEqual(self.window.segany.predictor.visual_calls[-1],
                         ([[4.0, 14.0, 12.0, 22.0], [18.0, 4.0, 26.0, 12.0]], [True, False]))
        self.assertEqual(dock.label_num_objects.text(), "2")

    def test_no_match_is_reported(self):
        dock = self.prepare()
        self.set_points([[30, 22]], [True])  # background click: nothing to use as exemplar
        self.predict(dock)
        self.assertEqual(dock.label_num_objects.text(), "0")
        self.assertEqual(dock.label_result_note.text(), "No objects matched the visual prompt in this image.")
        self.assertEqual(self.window.statusbar.currentMessage(), dock.label_result_note.text())

    def test_predict_without_prompts_explains_what_to_do(self):
        dock = self.prepare()
        self.predict(dock)
        self.assertEqual(dock.label_result_note.text(), "Add a positive/negative box or point first.")


@unittest.skipUnless(hasattr(__import__("IVAS.widgets.canvas", fromlist=["AnnotationScene"]).AnnotationScene,
                             "start_segment_anything_visual_point"),
                     "tmp/sam-prompts/canvas.py.patch is not applied")
class VisualPointCanvasTests(GuiCase):
    def prepare(self):
        self.window.resize(900, 700)
        self.window.show()
        self.assertTrue(self.window.load_image_directory(str(self.root)))
        self.window.view.zoomfit()
        return self.window.scene

    def click(self, scene, x, y, button=QtCore.Qt.LeftButton):
        # Real mouse input through the view (scene events cannot be built in PyQt5).
        view = self.window.view
        QtTest.QTest.mouseClick(view.viewport(), button, QtCore.Qt.NoModifier,
                                view.mapFromScene(QtCore.QPointF(x, y)))

    def assertPoints(self, actual, expected):
        self.assertEqual(np.asarray(actual).shape, np.asarray(expected).shape)
        self.assertTrue(np.allclose(actual, expected, atol=0.2), (actual, expected))

    def test_left_uses_chosen_polarity_right_the_opposite_and_markers_are_visible(self):
        scene = self.prepare()
        dock = self.window.visual_prompt_dock_widget
        self.assertTrue(dock.pushButton_negative_point.isEnabled())
        dock.pushButton_negative_point.click()
        self.assertEqual((scene.mode, scene.prompt_visual_point_label), (STATUSMode.CREATE, False))
        self.click(scene, 5, 6)
        self.click(scene, 20, 8, QtCore.Qt.RightButton)
        self.assertPoints(scene.prompt_visual_point_positions, [[5, 6], [20, 8]])
        self.assertEqual(scene.prompt_visual_point_labels, [False, True])
        colors = [item.brush().color().name() for item in scene.prompt_visual_point_items]
        self.assertEqual(colors, ["#ff0000", "#00ff00"])
        self.assertTrue(all(item.isVisible() and item.scene() is scene
                            for item in scene.prompt_visual_point_items))
        scene.start_segment_anything_visual_point(True)  # switch polarity, same session
        self.click(scene, 9, 9)
        self.assertEqual(scene.prompt_visual_point_labels, [False, True, True])
        scene.backspace()
        self.assertEqual(len(scene.prompt_visual_point_items), 2)

    def test_points_survive_box_drawing_and_are_cleared_with_other_prompts(self):
        scene = self.prepare()
        scene.start_segment_anything_visual_point(True)
        self.click(scene, 5, 6)
        scene.start_segment_anything_visual(False)  # switch to a negative box
        self.click(scene, 18, 4)
        self.click(scene, 30, 20, QtCore.Qt.RightButton)  # used to raise ValueError mid-box
        self.assertEqual(len(scene.prompt_visual_point_items), 1)
        self.click(scene, 26, 12)
        self.assertEqual(scene.mode, STATUSMode.VIEW)
        self.assertEqual((len(scene.prompt_visual_items), scene.prompt_visual_point_labels), (1, [True]))
        boxes, box_labels, points, point_labels = self.window.visual_prompt_dock_widget.prompts()
        self.assertPoints(boxes, [[18, 4, 26, 12]])
        self.assertPoints(points, [[5, 6]])
        self.assertEqual((box_labels, point_labels), ([False], [True]))
        markers = list(scene.prompt_visual_point_items)
        self.window.visual_prompt_dock_widget.pushButton_clear.click()
        self.assertEqual((scene.prompt_visual_items, scene.prompt_visual_point_positions,
                          scene.prompt_visual_point_items), ([], [], []))
        self.assertTrue(all(item.scene() is None for item in markers))


if __name__ == "__main__":
    unittest.main()
