"""Qt responsiveness and stale-result regressions with a controllable SAM double."""
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS.configs import CONTOURMode


class InteractiveMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from IVAS.widgets.mainwindow import MainWindow, CheckLatestVersionThread
        with mock.patch.object(CheckLatestVersionThread, "start"):
            self.window = MainWindow()
        self.window._version_check_timer.stop()
        self.window.save_cfg = mock.Mock()
        self.window.save_software_cfg = mock.Mock()
        self.window.cfg["software"]["auto_save"] = False
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        Image.new("RGB", (64, 48), "white").save(self.root / "image.png")
        self.window.load_image_directory(str(self.root))
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = []
        self.states = []
        self.segany = types.SimpleNamespace(
            inference_lock=threading.RLock(), model_source="sam3.1",
            predictor=types.SimpleNamespace(set_cached_state=self.states.append),
            predict_with_point_prompt=self._predict,
            predict_with_box_prompt=lambda box: self._predict([box], [1]),
        )
        self.window.segany = self.segany
        self.window.seganythread = types.SimpleNamespace(results_dict={0: {
            "features": "frame-one", "original_size": (48, 64), "input_size": (48, 64),
            "image_path": str(self.root / "image.png"),
        }}, invalidate=mock.Mock(), request_image=mock.Mock())
        self.window.use_segment_anything = True
        self.scene = self.window.scene
        self.scene.start_segment_anything()

    def tearDown(self):
        self.gate.set()
        self.scene.cancel_draw()
        self._wait(lambda: self.scene._mask_worker is None)
        self.window.use_segment_anything = False
        self.window.saved = True
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def _wait(self, predicate):
        timer = QtCore.QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < 4000:
            QtTest.QTest.qWait(10)
        self.assertTrue(predicate(), "Mask job did not settle")

    def _predict(self, points, labels):
        self.calls.append(tuple(points))
        self.entered.set()
        if not self.gate.wait(3):
            raise TimeoutError("Test did not release predictor")
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[8:32, 8:32] = 1
        return mask

    def _point(self, x):
        self.scene.prompt_point_positions.append([x, 12])
        self.scene.prompt_point_labels.append(1)
        self.scene.update_mask()

    def test_slow_inference_keeps_qt_alive_and_coalesces_to_latest_prompts(self):
        self._point(10)
        self._wait(self.entered.is_set)
        self._point(20)
        self._point(30)
        heartbeat = []
        QtCore.QTimer.singleShot(0, lambda: heartbeat.append(True))
        self._wait(lambda: bool(heartbeat))
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.scene.mask)
        self.gate.set()
        self._wait(lambda: self.scene._mask_worker is None)
        self.assertEqual([len(points) for points in self.calls], [1, 3])
        self.assertIsNotNone(self.scene.mask)
        self.assertFalse(self.scene._mask_pending)

    def test_finish_waits_for_latest_mask_without_saving_an_old_preview(self):
        label_path = Path(self.window.current_label.label_path)
        self._point(10)
        self._wait(self.entered.is_set)
        self._point(20)
        self.scene.finish_draw()
        self.assertTrue(self.scene._finish_mask_when_ready)
        self.assertEqual(self.window.polygons, [])
        self.assertFalse(label_path.exists())
        with mock.patch.object(self.window, "save", wraps=self.window.save) as save:
            self.gate.set()
            self._wait(lambda: self.scene._mask_worker is None)
        save.assert_called_once_with()
        self.assertEqual([len(points) for points in self.calls], [1, 2])
        self.assertEqual(len(self.window.polygons), 1)
        self.assertTrue(self.window.saved)
        objects = json.loads(label_path.read_text(encoding="utf-8"))["objects"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["category"], self.window.current_category)

    def test_cancel_drops_pending_and_late_results(self):
        self._point(10)
        self._wait(self.entered.is_set)
        self._point(20)
        self.scene.finish_draw()
        self.scene.cancel_draw()
        self.gate.set()
        self._wait(lambda: self.scene._mask_worker is None)
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.scene.mask)
        self.assertEqual(self.window.polygons, [])
        self.assertFalse(Path(self.window.current_label.label_path).exists())

    def test_finish_saves_all_mask_components_in_one_write_with_auto_save_enabled(self):
        self.window.cfg["software"]["auto_save"] = True
        self.scene.contour_mode = CONTOURMode.SAVE_EXTERNAL
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        mask[25:40, 35:50] = 1
        self.segany.predict_with_point_prompt = mock.Mock(return_value=mask)
        self._point(10)
        self._wait(lambda: self.scene._mask_worker is None)
        with mock.patch.object(self.window, "save", wraps=self.window.save) as save:
            self.window.actionFinish.trigger()
        save.assert_called_once_with()
        objects = json.loads(Path(self.window.current_label.label_path).read_text(encoding="utf-8"))["objects"]
        self.assertEqual(len(objects), 2)
        self.assertTrue(self.window.saved)

    def test_old_frame_result_cannot_paint_new_image(self):
        self._point(10)
        self._wait(self.entered.is_set)
        self.scene.cancel_draw()
        other = self.root / "other"
        other.mkdir()
        Image.new("RGB", (64, 48), "black").save(other / "image.png")
        self.window.load_image_directory(str(other))
        self.gate.set()
        self._wait(lambda: self.scene._mask_worker is None)
        self.assertIsNone(self.scene.mask)
        self.assertEqual(self.window.polygons, [])
        self.assertEqual(self.window.image_root, str(other))

    def test_failed_prediction_can_be_retried_and_does_not_finish_shape(self):
        self.segany.predict_with_point_prompt = mock.Mock(side_effect=RuntimeError("inference failed"))
        self._point(10)
        self.scene.finish_draw()
        self._wait(lambda: self.scene._mask_worker is None)
        self.assertFalse(self.scene._mask_pending)
        self.assertFalse(self.scene._finish_mask_when_ready)
        self.assertEqual(self.window.polygons, [])
        self.assertIn("inference failed", self.window.statusbar.currentMessage())
        self.assertFalse(Path(self.window.current_label.label_path).exists())
        self.segany.predict_with_point_prompt = self._predict
        self.gate.set()
        self.scene.update_mask()
        self._wait(lambda: self.scene._mask_worker is None)
        self.assertIsNotNone(self.scene.mask)

    def test_box_predictions_keep_numpy_contract_and_latest_rectangle(self):
        self.scene.cancel_draw()
        self.scene.start_segment_anything_box()
        boxes = []

        def predict(box):
            self.assertIsInstance(box, np.ndarray)
            boxes.append(box.copy())
            return self._predict([box], [1])

        self.segany.predict_with_box_prompt = predict
        self.scene.prompt_box_item.addPoint(QtCore.QPointF(5, 6))
        self.scene.prompt_box_item.addPoint(QtCore.QPointF(20, 22))
        self.scene.update_mask()
        self._wait(self.entered.is_set)
        self.scene.prompt_box_item.movePoint(1, QtCore.QPointF(35, 40))
        self.scene.update_mask()
        self.gate.set()
        self._wait(lambda: self.scene._mask_worker is None)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_array_equal(boxes[-1], [5, 6, 35, 40])
        self.window.actionFinish.trigger()
        self.assertTrue(self.window.saved)
        objects = json.loads(Path(self.window.current_label.label_path).read_text(encoding="utf-8"))["objects"]
        self.assertEqual(len(objects), 1)
