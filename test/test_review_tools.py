import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS.configs import CONTOURMethod, CONTOURMode, load_config, save_config
from IVAS.utils.batch import video_to_frames
from test.test_ui_mode_boundaries import WorkbenchFixture


class ToolServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_config_failed_replace_preserves_previous_data(self):
        path = self.root / "settings.yaml"
        save_config({"value": "old"}, str(path))
        before = path.read_bytes()
        with mock.patch("IVAS.utils.persistence.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_config({"value": "new"}, str(path))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_config_handles_empty_and_missing_but_rejects_malformed(self):
        path = self.root / "settings.yaml"
        self.assertEqual(load_config(str(path)), {})
        path.write_text("")
        self.assertEqual(load_config(str(path)), {})
        for text in ("[1, 2]", "key: ["):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_config_import_does_not_create_user_data_directory(self):
        target = self.root / "new-data"
        result = subprocess.run([sys.executable, "-c", "import IVAS.configs"],
                                env={**os.environ, "IVAS_DATA_DIR": str(target)}, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(target.exists())

    def test_auto_segment_accepts_missing_optional_xml_and_isolates_bad_files(self):
        from IVAS.widgets.auto_segment_dialog import AutoSegmentThread
        output = self.root / "output"
        output.mkdir()
        for name in ("a-bad", "b-good"):
            Image.new("L", (8, 8)).save(self.root / (name + ".png"))
        (self.root / "a-bad.xml").write_text("<broken>")
        (self.root / "b-good.xml").write_text("<annotation><object><name>tool</name><bndbox>"
            "<xmin>1</xmin><ymin>1</ymin><xmax>6</xmax><ymax>6</ymax></bndbox></object></annotation>")
        lock = threading.RLock()
        def encode(pixels):
            self.assertTrue(lock._is_owned())
            self.assertEqual(pixels.shape, (8, 8, 3))
        def predict(**kwargs):
            self.assertTrue(lock._is_owned())
            return np.ones((8, 8), dtype=np.uint8)
        worker = AutoSegmentThread(None)
        worker.model = SimpleNamespace(inference_lock=lock, set_image=encode, predict_with_box_prompt=predict)
        worker.contour_options = (CONTOURMode.SAVE_MAX_ONLY, CONTOURMethod.SIMPLE, True)
        worker.image_dir = worker.label_dir = str(self.root)
        worker.save_dir = str(output)
        messages = []
        worker.message.connect(lambda a, b, text: messages.append(text))
        worker.run()
        self.assertFalse((output / "a-bad.json").exists())
        self.assertEqual(len(json.loads((output / "b-good.json").read_text())["objects"]), 1)
        self.assertIn("Finished: 1 error(s)", messages)

    def test_frame_export_reports_write_failure_and_releases_capture(self):
        source = self.root / "case.part.avi"
        source.write_bytes(b"fixture")
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 2
        capture.read.return_value = True, np.zeros((8, 8, 3), np.uint8)
        messages = []
        with mock.patch("IVAS.utils.batch.cv2.VideoCapture", return_value=capture), \
             mock.patch("IVAS.utils.batch.atomic_write_bytes", side_effect=[OSError("full"), None]) as write:
            result = video_to_frames(str(source), str(self.root), 1, threading.Event(), lambda *args: messages.append(args))
        self.assertEqual(result, "1 error(s)")
        self.assertEqual(write.call_args_list[1].args[0].name, "case.part_000000000001.jpg")
        capture.release.assert_called_once()


class ToolWidgetTests(WorkbenchFixture):
    def test_bad_configuration_is_backed_up_and_empty_shortcuts_work(self):
        path = self.root / "software.yaml"
        path.write_text("broken: [", encoding="utf-8")
        self.window.software_config_file = str(path)
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.window._reload_cfg_impl()
        self.assertTrue(warning.called)
        backups = list(self.root.glob("software.yaml.invalid-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "broken: [")
        self.window.cfg["shortcut"] = {}
        self.window.load_actions_shortcut()

    def test_remote_failure_clears_old_result_and_keeps_ui_responsive(self):
        from IVAS.widgets.remote_sam_dialog import RemoteSamDialog
        dialog = RemoteSamDialog(self.window, self.window)
        self.addCleanup(dialog.deleteLater)
        dialog.lineEdit_host.setText("127.0.0.1")
        dialog.lineEdit_port.setText("8000")
        entered, release = threading.Event(), threading.Event()
        def response(*args, **kwargs):
            entered.set()
            release.wait(3)
            item = mock.MagicMock()
            item.__enter__.return_value = item
            item.json.return_value = {"checkpoint": "missing fields"}
            return item
        dialog.checkBox_use_remote.setEnabled(True)
        ticks = []
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        try:
            with mock.patch("IVAS.widgets.remote_sam_dialog.requests.get", side_effect=response), \
                 mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
                dialog.check_connection()
                self._wait(entered.is_set)
                QtTest.QTest.qWait(40)
                self.assertGreater(len(ticks), 1)
                self.assertFalse(dialog.checkBox_use_remote.isEnabled())
                release.set()
                self._wait(lambda: dialog._worker is None)
                self.assertTrue(warning.called)
                self.assertFalse(dialog.checkBox_use_remote.isEnabled())
                self.assertIsNone(dialog.remote_sam_model_name)
        finally:
            release.set()
            timer.stop()
            self._wait(lambda: dialog._worker is None)

    def test_visual_prompt_cancel_drops_late_result(self):
        self._load_image()
        window, scene = self.window, self.window.scene
        dock = window.visual_prompt_dock_widget
        entered, release = threading.Event(), threading.Event()
        def predict(*args):
            entered.set()
            release.wait(3)
            return np.ones((1, 48, 64), np.uint8), [0.9]
        window.segany = SimpleNamespace(model_source="sam3", inference_lock=threading.RLock(),
                                       predictor=SimpleNamespace(predict_with_visual_prompt=predict))
        window.seganythread = SimpleNamespace(results_dict={})
        window.use_segment_anything = True
        scene.prompt_visual_items = [SimpleNamespace(points=[QtCore.QPointF(2, 2), QtCore.QPointF(8, 8)])]
        scene.prompt_visual_labels = [True]
        try:
            with mock.patch.object(window, "_process_sam_masks") as process:
                dock._predict()
                self._wait(entered.is_set)
                dock.invalidate_prediction()
                release.set()
                self._wait(lambda: dock._worker is None)
                process.assert_not_called()
        finally:
            release.set()
            scene.prompt_visual_items = []
            scene.prompt_visual_labels = []
            window.use_segment_anything = False
            self._wait(lambda: dock._worker is None)
