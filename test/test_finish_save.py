"""Finish persists completed shapes, including video-frame links and failed writes."""

import json
from pathlib import Path
from unittest import mock

from PyQt5 import QtCore, QtWidgets

from IVAS.configs import STATUSMode
from test import test_ui_mode_boundaries as fixtures


class FinishSaveTests(fixtures.WorkbenchFixture):
    def _start_polygon(self):
        self.window.current_category = "lens"
        self.window.actionPolygon.trigger()
        self.assertEqual(self.window.scene.mode, STATUSMode.CREATE)
        # Three anchors plus the trailing mouse position that Finish removes.
        for x, y in ((8, 8), (32, 8), (32, 32), (32, 32)):
            self.window.scene.current_graph.addPoint(QtCore.QPointF(x, y))

    def _click_finish(self):
        button = self.window.toolBar.widgetForAction(self.window.actionFinish)
        self.assertTrue(button.isEnabled())
        button.click()

    def _saved_objects(self):
        label_path = Path(self.window.current_label.label_path)
        return json.loads(label_path.read_text(encoding="utf-8"))["objects"]

    def test_finish_saves_each_polygon_once_with_either_auto_save_setting(self):
        self._load_image()
        for count, auto_save in enumerate((False, True), start=1):
            with self.subTest(auto_save=auto_save):
                self.window.cfg["software"]["auto_save"] = auto_save
                self._start_polygon()
                with mock.patch.object(self.window, "save", wraps=self.window.save) as save:
                    self._click_finish()
                save.assert_called_once_with()
                objects = self._saved_objects()
                self.assertEqual(len(objects), count)
                self.assertEqual(objects[-1]["category"], "lens")
                self.assertEqual(objects[-1]["segmentation"], [[8, 8], [32, 8], [32, 32]])
                self.assertTrue(self.window.saved)
                self.assertFalse(self.window.windowTitle().startswith("*"))
                self.assertEqual(self.window.cfg["software"]["auto_save"], auto_save)
                self.assertEqual(self.window.scene.mode, STATUSMode.VIEW)

    def test_invalid_finish_and_cancel_do_not_save(self):
        self._load_image()
        label_path = Path(self.window.current_label.label_path)
        with mock.patch.object(self.window, "save", wraps=self.window.save) as save:
            self.window.actionPolygon.trigger()
            self.window.scene.current_graph.addPoint(QtCore.QPointF(8, 8))
            self._click_finish()
            self._start_polygon()
            self.window.actionCancel.trigger()
        save.assert_not_called()
        self.assertFalse(label_path.exists())
        self.assertEqual(self.window.polygons, [])

    def test_failed_finish_save_retains_previous_file_and_can_be_retried(self):
        self._load_image()
        self._start_polygon()
        self._click_finish()
        label_path = Path(self.window.current_label.label_path)
        original = label_path.read_bytes()
        self._start_polygon()
        with mock.patch("IVAS.annotation.os.replace", side_effect=OSError("disk full")), \
             mock.patch.object(QtWidgets.QMessageBox, "critical") as error:
            self._click_finish()
        error.assert_called_once()
        self.assertIn("disk full", error.call_args.args[2])
        self.assertFalse(self.window.saved)
        self.assertTrue(self.window.windowTitle().startswith("*"))
        self.assertEqual(label_path.read_bytes(), original)
        self.assertEqual(len(self.window.polygons), 2)
        self.window.actionSave.trigger()
        self.assertTrue(self.window.saved)
        self.assertEqual(len(self._saved_objects()), 2)

    def test_finish_saves_video_frame_and_its_project_link(self):
        workspace = self._video()
        self.window.request_frame_mode()
        self._wait(lambda: self.window.annotation_mode == "Frame" and not workspace._busy)
        self._start_polygon()
        self._click_finish()
        self.assertTrue(self.window.saved)
        self.assertEqual(len(self._saved_objects()), 1)
        project = json.loads(workspace.store.path.read_text(encoding="utf-8"))
        self.assertEqual(project["spatial_frames"][0]["annotation_status"], "saved")
        self.window.actionReturn_to_video.trigger()
        self.assertEqual(self.window.annotation_mode, "Video")
