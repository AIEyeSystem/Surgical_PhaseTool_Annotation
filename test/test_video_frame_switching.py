"""Regressions for switching between Video preview and Frame segmentation.

``WorkspaceSideFixes`` covers fixes inside ``video_annotation_workspace.py``.
``MainWindowSideFixes`` covers the proposed ``mainwindow.py`` changes in
``tmp/video-frame-audit/mainwindow.patch``; those tests fail until that patch
is applied.  Run one class with e.g.::

    python -m unittest test.test_video_frame_switching.WorkspaceSideFixes
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS.annotation import Object
from IVAS.configs import MAPMode
from IVAS.widgets.polygon import Polygon

PROJECT_TMP = Path(__file__).resolve().parents[1] / "tmp" / "test-video-frame-switching"


class _FakeVlcPlayer:
    """Minimal libVLC stand-in whose clock follows set_time()."""

    def __init__(self, time_ms):
        self.time_ms = int(time_ms)
        self.set_time_calls = []

    def get_time(self):
        return self.time_ms

    def set_time(self, time_ms):
        self.time_ms = int(time_ms)
        self.set_time_calls.append(int(time_ms))

    def set_pause(self, _value):
        pass

    def is_playing(self):
        return False

    def play(self):
        pass

    def pause(self):
        pass

    def set_rate(self, _rate):
        pass

    def stop(self):
        pass

    def release(self):
        pass


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from IVAS.widgets import mainwindow

        patcher = mock.patch.object(mainwindow.CheckLatestVersionThread, "start")
        patcher.start()
        self.addCleanup(patcher.stop)
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=PROJECT_TMP)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.window = mainwindow.MainWindow()
        # Never overwrite the user's YAML configuration from tests.
        self.window.save_cfg = mock.Mock()
        self.window.save_software_cfg = mock.Mock()
        self.window.cfg["software"]["auto_save"] = False
        self.addCleanup(self._close_window)

    def _close_window(self):
        window = self.window
        window.saved = True
        window.use_segment_anything = False
        window.scene.cancel_draw()
        workspace = window.video_annotation_workspace
        if workspace is not None:
            workspace._clear_pending_interval()
            workspace.vlc_player = None
            workspace.shutdown()
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def _wait(self, predicate, message="Timed out"):
        timer = QtCore.QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < 8000:
            QtTest.QTest.qWait(20)
        self.app.processEvents()
        self.assertTrue(predicate(), message)

    def _images(self, name, count=1):
        directory = self.root / name
        directory.mkdir()
        for index in range(count):
            Image.new("RGB", (64, 48), "white").save(directory / "img{}.png".format(index))
        return directory

    def _make_video(self, name="case.avi"):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 15, dtype=np.uint8))
        writer.release()
        return path

    def _video(self, name="case.avi"):
        path = self._make_video(name)
        workspace = self.window._ensure_video_workspace()
        workspace.output_root = self.root / "projects"
        if workspace.player is not None:
            workspace.player.stop()
            workspace.player = None
        workspace._load_player_media = mock.Mock()
        workspace.open_video(path)
        self._wait(lambda: not workspace._background_work_running())
        self.window._activate_video_workspace()
        return workspace

    def _annotate_at(self, workspace, position_ms):
        if self.window.annotation_mode != "Video":
            self.window.show_video_workspace()
        workspace.seek(position_ms)
        self.window.request_frame_mode()
        self._wait(lambda: self.window.annotation_mode == "Frame" and not workspace._busy,
                   "Video frame handoff timed out")

    def _capture_keyframe_synchronously(self, workspace, interrupt):
        """Start a key-frame capture, run ``interrupt``, then finish the job."""
        from IVAS.widgets import video_annotation_workspace as module

        with mock.patch.object(module.FrameExtractionWorker, "start"):
            self.assertTrue(workspace.capture_current_keyframe())
            worker = workspace._extraction_worker
            interrupt()
            worker.run()
            worker.finished.emit()
        self._wait(lambda: not workspace._busy)
        return worker

    def _add_polygon(self):
        polygon = Polygon()
        self.window.scene.addItem(polygon)
        polygon.load_object(Object("lens", 1, [[8, 8], [32, 8], [32, 32]], 288, 1, [8, 8, 32, 32]))
        self.window.polygons.append(polygon)
        return polygon

    def _fake_segany(self):
        import torch

        return types.SimpleNamespace(
            device="cpu", checkpoint="checkpoints/sam_vit_b.pth", model_dtype=torch.float32,
            model_source="sam", inference_lock=threading.RLock(),
        )


class WorkspaceSideFixes(_Fixture):
    def test_open_project_discard_clears_pending_interval(self):
        workspace = self._video("a.avi")
        workspace.open_video(self._make_video("b.avi"))
        project_b = workspace.store.path
        self._wait(lambda: not workspace._background_work_running())
        workspace.open_video(self.root / "a.avi")
        self._wait(lambda: not workspace._background_work_running())
        workspace.label_combo.setCurrentText("Preparation")
        workspace.seek(700)
        workspace.mark_start()

        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName",
                               return_value=(str(project_b), "")), \
             mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Discard) as asked:
            self.window.open_video_project_manifest()

        asked.assert_called_once()
        self.assertEqual(workspace.store.path, project_b)
        self.assertIsNone(workspace._pending_interval)
        workspace.seek(900)
        with mock.patch.object(QtWidgets.QMessageBox, "information"):
            workspace.mark_end()
        self.assertEqual(workspace.store.temporal_annotations, [])

    def test_open_project_cancel_keeps_case_and_pending_interval(self):
        workspace = self._video("a.avi")
        workspace.open_video(self._make_video("b.avi"))
        project_b = workspace.store.path
        self._wait(lambda: not workspace._background_work_running())
        store_a = workspace.open_video(self.root / "a.avi")
        self._wait(lambda: not workspace._background_work_running())
        workspace.label_combo.setCurrentText("Preparation")
        workspace.mark_start()
        pending = workspace._pending_interval

        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName",
                               return_value=(str(project_b), "")), \
             mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Cancel):
            self.window.open_video_project_manifest()

        self.assertIs(workspace.store, store_a)
        self.assertEqual(workspace._pending_interval, pending)

    def test_media_relink_retry_asks_about_pending_interval_once(self):
        workspace = self._video("a.avi")
        workspace.open_video(self._make_video("b.avi"))
        project_b = workspace.store.path
        self._wait(lambda: not workspace._background_work_running())
        workspace.open_video(self.root / "a.avi")
        self._wait(lambda: not workspace._background_work_running())
        moved = self.root / "moved"
        moved.mkdir()
        shutil.move(str(self.root / "b.avi"), str(moved / "b.avi"))
        workspace.label_combo.setCurrentText("Preparation")
        workspace.mark_start()

        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName",
                               side_effect=[(str(project_b), ""), (str(moved / "b.avi"), "")]), \
             mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Discard) as asked, \
             mock.patch.object(QtWidgets.QMessageBox, "warning",
                               return_value=QtWidgets.QMessageBox.Yes), \
             mock.patch.object(QtWidgets.QMessageBox, "critical") as failed:
            workspace.open_project_dialog()

        failed.assert_not_called()
        asked.assert_called_once()
        self.assertEqual(workspace.store.path, project_b)
        self.assertIsNone(workspace._pending_interval)

    def test_keyframe_capture_survives_media_and_mode_changes(self):
        workspace = self._video()
        standalone = self._images("standalone")
        workspace.seek(500)
        worker = self._capture_keyframe_synchronously(
            workspace, lambda: self.window.load_image_directory(str(standalone)))
        self.assertFalse(worker.cancel_event.is_set())
        self.assertEqual(Path(self.window.image_root), standalone)
        self.assertEqual([(item.kind, item.start_ms) for item in workspace.store.temporal_annotations],
                         [("point", 500)])

        self.window.show_video_workspace()
        workspace.seek(800)
        self._capture_keyframe_synchronously(
            workspace, self.window.actionShow_video_workspace.trigger)
        self.assertEqual(len(workspace.store.temporal_annotations), 2)

    def test_explicit_cancel_still_stops_keyframe_capture_and_reports_it(self):
        workspace = self._video()
        messages = []
        workspace.status_message.connect(messages.append)
        workspace.seek(500)
        worker = self._capture_keyframe_synchronously(
            workspace, lambda: workspace.cancel_frame_extraction(include_keyframe=True))
        self.assertTrue(worker.cancel_event.is_set())
        self.assertEqual(workspace.store.temporal_annotations, [])
        self.assertTrue(any("cancel" in message.lower() or "取消" in message for message in messages))

    def test_return_to_video_restores_playhead_during_keyframe_capture(self):
        workspace = self._video()
        self._annotate_at(workspace, 300)
        self._annotate_at(workspace, 800)
        player = _FakeVlcPlayer(800)
        workspace.vlc_player = player
        self.window.prev_image()
        self.assertEqual(workspace._frame_edit_position, 300)
        from IVAS.widgets import video_annotation_workspace as module

        with mock.patch.object(module.FrameExtractionWorker, "start"):
            self.window.actionTemporal_capture_keyframe.trigger()
            worker = workspace._extraction_worker
            self.assertTrue(workspace._busy)
            self.window.actionReturn_to_video.trigger()
            self.assertEqual(self.window.annotation_mode, "Video")
            self.assertEqual(player.set_time_calls, [300])
            worker.run()
            worker.finished.emit()
        self._wait(lambda: not workspace._busy)
        self.assertEqual(workspace.position_ms(), 300)
        self.assertEqual(workspace.store.temporal_annotations[-1].start_ms, 300)

    def test_linked_frame_filenames_exclude_keyframe_captures(self):
        workspace = self._video()
        workspace.seek(800)
        self._capture_keyframe_synchronously(workspace, lambda: None)
        self._annotate_at(workspace, 300)
        on_disk = sorted(path.name for path in workspace.store.frames_directory.glob("*.png"))
        self.assertEqual(on_disk, ["frame_000000000003.png", "frame_000000000008.png"])
        self.assertEqual(workspace.linked_frame_filenames(workspace.store.frames_directory),
                         ["frame_000000000003.png"])
        self.assertEqual(workspace.linked_frame_filenames(self.root), [])


class MainWindowSideFixes(_Fixture):
    """Requires tmp/video-frame-audit/mainwindow.patch to be applied."""

    def test_video_case_lists_only_linked_frames(self):
        workspace = self._video()
        workspace.seek(800)
        self._capture_keyframe_synchronously(workspace, lambda: None)
        self._annotate_at(workspace, 300)
        self.assertEqual(self.window.files_list, ["frame_000000000003.png"])
        self.assertFalse(self.window.actionNext_image.isEnabled())
        self._add_polygon()
        self.window.saved = False
        with mock.patch.object(QtWidgets.QMessageBox, "critical") as failed:
            self.assertIsNot(self.window.save_current_work(), False)
        failed.assert_not_called()
        self.assertTrue(self.window.saved)

    def test_sam_loaded_in_video_encodes_frame_on_return(self):
        from IVAS.widgets import mainwindow

        workspace = self._video()
        self._annotate_at(workspace, 300)
        self.window.show_video_workspace()
        requests = []
        self.window.segany = self._fake_segany()
        with mock.patch.object(mainwindow.SegAnyThread, "request_image",
                               lambda thread, index: requests.append(index)):
            self.window.init_sam_finish(True, False)
            self.assertEqual(requests, [])
            self.window.request_frame_mode()
            self._wait(lambda: self.window.annotation_mode == "Frame" and not workspace._busy)
        self.assertEqual(requests, [self.window.current_index])

    def test_settings_change_in_video_defers_canvas_reload(self):
        window = self.window
        window.load_image_directory(str(self._images("standalone", count=3)))
        self._add_polygon()
        window.save()
        window.show_image(0)
        window.change_bit_map_to_semantic()
        window.show_video_workspace()
        title = window.windowTitle()
        window.change_vertex_size(2)
        for name in ("actionNext_image", "actionPrev_image", "actionSave", "actionPolygon"):
            self.assertFalse(getattr(window, name).isEnabled(), name)
        self.assertEqual(window.windowTitle(), title)
        window.actionNext_image.trigger()
        self.assertEqual(window.current_index, 0)

        window.request_frame_mode()
        polygon = window.polygons[0]
        self.assertEqual(polygon.isEnabled(), window.map_mode == MAPMode.LABEL)
        self.assertEqual(any(vertex.isVisible() for vertex in polygon.vertices),
                         window.map_mode == MAPMode.LABEL)

    def test_cancelled_open_video_keeps_mode(self):
        workspace = self._video()
        workspace.label_combo.setCurrentText("Preparation")
        workspace.mark_start()
        standalone = self._images("standalone")
        self.window.load_image_directory(str(standalone))
        other = self._make_video("other.avi")
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName",
                               return_value=(str(other), "")), \
             mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Cancel):
            self.window.open_video_workspace()
        self.assertEqual(self.window.annotation_mode, "Frame")
        self.assertTrue(self.window.actionShow_image_workspace.isChecked())
        self.assertEqual(Path(self.window.image_root), standalone)
        self.assertEqual(workspace.video_path.name, "case.avi")
        self.assertIsNotNone(workspace._pending_interval)

    def test_language_switch_keeps_frame_title_and_unsaved_marker(self):
        window = self.window
        window.load_image_directory(str(self._images("standalone")))
        window.set_saved_state(False)
        title = window.windowTitle()
        self.assertTrue(title.startswith("*"))
        window.translate("en" if window.cfg["software"]["language"] == "zh" else "zh")
        self.assertEqual(window.windowTitle(), title)
        workspace = self._video()
        video_title = window.windowTitle()
        window.translate("zh")
        self.assertEqual(window.windowTitle(), video_title)
        self.assertIn(workspace.video_path.name, video_title)

    def test_failed_sam_load_in_video_keeps_prompt_docks_hidden_on_return(self):
        window = self.window
        window.load_image_directory(str(self._images("standalone")))
        window.text_prompt_dock.show()
        window.visual_prompt_dock.show()
        window.show_video_workspace()
        window.init_segany_thread.error_message = ""
        window.init_sam_finish(False, False)
        window.request_frame_mode()
        self.assertEqual(window.annotation_mode, "Frame")
        self.assertTrue(window.text_prompt_dock.isHidden())
        self.assertTrue(window.visual_prompt_dock.isHidden())

    def test_save_does_not_silently_overwrite_label_changed_on_disk(self):
        workspace = self._video()
        self._annotate_at(workspace, 300)
        window = self.window
        label = Path(window.current_label.label_path)
        external = {"info": {"description": "IVAS", "note": "tracking commit"}, "objects": []}
        label.write_text(json.dumps(external), encoding="utf-8")
        self._add_polygon()
        window.saved = False
        with mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Cancel) as asked, \
             mock.patch.object(QtWidgets.QMessageBox, "critical"):
            self.assertFalse(window.save_current_work())
        asked.assert_called_once()
        self.assertFalse(window.saved)
        self.assertEqual(json.loads(label.read_text(encoding="utf-8")), external)
        with mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Yes):
            window.save_current_work()
        self.assertTrue(window.saved)
        self.assertEqual(json.loads(label.read_text(encoding="utf-8"))["objects"][0]["category"], "lens")

    def test_retry_after_failed_link_does_not_report_own_write_as_external(self):
        workspace = self._video()
        self._annotate_at(workspace, 300)
        window = self.window
        self._add_polygon()
        window.saved = False
        with mock.patch.object(workspace, "mark_frame_annotation_saved", return_value=False), \
             mock.patch.object(QtWidgets.QMessageBox, "critical"):
            self.assertFalse(window.save_current_work())
        with mock.patch.object(QtWidgets.QMessageBox, "question") as asked:
            window.save_current_work()
        asked.assert_not_called()
        self.assertTrue(window.saved)


if __name__ == "__main__":
    unittest.main()
