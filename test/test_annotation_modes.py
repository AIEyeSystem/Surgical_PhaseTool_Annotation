"""Acceptance tests for independent and mixed annotation workflows."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS.annotation import Object
from IVAS.video.project import UnifiedVideoProjectStore
from IVAS.widgets.polygon import Polygon


class AnnotationModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from IVAS.widgets import mainwindow
        patcher = mock.patch.object(mainwindow.CheckLatestVersionThread, "start")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.window = mainwindow.MainWindow()
        self.window.save_cfg = mock.Mock()
        self.window.save_software_cfg = mock.Mock()
        self.window.cfg["software"]["auto_save"] = False
        self.addCleanup(self._close_window)

    def _close_window(self):
        window = self.window
        window.saved = True
        window.scene.cancel_draw()
        workspace = window.video_annotation_workspace
        if workspace is not None:
            workspace._clear_pending_interval()
            workspace.shutdown()
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def _images(self, name):
        directory = self.root / name
        directory.mkdir()
        Image.new("RGB", (64, 48), "white").save(directory / "same.png")
        return directory

    def _wait(self, predicate):
        timer = QtCore.QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < 8000:
            QtTest.QTest.qWait(20)
        self.app.processEvents()
        self.assertTrue(predicate(), "Video frame handoff timed out")

    def _video(self):
        path = self.root / "case.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 15, dtype=np.uint8))
        writer.release()
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

    def test_mode_switch_leads_toolbar_and_switches_without_loading_a_model(self):
        window = self.window
        widgets = [window.toolBar.widgetForAction(action) for action in window.toolBar.actions()]
        # The segmented Video | Frame switch is the first control; project
        # locations are File commands, not a duplicate toolbar menu.
        self.assertIs(widgets[0], window.mode_tools_button)
        self.assertFalse(hasattr(window, "project_tools_button"))
        self.assertTrue(window.mode_segment.button("Frame").isChecked())
        self.assertEqual(window.annotation_mode, "Frame")
        window.actionShow_video_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Video")
        self.assertEqual(window.mode_tools_button.text(), "Mode: Video")
        self.assertFalse(window.actionSave.isEnabled())
        self.assertIsNone(window.video_annotation_workspace.store)
        self.assertFalse(window.use_segment_anything)
        window.actionShow_image_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Frame")
        window.translate("en")
        self.assertEqual(window.menuTemporal.title(), "Video annotation")
        self.assertEqual(window.actionAnnotate_current_video_frame.text(), "Annotate current video frame")
        window.translate("zh")
        self.assertEqual(window.actionAnnotate_current_video_frame.text(), "标注当前视频帧")
        self.assertEqual(window.mode_tools_button.text(), "Mode: Frame")

    def test_phase_frame_save_return_undo_and_reopen_share_one_case(self):
        window = self.window
        workspace = self._video()
        workspace.label_combo.setCurrentText("Preparation")
        workspace.seek(100)
        workspace.mark_start()
        pending = copy.deepcopy(workspace._pending_interval)
        workspace.seek(300)
        window.actionShow_image_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Video")
        self._wait(lambda: window.annotation_mode == "Frame" and not workspace._busy)
        self.assertEqual(workspace._pending_interval, pending)
        self.assertEqual(window.mode_tools_button.text(), "Mode: Frame")
        self.assertTrue(window.actionPolygon.isEnabled())
        self.assertTrue(window._video_frame_belongs_to_active_project())
        self.assertFalse(window.actionLabel_dir.isEnabled())
        self.assertEqual(workspace.store.project["spatial_frames"][0]["frame_index"], 3)

        polygon = Polygon()
        window.scene.addItem(polygon)
        polygon.load_object(Object("lens", 1, [[8, 8], [32, 8], [32, 32]], 288, 1, [8, 8, 32, 32]))
        window.polygons.append(polygon)
        window.saved = False
        window.save_current_work()
        saved_path = Path(window.current_label.label_path)
        saved_json = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_json["info"]["description"], "IVAS")
        self.assertEqual(saved_json["objects"][0]["category"], "lens")
        self.assertEqual(workspace.store.project["spatial_frames"][0]["annotation_status"], "saved")
        frames = copy.deepcopy(workspace.store.project["spatial_frames"])
        window.actionShow_video_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Video")
        self.assertEqual(workspace._pending_interval, pending)
        workspace.mark_end()
        self.assertEqual(len(workspace.store.temporal_annotations), 1)
        workspace.undo()
        self.assertEqual(workspace.store.temporal_annotations, [])
        self.assertEqual(workspace.store.project["spatial_frames"], frames)
        workspace.redo()
        self.assertEqual(len(workspace.store.temporal_annotations), 1)

        window.actionShow_image_workspace.trigger()
        self._wait(lambda: window.annotation_mode == "Frame" and not workspace._busy)
        self.assertEqual(workspace.store.project["spatial_frames"][0]["annotation_status"], "saved")
        self.assertEqual(len(window.polygons), 1)
        reopened = UnifiedVideoProjectStore.load(workspace.store.path)
        self.assertEqual(len(reopened.temporal_annotations), 1)
        self.assertEqual(reopened.project["spatial_frames"][0]["annotation_status"], "saved")

    def test_cancel_or_failed_save_keeps_mode_and_image_context(self):
        window = self.window
        original = self._images("original")
        other = self._images("other")
        window.load_image_directory(str(original))
        window.saved = False
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.assertFalse(window.load_image_directory(str(other)))
            window.actionShow_video_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Frame")
        self.assertTrue(window.actionShow_image_workspace.isChecked())
        self.assertEqual(Path(window.image_root), original)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save), \
             mock.patch.object(QtWidgets.QMessageBox, "critical"), \
             mock.patch.object(window, "save", side_effect=OSError("disk full")):
            window.actionShow_video_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Frame")
        self.assertFalse(window.saved)

    def test_standalone_image_disables_retained_video_actions(self):
        workspace = self._video()
        workspace.label_combo.setCurrentText("Preparation")
        workspace.mark_start()
        pending = copy.deepcopy(workspace._pending_interval)
        window = self.window
        window.load_image_directory(str(self._images("standalone")))
        self.assertEqual(window.annotation_mode, "Frame")
        self.assertFalse(window._active_video_frame_case)
        self.assertTrue(window.actionPolygon.isEnabled())
        self.assertFalse(window.actionTemporal_mark_end.isEnabled())
        self.assertFalse(window.actionTracking_start.isEnabled())
        self.assertFalse(window.temporal_dock.isEnabled())
        self.assertTrue(window.video_transport_toolbar.isHidden())
        self.assertEqual(workspace._pending_interval, pending)
        window.actionShow_video_workspace.trigger()
        self.assertEqual(window.annotation_mode, "Video")
        self.assertTrue(window.actionTemporal_mark_end.isEnabled())

    def test_failed_manifest_save_keeps_frame_unsaved(self):
        workspace = self._video()
        window = self.window
        window.request_frame_mode()
        self._wait(lambda: window.annotation_mode == "Frame" and not workspace._busy)
        window.saved = False
        with mock.patch.object(workspace, "mark_frame_annotation_saved", return_value=False), \
             mock.patch.object(QtWidgets.QMessageBox, "critical"):
            self.assertFalse(window.save_current_work())
        self.assertFalse(window.saved)
        self.assertTrue(Path(window.current_label.label_path).is_file())

    def test_old_video_frame_never_inherits_another_project(self):
        workspace = self._video()
        window = self.window
        window.request_frame_mode()
        self._wait(lambda: window.annotation_mode == "Frame" and not workspace._busy)
        workspace.store = UnifiedVideoProjectStore.create(
            self.root / "another" / "project.ivas.json", workspace.video_path
        )
        window._sync_video_context_actions()
        self.assertFalse(window._video_frame_belongs_to_active_project())
        self.assertFalse(window.actionTemporal_mark_start.isEnabled())
        with mock.patch.object(workspace, "mark_frame_annotation_saved") as linked:
            window.save()
        linked.assert_not_called()

    def test_text_prediction_rejects_same_filename_in_another_directory(self):
        window = self.window
        first, second = self._images("first"), self._images("second")
        window.load_image_directory(str(first))
        dock = window.text_prompt_dock_widget
        dock._job_context = (1, window.sam_model_generation, 0, "same.png",
                             os.path.normcase(str(first / "same.png")))
        self.assertTrue(dock._context_is_current(1))
        window.load_image_directory(str(second))
        self.assertFalse(dock._context_is_current(1))

    def test_cancelled_extraction_cannot_replace_a_standalone_image(self):
        workspace = self._video()
        self.window.request_frame_mode()
        standalone = self._images("standalone")
        self.window.load_image_directory(str(standalone))
        self._wait(lambda: not workspace._background_work_running())
        self.assertEqual(self.window.annotation_mode, "Frame")
        self.assertEqual(Path(self.window.image_root), standalone)
        self.assertFalse(self.window._active_video_frame_case)

    def test_corrupt_image_does_not_replace_current_frame(self):
        original = self._images("original")
        broken = self._images("broken")
        (broken / "same.png").write_bytes(b"corrupt image")
        self.window.load_image_directory(str(original))
        with mock.patch.object(QtWidgets.QMessageBox, "warning"):
            self.assertFalse(self.window.load_image_directory(str(broken)))
        self.assertEqual(Path(self.window.image_root), original)

    def test_failed_auto_save_retains_dirty_frame_and_edits(self):
        window = self.window
        window.load_image_directory(str(self._images("original")))
        window.cfg["software"]["auto_save"] = True
        window.info_dock_widget.lineEdit_note.setText("keep this edit")
        with mock.patch.object(window, "save", side_effect=OSError("disk full")), \
             mock.patch.object(QtWidgets.QMessageBox, "critical") as error:
            window.set_saved_state(False)
        error.assert_called_once()
        self.assertFalse(window.saved)
        self.assertEqual(window.info_dock_widget.lineEdit_note.text(), "keep this edit")

    def test_frame_canvas_shortcuts_do_not_undo_video_annotations(self):
        window = self.window
        window.show()
        workspace = self._video()
        window.request_frame_mode()
        self._wait(lambda: window.annotation_mode == "Frame" and not workspace._busy)
        window.view.setFocus()
        self.app.processEvents()
        self.assertFalse(window._video_shortcut_context_active())
        self.assertEqual(window.actionBit_map.shortcut().toString(), "Space")
        workspace.timeline.setFocus()
        self.app.processEvents()
        self.assertTrue(window._video_shortcut_context_active())
        self.assertTrue(window.actionBit_map.shortcut().isEmpty())

    def test_late_sam_encoder_cannot_enable_frame_actions_in_video(self):
        window = self.window
        window.show_video_workspace()
        window.actionSegment_anything_point.setEnabled(True)
        window.actionSegment_anything_box.setEnabled(True)
        window.SeganyEnabled()
        self.assertFalse(window.actionSegment_anything_point.isEnabled())
        self.assertFalse(window.actionSegment_anything_box.isEnabled())

    def test_video_undo_keeps_tracking_and_redo_survives_frame_save(self):
        workspace = self._video()
        store = workspace.store
        store.add_interval("phase", "Preparation", 0, 100)
        instance = store.ensure_instance("lens", "1")
        run = store.create_tracking_run("forward", 0, 2, [instance["object_id"]], "fake")
        store.undo()
        store.add_spatial_frame(0, 0, store.frames_directory / "frame.png",
                                store.labels_directory / "frame.json", "saved")
        self.assertTrue(store.redo())
        self.assertEqual(len(store.temporal_annotations), 1)
        self.assertEqual(store.project["tracking_runs"][0]["id"], run["id"])
        self.assertEqual(len(store.project["spatial_frames"]), 1)

    def test_vlc_pause_is_idempotent_and_frame_transport_stays_pinned(self):
        workspace = self._video()
        workspace.seek(300)
        workspace.label_combo.setCurrentText("Preparation")
        workspace.mark_start()
        pending = copy.deepcopy(workspace._pending_interval)
        self.window.request_frame_mode()
        self._wait(lambda: not workspace._busy)
        player = mock.Mock()
        player.get_time.return_value = 900
        workspace.vlc_player = player
        workspace.pause()
        workspace.pause()
        player.set_pause.assert_has_calls([mock.call(1), mock.call(1)])
        player.pause.assert_not_called()
        workspace.seek(900)
        workspace.timeline.seek_requested.emit(900)
        workspace._on_position_changed(900)
        player.play.assert_not_called()
        self.assertEqual(workspace.position_ms(), 300)
        self.assertEqual(workspace.timeline._position_ms, 300)
        self.assertTrue(workspace.play_button.isEnabled())
        self.assertIn(workspace.play_button.text(), ("返回视频", "Return to video"))
        self.assertFalse(self.window.actionVideo_seek_forward_1s.isEnabled())
        workspace.toggle_playback()
        self.assertEqual(self.window.annotation_mode, "Video")
        player.set_time.assert_called_once_with(300)
        self.assertTrue(workspace.play_button.isEnabled())
        self.assertEqual(workspace._pending_interval, pending)
        workspace.vlc_player = None

    def test_reopening_same_frame_preserves_canvas_and_reuses_extraction(self):
        workspace = self._video()
        workspace.seek(300)
        self.window.request_frame_mode()
        self._wait(lambda: not workspace._busy)
        original_item = self.window.scene.image_item
        self.window.view.scale(1.5, 1.5)
        transform = self.window.view.transform()
        self.window.show_video_workspace()
        with mock.patch("IVAS.widgets.video_annotation_workspace.decode_frame") as decode, \
             mock.patch("IVAS.widgets.video_annotation_workspace.atomic_write_png") as write, \
             mock.patch.object(self.window, "load_image_directory") as reload, \
             mock.patch.object(workspace.store, "add_spatial_frame") as relink:
            self.window.request_frame_mode()
            self._wait(lambda: not workspace._busy)
        decode.assert_not_called()
        write.assert_not_called()
        reload.assert_not_called()
        relink.assert_not_called()
        self.assertIs(self.window.scene.image_item, original_item)
        self.assertEqual(self.window.view.transform(), transform)
        self.assertEqual(self.window.annotation_mode, "Frame")

    def test_duplicate_extraction_and_cancel_cannot_steal_video_mode(self):
        workspace = self._video()
        with mock.patch("IVAS.widgets.video_annotation_workspace.FrameExtractionWorker.start"):
            self.window.request_frame_mode()
            worker = workspace._extraction_worker
            self.window.request_frame_mode()
            self.assertIs(workspace._extraction_worker, worker)
            self.assertFalse(workspace.play_button.isEnabled())
            self.assertFalse(self.window.actionShow_image_workspace.isEnabled())
            self.window.show_video_workspace()
            self.assertTrue(worker.cancel_event.is_set())
            worker.ready.emit(workspace._project_token, 0, 0, "missing.png", "missing.json")
            worker.finished.emit()
        self.assertEqual(self.window.annotation_mode, "Video")
        self.assertFalse(workspace._busy)
        self.assertEqual(workspace.store.project["spatial_frames"], [])

    def test_unfinished_point_prompt_blocks_mode_and_text_segmentation(self):
        self.window.load_image_directory(str(self._images("image")))
        self.window.scene.start_segment_anything()
        self.assertIsNone(self.window.scene.current_graph)
        with mock.patch.object(QtWidgets.QMessageBox, "information"):
            self.window.show_video_workspace()
        self.assertEqual(self.window.annotation_mode, "Frame")
        self.assertFalse(self.window.text_prompt_dock_widget._validate_ready())

    def test_changed_label_on_disk_is_reloaded_when_reopening_frame(self):
        workspace = self._video()
        self.window.request_frame_mode()
        self._wait(lambda: not workspace._busy)
        self.window.save()
        self.window.show_video_workspace()
        label = Path(self.window.current_label.label_path)
        payload = json.loads(label.read_text(encoding="utf-8"))
        payload["info"]["note"] = "updated by review"
        label.write_text(json.dumps(payload), encoding="utf-8")
        self.window.request_frame_mode()
        self._wait(lambda: not workspace._busy)
        self.assertEqual(self.window.info_dock_widget.lineEdit_note.text(), "updated by review")


if __name__ == "__main__":
    unittest.main()
