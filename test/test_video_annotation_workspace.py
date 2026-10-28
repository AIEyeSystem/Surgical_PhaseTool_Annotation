import os
import sys
import tempfile
import types
import threading
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS.video.project import TemporalAnnotation, UnifiedVideoProjectStore
from IVAS.video.media import FrameTimestampMap, OperationCancelled
from IVAS.widgets.video_annotation_workspace import (
    TemporalTimeline,
    VideoAnnotationWorkspace,
)


class VideoAnnotationWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    @staticmethod
    def _synthetic_video(path: Path) -> bool:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        if not writer.isOpened():
            return False
        try:
            for index in range(6):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :, 1] = index * 30
                frame[8:24, 8 + index : 24 + index, 2] = 255
                writer.write(frame)
        finally:
            writer.release()
        return True

    @staticmethod
    def _disable_player(workspace: VideoAnnotationWorkspace) -> None:
        if workspace.player is not None:
            workspace.player.stop()
            workspace.player = None
        if workspace.vlc_player is not None:
            workspace.vlc_player.stop()
            workspace.vlc_player = None
        workspace._activate_vlc_fallback = lambda _path: False

    def _wait_until(self, predicate, timeout_ms=5000):
        elapsed = 0
        while not predicate() and elapsed < timeout_ms:
            self.app.processEvents()
            QtTest.QTest.qWait(20)
            elapsed += 20
        self.app.processEvents()
        self.assertTrue(predicate(), "asynchronous operation timed out")

    def test_rebuilding_index_defers_until_cancelled_worker_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "case.avi"
            self.assertTrue(self._synthetic_video(media))
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            entered, release = threading.Event(), threading.Event()
            calls = []

            def scan(path, cancel_event=None, progress=None):
                calls.append(path)
                if len(calls) == 1:
                    entered.set()
                    release.wait(3)
                if cancel_event.is_set():
                    raise OperationCancelled("cancelled")
                return FrameTimestampMap([0, 100, 200, 300, 400, 500])

            try:
                with mock.patch("IVAS.widgets.video_annotation_workspace.scan_frame_timestamps", side_effect=scan):
                    workspace.open_video(media)
                    self._wait_until(entered.is_set)
                    first = workspace._frame_map_worker
                    with mock.patch.object(first, "wait") as blocking_wait:
                        workspace._start_frame_map_scan(force=True)
                    blocking_wait.assert_not_called()
                    self.assertTrue(first.cancel_event.is_set())
                    self.assertIn(first, workspace._workers)
                    heartbeat = []
                    QtCore.QTimer.singleShot(0, lambda: heartbeat.append(True))
                    self._wait_until(lambda: bool(heartbeat))
                    release.set()
                    self._wait_until(lambda: len(calls) == 2 and not workspace._background_work_running())
                    self.assertEqual(workspace.frame_map.method, "decoded_pts")
                    self.assertEqual(len(workspace._workers), 0)
            finally:
                release.set()
                workspace.shutdown()

    def test_interval_and_current_frame_handoff_share_one_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            emitted = []
            workspace.frame_annotation_requested.connect(
                lambda image_root, label_root, filename: emitted.append(
                    (image_root, label_root, filename)
                )
            )

            store = workspace.open_video(video_path)
            workspace.label_combo.setCurrentText("Capsulorhexis")
            workspace.seek(100)
            workspace.mark_start()
            workspace.seek(300)
            workspace.mark_end()
            activated = workspace.annotation_table.item(0, 0)
            workspace.annotation_table.itemActivated.emit(activated)
            self.assertEqual(workspace.position_ms(), 100)
            self.assertEqual(
                workspace.timeline._selected_annotation_id,
                store.temporal_annotations[0].id,
            )
            workspace.seek(300)
            workspace.annotate_current_frame()
            self._wait_until(lambda: not workspace._busy)

            self.assertEqual(store.temporal_annotations[0].category, "phase")
            self.assertEqual(store.temporal_annotations[0].start_ms, 100)
            self.assertEqual(store.temporal_annotations[0].end_ms, 300)
            self.assertEqual(len(store.project["spatial_frames"]), 1)
            self.assertEqual(len(emitted), 1)
            image_root, label_root, filename = emitted[0]
            self.assertTrue((Path(image_root) / filename).is_file())
            self.assertEqual(Path(label_root), store.labels_directory)
            self.assertTrue(store.path.is_file())
            workspace.shutdown()

    def test_mainwindow_switches_from_video_to_existing_image_canvas(self):
        from IVAS.widgets import mainwindow as mainwindow_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            with mock.patch.object(
                mainwindow_module.CheckLatestVersionThread,
                "start",
                lambda thread: None,
            ):
                window = mainwindow_module.MainWindow()
            # Never write the user's IVAS/software.yaml or category file from tests.
            window.save_cfg = mock.Mock()
            window.save_software_cfg = mock.Mock()
            try:
                workspace = window._ensure_video_workspace()
                self._disable_player(workspace)
                workspace.output_root = root / "outputs"
                store = workspace.open_video(video_path)
                window.active_video_project_path = str(store.path)
                window._activate_video_workspace()
                self.assertIs(
                    window.workspace_stack.currentWidget(),
                    window.video_annotation_workspace,
                )

                workspace.seek(200)
                workspace.annotate_current_frame()
                self._wait_until(lambda: not workspace._busy)

                self.assertIs(window.workspace_stack.currentWidget(), window.view)
                self.assertEqual(Path(window.image_root), store.frames_directory)
                self.assertEqual(Path(window.label_root), store.labels_directory)
                self.assertEqual(window.current_label.img_name, "frame_000000000002.png")
            finally:
                if window.video_annotation_workspace is not None:
                    window.video_annotation_workspace.shutdown()
                window.plugin_manager_dialog.trigger_application_shutdown()
                window.deleteLater()
                self.app.processEvents()

    def test_timeline_supports_keyboard_selection_and_seek(self):
        timeline = TemporalTimeline()
        timeline.set_duration(5_000)
        timeline.set_annotations(
            [
                TemporalAnnotation(
                    id="phase-1",
                    kind="interval",
                    category="phase",
                    label="Preparation",
                    start_ms=500,
                    end_ms=1_500,
                ),
                TemporalAnnotation(
                    id="task-1",
                    kind="interval",
                    category="task",
                    label="Incision",
                    start_ms=2_000,
                    end_ms=3_000,
                ),
            ]
        )
        selected = []
        seeks = []
        timeline.annotation_selected.connect(selected.append)
        timeline.seek_requested.connect(seeks.append)
        timeline.show()
        timeline.setFocus()
        self.app.processEvents()

        QtTest.QTest.keyClick(timeline, QtCore.Qt.Key_Down)
        self.assertEqual(selected[-1], "phase-1")
        self.assertEqual(seeks[-1], 500)
        self.assertEqual(timeline._selected_annotation_id, "phase-1")
        QtTest.QTest.keyClick(timeline, QtCore.Qt.Key_End)
        self.assertEqual(seeks[-1], 5_000)
        self.assertEqual(timeline.focusPolicy(), QtCore.Qt.StrongFocus)
        self.assertTrue(timeline.accessibleName())
        timeline.close()

    def test_timeline_zoom_pan_and_reset_keep_a_bounded_view(self):
        timeline = TemporalTimeline()
        timeline.set_duration(10_000)
        timeline.zoom_by(2.0, 5_000)
        self.assertEqual(timeline._view_end_ms - timeline._view_start_ms, 5_000)
        initial_start = timeline._view_start_ms
        timeline.pan_by(1_000)
        self.assertGreater(timeline._view_start_ms, initial_start)
        self.assertLessEqual(timeline._view_end_ms, 10_000)
        timeline.reset_zoom()
        self.assertEqual((timeline._view_start_ms, timeline._view_end_ms), (0, 10_000))

    def test_video_shortcuts_play_and_step_but_ignore_text_input_focus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            workspace.open_video(video_path)
            workspace.show()
            self.app.processEvents()
            shortcuts = {
                shortcut.key().toString(): shortcut
                for shortcut in workspace.findChildren(QtWidgets.QShortcut)
            }

            workspace.play_button.setFocus()
            self.app.processEvents()
            workspace.seek(200)
            shortcuts["Left"].activated.emit()
            self.assertEqual(workspace.position_ms(), 100)

            workspace.label_combo.lineEdit().setFocus()
            self.app.processEvents()
            shortcuts["Right"].activated.emit()
            self.assertEqual(workspace.position_ms(), 100)

            fake_player = mock.Mock()
            fake_player.state.return_value = 0
            workspace.player = fake_player
            shortcuts["Space"].activated.emit()
            fake_player.play.assert_not_called()

            workspace.play_button.setFocus()
            self.app.processEvents()
            shortcuts["Space"].activated.emit()
            fake_player.play.assert_called_once()
            workspace.player = None
            workspace.shutdown()

    def test_more_menu_and_responsive_layout_keep_primary_controls_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = VideoAnnotationWorkspace(Path(directory) / "outputs")
            self._disable_player(workspace)
            self.assertFalse(workspace._panels_detached)
            self.assertGreaterEqual(workspace.splitter.indexOf(workspace.editor_panel), 0)
            self.assertIn(workspace.undo_action, workspace.more_menu.actions())
            self.assertIn(workspace.redo_action, workspace.more_menu.actions())
            self.assertIn(workspace.delete_action, workspace.more_menu.actions())
            self.assertIn(workspace.open_project_action, workspace.more_menu.actions())
            self.assertIn(workspace.output_root_action, workspace.more_menu.actions())
            self.assertIn(workspace.template_action, workspace.more_menu.actions())
            self.assertIn(workspace.import_action, workspace.more_menu.actions())
            self.assertIn(workspace.start_tracking_action, workspace.more_menu.actions())
            self.assertIn(workspace.review_tracking_action, workspace.more_menu.actions())
            self.assertEqual(
                [workspace.rate_combo.itemText(index) for index in range(workspace.rate_combo.count())],
                ["0.25×", "0.5×", "1×", "1.5×", "2×"],
            )

            # The main window's persistent mode toolbar leaves roughly this
            # central-workspace area at the supported 800x600 shell size.
            workspace.resize(680, 500)
            workspace.show()
            self.app.processEvents()
            self.assertEqual(workspace.splitter.orientation(), QtCore.Qt.Vertical)
            self.assertFalse(workspace.title_label.isVisible())
            self.assertGreaterEqual(
                workspace.interval_group.height(),
                workspace.interval_group.minimumSizeHint().height(),
            )
            self.assertGreater(
                workspace.annotation_table.height(),
                workspace.annotation_table.horizontalHeader().height(),
            )

            workspace.resize(1_200, 760)
            self.app.processEvents()
            self.assertEqual(workspace.splitter.orientation(), QtCore.Qt.Horizontal)
            self.assertTrue(workspace.title_label.isVisible())
            workspace.shutdown()

    def test_detached_project_panels_remain_functional_and_context_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            workspace.resize(1_200, 760)
            workspace.show()
            self.app.processEvents()
            editor = workspace.editor_panel
            timeline = workspace.timeline
            changes = []
            workspace.context_changed.connect(lambda: changes.append(True))

            detached_editor, detached_timeline = workspace.detach_project_panels()

            self.assertIs(detached_editor, editor)
            self.assertIs(detached_timeline, timeline)
            self.assertIs(workspace.detach_project_panels()[0], editor)
            self.assertTrue(workspace._panels_detached)
            self.assertEqual(workspace.splitter.indexOf(editor), -1)
            self.assertEqual(workspace.layout().indexOf(timeline), -1)
            self.assertEqual(workspace.splitter.count(), 1)

            def item_position(widget):
                return workspace.form_layout.getItemPosition(
                    workspace.form_layout.indexOf(widget)
                )

            self.assertEqual(item_position(workspace.category_label)[:2], (0, 0))
            self.assertEqual(item_position(workspace.annotation_label)[:2], (0, 2))
            self.assertEqual(item_position(workspace.mark_start_button)[:2], (1, 0))
            self.assertEqual(item_position(workspace.pending_label)[:2], (2, 0))

            workspace.resize(680, 500)
            self.app.processEvents()
            workspace.resize(1_200, 760)
            self.app.processEvents()
            self.assertEqual(workspace.splitter.indexOf(editor), -1)
            self.assertEqual(item_position(workspace.annotation_label)[:2], (0, 2))

            store = workspace.open_video(video_path)
            self.assertTrue(workspace.mark_start_button.isEnabled())
            workspace.label_combo.setCurrentText("Incision")
            workspace.seek(100)
            workspace.mark_start()
            workspace.seek(300)
            workspace.mark_end()
            self.assertEqual(len(store.temporal_annotations), 1)
            self.assertEqual(workspace.annotation_table.rowCount(), 1)
            self.assertEqual(len(timeline._annotations), 1)

            before_busy = len(changes)
            workspace._set_frame_extraction_busy(True)
            workspace._set_frame_extraction_busy(False)
            self.assertGreaterEqual(len(changes), before_busy + 2)
            self.assertTrue(changes)
            workspace.shutdown()

    def test_output_root_choice_is_compact_and_applies_only_to_new_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = VideoAnnotationWorkspace(root / "initial-output")
            self._disable_player(workspace)
            chosen = root / "chosen-output"
            chosen.mkdir()
            emitted = []
            statuses = []
            workspace.output_root_changed.connect(emitted.append)
            workspace.status_message.connect(statuses.append)

            with mock.patch.object(
                QtWidgets.QFileDialog,
                "getExistingDirectory",
                return_value=str(chosen),
            ):
                workspace.choose_output_root_dialog()

            self.assertEqual(workspace.output_root, chosen.resolve())
            self.assertEqual(emitted, [str(chosen.resolve())])
            self.assertTrue(statuses)
            self.assertEqual(workspace.output_root_action.toolTip(), str(chosen.resolve()))
            workspace.shutdown()

    def test_vlc_fallback_binds_media_rate_window_and_releases_dll_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = VideoAnnotationWorkspace(Path(directory) / "outputs")
            self._disable_player(workspace)
            # Restore the real method after the player-neutral helper above.
            workspace._activate_vlc_fallback = VideoAnnotationWorkspace._activate_vlc_fallback.__get__(
                workspace, VideoAnnotationWorkspace
            )
            fake_player = mock.Mock()
            fake_instance = mock.Mock()
            fake_instance.media_player_new.return_value = fake_player
            fake_instance.media_new.return_value = "media-token"
            fake_vlc = types.SimpleNamespace(Instance=mock.Mock(return_value=fake_instance))
            dll_handle = mock.Mock()
            with mock.patch.dict(sys.modules, {"vlc": fake_vlc}):
                with mock.patch(
                    "IVAS.widgets.video_annotation_workspace.os.add_dll_directory",
                    return_value=dll_handle,
                ):
                    self.assertTrue(workspace._activate_vlc_fallback(Path("case.mp4")))
            fake_player.set_media.assert_called_once_with("media-token")
            fake_player.set_rate.assert_called_once_with(1.0)
            fake_player.set_hwnd.assert_called_once()
            workspace.shutdown()
            fake_player.release.assert_called_once()
            dll_handle.close.assert_called_once()

    def test_pending_interval_is_not_silently_replaced_or_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_video = root / "first.avi"
            second_video = root / "second.avi"
            if not self._synthetic_video(first_video) or not self._synthetic_video(
                second_video
            ):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            first_store = workspace.open_video(first_video)
            workspace.label_combo.setCurrentText("Incision")
            workspace.seek(100)
            workspace.mark_start()
            pending = workspace._pending_interval

            workspace.label_combo.setCurrentText("Preparation")
            workspace.seek(200)
            with mock.patch.object(
                workspace, "_confirm_discard_pending_interval", return_value=False
            ):
                workspace.mark_start()
            self.assertEqual(workspace._pending_interval, pending)
            with mock.patch.object(
                workspace, "_confirm_discard_pending_interval", return_value=True
            ):
                workspace.mark_start()
            pending = workspace._pending_interval
            self.assertEqual(pending, ("phase", "Preparation", 200))

            with mock.patch.object(
                workspace, "_confirm_discard_pending_interval", return_value=False
            ):
                returned = workspace.open_video(second_video)
            self.assertIs(returned, first_store)
            self.assertEqual(workspace.video_path, first_video.resolve())
            self.assertEqual(workspace._pending_interval, pending)

            with mock.patch.object(
                workspace, "_confirm_discard_pending_interval", return_value=True
            ):
                second_store = workspace.open_video(second_video)
            self.assertIsNot(second_store, first_store)
            self.assertEqual(workspace.video_path, second_video.resolve())
            self.assertIsNone(workspace._pending_interval)
            workspace.shutdown()

    def test_failed_interval_save_rolls_back_and_retry_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            store = workspace.open_video(video_path)
            workspace.label_combo.setCurrentText("Incision")
            workspace.seek(100)
            workspace.mark_start()
            workspace.seek(300)

            real_save = store.save
            with mock.patch.object(store, "save", side_effect=OSError("disk full")):
                with mock.patch.object(QtWidgets.QMessageBox, "critical"):
                    workspace.mark_end()
            self.assertEqual(store.temporal_annotations, [])
            self.assertIsNotNone(workspace._pending_interval)

            with mock.patch.object(store, "save", side_effect=real_save):
                workspace.mark_end()
            self.assertEqual(len(store.temporal_annotations), 1)
            self.assertIsNone(workspace._pending_interval)
            workspace.shutdown()

    def test_clean_existing_project_is_not_rewritten_when_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            first = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(first)
            first.open_video(video_path)
            first.shutdown()

            calls = []
            real_save = UnifiedVideoProjectStore.save

            def counting_save(store):
                calls.append(store.path)
                return real_save(store)

            second = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(second)
            with mock.patch.object(UnifiedVideoProjectStore, "save", counting_save):
                second.open_video(video_path)
            self.assertEqual(calls, [])
            second.shutdown()

    def test_frame_extraction_busy_state_recovers_after_decode_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            workspace.open_video(video_path)
            self._wait_until(
                lambda: workspace._frame_map_worker is None
                or not workspace._frame_map_worker.isRunning()
            )
            messages = []
            workspace.status_message.connect(messages.append)
            failed_capture = mock.Mock()
            failed_capture.isOpened.return_value = False

            with mock.patch(
                "IVAS.video.media.cv2.VideoCapture",
                return_value=failed_capture,
            ):
                with mock.patch.object(QtWidgets.QMessageBox, "critical"):
                    workspace.annotate_current_frame()
                    self._wait_until(lambda: not workspace._busy)
            self.assertFalse(workspace._busy)
            self.assertTrue(workspace.annotate_frame_button.isEnabled())
            self.assertEqual(workspace.annotate_frame_button.text(), "标注当前帧")
            self.assertTrue(any(message.startswith("正在") and "提取" in message for message in messages))
            self.assertGreaterEqual(failed_capture.release.call_count, 1)
            workspace.shutdown()

    def test_saved_frame_annotation_updates_unified_project_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            store = workspace.open_video(video_path)
            workspace.seek(200)
            workspace.annotate_current_frame()
            self._wait_until(lambda: not workspace._busy)
            label_path = store.labels_directory / "frame_000000000002.json"
            label_path.write_text("{}", encoding="utf-8")

            self.assertTrue(workspace.mark_frame_annotation_saved(str(label_path)))
            loaded = UnifiedVideoProjectStore.load(store.path)
            spatial = loaded.project["spatial_frames"][0]
            self.assertEqual(spatial["annotation_status"], "saved")
            self.assertEqual(
                workspace._resolve_project_asset(spatial["annotation_path"]),
                label_path.resolve(),
            )
            workspace.shutdown()

    def test_public_pending_confirmation_and_shutdown_are_cancellable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            workspace.open_video(video_path)
            workspace.label_combo.setCurrentText("Incision")
            workspace.mark_start()

            with mock.patch.object(
                QtWidgets.QMessageBox,
                "question",
                return_value=QtWidgets.QMessageBox.Cancel,
            ):
                self.assertFalse(workspace.confirm_discard_pending())
            self.assertIsNotNone(workspace._pending_interval)

            with mock.patch.object(
                QtWidgets.QMessageBox,
                "question",
                return_value=QtWidgets.QMessageBox.Discard,
            ):
                self.assertTrue(workspace.confirm_discard_pending())
            self.assertIsNone(workspace._pending_interval)
            self.assertTrue(workspace.shutdown())

    def test_compact_temporal_actions_seek_discard_and_reset_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            workspace.open_video(video_path)

            menu_actions = workspace.more_menu.actions()
            for action in (
                workspace.discard_pending_interval_action,
                workspace.seek_backward_5s_action,
                workspace.seek_backward_1s_action,
                workspace.seek_forward_1s_action,
                workspace.seek_forward_5s_action,
                workspace.capture_keyframe_action,
                workspace.timeline_zoom_in_action,
                workspace.timeline_zoom_out_action,
                workspace.timeline_reset_zoom_action,
                workspace.save_project_action,
            ):
                self.assertIn(action, menu_actions)
                self.assertTrue(action.statusTip())

            workspace.seek(300)
            workspace.seek_backward_1s_action.trigger()
            self.assertEqual(workspace.position_ms(), 0)
            workspace.seek_forward_5s_action.trigger()
            self.assertEqual(workspace.position_ms(), workspace.duration_ms())

            workspace.timeline.set_duration(10_000)
            workspace.timeline_zoom_in_action.trigger()
            self.assertLess(
                workspace.timeline._view_end_ms - workspace.timeline._view_start_ms,
                10_000,
            )
            workspace.timeline_reset_zoom_action.trigger()
            self.assertEqual(
                (workspace.timeline._view_start_ms, workspace.timeline._view_end_ms),
                (0, 10_000),
            )

            workspace.label_combo.setCurrentText("Incision")
            workspace.mark_start()
            self.assertTrue(workspace.discard_pending_interval_action.isEnabled())
            workspace.discard_pending_interval_action.trigger()
            self.assertIsNone(workspace._pending_interval)
            self.assertFalse(workspace.discard_pending_interval_action.isEnabled())

            workspace.set_language("en")
            self.assertIn("Capture", workspace.capture_keyframe_action.text())
            self.assertIn("linked", workspace.capture_keyframe_action.statusTip())
            workspace.shutdown()

    def test_capture_current_frame_creates_linked_keyframe_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            store = workspace.open_video(video_path)
            handoffs = []
            workspace.frame_annotation_requested.connect(
                lambda *arguments: handoffs.append(arguments)
            )

            workspace.seek(200)
            self.assertTrue(workspace.capture_current_keyframe())
            self._wait_until(lambda: not workspace._busy)

            points = [
                annotation
                for annotation in store.temporal_annotations
                if annotation.kind == "point"
            ]
            self.assertEqual(len(points), 1)
            point = points[0]
            self.assertEqual(point.frame_index, 2)
            self.assertEqual(point.start_ms, 200)
            self.assertEqual(point.status, "ready")
            self.assertTrue(point.asset_path)
            self.assertTrue(workspace._resolve_project_asset(point.asset_path).is_file())
            self.assertEqual(handoffs, [])
            self.assertTrue((store.case_directory / "exports" / "annotations.csv").is_file())
            workspace.shutdown()

    def test_quick_rating_menu_commits_and_replaces_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            store = workspace.open_video(video_path)

            option = next(
                action
                for action in workspace.rating_menu.actions()
                if action.data() and int(action.data()[1]) == 4
            )
            option.trigger()
            ratings = [
                annotation
                for annotation in store.temporal_annotations
                if annotation.kind == "rating"
            ]
            self.assertEqual(len(ratings), 1)
            self.assertEqual(ratings[0].score, 4)
            self.assertEqual(ratings[0].status, "ready")

            self.assertTrue(workspace.set_rating_from_menu(ratings[0].template_id, 1))
            ratings = [
                annotation
                for annotation in store.temporal_annotations
                if annotation.kind == "rating"
            ]
            self.assertEqual(len(ratings), 1)
            self.assertEqual(ratings[0].score, 1)
            workspace.shutdown()

    def test_export_sync_failure_does_not_roll_back_saved_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "case.avi"
            if not self._synthetic_video(video_path):
                self.skipTest("OpenCV MJPG writer is unavailable")
            workspace = VideoAnnotationWorkspace(root / "outputs")
            self._disable_player(workspace)
            store = workspace.open_video(video_path)
            workspace.label_combo.setCurrentText("Incision")
            workspace.seek(100)
            workspace.mark_start()
            workspace.seek(300)
            messages = []
            workspace.status_message.connect(messages.append)

            with mock.patch.object(
                store, "sync_default_exports", side_effect=OSError("export locked")
            ):
                workspace.mark_end()

            self.assertEqual(len(store.temporal_annotations), 1)
            loaded = UnifiedVideoProjectStore.load(store.path)
            self.assertEqual(len(loaded.temporal_annotations), 1)
            self.assertTrue(any("export locked" in message for message in messages))
            workspace.shutdown()


if __name__ == "__main__":
    unittest.main()
