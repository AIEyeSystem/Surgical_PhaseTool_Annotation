import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from IVAS.video.project import UnifiedVideoProjectStore


class FusedWorkbenchTests(unittest.TestCase):
    """Acceptance coverage for the shared Frame/Video workbench shell."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _window(self):
        from IVAS.widgets import mainwindow as mainwindow_module

        patcher = mock.patch.object(
            mainwindow_module.CheckLatestVersionThread,
            "start",
            lambda thread: None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        window = mainwindow_module.MainWindow()

        def cleanup_window():
            timer = getattr(window, "_version_check_timer", None)
            if timer is not None:
                timer.stop()
            workspace = getattr(window, "video_annotation_workspace", None)
            if workspace is not None:
                workspace.shutdown()
            plugin_manager = getattr(window, "plugin_manager_dialog", None)
            if plugin_manager is not None:
                plugin_manager.trigger_application_shutdown()
            window.close()
            window.deleteLater()
            self.app.processEvents()

        self.addCleanup(cleanup_window)
        return window

    @classmethod
    def _leaf_actions(cls, menu):
        actions = []
        for action in menu.actions():
            submenu = action.menu()
            if submenu is not None:
                actions.extend(cls._leaf_actions(submenu))
            elif not action.isSeparator():
                actions.append(action)
        return tuple(actions)

    @classmethod
    def _walk_menu(cls, menu, seen=None):
        if seen is None:
            seen = set()
        if menu in seen:
            return ()
        seen.add(menu)
        menus = [menu]
        for action in menu.actions():
            submenu = action.menu()
            if submenu is not None:
                menus.extend(cls._walk_menu(submenu, seen))
        return tuple(menus)

    @classmethod
    def _public_menus(cls, window):
        menus = []
        seen = set()
        for action in window.menuBar().actions():
            menu = action.menu()
            if menu is not None:
                menus.extend(cls._walk_menu(menu, seen))
        for button in window.findChildren(QtWidgets.QToolButton):
            menu = button.menu()
            if menu is not None:
                menus.extend(cls._walk_menu(menu, seen))
        return tuple(menus)

    @classmethod
    def _discoverable_actions(cls, window):
        actions = set()
        for menu in cls._public_menus(window):
            actions.update(cls._leaf_actions(menu))
        for toolbar in (window.toolBar, window.toolBar_2):
            actions.update(
                action for action in toolbar.actions() if not action.isSeparator()
            )
        return actions

    @staticmethod
    def _menu_name(menu):
        return menu.objectName() or menu.title() or repr(menu)

    def _assert_clean_separators(self, owner, actions):
        if not actions:
            return
        name = (
            owner.objectName()
            or getattr(owner, "title", lambda: "")()
            or repr(owner)
        )
        self.assertFalse(actions[0].isSeparator(), "{} starts with a separator".format(name))
        self.assertFalse(actions[-1].isSeparator(), "{} ends with a separator".format(name))
        for previous, current in zip(actions, actions[1:]):
            self.assertFalse(
                previous.isSeparator() and current.isSeparator(),
                "{} contains adjacent separators".format(name),
            )

    def test_cold_start_exposes_video_commands_without_creating_video_qt(self):
        window = self._window()

        self.assertIsNone(window.video_annotation_workspace)
        self.assertIsInstance(window.menuTemporal, QtWidgets.QMenu)
        self.assertIsInstance(window.menuTracking, QtWidgets.QMenu)
        self.assertIsInstance(window.project_tools_button, QtWidgets.QToolButton)
        self.assertIs(window.workspace_tools_button, window.mode_tools_button)
        self.assertIsInstance(window.temporal_lane_combo, QtWidgets.QComboBox)
        self.assertIsInstance(window.temporal_label_combo, QtWidgets.QComboBox)
        self.assertTrue(window.temporal_lane_combo.accessibleName())
        self.assertTrue(window.temporal_label_combo.accessibleName())
        self.assertFalse(window.temporal_lane_combo.isEnabled())
        self.assertFalse(window.temporal_label_combo.isEnabled())
        self.assertFalse(window.temporal_dock.isHidden())
        self.assertTrue(window.annotate_video_frame_button.isHidden())
        self.assertTrue(window.return_to_video_button.isHidden())

        temporal_actions = set(self._leaf_actions(window.menuTemporal))
        tracking_actions = set(self._leaf_actions(window.menuTracking))
        self.assertTrue(tracking_actions, "Tracking must be a real discoverable menu")
        required_temporal_actions = {
            window.actionTemporal_mark_start,
            window.actionTemporal_mark_end,
            window.actionAnnotate_current_video_frame,
            window.actionReturn_to_video,
        }
        self.assertTrue(required_temporal_actions.issubset(temporal_actions))

        major_video_actions = required_temporal_actions | tracking_actions | {
            window.actionOpen_video_workspace,
            window.actionOpen_video_project_manifest,
            window.actionVideo_output_root,
        }
        self.assertTrue(
            major_video_actions.issubset(self._discoverable_actions(window)),
            "Major video commands must be reachable before the lazy workspace exists",
        )

        project_actions = temporal_actions | tracking_actions
        for action in project_actions:
            with self.subTest(action=action.objectName() or action.text()):
                self.assertTrue(action.isVisible())
                self.assertFalse(action.isEnabled())
                self.assertTrue(action.statusTip() or action.toolTip())

        for action in (
            window.actionImages_dir,
            window.actionOpen_video_workspace,
            window.actionOpen_video_project_manifest,
            window.actionVideo_output_root,
        ):
            with self.subTest(open_action=action.objectName()):
                self.assertTrue(action.isVisible())
                self.assertTrue(action.isEnabled())

    def test_release_smoke_validates_the_lazy_fused_video_shell_without_media(self):
        from IVAS.main import _validate_video_workbench_smoke

        window = self._window()
        self.assertIsNone(window.video_annotation_workspace)

        workspace = _validate_video_workbench_smoke(window)

        self.assertIs(window.video_annotation_workspace, workspace)
        self.assertIsNone(workspace.store)
        self.assertIsNone(workspace.video_path)
        self.assertIs(
            window.video_transport_toolbar.widgetForAction(
                window.video_transport_widget_action
            ),
            workspace.transport_widget,
        )
        self.assertIs(window.temporal_dock.widget().widget(), workspace.editor_panel)
        self.assertIs(window.timeline_dock.widget(), workspace.timeline)

    def test_file_groups_media_and_project_keeps_location_shortcuts(self):
        window = self._window()

        file_actions = set(self._leaf_actions(window.menuFile))
        expected_file_entries = {
            window.actionImages_dir,
            window.actionOpen_video_workspace,
            window.actionOpen_video_project_manifest,
        }
        self.assertTrue(expected_file_entries.issubset(file_actions))

        public_menu_actions = set()
        for menu in self._public_menus(window):
            public_menu_actions.update(self._leaf_actions(menu))
        project_actions = set(self._leaf_actions(window.project_tools_button.menu()))
        self.assertTrue(project_actions.issubset(file_actions))
        self.assertIn(window.actionVideo_output_root, project_actions)
        for mode_action in (
            window.actionShow_image_workspace,
            window.actionShow_video_workspace,
        ):
            with self.subTest(mode_action=mode_action.objectName()):
                self.assertTrue(mode_action.isVisible())
                self.assertIn(mode_action, public_menu_actions)

    def test_public_menus_and_tool_button_menus_have_clean_separators(self):
        window = self._window()

        public_menus = self._public_menus(window)
        self.assertIn(window.menuTemporal, public_menus)
        self.assertIn(window.menuTracking, public_menus)
        self.assertIn(window.project_tools_button.menu(), public_menus)
        for menu in public_menus:
            with self.subTest(menu=self._menu_name(menu)):
                self._assert_clean_separators(menu, menu.actions())

    def test_advanced_temporal_commands_are_public_proxies_not_extra_buttons(self):
        window = self._window()
        proxy_to_target_name = {
            window.actionTemporal_capture_keyframe: "capture_keyframe_action",
            window.actionTemporal_discard_pending: "discard_pending_interval_action",
            window.actionVideo_seek_backward_5s: "seek_backward_5s_action",
            window.actionVideo_seek_backward_1s: "seek_backward_1s_action",
            window.actionVideo_seek_forward_1s: "seek_forward_1s_action",
            window.actionVideo_seek_forward_5s: "seek_forward_5s_action",
            window.actionTimeline_zoom_in: "timeline_zoom_in_action",
            window.actionTimeline_zoom_out: "timeline_zoom_out_action",
            window.actionTimeline_reset_zoom: "timeline_reset_zoom_action",
        }
        proxies = set(proxy_to_target_name)
        self.assertTrue(proxies.issubset(set(self._leaf_actions(window.menuTemporal))))
        self.assertTrue(
            proxies.issubset(
                set(self._leaf_actions(window.temporal_tools_button.menu()))
            )
        )
        direct_toolbar_actions = set(window.toolBar.actions()) | set(
            window.toolBar_2.actions()
        )
        self.assertTrue(proxies.isdisjoint(direct_toolbar_actions))
        for proxy in proxies:
            self.assertFalse(proxy.isEnabled())
            self.assertTrue(proxy.statusTip())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "temporal-proxy-case.mp4"
            media_path.write_bytes(b"temporal-proxy-test-media")
            store = UnifiedVideoProjectStore.create(
                root / "temporal-proxy.ivas.json",
                media_path,
                duration_ms=10_000,
                fps=25.0,
                frame_count=250,
                width=16,
                height=12,
            )
            workspace = window._ensure_video_workspace()
            if workspace.player is not None:
                workspace.player.stop()
                workspace.player = None
            if workspace.vlc_player is not None:
                workspace.vlc_player.stop()
                workspace.vlc_player = None
            workspace.store = store
            workspace.video_path = media_path
            workspace._set_project_controls_enabled(True)
            workspace.timeline.set_duration(10_000)
            window._activate_video_workspace()

            for proxy, target_name in proxy_to_target_name.items():
                with self.subTest(proxy=proxy.objectName()):
                    self.assertIs(
                        window._video_proxy_targets[proxy],
                        getattr(workspace, target_name),
                    )
                    self.assertEqual(
                        proxy.isEnabled(), getattr(workspace, target_name).isEnabled()
                    )

            workspace.seek(5_000)
            window.actionVideo_seek_backward_1s.trigger()
            self.assertEqual(workspace.position_ms(), 4_000)
            original_span = (
                workspace.timeline._view_end_ms
                - workspace.timeline._view_start_ms
            )
            window.actionTimeline_zoom_in.trigger()
            self.assertLess(
                workspace.timeline._view_end_ms
                - workspace.timeline._view_start_ms,
                original_span,
            )
            window.actionTimeline_reset_zoom.trigger()
            self.assertEqual(
                (
                    workspace.timeline._view_start_ms,
                    workspace.timeline._view_end_ms,
                ),
                (0, 10_000),
            )

            workspace.label_combo.setCurrentText("Incision")
            workspace.mark_start()
            self.assertTrue(window.actionTemporal_discard_pending.isEnabled())
            window.actionTemporal_discard_pending.trigger()
            self.assertIsNone(workspace._pending_interval)

        window.translate("en")
        self.assertEqual(
            window.actionTemporal_capture_keyframe.text(),
            "Capture current frame as key frame",
        )
        self.assertIn(
            "exact current frame",
            window.actionTemporal_capture_keyframe.statusTip(),
        )

    def test_toolbars_use_semantic_groups_without_command_pileup(self):
        window = self._window()

        primary_actions = window.toolBar.actions()
        secondary_actions = window.toolBar_2.actions()
        self._assert_clean_separators(window.toolBar, primary_actions)
        self._assert_clean_separators(window.toolBar_2, secondary_actions)

        primary_controls = [
            action for action in primary_actions if not action.isSeparator()
        ]
        secondary_controls = [
            action for action in secondary_actions if not action.isSeparator()
        ]
        # Four direct drawing tools replace the former annotation dropdown;
        # advanced temporal/tracking commands remain grouped in menus.
        self.assertLessEqual(len(primary_controls), 15)
        self.assertLessEqual(len(secondary_controls), 9)
        self.assertGreaterEqual(
            sum(action.isSeparator() for action in primary_actions),
            2,
            "The primary toolbar must visually separate navigation and annotation groups",
        )
        direct_toolbar_actions = set(primary_actions) | set(secondary_actions)
        for advanced_action in (
            window.actionTemporal_add_point,
            window.actionTemporal_add_rating,
            window.actionTemporal_capture_keyframe,
            window.actionTemporal_discard_pending,
            window.actionTemporal_templates,
            window.actionTemporal_import,
            window.actionTemporal_export,
            window.actionVideo_seek_backward_5s,
            window.actionVideo_seek_backward_1s,
            window.actionVideo_seek_forward_1s,
            window.actionVideo_seek_forward_5s,
            window.actionTimeline_zoom_in,
            window.actionTimeline_zoom_out,
            window.actionTimeline_reset_zoom,
            window.actionTracking_start,
            window.actionTracking_review,
            window.actionTracking_rollback,
        ):
            with self.subTest(advanced_action=advanced_action.objectName()):
                self.assertNotIn(advanced_action, direct_toolbar_actions)

        toolbar_widgets = {
            toolbar.widgetForAction(action)
            for toolbar in (window.toolBar, window.toolBar_2)
            for action in toolbar.actions()
        }
        self.assertIn(window.project_tools_button, toolbar_widgets)
        self.assertEqual(
            window.toolBarArea(window.toolBar), QtCore.Qt.TopToolBarArea
        )
        self.assertEqual(
            window.toolBarArea(window.toolBar_2), QtCore.Qt.TopToolBarArea
        )
        self.assertTrue(
            window.toolBarBreak(window.toolBar_2),
            "Temporal and tracking controls must stay on a visible second row",
        )

    def test_activating_video_surface_never_hides_the_shared_toolbar(self):
        window = self._window()
        window.show()
        self.app.processEvents()
        self.assertFalse(window.toolBar.isHidden())

        workspace = window._ensure_video_workspace()
        if workspace.player is not None:
            workspace.player.stop()
            workspace.player = None
        if workspace.vlc_player is not None:
            workspace.vlc_player.stop()
            workspace.vlc_player = None
        self.assertIs(window.temporal_dock.widget().widget(), workspace.editor_panel)
        self.assertIs(window.timeline_dock.widget(), workspace.timeline)
        self.app.processEvents()
        # Replacing the cold-start temporal call-to-action must not leave
        # dangling Qt objects in the runtime translation path.
        window.translate("zh")
        window.translate("en")
        window._activate_video_workspace()
        self.app.processEvents()

        self.assertIs(window.workspace_stack.currentWidget(), workspace)
        self.assertFalse(window.toolBar.isHidden())
        self.assertTrue(window.toolBar.isVisibleTo(window))
        self.assertTrue(window.annotate_video_frame_button.isHidden())
        self.assertTrue(window.return_to_video_button.isHidden())

    def test_shared_transport_and_shortcuts_survive_central_surface_changes(self):
        window = self._window()
        window.show()
        self.app.processEvents()
        bitmap_shortcut = window.actionBit_map.shortcut().toString()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "case.mp4"
            media_path.write_bytes(b"shared-transport-test-media")
            store = UnifiedVideoProjectStore.create(
                root / "project.ivas.json",
                media_path,
                duration_ms=2_000,
                fps=25.0,
                frame_count=50,
                width=16,
                height=12,
            )
            workspace = window._ensure_video_workspace()
            if workspace.player is not None:
                workspace.player.stop()
                workspace.player = None
            if workspace.vlc_player is not None:
                workspace.vlc_player.stop()
                workspace.vlc_player = None
            workspace.store = store
            workspace.video_path = media_path
            workspace._set_project_controls_enabled(True)
            window._activate_video_workspace()
            self.app.processEvents()

            self.assertIs(
                window.video_transport_toolbar.widgetForAction(
                    window.video_transport_widget_action
                ),
                workspace.transport_widget,
            )
            self.assertEqual(
                workspace.video_layout.indexOf(workspace.transport_widget), -1
            )
            self.assertFalse(window.video_transport_toolbar.isHidden())
            self.assertTrue(
                window.video_transport_toolbar.isVisibleTo(window)
            )
            self.assertGreater(
                window.video_transport_toolbar.height(),
                40,
                "The shared transport must not collapse to an empty toolbar strip",
            )
            self.assertGreater(workspace.transport_widget.height(), 35)
            for control in (
                workspace.back_button,
                workspace.play_button,
                workspace.forward_button,
                workspace.position_label,
                workspace.rate_combo,
                workspace.position_slider,
            ):
                with self.subTest(transport_control=control.accessibleName()):
                    self.assertTrue(control.isVisibleTo(window))
            self.assertTrue(workspace.annotate_frame_button.isHidden())
            self.assertTrue(
                all(
                    not shortcut.isEnabled()
                    for shortcut in workspace._workspace_shortcuts
                )
            )
            self.assertTrue(window.actionBit_map.shortcut().isEmpty())

            shortcuts = {
                shortcut.key().toString(): shortcut
                for shortcut in window._video_application_shortcuts
            }
            self.assertEqual(
                set(shortcuts),
                {"Space", "Left", "Right", "I", "O", "Ctrl+Z", "Ctrl+Shift+Z"},
            )
            workspace.play_button.setFocus()
            self.app.processEvents()
            self.assertTrue(all(shortcut.isEnabled() for shortcut in shortcuts.values()))
            with (
                mock.patch.object(workspace, "toggle_playback") as play,
                mock.patch.object(workspace, "step_frames") as step,
                mock.patch.object(workspace, "mark_start") as mark_start,
                mock.patch.object(workspace, "mark_end") as mark_end,
                mock.patch.object(workspace, "undo") as undo,
                mock.patch.object(workspace, "redo") as redo,
            ):
                shortcuts["Space"].activated.emit()
                shortcuts["Left"].activated.emit()
                shortcuts["Right"].activated.emit()
                shortcuts["I"].activated.emit()
                shortcuts["O"].activated.emit()
                shortcuts["Ctrl+Z"].activated.emit()
                shortcuts["Ctrl+Shift+Z"].activated.emit()
                play.assert_called_once_with()
                self.assertEqual(
                    [call.args for call in step.call_args_list], [(-1,), (1,)]
                )
                mark_start.assert_called_once_with()
                mark_end.assert_called_once_with()
                undo.assert_called_once_with()
                redo.assert_called_once_with()

                step.reset_mock()
                window.temporal_label_combo.lineEdit().setFocus()
                self.app.processEvents()
                self.assertTrue(
                    all(not shortcut.isEnabled() for shortcut in shortcuts.values())
                )
                shortcuts["Right"].activated.emit()
                step.assert_not_called()

            # A standalone image surface gives Space back to bitmap display.
            window._active_video_frame_case = False
            window.show_image_workspace()
            self.app.processEvents()
            self.assertTrue(window.video_transport_toolbar.isHidden())
            self.assertEqual(
                window.actionBit_map.shortcut().toString(), bitmap_shortcut
            )

    def test_one_save_action_routes_to_the_visible_part_of_the_case(self):
        window = self._window()
        window.show()
        self.app.processEvents()
        window.actionSave.setEnabled(True)

        self.assertIn(window.actionSave, self._leaf_actions(window.menuFile))
        self.assertIn(window.actionSave, window.toolBar.actions())
        self.assertEqual(window.actionSave.shortcut().toString(), "Ctrl+S")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "save-case.mp4"
            media_path.write_bytes(b"unified-save-test-media")
            from PIL import Image
            Image.new("RGB", (16, 12), "white").save(root / "frame.png")
            window.load_image_directory(str(root))
            store = UnifiedVideoProjectStore.create(
                root / "save-project.ivas.json",
                media_path,
                duration_ms=1_000,
                fps=25.0,
                frame_count=25,
                width=16,
                height=12,
            )
            workspace = window._ensure_video_workspace()
            if workspace.player is not None:
                workspace.player.stop()
                workspace.player = None
            if workspace.vlc_player is not None:
                workspace.vlc_player.stop()
                workspace.vlc_player = None
            workspace.store = store
            workspace.video_path = media_path
            workspace._set_project_controls_enabled(True)
            window._activate_video_workspace()
            self.app.processEvents()

            self.assertTrue(window.actionSave.isEnabled())
            with mock.patch.object(
                workspace, "save_project", return_value=True
            ) as save_project:
                window.actionSave.trigger()
            save_project.assert_called_once_with()

            workspace.save_project_action.setEnabled(False)
            self.app.processEvents()
            self.assertFalse(window.actionSave.isEnabled())
            workspace.save_project_action.setEnabled(True)
            self.app.processEvents()
            self.assertTrue(window.actionSave.isEnabled())

            window._active_video_frame_case = False
            window.show_image_workspace()
            self.app.processEvents()
            self.assertTrue(
                window.actionSave.isEnabled(),
                "The frame save state from before video preview must be restored",
            )
            with mock.patch.object(window, "save") as save_frame:
                window.actionSave.trigger()
            save_frame.assert_called_once_with()

            window.translate("zh")
            self.assertEqual(window.actionSave.text(), "保存当前工作")
            self.assertEqual(window.actionSave.shortcut().toString(), "Ctrl+S")
            window.translate("en")
            self.assertEqual(window.actionSave.text(), "Save current work")
            self.assertEqual(window.actionSave.shortcut().toString(), "Ctrl+S")

    def test_video_frame_context_keeps_temporal_surface_and_image_tools_together(self):
        window = self._window()
        window.show()
        self.app.processEvents()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_directory = root / "frames"
            labels_directory = root / "labels"
            frames_directory.mkdir()
            labels_directory.mkdir()
            frame_name = "frame_000000000000.png"
            image = QtGui.QImage(16, 12, QtGui.QImage.Format_RGB32)
            image.fill(QtGui.QColor("black"))
            self.assertTrue(image.save(str(frames_directory / frame_name)))

            media_path = root / "case.mp4"
            media_path.write_bytes(b"fused-workbench-test-media")
            store = UnifiedVideoProjectStore.create(
                root / "project.ivas.json",
                media_path,
                duration_ms=1_000,
                fps=25.0,
                frame_count=25,
                width=16,
                height=12,
            )
            workspace = window._ensure_video_workspace()
            if workspace.player is not None:
                workspace.player.stop()
                workspace.player = None
            if workspace.vlc_player is not None:
                workspace.vlc_player.stop()
                workspace.vlc_player = None
            workspace.store = store
            workspace.video_path = media_path
            workspace._set_project_controls_enabled(True)
            window.active_video_project_path = str(store.path)
            window._activate_video_workspace()
            window.temporal_dock.show()
            window.timeline_dock.show()
            self.app.processEvents()

            window.open_video_frame_for_annotation(
                str(frames_directory),
                str(labels_directory),
                frame_name,
            )
            self.app.processEvents()

            self.assertIs(window.workspace_stack.currentWidget(), window.view)
            self.assertTrue(window._active_video_frame_case)
            self.assertFalse(window.temporal_dock.isHidden())
            self.assertFalse(window.timeline_dock.isHidden())
            self.assertTrue(window.temporal_dock.isVisibleTo(window))
            self.assertTrue(window.timeline_dock.isVisibleTo(window))
            self.assertTrue(window.actionPolygon.isEnabled())
            self.assertTrue(window.actionTemporal_mark_start.isEnabled())
            self.assertTrue(window.actionReturn_to_video.isEnabled())
            self.assertTrue(window.annotate_video_frame_button.isHidden())
            self.assertFalse(window.return_to_video_button.isHidden())
            self.assertFalse(window.video_transport_toolbar.isHidden())
            self.assertTrue(
                window.video_transport_toolbar.isVisibleTo(window)
            )
            self.assertIs(
                window.video_transport_toolbar.widgetForAction(
                    window.video_transport_widget_action
                ),
                workspace.transport_widget,
            )


if __name__ == "__main__":
    unittest.main()
