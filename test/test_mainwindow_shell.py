import os
import unittest
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets


class MainWindowShellTests(unittest.TestCase):
    OPTIONAL_DIALOG_ATTRIBUTES = (
        "category_setting_dialog",
        "category_edit_widget",
        "Converter_dialog",
        "video2frames_dialog",
        "auto_segment_dialog",
        "annos_validator_dialog",
        "process_exif_dialog",
        "shortcut_dialog",
        "about_dialog",
        "model_manager_dialog",
        "remote_sam_dialog",
    )
    OPTIONAL_DIALOG_ENSURERS = (
        ("_ensure_category_setting_dialog", "category_setting_dialog"),
        ("_ensure_category_edit_dialog", "category_edit_widget"),
        ("_ensure_converter_dialog", "Converter_dialog"),
        ("_ensure_video2frames_dialog", "video2frames_dialog"),
        ("_ensure_auto_segment_dialog", "auto_segment_dialog"),
        ("_ensure_annos_validator_dialog", "annos_validator_dialog"),
        ("_ensure_process_exif_dialog", "process_exif_dialog"),
        ("_ensure_shortcut_dialog", "shortcut_dialog"),
        ("_ensure_about_dialog", "about_dialog"),
        ("_ensure_model_manager_dialog", "model_manager_dialog"),
        ("_ensure_remote_sam_dialog", "remote_sam_dialog"),
    )

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _window(self):
        from IVAS.widgets import mainwindow as mainwindow_module

        patcher = mock.patch.object(
            mainwindow_module.CheckLatestVersionThread, "start", lambda thread: None
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        window = mainwindow_module.MainWindow()

        def cleanup_window():
            window._version_check_timer.stop()
            workspace = window.video_annotation_workspace
            if workspace is not None:
                workspace.shutdown()
            window.plugin_manager_dialog.trigger_application_shutdown()
            window.deleteLater()
            self.app.processEvents()

        self.addCleanup(cleanup_window)
        return window

    def test_shell_keeps_video_player_lazy_and_toolbars_compact(self):
        window = self._window()

        self.assertIsNone(window.video_annotation_workspace)
        self.assertLessEqual(
            len([action for action in window.toolBar.actions() if not action.isSeparator()]),
            12,
        )
        self.assertLessEqual(
            len([action for action in window.toolBar_2.actions() if not action.isSeparator()]),
            9,
        )
        self.assertTrue(window.workspace_tools_button.accessibleName())
        self.assertIs(window.workspace_tools_button, window.mode_tools_button)
        self.assertIn(window.actionDelete, window.image_more_button.menu().actions())
        self.assertIn(
            window.actionTracking_start,
            window.transaction_tracking_button.menu().actions(),
        )

    def test_canvas_first_layout_keeps_tools_available_without_clutter(self):
        window = self._window()

        self.assertEqual(
            window.toolBarArea(window.toolBar_2), QtCore.Qt.TopToolBarArea
        )
        self.assertGreater(window.toolBar_2.maximumWidth(), 118)

        inspector_docks = (
            window.categories_dock,
            window.annos_dock,
            window.files_dock,
            window.info_dock,
            window.text_prompt_dock,
            window.visual_prompt_dock,
        )
        for dock in inspector_docks:
            with self.subTest(dock=dock.objectName()):
                self.assertEqual(
                    window.dockWidgetArea(dock), QtCore.Qt.RightDockWidgetArea
                )
        tabbed_with_categories = set(
            window.tabifiedDockWidgets(window.categories_dock)
        )
        self.assertEqual(
            tabbed_with_categories,
            set(inspector_docks[1:]) | {window.temporal_dock},
        )

        for dock in inspector_docks[:3]:
            self.assertFalse(dock.isHidden(), dock.objectName())
        for dock in inspector_docks[3:]:
            self.assertTrue(dock.isHidden(), dock.objectName())

        panel_actions = window.panels_menu.actions()
        for widget in (
            *inspector_docks,
            window.temporal_dock,
            window.timeline_dock,
            window.toolBar,
            window.toolBar_2,
        ):
            with self.subTest(toggle=widget.objectName()):
                self.assertIn(widget.toggleViewAction(), panel_actions)
        self.assertIn(window.panels_menu.menuAction(), window.menuView.actions())
        self.assertIn(window.actionReset_layout, window.menuView.actions())

    def test_reset_layout_restores_shell_and_legacy_tracking_stays_off_menu(self):
        window = self._window()
        window.addToolBar(QtCore.Qt.LeftToolBarArea, window.toolBar_2)
        window.categories_dock.hide()
        window.info_dock.show()

        window._reset_workspace_layout()

        self.assertEqual(
            window.toolBarArea(window.toolBar_2), QtCore.Qt.TopToolBarArea
        )
        self.assertFalse(window.categories_dock.isHidden())
        self.assertTrue(window.info_dock.isHidden())
        for action in (
            window.actionVideo_segment,
            window.actionVideo_segment_once,
            window.actionVideo_segment_five_times,
        ):
            self.assertIsNotNone(action.parent())
            self.assertNotIn(action, window.menuEdit.actions())

    def test_secondary_dialogs_stay_lazy_until_explicitly_requested(self):
        window = self._window()

        for attribute_name in self.OPTIONAL_DIALOG_ATTRIBUTES:
            self.assertIsNone(getattr(window, attribute_name), attribute_name)

        for ensure_name, attribute_name in self.OPTIONAL_DIALOG_ENSURERS:
            with self.subTest(attribute_name=attribute_name):
                ensure = getattr(window, ensure_name)
                dialog = ensure()
                self.assertIs(getattr(window, attribute_name), dialog)
                self.assertIs(dialog, ensure())

    def test_video_workspace_disables_and_restores_image_shortcuts(self):
        window = self._window()
        window.actionBit_map.setEnabled(True)

        workspace = window._ensure_video_workspace()
        if workspace.player is not None:
            workspace.player.stop()
            workspace.player = None
        window._activate_video_workspace()

        self.assertFalse(window.actionBit_map.isEnabled())
        self.assertFalse(window.actionPolygon.isEnabled())
        window.show_image_workspace()
        self.assertTrue(window.actionBit_map.isEnabled())

        workspace.shutdown()

    def test_legacy_direct_write_tracking_is_retired_for_every_image_sequence(self):
        window = self._window()
        window.use_segment_anything_video = True
        for sparse_case in (False, True):
            with self.subTest(sparse_case=sparse_case):
                window._active_video_frame_case = sparse_case
                window._update_tracking_action_safety()
                self.assertFalse(window.actionVideo_segment.isEnabled())
                self.assertFalse(window.actionVideo_segment_once.isEnabled())
                self.assertFalse(window.actionVideo_segment_five_times.isEnabled())
                self.assertFalse(window.tracking_tools_button.isVisible())
                with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
                    window.seg_video_start()
                warning.assert_called_once()

    def test_inspector_tabs_keep_full_text_at_compact_and_wide_sizes(self):
        from IVAS.theme import apply_workbench_theme
        old_style = self.app.styleSheet()
        self.addCleanup(self.app.setStyleSheet, old_style)
        apply_workbench_theme(self.app)
        window = self._window()
        window.show()
        for language in ("zh", "en"):
            window.translate(language)
            for size in ((800, 600), (1440, 900)):
                window.resize(*size)
                for dock in window._inspector_docks:
                    dock.show()
                self.app.processEvents()
                bars = window.findChildren(QtWidgets.QTabBar, options=QtCore.Qt.FindDirectChildrenOnly)
                self.assertTrue(bars)
                for bar in bars:
                    if not bar.count():
                        continue
                    self.assertEqual(bar.elideMode(), QtCore.Qt.ElideNone)
                    self.assertTrue(bar.usesScrollButtons())
                    scroll_buttons = [button for button in bar.findChildren(QtWidgets.QToolButton)
                                      if button.isVisible()]
                    for button in scroll_buttons:
                        self.assertGreaterEqual(button.width(), 24)
                        self.assertTrue(bar.rect().contains(button.geometry()))
                    if len(scroll_buttons) == 2:
                        overlap = scroll_buttons[0].geometry().intersected(scroll_buttons[1].geometry())
                        self.assertLessEqual(overlap.width(), 1)  # shared border only
                    for index in range(bar.count()):
                        self.assertGreaterEqual(bar.tabRect(index).width(),
                                                bar.fontMetrics().horizontalAdvance(bar.tabText(index)))
                        bar.setCurrentIndex(index)
                        self.app.processEvents()
                        self.assertTrue(bar.rect().intersects(bar.tabRect(index)))


if __name__ == "__main__":
    unittest.main()
