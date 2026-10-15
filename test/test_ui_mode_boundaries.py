"""Mode guards and screen geometry regressions, without loading a SAM model."""

import os
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from IVAS.annotation import Object
from IVAS.configs import DRAWMode, MAPMode, STATUSMode
from IVAS.theme import apply_workbench_theme
from IVAS.widgets.polygon import Polygon
from IVAS.widgets.video_annotation_dialogs import (TrackingRangeDialog, TrackingReviewDialog,
    TemplateEditorDialog, TemplateManagerDialog, TemporalAnnotationDialog)
from test import test_annotation_modes as fixtures
from test import test_mainwindow_shell as shell_fixtures


class WorkbenchFixture(unittest.TestCase):
    setUpClass = classmethod(fixtures.AnnotationModeTests.setUpClass.__func__)
    setUp = fixtures.AnnotationModeTests.setUp
    _close_window = fixtures.AnnotationModeTests._close_window
    _images = fixtures.AnnotationModeTests._images
    _video = fixtures.AnnotationModeTests._video
    _wait = fixtures.AnnotationModeTests._wait

    def _load_image(self):
        self.assertTrue(self.window.load_image_directory(str(self._images("images"))))

    def _polygon(self):
        polygon = Polygon()
        self.window.scene.addItem(polygon)
        polygon.load_object(Object("lens", 1, [[8, 8], [32, 8], [32, 32]], 288, 1, [8, 8, 32, 32]))
        self.window.polygons.append(polygon)
        self.window.annos_dock_widget.listwidget_add_polygon(polygon)
        return polygon


class CanvasModeBoundaryTests(WorkbenchFixture):
    def test_language_and_mode_changes_preserve_pending_interval_caption(self):
        workspace = self._video()
        workspace.label_combo.setCurrentText("Preparation")
        workspace.seek(100)
        workspace.mark_start()
        pending = workspace._pending_interval
        self.window.translate("en")
        self.assertIn("In progress:", workspace.pending_label.text())
        self.assertIn("00:00:00.100", workspace.pending_label.text())
        self.window.request_frame_mode()
        self._wait(lambda: self.window.annotation_mode == "Frame" and not workspace._busy)
        self.window.translate("zh")
        self.assertIn("进行中", workspace.pending_label.text())
        self.assertEqual(workspace._pending_interval, pending)

    def test_temporal_label_sync_preserves_typing_cursor_and_selection(self):
        workspace = self._video()
        combo = self.window.temporal_label_combo
        combo.setCurrentText("Preparation")
        editor = combo.lineEdit()
        editor.setCursorPosition(3)
        editor.insert("X")
        self.assertEqual(editor.text(), "PreXparation")
        self.assertEqual(editor.cursorPosition(), 4)
        editor.setSelection(1, 3)
        self.window._sync_video_context_actions()
        self.assertEqual(editor.selectedText(), "reX")
        self.assertEqual(workspace.label_combo.currentText(), "PreXparation")

    def test_reloaded_bitmap_shortcut_survives_video_scope(self):
        self._video()
        window = self.window
        self.assertIsNotNone(window._suspended_bitmap_shortcut)
        window.cfg["shortcut"]["actionBit_map"]["key"] = "Ctrl+9"
        window.load_actions_shortcut()
        self.assertTrue(window.actionBit_map.shortcut().isEmpty())
        window.show_image_workspace()
        self.assertEqual(window.actionBit_map.shortcut().toString(), "Ctrl+9")

    def test_corrupt_next_image_preserves_canvas_and_display_mode(self):
        self._load_image()
        window = self.window
        (Path(window.image_root) / "z-broken.png").write_bytes(b"not an image")
        window.files_list.append("z-broken.png")
        window.change_bit_map_to_semantic()
        image_item = window.scene.image_item
        current_label = window.current_label
        with mock.patch.object(QtWidgets.QMessageBox, "warning"):
            window.show_image(1)
        self.assertIs(window.scene.image_item, image_item)
        self.assertIs(window.current_label, current_label)
        self.assertEqual(window.current_index, 0)
        self.assertEqual(window.map_mode, MAPMode.SEMANTIC)

    def test_drawing_tool_switch_does_not_mutate_unfinished_polygon(self):
        self._load_image()
        scene = self.window.scene
        scene.start_draw_polygon()
        graph = scene.current_graph
        graph.addPoint(QtCore.QPointF(8, 8))
        for start in (scene.start_segment_anything, scene.start_segment_anything_box,
                      scene.start_segment_anything_visual, scene.start_draw_obb,
                      scene.change_mode_to_repaint):
            start()
            self.assertEqual(scene.draw_mode, DRAWMode.POLYGON)
            self.assertEqual(scene.mode, STATUSMode.CREATE)
            self.assertIs(scene.current_graph, graph)
            self.assertEqual(len(graph.points), 1)
            self.assertIsNone(scene.prompt_box_item)
            self.assertIsNone(scene.prompt_visual_current_item)

    def test_preview_switch_preserves_unfinished_polygon_and_repaint(self):
        self._load_image()
        scene = self.window.scene
        scene.start_draw_polygon()
        graph = scene.current_graph
        graph.addPoint(QtCore.QPointF(8, 8))
        self.window.change_bit_map()
        self.assertEqual(self.window.map_mode, MAPMode.LABEL)
        self.assertIs(scene.current_graph, graph)
        scene.cancel_draw()
        scene.change_mode_to_repaint()
        self.window.change_bit_map_to_instance()
        self.assertEqual(self.window.map_mode, MAPMode.LABEL)
        self.assertEqual(scene.mode, STATUSMode.REPAINT)

    def test_preview_rejects_drawing_repaint_and_late_encoder_enable(self):
        self._load_image()
        window = self.window
        window.segany = types.SimpleNamespace(inference_lock=threading.RLock())
        window.seganythread = types.SimpleNamespace(results_dict={0: {
            "features": object(), "original_size": (48, 64), "input_size": (48, 64),
        }})
        window.use_segment_anything = True
        self.addCleanup(setattr, window, "use_segment_anything", False)
        for switch in (window.change_bit_map_to_semantic, window.change_bit_map_to_instance):
            switch()
            window.SeganyEnabled()
            for start in (window.scene.start_draw_obb, window.scene.start_draw_polygon,
                          window.scene.start_segment_anything_visual, window.scene.change_mode_to_repaint):
                start()
            self.assertEqual(window.scene.mode, STATUSMode.VIEW)
            self.assertIsNone(window.scene.current_graph)
            self.assertFalse(window.actionPolygon.isEnabled())
            self.assertFalse(window.actionRepaint.isEnabled())
            self.assertFalse(window.actionSegment_anything_point.isEnabled())
            self.assertFalse(window.actionSegment_anything_box.isEnabled())

    def test_preview_deselects_edit_and_label_restores_only_valid_actions(self):
        self._load_image()
        polygon = self._polygon()
        polygon.setSelected(True)
        self.assertEqual(self.window.scene.mode, STATUSMode.EDIT)
        self.window.change_bit_map_to_semantic()
        self.assertFalse(polygon.isSelected())
        self.assertFalse(self.window.actionDelete.isEnabled())
        self.assertFalse(self.window.actionCopy.isEnabled())
        self.window.use_segment_anything_video = True
        self.window.change_bit_map_to_label()
        self.assertTrue(self.window.actionPolygon.isEnabled())
        self.assertFalse(self.window.actionVideo_segment.isEnabled())
        self.assertFalse(self.window.actionPrev_image.isEnabled())
        self.assertFalse(self.window.actionNext_image.isEnabled())

    def test_display_change_invalidates_text_job_even_after_return_to_label(self):
        self._load_image()
        window = self.window
        dock = window.text_prompt_dock_widget
        dock._job_context = (4, window.sam_model_generation, 0, "same.png",
                             os.path.normcase(str(Path(window.image_root) / "same.png")))
        self.assertTrue(dock._context_is_current(4))
        window.change_bit_map_to_semantic()
        window.change_bit_map_to_label()
        self.assertFalse(dock._context_is_current(4))

    def test_video_return_restores_frame_title_and_readonly_state(self):
        self._load_image()
        window = self.window
        frame_title = window.windowTitle()
        window.change_bit_map_to_semantic()
        window.show_video_workspace()
        window.scene.cancel_draw()
        self.assertFalse(window.actionPolygon.isEnabled())
        window.request_frame_mode()
        self.assertEqual(window.windowTitle(), frame_title)
        self.assertEqual(window.map_mode, MAPMode.SEMANTIC)
        self.assertFalse(window.actionPolygon.isEnabled())
        window.change_bit_map_to_label()
        self.assertTrue(window.annos_dock_widget.isEnabled())
        self.assertTrue(window.actionPolygon.isEnabled())

    def test_palette_image_is_displayed_without_enabling_annotation(self):
        directory = self.root / "palette"
        directory.mkdir()
        image = Image.new("P", (64, 48))
        image.putpalette([value for i in range(256) for value in (i, i, i)])
        image.save(directory / "label.png")
        self.window.load_image_directory(str(directory))
        self.assertTrue(self.window.load_finished)
        self.assertEqual(self.window.scene.image_data.shape, (48, 64, 3))
        self.assertFalse(self.window.can_be_annotated)
        self.window.scene.start_draw_obb()
        self.assertEqual(self.window.scene.mode, STATUSMode.VIEW)
        self.assertFalse(self.window.actionPolygon.isEnabled())

    def test_focus_loss_clears_modifiers_and_modified_obb_key_is_ignored(self):
        self._load_image()
        view = self.window.view
        view.ctrl_pressed = view.shift_pressed = True
        self.window.scene.ctrl_pressed = self.window.scene.shift_pressed = True
        view.focusOutEvent(QtGui.QFocusEvent(QtCore.QEvent.FocusOut))
        self.assertFalse(view.ctrl_pressed or view.shift_pressed)
        self.assertFalse(self.window.scene.ctrl_pressed or self.window.scene.shift_pressed)
        view.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_O,
                                         QtCore.Qt.ControlModifier))
        self.assertEqual(self.window.scene.mode, STATUSMode.VIEW)

    def test_fit_follows_resize_but_manual_zoom_survives(self):
        self._load_image()
        window = self.window
        window.show()
        window.resize(800, 600)
        self.app.processEvents()
        window.view.zoomfit()
        before = window.view.transform().m11()
        window.resize(1200, 850)
        self.app.processEvents()
        self.assertGreater(window.view.transform().m11(), before)
        window.view.zoom_out()
        manual = window.view.transform().m11()
        window.resize(900, 650)
        self.app.processEvents()
        self.assertAlmostEqual(window.view.transform().m11(), manual)


class ResponsiveLayoutTests(WorkbenchFixture):
    def setUp(self):
        super().setUp()
        stylesheet = self.app.styleSheet()
        self.addCleanup(self.app.setStyleSheet, stylesheet)
        apply_workbench_theme(self.app)

    def test_all_secondary_dialogs_fit_screen_and_keep_overflow_reachable(self):
        window = self.window
        workspace = self._video()
        for language in ("zh", "en"):
            window.translate(language)
            dialogs = [window.setting_dialog, window.plugin_manager_dialog]
            dialogs.extend(getattr(window, ensure)() for ensure, _ in
                           shell_fixtures.MainWindowShellTests.OPTIONAL_DIALOG_ENSURERS)
            dialogs.extend((TrackingRangeDialog(100, [], window),
                            TrackingReviewDialog({}, lambda _: None, Path, parent=window),
                            TemplateEditorDialog(parent=window),
                            TemplateManagerDialog(workspace.store, window),
                            TemporalAnnotationDialog(workspace.store.temporal_templates, 1200, 100, parent=window)))
            for dialog in dialogs:
                with self.subTest(language=language, dialog=type(dialog).__name__):
                    dialog.show()
                    QtTest.QTest.qWait(30)
                    self.assertTrue(dialog.screen().availableGeometry().contains(dialog.frameGeometry()),
                                    str(dialog.frameGeometry()))
                    scroll = dialog.findChild(QtWidgets.QScrollArea, "responsiveDialogScroll")
                    self.assertIsNotNone(scroll)
                    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
                    self.app.processEvents()
                    self.assertGreaterEqual(scroll.widget().height(), scroll.viewport().height())
                    dialog.hide()

    def test_shortcut_reflow_preserves_editor_values(self):
        dialog = self.window._ensure_shortcut_dialog()
        dialog.show()
        self.app.processEvents()
        editors = list(dialog.findChildren(QtWidgets.QKeySequenceEdit))
        editors[0].setKeySequence(QtGui.QKeySequence("Ctrl+9"))
        for width in (520, 1100, 650):
            dialog.resize(width, 400)
            self.app.processEvents()
            self.assertEqual(editors[0].keySequence().toString(), "Ctrl+9")
            self.assertEqual(len(dialog.findChildren(QtWidgets.QKeySequenceEdit)), len(editors))
        self.assertLess(dialog.columns, 3)
        dialog.hide()

    def test_video_frame_sizes_and_transport_do_not_depend_on_hidden_surface(self):
        workspace = self._video()
        window = self.window
        window.show()
        for language in ("zh", "en"):
            window.translate(language)
            for size in ((640, 480), (800, 600), (1440, 600)):
                window.resize(*size)
                self.app.processEvents()
                self.assertEqual((window.width(), window.height()), size)
                for control in (workspace.play_button, workspace.back_button, workspace.forward_button,
                                workspace.position_slider, workspace.rate_combo):
                    self.assertTrue(control.isVisible(), control.objectName())
                    self.assertTrue(workspace.transport_widget.rect().contains(control.geometry()),
                                    str(control.geometry()))
        window.request_frame_mode()
        self._wait(lambda: window.annotation_mode == "Frame" and not workspace._busy)
        window.resize(640, 480)
        self.app.processEvents()
        self.assertEqual((window.width(), window.height()), (640, 480))
        self.assertTrue(workspace.play_button.isVisible())
        self.assertGreater(window.view.viewport().height(), 50)


if __name__ == "__main__":
    unittest.main()
