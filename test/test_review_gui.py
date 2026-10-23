"""Exercise editing and worker lifetime through real offscreen widgets."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from shapely.geometry import Polygon as GeometryPolygon
from shapely.ops import unary_union

from IVAS.annotation import Object
from IVAS.widgets import mainwindow
from IVAS.widgets.polygon import Polygon
from test.test_ui_mode_boundaries import WorkbenchFixture


class ReviewGuiTests(WorkbenchFixture):
    def polygon(self, points):
        polygon = Polygon()
        self.window.scene.addItem(polygon)
        polygon.load_object(Object("lens", 1, points, 0, 2, [], source="tracking", review_status="draft"))
        self.window.polygons.append(polygon)
        self.window.annos_dock_widget.update_listwidget()
        return polygon

    def test_corrupt_label_does_not_replace_current_canvas_or_source_file(self):
        self._load_image()
        polygon = self._polygon()
        self.window.save()
        other = self._images("broken")
        label = other / "same.json"
        label.write_text('{"info":', encoding="utf-8")
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.assertFalse(self.window.load_image_directory(str(other)))
        self.assertTrue(warning.called)
        self.assertIs(self.window.polygons[0], polygon)
        self.window.save()
        self.assertEqual(label.read_text(), '{"info":')

    def test_vertex_delete_marks_dirty_and_preserves_tracking_origin(self):
        self._load_image()
        polygon = self.polygon([[2, 2], [20, 2], [20, 20], [2, 20]])
        self.window.save()
        self.window.scene.change_mode_to_edit()
        polygon.vertices[0].setSelected(True)
        self.window.scene.delete_selected_graph()
        self.assertEqual(len(polygon.points), 3)
        self.assertFalse(self.window.saved)
        edited = polygon.to_object()
        self.assertEqual((edited.source, edited.origin_source, edited.review_status), ("manual", "tracking", "draft"))
        self.window.save()
        data = json.loads(Path(self.window.current_label.label_path).read_text())
        self.assertEqual(len(data["objects"][0]["segmentation"]), 3)

    def test_vertex_move_marks_manual_and_updates_area(self):
        self._load_image()
        polygon = self.polygon([[2, 2], [20, 2], [20, 20], [2, 20]])
        self.window.save()
        before = polygon.calculate_area()
        polygon.movePoint(1, QtCore.QPointF(25, 2))
        self.assertFalse(self.window.saved)
        self.assertNotEqual(polygon.calculate_area(), before)
        self.assertEqual(polygon.to_object().source, "manual")

    def test_difference_keeps_hole_and_nonoverlapping_instance_parts(self):
        self._load_image()
        outer = self.polygon([[2, 2], [22, 2], [22, 22], [2, 22]])
        inner = self.polygon([[7, 7], [17, 7], [17, 17], [7, 17]])
        outer.object_id, outer.track_id = "object", "track"
        self.window.scene.selected_polygons_list[:] = [outer, inner]
        self.window.scene.polygons_difference()
        geometries = [GeometryPolygon(p.to_object().segmentation) for p in self.window.polygons]
        result = unary_union(geometries)
        self.assertAlmostEqual(result.area, 300)
        self.assertAlmostEqual(sum(p.area for p in geometries), 300)
        self.assertEqual(len(result.interiors), 1)
        self.assertEqual({p.track_id for p in self.window.polygons}, {"track"})

    def test_same_path_image_replacement_invalidates_encoder_cache(self):
        path = self.root / "image.png"
        Image.new("RGB", (8, 8), "red").save(path)
        owner = SimpleNamespace(segany=SimpleNamespace(model_source="sam"), image_root=str(self.root),
                                files_list=[path.name], _shutdown_started=False)
        worker = mainwindow.SegAnyThread(owner)
        worker.blockSignals(True)
        calls = []
        worker.sam_encoder = lambda data: (calls.append(data[0, 0].tolist()) or data.copy(), (8, 8), (8, 8))
        with mock.patch.object(worker, "start"):
            worker.request_image(0); worker.run()
            Image.new("RGB", (8, 8), "blue").save(path)
            worker.request_image(0); worker.run()
        self.assertEqual(calls, [[255, 0, 0], [0, 0, 255]])
        np.testing.assert_array_equal(worker.results_dict[0]["features"], np.asarray(Image.open(path)))

    def test_close_requests_cancellation_and_waits_for_text_worker(self):
        self._load_image()
        class WaitingWorker(QtCore.QThread):
            def __init__(self, parent):
                super().__init__(parent)
                self.done = threading.Event()
            def run(self):
                self.done.wait(3)
        dock = self.window.text_prompt_dock_widget
        worker = dock._worker = WaitingWorker(dock)
        worker.start()
        event = QtGui.QCloseEvent()
        try:
            self.window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.assertTrue(worker.isInterruptionRequested())
            self.assertTrue(worker.isRunning())
        finally:
            worker.done.set()
            self.assertTrue(worker.wait(3000))
            dock._worker = None
            self.app.processEvents()

    def test_conversion_runs_off_gui_thread_and_heartbeat_continues(self):
        from IVAS.widgets.converter_dialog import YOLOConverter
        from IVAS.widgets.converter_dialog import ConverterDialog
        dialog = ConverterDialog(self.window, self.window)
        self.addCleanup(dialog.deleteLater)
        entered, release = threading.Event(), threading.Event()
        threads = []
        def slow_convert(converter):
            threads.append(QtCore.QThread.currentThread())
            entered.set()
            release.wait(3)
        dialog.tabWidget.setCurrentWidget(dialog.tab_YOLO)
        dialog.toolBox_yolo.setCurrentWidget(dialog.toolbox_item_yolo2ivas)
        for editor in (dialog.lineEdit_yolo2ivas_yolo_image_root, dialog.lineEdit_yolo2ivas_yolo_txt_root,
                       dialog.lineEdit_yolo2ivas_ivas_json_root):
            editor.setText(str(self.root))
        ticks = []
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        try:
            with mock.patch.object(YOLOConverter, "yolo_to_ivas", slow_convert):
                dialog.apply()
                self._wait(entered.is_set)
                QtTest.QTest.qWait(40)
                self.assertGreater(len(ticks), 1)
                self.assertFalse(dialog.pushButton_start.isEnabled())
                self.assertIsNot(threads[0], self.app.thread())
                event = QtGui.QCloseEvent()
                dialog.closeEvent(event)
                self.assertFalse(event.isAccepted())
                self.assertTrue(dialog.converter.cancel)
        finally:
            release.set()
            self._wait(lambda: not dialog.converter.isRunning())
            timer.stop()
        self.assertTrue(dialog.pushButton_start.isEnabled())

    def test_empty_exif_output_is_rejected_before_launch(self):
        from IVAS.widgets.process_exif_dialog import ProcessExifDialog
        dialog = ProcessExifDialog(self.window, self.window)
        self.addCleanup(dialog.deleteLater)
        dialog.lineEdit_image_root.setText(str(self.root))
        dialog.lineEdit_save_root.clear()
        with mock.patch.object(dialog, "start_task") as start:
            dialog.start()
        start.assert_not_called()
        self.assertIn("Error", dialog.textBrowser.toPlainText())

    def test_plugins_isolate_notifications_but_abort_rejected_save(self):
        manager = self.window.plugin_manager_dialog
        broken = SimpleNamespace(enabled=True, after_annotations_saved_event=mock.Mock(side_effect=ValueError("boom")),
                                 before_annotations_save_event=mock.Mock(side_effect=ValueError("rejected")))
        healthy = SimpleNamespace(enabled=True, after_annotations_saved_event=mock.Mock())
        manager.plugins = [broken, healthy]
        manager.trigger_after_annotations_saved()
        healthy.after_annotations_saved_event.assert_called_once()
        with self.assertRaises(RuntimeError):
            manager.trigger_before_annotations_save()
        manager.plugins = []

    def test_repaint_guide_vertex_removal_does_not_mark_annotation_dirty(self):
        from IVAS.widgets.polygon import Line
        self._load_image()
        line = Line()
        self.window.scene.addItem(line)
        line.addPoint(QtCore.QPointF(2, 2))
        line.addPoint(QtCore.QPointF(4, 4))
        line.removePoint(0)
        self.assertEqual(len(line.points), 1)
        self.assertTrue(self.window.saved)
