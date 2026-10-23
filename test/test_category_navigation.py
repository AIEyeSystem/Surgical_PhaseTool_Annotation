"""Regressions for responsive navigation, display aliases and usable category controls."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets
import yaml

from IVAS.annotation import Object
from IVAS.configs import STATUSMode
from IVAS.category_names import load_category_aliases
from IVAS.theme import apply_workbench_theme
from IVAS.widgets import mainwindow
from IVAS.widgets.polygon import OBB, Polygon
from IVAS.widgets.text_prompt_dock_widget import TextPromptThread


class CategoryAliasTests(unittest.TestCase):
    def test_missing_and_partial_aliases_use_only_canonical_config(self):
        labels = [{"name": "__background__"}, {"name": "Cornea"}, {"name": "Forceps", "id": 8}]
        fallback = {"__background__": "0", "Cornea": "1", "Forceps": "8"}
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ivas.yaml"
            rename = config.with_name("ivas_rename.yaml")
            self.assertEqual(load_category_aliases(labels, config), fallback)
            for contents in ("label:\n- name: 背景\n- name: 角膜\n",
                             "label:\n- id: 8\n  name: 镊子\n", "Cornea: 角膜\n"):
                with self.subTest(contents=contents):
                    rename.write_text(contents, encoding="utf-8")
                    with self.assertLogs("IVAS.category_names", level="WARNING"):
                        self.assertEqual(load_category_aliases(labels, config), fallback)

    def test_complete_aliases_match_by_name_id_or_position(self):
        labels = [{"name": "Cornea", "color": "#AABBCC"}, {"name": "Forceps"}]
        expected = {"Cornea": "角膜", "Forceps": "镊子"}
        variants = [
            {"Cornea": "角膜", "Forceps": "镊子"},
            {0: "角膜", "1": "镊子"},
            {"label": [{"id": 1, "name": "镊子"}, {"id": 0, "name": "角膜"}]},
            {"label": [{"name": "角膜", "color": "#aabbcc"}, {"name": "镊子"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ivas.yaml"
            for data in variants:
                with self.subTest(data=data):
                    config.with_name("ivas_rename.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
                    self.assertEqual(load_category_aliases(labels, config), expected)

    def test_any_mismatch_discards_all_aliases_without_changing_files(self):
        labels = [{"name": "Cornea", "id": 0, "color": "#ff0000"}, {"name": "Forceps", "id": 1}]
        variants = [
            ["角膜", "镊子", "多余类别"],
            [{"id": 0, "name": "角膜"}, {"id": 7, "name": "镊子"}],
            [{"id": 0, "name": "角膜"}, {"id": 0, "name": "镊子"}],
            [{"id": 0, "name": "角膜"}, {"name": "镊子"}],
            [{"name": "角膜", "color": "#00ff00"}, {"name": "镊子"}],
            {"Cornea": "角膜", "Other": "镊子"},
            {"Cornea": "角膜", "Forceps": {"id": 8, "name": "镊子"}},
            ["角膜", None], ["角膜", ""], ["角膜", []], None,
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ivas.yaml"
            config.write_text(yaml.safe_dump({"label": labels}), encoding="utf-8")
            original = config.read_bytes()
            rename = config.with_name("ivas_rename.yaml")
            for data in variants:
                with self.subTest(data=data):
                    rename.write_text(yaml.safe_dump({"label": data}), encoding="utf-8")
                    rename_bytes = rename.read_bytes()
                    with self.assertLogs("IVAS.category_names", level="WARNING"):
                        self.assertEqual(load_category_aliases(labels, config), {"Cornea": "0", "Forceps": "1"})
                    self.assertEqual(config.read_bytes(), original)
                    self.assertEqual(rename.read_bytes(), rename_bytes)

    def test_invalid_rename_file_falls_back_to_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ivas.yaml"
            config.with_name("ivas_rename.yaml").write_text("label: [broken", encoding="utf-8")
            with self.assertLogs("IVAS.category_names", level="WARNING"):
                self.assertEqual(load_category_aliases([{"name": "Cornea"}], config), {"Cornea": "0"})


class CategoryNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.previous_stylesheet = cls.app.styleSheet()
        apply_workbench_theme(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.setStyleSheet(cls.previous_stylesheet)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "ivas.yaml"
        self.config.write_text(yaml.safe_dump({"label": [
            {"name": "__background__", "color": "#000000", "id": 0},
            {"name": "Cornea", "color": "#ff0000", "id": 1},
            {"name": "Forceps", "color": "#00ff00", "id": 2},
        ]}), encoding="utf-8")
        self.config.with_name("ivas_rename.yaml").write_text(
            "label:\n- name: 背景\n- name: 角膜\n- name: 镊子\n", encoding="utf-8")
        with mock.patch.object(mainwindow, "CONFIG_FILE", str(self.config)), \
             mock.patch.object(mainwindow.CheckLatestVersionThread, "start"):
            self.window = mainwindow.MainWindow()
        self.window._version_check_timer.stop()
        self.window.save_cfg = mock.Mock()
        self.window.save_software_cfg = mock.Mock()
        self.window.cfg["software"]["auto_save"] = False
        self.addCleanup(self._close_window)
        for index in range(4):
            Image.new("RGB", (32, 24), (index, index, index)).save(self.root / f"{index}.png")

    def _close_window(self):
        window = self.window
        worker = getattr(window, "seganythread", None)
        if worker is not None:
            worker.invalidate()
            worker.requestInterruption()
            worker.wait(6000)
        window.saved = True
        window.scene.cancel_draw()
        window.close()
        window.deleteLater()
        self.app.processEvents()

    def _wait(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.app.processEvents()
        self.assertTrue(predicate(), "Background work did not settle")

    def _select_category(self, name):
        dock = self.window.categories_dock_widget
        for row in range(dock.listWidget.count()):
            item = dock.listWidget.item(row)
            if item.data(QtCore.Qt.UserRole) == name:
                dock.item_choice(item)
                return
        self.fail("Missing category " + name)

    def _add_polygon(self, category="Cornea", group=7, shape=Polygon):
        polygon = shape()
        self.window.scene.addItem(polygon)
        polygon.load_object(Object(category, group, [[2, 2], [14, 2], [14, 14], [2, 14]],
                                   144, 2, [2, 2, 14, 14], note="keep note",
                                   object_id=f"object-{group}", track_id=f"track-{group}",
                                   source="tracking", review_status="reviewed"))
        self.window.polygons.append(polygon)
        self.window.annos_dock_widget.listwidget_add_polygon(polygon)
        return polygon

    def test_canvas_and_annos_selection_highlight_category_without_editing(self):
        window = self.window
        window.load_image_directory(str(self.root))
        polygon = self._add_polygon()
        dock = window.categories_dock_widget
        for origin in ("canvas", "annos"):
            with self.subTest(origin=origin):
                window.scene.clearSelection()
                self._select_category("Forceps")
                window.set_saved_state(True)
                if origin == "canvas":
                    polygon.setSelected(True)
                else:
                    window.annos_dock_widget.polygon_item_dict[polygon].setSelected(True)
                self.assertTrue(polygon.isSelected())
                self.assertEqual(window.current_category, "Cornea")
                item = dock.listWidget.currentItem()
                self.assertEqual(item.data(QtCore.Qt.UserRole), "Cornea")
                self.assertTrue(dock.listWidget.itemWidget(item).findChild(QtWidgets.QRadioButton).isChecked())
                self.assertEqual(window.text_prompt_dock_widget.lineEdit_prompt_text_for_single_category.text(), "Cornea")
                self.assertEqual(polygon.category, "Cornea")
                self.assertTrue(window.saved)
                self.assertFalse(dock.pushButton_change.isEnabled())

    def test_change_updates_selected_mask_annos_and_saved_json_preserving_identity(self):
        window = self.window
        window.load_image_directory(str(self.root))
        polygon, other = self._add_polygon(), self._add_polygon(group=3)
        annos, dock = window.annos_dock_widget, window.categories_dock_widget
        annos.comboBox_group_select.setCurrentText("7")
        polygon.setSelected(True)
        window.set_saved_state(True)
        before = polygon.to_object().to_dict()
        original_item = annos.polygon_item_dict[polygon]
        self._select_category("Forceps")
        self.assertEqual(polygon.category, "Cornea")
        self.assertTrue(window.saved)
        self.assertTrue(polygon.isSelected())
        self.assertTrue(dock.pushButton_change.isEnabled())
        dock.pushButton_change.click()
        self.assertEqual(polygon.category, "Forceps")
        self.assertEqual(polygon.color.name(), "#00ff00")
        self.assertEqual(other.category, "Cornea")
        self.assertFalse(other.isVisible())
        self.assertTrue(polygon.isSelected())
        self.assertIs(annos.polygon_item_dict[polygon], original_item)
        self.assertTrue(original_item.isSelected())
        self.assertEqual(annos.comboBox_group_select.currentText(), "7")
        category_label = annos.listWidget.itemWidget(original_item).findChild(QtWidgets.QLabel, "category")
        self.assertEqual(category_label.text(), "Forceps [镊子]")
        self.assertFalse(window.saved)
        self.assertFalse(dock.pushButton_change.isEnabled())
        after = polygon.to_object().to_dict()
        for key in ("segmentation", "group", "layer", "iscrowd", "note", "object_id", "track_id"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual((after["source"], after["origin_source"], after["review_status"]),
                         ("manual", "tracking", "draft"))
        self.assertEqual(window.current_label.objects[0].category, "Forceps")
        window.save()
        saved = json.loads((self.root / "0.json").read_text(encoding="utf-8"))
        self.assertEqual([obj["category"] for obj in saved["objects"]], ["Forceps", "Cornea"])
        self.assertEqual(saved["objects"][0]["track_id"], "track-7")
        window.scene.cancel_draw()
        window.show_image(0)
        self.assertEqual([p.category for p in window.polygons], ["Forceps", "Cornea"])

    def test_change_handles_multiple_selected_masks_and_obb(self):
        window = self.window
        window.load_image_directory(str(self.root))
        polygons = [self._add_polygon(), self._add_polygon(group=8, shape=OBB)]
        for polygon in polygons:
            polygon.setSelected(True)
        self.assertEqual(set(window.categories_dock_widget.selected_polygons()), set(polygons))
        self.assertEqual(len(window.annos_dock_widget.listWidget.selectedItems()), 2)
        self._select_category("Forceps")
        window.categories_dock_widget.pushButton_change.click()
        self.assertEqual([p.category for p in polygons], ["Forceps", "Forceps"])
        self.assertTrue(all(p.isSelected() for p in polygons))
        self.assertTrue(polygons[1].to_object().is_obb)

    def test_selection_reveals_filtered_category_and_unknown_categories_do_not_select_wrong_row(self):
        window = self.window
        window.load_image_directory(str(self.root))
        polygon = self._add_polygon()
        dock = window.categories_dock_widget
        dock.lineEdit_search_category.setText("zzzz")
        self.assertEqual(dock.listWidget.count(), 0)
        polygon.setSelected(True)
        self.assertEqual(dock.lineEdit_search_category.text(), "")
        self.assertEqual(dock.listWidget.currentItem().data(QtCore.Qt.UserRole), "Cornea")
        window.scene.clearSelection()
        unknown = self._add_polygon(category="NotInConfig", group=9)
        unknown.setSelected(True)
        self.assertEqual(window.current_category, "NotInConfig")
        self.assertIsNone(dock.listWidget.currentItem())
        self.assertFalse(any(button.isChecked() for button in dock.button_group.buttons()))
        self.assertFalse(dock.pushButton_change.isEnabled())
        self._select_category("Forceps")
        dock.pushButton_change.click()
        self.assertEqual(unknown.category, "Forceps")
        self.assertNotIn("NotInConfig", window.category_color_dict)

    def test_change_auto_saves_all_selected_masks_once(self):
        window = self.window
        window.load_image_directory(str(self.root))
        polygons = [self._add_polygon(), self._add_polygon(group=8)]
        for polygon in polygons:
            polygon.setSelected(True)
        self._select_category("Forceps")
        window.cfg["software"]["auto_save"] = True
        with mock.patch.object(window, "save", wraps=window.save) as save:
            window.categories_dock_widget.pushButton_change.click()
        save.assert_called_once()
        payload = json.loads((self.root / "0.json").read_text(encoding="utf-8"))
        self.assertEqual([obj["category"] for obj in payload["objects"]], ["Forceps", "Forceps"])
        self.assertTrue(window.saved)

    def test_change_is_disabled_without_selection_or_during_drawing(self):
        window = self.window
        window.load_image_directory(str(self.root))
        dock = window.categories_dock_widget
        self._select_category("Forceps")
        self.assertFalse(dock.pushButton_change.isEnabled())
        polygon = self._add_polygon()
        polygon.setSelected(True)
        self._select_category("Forceps")
        self.assertTrue(dock.pushButton_change.isEnabled())
        for mode in (STATUSMode.CREATE, STATUSMode.REPAINT):
            window.scene.mode = mode
            dock.change_selected_category()
            self.assertFalse(dock.pushButton_change.isEnabled())
            self.assertEqual(polygon.category, "Cornea")
        window.scene.mode = STATUSMode.EDIT
        window.scene.clearSelection()
        self.assertFalse(dock.pushButton_change.isEnabled())

    def test_segment_and_change_buttons_share_a_row_and_mouse_click_keeps_selection(self):
        window = self.window
        window.load_image_directory(str(self.root))
        polygon = self._add_polygon()
        window.show()
        self.app.processEvents()
        dock = window.categories_dock_widget
        self.assertEqual(dock.pushButton_segment.geometry().top(), dock.pushButton_change.geometry().top())
        self.assertFalse(dock.pushButton_segment.geometry().intersects(dock.pushButton_change.geometry()))
        polygon.setSelected(True)
        item = dock.listWidget.item(2)
        radio = dock.listWidget.itemWidget(item).findChild(QtWidgets.QRadioButton)
        QtTest.QTest.mouseClick(radio, QtCore.Qt.LeftButton)
        self.assertTrue(polygon.isSelected())
        self.assertEqual(polygon.category, "Cornea")
        QtTest.QTest.mouseClick(dock.pushButton_change, QtCore.Qt.LeftButton)
        self.assertEqual(polygon.category, "Forceps")
        self.assertTrue(polygon.isSelected())

    def test_category_selection_editing_and_saved_annotations_keep_canonical_names(self):
        window = self.window
        window.load_image_directory(str(self.root))
        self._select_category("Cornea")
        self.assertEqual(window.current_category, "Cornea")
        self.assertEqual(window.category_display_name("Cornea"), "Cornea [角膜]")
        polygon = Polygon()
        window.scene.addItem(polygon)
        polygon.load_object(Object(window.current_category, 1, [[1, 1], [12, 1], [12, 12]],
                                   60, 0, [1, 1, 12, 12]))
        window.polygons.append(polygon)
        editor = window._ensure_category_edit_dialog()
        editor.polygons = [polygon]
        editor.load_cfg()
        editor.get_category(editor.listWidget.item(2))
        self.assertEqual(editor.lineEdit_category.text(), "Forceps")
        window.save()
        objects = json.loads((self.root / "0.json").read_text(encoding="utf-8"))["objects"]
        self.assertEqual(objects[0]["category"], "Cornea")
        settings = window._ensure_category_setting_dialog()
        settings.load_cfg()
        settings.apply()
        saved = yaml.safe_load(self.config.read_text(encoding="utf-8"))["label"]
        self.assertEqual([label["name"] for label in saved], ["__background__", "Cornea", "Forceps"])
        self.assertEqual([label["id"] for label in saved], [0, 1, 2])

    def test_removing_category_invalidates_entire_rename_file(self):
        window = self.window
        for label in window.cfg["label"]:
            label.pop("id")
        settings = window._ensure_category_setting_dialog()
        settings.load_cfg()
        settings.category_list_widget.takeItem(1)
        settings.apply()
        self.assertEqual(window.category_display_name("Forceps"), "Forceps [2]")
        self.assertEqual(window.category_display_name("__background__"), "__background__ [0]")

    def test_queued_encoder_status_is_discarded_after_directory_invalidation(self):
        window = self.window
        window.load_image_directory(str(self.root))
        window.segany = SimpleNamespace(model_source="sam3.1")
        worker = mainwindow.SegAnyThread(window)
        window.seganythread = worker
        worker.sam_encoder = lambda image: ("cached", image.shape[:2], image.shape[:2])
        observed = []
        worker.tag.connect(lambda *status: observed.append(status))
        worker.request_image(0)
        self.assertTrue(worker.wait(2000))
        worker.invalidate()
        self.app.processEvents()
        self.assertEqual(observed, [])

    def test_category_button_and_automatic_manual_prompts(self):
        window = self.window
        dock = window.text_prompt_dock_widget
        self._select_category("Cornea")
        self.assertEqual(dock.lineEdit_prompt_text_for_single_category.text(), "Cornea")
        with mock.patch.object(dock, "_start_prediction") as start:
            window.categories_dock_widget.pushButton_segment.click()
            start.assert_called_once_with([("Cornea", "Cornea")])
        dock.checkBox_use_selected_category.setChecked(False)
        dock.lineEdit_prompt_text_for_single_category.setText("surgical instrument")
        self._select_category("Forceps")
        self.assertEqual(dock.lineEdit_prompt_text_for_single_category.text(), "surgical instrument")
        with mock.patch.object(dock, "_start_prediction") as start:
            dock.predict_for_single_category()
            start.assert_called_once_with([("Forceps", "surgical instrument")])
        dock._choose_prompt(dock.comboBox_prompt.findData("Cornea"))
        self.assertEqual(dock.lineEdit_prompt_text_for_single_category.text(), "Cornea")
        dock.checkBox_use_selected_category.setChecked(True)
        self.assertEqual(dock.lineEdit_prompt_text_for_single_category.text(), "Forceps")

    def _blocked_encoder(self):
        window = self.window
        window.load_image_directory(str(self.root))
        lock = threading.RLock()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        predictor = SimpleNamespace(set_cached_state=mock.Mock())
        window.segany = SimpleNamespace(model_source="sam3.1", inference_lock=lock, predictor=predictor)
        window.use_segment_anything = True
        worker = mainwindow.SegAnyThread(window)
        window.seganythread = worker
        worker.tag.connect(window.sam_encoder_finish)
        worker.finished.connect(window.SeganyEnabled)
        calls = []

        def encode(image):
            with lock:
                calls.append(int(image[0, 0, 0]))
                if len(calls) == 1:
                    entered.set()
                    release.wait(5)
                return f"frame:{calls[-1]}", image.shape[:2], image.shape[:2]

        worker.sam_encoder = encode
        window.show_image(0)
        self.assertTrue(entered.wait(2))
        return worker, release, calls

    def test_repeated_navigation_stays_responsive_and_encodes_latest_frame(self):
        window = self.window
        worker, release, calls = self._blocked_encoder()
        started = time.monotonic()
        for _ in range(20):
            window.next_image()
            window.prev_image()
        window.show_image(3)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(window.actionSegment_anything_point.isEnabled())
        ticks = []
        QtCore.QTimer.singleShot(0, lambda: ticks.append(True))
        QtTest.QTest.qWait(30)
        self.assertEqual(ticks, [True])
        release.set()
        self._wait(lambda: not worker.isRunning() and 3 in worker.results_dict)
        self.assertEqual(calls, [0, 3])
        self.assertEqual(worker.results_dict[3]["features"], "frame:3")
        self.assertTrue(window.actionSegment_anything_point.isEnabled())

    def test_directory_change_does_not_wait_or_publish_old_features(self):
        worker, release, calls = self._blocked_encoder()
        another = self.root / "another"
        another.mkdir()
        Image.new("RGB", (32, 24), (9, 9, 9)).save(another / "0.png")
        started = time.monotonic()
        self.assertTrue(self.window.load_image_directory(str(another)))
        self.assertLess(time.monotonic() - started, 1)
        release.set()
        self._wait(lambda: not worker.isRunning() and 0 in worker.results_dict)
        self.assertEqual(calls, [0, 9])
        self.assertEqual(worker.results_dict[0]["image_path"], str(another / "0.png"))
        self.assertEqual(worker.results_dict[0]["features"], "frame:9")

    def test_cached_features_wait_asynchronously_for_inference_lock(self):
        worker, release, _ = self._blocked_encoder()
        worker.results_dict[1] = {"features": "cached", "original_size": (24, 32), "input_size": (24, 32)}
        self.window.current_index = 1
        started = time.monotonic()
        self.window.SeganyEnabled()
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(self.window._sam_ready_timer.isActive())
        self.assertFalse(self.window.actionSegment_anything_point.isEnabled())
        release.set()
        self._wait(lambda: not worker.isRunning())
        self.window.current_index = 0
        self.window.SeganyEnabled()
        self.assertTrue(self.window.actionSegment_anything_point.isEnabled())

    def test_text_job_uses_captured_frame_and_leaving_invalidates_result(self):
        window = self.window
        window.load_image_directory(str(self.root))
        dock = window.text_prompt_dock_widget
        dock._job_context = (1, window.sam_model_generation, 0, "0.png",
                             os.path.normcase(str(self.root / "0.png")))
        window.next_image()
        window.prev_image()
        self.assertFalse(dock._context_is_current(1))
        predictor = SimpleNamespace(set_cached_state=mock.Mock())
        segany = SimpleNamespace(model_source="sam3.1", predictor=predictor,
                                 inference_lock=threading.RLock(),
                                 predict_with_text_prompts=mock.Mock(return_value=[([], [])]))
        worker = TextPromptThread(1, segany, str(self.root / "0.png"), [("Cornea", "cornea")],
                                  cached_state="captured-frame")
        worker.run()
        predictor.set_cached_state.assert_called_once_with("captured-frame")

    def test_category_rows_and_dialog_footers_fit_themed_controls(self):
        window = self.window
        dialog = window._ensure_category_setting_dialog()
        dialog.load_cfg()
        dialog.show()
        self.app.processEvents()
        previous = None
        for index in range(dialog.category_list_widget.count()):
            item = dialog.category_list_widget.item(index)
            row = dialog.category_list_widget.itemWidget(item)
            self.assertGreaterEqual(row.height(), row.minimumSizeHint().height())
            for button in row.findChildren(QtWidgets.QPushButton):
                self.assertGreaterEqual(button.height(), button.minimumSizeHint().height())
                self.assertTrue(row.rect().contains(button.geometry()))
            rect = dialog.category_list_widget.visualItemRect(item)
            if previous is not None:
                self.assertFalse(previous.intersects(rect))
            previous = rect
        dialog.hide()
        for dialog in (window.setting_dialog, window.plugin_manager_dialog,
                       window._ensure_model_manager_dialog(), window._ensure_shortcut_dialog()):
            dialog.show()
            self.app.processEvents()
            for button in dialog.findChildren(QtWidgets.QPushButton):
                if button.isVisible():
                    button.setFocus()
                    self.app.processEvents()
                    self.assertGreaterEqual(button.height(), button.minimumSizeHint().height(),
                                            f"{type(dialog).__name__}: {button.objectName()}")
                    self.assertTrue(button.parentWidget().rect().contains(button.geometry()),
                                    f"{type(dialog).__name__}: {button.objectName()} outside parent")
            dialog.hide()


if __name__ == "__main__":
    unittest.main()

