"""Category index, batch edits, history/undo and the Files dock filter."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from IVAS import batch_ops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tmp" / "batch-ops"


def label(*objects, note="", description="IVAS", **info):
    data = {"info": {"description": description, "folder": "", "name": "x.png",
                     "width": 64, "height": 48, "depth": 3, "note": note, **info},
            "objects": list(objects)}
    return data


def obj(category, x=2, **extra):
    return {"category": category, "group": 1,
            "segmentation": [[x, 2], [x + 10, 2], [x + 10, 12], [x, 12]],
            "area": 100.0, "layer": 1, "bbox": [x, 2, x + 10, 12], "iscrowd": False,
            "note": "", **extra}


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=str(TMP_ROOT))
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name, data):
        path = self.root / name
        if isinstance(data, (dict, list)):
            path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(data, encoding="utf-8")
        return path

    def load(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))


class LabelIndexTests(TempDirTestCase):
    def test_scan_counts_categories_and_reports_bad_files(self):
        self.write("a.json", label(obj("lens"), obj("lens", 20), obj("tool")))
        self.write("b.json", label())
        self.write("c.json", '{"info":')
        self.write("d.json", {"images": [], "annotations": []})
        self.write("e.json", label(description="COCO"))
        self.write("f.json", {"info": {"description": "ISAT"}, "objects": [obj("lens")]})
        index = batch_ops.scan_label_dir(self.root, ["a", "b", "c", "d", "e", "f", "missing"])
        self.assertEqual(index.counts["a"], {"lens": 2, "tool": 1})
        self.assertEqual(index.counts["b"], {})
        self.assertEqual(index.counts["f"], {"lens": 1})  # legacy description
        self.assertEqual(set(index.errors), {"c", "d", "e"})
        self.assertIn("invalid JSON", index.errors["c"])
        self.assertIn("not an IVAS annotation", index.errors["e"])
        self.assertFalse(index.has_label("missing"))
        self.assertEqual(index.image_counts(), {"lens": 2, "tool": 1})
        self.assertEqual(index.object_counts(), {"lens": 3, "tool": 1})
        self.assertEqual(index.stems_with("lens"), ["a", "f"])
        # Unrestricted scans index every top-level JSON but skip hidden files.
        self.write(".hidden.json", label(obj("lens")))
        everything = batch_ops.scan_label_dir(self.root)
        self.assertEqual(set(everything.stems), {"a", "b", "c", "d", "e", "f"})

    def test_missing_directory_is_reported_not_raised(self):
        index = batch_ops.scan_label_dir(self.root / "nope", ["a"])
        self.assertTrue(index.root_error)
        self.assertEqual(index.counts, {})

    def test_rescan_only_parses_changed_files(self):
        for name in "abc":
            self.write(name + ".json", label(obj("lens")))
        first = batch_ops.scan_label_dir(self.root, list("abc"))
        with mock.patch.object(batch_ops, "read_label", wraps=batch_ops.read_label) as reader:
            second = batch_ops.scan_label_dir(self.root, list("abc"), previous=first)
            self.assertEqual(reader.call_count, 0)
            self.assertEqual(second.counts, first.counts)
            self.write("b.json", label(obj("lens"), obj("tool")))
            third = batch_ops.scan_label_dir(self.root, list("abc"), previous=second)
            self.assertEqual(reader.call_count, 1)
        self.assertEqual(third.counts["b"], {"lens": 1, "tool": 1})

    def test_cancelled_scan_returns_none(self):
        self.write("a.json", label(obj("lens")))
        event = mock.Mock(is_set=mock.Mock(return_value=True))
        self.assertIsNone(batch_ops.scan_label_dir(self.root, ["a"], cancelled=event))


class BatchEditTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.a = self.write("a.json", label(
            obj("lens", custom={"keep": [1, 2]}),
            obj("tool", shape_type="obb", is_obb=True, object_id="o-1"),
            obj("lens", 30), custom_info="kept"))
        self.b = self.write("b.json", label(obj("lens"), obj("gauze")))
        self.c = self.write("c.json", label(obj("tool")))
        self.originals = {name: (self.root / name).read_bytes() for name in ("a.json", "b.json", "c.json")}

    def history(self):
        return batch_ops.read_history(self.root)

    def test_delete_category_limited_to_images_keeps_unknown_keys(self):
        progress = []
        result = batch_ops.delete_category(self.root, "lens", stems=["a", "c"], scope="filtered",
                                           progress=lambda done, total: progress.append((done, total)))
        self.assertEqual((result["images"], result["objects"]), (1, 2))
        self.assertTrue(progress)
        self.assertEqual(progress[-1][0], progress[-1][1])  # finishes at 100 %
        data = self.load("a.json")
        self.assertEqual([o["category"] for o in data["objects"]], ["tool"])
        self.assertEqual(data["objects"][0]["shape_type"], "obb")
        self.assertEqual(data["objects"][0]["object_id"], "o-1")
        self.assertEqual(data["info"]["custom_info"], "kept")
        self.assertEqual(self.b.read_bytes(), self.originals["b.json"])  # out of scope
        self.assertEqual(self.c.read_bytes(), self.originals["c.json"])  # no lens

        # Backup + history record
        record = self.history()[-1]
        self.assertEqual(record["id"], result["id"])
        self.assertEqual(record["op"], batch_ops.OP_DELETE_CATEGORY)
        self.assertEqual(record["params"], {"category": "lens", "remove_empty_files": False})
        self.assertEqual(record["scope"], "filtered")
        self.assertEqual(record["scope_images"], 2)
        self.assertEqual((record["images"], record["objects"]), (1, 2))
        entry, = record["files"]
        self.assertEqual((entry["file"], entry["objects_before"], entry["objects_after"]),
                         ("a.json", 3, 1))
        self.assertTrue(entry["before"]["sha256"] and entry["after"]["sha256"])
        self.assertIsInstance(entry["before"]["mtime_ns"], int)
        backup = self.root / record["backup_dir"] / "a.json"
        self.assertTrue(record["backup_dir"].startswith(batch_ops.BATCH_DIR_NAME + "/"))
        self.assertEqual(backup.read_bytes(), self.originals["a.json"])

    def test_rename_and_merge_into_existing_category(self):
        result = batch_ops.rename_category(self.root, "lens", "gauze")
        self.assertEqual((result["images"], result["objects"]), (2, 3))
        self.assertEqual([o["category"] for o in self.load("b.json")["objects"]], ["gauze", "gauze"])
        a = self.load("a.json")["objects"]
        self.assertEqual([o["category"] for o in a], ["gauze", "tool", "gauze"])
        self.assertEqual(a[0]["custom"], {"keep": [1, 2]})
        index = batch_ops.scan_label_dir(self.root, ["a", "b", "c"])
        self.assertEqual(index.object_counts(), {"gauze": 4, "tool": 2})
        with self.assertRaises(batch_ops.BatchError):
            batch_ops.rename_category(self.root, "tool", "tool")
        with self.assertRaises(batch_ops.BatchError):
            batch_ops.rename_category(self.root, "tool", "  ")

    def test_dry_run_reports_without_writing(self):
        result = batch_ops.delete_category(self.root, "lens", dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual((result["images"], result["objects"]), (2, 3))
        for name, raw in self.originals.items():
            self.assertEqual((self.root / name).read_bytes(), raw)
        self.assertFalse((self.root / batch_ops.BATCH_DIR_NAME).exists())

    def test_noop_writes_no_history(self):
        result = batch_ops.delete_category(self.root, "absent")
        self.assertEqual(result["images"], 0)
        self.assertIsNone(result["id"])
        self.assertEqual(self.history(), [])

    def test_unparsable_or_invalid_labels_are_never_rewritten(self):
        bad_json = self.write("d.json", '{"info": {"description": "IVAS"}, "objects": [')
        two_vertices = label(obj("lens"))
        two_vertices["objects"][0]["segmentation"] = [[1, 1], [2, 2]]
        invalid = self.write("e.json", two_vertices)
        invalid_raw = invalid.read_bytes()
        result = batch_ops.delete_category(self.root, "lens")
        self.assertEqual({item["file"] for item in result["errors"]}, {"d.json", "e.json"})
        self.assertEqual(result["error_count"], 2)
        self.assertEqual(bad_json.read_text(encoding="utf-8"),
                         '{"info": {"description": "IVAS"}, "objects": [')
        self.assertEqual(invalid.read_bytes(), invalid_raw)
        self.assertEqual(result["images"], 2)
        self.assertEqual(self.history()[-1]["error_count"], 2)

    def test_remove_empty_files_is_opt_in_and_keeps_noted_files(self):
        self.write("n.json", label(obj("tool"), note="check later"))
        batch_ops.delete_category(self.root, "tool", stems=["c"])
        self.assertEqual(self.load("c.json")["objects"], [])  # default keeps the file
        batch_ops.undo_last(self.root)
        result = batch_ops.delete_category(self.root, "tool", stems=["c", "n"],
                                           remove_empty_files=True)
        self.assertFalse(self.c.exists())
        self.assertEqual(self.load("n.json")["objects"], [])  # note keeps it
        actions = {item["file"]: item["action"] for item in result["files"]}
        self.assertEqual(actions, {"c.json": "deleted", "n.json": "modified"})
        self.assertIsNone(self.history()[-1]["files"][0]["after"])
        batch_ops.undo_last(self.root)
        self.assertEqual(self.c.read_bytes(), self.originals["c.json"])

    def test_delete_empty_labels(self):
        self.write("x.json", label())
        self.write("y.json", label(note="keep me"))
        result = batch_ops.delete_empty_labels(self.root)
        self.assertEqual([item["file"] for item in result["files"]], ["x.json"])
        self.assertFalse((self.root / "x.json").exists())
        self.assertTrue((self.root / "y.json").exists())

    def test_undo_restores_bytes_and_marks_history(self):
        first = batch_ops.delete_category(self.root, "lens")
        second = batch_ops.rename_category(self.root, "tool", "instrument")
        self.assertEqual(batch_ops.last_undoable(self.root)["id"], second["id"])
        undo = batch_ops.undo_last(self.root)
        self.assertEqual(undo["op"], batch_ops.OP_UNDO)
        self.assertEqual(undo["params"]["target"], second["id"])
        self.assertEqual(batch_ops.last_undoable(self.root)["id"], first["id"])
        batch_ops.undo_last(self.root)
        for name, raw in self.originals.items():
            self.assertEqual((self.root / name).read_bytes(), raw, name)
        records = self.history()
        self.assertEqual([r["op"] for r in records], [
            batch_ops.OP_DELETE_CATEGORY, batch_ops.OP_RENAME_CATEGORY,
            batch_ops.OP_UNDO, batch_ops.OP_UNDO])
        self.assertTrue(records[0]["undone"] and records[1]["undone"])
        self.assertTrue(records[0]["undone_at"])
        with self.assertRaises(batch_ops.UndoRefused):
            batch_ops.undo_last(self.root)

    def test_undo_refuses_after_external_modification(self):
        result = batch_ops.delete_category(self.root, "lens")
        after_b = self.b.read_bytes()
        data = self.load("a.json")
        data["objects"].append(obj("lens", 40))
        self.write("a.json", data)
        edited_a = self.a.read_bytes()
        with self.assertRaises(batch_ops.UndoRefused) as caught:
            batch_ops.undo_last(self.root)
        self.assertIn("a.json", str(caught.exception))
        self.assertIn("modified", str(caught.exception))
        # Nothing was restored and the record stays undoable.
        self.assertEqual(self.a.read_bytes(), edited_a)
        self.assertEqual(self.b.read_bytes(), after_b)
        self.assertFalse(self.history()[-1]["undone"])
        self.assertEqual(batch_ops.last_undoable(self.root)["id"], result["id"])

    def test_undo_refuses_when_deleted_file_was_recreated(self):
        batch_ops.delete_category(self.root, "tool", stems=["c"], remove_empty_files=True)
        self.write("c.json", label(obj("new")))
        with self.assertRaises(batch_ops.UndoRefused) as caught:
            batch_ops.undo_last(self.root)
        self.assertIn("c.json", str(caught.exception))

    def test_touched_but_identical_file_can_still_be_undone(self):
        batch_ops.delete_category(self.root, "gauze")
        os.utime(self.b, ns=(1_000_000_000, 1_000_000_000))
        batch_ops.undo_last(self.root)
        self.assertEqual(self.b.read_bytes(), self.originals["b.json"])

    def test_history_tolerates_truncated_line(self):
        batch_ops.delete_category(self.root, "gauze")
        path = Path(batch_ops.history_path(self.root))
        with open(path, "ab") as stream:
            stream.write(b'{"id": "broken')
        batch_ops.delete_category(self.root, "tool", stems=["c"])
        records = batch_ops.read_history(self.root)
        self.assertEqual([r["op"] for r in records], [batch_ops.OP_DELETE_CATEGORY] * 2)
        batch_ops.undo_last(self.root)
        self.assertIn(b'{"id": "broken', path.read_bytes())  # kept verbatim
        self.assertTrue(batch_ops.read_history(self.root)[1]["undone"])


class FilesDockFilterTests(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        super().setUp()
        from PIL import Image
        from IVAS.widgets import mainwindow
        patcher = mock.patch.object(mainwindow.CheckLatestVersionThread, "start")
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("a", "b", "c", "d"):
            Image.new("RGB", (64, 48), "white").save(self.root / (name + ".png"))
        self.write("a.json", label(obj("lens"), obj("tool")))
        self.write("b.json", label(obj("tool")))
        self.write("c.json", '{"info":')
        self.window = mainwindow.MainWindow()
        self.window.save_cfg = mock.Mock()
        self.window.save_software_cfg = mock.Mock()
        self.window.cfg["software"]["auto_save"] = False
        self.addCleanup(self._close_window)
        self.dock = self.window.files_dock_widget
        self.assertTrue(self.window.load_image_directory(str(self.root)))
        self._wait_index()

    def _close_window(self):
        from PyQt5 import QtWidgets
        from IVAS.widgets.batch_ops_dialog import wait_for_background_tasks
        wait_for_background_tasks()
        window = self.window
        window.saved = True
        window.scene.cancel_draw()
        workspace = window.video_annotation_workspace
        if workspace is not None:
            workspace.shutdown()
        if self.dock.batch_dialog is not None:
            self.dock.batch_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()

    def _wait_index(self):
        from PyQt5 import QtCore, QtTest
        timer = QtCore.QElapsedTimer()
        timer.start()
        QtTest.QTest.qWait(10)
        while self.dock.is_index_busy() and timer.elapsed() < 8000:
            QtTest.QTest.qWait(20)
        self.assertFalse(self.dock.is_index_busy(), "label index scan timed out")
        self.assertIsNotNone(self.dock.index)

    def visible(self):
        return self.dock.visible_rows()

    def test_filter_hides_rows_without_touching_files_list(self):
        dock, window = self.dock, self.window
        files = list(window.files_list)
        self.assertEqual(files, ["a.png", "b.png", "c.png", "d.png"])
        self.assertEqual(self.visible(), [0, 1, 2, 3])
        self.assertIn("4 / 4", dock.label_filter_count.text())

        dock.set_category_filter("lens")
        self.assertEqual(self.visible(), [0])
        self.assertIn("1 / 4", dock.label_filter_count.text())
        dock.set_category_filter("tool")
        self.assertEqual(self.visible(), [0, 1])
        dock.set_category_filter("lens", lacks=True)
        self.assertEqual(self.visible(), [1, 3])  # unreadable c is excluded
        from IVAS.widgets.files_dock_widget import FILTER_ANY, FILTER_UNREADABLE, FILTER_ALL
        dock.set_category_filter(FILTER_UNREADABLE)
        self.assertEqual(self.visible(), [2])
        dock.set_category_filter(FILTER_ANY, lacks=True)
        self.assertEqual(self.visible(), [3])
        dock.set_category_filter(FILTER_ALL)
        dock.lineEdit_search.setText("B")
        dock.apply_filter()
        self.assertEqual(self.visible(), [1])
        dock.lineEdit_search.setText("*.png")
        dock.apply_filter()
        self.assertEqual(self.visible(), [0, 1, 2, 3])
        dock.lineEdit_search.clear()

        # Clicking a visible row opens the same image index.
        dock.set_category_filter("tool")
        model_index = dock.listWidget.model().index(1, 0)
        dock.listWidget.setCurrentRow(1)
        dock.listwidget_doubleclick(model_index)
        self.assertEqual(window.current_index, 1)
        self.assertEqual(window.current_label.label_path, str(self.root / "b.json"))
        self.assertEqual(window.files_list, files)
        self.assertEqual(dock.listWidget.count(), 4)
        self.assertEqual(dock.listWidget.item(3).text(), "[4] d.png")

        # Category combo lists configured + on-disk categories with image counts.
        texts = [dock.comboBox_category.itemText(i) for i in range(dock.comboBox_category.count())]
        self.assertIn("lens (1)", texts)
        self.assertIn("tool (2)", texts)

    def test_index_refreshes_after_saves_and_batch_ops(self):
        dock = self.dock
        dock.set_category_filter("lens")
        self.assertEqual(self.visible(), [0])
        self.write("d.json", label(obj("lens")))
        dock.refresh_index()
        self._wait_index()
        self.assertEqual(self.visible(), [0, 3])
        batch_ops.delete_category(self.window.label_root, "lens", stems=["a"])
        dock.refresh_index()
        self._wait_index()
        self.assertEqual(self.visible(), [3])

    def test_batch_dialog_runs_reloads_canvas_and_undoes(self):
        from PyQt5 import QtWidgets
        window, dock = self.window, self.dock
        self.assertEqual([p.category for p in window.polygons], ["lens", "tool"])
        dock.set_category_filter("lens")
        dialog = dock.open_batch_dialog()
        # The dock's filter preselects the category and the filtered scope.
        self.assertEqual(dialog.selected_category(), "lens")
        self.assertEqual(dialog.scope(), "filtered")
        self.assertEqual(dialog.scope_stems(), ["a"])
        self.assertEqual(dialog._category_keys[:2], ["lens", "tool"])  # on-disk first
        dialog.comboBox_operation.setCurrentIndex(0)
        dialog.radio_scope["all"].setChecked(True)
        with mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Yes):
            dialog.run_operation()
            self._wait_dialog(dialog)
        self.assertEqual([o["category"] for o in self.load("a.json")["objects"]], ["tool"])
        self.assertEqual([p.category for p in window.polygons], ["tool"])  # canvas reloaded
        self.assertEqual(self.load_text("c.json"), '{"info":')
        self._wait_index()
        self.assertEqual(self.visible(), [])  # the dock index followed the batch edit
        self.assertEqual(dialog.treeWidget_history.topLevelItemCount(), 1)

        with mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Yes):
            dialog.undo_last()
            self._wait_dialog(dialog)
        self.assertEqual([p.category for p in window.polygons], ["lens", "tool"])
        self.assertEqual(dialog.treeWidget_history.topLevelItemCount(), 2)
        self._wait_index()
        self.assertEqual(self.visible(), [0])

    def test_batch_requires_confirming_unsaved_edits(self):
        from PyQt5 import QtWidgets
        dialog = self.dock.open_batch_dialog()
        dialog.comboBox_category.setCurrentIndex(dialog._category_keys.index("lens"))
        with mock.patch.object(self.window, "_confirm_leave_unsaved_image", return_value=False), \
                mock.patch.object(QtWidgets.QMessageBox, "question") as question:
            dialog.run_operation()
        question.assert_not_called()
        self.assertFalse(dialog._busy)
        self.assertEqual(batch_ops.read_history(self.root), [])

    def test_language_switch_retranslates_filter_and_dialog(self):
        dialog = self.dock.open_batch_dialog()
        self.window.translate("zh")
        self.assertEqual(self.dock.toolButton_batch.text(), "批量…")
        self.assertEqual(self.dock.comboBox_filter_mode.itemText(1), "不含")
        self.assertEqual(dialog.tabWidget.tabText(1), "记录")
        self.window.translate("en")
        self.assertEqual(self.dock.toolButton_batch.text(), "Batch…")
        self.assertEqual(dialog.tabWidget.tabText(1), "History")
        self.assertIn("images", self.dock.label_filter_count.text())

    def load_text(self, name):
        return (self.root / name).read_text(encoding="utf-8")

    def _wait_dialog(self, dialog):
        from PyQt5 import QtCore, QtTest
        timer = QtCore.QElapsedTimer()
        timer.start()
        QtTest.QTest.qWait(10)
        while (dialog._busy or dialog._pending is not None) and timer.elapsed() < 8000:
            QtTest.QTest.qWait(20)
        self.assertFalse(dialog._busy, "batch dialog task timed out")


if __name__ == "__main__":
    unittest.main()
