"""Themes, toolbar chrome, bbox drawing, frame undo/redo and dataset export."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
from PIL import Image
from PyQt5 import QtCore, QtTest, QtWidgets

from IVAS import theme
from IVAS.widgets.polygon import BBox, OBB, Polygon

ROOT = Path(__file__).resolve().parents[1]


def _luminance(colour):
    colour = colour.lstrip("#")
    channels = [int(colour[index:index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first, second):
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class ThemePaletteTests(unittest.TestCase):
    def test_every_theme_declares_the_same_roles(self):
        keys = [key for key, *_names in theme.available_themes()]
        self.assertEqual(
            keys, ["medical", "geek", "graphite", "surgical", "aurora"]
        )
        reference = set(theme.THEMES["medical"]["roles"])
        for key in keys:
            with self.subTest(theme=key):
                self.assertEqual(set(theme.THEMES[key]["roles"]), reference)

    def test_text_and_controls_meet_contrast_targets(self):
        pairs = (
            ("text", "surface", 4.5), ("text", "surface-raised", 4.5), ("text", "chrome", 4.5),
            ("text", "input", 4.5), ("text", "hover", 4.5),
            ("text-muted", "surface", 4.5), ("text-muted", "surface-raised", 4.5),
            ("text-muted", "chrome", 4.5),
            ("on-primary", "primary", 4.5), ("on-primary", "primary-hover", 4.5),
            ("on-primary", "primary-active", 4.5), ("on-danger", "danger", 4.5),
            ("on-selection", "selection", 4.5), ("canvas-text", "canvas", 4.5),
            ("canvas-text", "canvas-raised", 4.5),
            ("icon", "chrome", 3), ("icon", "hover", 3), ("icon", "surface-raised", 3),
            ("border", "surface", 3), ("focus", "surface", 3), ("focus", "input", 3),
            ("primary", "chrome", 3), ("timeline-phase", "canvas", 3),
            ("timeline-task", "canvas", 3),
        )
        for key, spec in theme.THEMES.items():
            roles = spec["roles"]
            for foreground, background, minimum in pairs:
                with self.subTest(theme=key, pair=(foreground, background)):
                    self.assertGreaterEqual(
                        _contrast(roles[foreground], roles[background]), minimum
                    )

    def test_stylesheets_build_and_theme_tokens_follow_the_active_theme(self):
        previous = theme.active_theme()
        self.addCleanup(theme.set_active_theme, previous)
        for key, *_names in theme.available_themes():
            sheet = theme.build_stylesheet(key)
            self.assertIn(theme.THEMES[key]["roles"]["primary"], sheet)
            self.assertNotIn("{", sheet.split("QWidget", 1)[0])
            theme.set_active_theme(key)
            self.assertEqual(theme.THEME_TOKENS["primary"], theme.THEMES[key]["roles"]["primary"])
            self.assertEqual(theme.theme_hex("canvas"), theme.THEMES[key]["roles"]["canvas"])
        self.assertEqual(theme.normalize_theme("does-not-exist"), theme.DEFAULT_THEME)


class WorkbenchTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from IVAS.widgets import mainwindow

        patcher = mock.patch.object(mainwindow.CheckLatestVersionThread, "start")
        patcher.start()
        self.addCleanup(patcher.stop)
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self._previous_theme = theme.active_theme()
        self._previous_style = (self.app.styleSheet(), self.app.palette(), self.app.font())
        self.window = mainwindow.MainWindow()
        self.window.save_cfg = mock.Mock()
        self.window.save_software_cfg = mock.Mock()
        self.window.cfg["software"]["auto_save"] = False
        self.addCleanup(self._close_window)

    def _close_window(self):
        window = self.window
        window.saved = True
        window.scene.cancel_draw()
        if window.export_dialog is not None:
            window.export_dialog.close()
        workspace = window.video_annotation_workspace
        if workspace is not None:
            workspace._clear_pending_interval()
            workspace.shutdown()
        window.close()
        window.deleteLater()
        self.app.processEvents()
        # Later test modules size widgets against the unthemed application.
        theme.set_active_theme(self._previous_theme)
        style_sheet, palette, font = self._previous_style
        self.app.setStyleSheet(style_sheet)
        self.app.setPalette(palette)
        self.app.setFont(font)

    def _open_images(self, count=2, size=(200, 160)):
        directory = self.root / "images"
        directory.mkdir()
        for index in range(count):
            Image.new("RGB", size, (40 + index * 30, 60, 80)).save(directory / "img{}.png".format(index))
        self.assertTrue(self.window.load_image_directory(str(directory)))
        self.window.resize(1280, 800)
        self.window.show()
        QtTest.QTest.qWait(50)
        return directory

    def _viewport_point(self, x, y):
        view = self.window.view
        return view.mapFromScene(QtCore.QPointF(x, y))

    def _click(self, x, y):
        viewport = self.window.view.viewport()
        point = self._viewport_point(x, y)
        QtTest.QTest.mouseMove(viewport, point)
        QtTest.QTest.mousePress(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
        QtTest.QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)


class ChromeTests(WorkbenchTestCase):
    def test_runtime_theme_switch_updates_actions_config_and_icons(self):
        window = self.window
        before = window.actionPolygon.icon().pixmap(20, 20).toImage()
        window.change_theme("geek")
        self.assertEqual(theme.active_theme(), "geek")
        self.assertEqual(window.cfg["software"]["theme"], "geek")
        self.assertTrue(window._theme_actions["geek"].isChecked())
        self.assertFalse(window._theme_actions["medical"].isChecked())
        self.assertIn(theme.THEMES["geek"]["roles"]["primary"], self.app.styleSheet())
        after = window.actionPolygon.icon().pixmap(20, 20).toImage()
        self.assertNotEqual(before, after)
        window.save_software_cfg.assert_called()
        window.change_theme("unknown")
        self.assertEqual(theme.active_theme(), theme.DEFAULT_THEME)

    def test_mode_switch_mirrors_actions_and_file_menu_owns_export(self):
        window = self.window
        segment = window.mode_segment
        self.assertTrue(segment.button("Frame").isChecked())
        segment.button("Video").click()
        self.assertEqual(window.annotation_mode, "Video")
        self.assertTrue(segment.button("Video").isChecked())
        self.assertEqual(segment.text(), "Mode: Video")
        segment.button("Frame").click()
        self.assertEqual(window.annotation_mode, "Frame")
        self.assertIn(window.actionExport, window.menuFile.actions())
        self.assertIn(window.actionImport, window.menuFile.actions())
        self.assertIn(window.menuTheme.menuAction(), window.menuView.actions())
        window.translate("zh")
        self.assertEqual(segment.button("Video").text(), "视频")
        self.assertEqual(window.actionBBox.iconText(), "矩形框")
        self.assertGreaterEqual(segment.button("Frame").width(), segment.button("Frame").sizeHint().width())

    def test_each_command_has_one_menu_home_and_no_settings_toolbar(self):
        window = self.window
        homes = {}

        def walk(menu, path):
            for action in menu.actions():
                if action.isSeparator() or isinstance(action, QtWidgets.QWidgetAction):
                    continue
                if action.menu() is not None:
                    walk(action.menu(), path + (action.menu().objectName(),))
                else:
                    homes.setdefault(action, []).append(path)

        for top in window.menuBar().actions():
            walk(top.menu(), (top.menu().objectName(),))
        duplicates = {action.objectName(): paths for action, paths in homes.items() if len(paths) > 1}
        self.assertEqual(duplicates, {})
        self.assertIn(window.actionShortcut, window.menuFile.actions())
        self.assertIn(window.actionLanguage, window.menuView.actions())

        self.assertFalse(hasattr(window, "application_more_button"))
        toolbar_actions = set(window.toolBar.actions()) | set(window.toolBar_2.actions())
        for action in (window.actionModel_manage, window.actionSetting, window.actionLanguage,
                       window.actionPlugins, window.actionAbout):
            self.assertNotIn(action, toolbar_actions)
        self.assertIn(window.actionExport, window.toolBar.actions())
        self.assertIs(window.export_tool_button, window.toolBar.widgetForAction(window.actionExport))

    def test_video_row_appears_only_with_a_video_case(self):
        window = self.window
        window.show()
        self.assertTrue(window.toolBar_2.isHidden())
        window.actionShow_video_workspace.trigger()
        self.assertFalse(window.toolBar_2.isHidden())
        window.actionShow_image_workspace.trigger()
        self.assertTrue(window.toolBar_2.isHidden())

    def test_model_indicator_reports_state_and_opens_the_manager(self):
        window = self.window
        button = window.model_status_button
        self.assertIn(button, window.statusbar.findChildren(QtWidgets.QToolButton))
        self.assertEqual(button.property("modelState"), "none")
        window._loading_model_name = "sam3.1_hiera"
        window._refresh_model_status("loading")
        self.assertIn("sam3.1_hiera", button.text())
        window.segany = mock.Mock(checkpoint="C:/models/sam3.1_hiera.pt")
        window._refresh_model_status("ready")
        self.assertEqual(button.text(), "SAM · sam3.1_hiera")
        window.translate("zh")
        self.assertEqual(button.text(), "SAM · sam3.1_hiera")
        window._refresh_model_status("failed")
        self.assertEqual(button.text(), "SAM 加载失败")
        with mock.patch.object(window, "model_manage") as manage:
            button.click()
        manage.assert_called_once()

    def test_recent_folders_remember_pairs_and_reopen(self):
        window = self.window
        directory = self._open_images(count=1)
        recent = window.cfg["software"]["recent_folders"]
        self.assertEqual(recent[0]["images"], str(directory))
        window._rebuild_recent_menu()
        entries = [action for action in window.menuRecent.actions() if not action.isSeparator()]
        self.assertEqual(entries[0].text(), str(directory))
        with mock.patch.object(window, "load_image_directory", return_value=True) as load:
            entries[0].trigger()
        load.assert_called_once_with(str(directory), str(directory))
        window._clear_recent_folders()
        self.assertEqual(window.cfg["software"]["recent_folders"], [])

    def test_right_inspector_widens_so_every_tab_is_visible(self):
        window = self.window
        window.show()
        for width, height in ((1366, 768), (1920, 1080)):
            window.resize(width, height)
            QtTest.QTest.qWait(80)
            bars = [
                bar for bar in window.findChildren(
                    QtWidgets.QTabBar, options=QtCore.Qt.FindDirectChildrenOnly)
                if bar.isVisible() and bar.count() > 1
            ]
            self.assertTrue(bars)
            for bar in bars:
                with self.subTest(size=(width, height)):
                    last = bar.tabRect(bar.count() - 1)
                    self.assertLessEqual(last.right(), bar.width())
                    self.assertGreaterEqual(bar.tabRect(0).left(), 0)
            self.assertGreater(window.view.width(), width // 2)

    def test_frame_box_key_yields_to_video_mark_end_key(self):
        window = self.window
        self.assertEqual(window.actionOBB.shortcut().toString(), "O")
        with mock.patch.object(window, "_video_shortcut_context_active", return_value=True):
            window._update_unified_video_shortcuts()
            self.assertTrue(window.actionOBB.shortcut().isEmpty())
            self.assertTrue(window.actionUndo.shortcut().isEmpty())
        window._update_unified_video_shortcuts()
        self.assertEqual(window.actionOBB.shortcut().toString(), "O")
        self.assertEqual(window.actionUndo.shortcut().toString(), "Ctrl+Z")


class BBoxAndHistoryTests(WorkbenchTestCase):
    def test_two_click_and_drag_boxes_save_reload_and_stay_axis_aligned(self):
        window = self.window
        directory = self._open_images()
        window.current_category = "__background__"
        self.assertTrue(window.actionBBox.isEnabled())

        window.actionBBox.trigger()
        self._click(20, 30)
        self._click(80, 90)
        self.assertEqual(len(window.polygons), 1)
        box = window.polygons[0]
        self.assertIsInstance(box, BBox)
        stored = box.to_object()
        self.assertEqual(stored.shape_type, "rectangle")
        xmin, ymin, xmax, ymax = stored.bbox
        self.assertAlmostEqual(xmin, 20, delta=1.5)
        self.assertAlmostEqual(ymax, 90, delta=1.5)

        # One drag also completes a box.
        window.actionBBox.trigger()
        viewport = window.view.viewport()
        start, end = self._viewport_point(120, 20), self._viewport_point(180, 70)
        QtTest.QTest.mouseMove(viewport, start)
        QtTest.QTest.mousePress(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
        QtTest.QTest.mouseMove(viewport, end)
        QtTest.QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, end)
        self.assertEqual(len(window.polygons), 2)

        # Finish saves immediately; the JSON keeps the shape type.
        label = json.loads((directory / "img0.json").read_text(encoding="utf-8"))
        self.assertEqual([obj["shape_type"] for obj in label["objects"]], ["rectangle", "rectangle"])

        # Dragging a corner keeps the box upright; rotation keys ignore it.
        box = window.polygons[0]
        box.movePoint(2, QtCore.QPointF(100, 120))
        xs = sorted({round(point.x(), 3) for point in box.points})
        ys = sorted({round(point.y(), 3) for point in box.points})
        self.assertEqual(len(xs), 2)
        self.assertEqual(len(ys), 2)
        box.setSelected(True)
        window.scene.mode = window.scene.mode.__class__.EDIT
        window.scene.rotate_selected_obb(0.5)
        self.assertEqual(len({round(point.x(), 3) for point in box.points}), 2)
        window.scene.cancel_draw()

        window.save()
        window.show_image(1)
        window.show_image(0)
        self.assertEqual([type(shape) for shape in window.polygons], [BBox, BBox])

    def test_undo_and_redo_restore_frame_shapes_without_touching_other_images(self):
        window = self.window
        self._open_images()
        window.actionBBox.trigger()
        self._click(10, 10)
        self._click(60, 60)
        window.actionPolygon.trigger()
        for point in ((100, 20), (150, 20), (150, 80)):
            self._click(*point)
        window.actionFinish.trigger()
        self.assertEqual([type(shape) for shape in window.polygons], [BBox, Polygon])
        window._commit_frame_history()
        self.assertTrue(window.actionUndo.isEnabled())

        window.actionUndo.trigger()
        self.assertEqual([type(shape) for shape in window.polygons], [BBox])
        self.assertTrue(window.actionRedo.isEnabled())
        window.actionUndo.trigger()
        self.assertEqual(window.polygons, [])
        self.assertFalse(window.actionUndo.isEnabled())
        window.actionRedo.trigger()
        window.actionRedo.trigger()
        self.assertEqual([type(shape) for shape in window.polygons], [BBox, Polygon])
        self.assertEqual(window.annos_dock_widget.listWidget.count(), 2)

        # A new edit after undo discards the redo branch.
        window.actionUndo.trigger()
        window.actionBBox.trigger()
        self._click(120, 100)
        self._click(170, 140)
        window._commit_frame_history()
        self.assertFalse(window.actionRedo.isEnabled())

        window.saved = True
        window.show_image(1)
        self.assertFalse(window.actionUndo.isEnabled())
        self.assertEqual(len(window._frame_history), 1)

    def test_obb_action_replaces_the_hidden_key_handler(self):
        window = self.window
        self._open_images()
        window.actionOBB.trigger()
        for point in ((20, 20), (80, 20), (80, 60)):
            self._click(*point)
        window.actionFinish.trigger()
        self.assertIsInstance(window.polygons[0], OBB)
        self.assertNotIsInstance(window.polygons[0], BBox)


class ExportImportTests(WorkbenchTestCase):
    def _labelled_folder(self):
        directory = self._open_images(count=1)
        window = self.window
        window.current_category = "__background__"
        window.actionBBox.trigger()
        self._click(10, 20)
        self._click(70, 90)
        window.actionPolygon.trigger()
        for point in ((100, 20), (150, 20), (150, 80)):
            self._click(*point)
        window.actionFinish.trigger()
        window.save()
        return directory

    def _run(self, worker):
        worker.run()
        self.assertEqual(worker.errors, [])

    def test_every_export_format_writes_boxes(self):
        window = self.window
        self._labelled_folder()
        window.actionExport.trigger()
        dialog = window.export_dialog
        self.assertTrue(dialog.isVisible())
        out = self.root / "exports"
        coco = out / "coco.json"
        self._run(dialog.build_export_worker("coco", str(coco)))
        data = json.loads(coco.read_text(encoding="utf-8"))
        shapes = sorted(annotation.get("ivas_shape_type", "polygon") for annotation in data["annotations"])
        self.assertEqual(shapes, ["polygon", "rectangle"])

        for key in ("yolo-det", "yolo-seg", "labelme", "voc-det", "voc-seg", "ivas"):
            target = out / key
            target.mkdir(parents=True)
            self._run(dialog.build_export_worker(key, str(target)))
            self.assertTrue(any(target.iterdir()), key)
        detection = (out / "yolo-det" / "img0.txt").read_text(encoding="utf-8").split("\n")
        self.assertTrue(all(len(line.split()) == 5 for line in detection if line.strip()))
        labelme = json.loads((out / "labelme" / "img0.json").read_text(encoding="utf-8"))
        self.assertIn("rectangle", [shape["shape_type"] for shape in labelme["shapes"]])

    def test_import_merges_categories_and_keeps_existing_colours(self):
        window = self.window
        self._labelled_folder()
        window.actionExport.trigger()
        dialog = window.export_dialog
        coco = self.root / "coco.json"
        self._run(dialog.build_export_worker("coco", str(coco)))

        target = self.root / "imported"
        target.mkdir()
        (target / "ivas.yaml").write_text(
            yaml.safe_dump({"label": [{"name": "kept", "color": "#123456"}]}), encoding="utf-8"
        )
        with open(target / "ivas.yaml", encoding="utf-8") as stream:
            dialog._category_backup = (str(target / "ivas.yaml"), yaml.safe_load(stream))
        worker = dialog.build_import_worker("coco", str(coco), str(target))
        self._run(worker)
        dialog._merge_category_backup(str(target))
        labels = yaml.safe_load((target / "ivas.yaml").read_text(encoding="utf-8"))["label"]
        self.assertEqual(labels[0], {"name": "kept", "color": "#123456"})
        self.assertIn("__background__", [label["name"] for label in labels])
        imported = json.loads((target / "img0.json").read_text(encoding="utf-8"))
        self.assertIn("rectangle", [obj["shape_type"] for obj in imported["objects"]])


if __name__ == "__main__":
    unittest.main()
