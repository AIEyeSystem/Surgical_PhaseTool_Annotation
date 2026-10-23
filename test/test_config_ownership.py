"""Category files must survive preference changes, annotation saves and exit."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt5 import QtWidgets
import yaml

from IVAS.widgets import mainwindow


class ConfigOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.categories = self.root / "ivas.yaml"
        self.preferences = self.root / "software.yaml"
        self.categories.write_text(
            "# Keep user formatting\nlabel:\n- name: __background__\n  color: '#000000'\n"
            "- name: Original\n  color: '#ff0000'\n", encoding="utf-8")
        self.preferences.write_text(yaml.safe_dump({
            "label": [{"name": "StaleCategory", "color": "#00ff00"}],
            "software": {"language": "en", "mask_alpha": 0.4, "auto_save": False},
            "shortcut": {"actionSave": {"key": "Ctrl+S", "default_key": "Ctrl+S"}},
            "plugin_preferences": {"enabled": True},
        }), encoding="utf-8")
        for name, value in (("CONFIG_FILE", str(self.categories)),
                            ("SOFTWARE_CONFIG_FILE", str(self.preferences))):
            patch = mock.patch.object(mainwindow, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(mainwindow.CheckLatestVersionThread, "start")
        patch.start()
        self.addCleanup(patch.stop)

    def window(self):
        window = mainwindow.MainWindow()
        window._version_check_timer.stop()

        def close():
            window.saved = True
            window.scene.cancel_draw()
            window.close()
            window.deleteLater()
            self.app.processEvents()

        self.addCleanup(close)
        return window

    def test_preference_changes_drop_legacy_categories_without_touching_category_file(self):
        category_bytes = self.categories.read_bytes()
        window = self.window()
        self.assertEqual(list(window.category_color_dict), ["__background__", "Original"])
        window.change_language()
        saved = yaml.safe_load(self.preferences.read_text(encoding="utf-8"))
        self.assertNotIn("label", saved)
        self.assertEqual(saved["software"]["language"], "zh")
        self.assertEqual(saved["software"]["mask_alpha"], 0.4)
        self.assertEqual(saved["shortcut"]["actionSave"]["key"], "Ctrl+S")
        self.assertEqual(saved["plugin_preferences"], {"enabled": True})
        self.assertEqual(self.categories.read_bytes(), category_bytes)

    def test_missing_empty_and_invalid_categories_never_fall_back_to_software(self):
        for content in (None, "", "metadata: example\n", "label: []\n", "label: [broken"):
            with self.subTest(content=content):
                if content is None:
                    self.categories.unlink(missing_ok=True)
                else:
                    self.categories.write_text(content, encoding="utf-8")
                # Reintroduce an old software.yaml on each run.
                self.preferences.write_text("label:\n- name: StaleCategory\n", encoding="utf-8")
                with mock.patch.object(QtWidgets.QMessageBox, "warning"):
                    window = self.window()
                self.assertEqual(list(window.category_color_dict), ["__background__"])
                self.assertTrue(window.close())
                if content is None:
                    self.assertFalse(self.categories.exists())
                else:
                    self.assertEqual(self.categories.read_text(encoding="utf-8"), content)
                self.assertNotIn("label", yaml.safe_load(self.preferences.read_text(encoding="utf-8")))

    def test_external_category_edit_survives_annotation_save_and_exit(self):
        Image.new("RGB", (8, 8)).save(self.root / "image.png")
        window = self.window()
        self.assertTrue(window.load_image_directory(str(self.root)))
        edited = "# Edited while the app is open\nlabel:\n- name: NewCategory\n  color: '#123456'\n"
        self.categories.write_text(edited, encoding="utf-8")
        window.save()
        self.assertTrue((self.root / "image.json").is_file())
        self.assertEqual(self.categories.read_text(encoding="utf-8"), edited)
        self.assertTrue(window.close())
        self.assertEqual(self.categories.read_text(encoding="utf-8"), edited)
        reopened = self.window()
        self.assertEqual(list(reopened.category_color_dict), ["__background__", "NewCategory"])

    def test_category_config_cannot_override_application_preferences(self):
        categories = yaml.safe_load(self.categories.read_text(encoding="utf-8"))
        categories.update(software={"language": "zh", "mask_alpha": 0.9},
                          shortcut={"actionSave": {"key": "Ctrl+9"}},
                          plugin_preferences={"enabled": False})
        self.categories.write_text(yaml.safe_dump(categories), encoding="utf-8")
        window = self.window()
        self.assertEqual(window.cfg["software"]["language"], "en")
        self.assertEqual(window.cfg["software"]["mask_alpha"], 0.4)
        self.assertEqual(window.cfg["shortcut"]["actionSave"]["key"], "Ctrl+S")
        self.assertEqual(window.cfg["plugin_preferences"], {"enabled": True})

    def test_saving_original_isat_dataset_does_not_create_a_shadow_category_file(self):
        Image.new("RGB", (8, 8)).save(self.root / "image.png")
        legacy = self.categories.with_name("isat.yaml")
        self.categories.rename(legacy)
        original = legacy.read_bytes()
        window = self.window()
        self.assertTrue(window.load_image_directory(str(self.root)))
        window.save()
        self.assertEqual(legacy.read_bytes(), original)
        self.assertFalse(self.categories.exists())
        self.assertTrue(window.close())
        self.assertFalse(self.categories.exists())

    def test_new_label_directory_gets_categories_once(self):
        images, labels = self.root / "images", self.root / "labels"
        images.mkdir()
        labels.mkdir()
        Image.new("RGB", (8, 8)).save(images / "image.png")
        window = self.window()
        self.assertTrue(window.load_image_directory(str(images), str(labels)))
        window.save()
        created = labels / "ivas.yaml"
        self.assertEqual(yaml.safe_load(created.read_text(encoding="utf-8"))["label"], window.cfg["label"])
        created.write_text("# Dataset owner edit\nlabel:\n- name: DatasetCategory\n", encoding="utf-8")
        edited = created.read_bytes()
        window.save()
        self.assertEqual(created.read_bytes(), edited)

    def test_explicit_category_edits_and_exports_still_save(self):
        window = self.window()
        dialog = window._ensure_category_setting_dialog()
        dialog.load_cfg()
        dialog.category_input.setText("AddedInDialog")
        dialog.add_new_category()
        dialog.apply()
        saved = yaml.safe_load(self.categories.read_text(encoding="utf-8"))
        self.assertEqual(saved["label"][-1]["name"], "AddedInDialog")
        exported = self.root / "exported.yaml"
        window.save_cfg(str(exported))
        self.assertEqual(yaml.safe_load(exported.read_text(encoding="utf-8")), {"label": saved["label"]})
        window.save_software_cfg()
        self.assertNotIn("label", yaml.safe_load(self.preferences.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
