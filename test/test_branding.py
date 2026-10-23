import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt5 import QtGui, QtWidgets

import IVAS
from IVAS import icons_rc  # noqa: F401 - registers the compiled Qt resources
from IVAS.annotation import Annotation, Object
from IVAS.configs import load_config
from IVAS.formats.ivas import IVAS as IVASFormat
from IVAS.branding import (
    ANNOTATION_FORMAT,
    APP_NAME,
    AUTHOR,
    LOGO_RESOURCE,
    VIDEO_PROJECT_NAME,
)
from IVAS.video.project import (
    UnifiedVideoProjectStore,
    case_directory,
    discover_video_projects,
)


class BrandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_identity_and_compiled_logo_resource(self):
        self.assertEqual(APP_NAME, "IVAS")
        self.assertEqual(AUTHOR, "zack zhao")
        self.assertEqual(IVAS.__author__, AUTHOR)

        pixmap = QtGui.QPixmap(LOGO_RESOURCE)

        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 0)
        self.assertGreater(pixmap.height(), 0)

    def test_logo_assets_are_in_distribution_manifest(self):
        repository = Path(__file__).resolve().parents[1]
        manifest = (repository / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertTrue((repository / "logo" / "logo.png").is_file())
        self.assertTrue((repository / "logo" / "IVAS.ico").is_file())
        self.assertIn("include logo/logo.png", manifest)
        self.assertIn("include logo/IVAS.ico", manifest)

    def test_segmentation_actions_use_the_compiled_logo(self):
        repository = Path(__file__).resolve().parents[1]
        ui_source = (repository / "IVAS" / "ui" / "MainWindow.ui").read_text(
            encoding="utf-8"
        )
        generated_source = (
            repository / "IVAS" / "ui" / "MainWindow.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(":/icon/icons/M_Favicon.ico", ui_source)
        self.assertNotIn(":/icon/icons/M_Favicon.ico", generated_source)
        self.assertGreaterEqual(ui_source.count(LOGO_RESOURCE), 2)
        self.assertIn(LOGO_RESOURCE, generated_source)

    def test_user_documentation_uses_current_identity_and_support_channel(self):
        repository = Path(__file__).resolve().parents[1]
        documents = [repository / "README.md", repository / "README-cn.md"]
        documents.extend((repository / "docs" / "source").glob("*.rst"))
        combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)

        self.assertIn("Welcome to IVAS's documentation!", combined)
        self.assertIn("zack zhao", combined)
        for stale_text in (
            "IVAS-SAM",
            "Shuwei Ji",
            "Hongyuan Zhang",
            "yatengLG/IVAS_with_segment_anything/issues",
            "yatengLG/IVAS_with_segment_anything/discussions",
        ):
            self.assertNotIn(stale_text, combined)

    def test_new_annotation_uses_ivas_and_legacy_description_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            label_path = root / "image.json"
            Image.new("RGB", (8, 6), "white").save(image_path)
            annotation = Annotation(str(image_path), str(label_path))
            annotation.objects.append(
                Object("lens", 1, [[0, 0], [4, 0], [4, 4]], 8, 0, [0, 0, 4, 4])
            )

            annotation.save_annotation()
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["info"]["description"], ANNOTATION_FORMAT)

            for legacy_format in ("ISAT", "IVAT"):
                payload["info"]["description"] = legacy_format
                label_path.write_text(json.dumps(payload), encoding="utf-8")
                loaded = Annotation(str(image_path), str(label_path)).load_annotation()
                self.assertEqual(len(loaded.objects), 1)
                self.assertEqual(loaded.objects[0].category, "lens")

    def test_original_isat_roundtrip_in_gui_and_converter_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            label_path = root / "image.json"
            Image.new("RGB", (8, 6), "white").save(image_path)
            obj = {"category": "晶状体套圈", "group": 2, "segmentation": [[1.25, 1], [5, 1], [5, 5]],
                   "area": 7.5, "layer": 3, "bbox": [1.25, 1, 5, 5], "iscrowd": False, "note": "原标注"}
            label_path.write_text(json.dumps({"info": {"description": "ISAT", "name": "image.png",
                                                       "width": 8, "height": 6, "depth": 3, "note": "原图"},
                                             "objects": [obj]}, ensure_ascii=False), encoding="utf-8")
            original = label_path.read_bytes()
            annotation = Annotation(str(image_path), str(label_path)).load_annotation()
            converter = IVASFormat()
            converted = converter.load_one_ivas_json(str(label_path))
            self.assertEqual(label_path.read_bytes(), original)
            for loaded in (annotation.objects[0], converted.objs[0]):
                for key, value in obj.items():
                    self.assertEqual(getattr(loaded, key), value)
            annotation.label_path = str(root / "gui-output.json")
            annotation.save_annotation()
            converter.save_one_ivas_json(converted, str(root / "converter-output.json"))
            for filename in ("gui-output.json", "converter-output.json"):
                saved = json.loads((root / filename).read_text(encoding="utf-8"))
                self.assertEqual(saved["info"]["description"], "IVAS")
                self.assertEqual(saved["info"]["note"], "原图")
                for key, value in obj.items():
                    self.assertEqual(saved["objects"][0][key], value)

    def test_original_isat_category_config_is_discovered_with_ivas_priority(self):
        from IVAS.widgets.mainwindow import _find_category_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "isat.yaml"
            current = root / "ivas.yaml"
            legacy.write_text("label:\n- name: 晶状体套圈\n", encoding="utf-8")
            self.assertEqual(_find_category_config(directory), str(legacy))
            self.assertEqual(load_config(str(current))["label"][0]["name"], "晶状体套圈")
            current.write_text("label:\n- name: Vectis\n", encoding="utf-8")
            self.assertEqual(_find_category_config(directory), str(current))
            self.assertEqual(load_config(str(current))["label"][0]["name"], "Vectis")

    def test_foreign_annotation_format_still_cannot_be_loaded_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, label_path = root / "image.png", root / "image.json"
            Image.new("RGB", (8, 6)).save(image_path)
            label_path.write_text('{"info": {"description": "unrelated"}, "objects": []}', encoding="utf-8")
            original = label_path.read_bytes()
            annotation = Annotation(str(image_path), str(label_path))
            with self.assertRaises(ValueError):
                annotation.load_annotation()
            with self.assertRaises(ValueError):
                annotation.save_annotation()
            with self.assertRaises(AttributeError):
                IVASFormat().load_one_ivas_json(str(label_path))
            self.assertEqual(label_path.read_bytes(), original)

    def test_new_video_project_name_and_legacy_project_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "case.mp4"
            video.write_bytes(b"synthetic-video")

            new_output = root / "new-output"
            created = UnifiedVideoProjectStore.open_or_create(new_output, video)
            created.save()
            self.assertEqual(created.path.name, VIDEO_PROJECT_NAME)
            self.assertEqual(created.project["app_version"], IVAS.__version__)

            legacy_output = root / "legacy-output"
            legacy_case = case_directory(legacy_output, video)
            legacy_case.mkdir(parents=True)
            legacy_path = legacy_case / "project.ivat.json"
            UnifiedVideoProjectStore.create(legacy_path, video).save()

            opened = UnifiedVideoProjectStore.open_or_create(legacy_output, video)
            self.assertEqual(opened.path, legacy_path)
            self.assertIn(legacy_path.resolve(), discover_video_projects(legacy_output))




if __name__ == "__main__":
    unittest.main()
