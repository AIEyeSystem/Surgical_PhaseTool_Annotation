import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from IVAS.annotation import Annotation, Object


class AnnotationSafetyTests(unittest.TestCase):
    def test_save_is_replace_safe_and_reload_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            label_path = root / "image.json"
            Image.new("RGB", (8, 6), "white").save(image_path)

            annotation = Annotation(str(image_path), str(label_path))
            annotation.objects.append(
                Object("lens", 1, [[0, 0], [4, 0], [4, 4]], 8, 0, [0, 0, 4, 4])
            )
            self.assertTrue(annotation.save_annotation())
            self.assertTrue(label_path.is_file())
            self.assertEqual(list(root.glob(".ivas-*.json.tmp")), [])

            annotation.load_annotation()
            annotation.load_annotation()
            self.assertEqual(len(annotation.objects), 1)
            self.assertEqual(annotation.objects[0].category, "lens")

    def test_string_false_iscrowd_is_not_treated_as_true(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            label_path = root / "image.json"
            Image.new("L", (4, 4), 0).save(image_path)
            label_path.write_text(
                json.dumps(
                    {
                        "info": {"description": "IVAS"},
                        "objects": [
                            {
                                "category": "lens",
                                "group": 1,
                                "segmentation": [],
                                "iscrowd": "false",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            annotation = Annotation(str(image_path), str(label_path)).load_annotation()

            self.assertFalse(annotation.objects[0].iscrowd)


if __name__ == "__main__":
    unittest.main()
