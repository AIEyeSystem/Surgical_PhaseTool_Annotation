import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from IVAS.annotation import Annotation, Object
from IVAS.formats.coco import COCO
from IVAS.formats.ivas import IVAS
from IVAS.formats.labelme import LABELME
from IVAS.formats.voc import VOC
from IVAS.formats.voc_detection import VOCDetect
from IVAS.formats.yolo import YOLO
from IVAS.utils.batch import process_exif, validate_annotations


def example_anno():
    anno = IVAS.ANNO()
    anno.info = IVAS.ANNO.INFO()
    anno.info.name = "image.png"
    anno.info.description = "IVAS"
    anno.info.width = anno.info.height = 16
    anno.info.depth = 3
    anno.objs = (Object("tool", 1, [[2, 2], [8, 2], [8, 8], [2, 8]], 36, 1, [2, 2, 8, 8],
                        note="keep this note", object_id="object", track_id="track", source="tracking",
                        review_status="draft", is_obb=True, origin_source="sam"),)
    return anno


class FormatReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_converter_instances_and_repeated_batches_are_isolated(self):
        from IVAS.widgets.converter_dialog import YOLOConverter
        left, right = COCO(), YOLO()
        left.annos["sentinel"] = example_anno()
        right.annos.clear()
        self.assertIn("sentinel", left.annos)
        for name in ("first", "second"):
            folder = self.root / name
            folder.mkdir()
            output = folder / "converted"
            output.mkdir()
            Image.new("RGB", (16, 16)).save(folder / (name + ".png"))
            (folder / (name + ".txt")).write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
            converter = YOLOConverter()
            converter.yolo_images_root = converter.yolo_txt_root = str(folder)
            converter.ivas_json_root = str(output)
            converter.yolo_to_ivas()
            self.assertEqual([path.name for path in output.glob("*.json")], [name + ".json"])

    def test_voc_reserves_background_and_rejects_overflow(self):
        converter = VOC()
        converter.annos = {"image": example_anno()}
        for categories in (("tool",), ("tool", "__background__")):
            converter.cates = categories
            converter.save_to_voc(str(self.root))
            with Image.open(self.root / "image.png") as image:
                pixels = np.asarray(image)
                self.assertEqual(int(pixels[0, 0]), 0)
                self.assertEqual(int(pixels[4, 4]), 1)
        converter.cates = tuple("category-%d" % i for i in range(256))
        with self.assertRaises(ValueError):
            converter.semantic_categories()

    def test_labelme_null_groups_are_independent_and_notes_roundtrip(self):
        path = self.root / "labelme.json"
        data = {"imagePath": "image.png", "imageWidth": 16, "imageHeight": 16,
                "shapes": [{"label": "tool", "group_id": group, "shape_type": "polygon",
                            "points": [[1, 1], [4, 1], [4, 4]], "description": "note"}
                           for group in (None, None, 0, 1, 1)]}
        path.write_text(json.dumps(data), encoding="utf-8")
        converter = LABELME()
        anno = converter.load_one_labelme_json(str(path))
        self.assertEqual(len(set(obj.group for obj in anno.objs)), 4)
        self.assertEqual(anno.objs[2].group, 0)
        converter.save_one_labelme_json(anno, str(path))
        restored = converter.load_one_labelme_json(str(path))
        self.assertEqual([obj.note for obj in restored.objs], ["note"] * 5)
        self.assertEqual([obj.group for obj in restored.objs], [obj.group for obj in anno.objs])

    def test_excluding_all_crowd_produces_valid_empty_outputs(self):
        anno = example_anno()
        anno.objs[0].iscrowd = True
        yolo, voc = YOLO(), VOCDetect()
        yolo.keep_crowd = voc.keep_crowd = False
        yolo.save_one_yolo_txt(anno, str(self.root / "image.txt"), {"tool": 0})
        voc.save_one_voc_xml(anno, str(self.root / "image.xml"))
        self.assertEqual((self.root / "image.txt").read_text(), "")
        self.assertEqual(ElementTree.parse(self.root / "image.xml").findall("object"), [])

    def test_coco_group_with_mixed_categories_has_no_phantom_objects(self):
        anno = example_anno()
        second = example_anno().objs[0]
        second.category, second.iscrowd = "other", True
        anno.objs += (second,)
        converter = COCO()
        converter.annos = {"image": anno}
        path = self.root / "coco.json"
        converter.save_to_coco(str(path), ("tool", "other"))
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["annotations"]), 2)
        self.assertTrue(all(item["segmentation"] and item["area"] > 0 for item in data["annotations"]))
        converter.keep_crowd = False
        converter.save_to_coco(str(path), ("tool", "other"))
        self.assertEqual(len(json.loads(path.read_text())["annotations"]), 1)

    def test_internal_roundtrip_preserves_identity_and_review_metadata(self):
        path = self.root / "image.json"
        converter, anno = IVAS(), example_anno()
        converter.save_one_ivas_json(anno, str(path))
        restored = converter.load_one_ivas_json(str(path))
        self.assertEqual(restored.objs[0].to_dict(), anno.objs[0].to_dict())

    def test_bad_annotation_refuses_overwrite_and_leaves_old_objects(self):
        image, label = self.root / "image.png", self.root / "image.json"
        Image.new("RGB", (16, 16)).save(image)
        for payload in ('{"info":', json.dumps({"info": {"description": "unknown"}, "objects": []}),
                        json.dumps({"info": {"description": "IVAS"}, "objects": [{"segmentation": []}]}),
                        json.dumps({"info": {"description": "IVAS"}, "objects": [{"segmentation": [[0, 0], [1, 1], [float("nan"), 2]]}]})):
            with self.subTest(payload=payload):
                label.write_text(payload, encoding="utf-8")
                annotation = Annotation(str(image), str(label))
                annotation.objects = [example_anno().objs[0]]
                with self.assertRaises(ValueError):
                    annotation.load_annotation()
                with self.assertRaises(ValueError):
                    annotation.save_annotation()
                self.assertEqual(label.read_text(encoding="utf-8"), payload)
                self.assertEqual(len(annotation.objects), 1)

    def test_exif_rejects_empty_output_and_copies_unrotated_images(self):
        source, target = self.root / "source", self.root / "target"
        source.mkdir(); target.mkdir()
        Image.new("RGB", (8, 6), "red").save(source / "plain.jpg")
        exif = Image.Exif(); exif[274] = 6
        Image.new("RGB", (8, 6)).save(source / "rotate.jpg", exif=exif)
        with self.assertRaises(ValueError):
            process_exif(str(source), "", True, threading.Event(), lambda *args: None)
        process_exif(str(source), str(target), True, threading.Event(), lambda *args: None)
        with Image.open(target / "rotate.jpg") as image:
            self.assertEqual(image.size, (6, 8))
            self.assertNotIn(274, image.getexif())
        self.assertTrue((target / "plain.jpg").is_file())

    def test_validator_continues_after_each_bad_object(self):
        payload = {"info": {"width": 16, "height": 16}, "objects": [
            {"segmentation": []}, {"segmentation": [[-1, 0], [8, 0], [8, 8]]}, example_anno().objs[0].to_dict()]}
        (self.root / "image.json").write_text(json.dumps(payload))
        result = validate_annotations(str(self.root), threading.Event(), lambda *args: None)
        self.assertEqual(result, "2 error(s)")

    def test_text_batch_assigns_one_group_per_mask_across_prompts(self):
        path = Path(__file__).resolve().parents[1] / "tools/auto_labeling_base_on_text_prompt/main.py"
        spec = importlib.util.spec_from_file_location("standalone_batch", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Image.new("RGB", (16, 16)).save(self.root / "image.jpg")
        masks = np.zeros((2, 16, 16), np.uint8)
        masks[0, 1:5, 1:5] = 1
        masks[0, 10:14, 10:14] = 1
        masks[1, 6:9, 6:9] = 1
        predictor = SimpleNamespace(predict_with_text_prompt=lambda *args: (masks, [0.9, 0.8]))
        module.text_prompt_and_save_to_ivas_json(SimpleNamespace(predictor=predictor), ["tool", "other"], str(self.root))
        data = json.loads((self.root / "image.json").read_text())
        self.assertEqual(len(data["objects"]), 6)
        self.assertEqual(len(set(obj["group"] for obj in data["objects"])), 4)


if __name__ == "__main__":
    unittest.main()
