"""Axis-aligned rectangle ("bbox") support in the data model and every format."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image

import IVAS.annotation as annotation_module
from IVAS.annotation import Object, rectangle_segmentation
from IVAS.formats.coco import COCO
from IVAS.formats.ivas import IVAS
from IVAS.formats.labelme import LABELME
from IVAS.formats.voc import VOC
from IVAS.formats.voc_detection import VOCDetect
from IVAS.formats.yolo import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def points(segmentation):
    return [[float(x), float(y)] for x, y in segmentation]


def make_anno(objs, width=100, height=50, name="image.png"):
    anno = IVAS.ANNO()
    anno.info = IVAS.ANNO.INFO()
    anno.info.description = "IVAS"
    anno.info.folder = ""
    anno.info.name = name
    anno.info.width = width
    anno.info.height = height
    anno.info.depth = 3
    anno.info.note = ""
    anno.objs = tuple(objs)
    return anno


def rectangle(category, group, xmin, ymin, xmax, ymax, **kwargs):
    return Object(category, group, rectangle_segmentation(xmin, ymin, xmax, ymax),
                  (xmax - xmin) * (ymax - ymin), 1, [xmin, ymin, xmax, ymax],
                  shape_type="rectangle", **kwargs)


def polygon(category, group, segmentation, **kwargs):
    xs = [x for x, _ in segmentation]
    ys = [y for _, y in segmentation]
    return Object(category, group, segmentation, 0, 1, [min(xs), min(ys), max(xs), max(ys)], **kwargs)


def obb(category, group):
    # A diamond: rotated square around (50, 25).
    return Object(category, group, [(50, 15), (60, 25), (50, 35), (40, 25)], 200, 1,
                  [40, 15, 60, 35], is_obb=True)


def mixed_anno():
    return make_anno([
        polygon("tool", 1, [(2, 2), (12, 2), (12, 9)]),
        rectangle("tool", 2, 20, 10, 30, 22, note="box"),
        obb("organ", 3),
    ])


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        tmp_root = REPO_ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=str(tmp_root), prefix="bbox-formats-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class ObjectShapeTypeTests(unittest.TestCase):
    def test_normalisation_and_is_obb_consistency(self):
        base = ("tool", 1, [(0, 0), (1, 0), (1, 1)], 1, 1, [0, 0, 1, 1])
        self.assertEqual(Object(*base).shape_type, "polygon")
        self.assertEqual(Object(*base, is_obb=True).shape_type, "obb")
        self.assertEqual(Object(*base, shape_type="rectangle").shape_type, "rectangle")
        self.assertFalse(Object(*base, shape_type="rectangle").is_obb)
        self.assertEqual(Object(*base, shape_type=" Rectangle ").shape_type, "rectangle")
        self.assertEqual(Object(*base, is_obb=True, shape_type="circle").shape_type, "obb")
        self.assertEqual(Object(*base, shape_type="circle").shape_type, "polygon")
        # shape_type is the LAST parameter; existing positional parameters keep their order.
        positional = Object(*base, False, "", True, "id", "track", "sam", "draft", "origin", "rectangle")
        self.assertEqual((positional.object_id, positional.shape_type), ("id", "rectangle"))
        self.assertEqual(IVAS.ANNO.OBJ().shape_type, "polygon")

        obj = Object(*base)
        obj.is_obb = True
        self.assertEqual((obj.shape_type, obj.is_obb), ("obb", True))
        obj.shape_type = "unknown"  # unknown keeps following is_obb
        self.assertEqual(obj.shape_type, "obb")
        obj.is_obb = False
        self.assertEqual(obj.shape_type, "polygon")
        obj.shape_type = "rectangle"
        obj.is_obb = False  # clearing the flag does not demote a rectangle
        self.assertEqual(obj.shape_type, "rectangle")
        obj.shape_type = "obb"
        self.assertTrue(obj.is_obb)

    def test_rectangle_segmentation_helper(self):
        self.assertIn("rectangle_segmentation", annotation_module.__all__)
        self.assertEqual(rectangle_segmentation(1, 2, 5, 7), [(1, 2), (5, 2), (5, 7), (1, 7)])
        self.assertEqual(rectangle_segmentation(5, 7, 1, 2), [(1, 2), (5, 2), (5, 7), (1, 7)])

    def test_legacy_dicts_and_to_dict(self):
        square = [[0, 0], [4, 0], [4, 4], [0, 4]]
        self.assertEqual(Object.from_dict({"segmentation": square}).shape_type, "polygon")
        self.assertEqual(Object.from_dict({"segmentation": square, "is_obb": True}).shape_type, "obb")
        self.assertEqual(Object.from_dict({"segmentation": square, "is_obb": "false"}).shape_type, "polygon")
        self.assertEqual(Object.from_dict({"segmentation": square, "is_obb": True,
                                           "shape_type": "bogus"}).shape_type, "obb")
        scrambled = Object.from_dict({"segmentation": [[4, 4], [0, 0], [4, 0], [0, 4]],
                                      "shape_type": "rectangle"})
        self.assertEqual(scrambled.shape_type, "rectangle")
        self.assertEqual(points(scrambled.segmentation), points(rectangle_segmentation(0, 0, 4, 4)))

        for shape_type in ("polygon", "rectangle", "obb"):
            data = Object.from_dict({"segmentation": square, "shape_type": shape_type}).to_dict()
            self.assertEqual(data["shape_type"], shape_type)
            self.assertEqual(data["is_obb"], shape_type == "obb")


class IVASJsonTests(TempDirTestCase):
    def test_mixed_round_trip(self):
        path = self.root / "image.json"
        converter, anno = IVAS(), mixed_anno()
        converter.save_one_ivas_json(anno, str(path))
        raw = json.loads(path.read_text(encoding="utf-8"))["objects"]
        self.assertEqual([item["shape_type"] for item in raw], ["polygon", "rectangle", "obb"])
        self.assertEqual([item["is_obb"] for item in raw], [False, False, True])

        restored = converter.load_one_ivas_json(str(path))
        self.assertEqual([obj.shape_type for obj in restored.objs], ["polygon", "rectangle", "obb"])
        for before, after in zip(anno.objs, restored.objs):
            self.assertEqual(points(after.segmentation), points(before.segmentation))
            self.assertEqual(list(after.bbox), list(before.bbox))
        self.assertEqual(restored.objs[1].note, "box")

    def test_legacy_file_without_shape_type(self):
        path = self.root / "legacy.json"
        objects = [obj.to_dict() for obj in mixed_anno().objs]
        for item in objects:
            item.pop("shape_type")
        path.write_text(json.dumps({"info": {"description": "ISAT", "width": 100, "height": 50},
                                    "objects": objects}), encoding="utf-8")
        restored = IVAS().load_one_ivas_json(str(path))
        self.assertEqual([obj.shape_type for obj in restored.objs], ["polygon", "polygon", "obb"])


class COCOTests(TempDirTestCase):
    def test_export_import_round_trip(self):
        anno = make_anno([
            polygon("tool", 1, [(2, 2), (12, 2), (12, 9)]),
            polygon("tool", 1, [(14, 2), (18, 2), (18, 6)]),
            rectangle("tool", 1, 20, 10, 30, 22),
            obb("organ", 2),
        ])
        converter = COCO()
        converter.annos = {"image": anno}
        path = self.root / "coco.json"
        converter.save_to_coco(str(path), ("tool", "organ"))
        data = json.loads(path.read_text(encoding="utf-8"))["annotations"]
        self.assertEqual(len(data), 3)
        merged = [item for item in data if "ivas_shape_type" not in item]
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["segmentation"]), 2)  # polygons still merge
        box = next(item for item in data if item.get("ivas_shape_type") == "rectangle")
        self.assertEqual(box["segmentation"], [[20, 10, 30, 10, 30, 22, 20, 22]])
        self.assertEqual(box["bbox"], [20.0, 10.0, 10.0, 12.0])
        self.assertEqual(box["area"], 120.0)
        oriented = next(item for item in data if item.get("ivas_shape_type") == "obb")
        self.assertEqual(oriented["bbox"], [40.0, 15.0, 20.0, 20.0])

        restored = COCO()
        restored.read_from_coco(str(path))
        objs = restored.annos["image"].objs
        self.assertEqual(sorted(obj.shape_type for obj in objs), ["obb", "polygon", "polygon", "rectangle"])
        rect = next(obj for obj in objs if obj.shape_type == "rectangle")
        self.assertEqual(points(rect.segmentation), points(rectangle_segmentation(20, 10, 30, 22)))
        self.assertEqual(list(rect.bbox), [20, 10, 30, 22])
        self.assertEqual(rect.area, 120)
        oriented = next(obj for obj in objs if obj.shape_type == "obb")
        self.assertTrue(oriented.is_obb)
        self.assertEqual(points(oriented.segmentation), points(obb("organ", 2).segmentation))

    def test_detection_only_annotations_and_integer_iscrowd(self):
        path = self.root / "detection.json"
        path.write_text(json.dumps({
            "images": [{"id": 7, "file_name": "frame.jpg", "width": 64, "height": 48}],
            "categories": [{"id": 1, "name": "tool"}, {"id": 2, "name": "organ"}],
            "annotations": [
                {"id": 1, "image_id": 7, "category_id": 1, "iscrowd": 0, "segmentation": [],
                 "bbox": [4, 6, 10, 5], "area": 50},
                {"id": 2, "image_id": 7, "category_id": 2, "iscrowd": 0,
                 "bbox": [30.5, 1, 2.5, 3], "area": 7.5},
                {"id": 3, "image_id": 7, "category_id": 1, "iscrowd": 0,
                 "segmentation": [[1, 1, 9, 1, 9, 9]], "bbox": [1, 1, 8, 8], "area": 32},
                {"id": 4, "image_id": 7, "category_id": 1, "iscrowd": 0, "segmentation": [],
                 "bbox": [], "area": 0},
            ],
        }), encoding="utf-8")
        converter = COCO()
        converter.read_from_coco(str(path))
        objs = converter.annos["frame"].objs
        self.assertEqual([obj.shape_type for obj in objs], ["rectangle", "rectangle", "polygon"])
        self.assertEqual([obj.category for obj in objs], ["tool", "organ", "tool"])
        self.assertEqual(len({obj.group for obj in objs}), 3)
        self.assertEqual(list(objs[0].bbox), [4, 6, 14, 11])
        self.assertEqual(points(objs[0].segmentation), points(rectangle_segmentation(4, 6, 14, 11)))
        self.assertEqual(objs[0].area, 50)
        self.assertEqual(list(objs[1].bbox), [30.5, 1, 33, 4])
        self.assertFalse(objs[2].iscrowd)


class YOLOTests(TempDirTestCase):
    def detection_anno(self):
        return make_anno([
            # group 1 / "a": union of rectangle (10,5)-(30,25) and polygon up to (40,20)
            rectangle("a", 1, 10, 5, 30, 25),
            polygon("a", 1, [(20, 10), (40, 10), (40, 20)]),
            # group 2 / "b": partly outside the 100x50 image -> clipped to (90,40)-(100,50)
            rectangle("b", 2, 90, 40, 120, 60),
            # skipped: crowd (keep_crowd False) and background (ignore_background True)
            rectangle("a", 3, 0, 0, 10, 10, iscrowd=True),
            rectangle("__background__", 4, 0, 0, 100, 50),
        ])

    def read_lines(self, path):
        return [[float(value) for value in line.split()] for line in path.read_text().splitlines()]

    def test_detection_export_values(self):
        converter = YOLO()
        self.assertFalse(converter.detection)
        converter.detection = True
        converter.keep_crowd = False
        converter.ignore_background = True
        path = self.root / "image.txt"
        converter.save_one_yolo_txt(self.detection_anno(), str(path), {"a": 0, "b": 1, "__background__": 2})
        lines = self.read_lines(path)
        expected = [[0, 0.25, 0.3, 0.3, 0.4], [1, 0.95, 0.9, 0.1, 0.2]]
        self.assertEqual(len(lines), len(expected))
        for line, values in zip(lines, expected):
            self.assertEqual(len(line), 5)
            for actual, value in zip(line, values):
                self.assertAlmostEqual(actual, value, places=9)

        converter.keep_crowd, converter.ignore_background = True, False
        converter.save_one_yolo_txt(self.detection_anno(), str(path), {"a": 0, "b": 1, "__background__": 2})
        self.assertEqual(len(self.read_lines(path)), 4)

    def test_segmentation_export_writes_rectangle_corners(self):
        converter = YOLO()
        path = self.root / "image.txt"
        converter.save_one_yolo_txt(make_anno([rectangle("a", 1, 10, 5, 30, 25)]), str(path), {"a": 0})
        self.assertEqual(self.read_lines(path), [[0, 0.1, 0.1, 0.3, 0.1, 0.3, 0.5, 0.1, 0.5]])

    def test_import_detection_and_mixed_lines(self):
        Image.new("RGB", (100, 50)).save(self.root / "image.png")
        txt = self.root / "image.txt"
        txt.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n\n1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        converter = YOLO()
        anno = converter.load_one_yolo_txt(str(self.root / "image.png"), str(txt), {0: "a", 1: "b"})
        self.assertEqual([obj.shape_type for obj in anno.objs], ["polygon", "rectangle"])
        self.assertEqual([obj.category for obj in anno.objs], ["a", "b"])
        self.assertEqual([obj.group for obj in anno.objs], [1, 2])
        box = anno.objs[1]
        self.assertEqual(list(box.bbox), [40, 15, 60, 35])
        self.assertEqual(points(box.segmentation), points(rectangle_segmentation(40, 15, 60, 35)))
        self.assertEqual(box.area, 400)

        for bad in ("0 0.1 0.2 0.3\n", "0 0.1 0.2 0.3 0.4 0.5\n"):
            txt.write_text(bad, encoding="utf-8")
            with self.assertRaises(ValueError):
                converter.load_one_yolo_txt(str(self.root / "image.png"), str(txt))

    def test_detection_round_trip(self):
        images, labels = self.root / "images", self.root / "labels"
        images.mkdir()
        Image.new("RGB", (100, 50)).save(images / "image.png")
        exporter = YOLO()
        exporter.detection = True
        exporter.cates = ("a", "b")
        exporter.annos = {"image": make_anno([rectangle("a", 1, 10, 5, 30, 25),
                                              rectangle("b", 2, 60, 20, 80, 45)])}
        exporter.save_to_YOLO(str(labels))
        importer = YOLO()
        importer.read_from_YOLO(str(images), str(labels), {0: "a", 1: "b"})
        objs = importer.annos["image"].objs
        self.assertEqual([obj.shape_type for obj in objs], ["rectangle", "rectangle"])
        self.assertEqual([list(obj.bbox) for obj in objs], [[10, 5, 30, 25], [60, 20, 80, 45]])


class LabelMeTests(TempDirTestCase):
    def test_rectangle_round_trip(self):
        path = self.root / "image.json"
        converter = LABELME()
        converter.save_one_labelme_json(mixed_anno(), str(path))
        shapes = json.loads(path.read_text(encoding="utf-8"))["shapes"]
        self.assertEqual([shape["shape_type"] for shape in shapes], ["polygon", "rectangle", "polygon"])
        self.assertEqual(shapes[1]["points"], [[20, 10], [30, 22]])
        self.assertEqual(len(shapes[2]["points"]), 4)  # OBB stays a 4-point polygon

        restored = converter.load_one_labelme_json(str(path))
        self.assertEqual([obj.shape_type for obj in restored.objs], ["polygon", "rectangle", "polygon"])
        rect = restored.objs[1]
        self.assertEqual(points(rect.segmentation), points(rectangle_segmentation(20, 10, 30, 22)))
        self.assertEqual((list(rect.bbox), rect.area, rect.group, rect.note), ([20, 10, 30, 22], 120, 2, "box"))

    def test_rectangle_corners_in_any_order(self):
        path = self.root / "labelme.json"
        path.write_text(json.dumps({"imagePath": "image.png", "imageWidth": 64, "imageHeight": 64, "shapes": [
            {"label": "tool", "shape_type": "rectangle", "points": [[30, 40], [10, 20]], "group_id": None},
            {"label": "tool", "shape_type": "rectangle", "points": [[30, 20], [10, 40]], "group_id": None},
            {"label": "tool", "shape_type": "circle", "points": [[1, 1], [2, 2]]},
        ]}), encoding="utf-8")
        objs = LABELME().load_one_labelme_json(str(path)).objs
        self.assertEqual(len(objs), 2)
        for obj in objs:
            self.assertEqual(obj.shape_type, "rectangle")
            self.assertEqual(points(obj.segmentation), points(rectangle_segmentation(10, 20, 30, 40)))
        self.assertEqual(len({obj.group for obj in objs}), 2)


class VOCTests(TempDirTestCase):
    def test_xml_export_import_round_trip(self):
        anno = make_anno([
            rectangle("a", 1, 10, 5, 30, 25),
            polygon("b", 2, [(50, 10), (70, 12), (60, 30)]),
            rectangle("a", 3, 80, 30, 90, 40),
            polygon("a", 3, [(85, 35), (95, 35), (95, 45)]),
            rectangle("a", 4, 0, 0, 5, 5, iscrowd=True),
        ], name="frame.jpg")
        exporter = VOCDetect()
        exporter.keep_crowd = False
        exporter.annos = {"frame": anno}
        exporter.save_to_XML(str(self.root))
        xml = ElementTree.parse(self.root / "frame.xml")
        self.assertEqual(len(xml.findall("object")), 3)  # crowd-only group emits nothing
        self.assertTrue(all(obj.find("bndbox") is not None for obj in xml.findall("object")))

        importer = VOCDetect()
        importer.read_from_XML(str(self.root))
        self.assertEqual(importer.cates, ("a", "b"))
        restored = importer.annos["frame"]
        self.assertEqual((restored.info.name, restored.info.width, restored.info.height, restored.info.depth),
                         ("frame.jpg", 100, 50, 3))
        self.assertEqual([obj.shape_type for obj in restored.objs], ["rectangle"] * 3)
        self.assertEqual([obj.group for obj in restored.objs], [1, 2, 3])
        boxes = {(obj.category, tuple(obj.bbox)) for obj in restored.objs}
        self.assertEqual(boxes, {("a", (10, 5, 30, 25)), ("b", (50, 10, 70, 30)), ("a", (80, 30, 95, 45))})
        for obj in restored.objs:
            self.assertEqual(points(obj.segmentation), points(rectangle_segmentation(*obj.bbox)))

    def test_xml_import_tolerates_float_coordinates(self):
        path = self.root / "float.xml"
        path.write_text(
            "<annotation><filename>float.png</filename>"
            "<size><width>64.0</width><height>32</height><depth>3</depth></size>"
            "<object><name> tool </name><bndbox><xmin>1.5</xmin><ymin>2</ymin>"
            "<xmax>10.25</xmax><ymax>12.0</ymax></bndbox></object>"
            "<object><name>nobox</name></object></annotation>", encoding="utf-8")
        anno = VOCDetect().load_one_voc_xml(str(path))
        self.assertEqual((anno.info.width, anno.info.height), (64, 32))
        self.assertEqual(len(anno.objs), 1)
        obj = anno.objs[0]
        self.assertEqual((obj.category, obj.shape_type, list(obj.bbox)), ("tool", "rectangle", [1.5, 2, 10.25, 12]))

    def test_segmentation_png_rasterises_rectangles_and_obbs(self):
        converter = VOC()
        converter.cates = ("tool", "organ")
        converter.annos = {"image": make_anno([rectangle("tool", 1, 10, 10, 20, 20), obb("organ", 2)])}
        converter.save_to_voc(str(self.root))
        with Image.open(self.root / "image.png") as image:
            pixels = np.asarray(image)
        self.assertEqual(int(pixels[15, 15]), 1)  # inside the rectangle
        self.assertEqual(int(pixels[25, 50]), 2)  # centre of the OBB diamond
        self.assertEqual(int(pixels[16, 41]), 0)  # inside the OBB bbox, outside the diamond
        self.assertEqual(int(pixels[0, 0]), 0)


class ConverterWorkerTests(TempDirTestCase):
    def test_voc_xml_to_ivas_and_yolo_detection_workers(self):
        from IVAS.widgets.converter_dialog import VOCConverterForDetection, YOLOConverter

        xml_root, ivas_root, yolo_root = self.root / "xml", self.root / "ivas", self.root / "yolo"
        xml_root.mkdir()
        exporter = VOCDetect()
        exporter.annos = {"frame": make_anno([rectangle("a", 1, 10, 5, 30, 25)], name="frame.jpg")}
        exporter.save_to_XML(str(xml_root))

        worker = VOCConverterForDetection()
        self.assertFalse(worker.voc2ivas)
        worker.voc2ivas = True
        worker.voc_xml_root, worker.ivas_json_root = str(xml_root), str(ivas_root)
        worker.convert()
        self.assertEqual(worker.errors, [])
        saved = json.loads((ivas_root / "frame.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["objects"][0]["shape_type"], "rectangle")
        self.assertEqual(saved["objects"][0]["bbox"], [10.0, 5.0, 30.0, 25.0])

        yolo_root.mkdir()
        worker = YOLOConverter()
        worker.yolo2ivas, worker.detection = False, True
        worker.ivas_json_root, worker.yolo_txt_root = str(ivas_root), str(yolo_root)
        worker.convert()
        self.assertEqual(worker.errors, [])
        values = [float(value) for value in (yolo_root / "frame.txt").read_text().split()]
        self.assertEqual(len(values), 5)
        for actual, expected in zip(values, [0, 0.2, 0.3, 0.2, 0.4]):
            self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    unittest.main()
