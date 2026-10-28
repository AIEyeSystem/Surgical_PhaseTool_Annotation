# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
from json import dump, load, dumps
from IVAS.utils.persistence import atomic_write_bytes

import tqdm

from IVAS.branding import ANNOTATION_FORMAT
from IVAS.formats.ivas import IVAS


class LABELME(IVAS):
    """
    LABELME format

    Attributes:
        keep_crowd (bool): keep the crowded objects
    """

    def __init__(self):
        super().__init__()
        self.keep_crowd = True

    def read_from_LABELME(self, json_root: str) -> bool:
        """
        Load annotations from the directory of labelme json files.

        Arguments:
            json_root (str): the directory of label json files.
        """
        self.annos.clear()
        self.cates = ()

        json_files = [file for file in os.listdir(json_root) if file.endswith(".json")]
        pbar = tqdm.tqdm(json_files)

        for file in pbar:
            name_without_suffix = self.remove_file_suffix(file)
            pbar.set_description(
                "Load labelme json {}".format(name_without_suffix + ".json")
            )
            anno = self.load_one_labelme_json(os.path.join(json_root, file))
            self.annos[name_without_suffix] = anno

        class_set = set()
        for _, anno in self.annos.items():
            for obj in anno.objs:
                class_set.add(obj.category)

        class_set = list(class_set)
        class_set.sort()
        self.cates = tuple(class_set)
        return True

    def save_to_LABELME(self, json_root: str) -> bool:
        """
        Save annotations to the directory of labelme json files.

        Arguments:
            json_root (str): the directory of label json files.
        """
        os.makedirs(json_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            json_path = os.path.join(json_root, name_without_suffix + ".json")
            try:
                self.save_one_labelme_json(anno, json_path)
                pbar.set_description(
                    "Save labelme to {}".format(name_without_suffix + ".json")
                )

            except Exception as e:
                raise ValueError("{}: {}".format(name_without_suffix, e)) from e
        return True

    def save_one_labelme_json(self, anno: IVAS.ANNO, json_path) -> bool:
        """
        Save annotation to a json file.

        Arguments:
            anno (IVAS.ANNO): the annotation.
            json_path (str): the path to save the json file.
        """
        labelme_anno = {}
        labelme_anno["version"] = "5.2.0.post4 | IVAS to LabelMe"
        labelme_anno["imagePath"] = anno.info.name
        labelme_anno["imageData"] = None
        labelme_anno["imageHeight"] = anno.info.height
        labelme_anno["imageWidth"] = anno.info.width
        labelme_anno["flags"] = {}
        labelme_anno["shapes"] = []

        for obj in anno.objs:
            category = obj.category
            group = obj.group
            segmentation = obj.segmentation
            iscrowd = obj.iscrowd
            if iscrowd:
                if not self.keep_crowd:
                    continue
            note = obj.note
            area = obj.area
            layer = obj.layer
            bbox = obj.bbox

            shape = {}
            shape["label"] = category
            if obj.shape_type == "rectangle":
                # LabelMe rectangles are two opposite corners: top-left, bottom-right.
                xmin, ymin, xmax, ymax = self.segmentation_bounds(segmentation)
                shape["points"] = [[xmin, ymin], [xmax, ymax]]
                shape["shape_type"] = "rectangle"
            else:
                # OBBs have no LabelMe equivalent and are kept as 4-point polygons.
                shape["points"] = segmentation
                shape["shape_type"] = "polygon"
            shape["group_id"] = int(group) if group is not None else None
            shape["description"] = note
            shape["flags"] = {}

            labelme_anno["shapes"].append(shape)

        atomic_write_bytes(json_path, dumps(labelme_anno, indent=4, ensure_ascii=False).encode("utf-8"))
        return True

    def load_one_labelme_json(self, json_path: str) -> IVAS.ANNO:
        """
        Load annotation from a labelme json file.

        Arguments:
            json_path (str): the path to save the json file.

        Returns:
            IVAS.ANNO: the annotation.
        """
        anno = self.ANNO()
        anno.info = self.ANNO.INFO()

        with open(json_path, "r", encoding="utf-8") as f:
            dataset = load(f)
            anno.info.description = ANNOTATION_FORMAT
            anno.info.folder = os.path.split(json_path)[0]
            anno.info.name = dataset.get("imagePath")
            anno.info.height = dataset.get("imageHeight")
            anno.info.width = dataset.get("imageWidth")
            anno.info.depth = 3
            anno.info.note = ""
            shapes = dataset.get("shapes", {})

            objs = []
            used_groups = {int(shape["group_id"]) for shape in shapes if shape.get("group_id") is not None}
            next_group = max(used_groups | {0}) + 1
            for shape in shapes:
                # 只加载多边形和矩形
                shape_type = shape.get("shape_type", "")
                if shape_type not in ("polygon", "rectangle"):
                    continue
                points = shape.get("points", [])
                if shape_type == "rectangle" and len(points) != 2:
                    continue
                obj = self.ANNO.OBJ()

                obj.category = shape.get("label", "unknow")
                obj.group = shape.get("group_id")
                if obj.group is None:
                    obj.group = next_group
                    next_group += 1
                obj.segmentation = points
                obj.area = shape.get("area", 0)
                obj.layer = shape.get("layer", 1)
                obj.bbox = shape.get("bbox", [])
                obj.iscrowd = shape.get("iscrowd", False)
                obj.note = shape.get("description", shape.get("note", "")) or ""
                if shape_type == "rectangle":
                    # The two corners may be given in any order.
                    obj = self.make_rectangle_obj(
                        obj.category, obj.group, *self.segmentation_bounds(points),
                        layer=obj.layer, iscrowd=obj.iscrowd, note=obj.note,
                    )

                objs.append(obj)
            anno.objs = tuple(objs)
        return anno
