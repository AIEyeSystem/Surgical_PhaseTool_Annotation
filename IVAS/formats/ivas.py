# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
from json import dump, load
from typing import Dict, Tuple
from IVAS.annotation import Object, rectangle_segmentation
from IVAS.utils.persistence import atomic_write_bytes
import json

import imgviz
import tqdm
import yaml
from IVAS.configs import save_config

from IVAS.branding import (
    ANNOTATION_FORMAT,
    CATEGORY_CONFIG_NAME,
    LEGACY_CATEGORY_CONFIG_NAMES,
    SUPPORTED_ANNOTATION_FORMATS,
)


class IVAS:
    """
    The IVAS format provides a structured approach for representing image annotations.

    File Naming: Each image has a corresponding .json file named after the image file (without the image extension).

    Attributes:
        annos (dict): Dictionary of image annotations. {name: ANNO}
        cates (tuple): categories.

    """

    class ANNO:
        r"""Annotation class.

        Attributes:
            info (INFO): INFO class.
            objs (tuple[OBJ, ...]): tuple of OBJs.

        """

        class INFO:
            r"""Contains metadata about the dataset and image.

            Attributes:
                description (str): "IVAS" for new files; "ISAT" and "IVAT" are accepted.
                folder (str): The directory where the images are stored.
                name (str): The name of the image file
                width (int): The dimensions of the image.
                height (int): The dimensions of the image.
                depth (int): The dimensions of the image; depth is assumed to be 3 for RGB images.
                note: An optional field for any additional notes related to the image.
            """

            description = ""
            folder = ""
            name = ""
            width = None
            height = None
            depth = None
            note = ""

        class OBJ(Object):
            r"""Lists all the annotated objects in the image.

            Attributes:
                category (str): The class label of the object.
                group (int): An identifier that groups objects based on overlapping bounding boxes. If an object's bounding box is within another, they share the same group number.
                segmentation (list | tuple): A list of [x, y] coordinates forming the polygon around the object.
                area (float): The area covered by the object in pixels.
                layer (int): A float indicating the sequence of the object. It increments within the same group, starting at 1.0 .
                bbox (list | tuple): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
                iscrowd (bool): A boolean value indicating if the object is part of a crowd.
                note (str): An optional field for any additional notes related to the object.
                shape_type (str): "polygon" (default), "rectangle" or "obb". Rectangles store
                    their 4 corners clockwise from the top-left in ``segmentation``.
            """

            category = ""
            group = None
            segmentation = None
            area = None
            layer = None
            bbox = None
            iscrowd = False
            note = ""

            def __init__(self):
                super().__init__("", 0, [], 0, 2, [])

        info: INFO
        objs: Tuple[OBJ, ...] = ()

    cates: Tuple[str] = ()

    def __init__(self):
        self.annos = {}
        self.cates = ()

    def read_from_IVAS(self, json_root: str) -> bool:
        r"""Load annotations from a directory of json files.

        Arguments:
            json_root (str): The directory of json files.
        """
        self.annos.clear()
        self.cates = ()

        category_path = next(
            (
                os.path.join(json_root, name)
                for name in (CATEGORY_CONFIG_NAME, *LEGACY_CATEGORY_CONFIG_NAMES)
                if os.path.exists(os.path.join(json_root, name))
            ),
            None,
        )
        if category_path is not None:
            cates = []
            with open(category_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f.read()) or {}
            for label in cfg.get("label", []):
                cates.append(label.get("name"))
            self.cates = tuple(cates)

        pbar = tqdm.tqdm(
            [file for file in os.listdir(json_root) if file.endswith(".json")]
        )
        for file in pbar:
            pbar.set_description("Load IVAS from {}".format(file))
            anno = self.load_one_ivas_json(os.path.join(json_root, file))
            self.annos[self.remove_file_suffix(file)] = anno
        return True

    def save_to_IVAS(self, json_root: str) -> bool:
        r"""
        Save annotations to the directory of json files.

        Arguments:
            json_root (str): The directory of json files.
        """
        os.makedirs(json_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, Anno in pbar:
            json_name = name_without_suffix + ".json"
            pbar.set_description("Save IVAS to {}".format(json_name))
            self.save_one_ivas_json(Anno, os.path.join(json_root, json_name))

        # 类别文件
        cmap = imgviz.label_colormap()
        self.cates = sorted(self.cates)
        categories = []
        for index, cat in enumerate(self.cates):
            r, g, b = cmap[(index + 1) % len(cmap)]
            categories.append(
                {
                    "name": cat if isinstance(cat, str) else str(cat),
                    "color": "#{:02x}{:02x}{:02x}".format(r, g, b),
                }
            )
        save_config({"label": categories}, os.path.join(json_root, CATEGORY_CONFIG_NAME))

        return True

    def remove_file_suffix(self, file_name: str) -> str:
        r"""
        Remove the file suffix from the file name.

        Arguments:
            file_name (str): The file name.
        Returns:
            str: The file name without the file suffix.
        """
        return os.path.splitext(file_name)[0]

    @staticmethod
    def segmentation_bounds(segmentation) -> Tuple[float, float, float, float]:
        r"""
        Return the (xmin, ymin, xmax, ymax) bounds of a list of (x, y) vertices.
        """
        xs = [point[0] for point in segmentation]
        ys = [point[1] for point in segmentation]
        if not xs:
            raise ValueError("Cannot compute the bounds of an empty segmentation")
        return min(xs), min(ys), max(xs), max(ys)

    def make_rectangle_obj(
        self,
        category,
        group: int,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        layer: float = 1,
        iscrowd: bool = False,
        note: str = "",
    ) -> "IVAS.ANNO.OBJ":
        r"""
        Build an axis-aligned rectangle OBJ from its bounds.
        """
        segmentation = rectangle_segmentation(xmin, ymin, xmax, ymax)
        (xmin, ymin), (xmax, ymax) = segmentation[0], segmentation[2]
        obj = self.ANNO.OBJ()
        obj.category = category
        obj.group = group
        obj.segmentation = segmentation
        obj.area = (xmax - xmin) * (ymax - ymin)
        obj.layer = layer
        obj.bbox = [xmin, ymin, xmax, ymax]
        obj.iscrowd = iscrowd
        obj.note = note
        obj.shape_type = "rectangle"
        return obj

    def load_one_ivas_json(self, json_path: str) -> ANNO:
        r"""
        Load annotation from a json file.

        Arguments:
            json_path (str): The file path.
        Returns:
            ANNO: The instance of the ANNO.
        """
        anno = self.ANNO()
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = load(f)
            info = dataset.get("info", {})
            description = info.get("description", "")
            if description not in SUPPORTED_ANNOTATION_FORMATS:
                raise AttributeError(
                    "The json file {} is not an IVAS annotation JSON.".format(json_path)
                )
            folder = info.get("folder", "")
            img_name = info.get("name", "")
            width = info.get("width", None)
            height = info.get("height", None)
            depth = info.get("depth", None)
            note = info.get("note", "")

            anno.info = self.ANNO.INFO()
            anno.info.description = description
            anno.info.folder = folder
            anno.info.name = img_name
            anno.info.width = width
            anno.info.height = height
            anno.info.depth = depth
            anno.info.note = note

            objs = []
            objects = dataset.get("objects", [])
            for obj in objects:
                objs.append(Object.from_dict(obj))

            anno.objs = tuple(objs)
        return anno

    def save_one_ivas_json(self, anno: ANNO, save_path: str) -> bool:
        r"""
        Save annotation to a json file.

        Arguments:
            anno (ANNO): The instance of the ANNO.
            save_path (str): The IVAS annotation JSON file path.
        """
        anno.info.description = ANNOTATION_FORMAT
        dataset = {}
        dataset["info"] = {}
        dataset["info"]["description"] = anno.info.description
        dataset["info"]["folder"] = anno.info.folder
        dataset["info"]["name"] = anno.info.name
        dataset["info"]["width"] = anno.info.width
        dataset["info"]["height"] = anno.info.height
        dataset["info"]["depth"] = anno.info.depth
        dataset["info"]["note"] = anno.info.note
        dataset["objects"] = []
        for obj in anno.objs:
            dataset["objects"].append(obj.to_dict())

        atomic_write_bytes(save_path, json.dumps(dataset, indent=4, ensure_ascii=False).encode("utf-8"))
        return True
