# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
import tempfile
import math
from json import dump, load
from typing import List, Union

import numpy as np
from PIL import Image

from IVAS.branding import ANNOTATION_FORMAT, SUPPORTED_ANNOTATION_FORMATS

__all__ = [
    "Object",
    "Annotation",
    "SHAPE_TYPES",
    "normalize_shape_type",
    "rectangle_segmentation",
]

SHAPE_TYPES = ("polygon", "rectangle", "obb")


def normalize_shape_type(shape_type, is_obb: bool = False) -> str:
    r"""Return a valid shape type; empty/unknown values fall back on ``is_obb``."""
    value = shape_type.strip().lower() if isinstance(shape_type, str) else ""
    if value in SHAPE_TYPES:
        return value
    return "obb" if is_obb else "polygon"


def rectangle_segmentation(xmin, ymin, xmax, ymax) -> list:
    r"""Return the 4 axis-aligned corners clockwise (image coords) from top-left.

    The bounds are normalised so that ``xmin <= xmax`` and ``ymin <= ymax``.
    """
    xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
    ymin, ymax = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


class Object:
    r"""A class to represent an annotation object.

    Arguments:
        category (str): The category of the object.
        group (int): The group of the object.
        segmentation (list | tuple): The vertices of the object.[(x1, y1), (x2, y2), ...]
            Rectangles always store the 4 corners from :func:`rectangle_segmentation`.
        area (float): The area of the object.
        layer (int): The layer of the object.
        bbox (list | tuple): The bbox of the object. [xmin, ymin, xmax, ymax]
        iscrowd (bool): The crowd tag of the object.
        note (str): The note of the object.
        is_obb (bool): Legacy oriented-box flag; equivalent to ``shape_type == "obb"``.
        shape_type (str): ``"polygon"``, ``"rectangle"`` or ``"obb"``. Empty or
            unknown values become ``"obb"`` when ``is_obb`` is set, else ``"polygon"``.
    """

    def __init__(
        self,
        category: str,
        group: int,
        segmentation: Union[list, tuple],
        area: float,
        layer: int,
        bbox: Union[list, tuple],
        iscrowd: bool = False,
        note: str = "",
        is_obb: bool = False,
        object_id: str = "",
        track_id: str = "",
        source: str = "manual",
        review_status: str = "reviewed",
        origin_source: str = "",
        shape_type: str = "",
    ):
        self.category = category
        self.group = group
        self.segmentation = segmentation
        self.area = area
        self.layer = layer
        self.bbox = bbox
        self.iscrowd = iscrowd
        self.note = note
        self._shape_type = normalize_shape_type(shape_type, bool(is_obb))
        self.object_id = str(object_id or "")
        self.track_id = str(track_id or "")
        self.source = str(source or "manual")
        self.review_status = str(review_status or "reviewed")
        self.origin_source = str(origin_source or "")

    # ``shape_type`` is the source of truth; ``is_obb`` is a legacy view of it.
    @property
    def shape_type(self) -> str:
        return getattr(self, "_shape_type", "polygon")

    @shape_type.setter
    def shape_type(self, value):
        self._shape_type = normalize_shape_type(value, self.is_obb)

    @property
    def is_obb(self) -> bool:
        return self.shape_type == "obb"

    @is_obb.setter
    def is_obb(self, value):
        if value:
            self._shape_type = "obb"
        elif self.is_obb:
            self._shape_type = "polygon"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Annotation object must be a JSON object")
        points = value.get("segmentation", [])
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            raise ValueError("Annotation polygon requires at least three vertices")
        if any(not isinstance(point, (list, tuple)) or len(point) != 2
               or any(not isinstance(number, (int, float)) or not math.isfinite(number)
                      for number in point) for point in points):
            raise ValueError("Annotation vertices must contain finite x/y coordinates")
        def boolean(item):
            return item.strip().lower() in ("1", "true", "yes", "on") if isinstance(item, str) else bool(item)
        layer = value.get("layer", 2)
        if not isinstance(layer, (int, float)) or not math.isfinite(layer):
            raise ValueError("Annotation layer must be finite")
        is_obb = boolean(value.get("is_obb", False))
        shape_type = normalize_shape_type(value.get("shape_type", ""), is_obb)
        if shape_type == "rectangle":
            # Keep the documented corner order even for hand-edited files.
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            points = [list(point) for point in rectangle_segmentation(min(xs), min(ys), max(xs), max(ys))]
        return cls(str(value.get("category", "unknown")), int(value.get("group") or 0), points,
                   value.get("area") or 0, layer, value.get("bbox", []),
                   boolean(value.get("iscrowd", False)), str(value.get("note") or ""),
                   is_obb, value.get("object_id", ""),
                   value.get("track_id", ""), value.get("source", "manual"),
                   value.get("review_status", "reviewed"), value.get("origin_source", ""),
                   shape_type=shape_type)

    def to_dict(self):
        result = {name: getattr(self, name) for name in (
            "category", "group", "segmentation", "area", "layer", "bbox", "iscrowd", "note",
            "shape_type", "is_obb", "source", "review_status")}
        for name in ("object_id", "track_id", "origin_source"):
            if getattr(self, name, ""):
                result[name] = getattr(self, name)
        return result


class Annotation:
    r"""A class to represent an annotation containing many objects.

    Arguments:
        image_path (str): The path to the image.
        label_path (str): The path to the label file.

    Attributes:
        description (str): Always 'IVAS' for new files.
        img_folder (str): The path to the folder where the images are located.
        img_name (str): The name of the image.
        label_path (str): The path to the label file.
        note (str): The note of the image.
        height (int): The height of the image.
        width (int): The width of the image.
        depth (int): The depth of the image.
    """

    def __init__(self, image_path: str, label_path: str):
        img_folder, img_name = os.path.split(image_path)
        self.description = ANNOTATION_FORMAT
        self.img_folder = img_folder
        self.img_name = img_name
        self.label_path = label_path
        self.note = ""
        self._load_failed = False

        if image_path.lower().endswith(".dcm"):
            from IVAS.utils.dicom import load_dcm_as_image

            image = np.array(load_dcm_as_image(image_path))
        else:
            with Image.open(image_path) as source_image:
                image = np.array(source_image)
        if image.ndim == 3:
            self.height, self.width, self.depth = image.shape
        elif image.ndim == 2:
            self.height, self.width = image.shape
            self.depth = 0
        else:
            raise ValueError(
                "Expected a 2D or 3D image, got shape {} for {!r}.".format(
                    image.shape, image_path
                )
            )
        del image

        self.objects: List[Object,] = []

    def load_annotation(self):
        r"""
        Load annotation from self.label_path
        """
        self._load_failed = True
        parsed_objects = []
        info = {}
        if os.path.exists(self.label_path):
            with open(self.label_path, "r", encoding="utf-8") as f:
                dataset = load(f)
                if not isinstance(dataset, dict) or not isinstance(dataset.get("info"), dict):
                    raise ValueError("Annotation must contain an info object")
                info = dataset["info"]
                description = info.get("description", "")
                if description in SUPPORTED_ANNOTATION_FORMATS:
                    # Original ISAT and previous IVAT files share this schema.
                    objects = dataset.get("objects", [])
                    if not isinstance(objects, list):
                        raise ValueError("Annotation objects must be a list")
                    for obj in objects:
                        parsed_objects.append(Object.from_dict(obj))
                else:
                    raise ValueError("The file {} is not an IVAS annotation JSON".format(self.label_path))
        for name in ("width", "height", "depth"):
            if info.get(name) is None:
                info[name] = getattr(self, name)
            value = info[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid annotation image {}".format(name))
        self.objects = parsed_objects
        self.img_name = info.get("name", self.img_name)
        for name in ("width", "height", "depth"):
            setattr(self, name, info.get(name, getattr(self, name)))
        self.note = str(info.get("note") or "")
        self._load_failed = False
        return self

    def save_annotation(self):
        r"""
        Save annotation to self.label_path
        """
        if self._load_failed:
            raise ValueError("Cannot overwrite an annotation that failed to load")
        dataset = {}
        dataset["info"] = {}
        dataset["info"]["description"] = self.description
        dataset["info"]["folder"] = self.img_folder
        dataset["info"]["name"] = self.img_name
        dataset["info"]["width"] = self.width
        dataset["info"]["height"] = self.height
        dataset["info"]["depth"] = self.depth
        dataset["info"]["note"] = self.note
        dataset["objects"] = []
        for obj in self.objects:
            dataset["objects"].append(obj.to_dict())
        label_dir = os.path.dirname(os.path.abspath(self.label_path))
        os.makedirs(label_dir, exist_ok=True)
        handle, temporary_path = tempfile.mkstemp(
            prefix=".ivas-", suffix=".json.tmp", dir=label_dir
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                dump(dataset, stream, indent=4, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.label_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
        return True
