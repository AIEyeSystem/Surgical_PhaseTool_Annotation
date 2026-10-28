# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import math
from json import dump

import cv2
import numpy as np
import tqdm
from pycocotools import coco as mscoco
from pycocotools import mask as mscoco_mask

from IVAS.annotation import normalize_shape_type
from IVAS.branding import SUPPORTED_ANNOTATION_FORMATS
from IVAS.formats.ivas import IVAS

__all__ = ["COCO"]


class COCO(IVAS):
    r"""
    COCO format

    Attributes:
        keep_crowd (bool): keep the crowded objects
    """

    def __init__(self):
        super().__init__()
        self.keep_crowd = True

    def _coco_polygon_obj(self, category, group, points, iscrowd):
        obj = self.ANNO.OBJ()
        obj.category = category
        obj.group = group
        obj.area = None
        obj.segmentation = points
        obj.layer = 1
        obj.bbox = None
        obj.iscrowd = iscrowd
        obj.note = ""
        return obj

    @staticmethod
    def _valid_coco_bbox(bbox):
        r"""Return ``(x, y, w, h)`` as floats when ``bbox`` is a usable COCO box."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) for value in bbox
        ):
            return None
        x, y, w, h = (float(value) for value in bbox)
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

    def read_from_coco(self, annotation_file: str) -> bool:
        r"""
        Load annotations from a COCO json file.

        Arguments:
            annotation_file (str): coco json annotations file.
        """
        self.annos.clear()
        self.cates = ()

        dataset = mscoco.COCO(annotation_file)
        cats = dataset.cats
        self.cates = [cat.get("name", "UNKNOW") for _, cat in cats.items()]
        pbar = tqdm.tqdm(dataset.imgs.items())
        for img_id, image in pbar:
            pbar.set_description("Load COCO with image id {}".format(img_id))
            group = 1
            license = image.get("license", "")
            file_name = image.get("file_name", "")
            coco_url = image.get("license", "")
            height = image.get("height", None)
            width = image.get("width", None)
            date_captured = image.get("date_captured", "")
            flickr_url = image.get("flickr_url", "")
            id = image.get("id", None)

            anno = self.ANNO()
            anno.info = self.ANNO.INFO()
            anno.info.name = file_name
            anno.info.height = height
            anno.info.width = width
            objs = []

            ann_ids = dataset.getAnnIds(img_id)
            for ann_id in ann_ids:
                annos_coco = dataset.loadAnns(ann_id)
                anno_coco = annos_coco[0]

                segmentations = anno_coco.get("segmentation") or []  # 多个polygon
                area = anno_coco.get(
                    "area", None
                )  # coco中，area是组面积，ivas中是单个polygon面积
                # COCO stores iscrowd as 0/1; older IVAS exports used false/true.
                iscrowd = bool(anno_coco.get("iscrowd", False))
                image_id = anno_coco.get("image_id", None)
                bbox = anno_coco.get("bbox", [])
                category_id = anno_coco.get("category_id", None)
                id = anno_coco.get("id", None)
                shape_type = normalize_shape_type(anno_coco.get("ivas_shape_type", ""))

                if iscrowd and not self.keep_crowd:
                    group += 1
                    continue
                category = dataset.loadCats(category_id)[0].get("name", "UNKNOW")

                if isinstance(segmentations, dict) and "counts" in segmentations:
                    # RLE
                    rles = mscoco_mask.frPyObjects(segmentations, height, width)
                    masks = mscoco_mask.decode(rles)
                    contours, _ = cv2.findContours(
                        masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
                    )
                    for contour in contours:
                        points = []
                        for point in contour:
                            x, y = point[0]
                            points.append([float(x), float(y)])
                        objs.append(
                            self._coco_polygon_obj(category, group, points, iscrowd)
                        )

                elif segmentations:
                    # polygon (or an IVAS rectangle/OBB stored as its 4 corners)
                    for segmentation in segmentations:
                        xs = segmentation[::2]
                        ys = segmentation[1::2]
                        points = [[x, y] for x, y in zip(xs, ys)]
                        if shape_type == "rectangle" and points:
                            obj = self.make_rectangle_obj(
                                category, group, *self.segmentation_bounds(points),
                                layer=1, iscrowd=iscrowd,
                            )
                        else:
                            obj = self._coco_polygon_obj(
                                category, group, points, iscrowd
                            )
                            obj.shape_type = shape_type
                        objs.append(obj)

                else:
                    # Detection-only annotation: no segmentation, just [x, y, w, h].
                    box = self._valid_coco_bbox(bbox)
                    if box is not None:
                        x, y, w, h = box
                        objs.append(
                            self.make_rectangle_obj(
                                category, group, x, y, x + w, y + h,
                                layer=1, iscrowd=iscrowd,
                            )
                        )
                group += 1
            anno.objs = tuple(objs)
            self.annos[self.remove_file_suffix(file_name)] = anno
        return True

    def _coco_annotation(
        self, objs, crowd, image_id, image_name, category_index, ann_id, shape_type=None
    ) -> dict:
        r"""
        Build one COCO annotation from ``objs``.

        ``shape_type`` is None for merged polygons; for a single rectangle/OBB it
        is written as ``ivas_shape_type`` and the bbox is taken from its corners.
        """
        coco_anno_info = {}
        coco_anno_info["iscrowd"] = crowd
        coco_anno_info["image_id"] = image_id
        coco_anno_info["image_name"] = image_name
        coco_anno_info["category_id"] = category_index
        coco_anno_info["id"] = ann_id
        coco_anno_info["segmentation"] = []
        coco_anno_info["area"] = 0.0
        coco_anno_info["bbox"] = []

        for obj in objs:

            segmentation = obj.segmentation
            area = obj.area
            bbox = obj.bbox
            if shape_type is not None:
                # Shape bounds are exact; a stored bbox may include item padding.
                bbox = [float(value) for value in self.segmentation_bounds(segmentation)]
                if shape_type == "rectangle":
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            elif bbox is None or len(bbox) != 4:
                segmentation_nd = np.array(segmentation)
                bbox = [
                    min(segmentation_nd[:, 0]),
                    min(segmentation_nd[:, 1]),
                    max(segmentation_nd[:, 0]),
                    max(segmentation_nd[:, 1]),
                ]
                bbox = [float(value) for value in bbox]
                del segmentation_nd
            if not area:
                from shapely.geometry import Polygon
                area = Polygon(segmentation).area
            segmentation = [e for p in segmentation for e in p]

            if bbox != []:
                if coco_anno_info["bbox"] == []:
                    coco_anno_info["bbox"] = bbox
                else:
                    bbox_tmp = coco_anno_info["bbox"]
                    bbox_tmp = [
                        min(bbox_tmp[0], bbox[0]),
                        min(bbox_tmp[1], bbox[1]),
                        max(bbox_tmp[2], bbox[2]),
                        max(bbox_tmp[3], bbox[3]),
                    ]
                    coco_anno_info["bbox"] = bbox_tmp
            coco_anno_info["segmentation"].append(segmentation)
            if area is not None:
                coco_anno_info["area"] += float(area)

        # (xmin, ymin, xmax, ymax) 2 (xmin, ymin, w, h)
        bbox_tmp = coco_anno_info["bbox"]
        if len(bbox_tmp) >= 4:
            coco_anno_info["bbox"] = [
                bbox_tmp[0],
                bbox_tmp[1],
                bbox_tmp[2] - bbox_tmp[0],
                bbox_tmp[3] - bbox_tmp[1],
            ]
        else:
            coco_anno_info["bbox"] = []
        if shape_type is not None:
            coco_anno_info["ivas_shape_type"] = shape_type
        return coco_anno_info

    def save_to_coco(self, annotation_file: str, cates: tuple = ()) -> bool:
        r"""
        Save annotations to a COCO json file.

        Arguments:
            annotation_file (str): coco json annotations file.
            cates (tuple): coco cates.
        """
        coco_anno = {}
        # info
        coco_anno["info"] = {}
        coco_anno["info"]["description"] = "coco from IVAS"
        coco_anno["info"]["version"] = None
        coco_anno["info"]["year"] = None
        coco_anno["info"]["contributor"] = None
        coco_anno["info"]["date_created"] = None

        # licenses
        coco_anno["licenses"] = []
        license1 = {}
        license1["url"] = None
        license1["id"] = 0
        license1["name"] = None
        coco_anno["licenses"].append(license1)

        # images and annotations
        coco_anno["images"] = []
        coco_anno["annotations"] = []
        coco_anno["categories"] = []

        categories_dict = {cat: index for index, cat in enumerate(cates)}
        uncontained_dict = {}
        # Category definitions come from the IVAS or legacy category config.

        pbar = tqdm.tqdm(self.annos.items())
        for file_index, (name_without_suffix, anno) in enumerate(pbar):
            pbar.set_description("Integrate {}".format(name_without_suffix))
            try:
                if not any(
                    anno.info.description.startswith(name)
                    for name in SUPPORTED_ANNOTATION_FORMATS
                ):
                    continue

                # image
                coco_image_info = {}
                coco_image_info["license"] = ""
                coco_image_info["url"] = ""
                coco_image_info["file_name"] = anno.info.name
                coco_image_info["height"] = anno.info.height
                coco_image_info["width"] = anno.info.width
                coco_image_info["date_captured"] = ""
                coco_image_info["id"] = file_index
                coco_anno["images"].append(coco_image_info)

                objects = anno.objs

                objects_groups = [obj.group for obj in objects]
                objects_groups.sort()
                objects_groups = set(objects_groups)
                # 同group
                for group_index, group in enumerate(objects_groups):
                    objs_with_group = [obj for obj in objects if obj.group == group]
                    cats = [obj.category for obj in objs_with_group]
                    cats = set(cats)
                    # 同category
                    for cat in cats:
                        if cat not in categories_dict:
                            categories_dict[cat] = len(categories_dict) + 1
                            uncontained_dict[cat] = len(categories_dict) + 1
                        category_index = categories_dict.get(cat)

                        objs_with_cat = [
                            obj for obj in objs_with_group if obj.category == cat
                        ]
                        crowds = [obj.iscrowd for obj in objs_with_cat]
                        crowds = set(crowds)
                        # 同iscrowd
                        for crowd in crowds:
                            if crowd and not self.keep_crowd:
                                continue
                            objs_with_crowd = [
                                obj for obj in objs_with_cat if obj.iscrowd == crowd
                            ]
                            if not objs_with_crowd:
                                continue
                            # Polygons of one group/category merge into one
                            # annotation; rectangles and OBBs stay separate so
                            # their shape survives the round-trip.
                            polygon_objs = [
                                obj for obj in objs_with_crowd
                                if obj.shape_type == "polygon"
                            ]
                            box_objs = [
                                obj for obj in objs_with_crowd
                                if obj.shape_type != "polygon"
                            ]
                            entries = [(polygon_objs, None)] if polygon_objs else []
                            entries += [([obj], obj.shape_type) for obj in box_objs]
                            for entry_objs, shape_type in entries:
                                coco_anno["annotations"].append(
                                    self._coco_annotation(
                                        entry_objs,
                                        crowd,
                                        file_index,
                                        anno.info.name,
                                        category_index,
                                        len(coco_anno["annotations"]),
                                        shape_type,
                                    )
                                )
            except Exception as e:
                raise ValueError("{}: {}".format(name_without_suffix, e)) from e

        categories_dict = sorted(categories_dict.items(), key=lambda x: x[1])
        coco_anno["categories"] = [
            {"name": name, "id": id, "supercategory": None}
            for name, id in categories_dict
        ]

        from IVAS.utils.persistence import atomic_write_bytes
        from json import dumps
        atomic_write_bytes(annotation_file, dumps(coco_anno, indent=4, ensure_ascii=False).encode("utf-8"))

        return True
