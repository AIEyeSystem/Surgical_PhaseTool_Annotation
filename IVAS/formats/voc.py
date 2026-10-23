# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
import io
from IVAS.utils.persistence import atomic_write_bytes
from typing import Union

import imgviz
import numpy as np
import tqdm
from PIL import Image
from skimage.draw.draw import polygon

from IVAS.formats.ivas import IVAS


class VOC(IVAS):
    """
    VOC format

    Attributes:
        keep_crowd (bool): keep the crowded objects
        is_instance (bool): mark if an object is instance or sematic
    """

    def __init__(self):
        super().__init__()
        self.keep_crowd = True
        self.is_instance = False

    def save_to_voc(self, png_root: str) -> bool:
        """
        Save annotations to the directory of voc png files.

        Arguments:
            png_root (str): the directory to save the voc png files.
        """
        os.makedirs(png_root, exist_ok=True)

        # cmap
        cmap = imgviz.label_colormap()
        category_index_dict = {}
        with open(
            os.path.join(png_root, "classification.txt"), "w", encoding="utf-8"
        ) as f:
            for index, cate in enumerate(self.semantic_categories()):
                category_index_dict[cate] = index
                f.write("{}\n".format(cate))

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            pbar.set_description("Save to {}".format(name_without_suffix + ".png"))
            png_path = os.path.join(png_root, name_without_suffix + ".png")
            self.save_one_voc_png(anno, png_path, cmap, category_index_dict)
        return True

    def semantic_categories(self):
        categories = ("__background__",) + tuple(dict.fromkeys(cat for cat in self.cates if cat != "__background__"))
        if len(categories) > 256:
            raise ValueError("VOC PNG supports at most 255 foreground categories")
        return categories

    def save_one_voc_png(
        self, anno: IVAS.ANNO, png_path: str, cmap: np.ndarray, category_index_dict=None
    ) -> bool:
        """
        Save annotation to a VOC png file.

        Arguments:
            anno (IVAS.ANNO): the annotation.
            png_path (str): the path of the png file.
            cmap (np.ndarray): color map. shape [n, 3]
            category_index_dict (dict): the category index dict. {index: category}.
        """
        info = anno.info
        objects = anno.objs

        img_name = info.name
        width = info.width
        height = info.height
        depth = info.depth
        note = info.note
        img = np.zeros(shape=(height, width), dtype=np.uint8)

        objects = sorted(objects, key=lambda obj: obj.layer)
        groups = list(dict.fromkeys(obj.group for obj in objects if obj.category != "__background__"))
        instance_indices = {group: index + 1 for index, group in enumerate(groups)}
        if self.is_instance and len(instance_indices) > 255:
            raise ValueError("VOC PNG supports at most 255 instances")

        for obj in objects:
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
            segmentation = [(int(p[1]), int(p[0])) for p in segmentation]

            if self.is_instance:
                group = 0 if category == "__background__" else instance_indices[group]
                assert (
                    0 <= group < 256
                ), "When use VOC for segmentation, the group must in [0, 255], but get group={}".format(
                    group
                )
                self.fill_polygon(segmentation, img, color=group)
            else:
                if category not in (category_index_dict or {}):
                    raise ValueError("Unknown VOC category: {}".format(category))
                index = category_index_dict[category]
                assert (
                    0 <= index < 256
                ), "When use VOC for segmentation, the number of classifications must in [0, 255], but get {}".format(
                    index
                )
                self.fill_polygon(segmentation, img, color=index)

        img = Image.fromarray(img.astype(np.uint8), mode="P")

        img.putpalette(cmap.flatten())
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        atomic_write_bytes(png_path, buffer.getvalue())
        return True

    @staticmethod
    def fill_polygon(segmentation: Union[list, tuple], img: np.ndarray, color: int):
        """
        fill polygon with color on image.

        Arguments:
            segmentation (Union[list, tuple]): the vertices of the polygon. [(x1, y1), (x2, y2), ...] .
            img (np.ndarray): the image.
            color (int): the color of the polygon. save image as mode 'P' with PIL.
        """
        xs = [x for x, y in segmentation]
        ys = [y for x, y in segmentation]
        rr, cc = polygon(xs, ys, img.shape)
        img[rr, cc] = color
