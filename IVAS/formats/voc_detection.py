# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import math
import os
from IVAS.utils.persistence import atomic_write_bytes
from xml.etree import ElementTree as ET

import tqdm

from IVAS.branding import ANNOTATION_FORMAT
from IVAS.formats.ivas import IVAS


class VOCDetect(IVAS):
    """
    VOC format for object detection

    Export merges every (group, category) into one ``<object>`` with the union
    bndbox. Import creates one rectangle object (and group) per ``<object>``.

    Attributes:
        keep_crowd (bool): keep the crowded objects
    """

    def __init__(self):
        super().__init__()
        self.keep_crowd = True

    def read_from_XML(self, xml_root: str) -> bool:
        """
        Load annotations from the directory of voc xml files.

        Arguments:
            xml_root (str): the directory of voc xml files
        """
        self.annos.clear()
        self.cates = ()

        xml_files = [file for file in os.listdir(xml_root) if file.lower().endswith(".xml")]
        pbar = tqdm.tqdm(xml_files)
        for file in pbar:
            pbar.set_description("Load voc xml {}".format(file))
            anno = self.load_one_voc_xml(os.path.join(xml_root, file))
            self.annos[self.remove_file_suffix(file)] = anno

        class_set = set()
        for _, anno in self.annos.items():
            for obj in anno.objs:
                class_set.add(obj.category)
        self.cates = tuple(sorted(class_set))
        return True

    @staticmethod
    def _xml_number(element, tag: str, required: bool = False):
        r"""Read a finite number from ``element/tag``; tolerate "12" and "12.5"."""
        text = element.findtext(tag) if element is not None else None
        if text is None or not text.strip():
            if required:
                raise ValueError("VOC xml is missing <{}>".format(tag))
            return None
        try:
            value = float(text.strip())
        except ValueError:
            raise ValueError("VOC xml <{}> is not a number: {!r}".format(tag, text))
        if not math.isfinite(value):
            raise ValueError("VOC xml <{}> must be finite".format(tag))
        return value

    def load_one_voc_xml(self, xml_path: str) -> IVAS.ANNO:
        """
        Load annotation from one voc xml file. Every ``<object>`` with a
        ``<bndbox>`` becomes a rectangle object in its own group.

        Arguments:
            xml_path (str): the xml file path

        Returns:
            IVAS.ANNO: the annotation.
        """
        root = ET.parse(xml_path).getroot()
        if root.tag != "annotation":
            raise ValueError("The file {} is not a VOC annotation xml".format(xml_path))

        anno = self.ANNO()
        anno.info = self.ANNO.INFO()
        anno.info.description = ANNOTATION_FORMAT
        anno.info.folder = (root.findtext("folder") or "").strip() or os.path.dirname(xml_path)
        anno.info.name = (root.findtext("filename") or "").strip() or os.path.splitext(
            os.path.basename(xml_path))[0]
        size = root.find("size")
        for name in ("width", "height", "depth"):
            value = self._xml_number(size, name)
            setattr(anno.info, name, None if value is None else int(value))
        anno.info.note = ""

        objs = []
        group = 1
        for element in root.findall("object"):
            category = (element.findtext("name") or "").strip() or "unknown"
            bndbox = element.find("bndbox")
            if bndbox is None:
                continue
            xmin, ymin, xmax, ymax = (
                self._xml_number(bndbox, tag, required=True)
                for tag in ("xmin", "ymin", "xmax", "ymax")
            )
            objs.append(self.make_rectangle_obj(category, group, xmin, ymin, xmax, ymax))
            group += 1
        anno.objs = tuple(objs)
        return anno

    def save_to_XML(self, xml_root: str) -> bool:
        """
        Save annotations to the directory of voc xml files.

        Arguments:
            xml_root (str): the directory to save the voc xml files
        """
        os.makedirs(xml_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            xml_path = os.path.join(xml_root, name_without_suffix + ".xml")
            try:
                self.save_one_voc_xml(anno, xml_path)
                pbar.set_description(
                    "Save xml to {}".format(name_without_suffix + ".xml")
                )
            except Exception as e:
                raise ValueError("{}: {}".format(name_without_suffix, e)) from e
        return True

    def save_one_voc_xml(self, anno: IVAS.ANNO, xml_path: str) -> bool:
        """
        Save annotation to one voc xml file.

        Arguments:
            anno (IVAS.ANNO): the annotation
            xml_path (str): the xml file path
        """
        annotation = ET.Element("annotation")
        tree = ET.ElementTree(annotation)
        folder = ET.Element("folder")
        folder.text = "{}".format(anno.info.folder)
        annotation.append(folder)

        filename = ET.Element("filename")
        filename.text = "{}".format(anno.info.name)
        annotation.append(filename)

        explain = ET.Element("explain")
        explain.text = "{}".format("XML from IVAS")
        annotation.append(explain)

        size = ET.Element("size")
        width = ET.Element("width")
        width.text = "{}".format(anno.info.width)
        size.append(width)

        height = ET.Element("height")
        height.text = "{}".format(anno.info.height)
        size.append(height)

        depth = ET.Element("depth")
        depth.text = "{}".format(anno.info.depth)
        size.append(depth)
        annotation.append(size)

        objects = anno.objs
        objects_groups = [obj.group for obj in objects]
        objects_groups.sort()
        objects_groups = set(objects_groups)

        # 同group为同一个目标，需合并包围盒
        for group_index, group in enumerate(objects_groups):
            objs_with_group = [obj for obj in objects if obj.group == group]
            cats = [obj.category for obj in objs_with_group]
            cats = set(cats)

            # 同category
            for cat in cats:
                objs_with_cat = [obj for obj in objs_with_group if obj.category == cat]
                crowds = [obj.iscrowd for obj in objs_with_group]
                crowds = set(crowds)

                segmentations = []

                for obj in objs_with_cat:
                    if not self.keep_crowd and obj.iscrowd:
                        continue

                    segmentations.extend(obj.segmentation)

                xs = [x for x, y in segmentations]
                ys = [y for x, y in segmentations]
                if not xs or not ys:
                    # Nothing left after keep_crowd filtering: emit no <object>.
                    continue

                object = ET.Element("object")
                name = ET.Element("name")
                name.text = "{}".format(cat)
                object.append(name)

                pose = ET.Element("pose")
                pose.text = "Unspecified"
                object.append(pose)

                truncated = ET.Element("truncated")
                truncated.text = "0"
                object.append(truncated)

                difficult = ET.Element("difficult")
                difficult.text = "0"
                object.append(difficult)

                bndbox = ET.Element("bndbox")
                xmin = ET.Element("xmin")
                xmin.text = "{}".format(int(min(xs)))
                bndbox.append(xmin)

                ymin = ET.Element("ymin")
                ymin.text = "{}".format(int(min(ys)))
                bndbox.append(ymin)

                xmax = ET.Element("xmax")
                xmax.text = "{}".format(int(max(xs)))
                bndbox.append(xmax)

                ymax = ET.Element("ymax")
                ymax.text = "{}".format(int(max(ys)))
                bndbox.append(ymax)
                object.append(bndbox)
                annotation.append(object)
        atomic_write_bytes(xml_path, ET.tostring(annotation, encoding="utf-8", xml_declaration=True))
        return True
