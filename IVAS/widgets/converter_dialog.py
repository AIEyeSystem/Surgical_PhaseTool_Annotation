# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os

import imgviz
import numpy as np
import yaml
from IVAS.configs import save_config
from PIL import ImageColor
from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from PyQt5.QtCore import QThread, pyqtSignal

from IVAS.branding import CATEGORY_CONFIG_NAME, LEGACY_CATEGORY_CONFIG_NAMES
from IVAS.formats.coco import COCO
from IVAS.formats.ivas import IVAS
from IVAS.formats.labelme import LABELME
from IVAS.formats.voc import VOC
from IVAS.formats.voc_detection import VOCDetect
from IVAS.formats.yolo import YOLO
from IVAS.ui.Converter_dialog import Ui_Dialog


def _find_category_config(root):
    for name in (CATEGORY_CONFIG_NAME, *LEGACY_CATEGORY_CONFIG_NAMES):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    return None


class Converter(QThread, IVAS):
    message = pyqtSignal(int, int, str)

    def __init__(self):
        super(Converter, self).__init__()
        IVAS.__init__(self)
        self.cancel = False
        self.ivas_json_root = None
        self.errors = []

    def run(self):
        self.errors = []
        try:
            self.convert()
        except Exception as error:
            self.report_error(error)
        finally:
            state = "Cancelled" if self.cancel else "Completed"
            self.message.emit(-1, -1, "{}; {} error(s).".format(state, len(self.errors)))

    def convert(self):
        raise NotImplementedError

    def report_error(self, error):
        self.errors.append(str(error))
        self.message.emit(-1, -1, "Error: {}".format(error))

    def load_from_ivas(self):
        self.annos.clear()
        self.cates = ()
        json_files = [
            file for file in os.listdir(self.ivas_json_root) if file.endswith(".json")
        ]
        num_json_files = len(json_files)
        for index, file in enumerate(json_files):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            self.message.emit(
                index + 1,
                num_json_files,
                "{:>8d}/{:<8d} | Loading IVAS json {}".format(
                    index + 1, num_json_files, file
                ),
            )
            try:
                anno = self.load_one_ivas_json(os.path.join(self.ivas_json_root, file))
                self.annos[self.remove_file_suffix(file)] = anno
            except Exception as e:
                self.report_error(e)

        # load cates
        self.message.emit(-1, -1, " " * 18 + "| Loading cats.")
        category_config = _find_category_config(self.ivas_json_root)
        if category_config is not None:
            cates = []
            with open(category_config, "rb") as f:
                cfg = yaml.safe_load(f.read()) or {}
            for label in cfg.get("label", []):
                cates.append(label.get("name"))
            self.cates = tuple(cates)
        else:
            cates = set()
            for _, anno in self.annos.items():
                for obj in anno.objs:
                    cates.add(obj.category)
            cates = list(cates)
            cates.sort()
            self.cates = tuple(cates)
        self.message.emit(-1, -1, " " * 18 + "| Loaded.")
        return True

    def convert_to_ivas(self):
        self.message.emit(-1, -1, " " * 18 + "| Start convert to IVAS.")
        num_annos = len(self.annos)

        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            self.message.emit(
                index + 1,
                num_annos,
                "{:>8d}/{:<8d} | Converting to {}".format(
                    index + 1, num_annos, name_without_suffix + "json"
                ),
            )
            try:
                self.save_one_ivas_json(
                    anno,
                    os.path.join(
                        self.ivas_json_root, "{}.json".format(name_without_suffix)
                    ),
                )

            except Exception as e:
                self.report_error(e)

        # 类别文件
        cmap = imgviz.label_colormap()
        cates = sorted(self.cates)
        categories = []
        for index, cat in enumerate(cates):
            r, g, b = cmap[(index + 1) % len(cmap)]
            categories.append(
                {
                    "name": cat if isinstance(cat, str) else str(cat),
                    "color": "#{:02x}{:02x}{:02x}".format(r, g, b),
                }
            )
        save_config({"label": categories}, os.path.join(self.ivas_json_root, CATEGORY_CONFIG_NAME))

        return True


class COCOConverter(Converter, COCO):
    def __init__(self):
        super(COCOConverter, self).__init__()
        self.coco_json_path = None
        self.ivas_json_root = None
        self.keep_crowd = False
        self.coco2ivas = True

    def convert(self):
        if self.coco_json_path is None or self.ivas_json_root is None:
            return

        if self.coco2ivas:
            self.coco_to_ivas()
        else:
            self.ivas_to_coco()

    def coco_to_ivas(self):
        self.message.emit(-1, -1, " " * 18 + "| -- COCO to IVAS --")

        # load from coco
        self.message.emit(-1, -1, " " * 18 + "| Loading from COCO.")
        try:
            self.read_from_coco(self.coco_json_path)
            self.message.emit(-1, -1, " " * 18 + "| Loaded.")

        except Exception as e:
            self.report_error(e)
            return

        # convert to ivas
        self.convert_to_ivas()
        return True

    def ivas_to_coco(self):
        self.message.emit(-1, -1, " " * 18 + "| -- IVAS to COCO --")

        # Load from IVAS-compatible annotation data.
        self.load_from_ivas()

        # convert to coco
        self.message.emit(
            -1, -1, " " * 18 + "| Saving COCO json to {}.".format(self.coco_json_path)
        )
        try:
            self.save_to_coco(self.coco_json_path, self.cates)
        except Exception as e:
            self.report_error(e)
        return True


class YOLOConverter(Converter, YOLO):
    def __init__(self):
        super(YOLOConverter, self).__init__()
        self.yolo_images_root = None
        self.yolo_txt_root = None
        self.yolo_category_file = None
        self.ivas_json_root = None

        self.yolo2ivas = True

    def convert(self):
        if self.yolo2ivas:
            if (
                self.yolo_images_root is None
                or self.yolo_txt_root is None
                or self.ivas_json_root is None
            ):
                return
            self.yolo_to_ivas()
        else:
            if self.yolo_txt_root is None or self.ivas_json_root is None:
                return
            self.ivas_to_yolo()

    def yolo_to_ivas(self):
        self.annos.clear()
        self.cates = ()
        self.message.emit(-1, -1, " " * 18 + "| -- YOLO to IVAS --")

        # load category file
        if self.yolo_category_file is not None:
            self.message.emit(-1, -1, " " * 18 + "| Loading YOLO categorys.")
            class_dict = {}
            with open(self.yolo_category_file, "r") as f:
                lines = f.readlines()
                for index, line in enumerate(lines):
                    class_dict[index] = line.rstrip("\n")
                    self.message.emit(
                        -1, -1, " " * 18 + "| {} : {}".format(index, line.rstrip("\n"))
                    )
            self.message.emit(-1, -1, " " * 18 + "| Loaded.")
        else:
            class_dict = None

        # load from yolo
        self.message.emit(-1, -1, " " * 18 + "| Loading YOLO txts.")
        img_files = os.listdir(self.yolo_images_root)
        num_img_files = len(img_files)

        for index, img_name in enumerate(img_files):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return
            name_without_suffix = self.remove_file_suffix(img_name)
            txt_path = os.path.join(self.yolo_txt_root, name_without_suffix + ".txt")
            image_path = os.path.join(self.yolo_images_root, img_name)
            if not os.path.exists(txt_path):
                continue

            self.message.emit(
                index + 1,
                num_img_files,
                "{:>8d}/{:<8d} | Loading from {}".format(
                    index + 1, num_img_files, name_without_suffix + ".txt"
                ),
            )
            try:
                anno = self.load_one_yolo_txt(image_path, txt_path, class_dict)
                self.annos[name_without_suffix] = anno
            except Exception as e:
                self.report_error(e)

        # cates
        if class_dict is not None:
            self.cates = tuple(class_dict.values())
        else:
            class_set = set()
            for _, anno in self.annos.items():
                for obj in anno.objs:
                    class_set.add(obj.category)
            class_set = list(class_set)
            class_set.sort()
            self.cates = tuple(class_set)

        # convert to ivas
        self.convert_to_ivas()

        return True

    def ivas_to_yolo(self):
        self.message.emit(-1, -1, " " * 18 + "| -- IVAS to YOLO --")

        # Load from IVAS-compatible annotation data.
        self.load_from_ivas()

        # save to yolo
        cates_index_dict = {}
        index = 0
        for cat in self.cates:
            if self.ignore_background and cat == "__background__":
                continue
            cates_index_dict[cat] = index
            index += 1

        with open(os.path.join(self.yolo_txt_root, "classification.txt"), "w") as f:
            for cat in self.cates:
                if self.ignore_background and cat == "__background__":
                    continue
                f.write("{}\n".format(cat))

        num_annos = len(self.annos)
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            txt_path = os.path.join(self.yolo_txt_root, name_without_suffix + ".txt")
            try:
                self.message.emit(
                    index + 1,
                    num_annos,
                    "{:>8d}/{:<8d} | Save yolo txt to {}".format(
                        index + 1, num_annos, name_without_suffix + ".txt"
                    ),
                )
                self.save_one_yolo_txt(anno, txt_path, cates_index_dict)
            except Exception as e:
                self.report_error(e)
        return True


class LABELMEConverter(Converter, LABELME):
    def __init__(self):
        super(LABELMEConverter, self).__init__()
        self.ivas_json_root = None
        self.labelme_json_root = None
        self.labelme2ivas = True

    def convert(self):
        if self.labelme_json_root is None or self.ivas_json_root is None:
            return
        if self.labelme2ivas:
            self.labelme_to_ivas()
        else:
            self.ivas_to_labelme()

    def labelme_to_ivas(self):
        self.annos.clear()
        self.cates = ()
        self.message.emit(-1, -1, " " * 18 + "| -- IVAS to IVAS --")

        # load from labelme
        self.message.emit(-1, -1, " " * 18 + "| Loading from LABELME.")
        json_files = [
            file
            for file in os.listdir(self.labelme_json_root)
            if file.endswith(".json")
        ]
        num_json_files = len(json_files)
        for index, file in enumerate(json_files):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            name_without_suffix = self.remove_file_suffix(file)

            self.message.emit(
                index + 1,
                num_json_files,
                "{:>8d}/{:<8d} | Loading from {}".format(
                    index + 1, num_json_files, name_without_suffix + ".json"
                ),
            )
            try:
                anno = self.load_one_labelme_json(
                    os.path.join(self.labelme_json_root, file)
                )
                self.annos[name_without_suffix] = anno
            except Exception as e:
                self.report_error(e)

        # cats
        class_set = set()
        for _, anno in self.annos.items():
            for obj in anno.objs:
                class_set.add(obj.category)

        class_set = list(class_set)
        class_set.sort()
        self.cates = tuple(class_set)

        # save to ivas
        self.convert_to_ivas()

        return True

    def ivas_to_labelme(self):

        # Load from IVAS-compatible annotation data.
        self.load_from_ivas()

        # save to labelme
        num_annos = len(self.annos.items())
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            json_path = os.path.join(
                self.labelme_json_root, name_without_suffix + ".json"
            )
            self.message.emit(
                index + 1,
                num_annos,
                "{:>8d}/{:<8d} | Save LABELME json to {}".format(
                    index + 1, num_annos, name_without_suffix + ".json"
                ),
            )
            try:
                self.save_one_labelme_json(anno, json_path)
            except Exception as e:
                self.report_error(e)

        return True


class VOCConverter(Converter, VOC):
    def __init__(self):
        super(VOCConverter, self).__init__()
        self.ivas_json_root = None
        self.voc_png_root = None
        self.use_setting_color = False

    def convert(self):
        if self.ivas_json_root is not None and self.voc_png_root is not None:
            self.ivas_to_voc()

    def ivas_to_voc(self):
        os.makedirs(self.voc_png_root, exist_ok=True)
        self.message.emit(-1, -1, " " * 18 + "| -- IVAS to VOC --")

        # Load from IVAS-compatible annotation data.
        self.load_from_ivas()

        # default cmap
        cmap = imgviz.label_colormap()

        # segment cmap when use setting color
        category_config = _find_category_config(self.ivas_json_root)
        if not self.is_instance and self.use_setting_color:
            if category_config is not None:
                with open(category_config, "rb") as f:
                    cfg = yaml.safe_load(f.read()) or {}

                labels = cfg.get("label", [])
                indices = {category: index for index, category in enumerate(self.semantic_categories())}
                for label_dict in labels:
                    color = label_dict.get("color", "#000000")
                    if label_dict.get("name") in indices:
                        cmap[indices[label_dict["name"]]] = ImageColor.getrgb(color)
            else:
                self.message.emit(
                    -1,
                    -1,
                    " " * 18
                    + "| Warning: {}.".format(
                        "No IVAS category config was found; default colors will be used."
                    ),
                )

        if not self.is_instance:
            category_index_dict = {}
            with open(os.path.join(self.voc_png_root, "classification.txt"), "w") as f:
                for index, cate in enumerate(self.semantic_categories()):
                    category_index_dict[cate] = index
                    f.write("{}\n".format(cate))
        else:
            category_index_dict = None
        # save to voc
        num_annos = len(self.annos)
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            png_path = os.path.join(self.voc_png_root, name_without_suffix + ".png")
            try:
                self.message.emit(
                    index + 1,
                    num_annos,
                    "{:>8d}/{:<8d} | Save voc png to {}".format(
                        index + 1, num_annos, name_without_suffix + ".png"
                    ),
                )
                self.save_one_voc_png(anno, png_path, cmap, category_index_dict)
            except Exception as e:
                self.report_error(e)

        return True


class VOCConverterForDetection(Converter, VOCDetect):
    def __init__(self):
        super(VOCConverterForDetection, self).__init__()
        self.ivas_json_root = None
        self.voc_xml_root = None

    def convert(self):
        if self.ivas_json_root is not None and self.voc_xml_root is not None:
            self.ivas_to_voc_for_detection()

    def ivas_to_voc_for_detection(self):
        self.message.emit(-1, -1, " " * 18 + "| -- IVAS to VOC for Detection --")

        # Load from IVAS-compatible annotation data.
        self.load_from_ivas()

        # save to xml
        num_annos = len(self.annos)
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, " " * 18 + "| -- Cancel --")
                return

            xml_path = os.path.join(self.voc_xml_root, name_without_suffix + ".xml")
            try:
                self.message.emit(
                    index + 1,
                    num_annos,
                    "{:>8d}/{:<8d} | Save voc png to {}".format(
                        index + 1, num_annos, name_without_suffix + ".png"
                    ),
                )
                self.save_one_voc_xml(anno, xml_path)

            except Exception as e:
                self.report_error(e)
        return True


class ConverterDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ConverterDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.converter = None
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.tabWidget.currentChanged.connect(self.tab_widget_current_changed)
        self.pushButton_close.clicked.connect(self.close)

        self.init_connect()

    def tab_widget_current_changed(self):
        if self.tabWidget.currentWidget() == self.tab_COCO:
            self.mainwindow.cfg["software"]["current_converter"] = "coco"
        elif self.tabWidget.currentWidget() == self.tab_YOLO:
            self.mainwindow.cfg["software"]["current_converter"] = "yolo"
        elif self.tabWidget.currentWidget() == self.tab_LABELME:
            self.mainwindow.cfg["software"]["current_converter"] = "labelme"
        elif self.tabWidget.currentWidget() == self.tab_VOC:
            self.mainwindow.cfg["software"]["current_converter"] = "voc"
        elif self.tabWidget.currentWidget() == self.tab_VOC_DETECTION:
            self.mainwindow.cfg["software"]["current_converter"] = "voc for detection"
        else:
            self.mainwindow.cfg["software"]["current_converter"] = "coco"

    def apply(self):
        if self.converter is not None and self.converter.isRunning():
            return
        self.tabWidget.setEnabled(False)
        self.pushButton_start.setEnabled(False)
        self.progressBar.reset()
        self.textBrowser.clear()

        if self.tabWidget.currentWidget() == self.tab_COCO:
            # COCO
            if self.toolBox_coco.currentWidget() == self.toolbox_item_coco2ivas:
                # coco2ivas
                if (
                    self.lineEdit_coco2ivas_coco_json_path.text()
                    and self.lineEdit_coco2ivas_ivas_json_root.text()
                ):
                    self.converter = COCOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.coco2ivas = True
                    self.converter.coco_json_path = (
                        self.lineEdit_coco2ivas_coco_json_path.text()
                    )
                    self.converter.ivas_json_root = (
                        self.lineEdit_coco2ivas_ivas_json_root.text()
                    )
                    self.converter.keep_crowd = (
                        self.checkBox_coco2ivas_keep_crowd.isChecked()
                    )

                    self._start_converter()
                else:
                    QtWidgets.QMessageBox.warning(self, "", "")

            elif self.toolBox_coco.currentWidget() == self.toolbox_item_ivas2coco:
                # ivas2coco
                if (
                    self.lineEdit_ivas2coco_ivas_json_root.text()
                    and self.lineEdit_ivas2coco_coco_json_path.text()
                ):
                    self.converter = COCOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.coco2ivas = False
                    self.converter.ivas_json_root = (
                        self.lineEdit_ivas2coco_ivas_json_root.text()
                    )
                    self.converter.coco_json_path = (
                        self.lineEdit_ivas2coco_coco_json_path.text()
                    )
                    self.converter.keep_crowd = (
                        self.checkBox_ivas2coco_keep_crowd.isChecked()
                    )

                    self._start_converter()
                else:
                    QtWidgets.QMessageBox.warning(self, "", "")

        elif self.tabWidget.currentWidget() == self.tab_YOLO:
            # YOLO
            if self.toolBox_yolo.currentWidget() == self.toolbox_item_yolo2ivas:
                # yolo2ivas
                if (
                    self.lineEdit_yolo2ivas_yolo_image_root.text()
                    and self.lineEdit_yolo2ivas_yolo_txt_root.text()
                    and self.lineEdit_yolo2ivas_ivas_json_root.text()
                ):
                    self.converter = YOLOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.yolo2ivas = True
                    self.converter.yolo_images_root = (
                        self.lineEdit_yolo2ivas_yolo_image_root.text()
                    )
                    self.converter.yolo_txt_root = (
                        self.lineEdit_yolo2ivas_yolo_txt_root.text()
                    )
                    self.converter.ivas_json_root = (
                        self.lineEdit_yolo2ivas_ivas_json_root.text()
                    )
                    if self.lineEdit_yolo2ivas_yolo_cate_path.text():
                        self.converter.yolo_category_file = (
                            self.lineEdit_yolo2ivas_yolo_cate_path.text()
                        )

                    self._start_converter()
                else:
                    QtWidgets.QMessageBox.warning(self, "", "")

            elif self.toolBox_yolo.currentWidget() == self.toolbox_item_ivas2yolo:
                # ivas2yolo
                if (
                    self.lineEdit_ivas2yolo_ivas_json_root.text()
                    and self.lineEdit_ivas2yolo_yolo_txt_root.text()
                ):
                    self.converter = YOLOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.yolo2ivas = False
                    self.converter.ignore_background = self.checkBox_ignore_background.isChecked()
                    self.converter.ivas_json_root = (
                        self.lineEdit_ivas2yolo_ivas_json_root.text()
                    )
                    self.converter.yolo_txt_root = (
                        self.lineEdit_ivas2yolo_yolo_txt_root.text()
                    )

                    self._start_converter()
                else:
                    QtWidgets.QMessageBox.warning(self, "", "")

        elif self.tabWidget.currentWidget() == self.tab_LABELME:
            # LABELME
            if self.toolBox_labelme.currentWidget() == self.toolbox_item_labelme2ivas:
                # labelme2ivas
                if (
                    self.lineEdit_labelme2ivas_labelme_json_root.text()
                    and self.lineEdit_labelme2ivas_ivas_json_root.text()
                ):
                    self.converter = LABELMEConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.labelme2ivas = True
                    self.converter.labelme_json_root = (
                        self.lineEdit_labelme2ivas_labelme_json_root.text()
                    )
                    self.converter.ivas_json_root = (
                        self.lineEdit_labelme2ivas_ivas_json_root.text()
                    )

                    self._start_converter()
                else:
                    QtWidgets.QMessageBox.warning(self, "", "")

            elif self.toolBox_labelme.currentWidget() == self.toolbox_item_ivas2labelme:
                # ivas2labelme
                if (
                    self.lineEdit_ivas2labelme_ivas_json_root.text()
                    and self.lineEdit_ivas2labelme_labelme_json_root.text()
                ):
                    self.converter = LABELMEConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.labelme2ivas = False
                    self.converter.ivas_json_root = (
                        self.lineEdit_ivas2labelme_ivas_json_root.text()
                    )
                    self.converter.labelme_json_root = (
                        self.lineEdit_ivas2labelme_labelme_json_root.text()
                    )

                    self._start_converter()
                else:
                    QtWidgets.QMessageBox.warning(self, "", "")

        elif self.tabWidget.currentWidget() == self.tab_VOC:
            # VOC
            if (
                self.lineEdit_ivas2voc_ivas_json_root.text()
                and self.lineEdit_ivas2voc_voc_png_root.text()
            ):
                self.converter = VOCConverter()
                self.converter.message.connect(self.print_message)
                self.converter.ivas_json_root = (
                    self.lineEdit_ivas2voc_ivas_json_root.text()
                )
                self.converter.voc_png_root = self.lineEdit_ivas2voc_voc_png_root.text()
                self.converter.is_instance = self.checkBox_is_instance.isChecked()
                self.converter.use_setting_color = (
                    self.checkBox_use_setting_color.isChecked()
                )
                self.checkBox_use_setting_color.setEnabled(
                    not self.checkBox_is_instance.isChecked()
                )
                self._start_converter()
            else:
                QtWidgets.QMessageBox.warning(self, "", "")

        elif self.tabWidget.currentWidget() == self.tab_VOC_DETECTION:
            # VOC_DETECTION
            if (
                self.lineEdit_ivas2vocod_ivas_json_root.text()
                and self.lineEdit_ivas2vocod_voc_xml_root.text()
            ):
                self.converter = VOCConverterForDetection()
                self.converter.message.connect(self.print_message)
                self.converter.ivas_json_root = (
                    self.lineEdit_ivas2vocod_ivas_json_root.text()
                )
                self.converter.voc_xml_root = (
                    self.lineEdit_ivas2vocod_voc_xml_root.text()
                )
                self._start_converter()
            else:
                QtWidgets.QMessageBox.warning(self, "", "")

        else:
            pass
        if self.converter is None or not self.converter.isRunning():
            self._converter_finished()

    def reject(self):
        self.close()

    def _start_converter(self):
        self._close_when_finished = False
        self.converter.finished.connect(self._converter_finished)
        self.converter.start()

    def _converter_finished(self):
        self.tabWidget.setEnabled(True)
        self.pushButton_start.setEnabled(True)
        if getattr(self, "_close_when_finished", False):
            self.close()

    def closeEvent(self, event):
        if self.converter is not None and self.converter.isRunning():
            self._close_when_finished = True
            self.converter.cancel = True
            event.ignore()
            return
        super().closeEvent(event)

    def cancel(self):
        try:
            self.converter.cancel = True
        except:
            pass

    def open_file(self):

        # coco2ivas
        if self.sender() == self.pushButton_coco2ivas_coco_json_path:
            filter = "json (*.json)"
            lineEdit = self.lineEdit_coco2ivas_coco_json_path
        # ivas2coco
        elif self.sender() == self.pushButton_ivas2coco_coco_json_path:
            filter = "json (*.json)"
            lineEdit = self.lineEdit_ivas2coco_coco_json_path
        elif self.sender() == self.pushButton_yolo2ivas_yolo_cate_path:
            filter = "txt (*.txt)"
            lineEdit = self.lineEdit_yolo2ivas_yolo_cate_path
        else:
            filter = ""
            lineEdit = None

        path, suffix = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open file", filter=filter
        )

        if path:
            if lineEdit is not None:
                lineEdit.setText(path)
        else:
            if lineEdit is not None:
                lineEdit.setText(path)

    def save_file(self):
        path, suffix = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Save file", filter="json (*.json)"
        )
        # coco2ivas
        if self.sender() == self.pushButton_coco2ivas_coco_json_path:
            lineEdit = self.lineEdit_coco2ivas_coco_json_path
        # ivas2coco
        elif self.sender() == self.pushButton_ivas2coco_coco_json_path:
            lineEdit = self.lineEdit_ivas2coco_coco_json_path
        else:
            lineEdit = None

        if path:
            if not path.endswith(".json"):
                path += ".json"
            if lineEdit is not None:
                lineEdit.setText(path)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open dir")
        # coco2ivas
        if self.sender() == self.pushButton_coco2ivas_ivas_json_root:
            lineEdit = self.lineEdit_coco2ivas_ivas_json_root
        # ivas2coco
        elif self.sender() == self.pushButton_ivas2coco_ivas_json_root:
            lineEdit = self.lineEdit_ivas2coco_ivas_json_root
        # yolo2ivas
        elif self.sender() == self.pushButton_yolo2ivas_yolo_image_root:
            lineEdit = self.lineEdit_yolo2ivas_yolo_image_root
        elif self.sender() == self.pushButton_yolo2ivas_yolo_txt_root:
            lineEdit = self.lineEdit_yolo2ivas_yolo_txt_root
        elif self.sender() == self.pushButton_yolo2ivas_ivas_json_root:
            lineEdit = self.lineEdit_yolo2ivas_ivas_json_root
        # ivas2yolo
        elif self.sender() == self.pushButton_ivas2yolo_ivas_json_root:
            lineEdit = self.lineEdit_ivas2yolo_ivas_json_root
        elif self.sender() == self.pushButton_ivas2yolo_yolo_txt_root:
            lineEdit = self.lineEdit_ivas2yolo_yolo_txt_root
        # labelme2ivas
        elif self.sender() == self.pushButton_labelme2ivas_labelme_json_root:
            lineEdit = self.lineEdit_labelme2ivas_labelme_json_root
        elif self.sender() == self.pushButton_labelme2ivas_ivas_json_root:
            lineEdit = self.lineEdit_labelme2ivas_ivas_json_root
        # ivas2labelme
        elif self.sender() == self.pushButton_ivas2labelme_ivas_json_root:
            lineEdit = self.lineEdit_ivas2labelme_ivas_json_root
        elif self.sender() == self.pushButton_ivas2labelme_labelme_json_root:
            lineEdit = self.lineEdit_ivas2labelme_labelme_json_root
        # ivas2voc
        elif self.sender() == self.pushButton_ivas2voc_ivas_json_root:
            lineEdit = self.lineEdit_ivas2voc_ivas_json_root
        elif self.sender() == self.pushButton_ivas2voc_voc_png_root:
            lineEdit = self.lineEdit_ivas2voc_voc_png_root
        # ivas2vocod
        elif self.sender() == self.pushButton_ivas2vocod_ivas_json_root:
            lineEdit = self.lineEdit_ivas2vocod_ivas_json_root
        elif self.sender() == self.pushButton_ivas2vocod_voc_xml_root:
            lineEdit = self.lineEdit_ivas2vocod_voc_xml_root
        else:
            lineEdit = None

        if dir:
            if lineEdit is not None:
                lineEdit.setText(dir)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def save_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Save dir")
        # coco2ivas
        if self.sender() == self.pushButton_coco2ivas_ivas_json_root:
            lineEdit = self.lineEdit_coco2ivas_ivas_json_root
        # ivas2coco
        elif self.sender() == self.pushButton_ivas2coco_ivas_json_root:
            lineEdit = self.lineEdit_ivas2coco_ivas_json_root
        # yolo2ivas
        elif self.sender() == self.pushButton_yolo2ivas_yolo_image_root:
            lineEdit = self.lineEdit_yolo2ivas_yolo_image_root
        elif self.sender() == self.pushButton_yolo2ivas_yolo_txt_root:
            lineEdit = self.lineEdit_yolo2ivas_yolo_txt_root
        elif self.sender() == self.pushButton_yolo2ivas_ivas_json_root:
            lineEdit = self.lineEdit_yolo2ivas_ivas_json_root
        # ivas2yolo
        elif self.sender() == self.pushButton_ivas2yolo_ivas_json_root:
            lineEdit = self.lineEdit_ivas2yolo_ivas_json_root
        elif self.sender() == self.pushButton_ivas2yolo_yolo_txt_root:
            lineEdit = self.lineEdit_ivas2yolo_yolo_txt_root
        # labelme2ivas
        elif self.sender() == self.pushButton_labelme2ivas_labelme_json_root:
            lineEdit = self.lineEdit_labelme2ivas_labelme_json_root
        elif self.sender() == self.pushButton_labelme2ivas_ivas_json_root:
            lineEdit = self.lineEdit_labelme2ivas_ivas_json_root
        # ivas2labelme
        elif self.sender() == self.pushButton_ivas2labelme_ivas_json_root:
            lineEdit = self.lineEdit_ivas2labelme_ivas_json_root
        elif self.sender() == self.pushButton_ivas2labelme_labelme_json_root:
            lineEdit = self.lineEdit_ivas2labelme_labelme_json_root
        # ivas2voc
        elif self.sender() == self.pushButton_ivas2voc_ivas_json_root:
            lineEdit = self.lineEdit_ivas2voc_ivas_json_root
        elif self.sender() == self.pushButton_ivas2voc_voc_png_root:
            lineEdit = self.lineEdit_ivas2voc_voc_png_root
        # ivas2vocod
        elif self.sender() == self.pushButton_ivas2vocod_ivas_json_root:
            lineEdit = self.lineEdit_ivas2vocod_ivas_json_root
        elif self.sender() == self.pushButton_ivas2vocod_voc_xml_root:
            lineEdit = self.lineEdit_ivas2vocod_voc_xml_root
        else:
            lineEdit = None

        if dir:
            self.save_root = dir
            if lineEdit is not None:
                lineEdit.setText(dir)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def print_message(self, index, all, message):
        if all > 0:
            self.progressBar.setMaximum(all)
        if index > 0:
            self.progressBar.setValue(index)
        if message:
            self.textBrowser.append(message)

    def init_connect(self):
        self.pushButton_start.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)
        # coco2ivas
        self.pushButton_coco2ivas_coco_json_path.clicked.connect(self.open_file)
        self.pushButton_coco2ivas_ivas_json_root.clicked.connect(self.save_dir)
        # ivas2coco
        self.pushButton_ivas2coco_ivas_json_root.clicked.connect(self.open_dir)
        self.pushButton_ivas2coco_coco_json_path.clicked.connect(self.save_file)
        # yolo2ivas
        self.pushButton_yolo2ivas_yolo_image_root.clicked.connect(self.open_dir)
        self.pushButton_yolo2ivas_yolo_txt_root.clicked.connect(self.open_dir)
        self.pushButton_yolo2ivas_yolo_cate_path.clicked.connect(self.open_file)
        self.pushButton_yolo2ivas_ivas_json_root.clicked.connect(self.save_dir)
        # ivas2yolo
        self.pushButton_ivas2yolo_ivas_json_root.clicked.connect(self.open_dir)
        self.pushButton_ivas2yolo_yolo_txt_root.clicked.connect(self.save_dir)
        # labelme2ivas
        self.pushButton_labelme2ivas_labelme_json_root.clicked.connect(self.open_dir)
        self.pushButton_labelme2ivas_ivas_json_root.clicked.connect(self.save_dir)
        # ivas2labelme
        self.pushButton_ivas2labelme_ivas_json_root.clicked.connect(self.open_dir)
        self.pushButton_ivas2labelme_labelme_json_root.clicked.connect(self.save_dir)
        # voc
        self.pushButton_ivas2voc_ivas_json_root.clicked.connect(self.open_dir)
        self.pushButton_ivas2voc_voc_png_root.clicked.connect(self.save_dir)
        # voc-object-detection
        self.pushButton_ivas2vocod_ivas_json_root.clicked.connect(self.open_dir)
        self.pushButton_ivas2vocod_voc_xml_root.clicked.connect(self.save_dir)
