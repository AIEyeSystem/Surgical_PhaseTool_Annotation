# -*- coding: utf-8 -*-
"""Export the current label directory, or import a dataset into one."""

import os
import shutil

import imgviz
import yaml
from PyQt5 import QtCore, QtWidgets

from IVAS.branding import CATEGORY_CONFIG_NAME
from IVAS.configs import save_config
from IVAS.widgets.converter_dialog import (
    COCOConverter,
    Converter,
    LABELMEConverter,
    VOCConverter,
    VOCConverterForDetection,
    YOLOConverter,
    _find_category_config,
)
from IVAS.widgets.responsive import ResponsiveDialog
from IVAS.widgets.workbench_chrome import icon_for


class IVASCopyConverter(Converter):
    """Re-write IVAS JSON (normalising old ISAT/IVAT files) into another folder."""

    def __init__(self):
        super().__init__()
        self.output_root = None

    def convert(self):
        if self.ivas_json_root is None or self.output_root is None:
            return
        self.load_from_ivas()
        source_config = _find_category_config(self.ivas_json_root)
        destination = self.output_root
        self.ivas_json_root = destination
        self.convert_to_ivas()
        if source_config is not None:
            # Keep the user's category order and colours, not generated ones.
            shutil.copyfile(source_config, os.path.join(destination, CATEGORY_CONFIG_NAME))


# key, zh name, en name, output kind ("file"/"dir"), zh hint, en hint
EXPORT_FORMATS = (
    ("coco", "COCO（实例分割 + 检测框）", "COCO (instances + boxes)", "file",
     "单个 JSON；矩形框和旋转框带 ivas_shape_type，可无损导回。",
     "One JSON file; boxes keep ivas_shape_type for lossless re-import."),
    ("yolo-seg", "YOLO 分割", "YOLO segmentation", "dir",
     "每张图一个 txt：cls x1 y1 x2 y2 …（归一化多边形）。",
     "One txt per image: cls x1 y1 x2 y2 … (normalised polygons)."),
    ("yolo-det", "YOLO 检测框", "YOLO detection", "dir",
     "每张图一个 txt：cls cx cy w h；同一 group 的目标合并为一个框。",
     "One txt per image: cls cx cy w h; objects sharing a group form one box."),
    ("labelme", "LabelMe", "LabelMe", "dir",
     "每张图一个 JSON；矩形框导出为 LabelMe rectangle。",
     "One JSON per image; boxes become LabelMe rectangles."),
    ("voc-seg", "Pascal VOC 分割 PNG", "Pascal VOC segmentation PNG", "dir",
     "语义或实例调色板 PNG，附 classification.txt。",
     "Semantic or instance palette PNGs with classification.txt."),
    ("voc-det", "Pascal VOC 检测 XML", "Pascal VOC detection XML", "dir",
     "每张图一个 XML；同一 group 的目标合并为一个框。",
     "One XML per image; objects sharing a group form one box."),
    ("ivas", "IVAS JSON 副本", "IVAS JSON copy", "dir",
     "复制并统一为当前 IVAS 格式，保留类别配置。",
     "Copy and normalise to the current IVAS schema, keeping the category file."),
)

IMPORT_FORMATS = (
    ("coco", "COCO JSON", "COCO JSON", "file",
     "支持多边形、RLE 和仅含 bbox 的检测数据。",
     "Polygons, RLE and bbox-only detection datasets."),
    ("yolo", "YOLO（分割或检测）", "YOLO (segmentation or detection)", "dir",
     "需要图像目录；5 列的行导入为矩形框。",
     "Needs the image folder; 5-column lines import as boxes."),
    ("labelme", "LabelMe", "LabelMe", "dir",
     "导入 polygon 与 rectangle 形状。", "Imports polygon and rectangle shapes."),
    ("voc-det", "Pascal VOC 检测 XML", "Pascal VOC detection XML", "dir",
     "每个 object 导入为一个矩形框。", "Each object becomes one box."),
)


class ExportDialog(ResponsiveDialog):
    """Two tabs: export the active label folder, or import into a folder."""

    def __init__(self, parent, mainwindow):
        super().__init__(parent)
        self.mainwindow = mainwindow
        self.worker = None
        self._job = None
        self._category_backup = None
        self.setObjectName("exportDialog")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(720, 560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(10)
        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs)

        # ---------------- export tab
        export_page = QtWidgets.QWidget()
        form = QtWidgets.QGridLayout(export_page)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self.export_source_label = QtWidgets.QLabel()
        self.export_source = QtWidgets.QLabel()
        self.export_source.setWordWrap(True)
        self.export_source.setObjectName("workspaceMeta")
        form.addWidget(self.export_source_label, 0, 0)
        form.addWidget(self.export_source, 0, 1, 1, 2)
        self.export_format_label = QtWidgets.QLabel()
        self.export_formats = QtWidgets.QListWidget()
        self.export_formats.setObjectName("exportFormatList")
        self.export_formats.setMinimumHeight(len(EXPORT_FORMATS) * 34 + 6)
        for key, *_rest in EXPORT_FORMATS:
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, key)
            self.export_formats.addItem(item)
        form.addWidget(self.export_format_label, 1, 0, QtCore.Qt.AlignTop)
        form.addWidget(self.export_formats, 1, 1, 1, 2)
        self.export_hint = QtWidgets.QLabel()
        self.export_hint.setWordWrap(True)
        self.export_hint.setObjectName("workspaceMeta")
        form.addWidget(self.export_hint, 2, 1, 1, 2)
        self.export_options_label = QtWidgets.QLabel()
        options = QtWidgets.QHBoxLayout()
        self.keep_crowd = QtWidgets.QCheckBox()
        self.keep_crowd.setChecked(True)
        self.ignore_background = QtWidgets.QCheckBox()
        self.ignore_background.setChecked(True)
        self.voc_instance = QtWidgets.QCheckBox()
        self.voc_category_colors = QtWidgets.QCheckBox()
        self.voc_category_colors.setChecked(True)
        for box in (self.keep_crowd, self.ignore_background, self.voc_instance,
                    self.voc_category_colors):
            options.addWidget(box)
        options.addStretch(1)
        form.addWidget(self.export_options_label, 3, 0)
        form.addLayout(options, 3, 1, 1, 2)
        self.export_output_label = QtWidgets.QLabel()
        self.export_output = QtWidgets.QLineEdit()
        self.export_browse = QtWidgets.QPushButton()
        form.addWidget(self.export_output_label, 4, 0)
        form.addWidget(self.export_output, 4, 1)
        form.addWidget(self.export_browse, 4, 2)
        form.setColumnStretch(1, 1)
        self.tabs.addTab(export_page, "")

        # ---------------- import tab
        import_page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(import_page)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.import_format_label = QtWidgets.QLabel()
        self.import_formats = QtWidgets.QComboBox()
        for key, *_rest in IMPORT_FORMATS:
            self.import_formats.addItem("", key)
        grid.addWidget(self.import_format_label, 0, 0)
        grid.addWidget(self.import_formats, 0, 1, 1, 2)
        self.import_hint = QtWidgets.QLabel()
        self.import_hint.setWordWrap(True)
        self.import_hint.setObjectName("workspaceMeta")
        grid.addWidget(self.import_hint, 1, 1, 1, 2)
        self.import_source_label = QtWidgets.QLabel()
        self.import_source = QtWidgets.QLineEdit()
        self.import_source_browse = QtWidgets.QPushButton()
        grid.addWidget(self.import_source_label, 2, 0)
        grid.addWidget(self.import_source, 2, 1)
        grid.addWidget(self.import_source_browse, 2, 2)
        self.import_images_label = QtWidgets.QLabel()
        self.import_images = QtWidgets.QLineEdit()
        self.import_images_browse = QtWidgets.QPushButton()
        grid.addWidget(self.import_images_label, 3, 0)
        grid.addWidget(self.import_images, 3, 1)
        grid.addWidget(self.import_images_browse, 3, 2)
        self.import_classes_label = QtWidgets.QLabel()
        self.import_classes = QtWidgets.QLineEdit()
        self.import_classes_browse = QtWidgets.QPushButton()
        grid.addWidget(self.import_classes_label, 4, 0)
        grid.addWidget(self.import_classes, 4, 1)
        grid.addWidget(self.import_classes_browse, 4, 2)
        self.import_output_label = QtWidgets.QLabel()
        self.import_output = QtWidgets.QLineEdit()
        self.import_output_browse = QtWidgets.QPushButton()
        grid.addWidget(self.import_output_label, 5, 0)
        grid.addWidget(self.import_output, 5, 1)
        grid.addWidget(self.import_output_browse, 5, 2)
        self.import_use_current = QtWidgets.QPushButton()
        grid.addWidget(self.import_use_current, 6, 1, QtCore.Qt.AlignLeft)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(7, 1)
        self.tabs.addTab(import_page, "")

        # ---------------- progress + footer
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        layout.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(90)
        layout.addWidget(self.log, 1)
        footer = QtWidgets.QDialogButtonBox()
        self.run_button = footer.addButton("", QtWidgets.QDialogButtonBox.AcceptRole)
        self.run_button.setObjectName("primaryAction")
        self.close_button = footer.addButton("", QtWidgets.QDialogButtonBox.RejectRole)
        layout.addWidget(footer)

        self.export_formats.currentRowChanged.connect(self._export_format_changed)
        self.import_formats.currentIndexChanged.connect(self._import_format_changed)
        self.voc_instance.toggled.connect(self._export_format_changed)
        self.export_browse.clicked.connect(self._browse_export_output)
        self.import_source_browse.clicked.connect(self._browse_import_source)
        self.import_images_browse.clicked.connect(
            lambda: self._browse_directory(self.import_images))
        self.import_classes_browse.clicked.connect(self._browse_classes)
        self.import_output_browse.clicked.connect(
            lambda: self._browse_directory(self.import_output))
        self.import_use_current.clicked.connect(self._use_current_label_root)
        self.tabs.currentChanged.connect(self._update_run_button)
        self.run_button.clicked.connect(self.run)
        self.close_button.clicked.connect(self.reject)

        settings = self.mainwindow.cfg.get("software", {})
        last = settings.get("export_format", "coco")
        keys = [key for key, *_rest in EXPORT_FORMATS]
        self.export_formats.setCurrentRow(keys.index(last) if last in keys else 0)
        self.retranslateUi(self)

    # ------------------------------------------------------------ text
    def _zh(self):
        language = getattr(self.mainwindow, "_ui_language", None)
        if language is None:
            language = self.mainwindow.cfg.get("software", {}).get("language", "en")
        return language == "zh"

    def _t(self, chinese, english):
        return chinese if self._zh() else english

    def retranslateUi(self, _dialog=None):
        zh = self._zh()
        t = self._t
        self.setWindowTitle(t("导出 / 导入标注", "Export / import annotations"))
        self.tabs.setTabText(0, t("导出", "Export"))
        self.tabs.setTabText(1, t("导入", "Import"))
        self.tabs.setTabIcon(0, icon_for("export"))
        self.tabs.setTabIcon(1, icon_for("import"))
        self.export_source_label.setText(t("来源", "Source"))
        self.export_format_label.setText(t("格式", "Format"))
        self.export_options_label.setText(t("选项", "Options"))
        self.export_output_label.setText(t("输出", "Output"))
        self.keep_crowd.setText(t("保留 crowd 目标", "Keep crowd objects"))
        self.ignore_background.setText(t("忽略 __background__", "Skip __background__"))
        self.voc_instance.setText(t("实例 PNG", "Instance PNG"))
        self.voc_category_colors.setText(t("使用类别颜色", "Use category colours"))
        for row, (_key, chinese, english, _kind, _zh_hint, _en_hint) in enumerate(EXPORT_FORMATS):
            self.export_formats.item(row).setText(chinese if zh else english)
        self.export_browse.setText(t("浏览…", "Browse…"))
        self.import_format_label.setText(t("格式", "Format"))
        self.import_source_label.setText(t("来源", "Source"))
        self.import_images_label.setText(t("图像目录", "Image folder"))
        self.import_classes_label.setText(t("类别文件", "Class file"))
        self.import_output_label.setText(t("IVAS 标签目录", "IVAS label folder"))
        for index, (_key, chinese, english, _kind, _zh_hint, _en_hint) in enumerate(IMPORT_FORMATS):
            self.import_formats.setItemText(index, chinese if zh else english)
        for button in (self.import_source_browse, self.import_images_browse,
                       self.import_classes_browse, self.import_output_browse):
            button.setText(t("浏览…", "Browse…"))
        self.import_use_current.setText(t("使用当前标签目录", "Use current label folder"))
        self.import_classes.setPlaceholderText(
            t("可选：每行一个类别名", "Optional: one class name per line"))
        self.close_button.setText(t("关闭", "Close"))
        self._export_format_changed()
        self._import_format_changed()
        self._update_run_button()

    # ------------------------------------------------------------ state
    def prepare(self):
        """Refresh the source summary each time the dialog is opened."""
        label_root = self.mainwindow.label_root
        if label_root and os.path.isdir(label_root):
            count = len([name for name in os.listdir(label_root) if name.endswith(".json")])
            self.export_source.setText(
                self._t("{}（{} 个标签文件）", "{} ({} label files)").format(label_root, count)
            )
        else:
            self.export_source.setText(
                self._t("尚未打开图像目录或标签目录。", "No image or label folder is open."))
        self._suggest_export_output()
        self._update_run_button()

    def _export_key(self):
        item = self.export_formats.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else "coco"

    def _export_spec(self):
        key = self._export_key()
        return next(spec for spec in EXPORT_FORMATS if spec[0] == key)

    def _import_spec(self):
        key = self.import_formats.currentData() or "coco"
        return next(spec for spec in IMPORT_FORMATS if spec[0] == key)

    def _export_format_changed(self, *_args):
        key, _chinese, _english, _kind, zh_hint, en_hint = self._export_spec()
        self.export_hint.setText(zh_hint if self._zh() else en_hint)
        self.keep_crowd.setVisible(key in ("coco", "yolo-seg", "yolo-det", "labelme"))
        self.ignore_background.setVisible(key in ("yolo-seg", "yolo-det"))
        self.voc_instance.setVisible(key == "voc-seg")
        self.voc_category_colors.setVisible(key == "voc-seg")
        self.voc_category_colors.setEnabled(not self.voc_instance.isChecked())
        self._suggest_export_output()

    def _suggest_export_output(self):
        label_root = self.mainwindow.label_root
        if not label_root:
            return
        key, *_rest = self._export_spec()
        base = os.path.dirname(os.path.abspath(label_root))
        name = os.path.basename(os.path.abspath(label_root)) or "labels"
        suggestions = {
            "coco": os.path.join(base, name + "_coco.json"),
            "yolo-seg": os.path.join(base, name + "_yolo_seg"),
            "yolo-det": os.path.join(base, name + "_yolo_det"),
            "labelme": os.path.join(base, name + "_labelme"),
            "voc-seg": os.path.join(base, name + "_voc_png"),
            "voc-det": os.path.join(base, name + "_voc_xml"),
            "ivas": os.path.join(base, name + "_ivas_copy"),
        }
        current = self.export_output.text().strip()
        if not current or current in suggestions.values():
            self.export_output.setText(suggestions[key])

    def _import_format_changed(self, *_args):
        key, _chinese, _english, kind, zh_hint, en_hint = self._import_spec()
        self.import_hint.setText(zh_hint if self._zh() else en_hint)
        yolo = key == "yolo"
        for widget in (self.import_images_label, self.import_images, self.import_images_browse,
                       self.import_classes_label, self.import_classes, self.import_classes_browse):
            widget.setVisible(yolo)
        self.import_source.setPlaceholderText(
            self._t("COCO JSON 文件", "COCO JSON file") if kind == "file"
            else self._t("标注文件所在目录", "Folder with annotation files"))

    def _update_run_button(self, *_args):
        busy = self.worker is not None and self.worker.isRunning()
        exporting = self.tabs.currentIndex() == 0
        self.run_button.setText(
            self._t("取消", "Cancel") if busy
            else (self._t("导出", "Export") if exporting else self._t("导入", "Import")))
        self.run_button.setIcon(icon_for("export" if exporting else "import"))
        if exporting and not busy:
            self.run_button.setEnabled(bool(self.mainwindow.label_root))
        else:
            self.run_button.setEnabled(True)
        self.tabs.setEnabled(not busy)

    # ------------------------------------------------------------ browse
    def _browse_directory(self, line_edit):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "", line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_export_output(self):
        key, _chinese, _english, kind, *_hints = self._export_spec()
        if kind == "file":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "", self.export_output.text(), "JSON (*.json)")
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "", self.export_output.text())
        if path:
            self.export_output.setText(path)

    def _browse_import_source(self):
        key, _chinese, _english, kind, *_hints = self._import_spec()
        if kind == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "", self.import_source.text(), "JSON (*.json)")
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "", self.import_source.text())
        if path:
            self.import_source.setText(path)
            if not self.import_output.text():
                base = path[:-5] if path.lower().endswith(".json") else path.rstrip("/\\")
                self.import_output.setText(base + "_ivas")

    def _browse_classes(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "", self.import_classes.text(), "Text (*.txt)")
        if path:
            self.import_classes.setText(path)

    def _use_current_label_root(self):
        if self.mainwindow.label_root:
            self.import_output.setText(self.mainwindow.label_root)

    # ------------------------------------------------------------ run
    def _log(self, text):
        self.log.appendPlainText(text)

    def _progress(self, index, total, message):
        if index >= 0 and total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(index)
        if message:
            self._log(message)

    def run(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel = True
            return
        if self.tabs.currentIndex() == 0:
            self._run_export()
        else:
            self._run_import()

    def build_export_worker(self, key, output):
        """Return a configured converter thread (public for tests)."""
        label_root = self.mainwindow.label_root
        if key == "coco":
            worker = COCOConverter()
            worker.coco2ivas = False
            worker.coco_json_path = output
            worker.keep_crowd = self.keep_crowd.isChecked()
        elif key in ("yolo-seg", "yolo-det"):
            worker = YOLOConverter()
            worker.yolo2ivas = False
            worker.yolo_txt_root = output
            worker.detection = key == "yolo-det"
            worker.keep_crowd = self.keep_crowd.isChecked()
            worker.ignore_background = self.ignore_background.isChecked()
        elif key == "labelme":
            worker = LABELMEConverter()
            worker.labelme2ivas = False
            worker.labelme_json_root = output
            worker.keep_crowd = self.keep_crowd.isChecked()
        elif key == "voc-seg":
            worker = VOCConverter()
            worker.voc_png_root = output
            worker.is_instance = self.voc_instance.isChecked()
            worker.use_setting_color = self.voc_category_colors.isChecked()
        elif key == "voc-det":
            worker = VOCConverterForDetection()
            worker.voc_xml_root = output
        else:
            worker = IVASCopyConverter()
            worker.output_root = output
        worker.ivas_json_root = label_root
        return worker

    def _run_export(self):
        label_root = self.mainwindow.label_root
        output = self.export_output.text().strip()
        key, _chinese, _english, kind, *_hints = self._export_spec()
        if not label_root or not os.path.isdir(label_root):
            QtWidgets.QMessageBox.warning(self, self.windowTitle(),
                                          self._t("请先打开图像目录。", "Open an image folder first."))
            return
        if not output:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(),
                                          self._t("请选择输出位置。", "Choose an output location."))
            return
        if os.path.abspath(output) == os.path.abspath(label_root):
            QtWidgets.QMessageBox.warning(
                self, self.windowTitle(),
                self._t("输出不能是当前标签目录。", "The output cannot be the label folder itself."))
            return
        directory = os.path.dirname(os.path.abspath(output)) if kind == "file" else output
        os.makedirs(directory, exist_ok=True)
        self.mainwindow.cfg["software"]["export_format"] = key
        self.mainwindow.save_software_cfg()
        self._start(self.build_export_worker(key, output), ("export", key, output))

    def build_import_worker(self, key, source, output):
        if key == "coco":
            worker = COCOConverter()
            worker.coco2ivas = True
            worker.coco_json_path = source
            worker.keep_crowd = True
        elif key == "yolo":
            worker = YOLOConverter()
            worker.yolo2ivas = True
            worker.yolo_txt_root = source
            worker.yolo_images_root = self.import_images.text().strip() or None
            worker.yolo_category_file = self.import_classes.text().strip() or None
        elif key == "labelme":
            worker = LABELMEConverter()
            worker.labelme2ivas = True
            worker.labelme_json_root = source
        else:
            worker = VOCConverterForDetection()
            worker.voc2ivas = True
            worker.voc_xml_root = source
        worker.ivas_json_root = output
        return worker

    def _run_import(self):
        key, _chinese, _english, kind, *_hints = self._import_spec()
        source = self.import_source.text().strip()
        output = self.import_output.text().strip()
        valid_source = os.path.isfile(source) if kind == "file" else os.path.isdir(source)
        if not valid_source or not output:
            QtWidgets.QMessageBox.warning(
                self, self.windowTitle(),
                self._t("请选择有效的来源和输出目录。", "Choose a valid source and output folder."))
            return
        if key == "yolo" and not os.path.isdir(self.import_images.text().strip()):
            QtWidgets.QMessageBox.warning(
                self, self.windowTitle(),
                self._t("YOLO 导入需要图像目录。", "YOLO import needs the image folder."))
            return
        os.makedirs(output, exist_ok=True)
        existing = [name for name in os.listdir(output) if name.endswith(".json")]
        if existing:
            answer = QtWidgets.QMessageBox.question(
                self, self.windowTitle(),
                self._t("输出目录已有 {} 个 JSON 标签，同名文件会被覆盖。继续吗？",
                        "The output folder already has {} JSON labels; files with the same "
                        "name will be overwritten. Continue?").format(len(existing)))
            if answer != QtWidgets.QMessageBox.Yes:
                return
        if output == self.mainwindow.label_root and not self.mainwindow._confirm_leave_unsaved_image():
            return
        config = _find_category_config(output)
        self._category_backup = None
        if config is not None:
            with open(config, "r", encoding="utf-8") as stream:
                self._category_backup = (config, yaml.safe_load(stream) or {})
        self._start(self.build_import_worker(key, source, output), ("import", key, output))

    def _start(self, worker, job):
        self.log.clear()
        self.progress.setValue(0)
        self.worker = worker
        self._job = job
        worker.message.connect(self._progress)
        worker.finished.connect(self._finished)
        self._log(self._t("开始 → {}", "Starting → {}").format(job[2]))
        worker.start()
        self._update_run_button()

    def _merge_category_backup(self, output):
        """Keep existing categories (order, colours) and append imported ones."""
        if self._category_backup is None:
            return
        path, previous = self._category_backup
        self._category_backup = None
        generated = _find_category_config(output)
        labels = list(previous.get("label") or [])
        known = {label.get("name") for label in labels if isinstance(label, dict)}
        added = []
        if generated is not None:
            with open(generated, "r", encoding="utf-8") as stream:
                for label in (yaml.safe_load(stream) or {}).get("label") or []:
                    if isinstance(label, dict) and label.get("name") not in known:
                        added.append(label)
        cmap = imgviz.label_colormap()
        for offset, label in enumerate(added):
            if not label.get("color"):
                r, g, b = cmap[(len(labels) + offset + 1) % len(cmap)]
                label["color"] = "#{:02x}{:02x}{:02x}".format(r, g, b)
        merged = dict(previous)
        merged["label"] = labels + added
        save_config(merged, path)
        if generated is not None and os.path.abspath(generated) != os.path.abspath(path):
            os.remove(generated)

    def _finished(self):
        worker, job = self.worker, self._job
        self.worker = None
        if job is None:
            return
        direction, _key, output = job
        if direction == "import":
            try:
                self._merge_category_backup(output)
            except Exception as error:
                self._log("Category merge failed: {}".format(error))
        errors = len(getattr(worker, "errors", []) or [])
        cancelled = bool(getattr(worker, "cancel", False))
        if cancelled:
            summary = self._t("已取消。", "Cancelled.")
        elif errors:
            summary = self._t("完成，{} 个错误（见日志）。", "Finished with {} error(s); see the log.").format(errors)
        else:
            summary = self._t("完成：{}", "Done: {}").format(output)
        self._log(summary)
        self.mainwindow.statusbar.showMessage(summary, 6000)
        if (direction == "import" and not cancelled and self.mainwindow.label_root
                and os.path.abspath(output) == os.path.abspath(self.mainwindow.label_root)
                and self.mainwindow.current_index is not None):
            self.mainwindow.show_image(self.mainwindow.current_index, zoomfit=False)
        self._update_run_button()

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel = True
            self.worker.wait(5000)
        super().reject()
