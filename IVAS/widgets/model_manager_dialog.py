# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os
import shutil
import time
import re
import hashlib
from contextlib import closing
from urllib.error import HTTPError
from functools import partial
from urllib import request

from PyQt5 import QtCore, QtGui, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from PyQt5.QtCore import QThread, pyqtSignal

from IVAS.configs import CHECKPOINT_PATH
from IVAS.segment_any.model_zoo import model_dict
from IVAS.ui.model_manager_dialog import Ui_Dialog


class DownloadThread(QThread):
    """
    Thread for downloading checkpoint.

    Attributes:
        urls (list): Url list, which include: 'huggingface' and 'modelscope'.
        name (str): Checkpoint name.
        pause (bool): Pause download.
    """

    tag = pyqtSignal(float, float)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super(DownloadThread, self).__init__(parent)
        self.urls = None
        self.name = None
        self.pause = False
        self.sha256 = None
        self.last_error = ""

        self.block_size = 4096

    def setNameAndUrl(self, name, urls, sha256=None):
        self.urls = urls
        self.name = name
        self.pause = False
        self.sha256 = sha256
        self.last_error = ""

    def run(self):
        if not self.name or not self.urls:
            return
        try:
            tmp_root = os.path.join(CHECKPOINT_PATH, "tmp")
            os.makedirs(tmp_root, exist_ok=True)
            temporary = os.path.join(tmp_root, os.path.basename(self.name))
            errors = []
            for url in self.urls:
                if self.pause or self.isInterruptionRequested():
                    break
                try:
                    self._download(url, temporary)
                    if self.pause or self.isInterruptionRequested():
                        break
                    if self.sha256:
                        digest = hashlib.sha256()
                        with open(temporary, "rb") as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(block)
                        if digest.hexdigest().lower() != self.sha256.lower():
                            os.unlink(temporary)
                            raise ValueError("Checkpoint checksum does not match")
                    self.move_from_tmp(temporary, os.path.join(CHECKPOINT_PATH, os.path.basename(self.name)))
                    return
                except Exception as error:
                    errors.append(str(error))
            if errors and not (self.pause or self.isInterruptionRequested()):
                raise OSError("; ".join(errors))
        except Exception as error:
            self.last_error = str(error)
            self.failed.emit(self.last_error)
        self.tag.emit(-1, -1)

    def _download(self, url, temporary):
        offset = os.path.getsize(temporary) if os.path.exists(temporary) else 0
        headers = {"Range": "bytes={}-".format(offset)} if offset else {}
        try:
            response = request.urlopen(request.Request(url, headers=headers), timeout=10)
        except HTTPError as error:
            if error.code != 416:
                raise
            error.close()
            offset = 0
            response = request.urlopen(request.Request(url), timeout=10)
        with closing(response):
            status = response.getcode()
            content_length = int(response.headers.get("Content-Length", -1))
            if content_length < 0:
                raise ValueError("Download response is missing its length")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset or int(match[2]) - offset + 1 != content_length:
                    raise ValueError("Invalid download range response")
                total = int(match[3])
                if int(match[2]) != total - 1:
                    raise ValueError("Incomplete download range")
            elif status == 200:
                offset, total = 0, content_length
            else:
                raise ValueError("Unexpected download HTTP status {}".format(status))
            received = 0
            with open(temporary, "ab" if offset else "wb") as stream:
                while not (self.pause or self.isInterruptionRequested()):
                    block = response.read(self.block_size)
                    if not block:
                        break
                    stream.write(block)
                    received += len(block)
                    if received > content_length:
                        raise ValueError("Download body exceeds declared length")
                    self.tag.emit(offset + received, total)
                stream.flush()
                os.fsync(stream.fileno())
            if self.pause or self.isInterruptionRequested():
                return
            if received != content_length or os.path.getsize(temporary) != total or total <= 0:
                raise ValueError("Checkpoint download is incomplete")

    def move_from_tmp(self, from_path, to_path):
        os.replace(from_path, to_path)
        self.tag.emit(-2, -2)


class ModelManagerDialog(ResponsiveDialog, Ui_Dialog):
    """Model manager interface."""

    def __init__(self, parent, mainwindow):
        super(ModelManagerDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.download_thread_dict = {}
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.addButton(self.radioButton_ft_model)

        self.label_9.setWordWrap(True)
        self.init_ui()
        self.pushButton_clear_tmp.clicked.connect(self.clear_tmp)
        self.pushButton_checkpoint_dir.clicked.connect(self.open_checkpoint_dir)
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_open_ft_model.clicked.connect(self.load_fine_tuned_model)
        self.radioButton_ft_model.clicked.connect(self.use_fine_tune_model)

    def init_ui(self):
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)

        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()

        for index, (model_name, info_dict) in enumerate(model_dict.items()):
            url = info_dict.get("url", "")
            memory = info_dict.get("memory", "")
            bf16_memory = info_dict.get("bf16_memory", "")
            params = info_dict.get("params", "")
            image_segment = info_dict.get("image_segment", False)
            video_segment = info_dict.get("video_segment", False)
            # radio
            radio = QtWidgets.QRadioButton()
            radio.setFixedWidth(30)
            radio.toggled.connect(
                partial(
                    self.mainwindow.init_segment_anything,
                    os.path.join(CHECKPOINT_PATH, model_name),
                )
            )
            radio.setEnabled(False)
            self.button_group.addButton(radio)
            # image seg
            image_segment_label = QtWidgets.QLabel()
            pixmap = (
                QtGui.QPixmap(":/icon/icons/校验-小_check-small.svg")
                if image_segment
                else QtGui.QPixmap(":/icon/icons/关闭-小_close-small.svg")
            )
            image_segment_label.setPixmap(pixmap)
            image_segment_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image_segment_label.setFixedWidth(80)
            image_segment_label.setStyleSheet(
                "background-color: rgb(240, 240, 240);"
                if index % 2
                else "background-color: rgb(255, 255, 255);"
            )
            # video seg
            video_segment_label = QtWidgets.QLabel()
            pixmap = (
                QtGui.QPixmap(":/icon/icons/校验-小_check-small.svg")
                if video_segment
                else QtGui.QPixmap(":/icon/icons/关闭-小_close-small.svg")
            )
            video_segment_label.setPixmap(pixmap)
            video_segment_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            video_segment_label.setFixedWidth(80)
            video_segment_label.setStyleSheet(
                "background-color: rgb(240, 240, 240);"
                if index % 2
                else "background-color: rgb(255, 255, 255);"
            )
            # model name
            name_label = QtWidgets.QLabel()
            name_label.setFont(font)
            name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
            name_label.setText(model_name)
            name_label.setStyleSheet(
                "background-color: rgb(240, 240, 240);"
                if index % 2
                else "background-color: rgb(255, 255, 255);"
            )
            # 显存占用
            memory_label = QtWidgets.QLabel()
            memory_label.setFont(font)
            memory_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            memory_label.setText(bf16_memory)
            memory_label.setFixedWidth(100)
            memory_label.setStyleSheet(
                "background-color: rgb(240, 240, 240);"
                if index % 2
                else "background-color: rgb(255, 255, 255);"
            )
            # 权重大小
            params_label = QtWidgets.QLabel()
            params_label.setFont(font)
            params_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            params_label.setText(params)
            params_label.setFixedWidth(100)
            params_label.setStyleSheet(
                "background-color: rgb(240, 240, 240);"
                if index % 2
                else "background-color: rgb(255, 255, 255);"
            )
            # 下载/删除按钮
            ops_button = QtWidgets.QPushButton()
            ops_button.setMinimumWidth(130)
            ops_button.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
            ops_button.setFont(font)

            if os.path.exists(os.path.join(CHECKPOINT_PATH, model_name)):
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'red')
                ops_button.setStyleSheet("QWidget {color: %s}" % "red")
                ops_button.setText("delete")
                ops_button.clicked.connect(partial(self.delete, model_name))
                radio.setEnabled(True)
            else:
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'green')
                ops_button.setStyleSheet("QWidget {color: %s}" % "green")
                ops_button.setText("download")
                ops_button.clicked.connect(partial(self.download, model_name))

            self.gridLayout.addWidget(radio, index, 0, 1, 1)
            self.gridLayout.addWidget(image_segment_label, index, 1, 1, 1)
            self.gridLayout.addWidget(video_segment_label, index, 2, 1, 1)
            self.gridLayout.addWidget(name_label, index, 3, 1, 1)
            self.gridLayout.addWidget(memory_label, index, 4, 1, 1)
            self.gridLayout.addWidget(params_label, index, 5, 1, 1)
            self.gridLayout.addWidget(ops_button, index, 6, 1, 1)

    def update_ui(self):
        for index, (model_name, info_dict) in enumerate(model_dict.items()):
            url = info_dict.get("url", "")
            memory = info_dict.get("memory", "")
            bf16_memory = info_dict.get("bf16_memory", "")
            params = info_dict.get("params", "")
            image_segment = info_dict.get("image_segment", False)
            video_segment = info_dict.get("video_segment", False)

            radio = self.gridLayout.itemAtPosition(index, 0).widget()
            image_segment_label = self.gridLayout.itemAtPosition(index, 1).widget()
            video_segment_label = self.gridLayout.itemAtPosition(index, 2).widget()
            name_label = self.gridLayout.itemAtPosition(index, 3).widget()
            memory_label = self.gridLayout.itemAtPosition(index, 4).widget()
            params_label = self.gridLayout.itemAtPosition(index, 5).widget()
            ops_button = self.gridLayout.itemAtPosition(index, 6).widget()

            radio.blockSignals(True)
            if self.mainwindow.use_segment_anything:
                current_model_name = os.path.split(self.mainwindow.segany.checkpoint)[
                    -1
                ]
                radio.setChecked(current_model_name == model_name)
            else:
                radio.setChecked(False)
            radio.blockSignals(False)

            precision = self.mainwindow.cfg["software"].get(
                "sam_precision",
                (
                    "bfloat16"
                    if self.mainwindow.cfg["software"].get(
                        "use_bfloat16", False
                    )
                    else "float32"
                ),
            )
            if precision == "original":
                memory_label.setText("checkpoint")
            elif precision in ("bfloat16", "float16"):
                memory_label.setText(bf16_memory)
            else:
                memory_label.setText(memory)

            if os.path.exists(os.path.join(CHECKPOINT_PATH, model_name)):
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'red')
                ops_button.setStyleSheet("QWidget {color: %s}" % "red")
                ops_button.setText("delete")
                ops_button.disconnect()
                ops_button.clicked.connect(partial(self.delete, model_name))
                radio.setEnabled(True)
            else:
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'green')
                ops_button.setStyleSheet("QWidget {color: %s}" % "green")
                ops_button.setText("download")
                ops_button.disconnect()
                ops_button.clicked.connect(partial(self.download, model_name))
                radio.setEnabled(False)

    def download(self, model_name):
        # sam3 / sam3.1 模型特殊处理
        if "sam3.1" in model_name:
            message = (
                "Please manually download sam3.1_multiplex.pt.\n"
                "The download URL is https://huggingface.co/facebook/sam3.1 "
                "(research license required)."
                if self.mainwindow.cfg["software"]["language"] == "en"
                else "请手动下载sam3.1_multiplex.pt。\n"
                "下载网址 https://huggingface.co/facebook/sam3.1（需要研究许可）。"
            )
            QtWidgets.QMessageBox.warning(
                self, "warning", message
            )
            return
        if "sam3" in model_name:
            message = "Please manually download sam3.pt.\nThe download URL is https://huggingface.co/facebook/sam3." \
                if self.mainwindow.cfg["software"]["language"] == "en" \
                else "请手动下载sam3.pt。\n下载网址 https://huggingface.co/facebook/sam3"
            QtWidgets.QMessageBox.warning(
                self, "warning", message
            )
            return
        # -----

        button = self.sender()
        button.setText("downloading")
        info_dict = model_dict.get(model_name, {})
        urls = info_dict.get("urls", None)

        if model_name in self.download_thread_dict:
            download_thread = self.download_thread_dict[model_name]
        else:
            download_thread = DownloadThread(self)
            download_thread.failed.connect(lambda message: self.mainwindow.statusbar.showMessage(message, 10000))
            self.download_thread_dict[model_name] = download_thread

        download_thread.setNameAndUrl(model_name, urls, model_dict.get(model_name, {}).get("sha256"))
        download_thread.tag.connect(partial(self.download_process, button, model_name))
        download_thread.start()

        button.setEnabled(False)
        button.clicked.disconnect()
        button.clicked.connect(partial(self.pause, model_name))
        button.setEnabled(True)

    def download_process(
        self, button: QtWidgets.QPushButton, model_name, downloaded_size, total_size
    ):
        if downloaded_size == -1 and total_size == -1:
            button.setText("continue")
            button.setEnabled(False)
            button.clicked.disconnect()
            button.clicked.connect(partial(self.download, model_name))
            button.setEnabled(True)

        elif downloaded_size == -2 and total_size == -2:
            button.setText("delete")
            button.setStyleSheet("QWidget {color: %s}" % "red")
            button.setEnabled(False)
            button.clicked.disconnect()
            button.clicked.connect(partial(self.delete, model_name))
            button.setEnabled(True)
            # self.mainwindow.update_menuSAM()
            self.update_ui()
        else:
            button.setText(
                "{:.2f}% - {}/{}M".format(
                    downloaded_size / total_size * 100,
                    downloaded_size // 1000000,
                    total_size // 1000000,
                )
            )

    def pause(self, model_name):
        download_thread: DownloadThread = self.download_thread_dict[model_name]
        download_thread.pause = True

    def delete(self, model_name):
        button = self.sender()
        try:
            if os.path.exists(os.path.join(CHECKPOINT_PATH, model_name)):
                os.remove(os.path.join(CHECKPOINT_PATH, model_name))
            button.setText("download")
            button.setStyleSheet("QWidget {color: %s}" % "green")
            button.setEnabled(False)
            button.clicked.disconnect()
            button.clicked.connect(partial(self.download, model_name))
            button.setEnabled(True)
        except Exception as e:
            print(
                "Error when remove {}, {}".format(
                    os.path.join(CHECKPOINT_PATH, model_name), e
                )
            )
        self.update_ui()

    def clear_tmp(self):
        remove_list = []
        temporary_root = os.path.join(CHECKPOINT_PATH, "tmp")
        if not os.path.isdir(temporary_root):
            return
        for name in os.listdir(temporary_root):
            tmp_file = os.path.join(CHECKPOINT_PATH, "tmp", name)
            if name in self.download_thread_dict:
                thread = self.download_thread_dict.get(name, None)
                if thread is not None:
                    if thread.isRunning():
                        continue
            try:
                os.remove(tmp_file)
                remove_list.append(tmp_file)

            except Exception as e:
                print(
                    "Error when remove {}, {}".format(
                        os.path.join(CHECKPOINT_PATH, name), e
                    )
                )
        QtWidgets.QMessageBox.information(
            self, "clear tmp", "Remove tmp: [" + "],[".join(remove_list) + "]"
        )

    @staticmethod
    def open_checkpoint_dir(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(CHECKPOINT_PATH))

    def load_fine_tuned_model(self):
        path, suffix = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open file", filter="Checkpoint (*.pt *.pth *.pkl);;All files (*)"
        )
        if path:
            self.lineEdit_ft_model.setText(path)
            self.radioButton_ft_model.click()

    def use_fine_tune_model(self):
        model_path = self.lineEdit_ft_model.text()
        if model_path == "":
            QtWidgets.QMessageBox.warning(self, "Warning", "No checkpoint selected.")
            return

        supported_model_prefix = (
            "mobile_sam",
            "sam_hq_vit_h",
            "sam_hq_vit_l",
            "sam_hq_vit_b",
            "sam_hq_vit_tiny",
            "sam_vit_h",
            "sam_vit_l",
            "sam_vit_b",
            "edge_sam",
            "sam-med2d_b",
            "sam2_hiera_large",
            "sam2_hiera_base_plus",
            "sam2_hiera_small",
            "sam2_hiera_tiny",
            "sam2.1_hiera_large",
            "sam2.1_hiera_base_plus",
            "sam2.1_hiera_small",
            "sam2.1_hiera_tiny",
            "sam3.1_multiplex",
            "sam3"
        )

        if model_path and os.path.basename(model_path).startswith(
            supported_model_prefix
        ):
            self.mainwindow.init_segment_anything(model_path)
            return True
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "warning",
                "Checkpoint's name must start with: \n    {}".format(
                    "\n    ".join(supported_model_prefix)
                ),
            )
        return False
