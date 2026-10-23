# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import os.path

import requests
from PyQt5 import QtCore, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog
from IVAS.widgets.background import TaskWorker
from PyQt5.QtGui import QValidator

from IVAS.configs import CHECKPOINT_PATH
from IVAS.ui.remote_sam_dialog import Ui_Dialog


class RemoteSamDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(RemoteSamDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)

        self.remote_sam_model_name = None
        self._worker = None
        self._close_when_finished = False
        self._checked_endpoint = None

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.checkBox_use_remote.setEnabled(False)
        self.frame_info.setVisible(False)
        self.lineEdit_host.setValidator(IPv4Validator())

        self.pushButton_check.clicked.connect(self.check_connection)
        self.checkBox_use_remote.stateChanged.connect(self.use_remote_sam)
        self.pushButton_close.clicked.connect(self.close)
        self.lineEdit_host.textChanged.connect(self._reset_connection)
        self.lineEdit_port.textChanged.connect(self._reset_connection)

    def _reset_connection(self):
        self._checked_endpoint = None
        self.checkBox_use_remote.setChecked(False)
        self.checkBox_use_remote.setEnabled(False)
        self.frame_info.setVisible(False)
        self.remote_sam_model_name = None

    def check_connection(self):
        if self._worker is not None:
            return
        self._reset_connection()
        host, port = self.lineEdit_host.text().strip(), self.lineEdit_port.text().strip()
        if not host or not port.isdigit() or not 0 < int(port) < 65536:
            QtWidgets.QMessageBox.warning(self, "Error", "Enter a host and a port between 1 and 65535")
            return
        self._close_when_finished = False
        self.pushButton_check.setEnabled(False)
        self.lineEdit_host.setEnabled(False)
        self.lineEdit_port.setEnabled(False)
        self._checked_endpoint = (host, port)

        def check(cancelled, progress):
            with requests.get("http://{}:{}/api/info".format(host, port), timeout=(3, 10)) as response:
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or any(not isinstance(data.get(key), str) or not data[key]
                                                     for key in ("checkpoint", "device", "dtype")):
                    raise ValueError("Server response is missing checkpoint, device or dtype")
                if cancelled.is_set():
                    return None
                return data
        self._worker = TaskWorker(check)
        self._worker.result.connect(self._connection_result)
        self._worker.finished.connect(self._connection_finished)
        self._worker.start()

    def _connection_result(self, data, error):
        if (self._close_when_finished or getattr(self.mainwindow, "_closing_requested", False)
                or self._checked_endpoint != (self.lineEdit_host.text().strip(), self.lineEdit_port.text().strip())):
            return
        if error:
            self._reset_connection()
            QtWidgets.QMessageBox.warning(self, "Error", error)
        elif data is not None:
            model_name = data["checkpoint"].replace("\\", "/").split("/")[-1]
            self.remote_sam_model_name = model_name
            self.label_name.setText(model_name)
            self.label_device.setText(data["device"])
            self.label_dtype.setText(data["dtype"])
            self.frame_info.setVisible(True)
            self.checkBox_use_remote.setEnabled(True)

    def _connection_finished(self):
        self._worker = None
        self.pushButton_check.setEnabled(True)
        self.lineEdit_host.setEnabled(True)
        self.lineEdit_port.setEnabled(True)
        if self._close_when_finished:
            self.close()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._close_when_finished = True
            self._worker.cancel()
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        self.close()

    def use_remote_sam(self, check_state):
        if check_state == QtCore.Qt.CheckState.Checked:
            model_name = self.label_name.text()
            loadl_model_path = os.path.join(CHECKPOINT_PATH, model_name)
            if not os.path.exists(loadl_model_path):
                self.mainwindow.use_remote_sam = False
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Local model {model_name} does not exist."
                )
            else:
                try:
                    self.mainwindow.use_remote_sam = True
                    self.mainwindow.init_segment_anything(loadl_model_path)
                except Exception as e:
                    self.mainwindow.use_remote_sam = False
                    QtWidgets.QMessageBox.warning(
                        self, "Error", f"Init local sam failed.\n {e}"
                    )

        else:
            self.mainwindow.use_remote_sam = False

        state = (
            QtCore.Qt.CheckState.Checked
            if self.mainwindow.use_remote_sam
            else QtCore.Qt.CheckState.Unchecked
        )
        self.checkBox_use_remote.blockSignals(True)
        self.checkBox_use_remote.setCheckState(state)
        self.checkBox_use_remote.blockSignals(False)
        self.lineEdit_host.setEnabled(not self.mainwindow.use_remote_sam)
        self.lineEdit_port.setEnabled(not self.mainwindow.use_remote_sam)
        self.pushButton_check.setEnabled(not self.mainwindow.use_remote_sam)


class IPv4Validator(QValidator):
    def validate(self, input_str, pos):
        parts = input_str.split(".")
        if len(parts) > 4:
            # 返回 Invalid + 原始输入 + 光标位置
            return (QValidator.Invalid, input_str, pos)

        for i, part in enumerate(parts):
            if i < len(parts) - 1 and not part:
                return (QValidator.Invalid, input_str, pos)
            if part:
                if not part.isdigit() or len(part) > 3:
                    return (QValidator.Invalid, input_str, pos)
                num = int(part)
                if num < 0 or num > 255:
                    return (QValidator.Invalid, input_str, pos)

        if len(parts) == 4:
            if not parts[3]:
                # 允许中间状态（如 192.168.1.）
                return (QValidator.Intermediate, input_str, pos)
            elif 0 <= int(parts[3]) <= 255:
                return (QValidator.Acceptable, input_str, pos)
            else:
                return (QValidator.Invalid, input_str, pos)
        else:
            # 允许继续输入（如 192.168）
            return (QValidator.Intermediate, input_str, pos)
