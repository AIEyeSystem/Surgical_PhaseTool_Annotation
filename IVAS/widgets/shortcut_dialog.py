# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import functools

from PyQt5 import QtCore, QtGui, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog

from IVAS.ui.shortcut_dialog import Ui_Dialog


class ShortcutDialog(ResponsiveDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ShortcutDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.pushButton_reset.clicked.connect(self.reset_shortcut)
        self.pushButton_close.clicked.connect(self.close)

        self.columns = 3  # action列数
        self.column_columns = 3  # 每个action有多少列内容[icon, label, QKeySequenceEdit]

        # 可以自定义快捷键的action列表
        self.actions_list = [
            self.mainwindow.actionImages_dir,
            self.mainwindow.actionLabel_dir,
            self.mainwindow.actionPrev_image,
            self.mainwindow.actionNext_image,
            self.mainwindow.actionSetting,
            self.mainwindow.actionExit,
            self.mainwindow.actionSegment_anything_point,
            self.mainwindow.actionSegment_anything_box,
            self.mainwindow.actionPolygon,
            self.mainwindow.actionRepaint,
            self.mainwindow.actionVideo_segment,
            self.mainwindow.actionVideo_segment_once,
            self.mainwindow.actionVideo_segment_five_times,
            self.mainwindow.actionBackspace,
            self.mainwindow.actionFinish,
            self.mainwindow.actionCancel,
            self.mainwindow.actionEdit,
            self.mainwindow.actionDelete,
            self.mainwindow.actionSave,
            self.mainwindow.actionCopy,
            self.mainwindow.actionTo_top,
            self.mainwindow.actionTo_bottom,
            self.mainwindow.actionUnion,
            self.mainwindow.actionSubtract,
            self.mainwindow.actionIntersect,
            self.mainwindow.actionExclude,
            self.mainwindow.actionPrev_group,
            self.mainwindow.actionNext_group,
            self.mainwindow.actionVisible,
            self.mainwindow.actionBit_map,
            self.mainwindow.actionZoom_in,
            self.mainwindow.actionZoom_out,
            self.mainwindow.actionFit_window,
            self.mainwindow.actionScene_shot,
            self.mainwindow.actionWindow_shot,
            # self.mainwindow.actionModel_manage,
            # self.mainwindow.actionConverter,
            # self.mainwindow.actionVideo_to_frames,
            # self.mainwindow.actionAuto_segment_with_bounding_box,
            # self.mainwindow.actionAnno_validator,
            self.mainwindow.actionLanguage,
            # self.mainwindow.actionShortcut,
            # self.mainwindow.actionAbout,
        ]
        self.update_ui()

    def update_ui(self):
        while self.gridLayout.count():
            self.gridLayout.takeAt(0).widget().deleteLater()
        self._shortcut_rows = []

        for index, action in enumerate(self.actions_list):
            icon = QtWidgets.QLabel()
            icon.setFixedSize(30, 30)
            icon.setPixmap(QtGui.QPixmap(action.icon().pixmap(QtCore.QSize(24, 24))))
            label = QtWidgets.QLabel()
            label.setText(action.text())
            key_edit = QtWidgets.QKeySequenceEdit()
            key_edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
            key_edit.setMinimumWidth(150)
            key_edit.setKeySequence(action.shortcut())
            key_edit.editingFinished.connect(
                functools.partial(self.shortcut_change_finish, action)
            )
            self._shortcut_rows.append((icon, label, key_edit))
        self._reflow_shortcuts()

    def _reflow_shortcuts(self):
        rows = getattr(self, "_shortcut_rows", ())
        if not rows:
            return
        for row in rows:
            for widget in row:
                widget.ensurePolished()
        entry_width = max(sum(max(w.minimumWidth(), w.sizeHint().width()) for w in row)
                          for row in rows) + 24
        self.columns = max(1, min(3, (self.width() - 48) // entry_width))
        for index, row in enumerate(rows):
            for offset, widget in enumerate(row):
                self.gridLayout.addWidget(widget, index // self.columns,
                                          index % self.columns * self.column_columns + offset)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow_shortcuts()

    def reset_shortcut(self):
        self.mainwindow.load_actions_shortcut(default=True)
        self.update_ui()

    def shortcut_change_finish(self, action):
        ks = self.sender().keySequence()

        if ks.toString() in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            self.sender().setKeySequence(action.shortcut())
        else:
            if ks.toString() == "Backspace":
                ks = QtGui.QKeySequence(0)

            action.setShortcut(ks)  # 设置快捷键
            self.sender().setKeySequence(action.shortcut())  # 同步显示
            print("New shortcut [{}] for {}".format(ks.toString(), action.text()))

            for action_str in self.mainwindow.cfg["shortcut"]:
                try:
                    if eval("self.mainwindow." + action_str) == action:
                        self.mainwindow.cfg["shortcut"][action_str][
                            "key"
                        ] = ks.toString()
                        break
                except Exception:
                    pass

            self.sender().clearFocus()

            self.mainwindow.save_software_cfg()
