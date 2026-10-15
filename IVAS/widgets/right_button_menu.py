# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

from PyQt5 import QtWidgets


class RightButtonMenu(QtWidgets.QMenu):
    def __init__(self, mainwindow):
        super(RightButtonMenu, self).__init__()
        self.mainwindow = mainwindow
