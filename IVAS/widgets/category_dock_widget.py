# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import functools

from PyQt5 import QtCore, QtGui, QtWidgets

from IVAS.configs import STATUSMode
from IVAS.ui.category_dock import Ui_Form
from IVAS.widgets.list_item import fit_list_item


class CategoriesDockWidget(QtWidgets.QWidget, Ui_Form):
    categoryChanged = QtCore.pyqtSignal(str)

    def __init__(self, mainwindow):
        super(CategoriesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.pushButton_category_setting.clicked.connect(
            self.mainwindow.category_setting
        )
        self.listWidget.itemClicked.connect(self.item_choice)
        self.listWidget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.lineEdit_search_category.textChanged.connect(self.update_widget)
        self.lineEdit_search_category.setClearButtonEnabled(True)
        self.button_group = None
        self.pushButton_segment = QtWidgets.QPushButton("Segment", self)
        self.pushButton_segment.setObjectName("primaryAction")
        self.pushButton_segment.setToolTip("Segment the selected category with its category name.")
        self.pushButton_change = QtWidgets.QPushButton("Change", self)
        self.pushButton_change.setObjectName("changeCategory")
        self.pushButton_change.setEnabled(False)
        self.category_actions = QtWidgets.QHBoxLayout()
        self.category_actions.addWidget(self.pushButton_segment, 1)
        self.category_actions.addWidget(self.pushButton_change, 1)
        self.verticalLayout.insertLayout(2, self.category_actions)
        self.batch_actions = QtWidgets.QHBoxLayout()
        self.pushButton_segall = QtWidgets.QPushButton("SegAll", self)
        self.pushButton_segall.setObjectName("segmentAll")
        self.pushButton_segmiss = QtWidgets.QPushButton("SegMiss", self)
        self.pushButton_segmiss.setObjectName("segmentMissing")
        self.batch_actions.addWidget(self.pushButton_segall, 1)
        self.batch_actions.addWidget(self.pushButton_segmiss, 1)
        self.verticalLayout.insertLayout(3, self.batch_actions)
        self.pushButton_segment.clicked.connect(self.segment_selected_category)
        self.pushButton_segall.clicked.connect(self.segment_all)
        self.pushButton_segmiss.clicked.connect(self.segment_missing)
        self.pushButton_change.clicked.connect(self.change_selected_category)
        self.mainwindow.scene.selectionChanged.connect(self.sync_from_selection)
        self.mainwindow.actionEdit.changed.connect(self.update_change_enabled)
        for button in (self.pushButton_group_mode, self.pushButton_increase,
                       self.pushButton_decrease):
            button.setMaximumWidth(16777215)
            button.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)

        # 新增 手动/自动 group 选择
        self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))
        self.lineEdit_currentGroup.textChanged.connect(self.update_current_group)
        self.pushButton_increase.clicked.connect(self.increase_current_group)
        self.pushButton_decrease.clicked.connect(self.decrease_current_group)
        self.pushButton_group_mode.clicked.connect(self.toggle_group_mode)

        self.category_choice_shortcuts = {}
        for i in range(10):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("{}".format(i)), self)
            shortcut.activated.connect(self.choice_category)
            self.category_choice_shortcuts[shortcut] = i - 1 if i != 0 else 9

    def segment_selected_category(self):
        self.mainwindow.text_prompt_dock_widget.predict_for_selected_category()

    def segment_all(self):
        self.mainwindow.text_prompt_dock_widget.predict_for_folder(missing_only=False)

    def segment_missing(self):
        self.mainwindow.text_prompt_dock_widget.predict_for_folder(missing_only=True)

    def selected_polygons(self):
        return [item for item in self.mainwindow.scene.selectedItems()
                if item in self.mainwindow.polygons and not item.is_drawing]

    def sync_from_selection(self):
        """Follow canvas/Annos selection without editing any annotation."""
        selected = self.selected_polygons()
        if selected:
            polygon = next((item for item in reversed(self.mainwindow.scene.selected_polygons_list)
                            if item in selected), selected[0])
            self.select_category(polygon.category)
        self.update_change_enabled()

    def select_category(self, category):
        previous = self.mainwindow.current_category
        self.mainwindow.current_category = category
        if (self.lineEdit_search_category.text()
                and not any(self.listWidget.item(row).data(QtCore.Qt.UserRole) == category
                            for row in range(self.listWidget.count()))):
            self.lineEdit_search_category.clear()
        chosen = None
        if self.button_group is not None:
            self.button_group.setExclusive(False)
        for row in range(self.listWidget.count()):
            item = self.listWidget.item(row)
            radio = self.listWidget.itemWidget(item).findChild(QtWidgets.QRadioButton, "label_radio")
            matches = item.data(QtCore.Qt.UserRole) == category
            with QtCore.QSignalBlocker(radio):
                radio.setChecked(matches)
            if matches:
                chosen = item
        if self.button_group is not None:
            self.button_group.setExclusive(True)
        self.listWidget.setCurrentItem(chosen)
        if chosen is not None:
            self.listWidget.scrollToItem(chosen)
        else:
            self.listWidget.clearSelection()
        if previous != category:
            self.categoryChanged.emit(category)
        self.update_change_enabled()

    def update_change_enabled(self):
        window = self.mainwindow
        category = window.current_category
        enabled = (window.load_finished and window.scene._frame_is_editable()
                   and window.scene.mode in (STATUSMode.VIEW, STATUSMode.EDIT)
                   and category in window.category_color_dict
                   and any(polygon.category != category for polygon in self.selected_polygons()))
        self.pushButton_change.setEnabled(bool(enabled))

    def change_selected_category(self):
        """Apply the chosen canonical category only after an explicit click."""
        self.update_change_enabled()
        if not self.pushButton_change.isEnabled():
            return
        window = self.mainwindow
        category = window.current_category
        for polygon in self.selected_polygons():
            if polygon.category == category:
                continue
            polygon.set_drawed(category, polygon.group, polygon.iscrowd, polygon.note,
                               QtGui.QColor(window.category_color_dict[category]))
            polygon.mark_edited(notify=False)
            # Preserve the canvas selection highlight while updating its base color.
            highlight = QtGui.QColor("#00A0FF")
            highlight.setAlpha(polygon.hover_alpha)
            polygon.setBrush(highlight)
            window.annos_dock_widget.update_polygon_category(polygon)
        if window.current_label is not None:
            window.current_label.objects = [polygon.to_object() for polygon in window.polygons]
        # A multi-selection is one edit, including when auto-save is enabled.
        window.set_saved_state(False)
        self.update_change_enabled()

    def choice_category(self):
        """Shortcut function for category choice."""
        index = self.category_choice_shortcuts.get(self.sender(), 0)
        try:
            item = self.listWidget.item(index)
            widget = self.listWidget.itemWidget(item)
            label_radio = widget.findChild(QtWidgets.QRadioButton, "label_radio")
            label_radio.setChecked(True)
        except:
            pass

    def update_widget(self):
        """Update list widget."""
        self.listWidget.clear()
        if self.button_group is not None:
            self.button_group.deleteLater()
        self.button_group = QtWidgets.QButtonGroup(self)
        labels = self.mainwindow.cfg.get("label", [])
        search_text = self.lineEdit_search_category.text()

        name_label_dict = {label.get("name", "UNKNOW"): label for label in labels}

        label_names = [label.get("name", "UNKNOW") for label in labels]
        if search_text == "":
            show_label_names = label_names
        elif search_text.strip(" ") == "":
            show_label_names = label_names
        else:
            # Fuzzy matching pulls RapidFuzz/Levenshtein into the process.
            # Keep it off the first-paint path until a search is requested.
            from fuzzywuzzy import process

            display_names = {self.mainwindow.category_display_name(name): name
                             for name in label_names}
            matches = process.extract(search_text, list(display_names), limit=5)
            show_label_names = [display_names[name] for name, score in matches if score > 0]

        for index in range(len(show_label_names)):
            label = name_label_dict[show_label_names[index]]
            name = label.get("name", "UNKNOW")
            color = label.get("color", "#000000")
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, name)
            widget = QtWidgets.QWidget(self.listWidget)

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName("label_color")

            label_radio = QtWidgets.QRadioButton(self.mainwindow.category_display_name(name))
            label_radio.setProperty("category_name", name)
            label_radio.setToolTip(label_radio.text())
            label_radio.setObjectName("label_radio")
            label_radio.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.button_group.addButton(label_radio)
            if name == self.mainwindow.current_category:
                label_radio.setChecked(True)

            layout.addWidget(label_color)
            layout.addWidget(label_radio)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)
            fit_list_item(item, widget)
            label_radio.toggled.connect(functools.partial(self.radio_choice, index))
            if name == self.mainwindow.current_category:
                self.listWidget.setCurrentItem(item)
        self.update_change_enabled()

    def radio_choice(self, index: int):
        """
        Triggered when radio button selected and set current category.

        Arguments:
            index: Index of list widget item.
        """
        if isinstance(self.sender(), QtWidgets.QRadioButton):
            if self.sender().isChecked():
                self.mainwindow.current_category = self.sender().property("category_name")
                self.listWidget.setCurrentRow(index)
                self.categoryChanged.emit(self.mainwindow.current_category)
                self.update_change_enabled()

    def item_choice(self, item: QtWidgets.QListWidgetItem):
        """
        Triggered when an item in the category list is clicked.Will trigger radio button.

        Arguments:
            item: Item to be checked.
        """
        widget = self.listWidget.itemWidget(item)
        label_radio = widget.findChild(QtWidgets.QRadioButton, "label_radio")
        label_radio.setChecked(True)

    def update_current_group(self, text):
        """
        Update the current_group variable when the text in the QLineEdit changes.

        Arguments:
            text (str): The text of QLineEdit.
        """
        try:
            self.mainwindow.current_group = int(text)
        except ValueError:
            pass

    def increase_current_group(self):
        """Increase the current_group variable and update the QLineEdit text"""
        self.mainwindow.current_group += 1
        self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))

    def decrease_current_group(self):
        """Decrease the current_group variable and update the QLineEdit text"""
        if self.mainwindow.current_group > 1:
            self.mainwindow.current_group -= 1
            self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))

    def toggle_group_mode(self):
        """
        Toggle group mode.

        - auto: Group id auto add 1 when add a new polygon.
        - manual: Manual set group id.
        - track: Group id changed with the group of current polygons when use [TAB] or [`] to check.
        """
        _translate = QtCore.QCoreApplication.translate
        if self.mainwindow.group_select_mode == "auto":
            self.mainwindow.group_select_mode = "manual"
            self.pushButton_group_mode.setText("Manual")
            self.pushButton_group_mode.setStatusTip(
                _translate("MainWindow", "Manual set group id.")
            )
        elif self.mainwindow.group_select_mode == "manual":
            self.mainwindow.group_select_mode = "track"
            self.pushButton_group_mode.setText("Track")
            self.pushButton_group_mode.setStatusTip(
                _translate(
                    "MainWindow",
                    "Group id changed with the group of current polygons when use [TAB] or [`] to check.",
                )
            )
        elif self.mainwindow.group_select_mode == "track":
            self.mainwindow.group_select_mode = "auto"
            self.pushButton_group_mode.setText("Auto")
            self.pushButton_group_mode.setStatusTip(
                _translate("MainWindow", "Group id auto add 1 when add a new polygon.")
            )
