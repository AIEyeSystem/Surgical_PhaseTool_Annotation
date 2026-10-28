# -*- coding: utf-8 -*-
"""Batch category operations on the open label directory, with history and undo.

The dialog is opened from the Files dock. File work runs in a background
thread (see :func:`start_task`); the pure logic lives in :mod:`IVAS.batch_ops`.
"""

import os
from contextlib import nullcontext

from PyQt5 import QtCore, QtWidgets

from IVAS import batch_ops
from IVAS.widgets.responsive import ResponsiveDialog

# --------------------------------------------------------------------- strings
_STRINGS = {
    # Files dock
    "search_placeholder": ("Filter by file name (* ? wildcards)", "按文件名筛选（支持 * ? 通配符）"),
    "search_tip": ("Show only images whose file name contains this text.",
                   "只显示文件名包含该文本的图像。"),
    "batch_button": ("Batch…", "批量…"),
    "batch_tip": ("Batch delete or rename categories, with history and undo.",
                  "批量删除或重命名类别，可查看记录并撤销。"),
    "mode_has": ("has", "包含"),
    "mode_lacks": ("lacks", "不含"),
    "mode_tip": ("Show images that have / lack the selected category.",
                 "显示包含 / 不含所选类别的图像。"),
    "category_tip": ("Filter images by annotated category. Numbers are image counts.",
                     "按标注类别筛选图像，括号内为图像数量。"),
    "filter_all": ("All categories", "全部类别"),
    "filter_any": ("Any annotation", "任意标注"),
    "filter_unreadable": ("Unreadable labels", "无法读取的标注"),
    "count": ("{visible} / {total} images", "{visible} / {total} 张图像"),
    "count_scanning": ("{visible} / {total} images · scanning…", "{visible} / {total} 张图像 · 扫描中…"),
    "count_tip_errors": ("{n} label file(s) could not be read; choose \"Unreadable labels\" to list them.",
                         "{n} 个标注文件无法读取；选择“无法读取的标注”可列出它们。"),
    "row_no_label": ("No label file", "无标注文件"),
    "row_empty": ("No annotations", "没有标注"),
    "row_unreadable": ("Unreadable label: {error}", "标注无法读取：{error}"),
    # Dialog
    "title": ("Batch operations", "批量操作"),
    "tab_operation": ("Operation", "操作"),
    "tab_history": ("History", "记录"),
    "operation": ("Operation", "操作"),
    "op_delete": ("Delete a category", "删除类别"),
    "op_rename": ("Rename / merge a category", "重命名 / 合并类别"),
    "op_empty": ("Remove empty label files", "删除空标注文件"),
    "category": ("Category", "类别"),
    "target": ("New name", "新名称"),
    "target_tip": ("Type a new name, or pick an existing category to merge into it.",
                   "输入新名称，或选择已有类别以合并。"),
    "remove_empty": ("Also remove label files left without objects", "同时删除变为空的标注文件"),
    "remove_empty_tip": ("Off keeps the empty JSON (image size and note stay on disk). "
                         "Files with an image note are always kept.",
                         "关闭时保留空 JSON（保留图像尺寸与备注）；带图像备注的文件始终保留。"),
    "scope": ("Scope", "范围"),
    "scope_all": ("All images ({n})", "全部图像（{n}）"),
    "scope_filtered": ("Filtered images ({n})", "筛选结果（{n}）"),
    "scope_current": ("Current image ({name})", "当前图像（{name}）"),
    "scope_current_none": ("Current image (none)", "当前图像（无）"),
    "preview": ("Preview", "预览"),
    "run": ("Run…", "执行…"),
    "close": ("Close", "关闭"),
    "refresh": ("Refresh", "刷新"),
    "undo_last": ("Undo last", "撤销上一次"),
    "working": ("Working…", "处理中…"),
    "working_progress": ("Working… {percent}%", "处理中… {percent}%"),
    "preview_hint": ("Choose an operation and press Preview to count the affected images.",
                     "选择操作后点击“预览”以统计受影响的图像。"),
    "preview_result": ("{images} image(s) and {objects} object(s) would change.",
                       "将修改 {images} 张图像中的 {objects} 个目标。"),
    "preview_result_files": ("{images} label file(s) would be removed.", "将删除 {images} 个标注文件。"),
    "preview_deleted": ("{n} label file(s) would be removed.", "其中 {n} 个标注文件将被删除。"),
    "merge_note": ("\"{name}\" already exists; objects will be merged into it.",
                   "“{name}” 已存在，目标将合并到该类别。"),
    "errors_note": ("{n} label file(s) were skipped because they could not be read:",
                    "{n} 个标注文件无法读取，已跳过："),
    "nothing": ("Nothing to change in this scope.", "该范围内没有需要修改的内容。"),
    "confirm_title": ("Confirm batch operation", "确认批量操作"),
    "confirm_text": ("{description}\n\nScope: {scope}\n{summary}\n\nOriginal files are backed up to "
                     "{backup}; you can undo this from the History tab.",
                     "{description}\n\n范围：{scope}\n{summary}\n\n原文件将备份到 {backup}，"
                     "可在“记录”页撤销。"),
    "done": ("Done: {images} image(s), {objects} object(s) changed.",
             "完成：修改了 {images} 张图像中的 {objects} 个目标。"),
    "done_files": ("Done: {images} label file(s) removed.", "完成：删除了 {images} 个标注文件。"),
    "done_partial": ("Some files could not be changed:", "部分文件未能修改："),
    "failed_title": ("Batch operation failed", "批量操作失败"),
    "need_category": ("Choose a category first.", "请先选择类别。"),
    "need_target": ("Enter the new category name.", "请输入新的类别名称。"),
    "same_target": ("The new name equals the current name.", "新名称与原名称相同。"),
    "no_folder": ("Open an image folder first.", "请先打开图像文件夹。"),
    "no_current": ("No image is open.", "当前没有打开的图像。"),
    "undo_none": ("There is no batch operation to undo.", "没有可以撤销的批量操作。"),
    "undo_confirm_title": ("Undo batch operation", "撤销批量操作"),
    "undo_confirm": ("Undo \"{description}\" from {time}?\n\nThis restores {n} label file(s) from "
                     "its backup.",
                     "撤销 {time} 的“{description}”？\n\n将从备份恢复 {n} 个标注文件。"),
    "undo_refused_title": ("Undo refused", "无法撤销"),
    "undo_done": ("Undone: {n} label file(s) restored.", "已撤销：恢复了 {n} 个标注文件。"),
    "history_hint": ("Records are kept in {path}", "记录保存在 {path}"),
    "col_time": ("Time", "时间"),
    "col_operation": ("Operation", "操作"),
    "col_images": ("Images", "图像"),
    "col_objects": ("Objects", "目标"),
    "col_status": ("Status", "状态"),
    "status_done": ("Done", "完成"),
    "status_partial": ("Partial", "部分完成"),
    "status_undone": ("Undone", "已撤销"),
    "desc_delete": ("Delete category \"{category}\"", "删除类别“{category}”"),
    "desc_delete_empty": (" and remove emptied files", "并删除变空的文件"),
    "desc_rename": ("Rename \"{old}\" → \"{new}\"", "重命名“{old}” → “{new}”"),
    "desc_empty": ("Remove empty label files", "删除空标注文件"),
    "desc_undo": ("Undo: {target}", "撤销：{target}"),
    "scope_name_all": ("all images", "全部图像"),
    "scope_name_filtered": ("filtered images", "筛选结果"),
    "scope_name_current": ("current image", "当前图像"),
    "tip_record": ("Id: {id}\nScope: {scope}\nBackup: {backup}\nFiles:\n{files}",
                   "编号：{id}\n范围：{scope}\n备份：{backup}\n文件：\n{files}"),
}


def tr(key, language="en", **values):
    """Return the ``language`` ('zh'/'en') text for ``key`` formatted with ``values``."""
    english, chinese = _STRINGS[key]
    text = chinese if language == "zh" else english
    return text.format(**values) if values else text


def describe_record(record, language="en"):
    """One-line human description of a history record."""
    op = record.get("op")
    params = record.get("params") or {}
    if op == batch_ops.OP_DELETE_CATEGORY:
        text = tr("desc_delete", language, category=params.get("category", ""))
        if params.get("remove_empty_files"):
            text += tr("desc_delete_empty", language)
        return text
    if op == batch_ops.OP_RENAME_CATEGORY:
        return tr("desc_rename", language, old=params.get("old", ""), new=params.get("new", ""))
    if op == batch_ops.OP_DELETE_EMPTY:
        return tr("desc_empty", language)
    if op == batch_ops.OP_UNDO:
        target = {"op": params.get("target_op"), "params": params.get("target_params") or {}}
        return tr("desc_undo", language, target=describe_record(target, language))
    return str(op)


def _scope_name(scope, language):
    key = "scope_name_{}".format(scope)
    return tr(key, language) if key in _STRINGS else str(scope or "")


# ------------------------------------------------------------ background work
class BackgroundTask(QtCore.QThread):
    """Run ``function`` off the UI thread; report its result or exception."""

    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)

    def __init__(self, function):
        super().__init__()
        self._function = function

    def run(self):
        try:
            result = self._function()
        except Exception as error:  # reported to the UI thread
            self.failed.emit(error)
        else:
            self.succeeded.emit(result)


_ACTIVE_TASKS = set()


def _forget_task(task):
    _ACTIVE_TASKS.discard(task)


def start_task(function, on_success, on_failure=None):
    """Start ``function`` in a :class:`BackgroundTask` kept alive until it finishes.

    Connect bound methods of QObjects as callbacks: Qt drops queued results
    whose receiver was deleted, so a closing window never sees late results.
    """
    _install_shutdown_hook()
    task = BackgroundTask(function)
    _ACTIVE_TASKS.add(task)
    task.succeeded.connect(on_success)
    if on_failure is not None:
        task.failed.connect(on_failure)
    task.finished.connect(lambda task=task: _forget_task(task))
    task.finished.connect(task.deleteLater)
    task.start()
    return task


def wait_for_background_tasks(timeout_ms=10000):
    """Block until every running task finished (used at shutdown and in tests)."""
    for task in list(_ACTIVE_TASKS):
        try:
            task.wait(timeout_ms)
        except RuntimeError:  # already deleted
            pass


def _install_shutdown_hook():
    app = QtWidgets.QApplication.instance()
    if app is not None and not app.property("ivasBatchShutdownHook"):
        app.setProperty("ivasBatchShutdownHook", True)
        app.aboutToQuit.connect(wait_for_background_tasks)


# ---------------------------------------------------------------------- dialog
class BatchOperationsDialog(ResponsiveDialog):
    """Delete / rename-merge categories across images, with preview, history and undo."""

    # Emitted from worker threads; delivered queued to the dialog.
    progressed = QtCore.pyqtSignal(int, int)

    OPERATIONS = (batch_ops.OP_DELETE_CATEGORY, batch_ops.OP_RENAME_CATEGORY,
                  batch_ops.OP_DELETE_EMPTY)
    SCOPES = ("all", "filtered", "current")

    def __init__(self, files_dock, mainwindow, parent=None):
        super().__init__(parent or mainwindow)
        self.files_dock = files_dock
        self.mainwindow = mainwindow
        self._busy = False
        self._language = "en"
        self._category_keys = []
        self._pending = None
        self._mutating = False
        self.setObjectName("batchOperationsDialog")
        self.setModal(True)
        self.resize(560, 460)
        self._build_ui()
        self.progressed.connect(self._on_progress)
        self.retranslate(files_dock.language())

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.setObjectName("batchTabs")
        layout.addWidget(self.tabWidget, 1)

        # Operation tab
        page = QtWidgets.QWidget()
        form = QtWidgets.QGridLayout(page)
        form.setColumnStretch(1, 1)
        self.label_operation = QtWidgets.QLabel(page)
        self.comboBox_operation = QtWidgets.QComboBox(page)
        self.comboBox_operation.setObjectName("comboBox_batch_operation")
        self.label_category = QtWidgets.QLabel(page)
        self.comboBox_category = QtWidgets.QComboBox(page)
        self.comboBox_category.setObjectName("comboBox_batch_category")
        self.label_target = QtWidgets.QLabel(page)
        self.comboBox_target = QtWidgets.QComboBox(page)
        self.comboBox_target.setObjectName("comboBox_batch_target")
        self.comboBox_target.setEditable(True)
        self.comboBox_target.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        for combo in (self.comboBox_operation, self.comboBox_category, self.comboBox_target):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self.checkBox_remove_empty = QtWidgets.QCheckBox(page)
        self.checkBox_remove_empty.setChecked(False)

        self.groupBox_scope = QtWidgets.QGroupBox(page)
        scope_layout = QtWidgets.QVBoxLayout(self.groupBox_scope)
        self.radio_scope = {}
        self.scope_buttons = QtWidgets.QButtonGroup(self)
        for scope in self.SCOPES:
            button = QtWidgets.QRadioButton(self.groupBox_scope)
            button.setObjectName("radio_scope_{}".format(scope))
            self.radio_scope[scope] = button
            self.scope_buttons.addButton(button)
            scope_layout.addWidget(button)
        self.radio_scope["all"].setChecked(True)

        self.label_preview = QtWidgets.QLabel(page)
        self.label_preview.setObjectName("label_batch_preview")
        self.label_preview.setWordWrap(True)
        self.label_preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.label_preview.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label_preview.setMinimumHeight(60)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.pushButton_preview = QtWidgets.QPushButton(page)
        self.pushButton_run = QtWidgets.QPushButton(page)
        self.pushButton_run.setObjectName("pushButton_batch_run")
        buttons.addWidget(self.pushButton_preview)
        buttons.addWidget(self.pushButton_run)

        form.addWidget(self.label_operation, 0, 0)
        form.addWidget(self.comboBox_operation, 0, 1)
        form.addWidget(self.label_category, 1, 0)
        form.addWidget(self.comboBox_category, 1, 1)
        form.addWidget(self.label_target, 2, 0)
        form.addWidget(self.comboBox_target, 2, 1)
        form.addWidget(self.checkBox_remove_empty, 3, 0, 1, 2)
        form.addWidget(self.groupBox_scope, 4, 0, 1, 2)
        form.addWidget(self.label_preview, 5, 0, 1, 2)
        form.setRowStretch(5, 1)
        form.addLayout(buttons, 6, 0, 1, 2)
        self.tabWidget.addTab(page, "")

        # History tab
        history = QtWidgets.QWidget()
        history_layout = QtWidgets.QVBoxLayout(history)
        self.treeWidget_history = QtWidgets.QTreeWidget(history)
        self.treeWidget_history.setObjectName("treeWidget_batch_history")
        self.treeWidget_history.setRootIsDecorated(False)
        self.treeWidget_history.setUniformRowHeights(True)
        self.treeWidget_history.setColumnCount(5)
        self.treeWidget_history.setMinimumHeight(120)
        history_layout.addWidget(self.treeWidget_history, 1)
        self.label_history = QtWidgets.QLabel(history)
        self.label_history.setWordWrap(True)
        self.label_history.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        history_layout.addWidget(self.label_history)
        history_buttons = QtWidgets.QHBoxLayout()
        history_buttons.addStretch(1)
        self.pushButton_refresh = QtWidgets.QPushButton(history)
        self.pushButton_undo = QtWidgets.QPushButton(history)
        self.pushButton_undo.setObjectName("pushButton_batch_undo")
        history_buttons.addWidget(self.pushButton_refresh)
        history_buttons.addWidget(self.pushButton_undo)
        history_layout.addLayout(history_buttons)
        self.tabWidget.addTab(history, "")

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, self)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.comboBox_operation.currentIndexChanged.connect(self._sync_controls)
        self.comboBox_operation.currentIndexChanged.connect(self._clear_preview)
        self.comboBox_category.currentIndexChanged.connect(self._clear_preview)
        self.comboBox_target.editTextChanged.connect(self._clear_preview)
        self.checkBox_remove_empty.toggled.connect(self._clear_preview)
        self.scope_buttons.buttonToggled.connect(self._clear_preview)
        self.pushButton_preview.clicked.connect(self.preview)
        self.pushButton_run.clicked.connect(self.run_operation)
        self.pushButton_refresh.clicked.connect(self.refresh_history)
        self.pushButton_undo.clicked.connect(self.undo_last)
        self.tabWidget.currentChanged.connect(
            lambda index: self.refresh_history() if index == 1 else None)

    # --------------------------------------------------------------- language
    def retranslate(self, language):
        self._language = "zh" if language == "zh" else "en"
        lang = self._language
        self.setWindowTitle(tr("title", lang))
        self.tabWidget.setTabText(0, tr("tab_operation", lang))
        self.tabWidget.setTabText(1, tr("tab_history", lang))
        self.label_operation.setText(tr("operation", lang))
        operation = self.comboBox_operation.currentIndex()
        self.comboBox_operation.blockSignals(True)
        self.comboBox_operation.clear()
        self.comboBox_operation.addItems([tr("op_delete", lang), tr("op_rename", lang),
                                          tr("op_empty", lang)])
        self.comboBox_operation.setCurrentIndex(max(operation, 0))
        self.comboBox_operation.blockSignals(False)
        self.label_category.setText(tr("category", lang))
        self.label_target.setText(tr("target", lang))
        self.comboBox_target.setToolTip(tr("target_tip", lang))
        self.checkBox_remove_empty.setText(tr("remove_empty", lang))
        self.checkBox_remove_empty.setToolTip(tr("remove_empty_tip", lang))
        self.groupBox_scope.setTitle(tr("scope", lang))
        self.pushButton_preview.setText(tr("preview", lang))
        self.pushButton_run.setText(tr("run", lang))
        self.pushButton_refresh.setText(tr("refresh", lang))
        self.pushButton_undo.setText(tr("undo_last", lang))
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Close).setText(tr("close", lang))
        self.treeWidget_history.setHeaderLabels([
            tr("col_time", lang), tr("col_operation", lang), tr("col_images", lang),
            tr("col_objects", lang), tr("col_status", lang)])
        self.refresh_choices()
        self.refresh_history()
        if not self._busy:
            self.label_preview.setText(tr("preview_hint", lang))
        self._sync_controls()

    # ----------------------------------------------------------------- inputs
    def operation(self):
        return self.OPERATIONS[max(self.comboBox_operation.currentIndex(), 0)]

    def scope(self):
        return next(scope for scope, button in self.radio_scope.items() if button.isChecked())

    def scope_stems(self, scope=None):
        scope = scope or self.scope()
        if scope == "filtered":
            return self.files_dock.visible_stems()
        if scope == "current":
            stem = self.files_dock.current_stem()
            return [] if stem is None else [stem]
        return self.files_dock.all_stems()

    def selected_category(self):
        index = self.comboBox_category.currentIndex()
        return self._category_keys[index] if 0 <= index < len(self._category_keys) else None

    def refresh_choices(self):
        """Refresh category lists and scope counts from the Files dock index."""
        lang = self._language
        dock = self.files_dock
        current = self.selected_category()
        image_counts = dock.category_image_counts()
        names = dock.category_names(image_counts)
        # Categories that occur in this folder come first; configured-only ones follow.
        keys = ([name for name in names if image_counts.get(name)]
                + [name for name in names if not image_counts.get(name)])
        self._category_keys = list(keys)
        self.comboBox_category.blockSignals(True)
        self.comboBox_category.clear()
        indexed = dock.index is not None
        for name in keys:
            text = "{} ({})".format(name, image_counts.get(name, 0)) if indexed else name
            self.comboBox_category.addItem(dock.category_icon(name), text)
        if current in keys:
            self.comboBox_category.setCurrentIndex(keys.index(current))
        self.comboBox_category.blockSignals(False)

        target = self.comboBox_target.currentText()
        self.comboBox_target.blockSignals(True)
        self.comboBox_target.clear()
        for name in keys:
            self.comboBox_target.addItem(dock.category_icon(name), name)
        self.comboBox_target.setEditText(target)
        self.comboBox_target.blockSignals(False)

        self.radio_scope["all"].setText(tr("scope_all", lang, n=len(dock.all_stems())))
        self.radio_scope["filtered"].setText(tr("scope_filtered", lang, n=len(dock.visible_stems())))
        name = dock.current_file_name()
        self.radio_scope["current"].setText(
            tr("scope_current", lang, name=name) if name else tr("scope_current_none", lang))
        self.radio_scope["current"].setEnabled(bool(name))
        if not name and self.radio_scope["current"].isChecked():
            self.radio_scope["all"].setChecked(True)

    def preselect(self, category=None, filtered=False):
        """Start from the Files dock filter: its category and, if active, its scope."""
        self.refresh_choices()
        if category in self._category_keys:
            self.comboBox_category.setCurrentIndex(self._category_keys.index(category))
        self.radio_scope["filtered" if filtered else "all"].setChecked(True)

    def _sync_controls(self, *_):
        operation = self.operation()
        has_category = operation != batch_ops.OP_DELETE_EMPTY
        for widget in (self.label_category, self.comboBox_category):
            widget.setVisible(has_category)
        for widget in (self.label_target, self.comboBox_target):
            widget.setVisible(operation == batch_ops.OP_RENAME_CATEGORY)
        self.checkBox_remove_empty.setVisible(operation == batch_ops.OP_DELETE_CATEGORY)
        busy = self._busy
        for widget in (self.pushButton_preview, self.pushButton_run, self.pushButton_undo,
                       self.pushButton_refresh, self.comboBox_operation, self.comboBox_category,
                       self.comboBox_target, self.checkBox_remove_empty, self.groupBox_scope):
            widget.setEnabled(not busy)
        self.buttonBox.setEnabled(not busy)

    def _clear_preview(self, *_):
        if not self._busy:
            self.label_preview.setText(tr("preview_hint", self._language))

    def _request(self):
        """Return ``(call(dry_run, progress), description, scope)``; ValueError explains bad input."""
        lang = self._language
        root = self.mainwindow.label_root
        if not root or not self.files_dock.all_stems():
            raise ValueError(tr("no_folder", lang))
        scope = self.scope()
        stems = self.scope_stems(scope)
        if scope == "current" and not stems:
            raise ValueError(tr("no_current", lang))
        operation = self.operation()
        if operation == batch_ops.OP_DELETE_CATEGORY:
            category = self.selected_category()
            if not category:
                raise ValueError(tr("need_category", lang))
            remove_empty = self.checkBox_remove_empty.isChecked()
            description = describe_record({"op": operation, "params": {
                "category": category, "remove_empty_files": remove_empty}}, lang)

            def call(dry_run, progress=None):
                return batch_ops.delete_category(root, category, stems, remove_empty, dry_run,
                                                 scope, progress)
        elif operation == batch_ops.OP_RENAME_CATEGORY:
            category = self.selected_category()
            target = self.comboBox_target.currentText().strip()
            if not category:
                raise ValueError(tr("need_category", lang))
            if not target:
                raise ValueError(tr("need_target", lang))
            if target == category:
                raise ValueError(tr("same_target", lang))
            description = describe_record(
                {"op": operation, "params": {"old": category, "new": target}}, lang)

            def call(dry_run, progress=None):
                return batch_ops.rename_category(root, category, target, stems, dry_run, scope,
                                                 progress)
        else:
            description = describe_record({"op": operation, "params": {}}, lang)

            def call(dry_run, progress=None):
                return batch_ops.delete_empty_labels(root, stems, dry_run, scope, progress)
        return call, description, scope

    def _locked(self, call):
        """Serialise with tracking commits when the frames belong to a video project."""
        lock = nullcontext
        try:
            if self.mainwindow._video_frame_belongs_to_active_project():
                from IVAS.utils.persistence import project_write_lock
                path = self.mainwindow.video_annotation_workspace.store.path

                def lock():
                    return project_write_lock(path)
        except Exception:
            lock = nullcontext

        def locked(*args):
            with lock():
                return call(*args)
        return locked

    # ------------------------------------------------------------- operations
    def _summary_text(self, result, done=False):
        lang = self._language
        if result["op"] == batch_ops.OP_DELETE_EMPTY:
            text = tr("done_files" if done else "preview_result_files", lang, images=result["images"])
        else:
            text = tr("done" if done else "preview_result", lang,
                      images=result["images"], objects=result["objects"])
            deleted = sum(1 for item in result["files"] if item["action"] == "deleted")
            if deleted:
                text += "\n" + tr("preview_deleted", lang, n=deleted)
        if result["op"] == batch_ops.OP_RENAME_CATEGORY:
            target = result["params"]["new"]
            if target in self.files_dock.category_image_counts():
                text += "\n" + tr("merge_note", lang, name=target)
        if result.get("error_count"):
            lines = ["{}: {}".format(item["file"], item["error"]) for item in result["errors"][:5]]
            if result["error_count"] > len(lines):
                lines.append("…")
            heading = tr("done_partial" if done and result.get("record") else "errors_note", lang,
                         n=result["error_count"])
            text += "\n\n" + heading + "\n" + "\n".join(lines)
        return text

    def _on_progress(self, done, total):
        if self._busy:
            percent = int(100 * done / total) if total else 100
            self.label_preview.setText(tr("working_progress", self._language, percent=percent))

    def _set_busy(self, busy):
        self._busy = busy
        if busy:
            self.label_preview.setText(tr("working", self._language))
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._sync_controls()

    def preview(self):
        try:
            call, _description, _scope = self._request()
        except ValueError as error:
            self.label_preview.setText(str(error))
            return
        self._pending = None
        self._set_busy(True)
        progress = self.progressed.emit
        start_task(lambda: call(True, progress), self._on_preview, self._on_failed)

    def _on_preview(self, result):
        self._set_busy(False)
        self.label_preview.setText(self._summary_text(result) if result["images"]
                                   else tr("nothing", self._language))
        if self._pending is not None:
            call, description, scope = self._pending
            self._pending = None
            self._confirm_and_run(call, description, scope, result)

    def run_operation(self):
        """Preview, confirm, then apply the chosen operation."""
        try:
            call, description, scope = self._request()
        except ValueError as error:
            QtWidgets.QMessageBox.information(self, tr("title", self._language), str(error))
            return
        if not self.mainwindow._confirm_leave_unsaved_image():
            return
        self._pending = (call, description, scope)
        self._set_busy(True)
        progress = self.progressed.emit
        start_task(lambda: call(True, progress), self._on_preview, self._on_failed)

    def _confirm_and_run(self, call, description, scope, preview):
        lang = self._language
        if not preview["images"]:
            QtWidgets.QMessageBox.information(self, tr("title", lang), tr("nothing", lang))
            return
        backup = os.path.join(self.mainwindow.label_root, batch_ops.BATCH_DIR_NAME)
        answer = QtWidgets.QMessageBox.question(
            self, tr("confirm_title", lang),
            tr("confirm_text", lang, description=description, scope=_scope_name(scope, lang),
               summary=self._summary_text(preview), backup=backup),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        locked = self._locked(call)
        self._mutating = True
        self._set_busy(True)
        progress = self.progressed.emit
        start_task(lambda: locked(False, progress), self._on_run_finished, self._on_failed)

    def _on_run_finished(self, result):
        self._mutating = False
        self._set_busy(False)
        self._after_disk_change()
        text = self._summary_text(result, done=True)
        self.label_preview.setText(text)
        statusbar = getattr(self.mainwindow, "statusbar", None)
        if statusbar is not None:
            statusbar.showMessage(text.splitlines()[0], 8000)

    def _on_failed(self, error):
        mutating, self._mutating = self._mutating, False
        self._pending = None
        self._set_busy(False)
        lang = self._language
        if isinstance(error, batch_ops.UndoRefused):
            QtWidgets.QMessageBox.warning(self, tr("undo_refused_title", lang), str(error))
        else:
            QtWidgets.QMessageBox.critical(self, tr("failed_title", lang), str(error))
        self.label_preview.setText(str(error))
        if mutating and not isinstance(error, batch_ops.UndoRefused):
            self._after_disk_change()  # a partial run may have changed files
        else:
            self.refresh_history()

    def _after_disk_change(self):
        mainwindow = self.mainwindow
        if mainwindow.current_index is not None and mainwindow.files_list:
            mainwindow.show_image(mainwindow.current_index, zoomfit=False)
        self.files_dock.refresh_index()
        self.refresh_history()

    # ---------------------------------------------------------------- history
    def refresh_history(self):
        lang = self._language
        root = getattr(self.mainwindow, "label_root", None)
        tree = self.treeWidget_history
        tree.clear()
        records = batch_ops.read_history(root) if root else []
        for record in reversed(records):
            status = record.get("status", "done")
            if record.get("undone"):
                status_text = tr("status_undone", lang)
            elif status == "partial":
                status_text = tr("status_partial", lang)
            else:
                status_text = tr("status_done", lang)
            item = QtWidgets.QTreeWidgetItem([
                str(record.get("timestamp", "")).replace("T", " "),
                describe_record(record, lang),
                str(record.get("images", "")),
                str(record.get("objects", "")) if record.get("op") != batch_ops.OP_DELETE_EMPTY else "",
                status_text,
            ])
            files = [entry.get("file", "") for entry in record.get("files", [])]
            listed = "\n".join(files[:20]) + ("\n…" if len(files) > 20 else "")
            tooltip = tr("tip_record", lang, id=record.get("id", ""),
                         scope=_scope_name(record.get("scope"), lang),
                         backup=record.get("backup_dir", ""), files=listed)
            for column in range(tree.columnCount()):
                item.setToolTip(column, tooltip)
            tree.addTopLevelItem(item)
        for column in (0, 2, 3, 4):
            tree.resizeColumnToContents(column)
        self.label_history.setText(
            tr("history_hint", lang, path=batch_ops.history_path(root)) if root else "")
        self.pushButton_undo.setEnabled(
            not self._busy and bool(root) and any(
                record.get("op") in batch_ops.MUTATING_OPS and not record.get("undone")
                and record.get("files") for record in records))

    def undo_last(self):
        lang = self._language
        root = self.mainwindow.label_root
        record = batch_ops.last_undoable(root) if root else None
        if record is None:
            QtWidgets.QMessageBox.information(self, tr("undo_confirm_title", lang),
                                              tr("undo_none", lang))
            return
        answer = QtWidgets.QMessageBox.question(
            self, tr("undo_confirm_title", lang),
            tr("undo_confirm", lang, description=describe_record(record, lang),
               time=str(record.get("timestamp", "")).replace("T", " "),
               n=len(record.get("files", []))),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        if not self.mainwindow._confirm_leave_unsaved_image():
            return
        locked = self._locked(batch_ops.undo_last)
        self._mutating = True
        self._set_busy(True)
        progress = self.progressed.emit
        start_task(lambda: locked(root, progress), self._on_undo_finished, self._on_failed)

    def _on_undo_finished(self, record):
        self._mutating = False
        self._set_busy(False)
        self._after_disk_change()
        text = tr("undo_done", self._language, n=record.get("images", 0))
        self.label_preview.setText(text)
        statusbar = getattr(self.mainwindow, "statusbar", None)
        if statusbar is not None:
            statusbar.showMessage(text, 8000)

    # ------------------------------------------------------------------ Qt API
    def showEvent(self, event):
        self.refresh_choices()
        self.refresh_history()
        super().showEvent(event)

    def reject(self):
        if self._busy:
            return
        super().reject()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
            return
        super().closeEvent(event)
