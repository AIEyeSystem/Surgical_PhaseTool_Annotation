# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import fnmatch
import os
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from IVAS import batch_ops
from IVAS.ui.file_dock import Ui_Form
from IVAS.widgets.batch_ops_dialog import start_task, tr

# Keys of the special category-filter entries (real categories use their name).
FILTER_ALL = ("special", "all")
FILTER_ANY = ("special", "any")
FILTER_UNREADABLE = ("special", "unreadable")

_RESCAN_DELAY_MS = 400


class _CategoryFilterComboBox(QtWidgets.QComboBox):
    """Combo box that asks for fresh items right before its popup opens."""

    aboutToShowPopup = QtCore.pyqtSignal()

    def showPopup(self):
        self.aboutToShowPopup.emit()
        super().showPopup()


class FilesDockWidget(QtWidgets.QWidget, Ui_Form):
    """Image list of the open folder with a category/name filter and batch tools.

    Filtering only hides rows: row ``i`` always shows ``mainwindow.files_list[i]``,
    so clicks and navigation keep using the real image index.
    """

    index_updated = QtCore.pyqtSignal()

    def __init__(self, mainwindow):
        super(FilesDockWidget, self).__init__()
        self._filter_ready = False
        self.mainwindow = mainwindow
        self.setupUi(self)
        self._index = None
        self._scan_generation = 0
        self._scan_active = False
        self._scan_cancel = None
        self._scan_root = None
        self._rescan_pending = False
        self._watched_root = None
        self._filter_keys = [FILTER_ALL]
        self._last_filter = None
        self._icon_cache = {}
        self.batch_dialog = None

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_index)
        self._filter_timer = QtCore.QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(120)
        self._filter_timer.timeout.connect(self.apply_filter)
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._schedule_refresh)

        self._build_filter_ui()
        self.listWidget.clicked.connect(self.listwidget_doubleclick)
        self.lineEdit_jump.returnPressed.connect(self.mainwindow.jump_to)

        self.setAcceptDrops(True)
        self._filter_ready = True
        self.retranslate(self.language())

    # ------------------------------------------------------------------ layout
    def _build_filter_ui(self):
        self.widget_filter = QtWidgets.QWidget(self)
        self.widget_filter.setObjectName("widget_filter")
        grid = QtWidgets.QGridLayout(self.widget_filter)
        grid.setContentsMargins(0, 0, 0, 6)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        self.lineEdit_search = QtWidgets.QLineEdit(self.widget_filter)
        self.lineEdit_search.setObjectName("lineEdit_search")
        self.lineEdit_search.setClearButtonEnabled(True)
        self.lineEdit_search.setMinimumWidth(40)

        self.toolButton_batch = QtWidgets.QToolButton(self.widget_filter)
        self.toolButton_batch.setObjectName("toolButton_batch")
        self.toolButton_batch.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)

        self.comboBox_filter_mode = QtWidgets.QComboBox(self.widget_filter)
        self.comboBox_filter_mode.setObjectName("comboBox_filter_mode")
        self.comboBox_filter_mode.addItems(["has", "lacks"])
        self.comboBox_filter_mode.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.comboBox_category = _CategoryFilterComboBox(self.widget_filter)
        self.comboBox_category.setObjectName("comboBox_category_filter")
        self.comboBox_category.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.comboBox_category.setMinimumContentsLength(6)
        self.comboBox_category.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                             QtWidgets.QSizePolicy.Fixed)
        self.comboBox_category.setIconSize(QtCore.QSize(10, 10))

        self.label_filter_count = QtWidgets.QLabel(self.widget_filter)
        self.label_filter_count.setObjectName("label_filter_count")
        self.label_filter_count.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        # Long counts must not widen the dock; the label takes the row's width.
        self.label_filter_count.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                              QtWidgets.QSizePolicy.Preferred)

        grid.addWidget(self.lineEdit_search, 0, 0, 1, 2)
        grid.addWidget(self.toolButton_batch, 0, 2)
        grid.addWidget(self.comboBox_filter_mode, 1, 0)
        grid.addWidget(self.comboBox_category, 1, 1, 1, 2)
        grid.addWidget(self.label_filter_count, 2, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        self.verticalLayout.insertWidget(0, self.widget_filter)

        self.lineEdit_search.textChanged.connect(lambda _text: self._filter_timer.start())
        self.comboBox_filter_mode.currentIndexChanged.connect(self.apply_filter)
        self.comboBox_category.currentIndexChanged.connect(self._category_filter_changed)
        self.comboBox_category.aboutToShowPopup.connect(self._rebuild_category_combo)
        self.toolButton_batch.clicked.connect(self.open_batch_dialog)
        self._rebuild_category_combo()

    # --------------------------------------------------------------- language
    def language(self):
        mainwindow = self.mainwindow
        language = getattr(mainwindow, "_ui_language", None)
        if language is None:
            try:
                language = mainwindow.cfg["software"]["language"]
            except (AttributeError, KeyError, TypeError):
                language = "en"
        return "zh" if language == "zh" else "en"

    def retranslateUi(self, Form):
        # MainWindow.translate() calls this on every language change.
        Ui_Form.retranslateUi(self, Form)
        if getattr(self, "_filter_ready", False):
            self.retranslate(self.language())

    def retranslate(self, language):
        """Apply 'zh'/'en' texts to the filter row and an open batch dialog."""
        language = "zh" if language == "zh" else "en"
        self.lineEdit_search.setPlaceholderText(tr("search_placeholder", language))
        self.lineEdit_search.setToolTip(tr("search_tip", language))
        self.toolButton_batch.setText(tr("batch_button", language))
        self.toolButton_batch.setToolTip(tr("batch_tip", language))
        self.toolButton_batch.setStatusTip(tr("batch_tip", language))
        self.comboBox_filter_mode.setItemText(0, tr("mode_has", language))
        self.comboBox_filter_mode.setItemText(1, tr("mode_lacks", language))
        self.comboBox_filter_mode.setToolTip(tr("mode_tip", language))
        self.comboBox_category.setToolTip(tr("category_tip", language))
        self._rebuild_category_combo()
        self._update_row_tooltips()
        self.apply_filter()
        if self.batch_dialog is not None:
            self.batch_dialog.retranslate(language)

    # ------------------------------------------------------------------- rows
    def update_widget(self):
        """Update the file list widget."""
        self.listWidget.clear()
        if self.mainwindow.files_list is None:
            return

        for idx, file_name in enumerate(self.mainwindow.files_list):
            item = QtWidgets.QListWidgetItem()
            item.setText(f"[{idx + 1}] {file_name}")
            self.listWidget.addItem(item)

        self.label_all.setText("{}".format(len(self.mainwindow.files_list)))
        self._update_row_tooltips()
        self.apply_filter()
        # load_image_directory() assigns label_root after this call; scan once
        # control returns to the event loop.
        self._refresh_timer.start(0)

    def set_select(self, row):
        """Set the current file highlight."""
        self.listWidget.setCurrentRow(row)
        if not self._index_matches_root() and not self._scan_active:
            # e.g. the label directory was switched without reloading the list.
            self._refresh_timer.start(0)

    def listwidget_doubleclick(self, index=None):
        """Open the clicked image; rows map 1:1 to ``files_list`` even when hidden."""
        if isinstance(index, QtCore.QModelIndex) and index.isValid():
            row = index.row()
        else:
            row = self.listWidget.currentRow()
        self.mainwindow.show_image(row)

    # ----------------------------------------------------------------- stems
    @staticmethod
    def stem_of(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

    def all_stems(self):
        return list(dict.fromkeys(self.stem_of(name) for name in self.mainwindow.files_list or []))

    def visible_stems(self):
        files = self.mainwindow.files_list or []
        count = min(len(files), self.listWidget.count())
        return list(dict.fromkeys(self.stem_of(files[row]) for row in range(count)
                                  if not self.listWidget.isRowHidden(row)))

    def visible_rows(self):
        return [row for row in range(self.listWidget.count()) if not self.listWidget.isRowHidden(row)]

    def current_file_name(self):
        files = self.mainwindow.files_list or []
        index = self.mainwindow.current_index
        if index is None or not 0 <= index < len(files):
            return None
        return files[index]

    def current_stem(self):
        name = self.current_file_name()
        return None if name is None else self.stem_of(name)

    # ----------------------------------------------------------------- index
    @property
    def index(self):
        """The :class:`IVAS.batch_ops.LabelIndex` of the open label directory, if scanned."""
        return self._index if self._index_matches_root() else None

    def _index_matches_root(self):
        root = getattr(self.mainwindow, "label_root", None)
        return (self._index is not None and bool(root)
                and os.path.normcase(os.path.abspath(root)) == os.path.normcase(self._index.root))

    def is_index_busy(self):
        """True while a scan is running or scheduled."""
        return self._scan_active or self._refresh_timer.isActive()

    def _schedule_refresh(self, *_):
        self._refresh_timer.start(_RESCAN_DELAY_MS)

    def _watch(self, root):
        root = os.path.abspath(root) if root else None
        if root == self._watched_root:
            return
        directories = self._watcher.directories()
        if directories:
            self._watcher.removePaths(directories)
        self._watched_root = None
        if root and os.path.isdir(root) and self._watcher.addPath(root):
            self._watched_root = root

    def refresh_index(self):
        """Rescan the label directory in a background thread (incremental).

        Unchanged label files (same name, mtime and size) reuse their previous
        counts, so calling this after every save is cheap even for thousands of
        images. The filter and counts update when the scan finishes.
        """
        self._refresh_timer.stop()
        root = getattr(self.mainwindow, "label_root", None)
        files = self.mainwindow.files_list or []
        self._watch(root if files else None)
        if not root or not files:
            self._scan_generation += 1
            self._scan_active = False
            self._index = None
            self._apply_index()
            return
        root = os.path.abspath(root)
        if self._scan_active:
            if self._scan_root == root:
                self._rescan_pending = True
                return
            self._scan_cancel.set()
        stems = self.all_stems()
        previous = self._index
        self._scan_generation += 1
        generation = self._scan_generation
        cancel = threading.Event()
        self._scan_cancel, self._scan_root = cancel, root
        self._scan_active, self._rescan_pending = True, False

        def scan():
            try:
                return generation, batch_ops.scan_label_dir(
                    root, stems, previous=previous, cancelled=cancel), None
            except Exception as error:  # keep the dock usable
                return generation, None, str(error)

        start_task(scan, self._on_index_scanned)
        self._update_count_label()

    def _on_index_scanned(self, payload):
        generation, index, error = payload
        if generation != self._scan_generation:
            return  # superseded by a newer scan
        self._scan_active = False
        if index is not None:
            self._index = index
        elif error:
            self.mainwindow.statusbar.showMessage("Label index failed: {}".format(error), 5000)
        self._apply_index()
        if self._rescan_pending:
            self._rescan_pending = False
            self.refresh_index()

    def _apply_index(self):
        self.refresh_categories()
        self._update_row_tooltips()
        self.apply_filter()
        if self.batch_dialog is not None and self.batch_dialog.isVisible():
            self.batch_dialog.refresh_choices()
        self.index_updated.emit()

    # ------------------------------------------------------------- categories
    def category_image_counts(self):
        index = self.index
        return index.image_counts(self.all_stems()) if index is not None else {}

    def category_names(self, image_counts=None):
        """Configured categories (in config order) followed by those only found on disk."""
        if image_counts is None:
            image_counts = self.category_image_counts()
        configured = list(getattr(self.mainwindow, "category_color_dict", None) or {})
        names = [name for name in configured
                 if name != "__background__" or image_counts.get(name)]
        known = set(names)
        names.extend(sorted(name for name in image_counts if name not in known))
        return names

    def category_icon(self, name):
        color = (getattr(self.mainwindow, "category_color_dict", None) or {}).get(name)
        if not color:
            return QtGui.QIcon()
        icon = self._icon_cache.get(color)
        if icon is None:
            pixmap = QtGui.QPixmap(10, 10)
            pixmap.fill(QtGui.QColor(color))
            icon = self._icon_cache[color] = QtGui.QIcon(pixmap)
        return icon

    def refresh_categories(self):
        """Rebuild the category filter (e.g. after the category config changed)."""
        if not self.comboBox_category.view().isVisible():
            self._rebuild_category_combo()

    def _rebuild_category_combo(self):
        language = self.language()
        combo = self.comboBox_category
        current_index = combo.currentIndex()
        current = (self._filter_keys[current_index]
                   if 0 <= current_index < len(self._filter_keys) else FILTER_ALL)
        index = self.index
        stems = self.all_stems()
        image_counts = self.category_image_counts()

        def counted(text, count):
            return text if index is None else "{} ({})".format(text, count)

        entries = [(FILTER_ALL, tr("filter_all", language), QtGui.QIcon()),
                   (FILTER_ANY, counted(tr("filter_any", language),
                                        sum(1 for stem in stems if index is not None
                                            and index.counts.get(stem))), QtGui.QIcon())]
        names = self.category_names(image_counts)
        if (current[0] == "category" and current[1] not in names):
            names.append(current[1])  # keep a filter whose last image just disappeared
        entries.extend((("category", name), counted(name, image_counts.get(name, 0)),
                        self.category_icon(name)) for name in names)
        unreadable = sum(1 for stem in stems if index is not None and stem in index.errors)
        if unreadable or current == FILTER_UNREADABLE:
            entries.append((FILTER_UNREADABLE, counted(tr("filter_unreadable", language),
                                                       unreadable), QtGui.QIcon()))

        self._filter_keys = [key for key, _, _ in entries]
        combo.blockSignals(True)
        combo.clear()
        for _key, text, icon in entries:
            combo.addItem(icon, text)
        combo.setCurrentIndex(self._filter_keys.index(current)
                              if current in self._filter_keys else 0)
        combo.blockSignals(False)
        view = combo.view()
        view.setMinimumWidth(view.sizeHintForColumn(0) + 24)
        self._sync_mode_enabled()

    def _category_filter_changed(self, *_):
        self._sync_mode_enabled()
        self.apply_filter()

    def _sync_mode_enabled(self):
        key = self.filter_key()
        self.comboBox_filter_mode.setEnabled(key not in (FILTER_ALL, FILTER_UNREADABLE))

    def filter_key(self):
        row = self.comboBox_category.currentIndex()
        return self._filter_keys[row] if 0 <= row < len(self._filter_keys) else FILTER_ALL

    def set_category_filter(self, category, lacks=False):
        """Select ``category`` (or one of the FILTER_* keys) programmatically."""
        key = category if isinstance(category, tuple) else ("category", str(category))
        if key not in self._filter_keys:
            self._rebuild_category_combo()
        if key not in self._filter_keys:
            # A category not configured and not (yet) on disk.
            self._filter_keys.append(key)
            self.comboBox_category.addItem(self.category_icon(key[1]), "{} (0)".format(key[1]))
        self.comboBox_filter_mode.setCurrentIndex(1 if lacks else 0)
        self.comboBox_category.setCurrentIndex(self._filter_keys.index(key))
        self.apply_filter()

    # ---------------------------------------------------------------- filter
    @staticmethod
    def _name_matches(name, text):
        if not text:
            return True
        name = name.lower()
        if any(char in text for char in "*?["):
            return fnmatch.fnmatchcase(name, text if "*" in text else "*{}*".format(text))
        return text in name

    def _row_visible(self, name, text, key, lacks, index):
        if not self._name_matches(name, text):
            return False
        if key == FILTER_ALL or index is None:
            return True
        stem = self.stem_of(name)
        if key == FILTER_UNREADABLE:
            return stem in index.errors
        if stem in index.errors:
            return False  # content unknown; listed under "Unreadable labels"
        counts = index.counts.get(stem, {})
        has = bool(counts) if key == FILTER_ANY else key[1] in counts
        return has != lacks

    def apply_filter(self, *_):
        """Hide rows that do not match; never touches ``files_list`` or row order."""
        if not getattr(self, "_filter_ready", False):
            return
        self._filter_timer.stop()
        files = self.mainwindow.files_list or []
        text = self.lineEdit_search.text().strip().lower()
        key = self.filter_key()
        lacks = self.comboBox_filter_mode.currentIndex() == 1
        index = self.index
        list_widget = self.listWidget
        list_widget.setUpdatesEnabled(False)
        try:
            for row in range(min(len(files), list_widget.count())):
                hidden = not self._row_visible(files[row], text, key, lacks, index)
                if list_widget.isRowHidden(row) != hidden:
                    list_widget.setRowHidden(row, hidden)
        finally:
            list_widget.setUpdatesEnabled(True)
        self._update_count_label()
        # Keep the current image in view when the filter changes, but do not
        # jump while the user browses and a save triggers a rescan.
        state = (text, key, lacks)
        current = list_widget.currentItem()
        if state != self._last_filter and current is not None and not current.isHidden():
            list_widget.scrollToItem(current)
        self._last_filter = state

    def _update_count_label(self):
        language = self.language()
        self.toolButton_batch.setEnabled(bool(self.mainwindow.files_list))
        total = self.listWidget.count()
        visible = sum(1 for row in range(total) if not self.listWidget.isRowHidden(row))
        key = "count_scanning" if self._scan_active and self.index is None else "count"
        self.label_filter_count.setText(tr(key, language, visible=visible, total=total))
        index = self.index
        errors = sum(1 for stem in self.all_stems() if stem in index.errors) if index else 0
        self.label_filter_count.setToolTip(
            tr("count_tip_errors", language, n=errors) if errors else "")

    def _update_row_tooltips(self):
        if not getattr(self, "_filter_ready", False):
            return
        language = self.language()
        files = self.mainwindow.files_list or []
        index = self.index
        for row in range(min(len(files), self.listWidget.count())):
            item = self.listWidget.item(row)
            if index is None:
                item.setToolTip("")
                continue
            stem = self.stem_of(files[row])
            if stem in index.errors:
                tip = tr("row_unreadable", language, error=index.errors[stem])
            elif stem not in index.counts:
                tip = tr("row_no_label", language)
            elif not index.counts[stem]:
                tip = tr("row_empty", language)
            else:
                tip = "\n".join("{} × {}".format(name, count)
                                for name, count in sorted(index.counts[stem].items()))
            item.setToolTip(tip)

    # ----------------------------------------------------------------- batch
    def open_batch_dialog(self):
        from IVAS.widgets.batch_ops_dialog import BatchOperationsDialog

        if self.batch_dialog is None:
            self.batch_dialog = BatchOperationsDialog(self, self.mainwindow)
        else:
            self.batch_dialog.retranslate(self.language())
        key = self.filter_key()
        self.batch_dialog.preselect(
            key[1] if key[0] == "category" else None,
            filtered=len(self.visible_rows()) < self.listWidget.count())
        self.batch_dialog.show()
        self.batch_dialog.raise_()
        self.batch_dialog.activateWindow()
        return self.batch_dialog

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Support drop event for open a image file or image directory."""
        if len(event.mimeData().urls()) != 1:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Only support one path or dir."
            )
            return
        path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(path):
            loaded = self.mainwindow.load_image_directory(path)
        elif os.path.isfile(path):
            suffixes = tuple('.' + bytes(fmt).decode('ascii').lower()
                             for fmt in QtGui.QImageReader.supportedImageFormats())
            if not path.lower().endswith(suffixes + ('.dcm',)):
                event.ignore()
                return
            directory, filename = os.path.split(path)
            loaded = self.mainwindow.load_image_directory(
                directory, selected_filename=filename
            )
        else:
            event.ignore()
            return
        if loaded:
            event.acceptProposedAction()
        else:
            event.ignore()
