"""Compact secondary dialogs for the unified video workspace."""

from __future__ import annotations

import html
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from IVAS.widgets.responsive import ResponsiveDialog

from IVAS.video.project import TemporalAnnotation, TemporalTemplate, make_id


TRACKING_REVIEW_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 159, 67),
    (76, 215, 255),
    (105, 219, 124),
    (230, 126, 188),
    (170, 120, 255),
    (80, 205, 230),
)


def tracking_review_colour(object_id: str) -> Tuple[int, int, int]:
    """Return a stable BGR review colour for an object identifier."""

    checksum = zlib.crc32(str(object_id).encode("utf-8"))
    return TRACKING_REVIEW_PALETTE[checksum % len(TRACKING_REVIEW_PALETTE)]


def compose_tracking_overlay(
    frame_bgr: np.ndarray,
    masks: Sequence[Tuple[str, np.ndarray]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend staged object masks over a BGR frame without mutating the inputs."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Tracking overlay alpha must be between 0 and 1")
    frame = np.asarray(frame_bgr)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Tracking review frame must be a grayscale, BGR, or BGRA image")
    source = np.asarray(frame, dtype=np.uint8)
    result = source.copy()
    height, width = source.shape[:2]
    for object_id, raw_mask in sorted(masks, key=lambda item: str(item[0])):
        mask = np.asarray(raw_mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError("Tracking review masks must be two-dimensional")
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        selected = mask > 0
        if not np.any(selected):
            continue
        colour = np.asarray(tracking_review_colour(object_id), dtype=np.float32)
        blended = (
            source[selected].astype(np.float32) * (1.0 - float(alpha))
            + colour * float(alpha)
        )
        result[selected] = np.rint(blended).clip(0, 255).astype(np.uint8)
    return result


def _pixmap_from_bgr(image: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    qimage = QtGui.QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QtGui.QImage.Format_RGB888,
    ).copy()
    return QtGui.QPixmap.fromImage(qimage)


class TrackingReviewDialog(ResponsiveDialog):
    """Read-only visual review of a staged tracking run."""

    Discarded = 2

    def __init__(
        self,
        run: Mapping[str, Any],
        frame_loader: Callable[[int], np.ndarray],
        mask_path_resolver: Callable[[str], Path],
        instances: Optional[Mapping[str, Mapping[str, Any]]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._run = run
        self._frames = list(run.get("staged_frames", []))
        self._frame_loader = frame_loader
        self._mask_path_resolver = mask_path_resolver
        self._instances = dict(instances or {})
        self._original_pixmap = QtGui.QPixmap()
        self._overlay_pixmap = QtGui.QPixmap()

        self.setWindowTitle("审阅暂存追踪")
        self.resize(960, 560)
        self.setMinimumSize(720, 440)
        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QLabel(
            "方向：{} · 范围：{}–{} · 帧数：{}\n"
            "遮罩仅供审阅；人工对象默认优先，提交失败会自动恢复所有标签。".format(
                run.get("direction", "—"),
                run.get("start_frame", "—"),
                run.get("end_frame", "—"),
                len(self._frames),
            )
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.frame_table = QtWidgets.QTableWidget(len(self._frames), 2)
        self.frame_table.setHorizontalHeaderLabels(["帧", "对象"])
        self.frame_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.frame_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.frame_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.horizontalHeader().setStretchLastSection(True)
        self.frame_table.setMinimumWidth(150)
        self.frame_table.setMaximumWidth(230)
        for row, frame in enumerate(self._frames):
            self.frame_table.setItem(
                row,
                0,
                QtWidgets.QTableWidgetItem(str(frame.get("frame_index", "—"))),
            )
            self.frame_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(str(len(frame.get("objects", [])))),
            )
        splitter.addWidget(self.frame_table)

        preview = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        original_panel, self.original_label = self._preview_panel("原始帧")
        overlay_panel, self.overlay_label = self._preview_panel("遮罩叠加")
        preview_splitter.addWidget(original_panel)
        preview_splitter.addWidget(overlay_panel)
        preview_splitter.setSizes([1, 1])
        preview_layout.addWidget(preview_splitter, 1)
        self.legend_label = QtWidgets.QLabel()
        self.legend_label.setWordWrap(True)
        self.legend_label.setTextFormat(QtCore.Qt.RichText)
        preview_layout.addWidget(self.legend_label)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #d9534f;")
        preview_layout.addWidget(self.error_label)
        splitter.addWidget(preview)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        buttons = QtWidgets.QDialogButtonBox()
        self.commit_button = buttons.addButton(
            "提交到标签", QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.discard_button = buttons.addButton(
            "丢弃暂存", QtWidgets.QDialogButtonBox.DestructiveRole
        )
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.commit_button.clicked.connect(self.accept)
        self.discard_button.clicked.connect(lambda: self.done(self.Discarded))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.frame_table.currentCellChanged.connect(
            lambda row, _column, _previous_row, _previous_column: self._show_frame(row)
        )
        if self._frames:
            self.frame_table.setCurrentCell(0, 0)
        else:
            self.commit_button.setEnabled(False)
            self.error_label.setText("此暂存追踪没有可审阅的帧。")

    @staticmethod
    def _preview_panel(title: str) -> Tuple[QtWidgets.QWidget, QtWidgets.QLabel]:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 4, 0)
        heading = QtWidgets.QLabel(title)
        heading.setAlignment(QtCore.Qt.AlignCenter)
        image = QtWidgets.QLabel("选择左侧帧进行预览")
        image.setAlignment(QtCore.Qt.AlignCenter)
        image.setMinimumSize(220, 260)
        image.setAccessibleName(title)
        image.setStyleSheet("background: #202124; color: #b8b8b8;")
        layout.addWidget(heading)
        layout.addWidget(image, 1)
        return panel, image

    def _show_frame(self, row: int) -> None:
        if row < 0 or row >= len(self._frames):
            return
        frame_entry = self._frames[row]
        frame_index = int(frame_entry.get("frame_index", 0))
        errors = []
        try:
            original = self._frame_loader(frame_index)
            if original is None:
                raise ValueError("无法读取原始帧")
            masks = []
            visible_ids = []
            for item in frame_entry.get("objects", []):
                object_id = str(item.get("object_id", "unknown"))
                mask_path = self._mask_path_resolver(str(item.get("mask_path", "")))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    errors.append("缺少遮罩：{}".format(mask_path.name or object_id))
                    continue
                masks.append((object_id, mask))
                visible_ids.append(object_id)
            original_bgr = compose_tracking_overlay(original, ())
            overlay = compose_tracking_overlay(original_bgr, masks)
            self._original_pixmap = _pixmap_from_bgr(original_bgr)
            self._overlay_pixmap = _pixmap_from_bgr(overlay)
            self._update_scaled_pixmaps()
            self._set_legend(visible_ids)
        except Exception as error:
            self._original_pixmap = QtGui.QPixmap()
            self._overlay_pixmap = QtGui.QPixmap()
            self.original_label.setText("原始帧不可用")
            self.overlay_label.setText("预览不可用")
            self.legend_label.clear()
            errors.append(str(error))
        self.error_label.setText("；".join(errors))

    def _set_legend(self, object_ids: Sequence[str]) -> None:
        entries = []
        for object_id in object_ids:
            blue, green, red = tracking_review_colour(object_id)
            instance = self._instances.get(object_id, {})
            label = str(instance.get("category") or object_id)
            track_id = str(instance.get("track_id") or "")
            if track_id and track_id != label:
                label = "{} · {}".format(label, track_id)
            entries.append(
                '<span style="color: rgb({},{},{})">■</span> {}'.format(
                    red,
                    green,
                    blue,
                    html.escape(label),
                )
            )
        self.legend_label.setText("&nbsp;&nbsp;".join(entries) or "此帧没有有效遮罩。")

    def _update_scaled_pixmaps(self) -> None:
        for label, pixmap in (
            (self.original_label, self._original_pixmap),
            (self.overlay_label, self._overlay_pixmap),
        ):
            if pixmap.isNull():
                continue
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmaps()


class TemplateEditorDialog(ResponsiveDialog):
    def __init__(
        self,
        template: Optional[TemporalTemplate] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑时间标注模板" if template else "新增时间标注模板")
        self.setModal(True)
        self.setMinimumWidth(420)
        layout = QtWidgets.QFormLayout(self)
        self.label_edit = QtWidgets.QLineEdit(template.label if template else "")
        self.label_edit.setAccessibleName("模板标签")
        self.kind_combo = QtWidgets.QComboBox()
        for label, value in (("区间", "interval"), ("关键帧/点", "point"), ("评分", "rating")):
            self.kind_combo.addItem(label, value)
        self.category_edit = QtWidgets.QLineEdit(template.category if template else "phase")
        self.group_edit = QtWidgets.QLineEdit(template.group if template else "Default")
        self.colour_edit = QtWidgets.QLineEdit(template.colour if template else "")
        self.start_hint_edit = QtWidgets.QLineEdit(template.start_hint if template else "")
        self.end_hint_edit = QtWidgets.QLineEdit(template.end_hint if template else "")
        self.score_options_edit = QtWidgets.QLineEdit(
            ", ".join(str(value) for value in template.score_options) if template else "0, 1"
        )
        self.required_check = QtWidgets.QCheckBox("必填")
        self.required_check.setChecked(bool(template and template.required))
        layout.addRow("标签", self.label_edit)
        layout.addRow("类型", self.kind_combo)
        layout.addRow("分类/轨道", self.category_edit)
        layout.addRow("分组", self.group_edit)
        layout.addRow("颜色（可选）", self.colour_edit)
        layout.addRow("开始提示", self.start_hint_edit)
        layout.addRow("结束提示", self.end_hint_edit)
        layout.addRow("评分选项", self.score_options_edit)
        layout.addRow("要求", self.required_check)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        if template:
            index = self.kind_combo.findData(template.kind)
            self.kind_combo.setCurrentIndex(max(0, index))
            self.kind_combo.setEnabled(False)
        self.kind_combo.currentIndexChanged.connect(self._update_kind_fields)
        self._update_kind_fields()

    def _update_kind_fields(self) -> None:
        kind = self.kind_combo.currentData()
        self.score_options_edit.setEnabled(kind == "rating")
        self.start_hint_edit.setEnabled(kind == "interval")
        self.end_hint_edit.setEnabled(kind == "interval")

    def _accept_validated(self) -> None:
        if not self.label_edit.text().strip() or not self.category_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "模板无效", "标签和分类不能为空。")
            return
        if self.kind_combo.currentData() == "rating":
            try:
                options = self._score_options()
            except ValueError as error:
                QtWidgets.QMessageBox.warning(self, "评分选项无效", str(error))
                return
            if not options:
                QtWidgets.QMessageBox.warning(self, "评分选项无效", "至少需要一个整数评分。")
                return
        self.accept()

    def _score_options(self) -> List[int]:
        text = self.score_options_edit.text().strip()
        if not text:
            return []
        try:
            return sorted(set(int(value.strip()) for value in text.split(",") if value.strip()))
        except ValueError as error:
            raise ValueError("评分选项应为逗号分隔的整数。") from error

    def template(self, identifier: str = "") -> TemporalTemplate:
        return TemporalTemplate(
            id=str(identifier or make_id("tpl")),
            label=self.label_edit.text().strip(),
            kind=str(self.kind_combo.currentData()),
            category=self.category_edit.text().strip(),
            group=self.group_edit.text().strip() or "Default",
            colour=self.colour_edit.text().strip(),
            start_hint=self.start_hint_edit.text().strip(),
            end_hint=self.end_hint_edit.text().strip(),
            score_options=(self._score_options() if self.kind_combo.currentData() == "rating" else []),
            required=self.required_check.isChecked(),
        )


class TemplateManagerDialog(ResponsiveDialog):
    templates_changed = QtCore.pyqtSignal()

    def __init__(self, store, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("时间标注模板")
        self.resize(720, 430)
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["标签", "类型", "分类", "分组", "必填"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        actions = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("新增")
        self.edit_button = QtWidgets.QPushButton("编辑")
        self.delete_button = QtWidgets.QPushButton("删除")
        close_button = QtWidgets.QPushButton("关闭")
        actions.addWidget(self.add_button)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.delete_button)
        actions.addStretch(1)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        self.add_button.clicked.connect(self._add)
        self.edit_button.clicked.connect(self._edit)
        self.delete_button.clicked.connect(self._delete)
        close_button.clicked.connect(self.accept)
        self.table.itemDoubleClicked.connect(lambda _item: self._edit())
        self.table.itemSelectionChanged.connect(self._update_actions)
        self.refresh()

    def _selected(self) -> Optional[TemporalTemplate]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        template_id = rows[0].data(QtCore.Qt.UserRole)
        return next(
            (item for item in self.store.temporal_templates if item.id == template_id), None
        )

    def _commit(self, operation) -> bool:
        try:
            with self.store.transaction():
                operation()
                self.store.save()
            self.refresh()
            self.templates_changed.emit()
            return True
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "模板保存失败", str(error))
            return False

    def _add(self) -> None:
        dialog = TemplateEditorDialog(parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            template = dialog.template()
            self._commit(lambda: self.store.add_template(template))

    def _edit(self) -> None:
        current = self._selected()
        if current is None:
            return
        dialog = TemplateEditorDialog(current, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        updated = dialog.template(current.id)
        changes = asdict(updated)
        changes.pop("id", None)
        changes.pop("kind", None)
        changes.pop("source", None)
        self._commit(lambda: self.store.update_template(current.id, **changes))

    def _delete(self) -> None:
        current = self._selected()
        if current is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "删除模板",
            "确定删除“{}”吗？".format(current.label),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self._commit(lambda: self.store.delete_template(current.id))

    def _update_actions(self) -> None:
        selected = self._selected() is not None
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def refresh(self) -> None:
        templates = self.store.temporal_templates
        self.table.setRowCount(len(templates))
        for row, template in enumerate(templates):
            values = (
                template.label,
                template.kind,
                template.category,
                template.group,
                "是" if template.required else "否",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, template.id)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._update_actions()


class TemporalAnnotationDialog(ResponsiveDialog):
    def __init__(
        self,
        templates: Sequence[TemporalTemplate],
        duration_ms: int,
        position_ms: int,
        annotation: Optional[TemporalAnnotation] = None,
        kind: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑时间标注" if annotation else "新增时间标注")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._templates = [
            item for item in templates if kind is None or item.kind == kind
        ]
        if annotation is not None:
            self._templates = [item for item in self._templates if item.kind == annotation.kind]
        layout = QtWidgets.QFormLayout(self)
        self.template_combo = QtWidgets.QComboBox()
        for template in self._templates:
            self.template_combo.addItem(
                "{} · {}".format(template.category, template.label), template.id
            )
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, max(0, int(duration_ms)))
        self.start_spin.setSuffix(" ms")
        self.end_spin = QtWidgets.QSpinBox()
        self.end_spin.setRange(0, max(0, int(duration_ms)))
        self.end_spin.setSuffix(" ms")
        self.score_combo = QtWidgets.QComboBox()
        self.status_combo = QtWidgets.QComboBox()
        for label, value in (("草稿", "draft"), ("待复核", "ready"), ("已复核", "reviewed")):
            self.status_combo.addItem(label, value)
        self.note_edit = QtWidgets.QPlainTextEdit()
        self.note_edit.setMaximumHeight(90)
        layout.addRow("模板", self.template_combo)
        layout.addRow("开始/时间点", self.start_spin)
        layout.addRow("结束", self.end_spin)
        layout.addRow("评分", self.score_combo)
        layout.addRow("审阅状态", self.status_combo)
        layout.addRow("备注", self.note_edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.template_combo.currentIndexChanged.connect(self._update_template_fields)
        if annotation is not None:
            index = self.template_combo.findData(annotation.template_id)
            self.template_combo.setCurrentIndex(max(0, index))
            self.start_spin.setValue(annotation.start_ms)
            self.end_spin.setValue(annotation.end_ms or annotation.start_ms)
            status_index = self.status_combo.findData(annotation.status)
            self.status_combo.setCurrentIndex(max(0, status_index))
            self.note_edit.setPlainText(annotation.note)
        else:
            self.start_spin.setValue(max(0, int(position_ms)))
            self.end_spin.setValue(max(0, int(position_ms)))
        self._update_template_fields()
        if annotation is not None and annotation.score is not None:
            score_index = self.score_combo.findData(annotation.score)
            self.score_combo.setCurrentIndex(max(0, score_index))

    def _template(self) -> Optional[TemporalTemplate]:
        template_id = self.template_combo.currentData()
        return next((item for item in self._templates if item.id == template_id), None)

    def _update_template_fields(self) -> None:
        template = self._template()
        kind = template.kind if template else ""
        self.end_spin.setEnabled(kind == "interval")
        self.score_combo.setEnabled(kind == "rating")
        self.score_combo.clear()
        if template is not None and kind == "rating":
            for value in template.score_options:
                self.score_combo.addItem(str(value), value)
        if kind == "rating":
            self.start_spin.setValue(0)
            self.start_spin.setEnabled(False)
        else:
            self.start_spin.setEnabled(True)

    def _accept_validated(self) -> None:
        template = self._template()
        if template is None:
            QtWidgets.QMessageBox.warning(self, "缺少模板", "请先在更多菜单中创建对应模板。")
            return
        if template.kind == "interval" and self.end_spin.value() < self.start_spin.value():
            QtWidgets.QMessageBox.warning(self, "时间无效", "结束时间不能早于开始时间。")
            return
        self.accept()

    def values(self) -> Dict[str, Any]:
        template = self._template()
        if template is None:
            raise ValueError("No temporal template is selected")
        return {
            "template_id": template.id,
            "start_ms": self.start_spin.value(),
            "end_ms": self.end_spin.value() if template.kind == "interval" else None,
            "score": self.score_combo.currentData() if template.kind == "rating" else None,
            "status": str(self.status_combo.currentData()),
            "note": self.note_edit.toPlainText(),
        }


class TrackingRangeDialog(ResponsiveDialog):
    def __init__(
        self,
        frame_count: int,
        seed_options: Sequence[Dict[str, Any]],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("视频对象追踪")
        self.setMinimumWidth(500)
        layout = QtWidgets.QFormLayout(self)
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItem("双向", "bidirectional")
        self.direction_combo.addItem("向前", "forward")
        self.direction_combo.addItem("向后", "backward")
        self.start_spin = QtWidgets.QSpinBox()
        self.end_spin = QtWidgets.QSpinBox()
        maximum = max(0, int(frame_count) - 1)
        self.start_spin.setRange(0, maximum)
        self.end_spin.setRange(0, maximum)
        self.end_spin.setValue(maximum)
        self.seed_list = QtWidgets.QListWidget()
        self.seed_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.seed_list.setMinimumHeight(140)
        for option in seed_options:
            item = QtWidgets.QListWidgetItem(option["label"])
            item.setData(QtCore.Qt.UserRole, option)
            self.seed_list.addItem(item)
        self.replace_manual_check = QtWidgets.QCheckBox("允许替换同一 track_id 的人工对象")
        self.replace_manual_check.setChecked(False)
        layout.addRow("方向", self.direction_combo)
        layout.addRow("开始帧", self.start_spin)
        layout.addRow("结束帧", self.end_spin)
        layout.addRow("复核关键帧/纠正帧", self.seed_list)
        layout.addRow("冲突策略", self.replace_manual_check)
        hint = QtWidgets.QLabel("结果先进入暂存区；确认审阅后才会合并到 IVAS 标签。")
        hint.setWordWrap(True)
        layout.addRow(hint)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _accept_validated(self) -> None:
        if self.end_spin.value() < self.start_spin.value():
            QtWidgets.QMessageBox.warning(self, "范围无效", "结束帧不能早于开始帧。")
            return
        if not self.seed_list.selectedItems():
            QtWidgets.QMessageBox.warning(self, "缺少关键帧", "请选择至少一个已经复核的对象。")
            return
        self.accept()

    def values(self) -> Dict[str, Any]:
        return {
            "direction": str(self.direction_combo.currentData()),
            "start_frame": self.start_spin.value(),
            "end_frame": self.end_spin.value(),
            "seeds": [item.data(QtCore.Qt.UserRole) for item in self.seed_list.selectedItems()],
            "replace_manual": self.replace_manual_check.isChecked(),
        }
