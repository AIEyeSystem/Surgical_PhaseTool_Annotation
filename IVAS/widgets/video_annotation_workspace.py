"""Integrated video playback and video annotation workspace."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from PyQt5 import QtMultimedia, QtMultimediaWidgets
except ImportError:  # pragma: no cover - depends on the Qt distribution
    QtMultimedia = None
    QtMultimediaWidgets = None

from IVAS.video.project import (
    MediaRelinkRequired,
    TemporalAnnotation,
    TemporalTemplate,
    UnifiedVideoProjectStore,
    discover_video_projects,
    format_timecode,
    media_fingerprint,
    parse_interval_template,
    parse_rating_template,
)
from IVAS.video.media import (
    FrameTimestampMap,
    OperationCancelled,
    atomic_write_png,
    decode_frame,
    decode_frame_range,
    scan_frame_timestamps,
)
from IVAS.video.tracking import (
    DeterministicTrackingAdapter,
    SegAnyTrackingAdapter,
    TrackingRequest,
    TrackingSeed,
    VideoTrackingService,
)
from IVAS.theme import theme_color
from IVAS.annotation import Annotation
from IVAS.widgets.video_annotation_dialogs import (
    TemplateManagerDialog,
    TemporalAnnotationDialog,
    TrackingRangeDialog,
    TrackingReviewDialog,
)


def safe_identifier(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in str(value))
    return result.strip("_") or "object"


class FrameMapWorker(QtCore.QThread):
    ready = QtCore.pyqtSignal(int, object, str)
    failed = QtCore.pyqtSignal(int, str)
    progress = QtCore.pyqtSignal(int, int)

    def __init__(self, token: int, video_path: Path, output_path: Path) -> None:
        super().__init__()
        self.token = int(token)
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            mapping = scan_frame_timestamps(
                self.video_path,
                cancel_event=self.cancel_event,
                progress=lambda current, total: self.progress.emit(current, total),
            )
            if self.cancel_event.is_set():
                return
            mapping.save(self.output_path)
            self.ready.emit(self.token, mapping, str(self.output_path))
        except OperationCancelled:
            return
        except Exception as error:
            self.failed.emit(self.token, str(error))


class FrameExtractionWorker(QtCore.QThread):
    ready = QtCore.pyqtSignal(int, int, int, str, str)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        token: int,
        video_path: Path,
        frame_map: FrameTimestampMap,
        position_ms: int,
        frames_directory: Path,
        labels_directory: Path,
        cached_image_signature: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.token = int(token)
        self.video_path = Path(video_path)
        self.frame_map = frame_map
        self.position_ms = int(position_ms)
        self.frames_directory = Path(frames_directory)
        self.labels_directory = Path(labels_directory)
        self.cached_image_signature = cached_image_signature
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            frame_index = self.frame_map.frame_for_timestamp(self.position_ms)
            pts_ms = self.frame_map.timestamp_for_frame(frame_index)
            image_path = self.frames_directory / "frame_{:012d}.png".format(frame_index)
            label_path = self.labels_directory / "{}.json".format(image_path.stem)
            try:
                stat = image_path.stat()
                cached = self.cached_image_signature == (stat.st_size, stat.st_mtime_ns)
            except OSError:
                cached = False
            if cached:
                if not self.cancel_event.is_set():
                    self.ready.emit(self.token, frame_index, pts_ms, str(image_path), str(label_path))
                return
            frame = decode_frame(
                self.video_path, frame_index, cancel_event=self.cancel_event
            )
            if self.cancel_event.is_set():
                raise OperationCancelled("Frame extraction was cancelled")
            atomic_write_png(image_path, frame)
            self.ready.emit(
                self.token,
                frame_index,
                pts_ms,
                str(image_path),
                str(label_path),
            )
        except OperationCancelled:
            return
        except Exception as error:
            self.failed.emit(self.token, str(error))


class TrackingWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(int, str)
    failed = QtCore.pyqtSignal(int, str)
    progress = QtCore.pyqtSignal(int, int, str)

    def __init__(
        self,
        token: int,
        project_path: Path,
        request: TrackingRequest,
        adapter: Any,
    ) -> None:
        super().__init__()
        self.token = int(token)
        self.project_path = Path(project_path)
        self.request = request
        self.adapter = adapter
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            store = UnifiedVideoProjectStore.load(self.project_path)
            cache = (
                store.tracking_directory
                / "frame_cache"
                / store.media["fingerprint"][:12]
            )
            indices = list(range(self.request.start_frame, self.request.end_frame + 1))
            missing = [
                index
                for index in indices
                if not (cache / "frame_{:012d}.png".format(index)).is_file()
            ]
            if missing:
                decode_frame_range(
                    Path(store.media["path"]),
                    missing,
                    cache,
                    cancel_event=self.cancel_event,
                    progress=lambda current, total: self.progress.emit(
                        current, total, "decode"
                    ),
                )
            frame_paths = {
                index: cache / "frame_{:012d}.png".format(index) for index in indices
            }
            service = VideoTrackingService(store)
            run = service.stage(
                self.request,
                self.adapter,
                frame_paths,
                self.cancel_event,
                progress=lambda current, total: self.progress.emit(
                    current, total, "track"
                ),
            )
            self.completed.emit(self.token, str(run["id"]))
        except OperationCancelled:
            self.failed.emit(self.token, "cancelled")
        except Exception as error:
            self.failed.emit(self.token, str(error))


class TemporalTimeline(QtWidgets.QWidget):
    """Small multi-lane Phase/Task timeline with a seekable playhead."""

    seek_requested = QtCore.pyqtSignal(int)
    annotation_selected = QtCore.pyqtSignal(str)

    LANE_COLOUR_ROLES = {
        "phase": "timeline-phase",
        "task": "timeline-task",
    }
    KEYBOARD_SEEK_MS = 1_000

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._duration_ms = 0
        self._position_ms = 0
        self._view_start_ms = 0
        self._view_end_ms = 0
        self._annotations: List[TemporalAnnotation] = []
        self._selected_annotation_id: Optional[str] = None
        self.setMinimumHeight(94)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAccessibleName("阶段与动作时间轴")
        self.setAccessibleDescription(
            "左右方向键移动播放位置，上下方向键选择区间，Home 和 End 跳到首尾。"
        )
        self.setMouseTracking(True)
        self.setToolTip("单击跳转；单击区间可选择；方向键可进行键盘导航。")

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(0, int(duration_ms))
        self._view_start_ms = 0
        self._view_end_ms = self._duration_ms
        self.update()

    def set_position(self, position_ms: int) -> None:
        self._position_ms = max(0, int(position_ms))
        self.update()

    def set_annotations(self, annotations: Iterable[TemporalAnnotation]) -> None:
        self._annotations = list(annotations)
        available_ids = {annotation.id for annotation in self._annotations}
        if self._selected_annotation_id not in available_ids:
            self._selected_annotation_id = None
        self.update()

    def set_selected_annotation(self, annotation_id: Optional[str]) -> None:
        available_ids = {annotation.id for annotation in self._annotations}
        selected = str(annotation_id) if annotation_id in available_ids else None
        if selected == self._selected_annotation_id:
            return
        self._selected_annotation_id = selected
        self.update()

    def _content_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(56, 8, max(1, self.width() - 68), max(1, self.height() - 16))

    def _x_for_time(self, milliseconds: int) -> float:
        rect = self._content_rect()
        visible_duration = self._view_end_ms - self._view_start_ms
        if visible_duration <= 0:
            return rect.left()
        ratio = min(
            1.0,
            max(
                0.0,
                (milliseconds - self._view_start_ms) / float(visible_duration),
            ),
        )
        return rect.left() + rect.width() * ratio

    def _time_for_x(self, x: float) -> int:
        rect = self._content_rect()
        visible_duration = self._view_end_ms - self._view_start_ms
        if visible_duration <= 0 or rect.width() <= 0:
            return 0
        ratio = min(1.0, max(0.0, (x - rect.left()) / rect.width()))
        return int(round(self._view_start_ms + visible_duration * ratio))

    def reset_zoom(self) -> None:
        self._view_start_ms = 0
        self._view_end_ms = self._duration_ms
        self.update()

    def zoom_by(self, factor: float, anchor_ms: Optional[int] = None) -> None:
        if self._duration_ms <= 0 or factor <= 0:
            return
        current_span = max(1, self._view_end_ms - self._view_start_ms)
        minimum_span = min(self._duration_ms, 1_000)
        new_span = max(minimum_span, min(self._duration_ms, int(round(current_span / factor))))
        anchor = self._position_ms if anchor_ms is None else int(anchor_ms)
        ratio = (anchor - self._view_start_ms) / float(current_span)
        new_start = int(round(anchor - new_span * ratio))
        new_start = min(max(0, new_start), max(0, self._duration_ms - new_span))
        self._view_start_ms = new_start
        self._view_end_ms = new_start + new_span
        self.update()

    def pan_by(self, milliseconds: int) -> None:
        span = max(0, self._view_end_ms - self._view_start_ms)
        if span <= 0 or span >= self._duration_ms:
            return
        start = min(
            max(0, self._view_start_ms + int(milliseconds)),
            self._duration_ms - span,
        )
        self._view_start_ms = start
        self._view_end_ms = start + span
        self.update()

    @staticmethod
    def _lane(category: str) -> int:
        return 0 if category.casefold() == "phase" else 1

    def _lane_rect(self, lane: int) -> QtCore.QRectF:
        rect = self._content_rect()
        gap = 7.0
        lane_height = (rect.height() - gap) / 2.0
        return QtCore.QRectF(
            rect.left(), rect.top() + lane * (lane_height + gap), rect.width(), lane_height
        )

    def _annotation_rect(self, annotation: TemporalAnnotation) -> QtCore.QRectF:
        lane = self._lane(annotation.category)
        lane_rect = self._lane_rect(lane).adjusted(0, 3, 0, -3)
        left = self._x_for_time(annotation.start_ms)
        right = self._x_for_time(annotation.end_ms or annotation.start_ms)
        return QtCore.QRectF(
            left,
            lane_rect.top(),
            max(3.0, right - left),
            lane_rect.height(),
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), theme_color("canvas"))
        painter.setPen(theme_color("border-subtle"))
        for lane in (0, 1):
            lane_rect = self._lane_rect(lane)
            painter.setPen(theme_color("border-subtle"))
            painter.drawText(QtCore.QRectF(4, lane_rect.top(), 46, lane_rect.height()),
                             QtCore.Qt.AlignVCenter, ("Phase", "Task")[lane])
            painter.fillRect(lane_rect, theme_color("canvas-raised"))
            painter.setPen(theme_color("border"))
            painter.drawRoundedRect(lane_rect, 3, 3)

        for annotation in self._annotations:
            colour_role = self.LANE_COLOUR_ROLES.get(
                annotation.category.casefold(), "text-muted"
            )
            annotation_rect = self._annotation_rect(annotation)
            painter.setBrush(theme_color(colour_role))
            if annotation.id == self._selected_annotation_id:
                painter.setPen(QtGui.QPen(theme_color("on-primary"), 2))
            else:
                painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(annotation_rect, 3, 3)

            text_rect = annotation_rect.adjusted(5, 0, -5, 0)
            if text_rect.width() >= 12:
                painter.setPen(theme_color("canvas"))
                label = painter.fontMetrics().elidedText(
                    annotation.label,
                    QtCore.Qt.ElideRight,
                    max(1, int(text_rect.width())),
                )
                painter.drawText(
                    text_rect,
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                    label,
                )

        playhead_x = self._x_for_time(self._position_ms)
        painter.setPen(QtGui.QPen(theme_color("on-primary"), 2))
        painter.drawLine(
            QtCore.QPointF(playhead_x, self._content_rect().top()),
            QtCore.QPointF(playhead_x, self._content_rect().bottom()),
        )

        if self.hasFocus():
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(theme_color("focus"), 2))
            painter.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(1, 1, -2, -2), 4, 4)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(QtCore.Qt.MouseFocusReason)
        for annotation in reversed(self._annotations):
            if self._annotation_rect(annotation).contains(event.localPos()):
                self.set_selected_annotation(annotation.id)
                self.annotation_selected.emit(annotation.id)
                break
        self.seek_requested.emit(self._time_for_x(event.x()))
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if not delta:
            super().wheelEvent(event)
            return
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            span = max(1, self._view_end_ms - self._view_start_ms)
            self.pan_by(int((-1 if delta > 0 else 1) * span * 0.1))
        else:
            self.zoom_by(1.25 if delta > 0 else 0.8, self._time_for_x(event.pos().x()))
        event.accept()

    def _select_relative_annotation(self, offset: int) -> None:
        ordered = sorted(
            self._annotations,
            key=lambda annotation: (annotation.start_ms, annotation.category, annotation.label),
        )
        if not ordered:
            return
        selected_index = next(
            (
                index
                for index, annotation in enumerate(ordered)
                if annotation.id == self._selected_annotation_id
            ),
            -1 if offset > 0 else 0,
        )
        selected = ordered[(selected_index + offset) % len(ordered)]
        self.set_selected_annotation(selected.id)
        self.annotation_selected.emit(selected.id)
        self.seek_requested.emit(selected.start_ms)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        seek_step = self.KEYBOARD_SEEK_MS * (
            5 if event.modifiers() & QtCore.Qt.ShiftModifier else 1
        )
        if key == QtCore.Qt.Key_Left:
            self.seek_requested.emit(max(0, self._position_ms - seek_step))
        elif key == QtCore.Qt.Key_Right:
            self.seek_requested.emit(min(self._duration_ms, self._position_ms + seek_step))
        elif key == QtCore.Qt.Key_Home:
            self.seek_requested.emit(0)
        elif key == QtCore.Qt.Key_End:
            self.seek_requested.emit(self._duration_ms)
        elif key == QtCore.Qt.Key_Up:
            self._select_relative_annotation(-1)
        elif key == QtCore.Qt.Key_Down:
            self._select_relative_annotation(1)
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            selected = next(
                (
                    annotation
                    for annotation in self._annotations
                    if annotation.id == self._selected_annotation_id
                ),
                None,
            )
            if selected is None:
                self._select_relative_annotation(1)
            else:
                self.annotation_selected.emit(selected.id)
                self.seek_requested.emit(selected.start_ms)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.update()


class VideoAnnotationWorkspace(QtWidgets.QWidget):
    """Video editor that hands the indexed frame at the current position to IVAS."""

    frame_annotation_requested = QtCore.pyqtSignal(str, str, str)
    return_to_video_requested = QtCore.pyqtSignal()
    status_message = QtCore.pyqtSignal(str)
    output_root_changed = QtCore.pyqtSignal(str)
    context_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        output_root: Path,
        parent: Optional[QtWidgets.QWidget] = None,
        tracking_adapter_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.output_root = Path(output_root).expanduser().resolve()
        self.store: Optional[UnifiedVideoProjectStore] = None
        self.video_path: Optional[Path] = None
        self.fps = 0.0
        self.frame_count = 0
        self.frame_map: Optional[FrameTimestampMap] = None
        self._last_position_ms = 0
        self._pending_interval: Optional[Tuple[str, str, int]] = None
        self._scrubbing = False
        self._busy = False
        self._frame_edit_position: Optional[int] = None
        self._narrow_layout: Optional[bool] = None
        self._panels_detached = False
        self._transport_detached = False
        self._workspace_shortcuts: List[QtWidgets.QShortcut] = []
        self._row_by_annotation_id: Dict[str, int] = {}
        self._language = "zh"
        self._project_token = 0
        self._workers = set()
        self._frame_map_restart_requested = False
        self._frame_cache = {}
        self._frame_map_worker: Optional[FrameMapWorker] = None
        self._extraction_worker: Optional[FrameExtractionWorker] = None
        self._frame_extraction_intent = "spatial"
        self._keyframe_template_id = ""
        self._tracking_worker: Optional[TrackingWorker] = None
        self._tracking_adapter_provider = tracking_adapter_provider
        self._template_manager: Optional[TemplateManagerDialog] = None
        self._staged_tracking_run_id = ""
        self.vlc_player = None
        self.vlc_instance = None
        self._vlc_dll_directory = None
        self._vlc_timer = QtCore.QTimer(self)
        self._vlc_timer.setInterval(100)
        self._vlc_timer.timeout.connect(self._poll_vlc)

        self._build_ui()
        self._build_player()
        self._bind_shortcuts()
        self._set_project_controls_enabled(False)
        self._update_responsive_layout(self.width())
        self.set_language(self._language)

    def _t(self, chinese: Any, english: Any) -> Any:
        return chinese if self._language == "zh" else english

    def set_language(self, language: str) -> None:
        """Translate the hand-written workspace without regenerating Qt UI files."""

        self._language = "zh" if str(language).casefold().startswith("zh") else "en"
        self.title_label.setText(self._t("视频阶段与帧分割", "Video phases and frame masks"))
        if self.store is None:
            self.project_label.setText(self._t("尚未打开视频", "No video is open"))
        self.open_button.setText(self._t("打开视频…", "Open video…"))
        self.undo_action.setText(self._t("撤销", "Undo"))
        self.redo_action.setText(self._t("重做", "Redo"))
        self.delete_action.setText(self._t("删除所选标注", "Delete selected annotation"))
        self.open_project_action.setText(self._t("打开已有项目…", "Open existing project…"))
        self.output_root_action.setText(
            self._t("设置新项目输出目录…", "Set new-project output directory…")
        )
        self.relink_media_action.setText(self._t("重新关联视频…", "Relink media…"))
        self.save_project_action.setText(self._t("保存视频项目", "Save video project"))
        self.template_action.setText(self._t("管理模板…", "Manage templates…"))
        self.import_template_action.setText(self._t("导入模板 TXT…", "Import template TXT…"))
        self.add_point_action.setText(self._t("新增关键帧 / 点…", "Add key frame / point…"))
        self.capture_keyframe_action.setText(
            self._t("截取当前帧为关键帧", "Capture current frame as key frame")
        )
        self.add_rating_action.setText(self._t("新增评分…", "Add rating…"))
        self.edit_annotation_action.setText(self._t("编辑所选标注…", "Edit selected annotation…"))
        self.discard_pending_interval_action.setText(
            self._t("放弃未完成区间", "Discard unfinished interval")
        )
        self.validate_action.setText(self._t("校验与覆盖率…", "Validation and coverage…"))
        self.import_action.setText(self._t("导入 CSV / TXT…", "Import CSV / TXT…"))
        self.export_action.setText(self._t("导出 CSV / TXT…", "Export CSV / TXT…"))
        self.seek_backward_5s_action.setText(self._t("后退 5 秒", "Seek backward 5 seconds"))
        self.seek_backward_1s_action.setText(self._t("后退 1 秒", "Seek backward 1 second"))
        self.seek_forward_1s_action.setText(self._t("前进 1 秒", "Seek forward 1 second"))
        self.seek_forward_5s_action.setText(self._t("前进 5 秒", "Seek forward 5 seconds"))
        self.timeline_zoom_in_action.setText(self._t("时间轴放大", "Zoom timeline in"))
        self.timeline_zoom_out_action.setText(self._t("时间轴缩小", "Zoom timeline out"))
        self.timeline_reset_zoom_action.setText(
            self._t("重置时间轴缩放", "Reset timeline zoom")
        )
        self.rebuild_frame_map_action.setText(self._t("重建精确帧时间索引", "Rebuild exact frame-time index"))
        self.cancel_worker_action.setText(self._t("取消后台任务", "Cancel background task"))
        self.start_tracking_action.setText(self._t("开始事务化追踪…", "Start transactional tracking…"))
        self.review_tracking_action.setText(self._t("审阅 / 提交暂存追踪…", "Review / commit staged tracking…"))
        self.rollback_tracking_action.setText(self._t("回滚最近追踪提交", "Roll back latest tracking commit"))
        self.more_button.setText(self._t("更多", "More"))
        self.rating_button.setText(self._t("快速评分", "Quick rating"))
        action_status_tips = (
            (
                self.capture_keyframe_action,
                self._t(
                    "提取当前位置的精确帧，并保存为带图像链接的关键帧标注",
                    "Extract the exact current frame and save a key-frame annotation linked to its image",
                ),
            ),
            (
                self.save_project_action,
                self._t(
                    "保存统一项目清单，并刷新病例目录中的 CSV / TXT 导出",
                    "Save the unified manifest and refresh the case-local CSV / TXT exports",
                ),
            ),
            (
                self.discard_pending_interval_action,
                self._t(
                    "不保存并清除使用 I 开始的当前区间",
                    "Clear the interval started with I without saving it",
                ),
            ),
            (self.seek_backward_5s_action, self._t("播放位置后退 5 秒", "Move the playhead backward 5 seconds")),
            (self.seek_backward_1s_action, self._t("播放位置后退 1 秒", "Move the playhead backward 1 second")),
            (self.seek_forward_1s_action, self._t("播放位置前进 1 秒", "Move the playhead forward 1 second")),
            (self.seek_forward_5s_action, self._t("播放位置前进 5 秒", "Move the playhead forward 5 seconds")),
            (self.timeline_zoom_in_action, self._t("以当前播放位置为中心放大时间轴", "Zoom the timeline around the playhead")),
            (self.timeline_zoom_out_action, self._t("以当前播放位置为中心缩小时间轴", "Zoom the timeline out around the playhead")),
            (self.timeline_reset_zoom_action, self._t("显示完整视频时间范围", "Show the complete video time range")),
        )
        for action, status_tip in action_status_tips:
            action.setStatusTip(status_tip)
            action.setToolTip(status_tip)
        self._rebuild_rating_menu()
        self.back_button.setText(
            "◀" if self._narrow_layout else self._t("◀帧", "◀ Frame")
        )
        self.forward_button.setText(
            "▶" if self._narrow_layout else self._t("帧▶", "Frame ▶")
        )
        playing = bool(
            (self.vlc_player is not None and bool(self.vlc_player.is_playing()))
            or (
                self.player is not None
                and self.player.state() == QtMultimedia.QMediaPlayer.PlayingState
            )
        )
        self.play_button.setText(
            self._t("返回视频", "Return to video") if self._frame_edit_position is not None
            else self._t("暂停", "Pause") if playing else self._t("播放", "Play")
        )
        self.annotate_frame_button.setText(
            self._t("正在提取…", "Extracting…")
            if self._busy
            else self._t("标注当前帧", "Annotate frame")
        )
        self.interval_group.setTitle(self._t("阶段 / 动作区间", "Phase / task interval"))
        self.category_label.setText(self._t("轨道", "Lane"))
        self.annotation_label.setText(self._t("标签", "Label"))
        self.mark_start_button.setText(self._t("记录开始  I", "Mark start  I"))
        self.mark_end_button.setText(self._t("记录结束  O", "Mark end  O"))
        self._update_pending_label()
        self.annotation_table.setHorizontalHeaderLabels(
            self._t(
                ["类型", "分类", "标签", "开始", "结束", "状态"],
                ["Kind", "Category", "Label", "Start", "End", "Status"],
            )
        )
        if isinstance(self.video_widget, QtWidgets.QLabel) and self.player is None:
            self.video_widget.setText(
                self._t(
                    "当前 PyQt5 安装不包含 Qt Multimedia",
                    "This PyQt5 installation has no Qt Multimedia",
                )
            )
        self._update_position_label(self.position_ms())

    transport_layout_changed = QtCore.pyqtSignal()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("视频阶段与帧分割")
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        header.addWidget(self.title_label)
        self.project_label = QtWidgets.QLabel("尚未打开视频")
        self.project_label.setObjectName("workspaceMeta")
        self.project_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.project_label.setMinimumWidth(0)
        self.project_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        header.addWidget(self.project_label, 1)
        self.open_button = QtWidgets.QPushButton("打开视频…")
        self.open_button.setAccessibleName("打开视频项目")
        self.open_button.clicked.connect(self.open_video_dialog)
        header.addWidget(self.open_button)

        self.more_menu = QtWidgets.QMenu(self)
        self.open_project_action = self.more_menu.addAction("打开已有项目…")
        self.open_project_action.triggered.connect(self.open_project_dialog)
        self.output_root_action = self.more_menu.addAction("设置新项目输出目录…")
        self.output_root_action.triggered.connect(self.choose_output_root_dialog)
        self.output_root_action.setToolTip(str(self.output_root))
        self.relink_media_action = self.more_menu.addAction("重新关联视频…")
        self.relink_media_action.triggered.connect(self.relink_media_dialog)
        self.save_project_action = self.more_menu.addAction("保存视频项目")
        self.save_project_action.triggered.connect(self.save_project)
        self.more_menu.addSeparator()
        self.template_action = self.more_menu.addAction("管理模板…")
        self.template_action.triggered.connect(self.manage_templates)
        self.import_template_action = self.more_menu.addAction("导入模板 TXT…")
        self.import_template_action.triggered.connect(self.import_template_dialog)
        self.add_point_action = self.more_menu.addAction("新增关键帧 / 点…")
        self.add_point_action.triggered.connect(lambda: self.add_temporal_dialog("point"))
        self.capture_keyframe_action = self.more_menu.addAction("截取当前帧为关键帧")
        self.capture_keyframe_action.triggered.connect(self.capture_current_keyframe)
        self.add_rating_action = self.more_menu.addAction("新增评分…")
        self.add_rating_action.triggered.connect(lambda: self.add_temporal_dialog("rating"))
        self.edit_annotation_action = self.more_menu.addAction("编辑所选标注…")
        self.edit_annotation_action.triggered.connect(self.edit_selected_annotation)
        self.discard_pending_interval_action = self.more_menu.addAction("放弃未完成区间")
        self.discard_pending_interval_action.triggered.connect(self.discard_pending_interval)
        self.more_menu.addSeparator()
        self.validate_action = self.more_menu.addAction("校验与覆盖率…")
        self.validate_action.triggered.connect(self.show_validation_report)
        self.import_action = self.more_menu.addAction("导入 CSV / TXT…")
        self.import_action.triggered.connect(self.import_annotations_dialog)
        self.export_action = self.more_menu.addAction("导出 CSV / TXT…")
        self.export_action.triggered.connect(self.export_annotations_dialog)
        self.more_menu.addSeparator()
        self.seek_backward_5s_action = self.more_menu.addAction("后退 5 秒")
        self.seek_backward_5s_action.triggered.connect(lambda: self.seek_relative(-5_000))
        self.seek_backward_1s_action = self.more_menu.addAction("后退 1 秒")
        self.seek_backward_1s_action.triggered.connect(lambda: self.seek_relative(-1_000))
        self.seek_forward_1s_action = self.more_menu.addAction("前进 1 秒")
        self.seek_forward_1s_action.triggered.connect(lambda: self.seek_relative(1_000))
        self.seek_forward_5s_action = self.more_menu.addAction("前进 5 秒")
        self.seek_forward_5s_action.triggered.connect(lambda: self.seek_relative(5_000))
        self.more_menu.addSeparator()
        self.timeline_zoom_in_action = self.more_menu.addAction("时间轴放大")
        self.timeline_zoom_in_action.triggered.connect(
            lambda: self.timeline.zoom_by(2.0, self.position_ms())
        )
        self.timeline_zoom_out_action = self.more_menu.addAction("时间轴缩小")
        self.timeline_zoom_out_action.triggered.connect(
            lambda: self.timeline.zoom_by(0.5, self.position_ms())
        )
        self.timeline_reset_zoom_action = self.more_menu.addAction("重置时间轴缩放")
        self.timeline_reset_zoom_action.triggered.connect(
            lambda: self.timeline.reset_zoom()
        )
        self.more_menu.addSeparator()
        self.rebuild_frame_map_action = self.more_menu.addAction("重建精确帧时间索引")
        self.rebuild_frame_map_action.triggered.connect(lambda: self._start_frame_map_scan(force=True))
        self.cancel_worker_action = self.more_menu.addAction("取消后台任务")
        self.cancel_worker_action.triggered.connect(self.cancel_background_work)
        self.cancel_worker_action.setEnabled(False)
        self.more_menu.addSeparator()
        self.start_tracking_action = self.more_menu.addAction("开始事务化追踪…")
        self.start_tracking_action.triggered.connect(self.start_tracking_dialog)
        self.review_tracking_action = self.more_menu.addAction("审阅 / 提交暂存追踪…")
        self.review_tracking_action.triggered.connect(self.review_staged_tracking)
        self.rollback_tracking_action = self.more_menu.addAction("回滚最近追踪提交")
        self.rollback_tracking_action.triggered.connect(self.rollback_latest_tracking)
        self.more_menu.addSeparator()
        self.undo_action = self.more_menu.addAction("撤销")
        self.undo_action.triggered.connect(self.undo)
        self.redo_action = self.more_menu.addAction("重做")
        self.redo_action.triggered.connect(self.redo)
        self.more_menu.addSeparator()
        self.delete_action = self.more_menu.addAction("删除所选标注")
        self.delete_action.triggered.connect(self.delete_selected_annotation)
        self.more_button = QtWidgets.QToolButton()
        self.more_button.setText("更多")
        self.more_button.setAccessibleName("更多视频标注操作")
        self.more_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.more_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.more_button.setMenu(self.more_menu)
        header.addWidget(self.more_button)
        root.addLayout(header)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        self.video_panel = QtWidgets.QWidget()
        self.video_layout = QtWidgets.QVBoxLayout(self.video_panel)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(7)
        self.video_frame = QtWidgets.QFrame()
        self.video_frame.setObjectName("videoAnnotationStage")
        self.video_frame_layout = QtWidgets.QVBoxLayout(self.video_frame)
        self.video_frame_layout.setContentsMargins(1, 1, 1, 1)
        self.video_layout.addWidget(self.video_frame, 1)

        self.transport_widget = QtWidgets.QWidget(self.video_panel)
        self.transport_widget.setObjectName("sharedVideoTransport")
        self.transport_widget.setAccessibleName("视频播放与逐帧定位")
        self.transport_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        transport = QtWidgets.QGridLayout(self.transport_widget)
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setHorizontalSpacing(6)
        transport.setVerticalSpacing(4)
        self.back_button = QtWidgets.QToolButton()
        self.back_button.setText("◀帧")
        self.back_button.setAccessibleName("上一帧")
        self.back_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.back_button.clicked.connect(lambda: self.step_frames(-1))
        transport.addWidget(self.back_button, 0, 0)
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        transport.addWidget(self.play_button, 0, 1)
        self.forward_button = QtWidgets.QToolButton()
        self.forward_button.setText("帧▶")
        self.forward_button.setAccessibleName("下一帧")
        self.forward_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.forward_button.clicked.connect(lambda: self.step_frames(1))
        transport.addWidget(self.forward_button, 0, 2)
        self.position_label = QtWidgets.QLabel("00:00:00.000 / 00:00:00.000")
        self.position_label.setMinimumWidth(0)
        self.position_label.setAlignment(QtCore.Qt.AlignCenter)
        self.position_label.setAccessibleName("当前时间与视频时长")
        transport.addWidget(self.position_label, 0, 3)
        self.rate_combo = QtWidgets.QComboBox()
        self.rate_combo.setAccessibleName("播放速度")
        for label, value in (("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("1.5×", 1.5), ("2×", 2.0)):
            self.rate_combo.addItem(label, value)
        self.rate_combo.setCurrentIndex(self.rate_combo.findData(1.0))
        self.rate_combo.currentIndexChanged.connect(self.set_playback_rate)
        transport.addWidget(self.rate_combo, 0, 4)
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.position_slider.setAccessibleName("视频播放位置")
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._begin_scrub)
        self.position_slider.sliderReleased.connect(self._end_scrub)
        self.position_slider.sliderMoved.connect(self._preview_scrub)
        transport.addWidget(self.position_slider, 1, 0, 1, 6)
        self.annotate_frame_button = QtWidgets.QPushButton("标注当前帧")
        self.annotate_frame_button.setObjectName("primaryAction")
        self.annotate_frame_button.setToolTip(
            "提取当前位置对应的索引帧，并在 IVAS 多边形/SAM 画布中打开"
        )
        self.annotate_frame_button.clicked.connect(self.annotate_current_frame)
        transport.addWidget(self.annotate_frame_button, 0, 5)
        transport.setColumnStretch(3, 1)
        self.video_layout.addWidget(self.transport_widget)
        self.splitter.addWidget(self.video_panel)
        self._transport_compact = None
        self.transport_widget.installEventFilter(self)

        self.editor_panel = QtWidgets.QWidget()
        self.editor_panel.setMinimumWidth(340)
        editor_layout = QtWidgets.QVBoxLayout(self.editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(8)

        self.interval_group = QtWidgets.QGroupBox("阶段 / 动作区间")
        self.form_layout = QtWidgets.QGridLayout(self.interval_group)
        form_layout = self.form_layout
        self.category_label = QtWidgets.QLabel("轨道")
        form_layout.addWidget(self.category_label, 0, 0)
        self.category_combo = QtWidgets.QComboBox()
        self.category_label.setBuddy(self.category_combo)
        self.category_combo.addItem("Phase", "phase")
        self.category_combo.addItem("Task", "task")
        self.category_combo.currentIndexChanged.connect(self._refresh_label_templates)
        form_layout.addWidget(self.category_combo, 0, 1)
        self.annotation_label = QtWidgets.QLabel("标签")
        form_layout.addWidget(self.annotation_label, 1, 0)
        self.label_combo = QtWidgets.QComboBox()
        self.annotation_label.setBuddy(self.label_combo)
        self.label_combo.setEditable(True)
        self.label_combo.addItems(
            ["Preparation", "Incision", "Capsulorhexis", "Irrigation/Aspiration"]
        )
        self.label_combo.setCurrentText("")
        form_layout.addWidget(self.label_combo, 1, 1, 1, 3)
        self.mark_start_button = QtWidgets.QPushButton("记录开始  I")
        self.mark_start_button.clicked.connect(self.mark_start)
        form_layout.addWidget(self.mark_start_button, 2, 0, 1, 2)
        self.mark_end_button = QtWidgets.QPushButton("记录结束  O")
        self.mark_end_button.clicked.connect(self.mark_end)
        form_layout.addWidget(self.mark_end_button, 2, 2, 1, 2)
        self.pending_label = QtWidgets.QLabel("没有进行中的区间")
        self.pending_label.setWordWrap(True)
        form_layout.addWidget(self.pending_label, 3, 0, 1, 4)
        editor_layout.addWidget(self.interval_group)

        rating_row = QtWidgets.QHBoxLayout()
        rating_row.setContentsMargins(0, 0, 0, 0)
        self.coverage_label = QtWidgets.QLabel("阶段覆盖 0% · 待复核 0")
        self.coverage_label.setObjectName("workspaceMeta")
        self.coverage_label.setWordWrap(True)
        self.coverage_label.setAccessibleName("项目覆盖率和审阅状态")
        rating_row.addWidget(self.coverage_label, 1)
        self.rating_menu = QtWidgets.QMenu(self)
        self.rating_button = QtWidgets.QToolButton()
        self.rating_button.setText("快速评分")
        self.rating_button.setAccessibleName("快速评分")
        self.rating_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.rating_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.rating_button.setMenu(self.rating_menu)
        rating_row.addWidget(self.rating_button)
        editor_layout.addLayout(rating_row)

        self.annotation_table = QtWidgets.QTableWidget(0, 6)
        self.annotation_table.setAccessibleName("已保存的阶段与动作区间")
        self.annotation_table.setHorizontalHeaderLabels(["类型", "分类", "标签", "开始", "结束", "状态"])
        self.annotation_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.annotation_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        self.annotation_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch
        )
        self.annotation_table.itemSelectionChanged.connect(self._select_table_annotation)
        self.annotation_table.itemActivated.connect(self._activate_table_item)
        editor_layout.addWidget(self.annotation_table, 1)

        # Compatibility aliases for integrations that inspected the old buttons.
        self.undo_button = self.undo_action
        self.redo_button = self.redo_action
        self.delete_button = self.delete_action
        self.splitter.addWidget(self.editor_panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 3)

        self.timeline = TemporalTimeline()
        self.timeline.seek_requested.connect(self.seek)
        self.timeline.annotation_selected.connect(self.select_annotation)
        root.addWidget(self.timeline)

    def _build_player(self) -> None:
        if QtMultimedia is None or QtMultimediaWidgets is None:
            self.video_widget = QtWidgets.QLabel("当前 PyQt5 安装不包含 Qt Multimedia")
            self.video_widget.setAlignment(QtCore.Qt.AlignCenter)
            fallback_palette = self.video_widget.palette()
            fallback_palette.setColor(
                QtGui.QPalette.WindowText, theme_color("border-subtle")
            )
            self.video_widget.setPalette(fallback_palette)
            self.video_frame_layout.addWidget(self.video_widget)
            self.player = None
            return
        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.video_widget.setMinimumHeight(70)
        self.video_frame_layout.addWidget(self.video_widget)
        self.player = QtMultimedia.QMediaPlayer(
            self, QtMultimedia.QMediaPlayer.VideoSurface
        )
        self.player.setVideoOutput(self.video_widget)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.stateChanged.connect(self._on_state_changed)
        try:
            self.player.error.connect(self._on_player_error)
        except Exception:
            pass

    def _activate_vlc_fallback(self, media_path: Path) -> bool:
        if self.vlc_player is not None:
            try:
                self.vlc_player.stop()
            except Exception:
                pass
        try:
            import sys

            runtime_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            candidates = [
                runtime_root / "vlc",
                Path(__file__).resolve().parents[2]
                / "ref"
                / "Surgical_Video_Phrase_Tasks_Annotation",
            ]
            vlc_root = next(
                (candidate for candidate in candidates if (candidate / "libvlc.dll").is_file()),
                None,
            )
            if vlc_root is not None:
                os.environ.setdefault("PYTHON_VLC_LIB_PATH", str(vlc_root / "libvlc.dll"))
                os.environ.setdefault("PYTHON_VLC_MODULE_PATH", str(vlc_root / "plugins"))
                os.environ.setdefault("VLC_PLUGIN_PATH", str(vlc_root / "plugins"))
                if hasattr(os, "add_dll_directory"):
                    if self._vlc_dll_directory is not None:
                        self._vlc_dll_directory.close()
                    self._vlc_dll_directory = os.add_dll_directory(str(vlc_root))
            import vlc

            self.vlc_instance = vlc.Instance("--no-video-title-show", "--quiet")
            self.vlc_player = self.vlc_instance.media_player_new()
            media = self.vlc_instance.media_new(str(Path(media_path)))
            self.vlc_player.set_media(media)
            window_id = int(self.video_widget.winId())
            if os.name == "nt":
                self.vlc_player.set_hwnd(window_id)
            elif hasattr(self.vlc_player, "set_xwindow"):
                self.vlc_player.set_xwindow(window_id)
            elif hasattr(self.vlc_player, "set_nsobject"):
                self.vlc_player.set_nsobject(window_id)
            self.vlc_player.set_rate(float(self.rate_combo.currentData() or 1.0))
            self._vlc_timer.start()
            self.status_message.emit("Qt 播放失败，已切换到 VLC 后备播放器。")
            return True
        except Exception as error:
            self.vlc_player = None
            self.vlc_instance = None
            if self._vlc_dll_directory is not None:
                self._vlc_dll_directory.close()
                self._vlc_dll_directory = None
            self.status_message.emit("VLC 后备播放器不可用：{}".format(error))
            return False

    def _load_player_media(self, media_path: Path) -> None:
        if self.player is not None:
            media = QtMultimedia.QMediaContent(
                QtCore.QUrl.fromLocalFile(str(Path(media_path)))
            )
            self.player.setMedia(media)
            self.player.setPlaybackRate(float(self.rate_combo.currentData() or 1.0))
            self.player.setPosition(0)
            return
        self._activate_vlc_fallback(media_path)

    def _poll_vlc(self) -> None:
        if self.vlc_player is None:
            self._vlc_timer.stop()
            return
        try:
            position = int(self.vlc_player.get_time())
            duration = int(self.vlc_player.get_length())
        except Exception:
            return
        if duration > 0 and duration != self.duration_ms():
            self._on_duration_changed(duration)
        if position >= 0:
            self._on_position_changed(position)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "transport_widget", None) and event.type() in (
            QtCore.QEvent.Resize, QtCore.QEvent.LayoutRequest, QtCore.QEvent.Show,
        ):
            self._update_transport_layout(watched.width())
        return super().eventFilter(watched, event)

    def _update_transport_layout(self, width):
        controls = (self.back_button, self.play_button, self.forward_button,
                    self.position_label, self.rate_combo, self.annotate_frame_button)
        required = sum(control.sizeHint().width() for control in controls if not control.isHidden()) + 5 * 6
        compact = width < required
        if compact == self._transport_compact:
            return
        self._transport_compact = compact
        layout = self.transport_widget.layout()
        for control in (*controls, self.position_slider):
            layout.removeWidget(control)
        for column in range(6):
            layout.setColumnStretch(column, 0)
        if compact:
            for column, control in enumerate((self.back_button, self.play_button,
                                              self.forward_button, self.rate_combo)):
                layout.addWidget(control, 0, column)
            layout.addWidget(self.position_label, 1, 0, 1, 4 if self.annotate_frame_button.isHidden() else 2)
            layout.addWidget(self.annotate_frame_button, 1, 2, 1, 2)
            layout.addWidget(self.position_slider, 2, 0, 1, 4)
            layout.setColumnStretch(1, 1)
        else:
            for column, control in enumerate(controls):
                layout.addWidget(control, 0, column)
            layout.addWidget(self.position_slider, 1, 0, 1, 6)
            layout.setColumnStretch(3, 1)
        layout.invalidate()
        self.transport_widget.updateGeometry()
        self.transport_layout_changed.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout(event.size().width())

    def _update_responsive_layout(self, width: int) -> None:
        narrow = int(width) < 980
        if narrow == self._narrow_layout:
            return
        self._narrow_layout = narrow
        self.title_label.setVisible(not narrow)
        if self._panels_detached:
            self.back_button.setText(
                "◀" if narrow else self._t("◀帧", "◀ Frame")
            )
            self.forward_button.setText(
                "▶" if narrow else self._t("帧▶", "Frame ▶")
            )
            self.video_widget.setMinimumHeight(70)
            return

        self.editor_panel.setMinimumWidth(0 if narrow else 340)
        self.splitter.setOrientation(
            QtCore.Qt.Vertical if narrow else QtCore.Qt.Horizontal
        )
        if narrow:
            self.back_button.setText("◀")
            self.forward_button.setText("▶")
            self.video_widget.setMinimumHeight(70)
            self.timeline.setMinimumHeight(78)
            self._set_interval_form_narrow_layout()
            self.splitter.setStretchFactor(0, 3)
            self.splitter.setStretchFactor(1, 5)
            self.splitter.setSizes([135, 230])
        else:
            self.back_button.setText(self._t("◀帧", "◀ Frame"))
            self.forward_button.setText(self._t("帧▶", "Frame ▶"))
            self.video_widget.setMinimumHeight(70)
            self.timeline.setMinimumHeight(94)
            self._set_interval_form_wide_layout()
            self.splitter.setStretchFactor(0, 5)
            self.splitter.setStretchFactor(1, 3)
            self.splitter.setSizes([max(480, int(width) - 360), 340])

    def _set_interval_form_narrow_layout(self) -> None:
        self.form_layout.addWidget(self.category_label, 0, 0)
        self.form_layout.addWidget(self.category_combo, 0, 1)
        self.form_layout.addWidget(self.annotation_label, 0, 2)
        self.form_layout.addWidget(self.label_combo, 0, 3)
        self.form_layout.addWidget(self.mark_start_button, 1, 0, 1, 2)
        self.form_layout.addWidget(self.mark_end_button, 1, 2, 1, 2)
        self.form_layout.addWidget(self.pending_label, 2, 0, 1, 4)

    def _set_interval_form_wide_layout(self) -> None:
        self.form_layout.addWidget(self.category_label, 0, 0)
        self.form_layout.addWidget(self.category_combo, 0, 1)
        self.form_layout.addWidget(self.annotation_label, 1, 0)
        self.form_layout.addWidget(self.label_combo, 1, 1, 1, 3)
        self.form_layout.addWidget(self.mark_start_button, 2, 0, 1, 2)
        self.form_layout.addWidget(self.mark_end_button, 2, 2, 1, 2)
        self.form_layout.addWidget(self.pending_label, 3, 0, 1, 4)

    def detach_project_panels(
        self,
    ) -> Tuple[QtWidgets.QWidget, TemporalTimeline]:
        """Detach the project editor and timeline for hosting in external docks."""

        if self._panels_detached:
            return self.editor_panel, self.timeline

        self.editor_panel.hide()
        self.editor_panel.setParent(self)
        root_layout = self.layout()
        if root_layout is not None:
            root_layout.removeWidget(self.timeline)
        self.timeline.hide()
        self._panels_detached = True
        # The host supplies the open/project menus and interval toolbar.
        # Keep one visible entry point per operation, retaining the controls
        # as state adapters for standalone workspace use and signal bindings.
        for control in (
            self.open_button, self.more_button,
            self.category_label, self.category_combo,
            self.annotation_label, self.label_combo,
            self.mark_start_button, self.mark_end_button,
        ):
            control.hide()

        # Externally hosted panels keep a compact, width-independent form. The
        # remaining video surface can continue adapting without touching them.
        self.editor_panel.setMinimumWidth(0)
        self.timeline.setMinimumHeight(78)
        self._set_interval_form_narrow_layout()
        self.splitter.setStretchFactor(0, 1)
        return self.editor_panel, self.timeline

    def detach_transport(self) -> QtWidgets.QWidget:
        """Detach the one shared transport for hosting by the application shell.

        The controls themselves are moved, not copied, so playback state, seek
        position, timecode and playback rate remain continuous when the central
        surface changes between the video preview and an extracted IVAS frame.
        """

        if self._transport_detached:
            return self.transport_widget
        self.video_layout.removeWidget(self.transport_widget)
        self.transport_widget.hide()
        self._transport_detached = True
        # The unified shell already exposes this command in its primary bar;
        # keeping it inside the detached transport would duplicate the button.
        self.annotate_frame_button.hide()
        return self.transport_widget

    def set_local_shortcuts_enabled(self, enabled: bool) -> None:
        """Enable standalone shortcuts or yield them to an embedding window."""

        for shortcut in self._workspace_shortcuts:
            shortcut.setEnabled(bool(enabled))

    def _bind_shortcuts(self) -> None:
        for sequence, callback in (
            ("I", self.mark_start),
            ("O", self.mark_end),
            ("Space", self.toggle_playback),
            ("Left", lambda: self.step_frames(-1)),
            ("Right", lambda: self.step_frames(1)),
            ("Ctrl+Z", self.undo),
            ("Ctrl+Shift+Z", self.redo),
        ):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda callback=callback: self._activate_workspace_shortcut(callback)
            )
            self._workspace_shortcuts.append(shortcut)

    @staticmethod
    def _text_input_has_focus() -> bool:
        focused = QtWidgets.QApplication.focusWidget()
        return isinstance(
            focused,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QComboBox,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
            ),
        )

    def _activate_workspace_shortcut(self, callback) -> None:
        if not self._text_input_has_focus():
            callback()

    def _set_project_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.back_button,
            self.play_button,
            self.forward_button,
            self.position_slider,
            self.rate_combo,
            self.category_combo,
            self.label_combo,
            self.mark_start_button,
            self.mark_end_button,
            self.annotation_table,
            self.rating_button,
            self.undo_button,
            self.redo_button,
            self.delete_button,
        ):
            widget.setEnabled(enabled)
        self.more_button.setEnabled(True)
        for action in (
            self.relink_media_action,
            self.save_project_action,
            self.template_action,
            self.import_template_action,
            self.add_point_action,
            self.capture_keyframe_action,
            self.add_rating_action,
            self.edit_annotation_action,
            self.discard_pending_interval_action,
            self.validate_action,
            self.import_action,
            self.export_action,
            self.seek_backward_5s_action,
            self.seek_backward_1s_action,
            self.seek_forward_1s_action,
            self.seek_forward_5s_action,
            self.timeline_zoom_in_action,
            self.timeline_zoom_out_action,
            self.timeline_reset_zoom_action,
            self.rebuild_frame_map_action,
            self.start_tracking_action,
            self.review_tracking_action,
            self.rollback_tracking_action,
        ):
            action.setEnabled(enabled)
        self.discard_pending_interval_action.setEnabled(
            enabled and self._pending_interval is not None
        )
        self.capture_keyframe_action.setEnabled(
            enabled
            and not self._busy
            and bool(
                self.store
                and any(
                    template.kind == "point"
                    for template in self.store.temporal_templates
                )
            )
        )
        self.cancel_worker_action.setEnabled(self._background_work_running())
        self.annotate_frame_button.setEnabled(enabled and not self._busy)
        self._sync_transport_controls()
        self.context_changed.emit()

    def set_frame_edit_position(self, position_ms: Optional[int]) -> None:
        """Pin the shared playhead to the frame being edited until Video resumes."""
        position = None if position_ms is None else max(0, int(position_ms))
        if position == self._frame_edit_position:
            return
        previous = self._frame_edit_position
        self._frame_edit_position = position
        if position is not None:
            self.pause()
            self._on_position_changed(position)
        elif previous is not None:
            self.seek(previous)
        self._sync_transport_controls()
        self.context_changed.emit()

    def _sync_transport_controls(self) -> None:
        enabled = self.store is not None and not self._busy and self._frame_edit_position is None
        for control in (
            self.back_button, self.play_button, self.forward_button,
            self.position_slider, self.rate_combo,
            self.seek_backward_5s_action, self.seek_backward_1s_action,
            self.seek_forward_1s_action, self.seek_forward_5s_action,
        ):
            control.setEnabled(enabled)
        self.play_button.setEnabled(self.store is not None and not self._busy)
        if self._frame_edit_position is not None:
            self.play_button.setText(self._t("返回视频", "Return to video"))
        elif self.play_button.text() in ("返回视频", "Return to video"):
            self.play_button.setText(self._t("播放", "Play"))
        self.transport_widget.setToolTip(
            self._t("返回视频后继续播放或定位；当前分割帧已锁定。",
                    "Return to Video to play or seek; the annotation frame is pinned.")
            if self._frame_edit_position is not None else ""
        )

    def _background_work_running(self) -> bool:
        return any(worker.isRunning() for worker in self._workers)

    def _retain_worker(self, worker) -> None:
        # A cancelled old-project job can still be unwinding when a new one
        # starts. Keep it alive until Qt delivers finished; never wait in the UI.
        self._workers.add(worker)
        worker.finished.connect(self._release_worker)

    def _release_worker(self) -> None:
        worker = self.sender()
        self._workers.discard(worker)
        for name in ("_frame_map_worker", "_extraction_worker", "_tracking_worker"):
            if getattr(self, name) is worker:
                setattr(self, name, None)
        worker.deleteLater()
        if self._frame_map_restart_requested and self._frame_map_worker is None:
            self._frame_map_restart_requested = False
            self._start_frame_map_scan()

    def _set_frame_extraction_busy(self, busy: bool, message: str = "") -> None:
        self._busy = bool(busy)
        self.annotate_frame_button.setText(
            self._t("正在提取…", "Extracting…")
            if busy
            else self._t("标注当前帧", "Annotate frame")
        )
        self.annotate_frame_button.setEnabled(self.store is not None and not busy)
        self.capture_keyframe_action.setEnabled(
            self.store is not None
            and not busy
            and any(
                template.kind == "point"
                for template in self.store.temporal_templates
            )
        )
        if message:
            self.status_message.emit(message)
        self._sync_transport_controls()
        self.context_changed.emit()

    def _commit_store_mutation(
        self,
        operation: Callable[[], Any],
        error_title: str,
        validation_title: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        if self.store is None:
            return False, None
        manifest_saved = False
        try:
            with self.store.transaction():
                result = operation()
                if self.store.is_dirty:
                    self.store.save()
                    manifest_saved = True
        except Exception as error:
            title = (
                validation_title
                if validation_title and isinstance(error, ValueError)
                else error_title
            )
            self.status_message.emit("{}：{}".format(title, error))
            if validation_title and isinstance(error, ValueError):
                QtWidgets.QMessageBox.warning(self, title, str(error))
            else:
                QtWidgets.QMessageBox.critical(self, title, str(error))
            return False, None
        if manifest_saved:
            self._sync_default_exports(self.store)
        return True, result

    def _sync_default_exports(self, store: UnifiedVideoProjectStore) -> bool:
        """Refresh convenience exports without invalidating a saved manifest."""

        try:
            store.sync_default_exports()
            return True
        except Exception as error:
            self.status_message.emit(
                self._t(
                    "项目已保存，但自动刷新 CSV / TXT 失败：{}",
                    "The project was saved, but refreshing CSV / TXT exports failed: {}",
                ).format(error)
            )
            return False

    def save_project(self) -> bool:
        """Persist the unified manifest and its generated interchange views."""

        if self.store is None:
            return False
        try:
            self.store.save()
        except Exception as error:
            self.status_message.emit(
                self._t("保存视频项目失败：{}", "Saving the video project failed: {}").format(error)
            )
            QtWidgets.QMessageBox.critical(
                self,
                self._t("保存视频项目失败", "Save video project failed"),
                str(error),
            )
            return False
        exports_synced = self._sync_default_exports(self.store)
        if exports_synced:
            self.status_message.emit(
                self._t(
                    "已保存视频项目并刷新 CSV / TXT 导出。",
                    "Video project saved and CSV / TXT exports refreshed.",
                )
            )
        return True

    def _update_pending_label(self) -> None:
        if self._pending_interval is None:
            text = self._t("没有进行中的区间", "No interval in progress")
        else:
            category, label, start_ms = self._pending_interval
            text = self._t("进行中：{} / {}，开始于 {}", "In progress: {} / {}, started at {}").format(
                category.title(), label, format_timecode(start_ms)
            )
        self.pending_label.setText(text)

    def _clear_pending_interval(self) -> None:
        self._pending_interval = None
        self._update_pending_label()
        self.discard_pending_interval_action.setEnabled(False)
        self.context_changed.emit()

    def discard_pending_interval(self) -> bool:
        """Explicitly abandon the interval currently waiting for an end mark."""

        if self._pending_interval is None:
            return False
        category, label, _start_ms = self._pending_interval
        self._clear_pending_interval()
        self.status_message.emit(
            self._t(
                "已放弃未完成区间 {} / {}。",
                "Discarded unfinished interval {} / {}.",
            ).format(category.title(), label)
        )
        return True

    def _confirm_discard_pending_interval(self, action: str) -> bool:
        if self._pending_interval is None:
            return True
        category, label, start_ms = self._pending_interval
        answer = QtWidgets.QMessageBox.question(
            self,
            "未完成的区间",
            "{}会丢弃尚未结束的 {} / {}（开始于 {}）。是否继续？".format(
                action,
                category.title(),
                label,
                format_timecode(start_ms),
            ),
            QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        return answer == QtWidgets.QMessageBox.Discard

    def confirm_discard_pending(
        self, parent: Optional[QtWidgets.QWidget] = None
    ) -> bool:
        """Confirm and discard an unfinished interval before the workspace closes."""

        if self._pending_interval is None:
            return True
        category, label, start_ms = self._pending_interval
        answer = QtWidgets.QMessageBox.question(
            parent or self,
            "未完成的区间",
            "关闭会丢弃尚未结束的 {} / {}（开始于 {}）。是否继续？".format(
                category.title(), label, format_timecode(start_ms)
            ),
            QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Discard:
            return False
        self._clear_pending_interval()
        return True

    @staticmethod
    def probe_video(path: Path) -> Dict[str, object]:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("无法读取视频：{}".format(path))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_ms = int(round(frame_count * 1000.0 / fps)) if fps > 0 else 0
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_ms": duration_ms,
            }
        finally:
            capture.release()

    def open_video_dialog(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开视频",
            "",
            "Video (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if filename:
            try:
                self.open_video(Path(filename))
            except Exception as error:
                self.status_message.emit("打开视频失败：{}".format(error))
                QtWidgets.QMessageBox.critical(self, "打开视频失败", str(error))

    def open_project_dialog(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开已有统一视频项目",
            str(self.output_root),
            "IVAS video project (project.ivas.json project.ivat.json project.ivas.json *.json);;All files (*)",
        )
        if not filename:
            return
        try:
            self.open_project(Path(filename))
        except MediaRelinkRequired as error:
            media, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择项目对应的视频",
                str(error.recorded_path.parent),
                "Video (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
            )
            if not media:
                return
            try:
                self.open_project(Path(filename), Path(media))
            except MediaRelinkRequired:
                answer = QtWidgets.QMessageBox.warning(
                    self,
                    "视频身份不一致",
                    "所选视频与项目记录的指纹不同。继续会保留旧媒体身份历史，并显式绑定新视频。是否继续？",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Cancel,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return
                self.open_project(Path(filename), Path(media), allow_identity_change=True)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "打开项目失败", str(error))

    def choose_output_root_dialog(self) -> None:
        """Choose and remember where subsequently opened video cases are created."""

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self._t("选择新项目输出目录", "Choose new-project output directory"),
            str(self.output_root),
        )
        if not directory:
            return
        self.output_root = Path(directory).expanduser().resolve()
        self.output_root_action.setToolTip(str(self.output_root))
        self.output_root_changed.emit(str(self.output_root))
        message = self._t(
            "新视频项目将保存到：{}",
            "New video projects will be saved to: {}",
        ).format(self.output_root)
        if self.store is not None:
            message += self._t(
                "（当前项目位置不变）",
                " (the current project stays in place)",
            )
        self.status_message.emit(message)

    def open_project(
        self,
        project_path: Path,
        media_path: Optional[Path] = None,
        allow_identity_change: bool = False,
    ) -> UnifiedVideoProjectStore:
        if self._pending_interval is not None and not self._confirm_discard_pending_interval("打开项目"):
            return self.store
        if self.store is not None and self.store.is_dirty:
            self.store.save()
            self._sync_default_exports(self.store)
        store = UnifiedVideoProjectStore.open_existing(
            Path(project_path),
            media_path=media_path,
            allow_identity_change=allow_identity_change,
        )
        video_path = Path(store.media["path"]).resolve()
        metadata = self.probe_video(video_path)
        with store.transaction():
            store.update_media_info(**metadata)
            if store.is_dirty:
                store.save()
        self._sync_default_exports(store)
        self._activate_store(store, video_path, metadata)
        return store

    def relink_media_dialog(self) -> None:
        if self.store is None:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "重新关联项目视频",
            str(Path(self.store.media["path"]).parent),
            "Video (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if not filename:
            return
        candidate = Path(filename)
        allow_identity_change = media_fingerprint(candidate) != self.store.media["fingerprint"]
        if allow_identity_change:
            answer = QtWidgets.QMessageBox.warning(
                self,
                "视频身份不一致",
                "新视频指纹与项目记录不同。只有确认这是同一病例的替代媒体时才应继续。",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Cancel,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        try:
            with self.store.transaction():
                self.store.relink_media(candidate, allow_identity_change=allow_identity_change)
                metadata = self.probe_video(candidate)
                self.store.update_media_info(**metadata)
                self.store.save()
            self._sync_default_exports(self.store)
            self._activate_store(self.store, candidate.resolve(), metadata)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "重新关联失败", str(error))

    def open_video(self, path: Path) -> UnifiedVideoProjectStore:
        video_path = Path(path).resolve()
        if (
            self.store is not None
            and self.video_path is not None
            and self.video_path.resolve() == video_path
        ):
            return self.store

        discard_pending = self._pending_interval is not None
        if discard_pending and not self._confirm_discard_pending_interval("打开新视频"):
            # Returning the active store keeps callers compatible while making the
            # cancelled replacement a no-op.
            if self.store is not None:
                return self.store

        if self.store is not None and self.store.is_dirty:
            self.store.save()
            self._sync_default_exports(self.store)
        metadata = self.probe_video(video_path)
        store = UnifiedVideoProjectStore.open_or_create(
            self.output_root, video_path, **metadata
        )
        if store.is_dirty:
            store.save()
            self._sync_default_exports(store)
        if discard_pending:
            self._clear_pending_interval()
        self._activate_store(store, video_path, metadata)
        return store

    def _activate_store(
        self,
        store: UnifiedVideoProjectStore,
        video_path: Path,
        metadata: Dict[str, object],
    ) -> None:
        self.cancel_background_work(invalidate=True)
        self.store = store
        self.video_path = Path(video_path).resolve()
        self.fps = float(metadata["fps"])
        self.frame_count = int(metadata["frame_count"])
        self._last_position_ms = 0
        self._frame_edit_position = None
        self._frame_cache.clear()
        self.frame_map = None
        frame_map_info = store.project.get("frame_map", {})
        if frame_map_info.get("media_fingerprint") == store.media["fingerprint"]:
            try:
                cached_path = self._resolve_project_asset(frame_map_info["asset_path"])
                self.frame_map = FrameTimestampMap.load(cached_path)
            except Exception:
                self.frame_map = None
        if self.frame_map is None and self.frame_count and self.fps > 0:
            self.frame_map = FrameTimestampMap.from_fps(self.frame_count, self.fps)

        self._load_player_media(self.video_path)
        self.position_slider.setRange(0, int(metadata["duration_ms"]))
        self.timeline.set_duration(int(metadata["duration_ms"]))
        self.project_label.setText(video_path.name)
        self.project_label.setToolTip(
            "{}\n{}".format(video_path.resolve(), store.path.resolve())
        )
        self._set_project_controls_enabled(True)
        self._refresh_project_views()
        self.status_message.emit("已打开视频项目：{}".format(store.path))
        if not frame_map_info or frame_map_info.get("media_fingerprint") != store.media["fingerprint"]:
            self._start_frame_map_scan()

    def _start_frame_map_scan(self, force: bool = False) -> None:
        if self.store is None or self.video_path is None:
            return
        if self._frame_map_worker is not None and self._frame_map_worker.isRunning():
            if not force and self._frame_map_worker.token == self._project_token:
                return
            self._frame_map_worker.cancel()
            self._frame_map_restart_requested = True
            return
        token = self._project_token
        output_path = self.store.case_directory / "frame_timestamps.json"
        worker = FrameMapWorker(token, self.video_path, output_path)
        worker.ready.connect(self._on_frame_map_ready)
        worker.failed.connect(self._on_frame_map_failed)
        worker.progress.connect(self._on_frame_map_progress)
        worker.finished.connect(self._background_worker_finished)
        self._retain_worker(worker)
        self._frame_map_worker = worker
        self.cancel_worker_action.setEnabled(True)
        worker.start()

    def _on_frame_map_progress(self, current: int, total: int) -> None:
        worker = self.sender()
        if (worker is self._frame_map_worker and worker.token == self._project_token
                and not worker.cancel_event.is_set() and not self._busy
                and self._frame_edit_position is None):
            self.status_message.emit("正在建立 PTS 帧索引：{} / {}".format(current, total or "?"))

    def _on_frame_map_ready(
        self, token: int, mapping: FrameTimestampMap, asset_path: str
    ) -> None:
        if (
            token != self._project_token
            or self.store is None
            or self.sender() is not self._frame_map_worker
            or self._frame_map_worker.cancel_event.is_set()
        ):
            return
        saved, _result = self._commit_store_mutation(
            lambda: self.store.set_frame_map(
                Path(asset_path),
                len(mapping),
                mapping.duration_ms,
                mapping.method,
            ),
            "保存 PTS 帧索引失败",
        )
        if saved:
            self.frame_map = mapping
            self.frame_count = len(mapping)
            self.status_message.emit(
                "PTS 帧索引已就绪：{} 帧（{}）。".format(len(mapping), mapping.method)
            )

    def _on_frame_map_failed(self, token: int, message: str) -> None:
        if token != self._project_token or self.sender() is not self._frame_map_worker:
            return
        self.status_message.emit("PTS 索引失败，继续使用 FPS 后备映射：{}".format(message))

    def _background_worker_finished(self) -> None:
        self.cancel_worker_action.setEnabled(self._background_work_running())

    def cancel_background_work(self, invalidate: bool = False) -> None:
        if invalidate:
            self._project_token += 1
        self._frame_map_restart_requested = False
        for worker in self._workers:
            worker.cancel()
        self.status_message.emit("已请求取消后台视频任务。")

    def cancel_frame_extraction(self) -> None:
        """Cancel even a queued completion when a standalone image is opened."""
        if self._extraction_worker is not None:
            self._extraction_worker.cancel()

    def position_ms(self) -> int:
        if self._frame_edit_position is not None:
            return self._frame_edit_position
        if self._busy:
            return self._last_position_ms
        if self.vlc_player is not None:
            try:
                return max(0, int(self.vlc_player.get_time()))
            except Exception:
                pass
        if self.player is not None:
            return max(0, int(self.player.position()))
        return self._last_position_ms

    def duration_ms(self) -> int:
        if self.store is None:
            return 0
        return int(self.store.media.get("duration_ms", 0))

    def toggle_playback(self) -> None:
        if self.store is None or self._busy:
            return
        if self._frame_edit_position is not None:
            self.return_to_video_requested.emit()
            return
        if self.vlc_player is not None:
            try:
                if self.vlc_player.is_playing():
                    self.vlc_player.pause()
                    self.play_button.setText(self._t("播放", "Play"))
                else:
                    self.vlc_player.play()
                    self.play_button.setText(self._t("暂停", "Pause"))
            except Exception as error:
                self.status_message.emit("VLC 播放失败：{}".format(error))
            return
        if self.player is None:
            return
        if self.player.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def pause(self) -> None:
        if self.vlc_player is not None:
            try:
                # libVLC pause() toggles and resumes an already paused player.
                self.vlc_player.set_pause(1)
            except Exception:
                pass
        if self.player is not None:
            self.player.pause()
        self.play_button.setText(
            self._t("返回视频", "Return to video") if self._frame_edit_position is not None
            else self._t("播放", "Play")
        )

    def seek(self, milliseconds: int) -> None:
        if self._busy or self._frame_edit_position is not None:
            return
        position = min(self.duration_ms(), max(0, int(milliseconds)))
        self._last_position_ms = position
        if self.vlc_player is not None:
            try:
                self.vlc_player.set_time(position)
            except Exception:
                pass
        if self.player is not None:
            self.player.setPosition(position)
        self._on_position_changed(position)

    def seek_relative(self, milliseconds: int) -> None:
        """Seek by a bounded millisecond delta from the current playhead."""

        if self.store is None:
            return
        self.seek(self.position_ms() + int(milliseconds))

    def step_frames(self, count: int) -> None:
        self.pause()
        if self.frame_map is not None and len(self.frame_map):
            current = self.frame_map.frame_for_timestamp(self.position_ms())
            target = min(len(self.frame_map) - 1, max(0, current + int(count)))
            self.seek(self.frame_map.timestamp_for_frame(target))
            return
        fps = self.fps if self.fps > 0 else 25.0
        self.seek(self.position_ms() + int(round(count * 1000.0 / fps)))

    def set_playback_rate(self) -> None:
        rate = float(self.rate_combo.currentData() or 1.0)
        if self.vlc_player is not None:
            try:
                self.vlc_player.set_rate(rate)
            except Exception:
                pass
        if self.player is not None:
            self.player.setPlaybackRate(rate)

    def _begin_scrub(self) -> None:
        self._scrubbing = True

    def _preview_scrub(self, value: int) -> None:
        self._update_position_label(value)
        self.timeline.set_position(value)

    def _end_scrub(self) -> None:
        self._scrubbing = False
        self.seek(self.position_slider.value())

    def _on_position_changed(self, milliseconds: int) -> None:
        if self._frame_edit_position is not None:
            milliseconds = self._frame_edit_position
        elif self._busy:
            milliseconds = self._last_position_ms
        self._last_position_ms = max(0, int(milliseconds))
        if not self._scrubbing:
            self.position_slider.setValue(self._last_position_ms)
        self.timeline.set_position(self._last_position_ms)
        self._update_position_label(self._last_position_ms)

    def _on_duration_changed(self, milliseconds: int) -> None:
        if milliseconds <= 0:
            return
        self.position_slider.setRange(0, int(milliseconds))
        self.timeline.set_duration(int(milliseconds))
        if self.store is not None:
            self._commit_store_mutation(
                lambda: self.store.update_media_info(duration_ms=int(milliseconds)),
                "保存视频时长失败",
            )

    def _on_state_changed(self, state: int) -> None:
        if self._frame_edit_position is not None:
            self.play_button.setText(self._t("返回视频", "Return to video"))
            return
        playing = bool(
            self.player is not None
            and state == QtMultimedia.QMediaPlayer.PlayingState
        )
        self.play_button.setText(
            self._t("暂停", "Pause") if playing else self._t("播放", "Play")
        )

    def _on_player_error(self, *_args: object) -> None:
        if self.player is None:
            return
        message = self.player.errorString()
        if message:
            self.status_message.emit("视频播放器：{}".format(message))
        if self.video_path is not None and self.vlc_player is None:
            self._activate_vlc_fallback(self.video_path)

    def _update_position_label(self, position_ms: int) -> None:
        self.position_label.setText(
            (self._t("帧分割 · ", "Frame · ") if self._frame_edit_position is not None else "")
            + "{} / {}".format(format_timecode(position_ms), format_timecode(self.duration_ms()))
        )

    def mark_start(self) -> None:
        if self.store is None:
            return
        label = self.label_combo.currentText().strip()
        if not label:
            QtWidgets.QMessageBox.warning(self, "缺少标签", "请先输入阶段或动作标签。")
            return
        if self._pending_interval is not None:
            if not self._confirm_discard_pending_interval("记录新的开始时间"):
                return
        category = str(self.category_combo.currentData())
        self._pending_interval = (category, label, self.position_ms())
        self._update_pending_label()
        self.discard_pending_interval_action.setEnabled(True)
        self.context_changed.emit()
        self.status_message.emit("已记录区间开始；移动播放位置后按 O 完成。")

    def mark_end(self) -> None:
        if self.store is None or self._pending_interval is None:
            QtWidgets.QMessageBox.information(self, "没有开始时间", "请先按 I 记录区间开始。")
            return
        category, label, start_ms = self._pending_interval
        end_ms = self.position_ms()
        saved, _annotation = self._commit_store_mutation(
            lambda: self.store.add_interval(category, label, start_ms, end_ms),
            "保存区间失败",
            validation_title="区间无效",
        )
        if not saved:
            return
        self._clear_pending_interval()
        self._refresh_project_views()
        self.status_message.emit("已保存 {}：{}".format(category.title(), label))

    def _refresh_label_templates(self) -> None:
        current = self.label_combo.currentText().strip()
        category = str(self.category_combo.currentData() or "")
        labels = []
        if self.store is not None:
            labels = [
                template.label
                for template in self.store.temporal_templates
                if template.kind == "interval"
                and template.category.casefold() == category.casefold()
            ]
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItems(labels)
        self.label_combo.setCurrentText(current if current else (labels[0] if labels else ""))
        self.label_combo.blockSignals(False)

    def manage_templates(self) -> None:
        if self.store is None:
            return
        dialog = TemplateManagerDialog(self.store, self)
        dialog.templates_changed.connect(self._templates_changed)
        self._template_manager = dialog
        dialog.exec_()
        self._template_manager = None

    def import_template_dialog(self) -> None:
        if self.store is None:
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入模板 TXT",
            str(self.store.case_directory),
            "Text template (*.txt);;All files (*)",
        )
        if not filename:
            return
        route, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "模板类型",
            "将该文件导入为",
            ["Phase 区间", "Task 区间", "评分", "关键帧 / 点"],
            0,
            False,
        )
        if not accepted:
            return
        try:
            if route.startswith("Phase"):
                templates = parse_interval_template(Path(filename), "phase")
            elif route.startswith("Task"):
                templates = parse_interval_template(Path(filename), "task")
            else:
                category = "assessment" if route == "评分" else "keyframe"
                templates = parse_rating_template(Path(filename), category)
                if route != "评分":
                    for template in templates:
                        template.kind = "point"
                        template.score_options = []
            with self.store.transaction():
                self.store.append_templates(templates)
                self.store.save()
            self._sync_default_exports(self.store)
            self._templates_changed()
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "导入模板失败", str(error))

    def _templates_changed(self) -> None:
        self._refresh_label_templates()
        self._rebuild_rating_menu()
        self._refresh_project_views()

    def _rebuild_rating_menu(self) -> None:
        """Build a flat, one-selection menu from the project's rating templates."""

        self.rating_menu.clear()
        if self.store is None:
            placeholder = self.rating_menu.addAction(
                self._t("打开项目后可评分", "Open a project to rate")
            )
            placeholder.setEnabled(False)
            self.rating_button.setEnabled(False)
            return

        templates = [
            template
            for template in self.store.temporal_templates
            if template.kind == "rating"
        ]
        current_scores = {
            annotation.template_id: annotation.score
            for annotation in self.store.temporal_annotations
            if annotation.kind == "rating"
        }
        option_count = 0
        for template in templates:
            for score in template.score_options:
                action = self.rating_menu.addAction("{} · {}".format(template.label, score))
                action.setData((template.id, int(score)))
                action.setCheckable(True)
                action.setChecked(current_scores.get(template.id) == int(score))
                status_tip = self._t(
                    "立即将“{}”评分保存为 {}",
                    "Immediately save '{}' with score {}",
                ).format(template.label, score)
                action.setStatusTip(status_tip)
                action.setToolTip(status_tip)
                action.triggered.connect(
                    lambda _checked=False, template_id=template.id, value=int(score): self.set_rating_from_menu(
                        template_id, value
                    )
                )
                option_count += 1
        if not option_count:
            placeholder = self.rating_menu.addAction(
                self._t("没有可用的评分模板", "No rating templates available")
            )
            placeholder.setEnabled(False)
        self.rating_button.setEnabled(bool(option_count))

    def set_rating_from_menu(self, template_id: str, score: int) -> bool:
        """Commit a rating in one selection, replacing that template's prior score."""

        if self.store is None:
            return False
        template = next(
            (
                item
                for item in self.store.temporal_templates
                if item.id == str(template_id) and item.kind == "rating"
            ),
            None,
        )
        if template is None or int(score) not in template.score_options:
            return False
        saved, _annotation = self._commit_store_mutation(
            lambda: self.store.set_rating(template.id, int(score), status="ready"),
            self._t("保存评分失败", "Saving rating failed"),
            validation_title=self._t("评分无效", "Invalid rating"),
        )
        if not saved:
            return False
        self._refresh_project_views()
        self.status_message.emit(
            self._t("已保存评分 {}：{}", "Saved rating {}: {}").format(
                template.label, score
            )
        )
        return True

    def add_temporal_dialog(self, kind: str) -> None:
        if self.store is None:
            return
        templates = [
            template for template in self.store.temporal_templates if template.kind == kind
        ]
        if not templates:
            QtWidgets.QMessageBox.information(
                self, "缺少模板", "请先在“管理模板”中创建对应类型的模板。"
            )
            return
        dialog = TemporalAnnotationDialog(
            templates,
            self.duration_ms(),
            self.position_ms(),
            kind=kind,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.values()
        frame_index = None
        if kind == "point" and self.frame_map is not None:
            frame_index = self.frame_map.frame_for_timestamp(values["start_ms"])
        values["frame_index"] = frame_index
        saved, _annotation = self._commit_store_mutation(
            lambda: self.store.add_temporal_annotation(**values),
            "保存时间标注失败",
            validation_title="时间标注无效",
        )
        if saved:
            self._refresh_project_views()

    def _selected_annotation_id(self) -> Optional[str]:
        selected = self.annotation_table.selectedItems()
        if not selected:
            return None
        value = selected[0].data(QtCore.Qt.UserRole)
        return str(value) if value else None

    def edit_selected_annotation(self) -> None:
        if self.store is None:
            return
        annotation_id = self._selected_annotation_id()
        if not annotation_id:
            return
        annotation = next(
            (item for item in self.store.temporal_annotations if item.id == annotation_id),
            None,
        )
        if annotation is None:
            return
        dialog = TemporalAnnotationDialog(
            self.store.temporal_templates,
            self.duration_ms(),
            annotation.start_ms,
            annotation=annotation,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.values()
        if annotation.kind == "point" and self.frame_map is not None:
            values["frame_index"] = self.frame_map.frame_for_timestamp(values["start_ms"])
        saved, _updated = self._commit_store_mutation(
            lambda: self.store.update_temporal_annotation(annotation_id, **values),
            "更新时间标注失败",
            validation_title="时间标注无效",
        )
        if saved:
            self._refresh_project_views()

    def show_validation_report(self) -> None:
        if self.store is None:
            return
        issues = self.store.validate()
        coverage = round(self.store.phase_coverage_ratio() * 100)
        reviewed = sum(
            annotation.status == "reviewed"
            for annotation in self.store.temporal_annotations
        )
        lines = [
            "阶段覆盖率：{}%".format(coverage),
            "已复核：{} / {}".format(reviewed, len(self.store.temporal_annotations)),
        ]
        if issues:
            lines.append("")
            lines.extend(
                "[{}] {}".format(issue.severity.upper(), issue.message)
                for issue in issues[:40]
            )
            if len(issues) > 40:
                lines.append("…另有 {} 项".format(len(issues) - 40))
        else:
            lines.append("未发现校验问题。")
        QtWidgets.QMessageBox.information(self, "项目校验与覆盖率", "\n".join(lines))

    def import_annotations_dialog(self) -> None:
        if self.store is None:
            return
        filename, selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入时间标注",
            str(self.store.case_directory),
            "CSV annotations (*.csv);;Legacy TXT (*.txt)",
        )
        if not filename:
            return
        source = Path(filename)

        def import_operation():
            if source.suffix.casefold() == ".csv":
                return self.store.import_csv(source)
            return self.store.import_legacy(source.parent)

        saved, reports = self._commit_store_mutation(
            import_operation, "导入时间标注失败"
        )
        if saved:
            self._refresh_project_views()
            if reports:
                QtWidgets.QMessageBox.warning(
                    self,
                    "导入完成但有提示",
                    "\n".join(issue.message for issue in reports[:30]),
                )

    def export_annotations_dialog(self) -> None:
        if self.store is None:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择导出目录", str(self.store.case_directory / "exports")
        )
        if not directory:
            return
        try:
            destination = Path(directory)
            csv_path = self.store.export_csv(destination / "annotations.csv")
            legacy = self.store.export_legacy(destination)
            self.status_message.emit(
                "已导出 {} 和 {} 个 TXT 兼容文件。".format(csv_path, len(legacy))
            )
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(error))

    def _tracking_seed_options(self) -> List[Dict[str, Any]]:
        if self.store is None:
            return []
        options: List[Dict[str, Any]] = []
        for spatial in self.store.project["spatial_frames"]:
            if spatial.get("annotation_status") != "saved":
                continue
            image_path = self._resolve_project_asset(spatial.get("image_path", ""))
            label_path = self._resolve_project_asset(spatial.get("annotation_path", ""))
            if not image_path.is_file() or not label_path.is_file():
                continue
            try:
                annotation = Annotation(str(image_path), str(label_path)).load_annotation()
            except Exception:
                continue
            frame_index = int(spatial["frame_index"])
            for object_index, obj in enumerate(annotation.objects):
                options.append(
                    {
                        "frame_index": frame_index,
                        "object_index": object_index,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "label": "帧 {} · {} · 组 {}{}".format(
                            frame_index,
                            obj.category,
                            obj.group,
                            " · 纠正帧" if obj.track_id else "",
                        ),
                    }
                )
        return options

    def _prepare_tracking_seed(self, option: Dict[str, Any]) -> TrackingSeed:
        if self.store is None:
            raise ValueError("No video project is active")
        image_path = Path(option["image_path"])
        label_path = Path(option["label_path"])
        annotation = Annotation(str(image_path), str(label_path)).load_annotation()
        object_index = int(option["object_index"])
        if not 0 <= object_index < len(annotation.objects):
            raise ValueError("Selected tracking object no longer exists")
        obj = annotation.objects[object_index]
        object_id = str(obj.object_id or "")
        track_id = str(obj.track_id or "")
        if not object_id:
            object_id = "obj_{}_{}_{}".format(
                int(option["frame_index"]), safe_identifier(obj.category), int(obj.group)
            )
        if not track_id:
            track_id = object_id
        instance = self.store.ensure_instance(
            obj.category,
            obj.category,
            object_id=object_id,
            track_id=track_id,
            review_status="reviewed",
        )
        if obj.object_id != instance["object_id"] or obj.track_id != instance["track_id"]:
            obj.object_id = instance["object_id"]
            obj.track_id = instance["track_id"]
            obj.source = obj.source or "manual"
            obj.review_status = "reviewed"
            annotation.save_annotation()
        mask = np.zeros((annotation.height, annotation.width), dtype=np.uint8)
        points = np.asarray(obj.segmentation, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3:
            raise ValueError("Selected object does not have a valid polygon")
        cv2.fillPoly(mask, [points], 255)
        seed_directory = self.store.tracking_directory / "seeds" / instance["object_id"]
        seed_path = seed_directory / "frame_{:012d}.png".format(int(option["frame_index"]))
        atomic_write_png(seed_path, mask)
        return TrackingSeed(
            int(option["frame_index"]), instance["object_id"], seed_path
        )

    def start_tracking_dialog(self) -> None:
        if self.store is None or self.video_path is None:
            return
        if self._tracking_worker is not None and self._tracking_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "追踪正在运行", "请先等待或取消当前追踪任务。")
            return
        options = self._tracking_seed_options()
        if not options:
            QtWidgets.QMessageBox.information(
                self,
                "缺少复核关键帧",
                "请先提取视频帧、完成多边形/SAM 分割并保存，之后才能作为追踪种子。",
            )
            return
        facade = self._tracking_adapter_provider() if self._tracking_adapter_provider else None
        if facade is None:
            QtWidgets.QMessageBox.warning(
                self,
                "视频模型未就绪",
                "请先在模型管理中加载支持视频的 SAM2、SAM3 或 SAM3.1 模型。",
            )
            return
        dialog = TrackingRangeDialog(self.frame_count, options, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.values()
        try:
            seeds = [self._prepare_tracking_seed(option) for option in values["seeds"]]
            self.store.save()
            self._sync_default_exports(self.store)
            request = TrackingRequest(
                direction=values["direction"],
                start_frame=values["start_frame"],
                end_frame=values["end_frame"],
                seeds=seeds,
                engine="segany-video",
                replace_manual=values["replace_manual"],
            )
            adapter = SegAnyTrackingAdapter(facade)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "追踪准备失败", str(error))
            return
        self.pause()
        worker = TrackingWorker(
            self._project_token, self.store.path, request, adapter
        )
        worker.progress.connect(self._on_tracking_progress)
        worker.completed.connect(self._on_tracking_completed)
        worker.failed.connect(self._on_tracking_failed)
        worker.finished.connect(self._background_worker_finished)
        self._retain_worker(worker)
        self._tracking_worker = worker
        self.cancel_worker_action.setEnabled(True)
        self.start_tracking_action.setEnabled(False)
        worker.start()

    def _on_tracking_progress(self, current: int, total: int, stage: str) -> None:
        label = "解码" if stage == "decode" else "追踪"
        self.status_message.emit("{}：{} / {}".format(label, current, total or "?"))

    def _on_tracking_completed(self, token: int, run_id: str) -> None:
        if token != self._project_token or self.sender() is not self._tracking_worker:
            return
        if self.store is None:
            return
        self.store = UnifiedVideoProjectStore.load(self.store.path)
        self._staged_tracking_run_id = run_id
        self.start_tracking_action.setEnabled(True)
        self._refresh_project_views()
        self.status_message.emit("追踪结果已暂存，尚未覆盖任何人工标注。")

    def _on_tracking_failed(self, token: int, message: str) -> None:
        if token != self._project_token or self.sender() is not self._tracking_worker:
            return
        self.start_tracking_action.setEnabled(True)
        if message == "cancelled":
            self.status_message.emit("追踪已取消；人工标注未被修改。")
        else:
            QtWidgets.QMessageBox.critical(self, "追踪失败", message)

    def _tracking_frame_paths(self, run: Dict[str, Any]) -> Tuple[Callable[[int], Path], Callable[[int], Path]]:
        if self.store is None:
            raise ValueError("No project is active")
        cache = (
            self.store.tracking_directory
            / "frame_cache"
            / self.store.media["fingerprint"][:12]
        )
        return (
            lambda index: cache / "frame_{:012d}.png".format(int(index)),
            lambda index: self.store.labels_directory
            / "frame_{:012d}.json".format(int(index)),
        )

    def review_staged_tracking(self) -> None:
        if self.store is None:
            return
        staged = [
            run for run in self.store.project["tracking_runs"] if run.get("status") == "staged"
        ]
        if not staged:
            QtWidgets.QMessageBox.information(self, "没有暂存结果", "当前项目没有等待审阅的追踪结果。")
            return
        run = staged[-1]
        image_path, label_path = self._tracking_frame_paths(run)

        def load_frame(frame_index: int) -> np.ndarray:
            cached = image_path(frame_index)
            frame = cv2.imread(str(cached), cv2.IMREAD_COLOR)
            if frame is None:
                frame = decode_frame(Path(self.store.media["path"]), int(frame_index))
            return frame

        instances = {
            str(item.get("object_id")): item
            for item in self.store.project.get("instances", [])
            if item.get("object_id")
        }
        dialog = TrackingReviewDialog(
            run,
            load_frame,
            self._resolve_project_asset,
            instances,
            self,
        )
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Rejected:
            return
        if result == TrackingReviewDialog.Discarded:
            saved, _updated = self._commit_store_mutation(
                lambda: self.store.update_tracking_run(run["id"], status="cancelled"),
                "丢弃暂存追踪失败",
            )
            if saved:
                self._refresh_project_views()
            return
        try:
            service = VideoTrackingService(self.store)
            service.commit(
                run["id"],
                image_path,
                label_path,
                replace_manual=bool(run.get("replace_manual", False)),
            )
            self.store = UnifiedVideoProjectStore.load(self.store.path)
            self._refresh_project_views()
            self.status_message.emit("追踪结果已提交；仍可从更多菜单回滚。")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "提交追踪失败", str(error))

    def rollback_latest_tracking(self) -> None:
        if self.store is None:
            return
        committed = [
            run
            for run in self.store.project["tracking_runs"]
            if run.get("status") == "committed"
        ]
        if not committed:
            return
        run = committed[-1]
        answer = QtWidgets.QMessageBox.question(
            self,
            "回滚追踪",
            "恢复追踪提交前的所有标签吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            VideoTrackingService(self.store).rollback(run["id"])
            self.store = UnifiedVideoProjectStore.load(self.store.path)
            self._refresh_project_views()
            self.status_message.emit("追踪提交已回滚。")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "回滚追踪失败", str(error))

    def delete_selected_annotation(self) -> None:
        if self.store is None:
            return
        selected = self.annotation_table.selectedItems()
        if not selected:
            return
        annotation_id = selected[0].data(QtCore.Qt.UserRole)
        if not annotation_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "删除时间标注",
            "确定删除所选时间标注吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        saved, _deleted = self._commit_store_mutation(
            lambda: self.store.delete_temporal_annotation(str(annotation_id)),
            "删除时间标注失败",
        )
        if saved:
            self._refresh_project_views()

    def undo(self) -> None:
        if self.store is None:
            return
        saved, changed = self._commit_store_mutation(self.store.undo, "撤销失败")
        if saved and changed:
            self._refresh_project_views()

    def redo(self) -> None:
        if self.store is None:
            return
        saved, changed = self._commit_store_mutation(self.store.redo, "重做失败")
        if saved and changed:
            self._refresh_project_views()

    def _refresh_project_views(self) -> None:
        selected_items = self.annotation_table.selectedItems()
        selected_id = (
            str(selected_items[0].data(QtCore.Qt.UserRole))
            if selected_items and selected_items[0].data(QtCore.Qt.UserRole)
            else None
        )
        annotations = self.store.temporal_annotations if self.store is not None else []
        annotations.sort(key=lambda item: (item.start_ms, item.category, item.label))
        self.timeline.set_annotations(annotations)
        self.annotation_table.blockSignals(True)
        self.annotation_table.setRowCount(len(annotations))
        self._row_by_annotation_id.clear()
        for row, annotation in enumerate(annotations):
            self._row_by_annotation_id[annotation.id] = row
            values = (
                annotation.kind,
                annotation.category.title(),
                annotation.label,
                format_timecode(annotation.start_ms),
                format_timecode(annotation.end_ms),
                annotation.status,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, annotation.id)
                self.annotation_table.setItem(row, column, item)
        if selected_id in self._row_by_annotation_id:
            self.annotation_table.selectRow(self._row_by_annotation_id[selected_id])
            self.timeline.set_selected_annotation(selected_id)
        else:
            self.annotation_table.clearSelection()
            self.timeline.set_selected_annotation(None)
            selected_id = None
        self.annotation_table.blockSignals(False)
        self.delete_button.setEnabled(selected_id is not None)
        self.edit_annotation_action.setEnabled(selected_id is not None)
        self.undo_button.setEnabled(bool(self.store and self.store.can_undo))
        self.redo_button.setEnabled(bool(self.store and self.store.can_redo))
        if self.store is not None:
            coverage = round(self.store.phase_coverage_ratio() * 100)
            pending_review = sum(
                annotation.status != "reviewed" for annotation in annotations
            )
            self.coverage_label.setText(
                self._t(
                    "阶段覆盖 {}% · 待复核 {}".format(coverage, pending_review),
                    "Phase coverage {}% · {} to review".format(coverage, pending_review),
                )
            )
            self._refresh_label_templates()
            self._rebuild_rating_menu()
            self.review_tracking_action.setEnabled(
                any(run.get("status") == "staged" for run in self.store.project["tracking_runs"])
            )
            self.rollback_tracking_action.setEnabled(
                any(run.get("status") == "committed" for run in self.store.project["tracking_runs"])
            )
        self._update_position_label(self.position_ms())
        self.context_changed.emit()

    def select_annotation(self, annotation_id: str) -> None:
        row = self._row_by_annotation_id.get(annotation_id)
        if row is None:
            return
        self.timeline.set_selected_annotation(annotation_id)
        self.annotation_table.selectRow(row)

    def _select_table_annotation(self) -> None:
        selected = self.annotation_table.selectedItems()
        if selected:
            annotation_id = selected[0].data(QtCore.Qt.UserRole)
            self.timeline.set_selected_annotation(
                str(annotation_id) if annotation_id else None
            )
            self.delete_button.setEnabled(bool(annotation_id))
            self.edit_annotation_action.setEnabled(bool(annotation_id))
        else:
            self.timeline.set_selected_annotation(None)
            self.delete_button.setEnabled(False)
            self.edit_annotation_action.setEnabled(False)

    def _activate_table_item(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None or self.store is None:
            return
        annotation_id = item.data(QtCore.Qt.UserRole)
        if annotation_id:
            self.select_annotation(str(annotation_id))
        for annotation in self.store.temporal_annotations:
            if annotation.id == annotation_id:
                self.seek(annotation.start_ms)
                break

    def _seek_table_row(self, row: int, column: int = 0) -> None:
        """Compatibility wrapper for older integrations and tests."""

        del column
        item = self.annotation_table.item(row, 0)
        if item is not None:
            self._activate_table_item(item)

    def _point_template_for_capture(self) -> Optional[TemporalTemplate]:
        if self.store is None:
            return None
        templates = [
            template
            for template in self.store.temporal_templates
            if template.kind == "point"
        ]
        if not templates:
            return None
        current_label = self.label_combo.currentText().strip().casefold()
        if current_label:
            matched = next(
                (
                    template
                    for template in templates
                    if template.label.casefold() == current_label
                ),
                None,
            )
            if matched is not None:
                return matched
        return next(
            (template for template in templates if template.id == "keyframe_default"),
            templates[0],
        )

    def capture_current_keyframe(self) -> bool:
        """Extract and persist the current indexed frame as a linked point."""

        template = self._point_template_for_capture()
        if template is None:
            if self.store is not None:
                QtWidgets.QMessageBox.information(
                    self,
                    self._t("缺少关键帧模板", "No key-frame template"),
                    self._t(
                        "请先在“管理模板”中创建关键帧 / 点模板。",
                        "Create a key-frame / point template in Manage templates first.",
                    ),
                )
            return False
        return self._start_current_frame_extraction("keyframe", template.id)

    def _start_current_frame_extraction(
        self, intent: str, template_id: str = ""
    ) -> bool:
        if self.store is None or self.video_path is None or self._busy:
            return False
        self.pause()
        position_ms = self.position_ms()
        mapping = self.frame_map
        if mapping is None:
            if not self.frame_count or self.fps <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    self._t("帧索引不可用", "Frame index unavailable"),
                    self._t(
                        "视频尚未建立可用的帧时间索引。",
                        "The video does not yet have a usable frame-time index.",
                    ),
                )
                return False
            mapping = FrameTimestampMap.from_fps(self.frame_count, self.fps)
        frame_index = mapping.frame_for_timestamp(position_ms)
        self._last_position_ms = position_ms
        self._frame_extraction_intent = str(intent)
        self._keyframe_template_id = str(template_id)
        message = self._t(
            "正在异步提取第 {} 帧…",
            "Extracting frame {} in the background…",
        ).format(frame_index)
        self._set_frame_extraction_busy(True, message)
        worker = FrameExtractionWorker(
            self._project_token,
            self.video_path,
            mapping,
            position_ms,
            self.store.frames_directory,
            self.store.labels_directory,
            cached_image_signature=self._frame_cache.get(frame_index),
        )
        worker.ready.connect(self._on_extraction_ready)
        worker.failed.connect(self._on_extraction_failed)
        worker.finished.connect(self._on_extraction_finished)
        self._retain_worker(worker)
        self._extraction_worker = worker
        self.cancel_worker_action.setEnabled(True)
        worker.start()
        return True

    def annotate_current_frame(self) -> None:
        self._start_current_frame_extraction("spatial")

    def _stored_project_asset_path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        if self.store is None:
            return str(resolved)
        try:
            return resolved.relative_to(self.store.case_directory.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def _on_extraction_ready(
        self,
        token: int,
        frame_index: int,
        pts_ms: int,
        image_path: str,
        label_path: str,
    ) -> None:
        if (
            token != self._project_token
            or self.store is None
            or self.sender() is not self._extraction_worker
            or self._extraction_worker.cancel_event.is_set()
        ):
            return
        image = Path(image_path)
        label = Path(label_path)
        try:
            stat = image.stat()
            self._frame_cache[frame_index] = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            return
        if self._frame_extraction_intent == "keyframe":
            template_id = self._keyframe_template_id
            template = next(
                (
                    item
                    for item in self.store.temporal_templates
                    if item.id == template_id and item.kind == "point"
                ),
                None,
            )
            if template is None:
                self.status_message.emit(
                    self._t(
                        "关键帧模板已不存在，未创建时间标注。",
                        "The key-frame template no longer exists; no point was created.",
                    )
                )
                return
            saved, annotation = self._commit_store_mutation(
                lambda: self.store.add_point(
                    template.id,
                    start_ms=pts_ms,
                    frame_index=frame_index,
                    status="ready",
                    asset_path=self._stored_project_asset_path(image),
                ),
                self._t("保存关键帧失败", "Saving key frame failed"),
                validation_title=self._t("关键帧无效", "Invalid key frame"),
            )
            if not saved:
                return
            self._refresh_project_views()
            if annotation is not None:
                self.select_annotation(annotation.id)
            self.status_message.emit(
                self._t(
                    "已保存关键帧“{}”：第 {} 帧，并链接提取图像。",
                    "Saved key frame '{}': frame {}, linked to the extracted image.",
                ).format(template.label, frame_index)
            )
            return
        existing = next(
            (item for item in self.store.project["spatial_frames"]
             if int(item["frame_index"]) == frame_index), None
        )
        status = "saved" if (
            existing and existing.get("annotation_status") == "saved"
            and self._resolve_project_asset(existing["annotation_path"]) == label.resolve()
            and label.is_file()
        ) else "pending"
        unchanged = (
            existing and existing["pts_ms"] == pts_ms
            and existing.get("annotation_status") == status
            and self._resolve_project_asset(existing["image_path"]) == image.resolve()
            and self._resolve_project_asset(existing["annotation_path"]) == label.resolve()
        )
        if not unchanged:
            saved, _linked = self._commit_store_mutation(
                lambda: self.store.add_spatial_frame(
                    frame_index, pts_ms, image, label, annotation_status=status,
                ),
                "保存帧项目索引失败",
            )
            if not saved:
                return
        self.frame_annotation_requested.emit(
            str(self.store.frames_directory),
            str(self.store.labels_directory),
            image.name,
        )
        self.status_message.emit(
            self._t("已提取第 {} 帧并交给 IVAS 分割画布。",
                    "Extracted frame {} into the IVAS segmentation canvas.").format(frame_index)
        )

    def _on_extraction_failed(self, token: int, message: str) -> None:
        if (token != self._project_token or self.sender() is not self._extraction_worker
                or self._extraction_worker.cancel_event.is_set()):
            return
        title = self._t("提取视频帧失败", "Video frame extraction failed")
        self.status_message.emit("{}：{}".format(title, message))
        QtWidgets.QMessageBox.critical(self, title, message)

    def _on_extraction_finished(self) -> None:
        if self.sender() is self._extraction_worker:
            self._set_frame_extraction_busy(False)
            self._frame_extraction_intent = "spatial"
            self._keyframe_template_id = ""
        self._background_worker_finished()

    def _resolve_project_asset(self, stored_path: str) -> Path:
        path = Path(str(stored_path))
        if not path.is_absolute() and self.store is not None:
            path = self.store.case_directory / path
        return path.resolve()

    def mark_frame_annotation_saved(self, label_path: str) -> bool:
        """Mark the extracted frame matching ``label_path`` as safely persisted."""

        if self.store is None:
            return False
        target_path = Path(label_path).resolve()
        if not target_path.is_file():
            self.status_message.emit("图像标注文件尚未写入：{}".format(target_path))
            return False

        matched = next(
            (
                raw
                for raw in self.store.project["spatial_frames"]
                if self._resolve_project_asset(raw.get("annotation_path", ""))
                == target_path
            ),
            None,
        )
        if matched is None:
            self.status_message.emit("当前视频项目中未找到对应帧：{}".format(target_path))
            return False
        if matched.get("annotation_status") == "saved":
            return True

        frame_index = int(matched["frame_index"])
        pts_ms = int(matched["pts_ms"])
        image_path = self._resolve_project_asset(matched.get("image_path", ""))

        def mark_saved() -> Dict[str, Any]:
            return self.store.add_spatial_frame(
                frame_index,
                pts_ms,
                image_path,
                target_path,
                annotation_status="saved",
            )

        saved, _linked = self._commit_store_mutation(
            mark_saved, "确认帧标注保存失败"
        )
        if saved:
            self.status_message.emit("已确认第 {} 帧的图像标注。".format(frame_index))
        return saved

    def shutdown(self) -> bool:
        if not self.confirm_discard_pending(self):
            return False
        self.cancel_background_work(invalidate=True)
        for worker in tuple(self._workers):
            if worker is not None and worker.isRunning() and not worker.wait(5000):
                QtWidgets.QMessageBox.warning(
                    self,
                    "后台任务仍在停止",
                    "视频后台任务尚未安全停止，请稍后再次关闭。",
                )
                return False
        if self.store is not None and self.store.is_dirty:
            try:
                self.store.save()
                self._sync_default_exports(self.store)
            except Exception as error:
                self.status_message.emit("关闭前保存视频项目失败：{}".format(error))
                QtWidgets.QMessageBox.critical(self, "保存视频项目失败", str(error))
                return False
        if self.player is not None:
            self.player.stop()
            self.player.setMedia(QtMultimedia.QMediaContent())
        self._vlc_timer.stop()
        if self.vlc_player is not None:
            try:
                self.vlc_player.stop()
                self.vlc_player.release()
            except Exception:
                pass
            self.vlc_player = None
            self.vlc_instance = None
        if self._vlc_dll_directory is not None:
            self._vlc_dll_directory.close()
            self._vlc_dll_directory = None
        return True
