"""Capture real IVAS widgets using synthetic media and hand-authored masks.

Run `python tools/capture_readme_screenshots.py` from an app environment.
No clinical data or model inference is used. A generated video still replaces
the native video surface for reliable offscreen capture. Local preferences and
annotations are unchanged; all demo projects are created in a temporary folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from IVAS.annotation import Object
from IVAS.theme import apply_workbench_theme
from IVAS.widgets import mainwindow
from IVAS.widgets.polygon import Polygon
from IVAS.widgets.responsive import ResponsiveDialog
from IVAS.widgets.video_annotation_dialogs import TrackingReviewDialog


def demo_media(root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    frame = np.full((480, 800, 3), (32, 27, 23), dtype=np.uint8)
    cv2.ellipse(frame, (400, 240), (230, 180), 0, 0, 360, (112, 124, 143), -1)
    cv2.circle(frame, (400, 240), 128, (78, 100, 113), -1)
    cv2.circle(frame, (400, 240), 86, (39, 48, 52), -1)
    cv2.ellipse(frame, (370, 195), (21, 9), -35, 0, 360, (160, 176, 186), -1)
    points = np.array([[35, 78], [62, 58], [354, 252], [349, 270]], np.int32)
    cv2.fillPoly(frame, [points], (175, 191, 199))
    cv2.polylines(frame, [points], True, (216, 228, 234), 2)
    cv2.putText(frame, "IVAS | SYNTHETIC DEMO", (28, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (221, 231, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, "No clinical data / No model inference", (28, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (178, 191, 199), 1, cv2.LINE_AA)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    video = root / "demo_case.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (800, 480))
    if not writer.isOpened():
        raise RuntimeError("The OpenCV MJPG writer is unavailable")
    try:
        for _ in range(150):
            writer.write(frame)
    finally:
        writer.release()
    return video, frame, mask


def settle(app: QtWidgets.QApplication) -> None:
    for _ in range(5):
        app.processEvents()
        QtTest.QTest.qWait(30)


def grab(app: QtWidgets.QApplication, widget: QtWidgets.QWidget, path: Path) -> None:
    widget.show()
    settle(app)
    if isinstance(widget, TrackingReviewDialog):
        widget._update_scaled_pixmaps()
        settle(app)
    if not widget.grab().save(str(path)):
        raise RuntimeError("Could not save {}".format(path))
    print(path.name)


def capture(output: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("IVAS")
    # The Windows offscreen platform does not discover installed fonts itself.
    font_root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for filename in ("segoeui.ttf", "seguisb.ttf", "seguisym.ttf", "msyh.ttc"):
        font_path = font_root / filename
        if font_path.is_file():
            QtGui.QFontDatabase.addApplicationFont(str(font_path))
    apply_workbench_theme(app)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ivas-readme-") as directory, mock.patch.object(
        mainwindow.CheckLatestVersionThread, "start", lambda _thread: None
    ), mock.patch.object(mainwindow.MainWindow, "save_cfg"), mock.patch.object(
        mainwindow.MainWindow, "save_software_cfg"
    ), mock.patch.object(
        ResponsiveDialog, "_fit_available_screen"
    ):
        root = Path(directory)
        video, frame, mask = demo_media(root)
        window = mainwindow.MainWindow()
        window._version_check_timer.stop()
        category_path = root / "ivas.yaml"
        category_path.write_bytes((REPO_ROOT / "IVAS" / "software.yaml").read_bytes())
        window.config_file = str(category_path)
        window.reload_cfg()
        window.cfg["software"]["auto_save"] = False
        window.translate("zh")
        window.resize(1440, 900)
        workspace = window._ensure_video_workspace()
        workspace.set_language("zh")
        workspace.output_root = root / "outputs"
        if workspace.player is not None:
            workspace.player.stop()
            workspace.player = None
        workspace._load_player_media = lambda _path: None
        try:
            store = workspace.open_video(video)
            timer = QtCore.QElapsedTimer()
            timer.start()
            while workspace._background_work_running() and timer.elapsed() < 15000:
                QtTest.QTest.qWait(30)
            if workspace._background_work_running():
                raise RuntimeError("Demo video indexing timed out")
            with store.transaction():
                store.add_interval("phase", "Preparation", 0, 6000, status="reviewed")
                store.add_interval("phase", "Capsulorhexis", 6000, 18000, status="ready")
                store.add_interval("phase", "Lens removal", 18000, 30000, status="draft")
                store.add_interval("task", "Instrument positioning", 8000, 13000, status="reviewed")
                store.add_interval("task", "Capsule opening", 13000, 17500, status="draft")
                store.save()
            workspace.seek(11000)
            workspace._refresh_project_views()
            workspace.label_combo.setCurrentText("Capsulorhexis")
            window._activate_video_workspace()

            # QWidget.grab() cannot reliably capture a native video overlay.
            original_surface = workspace.video_widget
            workspace.video_frame_layout.removeWidget(original_surface)
            original_surface.hide()
            still = QtWidgets.QLabel()
            still.setAlignment(QtCore.Qt.AlignCenter)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(rgb.data, 800, 480, 2400, QtGui.QImage.Format_RGB888).copy()
            still.setPixmap(QtGui.QPixmap.fromImage(image))
            workspace.video_frame_layout.addWidget(still)
            grab(app, window, output / "video-phases.png")

            window.request_frame_mode()
            timer.restart()
            while window.annotation_mode != "Frame" and timer.elapsed() < 15000:
                QtTest.QTest.qWait(30)
            if window.annotation_mode != "Frame":
                raise RuntimeError("Demo frame extraction timed out")
            polygon = Polygon()
            window.scene.addItem(polygon)
            polygon.load_object(Object(
                "CapsulorhexisForceps", 1, [[35, 78], [62, 58], [354, 252], [349, 270]],
                float(cv2.countNonZero(mask)), 1, [35, 58, 354, 270],
            ))
            window.polygons.append(polygon)
            window.save_current_work()
            grab(app, window, output / "frame-segmentation.png")

            cv2.imwrite(str(root / "forceps-mask.png"), mask)
            run = {
                "id": "synthetic-demo", "status": "staged", "direction": "forward",
                "start_frame": 55, "end_frame": 59,
                "staged_frames": [
                    {"frame_index": index, "objects": [
                        {"object_id": "demo-forceps", "mask_path": "forceps-mask.png",
                         "review_status": "draft"}]}
                    for index in range(55, 60)
                ],
            }
            review = TrackingReviewDialog(
                run, lambda _index: frame.copy(), lambda value: root / value,
                {"demo-forceps": {"category": "CapsulorhexisForceps", "track_id": "demo-1"}},
            )
            review.resize(1120, 640)
            grab(app, review, output / "tracking-review.png")
            review.reject()

            for ensure, filename in (
                (window._ensure_model_manager_dialog, "model-manager.png"),
                (window._ensure_converter_dialog, "format-converter.png"),
            ):
                dialog = ensure()
                dialog.resize(1000, 660)
                grab(app, dialog, output / filename)
                dialog.hide()
        finally:
            window.saved = True
            window.scene.cancel_draw()
            workspace._clear_pending_interval()
            workspace.shutdown()
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "images")
    capture(parser.parse_args().output)
