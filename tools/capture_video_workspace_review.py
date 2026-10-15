"""Capture deterministic compact/wide screenshots of the unified video UI."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtTest, QtWidgets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IVAS.theme import apply_workbench_theme
from IVAS.widgets.video_annotation_workspace import VideoAnnotationWorkspace


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG writer is unavailable")
    try:
        for index in range(20):
            frame = np.full((360, 640, 3), (25, 34, 47), dtype=np.uint8)
            cv2.rectangle(frame, (90 + index * 4, 80), (260 + index * 4, 255), (58, 145, 163), -1)
            cv2.circle(frame, (390, 175), 65, (62, 73, 91), -1)
            writer.write(frame)
    finally:
        writer.release()


def capture(output: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_workbench_theme(app)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video = root / "cataract_case.avi"
        make_video(video)
        workspace = VideoAnnotationWorkspace(root / "outputs")
        if workspace.player is not None:
            workspace.player.stop()
            workspace.player = None
        workspace._activate_vlc_fallback = lambda _path: False
        store = workspace.open_video(video)
        with store.transaction():
            store.add_interval("phase", "Preparation", 0, 650, status="reviewed")
            store.add_interval("task", "Capsulorhexis", 220, 980, status="ready")
            store.add_interval("phase", "Lens removal", 650, 1_500, status="draft")
            store.save()
        workspace.seek(780)
        workspace._refresh_project_views()

        for name, size in (("wide", (1_200, 760)), ("compact", (680, 500))):
            workspace.resize(*size)
            workspace.show()
            for _ in range(8):
                app.processEvents()
                QtTest.QTest.qWait(20)
            if not workspace.grab().save(str(output / "video-workspace-parity-{}.png".format(name))):
                raise RuntimeError("Could not save {} UI capture".format(name))
        workspace.shutdown()
        workspace.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/ui-review"))
    capture(parser.parse_args().output)
