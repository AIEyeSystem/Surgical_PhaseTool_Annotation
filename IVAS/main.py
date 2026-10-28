# -*- coding: utf-8 -*-
# @Maintainer: zack zhao
# import os

import os
import sys
import time
import traceback
from pathlib import Path


def _smoke_log(message: str) -> None:
    log_path = (os.environ.get("IVAS_BUILD_SMOKE_LOG") or os.environ.get("IVAT_BUILD_SMOKE_LOG"))
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(message.rstrip() + "\n")


def _run_frozen_sam31_smoke() -> int:
    """Exercise the bundled checkpoint and text-prompt path during packaging."""
    import numpy as np
    import torch
    from PIL import Image

    from IVAS.configs import CHECKPOINT_PATH
    from IVAS.segment_any.segment_any import SegAny

    # Keep Torch ahead of any optional Qt import in the frozen inference-only
    # smoke path so its matching Visual C++ runtime wins DLL resolution.
    torch.cuda.is_available()

    segany = None
    try:
        _smoke_log("SAM31_SMOKE start")
        checkpoint = Path(CHECKPOINT_PATH) / "sam3.1_multiplex_bf16.pt"
        bundle_root = Path(
            getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])
        )
        image_path = bundle_root / "_build_smoke" / "000000000113.jpg"
        if not checkpoint.is_file() or not image_path.is_file():
            _smoke_log("SAM31_SMOKE missing bundled checkpoint or image")
            return 2
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        _smoke_log("SAM31_SMOKE inputs ready")
        segany = SegAny(str(checkpoint), precision="bfloat16")
        _smoke_log("SAM31_SMOKE model loaded")
        segany.set_image(image)
        _smoke_log("SAM31_SMOKE image encoded")
        masks, scores = segany.predict_with_text_prompt("person")
        _smoke_log("SAM31_SMOKE text prediction complete")
        masks = np.asarray(masks)
        scores = np.asarray(scores)
        valid = (
            segany.model_source == "sam3.1"
            and masks.ndim == 3
            and masks.shape[0] > 0
            and bool(masks.reshape(masks.shape[0], -1).any(axis=1).any())
            and scores.shape == (masks.shape[0],)
            and bool(np.isfinite(scores).all())
        )
        _smoke_log("SAM31_SMOKE passed" if valid else "SAM31_SMOKE invalid output")
        return 0 if valid else 3
    except Exception:
        _smoke_log(traceback.format_exc())
        return 4
    finally:
        if segany is not None:
            segany.release()


def _validate_video_workbench_smoke(mainwindow):
    """Validate the lazy, mode-free video workbench without opening media."""

    if mainwindow.video_annotation_workspace is not None:
        raise RuntimeError("video workspace was not lazy at MainWindow startup")

    workspace = mainwindow._ensure_video_workspace()
    if workspace is None or mainwindow._ensure_video_workspace() is not workspace:
        raise RuntimeError("video workspace lazy initialization is not stable")
    if workspace.store is not None or workspace.video_path is not None:
        raise RuntimeError("video workbench smoke unexpectedly opened media")

    expected_proxy_targets = (
        ("actionTemporal_add_point", "add_point_action"),
        ("actionTemporal_capture_keyframe", "capture_keyframe_action"),
        ("actionTemporal_validate", "validate_action"),
        ("actionTemporal_import", "import_action"),
        ("actionTemporal_export", "export_action"),
        ("actionTracking_start", "start_tracking_action"),
        ("actionTracking_review", "review_tracking_action"),
        ("actionTracking_rollback", "rollback_tracking_action"),
    )
    proxy_targets = getattr(mainwindow, "_video_proxy_targets", {})
    for proxy_name, target_name in expected_proxy_targets:
        proxy = getattr(mainwindow, proxy_name, None)
        target = getattr(workspace, target_name, None)
        if proxy is None or target is None or proxy_targets.get(proxy) is not target:
            raise RuntimeError(
                "video command proxy is not bound: {} -> {}".format(
                    proxy_name, target_name
                )
            )

    transport_action = mainwindow.video_transport_widget_action
    hosted_transport = (
        mainwindow.video_transport_toolbar.widgetForAction(transport_action)
        if transport_action is not None
        else None
    )
    if hosted_transport is not workspace.transport_widget:
        raise RuntimeError("shared video transport is not hosted by MainWindow")
    if workspace.video_layout.indexOf(workspace.transport_widget) != -1:
        raise RuntimeError("shared video transport is still duplicated in the workspace")
    if workspace.transport_widget.window() is not mainwindow:
        raise RuntimeError("shared video transport is outside the application shell")

    if mainwindow.temporal_dock.widget().widget() is not workspace.editor_panel:
        raise RuntimeError("temporal editor is not attached to the shared shell")
    if mainwindow.timeline_dock.widget() is not workspace.timeline:
        raise RuntimeError("temporal timeline is not attached to the shared shell")

    public_menus = {
        action.menu()
        for action in mainwindow.menuBar().actions()
        if action.menu() is not None
    }
    required_menus = {
        mainwindow.menuFile,
        mainwindow.menuEdit,
        mainwindow.menuTemporal,
        mainwindow.menuTracking,
    }
    if not required_menus.issubset(public_menus):
        raise RuntimeError("unified frame/temporal/tracking menus are not public")

    file_actions = set(mainwindow.menuFile.actions())
    required_file_actions = {
        mainwindow.actionImages_dir,
        mainwindow.actionOpen_video_workspace,
        mainwindow.actionOpen_video_project_manifest,
        mainwindow.actionSave,
    }
    if not required_file_actions.issubset(file_actions):
        raise RuntimeError("unified File menu is missing image/video project commands")

    temporal_actions = set(mainwindow.menuTemporal.actions())
    tracking_actions = set(mainwindow.menuTracking.actions())
    if mainwindow.actionTemporal_add_point not in temporal_actions:
        raise RuntimeError("temporal commands are missing from the unified menu")
    if not {
        mainwindow.actionTracking_start,
        mainwindow.actionTracking_review,
        mainwindow.actionTracking_rollback,
    }.issubset(tracking_actions):
        raise RuntimeError("tracking commands are missing from the unified menu")

    return workspace


def _configured_theme():
    """Read the saved theme before any widget exists (no flash of default)."""
    try:
        from IVAS.configs import SOFTWARE_CONFIG_FILE, load_config

        software = (load_config(SOFTWARE_CONFIG_FILE) or {}).get("software") or {}
        return software.get("theme")
    except Exception:
        return None


def main():
    startup_started = time.perf_counter()
    if "--build-sam31-smoke-test" in sys.argv:
        return _run_frozen_sam31_smoke()

    gui_smoke = "--build-smoke-test" in sys.argv
    brand_smoke = "--build-brand-smoke-test" in sys.argv
    video_workbench_smoke = "--build-video-workbench-smoke-test" in sys.argv

    def record_stage(name: str) -> None:
        if gui_smoke:
            _smoke_log(
                "GUI_STAGE {} seconds={:.3f}".format(
                    name, time.perf_counter() - startup_started
                )
            )

    from PyQt5 import QtCore, QtGui, QtWidgets

    record_stage("qt_imported")

    from IVAS.banner import print_banner
    from IVAS.branding import APP_NAME, AUTHOR, LOGO_RESOURCE
    from IVAS.theme import apply_workbench_theme
    from IVAS.widgets.mainwindow import MainWindow

    record_stage("mainwindow_imported")

    print_banner()
    if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(AUTHOR)
    app.setWindowIcon(QtGui.QIcon(LOGO_RESOURCE))
    record_stage("application_created")
    apply_workbench_theme(app, _configured_theme())
    record_stage("theme_applied")
    mainwindow = MainWindow()
    record_stage("mainwindow_created")
    if brand_smoke:
        about_dialog = mainwindow._ensure_about_dialog()
        brand_is_valid = all(
            (
                not QtGui.QPixmap(LOGO_RESOURCE).isNull(),
                not app.windowIcon().isNull(),
                not mainwindow.windowIcon().isNull(),
                APP_NAME in mainwindow.windowTitle(),
                APP_NAME in about_dialog.label_3.text(),
                AUTHOR in about_dialog.label_5.text(),
            )
        )
        _smoke_log("BRAND_SMOKE passed" if brand_is_valid else "BRAND_SMOKE failed")
        if not brand_is_valid:
            mainwindow.close()
            return 5
    if video_workbench_smoke:
        _smoke_log("VIDEO_WORKBENCH_SMOKE start")
        try:
            _validate_video_workbench_smoke(mainwindow)
        except Exception:
            _smoke_log("VIDEO_WORKBENCH_SMOKE failed")
            _smoke_log(traceback.format_exc())
            mainwindow.close()
            return 6
    mainwindow.show()
    record_stage("mainwindow_shown")
    if gui_smoke or brand_smoke or video_workbench_smoke:
        _smoke_log(
            "GUI_READY seconds={:.3f}".format(time.perf_counter() - startup_started)
        )
        QtCore.QTimer.singleShot(750, mainwindow.close)
    exit_code = app.exec_()
    if video_workbench_smoke:
        safely_closed = bool(getattr(mainwindow, "_shutdown_started", False))
        if exit_code == 0 and safely_closed:
            _smoke_log("VIDEO_WORKBENCH_SMOKE passed")
        else:
            _smoke_log(
                "VIDEO_WORKBENCH_SMOKE unsafe exit code={} shutdown_started={}".format(
                    exit_code, safely_closed
                )
            )
            return exit_code or 7
    return exit_code


if __name__ == "__main__":
    main()
