# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import functools
import gc
import os
import platform
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal

import IVAS.icons_rc
from IVAS.annotation import Annotation, Object
from IVAS.category_names import load_category_aliases
from IVAS.branding import (
    APP_NAME,
    CATEGORY_CONFIG_NAME,
    DISTRIBUTION_NAME,
    LEGACY_CATEGORY_CONFIG_NAMES,
)
from IVAS.configs import (CHECKPOINT_PATH, CONFIG_FILE, IVAS_ROOT,
                          SOFTWARE_CONFIG_FILE, CONTOURMode, MAPMode, CONTOURMethod,
                          STATUSMode, load_config, save_config)
from IVAS.theme import theme_hex
from IVAS.ui.MainWindow import Ui_MainWindow
from IVAS.widgets.annos_dock_widget import AnnosDockWidget
from IVAS.widgets.canvas import AnnotationScene, AnnotationView
from IVAS.widgets.mask_prediction import restore_cached_image
from IVAS.widgets.category_dock_widget import CategoriesDockWidget
from IVAS.widgets.files_dock_widget import FilesDockWidget
from IVAS.widgets.info_dock_widget import InfoDockWidget
from IVAS.widgets.text_prompt_dock_widget import TextPromptDockWidget
from IVAS.widgets.visuall_prompt_dock_widget import VisualPromptDockWidget
from IVAS.widgets.plugin_manager_dialog import PluginManagerDialog
from IVAS.widgets.polygon import OBB, Polygon, PromptPoint
from IVAS.widgets.right_button_menu import RightButtonMenu
from IVAS.widgets.setting_dialog import SettingDialog
from IVAS.widgets.responsive import CurrentWorkspaceStack, scrollable_content


def _find_category_config(directory):
    for name in (CATEGORY_CONFIG_NAME, *LEGACY_CATEGORY_CONFIG_NAMES):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def _default_video_output_root():
    """Return a writable, visible default beside the app or source checkout."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "outputs"
    return Path(IVAS_ROOT).parent / "outputs"


_PRECISION_ALIASES = {
    "auto": "original",
    "checkpoint": "original",
    "native": "original",
    "original": "original",
    "bf": "bfloat16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp": "float16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
    "float32": "float32",
}


def _normalize_precision_config(precision=None, use_bfloat16=None) -> str:
    """Normalize the persisted precision value without importing Torch.

    The full dtype-aware policy remains in ``segment_any.precision`` and is
    imported only when a local model is requested.  Startup configuration only
    contains strings and the legacy boolean, so pulling a multi-second ML
    runtime into the first-paint path is unnecessary.
    """

    if precision is None or str(precision).strip() == "":
        if use_bfloat16 is None:
            return "original"
        return "bfloat16" if bool(use_bfloat16) else "float32"
    value = str(precision).strip().lower().replace("torch.", "")
    try:
        return _PRECISION_ALIASES[value]
    except KeyError as exc:
        raise ValueError("Unsupported SAM precision: {!r}".format(precision)) from exc


def _record_startup_stage(name: str, started: float) -> None:
    """Write constructor timing only for the frozen build smoke route."""

    if "--build-smoke-test" not in sys.argv:
        return
    log_path = (os.environ.get("IVAS_BUILD_SMOKE_LOG") or os.environ.get("IVAT_BUILD_SMOKE_LOG"))
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(
            "MAINWINDOW_STAGE {} seconds={:.3f}\n".format(
                name, time.perf_counter() - started
            )
        )


class QtBoxStyleProgressBar(QtWidgets.QProgressBar):
    # copy from qtbox
    def __init__(self):
        super(QtBoxStyleProgressBar, self).__init__()
        self.setTextVisible(False)
        self.setStyleSheet(
            """
        QProgressBar {
            border: 1px solid %(border)s;
            border-radius: 5px;
        }

        QProgressBar::chunk {
            background-color: %(primary)s;
            border-radius: 2px;
            width: 9px;
            margin: 0.5px;
        }
        """
            % {"border": theme_hex("border"), "primary": theme_hex("primary")}
        )


def calculate_area(points: list) -> float:
    """
    Calculate the area of an array of points.

    Arguments:
        points: Vertices of polygon. [(x1, y1), (x2, y2), ...]

    Returns:
        The area of the polygon.
    """
    area = 0
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        d = p1[0] * p2[1] - p2[0] * p1[1]
        area += d
    return abs(area) / 2


class SegAnyThread(QThread):
    """
    Thread for encoder of the SAM model.

    Arguments:
        mainwindow (IVAS.widgets.mainwindow.MainWindow): Get attributes and functions from mainwindow.

    Attributes:
        tag (pyqtSignal): Signal.
        results_dict (dict): Dictionary of results. {image_index: {"feature": np.ndarray or tensor, ...}, ...}
        index: Current image index.
    """

    tag = pyqtSignal(int, int, str)
    statusReady = pyqtSignal(int, int, int, str)

    def __init__(self, mainwindow):
        super(SegAnyThread, self).__init__()
        self.mainwindow = mainwindow
        self.results_dict = {}
        self.index = None
        self.segany = mainwindow.segany
        self._request_lock = threading.Lock()
        self._request = None
        self._serial = 0
        self._completed_serial = 0
        self.finished.connect(self._restart_pending)
        self.statusReady.connect(self._forward_status)

    @QtCore.pyqtSlot(int, int, int, str)
    def _forward_status(self, serial, index, state, message):
        # Queued signals can outlive a frame or directory change.
        with self._request_lock:
            current = self._request is not None and self._request[0] == serial
        if current:
            self.tag.emit(index, state, message)

    def request_image(self, index):
        """Replace pending work with the latest frame without waiting for inference."""
        with self._request_lock:
            self.index = index
            self._serial += 1
            self._request = (
                self._serial, index, self.mainwindow.image_root,
                tuple(self.mainwindow.files_list),
            )
        if not self.isRunning():
            self.start()

    def invalidate(self):
        """Discard pending work and cached images when the directory changes."""
        with self._request_lock:
            self._serial += 1
            self._request = None
            self.results_dict.clear()

    @QtCore.pyqtSlot()
    def _restart_pending(self):
        # A request can arrive between run() returning and QThread finishing.
        with self._request_lock:
            pending = self._request and self._request[0] != self._completed_serial
        if pending and not self.isRunning() and not self.mainwindow._shutdown_started:
            self.start()

    def sam_encoder(self, image: np.ndarray):
        """
        SAM encoder.

        Arguments:
            image : Image to be encoded.
        """
        if self.mainwindow.use_remote_sam:
            return self._sam_encoder_remote(image)
        return self._sam_encoder_local(image)

    def _sam_encoder_remote(self, image: np.ndarray):
        """Encode image via remote SAM server."""
        import orjson
        import requests
        import torch

        shape = ",".join(map(str, image.shape))
        dtype = image.dtype.name
        response = requests.post(
            url=f"http://{self.mainwindow.remote_sam_dialog.lineEdit_host.text()}:{self.mainwindow.remote_sam_dialog.lineEdit_port.text()}/api/encode",
            files={"file": ("", image.tobytes(), "application/octet-stream")},
            data={"dtype": dtype, "shape": shape},
            timeout=(3, 120),
        )

        if response.status_code != 200:
            raise RuntimeError(response.status_code)

        data = orjson.loads(response.content)
        features = data["features"]
        original_size = data["original_size"]
        input_size = data["input_size"]

        dtype = self.segany.model_dtype
        device = self.segany.device

        if self.segany.model_source == "sam_hq":
            features, interm_features = features
            features = torch.tensor(features, dtype=torch.float32, device=device)
            interm_features = [
                torch.tensor(interm_feature, dtype=dtype, device=device)
                for interm_feature in interm_features
            ]
            features = (features, interm_features)
        elif "sam2" in self.segany.model_source:
            image_embed = features["image_embed"]
            high_res_feats = features["high_res_feats"]
            image_embed = torch.tensor(image_embed, dtype=torch.float32, device=device)
            high_res_feats = [
                torch.tensor(high_res_feat, dtype=dtype, device=device)
                for high_res_feat in high_res_feats
            ]
            features["image_embed"] = image_embed
            features["high_res_feats"] = high_res_feats
        else:
            features = torch.tensor(features, dtype=torch.float32, device=device)
        return features, original_size, input_size

    def _sam_encoder_local(self, image: np.ndarray):
        """Encode image via local SAM model."""
        import torch
        from IVAS.segment_any.precision import autocast_context

        lock = getattr(self.segany, "inference_lock", nullcontext())
        with lock, torch.inference_mode(), autocast_context(
            self.segany.device,
            self.segany.model_dtype,
        ):
            model_type = self.segany.model_type
            if "sam2" in model_type:
                return self._sam_encoder_local_sam2(image)
            elif "sam3" in model_type:
                return self._sam_encoder_local_sam3(image)
            else:
                return self._sam_encoder_local_sam1(image)

    def _sam_encoder_local_sam2(self, image: np.ndarray):
        """Local encode for SAM2 models."""
        _orig_hw = tuple([image.shape[:2]])
        input_image = self.segany.predictor._transforms(image)
        input_image = input_image[None, ...].to(
            self.segany.predictor.device
        )
        backbone_out = self.segany.predictor.model.forward_image(
            input_image
        )
        _, vision_feats, _, _ = (
            self.segany.predictor.model._prepare_backbone_features(
                backbone_out
            )
        )
        if self.segany.predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = (
                vision_feats[-1]
                + self.segany.predictor.model.no_mem_embed
            )
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1],
                self.segany.predictor._bb_feat_sizes[::-1],
            )
        ][::-1]
        _features = {
            "image_embed": feats[-1],
            "high_res_feats": tuple(feats[:-1]),
        }
        return _features, _orig_hw, _orig_hw

    def _sam_encoder_local_sam3(self, image: np.ndarray):
        """Local encode for SAM3 models."""
        _features, _orig_hw = self.segany.predictor.encode(image)
        return _features, _orig_hw, _orig_hw

    def _sam_encoder_local_sam1(self, image: np.ndarray):
        """Local encode for SAM1 models."""
        import torch

        input_image = self.segany.predictor.transform.apply_image(image)
        input_image_torch = torch.as_tensor(
            input_image, device=self.segany.predictor.device
        )
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]

        original_size = image.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])

        input_image = self.segany.predictor.model.preprocess(
            input_image_torch
        )
        features = self.segany.predictor.model.image_encoder(
            input_image
        )
        return features, original_size, input_size

    def run(self):
        while not self.isInterruptionRequested():
            with self._request_lock:
                request = self._request
                if request is None or request[0] == self._completed_serial:
                    return
                serial, index, image_root, files = request
                cached = self.results_dict.get(index)
            image_path = os.path.join(image_root, files[index])
            try:
                if cached is None or cached.get("image_path") != image_path:
                    self.statusReady.emit(serial, index, 2, "")
                    if image_path.lower().endswith(".dcm"):
                        from IVAS.utils.dicom import load_dcm_as_image

                        image_data = np.array(load_dcm_as_image(image_path).convert("RGB"))
                    else:
                        with Image.open(image_path) as image:
                            image_data = np.array(image.convert("RGB"))
                    features, original_size, input_size = self.sam_encoder(image_data)
                    cached = {
                        "features": features, "original_size": original_size,
                        "input_size": input_size, "image_path": image_path,
                    }
            except Exception as error:
                with self._request_lock:
                    self._completed_serial = serial
                    if self._request == request:
                        self.statusReady.emit(serial, index, 3, str(error))
                continue
            with self._request_lock:
                self._completed_serial = serial
                if self._request != request or self.isInterruptionRequested():
                    continue
                # Keep completed neighbours without starting speculative GPU
                # work while the user interacts with the current frame.
                radius = 0 if self.segany.model_source == "sam3.1" else 1
                for old_index in list(self.results_dict):
                    if abs(old_index - index) > radius:
                        del self.results_dict[old_index]
                        self.statusReady.emit(serial, old_index, 0, "")
                self.results_dict[index] = cached
                self.statusReady.emit(serial, index, 1, "")


class SegAnyVideoThread(QThread):
    """
    Thread for video segmentation of the SAM model.

    Arguments:
        mainwindow (IVAS.widgets.mainwindow.MainWindow): Get attributes and functions from mainwindow.

    Attributes:
        tag (pyqtSignal): Signal.
        start_frame_idx (int): The index of the start frame of the video.
        max_frame_num_to_track (int): The maximum number of frames to track.
    """

    tag = pyqtSignal(
        int, int, bool, bool, str
    )  # current, total, finished, is_error, message

    def __init__(self, mainwindow):
        super(SegAnyVideoThread, self).__init__()
        self.mainwindow = mainwindow
        self.start_frame_idx = 0
        self.max_frame_num_to_track = None

    def run(self):
        import torch
        from IVAS.segment_any.precision import autocast_context
        from skimage.draw.draw import polygon

        print("self.start_frame_idx: ", self.start_frame_idx)
        print("self.max_frame_num_to_track: ", self.max_frame_num_to_track)

        if self.max_frame_num_to_track is not None:
            total = self.max_frame_num_to_track
        else:
            total = len(self.mainwindow.files_list) - self.start_frame_idx + 1

        with torch.inference_mode(), autocast_context(
            self.mainwindow.segany_video.device,
            self.mainwindow.segany_video.model_dtype,
        ):

            if not self.mainwindow.use_segment_anything_video:
                self.mainwindow.actionVideo_segment.setEnabled(False)
                self.mainwindow.actionVideo_segment_once.setEnabled(False)
                self.mainwindow.actionVideo_segment_five_times.setEnabled(False)
                return

            if self.mainwindow.segany_video.inference_state == {}:
                self.mainwindow.segany_video.init_state(
                    self.mainwindow.image_root, self.mainwindow.files_list
                )
            self.mainwindow.segany_video.reset_state()

            current_file = self.mainwindow.files_list[self.start_frame_idx]
            current_file_path = os.path.join(self.mainwindow.image_root, current_file)
            current_label_path = os.path.join(
                self.mainwindow.label_root,
                ".".join(current_file.split(".")[:-1]) + ".json",
            )
            current_label = Annotation(current_file_path, current_label_path)

            current_label.load_annotation()

            group_object_dict = {}

            for object in current_label.objects:
                group = int(object.group)
                segmentation = [(int(p[1]), int(p[0])) for p in object.segmentation]
                category = object.category
                is_crowd = object.iscrowd
                layer = object.layer
                note = object.note

                # fill mask
                mask = np.zeros(
                    shape=(current_label.height, current_label.width), dtype=np.uint8
                )
                xs = [x for x, y in segmentation]
                ys = [y for x, y in segmentation]
                rr, cc = polygon(xs, ys, mask.shape)
                mask[rr, cc] = 1

                if group not in group_object_dict:
                    group_object_dict[group] = {}
                    group_object_dict[group]["mask"] = mask
                    group_object_dict[group]["category"] = category
                    group_object_dict[group]["is_crowd"] = is_crowd
                    group_object_dict[group]["layer"] = layer
                    group_object_dict[group]["note"] = note
                else:
                    group_object_dict[group]["mask"] = (
                        group_object_dict[group]["mask"] + mask
                    )

            if len(group_object_dict) < 1:
                self.tag.emit(
                    0, total, True, True, "Please label objects before video segment."
                )
                return
            try:
                for group, object_dict in group_object_dict.items():
                    mask = object_dict["mask"]
                    self.mainwindow.segany_video.add_new_mask(
                        self.start_frame_idx, group, mask
                    )

                for index, (out_frame_idxs, out_obj_ids, out_mask_logits) in enumerate(
                    self.mainwindow.segany_video.predictor.propagate_in_video(
                        self.mainwindow.segany_video.inference_state,
                        start_frame_idx=self.start_frame_idx,
                        max_frame_num_to_track=self.max_frame_num_to_track,
                        reverse=False,
                    )
                ):
                    if index == 0:  # 忽略当前图片
                        continue
                    file = self.mainwindow.files_list[out_frame_idxs]
                    file_path = os.path.join(self.mainwindow.image_root, file)
                    label_path = os.path.join(
                        self.mainwindow.label_root,
                        ".".join(file.split(".")[:-1]) + ".json",
                    )
                    annotation = Annotation(file_path, label_path)

                    objects = []
                    for index_mask, out_obj_id in enumerate(out_obj_ids):

                        mask = out_mask_logits[index_mask]  # [1, h, w]
                        mask = mask > 0
                        mask = mask.cpu().numpy()

                        # mask to polygon
                        contours, hierarchy = self.mainwindow.mask_to_polygon(mask)

                        for contour in contours:
                            if len(contour) < 3:
                                continue

                            segmentation = []
                            xmin, ymin, xmax, ymax = (
                                annotation.width,
                                annotation.height,
                                0,
                                0,
                            )
                            for point in contour:
                                x, y = point[0]
                                x, y = float(x), float(y)
                                xmin = min(x, xmin)
                                ymin = min(x, ymin)
                                xmax = max(y, xmax)
                                ymax = max(y, ymax)

                                segmentation.append((x, y))

                            area = calculate_area(segmentation)
                            # bbox = (xmin, ymin, xmax, ymax)
                            bbox = None
                            obj = Object(
                                category=group_object_dict[out_obj_id]["category"],
                                group=out_obj_id,
                                segmentation=segmentation,
                                area=area,
                                layer=group_object_dict[out_obj_id]["layer"],
                                bbox=bbox,
                                iscrowd=group_object_dict[out_obj_id]["is_crowd"],
                                note=group_object_dict[out_obj_id]["note"],
                            )
                            objects.append(obj)

                    annotation.objects = objects
                    annotation.save_annotation()
                    self.tag.emit(index, total, False, False, "")

                self.tag.emit(index, total, True, False, "")

            except Exception as e:
                self.tag.emit(index, total, True, True, "{}".format(e))


class InitSegAnyThread(QThread):
    """
    Thread for init SAM model.

    Arguments:
        mainwindow (IVAS.widgets.mainwindow.MainWindow): Get attributes and functions from mainwindow.

    Attributes:
        tag (pyqtSignal): Signal.
        model_path: Checkpoint file path.
    """

    tag = pyqtSignal(bool, bool)

    def __init__(self, mainwindow):
        super(InitSegAnyThread, self).__init__()
        self.mainwindow = mainwindow
        self.model_path: str = None
        self.error_message = ""

    def run(self):
        import torch
        from IVAS.segment_any.segment_any import SegAny, SegAnyVideo

        sam_tag = False
        sam_video_tag = False
        if self.model_path is not None:
            # 在后台线程中释放旧模型资源，避免主线程阻塞
            # 先保存旧模型引用并立即置空 mainwindow 的引用，防止主线程访问已释放的模型
            old_segany = getattr(self.mainwindow, 'segany', None)
            if old_segany is not None:
                self.mainwindow.segany = None
            old_segany_video = getattr(self.mainwindow, 'segany_video', None)
            if old_segany_video is not None:
                self.mainwindow.segany_video = None

            # 释放旧模型资源（在后台线程执行，不阻塞 UI）
            # 视频模型的 inference_state 可能包含大量帧数据，释放较慢
            if old_segany is not None:
                try:
                    old_segany.release()
                except Exception:
                    pass
            if old_segany_video is not None:
                try:
                    old_segany_video.release()
                except Exception:
                    pass

            # Drop the final model owners before returning unused allocator
            # blocks. Otherwise model replacement can transiently double VRAM.
            del old_segany
            del old_segany_video
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                self.mainwindow.segany = SegAny(
                    self.model_path,
                    precision=self.mainwindow.cfg["software"]["sam_precision"],
                )
                sam_tag = True
            except Exception as e:
                print("Init SAM Error: ", e)
                self.error_message = str(e)
                self.mainwindow.segany = None
                sam_tag = False
            model_name = os.path.basename(self.model_path).lower()
            supports_video_facade = "sam2" in model_name or "sam3" in model_name
            if (
                self.mainwindow.cfg["software"]["use_video_segmentation"]
                and supports_video_facade
            ):
                try:
                    self.mainwindow.segany_video = SegAnyVideo(
                        self.model_path,
                        precision=self.mainwindow.cfg["software"]["sam_precision"],
                    )
                    sam_video_tag = True
                except Exception as e:
                    print("Init SAM2 video Error: ", e)
                    sam_video_tag = False
            else:
                self.mainwindow.segany_video = None

        sam_video_tag = sam_tag and sam_video_tag
        self.tag.emit(sam_tag, sam_video_tag)


class CheckLatestVersionThread(QThread):
    """
    Thread for checking latest version.

    Arguments:
        mainwindow (IVAS.widgets.mainwindow.MainWindow): Get attributes and functions from mainwindow.

    Attributes:
        tag (pyqtSignal): Signal.
    """

    tag = pyqtSignal(bool, str)

    def __init__(self, mainwindow):
        super(CheckLatestVersionThread, self).__init__()
        self.mainwindow = mainwindow

    def run(self):
        try:
            import requests

            from IVAS import __version__

            response = requests.get(f"https://pypi.org/pypi/{DISTRIBUTION_NAME}/json", timeout=3)
            latest_version = response.json()["info"]["version"]
            if not self.isInterruptionRequested():
                self.tag.emit(__version__ == latest_version, latest_version)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.tag.emit(True, "")


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    The main window of IVAS, also contains most of the core functions.

    Attributes:
        image_root (str): Image root.
        label_root (str): Label root.
        files_list (list): List of images names.
        current_index (int): Current index for current image in files_list.
        current_group (int): Recode the current group. Default is 1.
        config_file (str): Categories config file path. Default is IVAS/ivas.yaml.
        software_config_file (str): Software config file path. Default is IVAS/software.yaml.
        saved (bool): The state of whether the annotations are written to disk.
        can_be_annotated (bool): The state of whether the annotation can be annotated. The png image with palette be used for VOC format, these can't be annotated.
        load_finished (bool): The state of whether the image load finished.
        polygons (list): List of polygons. These will be saved to disk as annotations.
        instance_cmap (np.ndarray): The color array. (N, 3), numpy.uint8
        map_mode (MAPMode): The map mode, means the current displayed state in the scene, which include 'LABEL', 'SEMANTIC' and 'INSTANCE'.
        current_label (Annotation): The current annotation.
        use_segment_anything (bool): The state of whether the SAM model can be used.
        use_segment_anything_video (bool): The state of whether the SAM model can be used to video segmentation.
        use_remote_sam (bool): The state of whether the remote sam is being used.
        group_select_mode (bool): The mode of group. Default is 'auto'.
    """

    # Color constants
    COLOR_DEFAULT = theme_hex("text-muted")
    COLOR_STATE_SUCCESS = theme_hex("success")
    COLOR_STATE_DELETED = theme_hex("text-muted")
    COLOR_STATE_WORKING = theme_hex("warning")
    # Mapping from SAM encoder state to display color
    STATE_COLOR_MAP = {
        0: theme_hex("text-muted"),
        1: theme_hex("success"),
        2: theme_hex("warning"),
        3: theme_hex("danger"),
    }

    def __init__(self):
        startup_started = time.perf_counter()
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setMinimumSize(480, 360)
        _record_startup_stage("base_ui", startup_started)

        self.image_root: str = None
        self.label_root: str = None

        self.files_list: list = []
        self.current_index = None

        self.current_group = 1
        self.current_category = "__background__"
        self.category_aliases = {}

        self.config_file = CONFIG_FILE
        self.software_config_file = SOFTWARE_CONFIG_FILE

        self.saved = True

        self.can_be_annotated = True
        self.load_finished = False
        self.polygons: list = []

        self.instance_cmap = None
        self.map_mode = MAPMode.LABEL
        # 标注目标
        self.current_label: Annotation = None
        self.use_segment_anything = False
        self.use_segment_anything_video = False
        self.use_remote_sam = False
        self.sam_model_generation = 0
        self.gpu_resource_thread = None
        self._shutdown_started = False
        self._loading_config = False
        self._active_video_frame_case = False
        self._video_frame_project_path = None
        self._loaded_frame_signature = None
        self._sam_ready_timer = QtCore.QTimer(self)
        self._sam_ready_timer.setSingleShot(True)
        self._sam_ready_timer.setInterval(50)
        self._sam_ready_timer.timeout.connect(self.SeganyEnabled)

        # 新增 手动/自动 group选择
        self.group_select_mode = "auto"
        self.init_ui()
        _record_startup_stage("workbench_ui", startup_started)
        self.reload_cfg()
        _record_startup_stage("config_loaded", startup_started)

        self.init_connect()
        self.reset_action()
        _record_startup_stage("actions_connected", startup_started)

        # sam初始化线程，大模型加载较慢
        self.init_segany_thread = InitSegAnyThread(self)
        self.init_segany_thread.tag.connect(self.init_sam_finish)

        # 检查最新版本
        self.check_latest_version_thread = CheckLatestVersionThread(self)
        self.check_latest_version_thread.tag.connect(self.latest_version_tip)
        # Let the first window paint before importing requests and touching the
        # network; this work is unrelated to opening a local annotation case.
        self._version_check_timer = QtCore.QTimer(self)
        self._version_check_timer.setSingleShot(True)
        self._version_check_timer.timeout.connect(self._start_version_check)
        self._version_check_timer.start(1200)
        _record_startup_stage("threads_deferred", startup_started)

        self.plugin_manager_dialog.trigger_application_start()
        _record_startup_stage("plugins_started", startup_started)

    def _start_version_check(self):
        if (
            not self._shutdown_started
            and not self.check_latest_version_thread.isRunning()
        ):
            self.check_latest_version_thread.start()

    def init_segment_anything(self, model_path: str = None, checked: bool = True):
        """
        Initialize segment anything model.

        Arguments:
            model_path (str): The path of the checkpoint file.
            checked (bool): If not checked, skip.
        """
        if not checked:
            return

        mask_worker = self.scene._mask_worker
        if mask_worker is not None:
            self.statusbar.showMessage("当前分割任务尚未结束，请完成或取消后再切换模型。", 4000)
            return

        if model_path is None:
            if self.use_segment_anything:
                model_path = self.segany.checkpoint
            else:
                return

        model_name = os.path.basename(model_path)

        if not self.saved:
            result = QtWidgets.QMessageBox.question(
                self,
                "Warning",
                "Proceed without saved?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if result == QtWidgets.QMessageBox.StandardButton.No:
                if isinstance(self.sender(), QtWidgets.QAction):
                    self.sender().setChecked(False)
                return

        # 等待sam线程完成
        self.actionSegment_anything_point.setEnabled(False)
        self.actionSegment_anything_box.setEnabled(False)
        try:
            self.seganythread.invalidate()
            self.seganythread.requestInterruption()
            self.seganythread.wait()
        except:
            if isinstance(self.sender(), QtWidgets.QAction):
                self.sender().setChecked(False)

        # 注意：模型资源的释放在后台线程 InitSegAnyThread.run() 中执行，
        # 避免在主线程同步释放大量显存（尤其是视频分割后）导致界面卡死

        if model_name == "":
            self.use_segment_anything = False
            self._ensure_model_manager_dialog().update_ui()
            return

        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "The checkpoint of [Segment anything] not existed. If you want use quick annotate, please download from {}".format(
                    "https://github.com/facebookresearch/segment-anything#model-checkpoints"
                ),
            )
            self.use_segment_anything = False
            self._ensure_model_manager_dialog().update_ui()
            return

        # 每次都创建新的线程，避免 QThread 重启带来的潜在问题
        self.sam_model_generation += 1
        self.init_segany_thread = InitSegAnyThread(self)
        self.init_segany_thread.tag.connect(self.init_sam_finish)
        self.init_segany_thread.model_path = model_path
        self.init_segany_thread.start()
        self.setEnabled(False)

    def init_sam_finish(self, sam_tag: bool, sam_video_tag: bool):
        """
        Triggered when SAM finish is finished. Then start thread of sam encoder and video segmentation.

        .. warning::

            Do not call this function manually

        Arguments:
            sam_tag (bool): The state of whether SAM can be used.
            sam_video_tag (bool): The state of whether video segmentation can be used.
        """
        if self.use_remote_sam:
            sam_video_tag = False

        torch = None
        if sam_tag:
            import torch as _torch

            torch = _torch

        print(
            "sam_tag:",
            f"\033[32m{sam_tag}\033[0m" if sam_tag else f"\033[31m{sam_tag}\033[0m",
            "sam_video_tag: ",
            (
                f"\033[32m{sam_video_tag}\033[0m"
                if sam_video_tag
                else f"\033[31m{sam_video_tag}\033[0m"
            ),
        )
        self.setEnabled(True)
        if sam_video_tag:
            self.use_segment_anything_video = True
            # The model facade remains available to the unified transactional
            # workspace. Do not instantiate the legacy direct-write worker.
            self.segany_video_thread = None

            # sam2 建议使用bfloat16
            if self.segany_video.model_dtype == torch.float32 and "sam2" in self.segany.model_source and torch.cuda.is_available():
                if self.cfg["software"]["language"] == "zh":
                    QtWidgets.QMessageBox.warning(
                        self,
                        "warning",
                        """建议使用bfloat16模式进行视频分割\n在[设置]界面打开该功能""",
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "warning",
                        """Suggest Use bfloat16 mode to segment video.\nYou can open it in [Setting].""",
                    )

        else:
            self.segany_video_thread = None
            self.use_segment_anything_video = False
            # torch.cuda.empty_cache() 已在后台线程 InitSegAnyThread.run() 中调用，
            # 此处无需在主线程重复调用，避免 GIL 阻塞 UI

        self.actionVideo_segment.setEnabled(self.use_segment_anything_video)
        self.actionVideo_segment_once.setEnabled(self.use_segment_anything_video)
        self.actionVideo_segment_five_times.setEnabled(self.use_segment_anything_video)
        self._update_tracking_action_safety()

        if sam_tag:
            self.use_segment_anything = True
            if self.use_segment_anything:
                if self.segany.device != "cpu":
                    self._start_gpu_monitor()
                else:
                    self._stop_gpu_monitor()
                    self.labelGPUResource.setText("cpu")
            else:
                self.labelGPUResource.setText("segment anything unused.")
            tooltip = "model: {}".format(os.path.split(self.segany.checkpoint)[-1])
            tooltip += "\ndtype: {}".format(self.segany.model_dtype)
            tooltip += "\ntorch: {}".format(torch.__version__)
            if self.segany.device == "cuda":
                try:
                    tooltip += "\ncuda : {}".format(torch.version.cuda)
                except:
                    pass
            self.labelGPUResource.setToolTip(tooltip)

            self.seganythread = SegAnyThread(self)
            self.seganythread.tag.connect(self.sam_encoder_finish)
            self.seganythread.finished.connect(self.SeganyEnabled)

            if self.current_index is not None and self.annotation_mode == "Frame":
                self.show_image(self.current_index)

            checkpoint_name = os.path.split(self.segany.checkpoint)[-1]
            self.statusbar.showMessage(
                "Use the checkpoint named {}.".format(checkpoint_name), 3000
            )

            # 非sam3/sam3.1时，隐藏text prompt 与 visual prompt窗口
            is_sam3_model = self.segany.model_source in ("sam3", "sam3.1")
            for dock in (self.text_prompt_dock, self.visual_prompt_dock):
                dock.setVisible(is_sam3_model and self.annotation_mode == "Frame")
                if self._image_workspace_dock_visibility is not None:
                    self._image_workspace_dock_visibility[dock] = is_sam3_model

        else:
            self._stop_gpu_monitor()
            self.use_segment_anything = False
            self.text_prompt_dock.setVisible(False)
            self.visual_prompt_dock.setVisible(False)
            error_message = getattr(
                self.init_segany_thread, "error_message", ""
            )
            if error_message:
                QtWidgets.QMessageBox.warning(
                    self, "SAM initialization failed", error_message
                )

        self._ensure_model_manager_dialog().update_ui()

    def _start_gpu_monitor(self):
        """Start exactly one low-overhead status-bar GPU monitor."""

        from IVAS.segment_any.gpu_resource import GPUResource_Thread

        self._stop_gpu_monitor()
        self.gpu_resource_thread = GPUResource_Thread(parent=self)
        self.gpu_resource_thread.message.connect(self.labelGPUResource.setText)
        self.gpu_resource_thread.start()

    def _stop_gpu_monitor(self):
        monitor = self.gpu_resource_thread
        self.gpu_resource_thread = None
        if monitor is not None:
            monitor.stop()
            monitor.deleteLater()

    def sam_encoder_finish(self, index: int, state: int, message: str):
        """
        Triggered when sam encoder is finished.

        .. warning::

            Do not call this function manually

        Arguments:
            index (int): The index of image.
            state (int): The state of sam encoder result. 0 - delete; 1 - success; 2 - working; 3 - error.
            message (str): Error message.
        """
        if self.current_index is None or self._shutdown_started:
            return
        if self.sender() is not None and self.sender() is not self.seganythread:
            return
        color = self.STATE_COLOR_MAP.get(state, self.COLOR_STATE_DELETED)

        if state == 3 and index == self.current_index:
            QtWidgets.QMessageBox.warning(
                self,
                "warning",
                "SAM not support the image: {}\nError: {}".format(
                    self.files_list[index], message
                ),
            )

        # Update file dock state indicators
        state_labels = {
            self.current_index: self.files_dock_widget.label_current_state,
            self.current_index - 1: self.files_dock_widget.label_prev_state,
            self.current_index + 1: self.files_dock_widget.label_next_state,
        }
        if index in state_labels:
            state_labels[index].setStyleSheet(
                "background-color: {};".format(color)
            )

        # item = self.files_dock_widget.listWidget.item(index)
        # widget = self.files_dock_widget.listWidget.itemWidget(item)
        # if widget is not None:
        #     state_color = widget.findChild(QtWidgets.QLabel, 'state_color')
        #     state_color.setStyleSheet("background-color: {};".format(color))

        if state == 1:  # 识别完
            # 如果当前图片刚识别完，需刷新segany状态
            if self.current_index == index:
                self.SeganyEnabled()
                self.plugin_manager_dialog.trigger_after_sam_encode_finished(index)

    def SeganyEnabled(self):
        """If current image has cached feature map by SAM encoder thread, enable semi-automatic annotation."""
        self.actionSegment_anything_point.setEnabled(False)
        self.actionSegment_anything_box.setEnabled(False)
        if self._shutdown_started:
            return
        worker = getattr(self.text_prompt_dock_widget, "_worker", None)
        if worker is not None and worker.isRunning():
            return
        if (self.annotation_mode != "Frame" or self.scene.mode != STATUSMode.VIEW
                or self.map_mode != MAPMode.LABEL or not self.can_be_annotated):
            self.actionSegment_anything_point.setEnabled(False)
            self.actionSegment_anything_box.setEnabled(False)
            return
        if not self.use_segment_anything:
            self.actionSegment_anything_point.setEnabled(False)
            self.actionSegment_anything_box.setEnabled(False)
            return

        if self.segany is None:
            self.actionSegment_anything_point.setEnabled(False)
            self.actionSegment_anything_box.setEnabled(False)
            return

        results = self.seganythread.results_dict.get(self.current_index, {})
        features = results.get("features", None)
        original_size = results.get("original_size", None)
        input_size = results.get("input_size", None)

        if (
            features is not None
            and original_size is not None
            and input_size is not None
        ):
            lock = self.segany.inference_lock
            if not lock.acquire(blocking=False):
                self._sam_ready_timer.start()
                return
            try:
                restore_cached_image(self.segany, results)
            finally:
                lock.release()

            self.actionSegment_anything_point.setEnabled(True)
            self.actionSegment_anything_box.setEnabled(True)
        # Missing features leave tools disabled. Resetting the predictor here
        # would wait on an encoder/text worker and freeze the Qt event loop.

    def seg_video_start(self, max_frame_num_to_track: int = None):
        """
        Start video segmentation.

        Arguments:
            max_frame_num_to_track (int): The maximum number of frames to track. Default: None, all frames.
        """
        del max_frame_num_to_track
        QtWidgets.QMessageBox.warning(
            self,
            "旧追踪已停用",
            "旧的图像序列追踪会直接覆盖目标标签，现已停用。\n"
            "请通过“文件 → 打开视频项目”，再从视频工作台“更多”菜单启动可审核、可回滚的事务追踪。",
        )

    def seg_video_finish(
        self, current: int, total: int, finished: bool, is_error: bool, message: str
    ):
        """
        Triggered when video segmentation finished.

        .. warning::

            Do not call this function manually

        Arguments:
            current (int): The index of current frame to segmentation.
            total (int): The total number of frames.
            finished (bool): The state of whether current frame is finished.
            is_error (bool): The state of whether current frame error.
            message (str): Error message.current, total, finished, is_error, message. int, int, bool, bool, str
        """
        if is_error:
            QtWidgets.QMessageBox.warning(self, "warning", message)

        self.progressbar.setMaximum(total)
        self.progressbar.setValue(current)
        if finished:
            self.statusbar_change_status(is_message=True)
            self.progressbar.setValue(0)
            self.setEnabled(True)

    def predict_current_image_with_text_prompt(
        self, prompt: str = None, category: str = None
    ):
        if not prompt:
            return
        if self.current_index is None or self.annotation_mode != "Frame":
            return
        if (not self.use_segment_anything) or self.segany.model_source not in (
            "sam3",
            "sam3.1",
        ):
            print("Only SAM3 is supported!")
            return

        category = category or self.current_category
        if category == "__background__":
            return 0
        if self.segany.model_source == "sam3.1":
            masks, scores = self.segany.predict_with_text_prompt(prompt)
        else:
            file_path = os.path.join(
                self.image_root, self.files_list[self.current_index]
            )
            image = Image.open(file_path).convert("RGB")
            with self.segany.inference_lock:
                masks, scores = (
                    self.segany.predictor.predict_with_text_prompt(image, prompt)
                )
        return self._process_sam_masks(masks, scores, category)

    def predict_current_image_with_visual_prompt(
        self, visual_category: str, boxes: list, labels: list
    ):
        if len(boxes) != len(labels):
            return
        if self.current_index is None or self.annotation_mode != "Frame":
            return
        if (not self.use_segment_anything) or self.segany.model_source not in (
            "sam3",
            "sam3.1",
        ):
            print("Only SAM3 is supported!")
            return

        file_path = os.path.join(self.image_root, self.files_list[self.current_index])
        with self.segany.inference_lock:
            image = (
                None
                if self.segany.model_source == "sam3.1"
                else Image.open(file_path).convert("RGB")
            )
            masks, scores = self.segany.predictor.predict_with_visual_prompt(
                image, boxes, labels
            )
        return self._process_sam_masks(masks, scores, visual_category)

    def mask_to_polygon(self, mask:np.ndarray):
        # OpenCV is only needed after an image has been segmented. Importing
        # it here keeps native DLL discovery off the frozen first-paint path.
        import cv2

        mask = mask.astype("uint8") * 255
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w)

        if self.scene.contour_method == CONTOURMethod.SIMPLE:
            contour_method = cv2.CHAIN_APPROX_SIMPLE
        elif self.scene.contour_method == CONTOURMethod.TC89_KCOS:
            contour_method = cv2.CHAIN_APPROX_TC89_KCOS
        elif self.scene.contour_method == CONTOURMethod.NONE:
            contour_method = cv2.CHAIN_APPROX_NONE
        else:
            contour_method = cv2.CHAIN_APPROX_SIMPLE

        if self.scene.contour_mode == CONTOURMode.SAVE_ALL:
            # 当保留所有轮廓时，检测所有轮廓，并建立二层等级关系
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, contour_method
            )
        else:
            # 当只保留外轮廓或单个mask时，只检测外轮廓
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, contour_method
            )

        if contours and (self.scene.contour_mode == CONTOURMode.SAVE_MAX_ONLY):
            largest_contour = max(
                contours, key=cv2.contourArea
            )  # 只保留面积最大的轮廓
            contours = [largest_contour]

        # polydp
        if self.cfg["software"]["use_polydp"]:
            epsilon_factor = 0.001
            polydp_contours = []
            for contour in contours:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
                polydp_contours.append(contour)
            contours = polydp_contours

        return contours, hierarchy

    def _setup_polygon_alpha(self, polygon):
        """Set hover and nohover alpha for a polygon from config."""
        polygon.hover_alpha = int(
            self.cfg["software"]["polygon_alpha_hover"] * 255
        )
        polygon.nohover_alpha = int(
            self.cfg["software"]["polygon_alpha_no_hover"] * 255
        )

    def _create_polygon_from_contour(
        self, contour, hierarchy, index: int, category: str, group: int
    ):
        """Create a Polygon from a single contour and add it to the scene.

        Arguments:
            contour: OpenCV contour points.
            hierarchy: OpenCV contour hierarchy (may be None for RETR_EXTERNAL).
            index: Index of this contour in the hierarchy.
            category: Category name for the polygon.
            group: Group id for the polygon.
        """
        if len(contour) < 3:
            return False

        graph = Polygon()
        self.scene.addItem(graph)
        self._setup_polygon_alpha(graph)

        for point in contour:
            x, y = point[0]
            x = max(0.1, x)
            y = max(0.1, y)
            graph.addPoint(QtCore.QPointF(x, y))

        # Inner contours get __background__ when SAVE_ALL mode
        if (
            self.scene.contour_mode == CONTOURMode.SAVE_ALL
            and hierarchy is not None
            and hierarchy[0][index][3] != -1
        ):
            category = "__background__"
            group = 0

        graph.set_drawed(
            category,
            group,
            False,
            "",
            QtGui.QColor(self.category_color_dict.get(category, self.COLOR_DEFAULT)),
            len(self.polygons) + 1,
        )

        self.polygons.append(graph)
        self.annos_dock_widget.listwidget_add_polygon(graph)
        return True

    def _process_sam_masks(
        self, masks, scores, category: str, notify: bool = True
    ) -> int:
        """Process SAM-predicted masks into polygons and add them to the scene.

        Arguments:
            masks: List of binary masks from SAM prediction.
            scores: List of confidence scores from SAM prediction.
            category: Category name for the resulting polygons.

        Returns:
            Number of masks processed.
        """
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(
                "SAM masks must have shape HxW, NxHxW, or Nx1xHxW."
            )

        created_masks = 0
        seen_masks = set()
        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                continue
            fingerprint = np.packbits(mask).tobytes()
            if fingerprint in seen_masks:
                continue
            seen_masks.add(fingerprint)
            contours, hierarchy = self.mask_to_polygon(mask)
            created_for_mask = False
            for index, contour in enumerate(contours):
                created_for_mask = self._create_polygon_from_contour(
                    contour, hierarchy, index, category, self.current_group
                ) or created_for_mask
            if not created_for_mask:
                continue
            created_masks += 1
            if self.group_select_mode == "auto":
                self.current_group += 1
                self.categories_dock_widget.lineEdit_currentGroup.setText(
                    str(self.current_group)
                )
        if created_masks and notify:
            self.set_saved_state(False)
            self.plugin_manager_dialog.trigger_after_annotation_created()
        return created_masks

    def init_ui(self):
        self.scene = AnnotationScene(mainwindow=self)

        # Secondary tools are imported and constructed on first use.  Keeping
        # the public attributes preserves the existing plugin API while
        # avoiding their converters, validators, network clients and large UI
        # trees on the first-paint path.
        self.category_setting_dialog = None
        self.category_edit_widget = None
        self.Converter_dialog = None
        self.video2frames_dialog = None
        self.auto_segment_dialog = None
        self.annos_validator_dialog = None
        self.process_exif_dialog = None
        self.shortcut_dialog = None
        self.about_dialog = None
        self.model_manager_dialog = None
        self.remote_sam_dialog = None

        self.categories_dock_widget = CategoriesDockWidget(mainwindow=self)
        self.categories_dock.setWidget(self.categories_dock_widget)

        self.annos_dock_widget = AnnosDockWidget(mainwindow=self)
        self.annos_dock.setWidget(self.annos_dock_widget)

        self.files_dock_widget = FilesDockWidget(mainwindow=self)
        self.files_dock.setWidget(self.files_dock_widget)

        self.info_dock_widget = InfoDockWidget(mainwindow=self)
        self.info_dock.setWidget(self.info_dock_widget)

        self.text_prompt_dock_widget = TextPromptDockWidget(mainwindow=self)
        self.text_prompt_dock.setWidget(self.text_prompt_dock_widget)

        self.visual_prompt_dock_widget = VisualPromptDockWidget(mainwindow=self)
        self.visual_prompt_dock.setWidget(self.visual_prompt_dock_widget)

        self.plugin_manager_dialog = PluginManagerDialog(self, self)

        self.view = AnnotationView(parent=self)
        self.view.setObjectName("annotationCanvas")
        self.view.setScene(self.scene)
        self.workspace_stack = CurrentWorkspaceStack(self)
        self.workspace_stack.setObjectName("mediaWorkspaceStack")
        self.workspace_stack.addWidget(self.view)
        # Qt Multimedia can take more than a second to initialize on Windows.
        # Construct the video workspace only when the user opens one so the
        # default image-labeling window can paint immediately.
        self.video_annotation_workspace = None
        self.setCentralWidget(self.workspace_stack)
        self._image_workspace_dock_visibility = None
        self.active_video_project_path = None

        self.actionOpen_video_workspace = QtWidgets.QAction("Open video project…", self)
        self.actionOpen_video_workspace.setObjectName("actionOpen_video_workspace")
        self.actionOpen_video_workspace.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        self.actionOpen_video_workspace.setStatusTip(
            "Open a video for Phase/Task and frame segmentation annotation."
        )
        self.actionShow_image_workspace = QtWidgets.QAction("Image workspace", self)
        self.actionShow_image_workspace.setCheckable(True)
        self.actionShow_video_workspace = QtWidgets.QAction("Video workspace", self)
        self.actionShow_video_workspace.setCheckable(True)
        self.workspace_action_group = QtWidgets.QActionGroup(self)
        self.workspace_action_group.setExclusive(True)
        self.workspace_action_group.addAction(self.actionShow_image_workspace)
        self.workspace_action_group.addAction(self.actionShow_video_workspace)
        self.actionShow_image_workspace.setChecked(True)
        self.actionShow_image_workspace.setObjectName("actionFrame_mode")
        self.actionShow_video_workspace.setObjectName("actionVideo_mode")
        self._create_unified_video_actions()
        self._configure_unified_menus()

        self.toolBar.setIconSize(QtCore.QSize(22, 22))
        self.toolBar_2.setIconSize(QtCore.QSize(22, 22))
        self.setDockOptions(
            QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.AnimatedDocks
        )

        self.right_button_menu = RightButtonMenu(mainwindow=self)
        self.right_button_menu.addAction(self.actionEdit)
        self.right_button_menu.addAction(self.actionCopy)
        self.right_button_menu.addAction(self.actionTo_top)
        self.right_button_menu.addAction(self.actionTo_bottom)
        self.right_button_menu.addAction(self.actionDelete)

        self.setting_dialog = SettingDialog(self, self)

        self.labelGPUResource = QtWidgets.QLabel("")
        self.labelGPUResource.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelGPUResource.setMaximumWidth(150)
        self.statusbar.addPermanentWidget(self.labelGPUResource)

        self.labelCoord = QtWidgets.QLabel("")
        self.labelCoord.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelCoord.setMaximumWidth(110)
        self.statusbar.addPermanentWidget(self.labelCoord)

        self.labelData = QtWidgets.QLabel("")
        self.labelData.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelData.setMaximumWidth(120)
        self.statusbar.addPermanentWidget(self.labelData)

        # Human-readable workspace state; colour comes from the semantic theme.
        self.modeState = QtWidgets.QLabel("图像")
        self.modeState.setObjectName("selectedCategory")
        self.modeState.setAccessibleName("当前工作区")
        self.modeState.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.modeState.setMinimumWidth(64)
        self.statusbar.addPermanentWidget(self.modeState)

        self.progressbar = QtBoxStyleProgressBar()
        self.progressbar.setTextVisible(False)
        self.progressbar.setMaximumWidth(360)
        self.progressbar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progressbar)

        # image saturation  调整图像饱和度
        self.image_saturation = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, self
        )
        self.image_saturation.setMinimumWidth(110)
        self.image_saturation.setStatusTip("Image saturation.")
        self.image_saturation.setToolTip("Image saturation")
        self.image_saturation.setMaximum(500)
        self.image_saturation.setMinimum(0)
        self.image_saturation.setPageStep(10)
        self.image_saturation.setTickInterval(10)
        self.image_saturation.valueChanged.connect(self.change_saturation)
        self._create_temporal_project_docks()
        self._create_shared_video_transport_toolbar()
        self._configure_compact_toolbars()
        self._configure_canvas_first_layout()
        self._bind_unified_video_shortcuts()

        self.trans = QtCore.QTranslator()

    def _ensure_dialog(self, attribute_name, dialog_class, *args, **kwargs):
        """Import and create one secondary dialog, retaining its public slot."""

        dialog = getattr(self, attribute_name)
        if dialog is not None:
            return dialog

        dialog = dialog_class(*args, **kwargs)
        setattr(self, attribute_name, dialog)

        # setupUi() already sees the installed application translator.  An
        # explicit pass also covers dialogs first requested by startup plugins
        # while the MainWindow itself is finishing a language change.
        retranslate = getattr(dialog, "retranslateUi", None)
        if callable(retranslate):
            retranslate(dialog)
        return dialog

    def _ensure_category_setting_dialog(self):
        from IVAS.widgets.category_setting_dialog import CategorySettingDialog

        return self._ensure_dialog(
            "category_setting_dialog",
            CategorySettingDialog,
            parent=self,
            mainwindow=self,
        )

    def _ensure_category_edit_dialog(self):
        from IVAS.widgets.category_edit_dialog import CategoryEditDialog

        return self._ensure_dialog(
            "category_edit_widget",
            CategoryEditDialog,
            self,
            self,
            self.scene,
        )

    def _ensure_converter_dialog(self):
        from IVAS.widgets.converter_dialog import ConverterDialog

        return self._ensure_dialog(
            "Converter_dialog",
            ConverterDialog,
            self,
            mainwindow=self,
        )

    def _ensure_video2frames_dialog(self):
        from IVAS.widgets.video_to_frames_dialog import Video2FramesDialog

        return self._ensure_dialog(
            "video2frames_dialog",
            Video2FramesDialog,
            self,
            self,
        )

    def _ensure_auto_segment_dialog(self):
        from IVAS.widgets.auto_segment_dialog import AutoSegmentDialog

        return self._ensure_dialog(
            "auto_segment_dialog",
            AutoSegmentDialog,
            self,
            self,
        )

    def _ensure_annos_validator_dialog(self):
        from IVAS.widgets.annos_validator_dialog import AnnosValidatorDialog

        return self._ensure_dialog(
            "annos_validator_dialog",
            AnnosValidatorDialog,
            self,
            self,
        )

    def _ensure_process_exif_dialog(self):
        from IVAS.widgets.process_exif_dialog import ProcessExifDialog

        return self._ensure_dialog(
            "process_exif_dialog",
            ProcessExifDialog,
            self,
            self,
        )

    def _ensure_shortcut_dialog(self):
        from IVAS.widgets.shortcut_dialog import ShortcutDialog

        return self._ensure_dialog(
            "shortcut_dialog",
            ShortcutDialog,
            self,
            self,
        )

    def _ensure_about_dialog(self):
        from IVAS.widgets.about_dialog import AboutDialog

        return self._ensure_dialog(
            "about_dialog",
            AboutDialog,
            self,
        )

    def _ensure_model_manager_dialog(self):
        from IVAS.widgets.model_manager_dialog import ModelManagerDialog

        return self._ensure_dialog(
            "model_manager_dialog",
            ModelManagerDialog,
            self,
            self,
        )

    def _ensure_remote_sam_dialog(self):
        from IVAS.widgets.remote_sam_dialog import RemoteSamDialog

        return self._ensure_dialog(
            "remote_sam_dialog",
            RemoteSamDialog,
            self,
            self,
        )

    def _new_workbench_action(
        self, object_name, text, callback, status_tip="", enabled=False
    ):
        action = QtWidgets.QAction(text, self)
        action.setObjectName(object_name)
        action.setEnabled(enabled)
        action.setStatusTip(status_tip)
        action.setToolTip(status_tip or text)
        action.triggered.connect(lambda _checked=False: callback())
        return action

    def _create_unified_video_actions(self):
        """Create discoverable project commands without constructing video Qt."""

        self.actionOpen_video_project_manifest = self._new_workbench_action(
            "actionOpen_video_project_manifest",
            "打开已有 IVAS 视频项目…",
            self.open_video_project_manifest,
            "打开 project.ivas.json，兼容 project.ivat.json。",
            True,
        )
        self.actionVideo_output_root = self._new_workbench_action(
            "actionVideo_output_root",
            "设置视频项目输出目录…",
            lambda: self._invoke_video_command(
                "choose_output_root_dialog", require_project=False
            ),
            "设置以后新建视频项目的输出目录。",
            True,
        )
        self.actionVideo_relink = self._new_workbench_action(
            "actionVideo_relink",
            "重新关联视频…",
            lambda: self._invoke_video_command("relink_media_dialog"),
            "为当前视频项目重新关联媒体文件。",
        )
        self.actionAnnotate_current_video_frame = self._new_workbench_action(
            "actionAnnotate_current_video_frame",
            "标注当前视频帧",
            self.request_frame_mode,
            "提取当前时间对应帧，并在同一视频 Case 中打开 IVAS 分割画布。",
        )
        self.actionReturn_to_video = self._new_workbench_action(
            "actionReturn_to_video",
            "返回视频预览",
            self.show_video_workspace,
            "返回当前视频 Case 的播放器；时序状态不会丢失。",
        )
        self.actionTemporal_mark_start = self._new_workbench_action(
            "actionTemporal_mark_start",
            "记录区间开始",
            lambda: self._invoke_video_command("mark_start"),
            "在当前视频时间记录 Phase/Task 区间开始点。",
        )
        self.actionTemporal_mark_end = self._new_workbench_action(
            "actionTemporal_mark_end",
            "记录区间结束",
            lambda: self._invoke_video_command("mark_end"),
            "在当前视频时间结束正在记录的 Phase/Task 区间。",
        )
        self.actionTemporal_discard_pending = self._new_workbench_action(
            "actionTemporal_discard_pending",
            "放弃未完成区间",
            lambda: self._invoke_video_workspace_action(
                "discard_pending_interval_action"
            ),
            "清除已记录开始点、但尚未记录结束点的区间。",
        )
        self.actionTemporal_add_point = self._new_workbench_action(
            "actionTemporal_add_point",
            "新增关键帧 / 点…",
            lambda: self._invoke_video_command("add_temporal_dialog", "point"),
            "在当前时间新增关键帧或点标注。",
        )
        self.actionTemporal_capture_keyframe = self._new_workbench_action(
            "actionTemporal_capture_keyframe",
            "截取当前帧为关键帧",
            lambda: self._invoke_video_workspace_action(
                "capture_keyframe_action"
            ),
            "提取当前位置的精确帧，并保存带图像链接的关键帧标注。",
        )
        self.actionTemporal_add_rating = self._new_workbench_action(
            "actionTemporal_add_rating",
            "新增评分…",
            lambda: self._invoke_video_command("add_temporal_dialog", "rating"),
            "为当前视频项目新增或更新评分。",
        )
        self.actionTemporal_templates = self._new_workbench_action(
            "actionTemporal_templates",
            "管理时序模板…",
            lambda: self._invoke_video_command("manage_templates"),
            "管理 Phase、Task、关键帧和评分模板。",
        )
        self.actionTemporal_import_templates = self._new_workbench_action(
            "actionTemporal_import_templates",
            "导入时序模板 TXT…",
            lambda: self._invoke_video_command("import_template_dialog"),
            "从 TXT 导入时序标签模板。",
        )
        self.actionTemporal_edit = self._new_workbench_action(
            "actionTemporal_edit",
            "编辑所选时序标注…",
            lambda: self._invoke_video_command("edit_selected_annotation"),
            "编辑当前选中的 Phase、Task、关键帧或评分。",
        )
        self.actionTemporal_delete = self._new_workbench_action(
            "actionTemporal_delete",
            "删除所选时序标注",
            lambda: self._invoke_video_command("delete_selected_annotation"),
            "删除当前选中的时序标注。",
        )
        self.actionTemporal_undo = self._new_workbench_action(
            "actionTemporal_undo",
            "撤销时序修改",
            lambda: self._invoke_video_command("undo"),
            "撤销当前视频项目的最近一次时序修改。",
        )
        self.actionTemporal_redo = self._new_workbench_action(
            "actionTemporal_redo",
            "重做时序修改",
            lambda: self._invoke_video_command("redo"),
            "重做当前视频项目的时序修改。",
        )
        self.actionVideo_seek_backward_5s = self._new_workbench_action(
            "actionVideo_seek_backward_5s",
            "后退 5 秒",
            lambda: self._invoke_video_workspace_action("seek_backward_5s_action"),
            "将当前视频播放位置后退 5 秒。",
        )
        self.actionVideo_seek_backward_1s = self._new_workbench_action(
            "actionVideo_seek_backward_1s",
            "后退 1 秒",
            lambda: self._invoke_video_workspace_action("seek_backward_1s_action"),
            "将当前视频播放位置后退 1 秒。",
        )
        self.actionVideo_seek_forward_1s = self._new_workbench_action(
            "actionVideo_seek_forward_1s",
            "前进 1 秒",
            lambda: self._invoke_video_workspace_action("seek_forward_1s_action"),
            "将当前视频播放位置前进 1 秒。",
        )
        self.actionVideo_seek_forward_5s = self._new_workbench_action(
            "actionVideo_seek_forward_5s",
            "前进 5 秒",
            lambda: self._invoke_video_workspace_action("seek_forward_5s_action"),
            "将当前视频播放位置前进 5 秒。",
        )
        self.actionTimeline_zoom_in = self._new_workbench_action(
            "actionTimeline_zoom_in",
            "时间轴放大",
            lambda: self._invoke_video_workspace_action("timeline_zoom_in_action"),
            "以当前播放位置为中心放大视频时间轴。",
        )
        self.actionTimeline_zoom_out = self._new_workbench_action(
            "actionTimeline_zoom_out",
            "时间轴缩小",
            lambda: self._invoke_video_workspace_action("timeline_zoom_out_action"),
            "以当前播放位置为中心缩小视频时间轴。",
        )
        self.actionTimeline_reset_zoom = self._new_workbench_action(
            "actionTimeline_reset_zoom",
            "重置时间轴缩放",
            lambda: self._invoke_video_workspace_action("timeline_reset_zoom_action"),
            "显示当前视频的完整时间范围。",
        )
        self.actionTemporal_validate = self._new_workbench_action(
            "actionTemporal_validate",
            "校验与阶段覆盖率…",
            lambda: self._invoke_video_command("show_validation_report"),
            "校验时序标注并查看 Phase 覆盖率。",
        )
        self.actionTemporal_import = self._new_workbench_action(
            "actionTemporal_import",
            "导入时序标注 CSV / TXT…",
            lambda: self._invoke_video_command("import_annotations_dialog"),
            "向当前视频项目导入时序标注。",
        )
        self.actionTemporal_export = self._new_workbench_action(
            "actionTemporal_export",
            "导出时序标注 CSV / TXT…",
            lambda: self._invoke_video_command("export_annotations_dialog"),
            "导出当前视频项目的时序标注。",
        )
        self.actionVideo_rebuild_index = self._new_workbench_action(
            "actionVideo_rebuild_index",
            "重建精确帧时间索引",
            lambda: self._invoke_video_command("_start_frame_map_scan", True),
            "重新扫描视频的精确 PTS 帧时间索引。",
        )
        self.actionVideo_cancel_task = self._new_workbench_action(
            "actionVideo_cancel_task",
            "取消视频后台任务",
            lambda: self._invoke_video_command("cancel_background_work"),
            "取消正在运行的帧索引、抽帧或追踪任务。",
        )
        self.actionTracking_start = self._new_workbench_action(
            "actionTracking_start",
            "开始事务化追踪…",
            lambda: self._invoke_video_command("start_tracking_dialog"),
            "从已保存的分割种子开始视频追踪，结果先进入暂存区。",
        )
        self.actionTracking_review = self._new_workbench_action(
            "actionTracking_review",
            "审阅 / 提交暂存追踪…",
            lambda: self._invoke_video_command("review_staged_tracking"),
            "逐帧审阅暂存遮罩后提交或丢弃。",
        )
        self.actionTracking_rollback = self._new_workbench_action(
            "actionTracking_rollback",
            "回滚最近追踪提交",
            lambda: self._invoke_video_command("rollback_latest_tracking"),
            "回滚当前视频项目最近一次已提交追踪。",
        )
        self._video_context_actions = (
            self.actionVideo_relink,
            self.actionAnnotate_current_video_frame,
            self.actionReturn_to_video,
            self.actionTemporal_mark_start,
            self.actionTemporal_mark_end,
            self.actionTemporal_discard_pending,
            self.actionTemporal_add_point,
            self.actionTemporal_capture_keyframe,
            self.actionTemporal_add_rating,
            self.actionTemporal_templates,
            self.actionTemporal_import_templates,
            self.actionTemporal_edit,
            self.actionTemporal_delete,
            self.actionTemporal_undo,
            self.actionTemporal_redo,
            self.actionVideo_seek_backward_5s,
            self.actionVideo_seek_backward_1s,
            self.actionVideo_seek_forward_1s,
            self.actionVideo_seek_forward_5s,
            self.actionTimeline_zoom_in,
            self.actionTimeline_zoom_out,
            self.actionTimeline_reset_zoom,
            self.actionTemporal_validate,
            self.actionTemporal_import,
            self.actionTemporal_export,
            self.actionVideo_rebuild_index,
            self.actionVideo_cancel_task,
            self.actionTracking_start,
            self.actionTracking_review,
            self.actionTracking_rollback,
        )

        self.temporal_lane_combo = QtWidgets.QComboBox(self)
        self.temporal_lane_combo.setObjectName("temporalLaneCombo")
        self.temporal_lane_combo.setAccessibleName("时序标注轨道")
        self.temporal_lane_combo.addItem("Phase", "phase")
        self.temporal_lane_combo.addItem("Task", "task")
        self.temporal_lane_combo.setEnabled(False)
        self.temporal_lane_combo.setMinimumWidth(88)
        self.temporal_lane_combo.currentIndexChanged.connect(
            self._set_video_temporal_lane
        )
        self.temporal_label_combo = QtWidgets.QComboBox(self)
        self.temporal_label_combo.setObjectName("temporalLabelCombo")
        self.temporal_label_combo.setAccessibleName("时序标注标签")
        self.temporal_label_combo.setEditable(True)
        self.temporal_label_combo.setEnabled(False)
        self.temporal_label_combo.setMinimumWidth(150)
        self.temporal_label_combo.setMaximumWidth(230)
        self.temporal_label_combo.currentTextChanged.connect(
            self._set_video_temporal_label
        )

    def _configure_unified_menus(self):
        """Expose image, temporal and tracking capabilities as peers."""

        self.menuNavigate = QtWidgets.QMenu(self.menubar)
        self.menuNavigate.setObjectName("menuNavigate")
        self.menuTemporal = QtWidgets.QMenu(self.menubar)
        self.menuTemporal.setObjectName("menuTemporal")
        self.menuTracking = QtWidgets.QMenu(self.menubar)
        self.menuTracking.setObjectName("menuTracking")
        self.menuTemporalSeek = QtWidgets.QMenu("播放定位", self.menuTemporal)
        self.menuTemporalSeek.setObjectName("menuTemporalSeek")
        for action in (
            self.actionVideo_seek_backward_5s,
            self.actionVideo_seek_backward_1s,
            self.actionVideo_seek_forward_1s,
            self.actionVideo_seek_forward_5s,
        ):
            self.menuTemporalSeek.addAction(action)
        self.menuTimelineZoom = QtWidgets.QMenu("时间轴缩放", self.menuTemporal)
        self.menuTimelineZoom.setObjectName("menuTimelineZoom")
        for action in (
            self.actionTimeline_zoom_in,
            self.actionTimeline_zoom_out,
            self.actionTimeline_reset_zoom,
        ):
            self.menuTimelineZoom.addAction(action)

        self.menuTemporalImport = QtWidgets.QMenu(self.menuFile)
        self.menuTemporalImport.setObjectName("menuTemporalImport")
        self.menuTemporalImport.addAction(self.actionTemporal_import)
        self.menuTemporalImport.addAction(self.actionTemporal_import_templates)
        self.menuTemporalExport = QtWidgets.QMenu(self.menuFile)
        self.menuTemporalExport.setObjectName("menuTemporalExport")
        self.menuTemporalExport.addAction(self.actionTemporal_export)

        self.menuFile.clear()
        self.menuFile.addAction(self.actionImages_dir)
        self.menuFile.addAction(self.actionOpen_video_workspace)
        self.menuFile.addAction(self.actionOpen_video_project_manifest)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addMenu(self.menuTemporalImport)
        self.menuFile.addMenu(self.menuTemporalExport)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSetting)
        self.menuFile.addAction(self.actionExit)

        self.menuNavigate.addAction(self.actionPrev_image)
        self.menuNavigate.addAction(self.actionNext_image)
        self.menuNavigate.addSeparator()
        self.menuNavigate.addAction(self.actionPrev_group)
        self.menuNavigate.addAction(self.actionNext_group)

        self.menuEdit.clear()
        for action in (
            self.actionSegment_anything_point,
            self.actionSegment_anything_box,
            self.actionPolygon,
            self.actionRepaint,
        ):
            self.menuEdit.addAction(action)
        self.menuEdit.addSeparator()
        for action in (self.actionBackspace, self.actionFinish, self.actionCancel):
            self.menuEdit.addAction(action)
        self.menuEdit.addSeparator()
        for action in (
            self.actionEdit,
            self.actionCopy,
            self.actionDelete,
            self.actionSave,
            self.actionTo_top,
            self.actionTo_bottom,
        ):
            self.menuEdit.addAction(action)
        self.menuEdit.addSeparator()
        for action in (
            self.actionUnion,
            self.actionSubtract,
            self.actionIntersect,
            self.actionExclude,
        ):
            self.menuEdit.addAction(action)

        for action in (
            self.actionTemporal_mark_start,
            self.actionTemporal_mark_end,
            self.actionTemporal_discard_pending,
            self.actionAnnotate_current_video_frame,
            self.actionReturn_to_video,
            None,
            self.actionTemporal_add_point,
            self.actionTemporal_capture_keyframe,
            self.actionTemporal_add_rating,
            None,
            self.menuTemporalSeek.menuAction(),
            self.menuTimelineZoom.menuAction(),
            None,
            self.actionTemporal_templates,
            self.actionTemporal_import_templates,
            None,
            self.actionTemporal_edit,
            self.actionTemporal_delete,
            self.actionTemporal_undo,
            self.actionTemporal_redo,
            None,
            self.actionTemporal_validate,
            self.actionTemporal_import,
            self.actionTemporal_export,
            self.actionVideo_rebuild_index,
            self.actionVideo_cancel_task,
        ):
            self.menuTemporal.addSeparator() if action is None else self.menuTemporal.addAction(action)

        self.menuTracking.addAction(self.actionTracking_start)
        self.menuTracking.addAction(self.actionTracking_review)
        self.menuTracking.addAction(self.actionTracking_rollback)

        self.menuView.clear()
        self.menuDisplay = self.menuView.addMenu("显示")
        self.menuDisplay.addAction(self.actionVisible)
        self.menuDisplay.addAction(self.actionBit_map)
        self.menuZoom = self.menuView.addMenu("缩放")
        self.menuZoom.addAction(self.actionZoom_in)
        self.menuZoom.addAction(self.actionZoom_out)
        self.menuZoom.addAction(self.actionFit_window)
        self.menuCapture = self.menuView.addMenu("截图")
        self.menuCapture.addAction(self.actionScene_shot)
        self.menuCapture.addAction(self.actionWindow_shot)

        self.menuSAM_model.clear()
        self.menuSAM_model.addAction(self.actionModel_manage)
        self.menuSAM_model.addSeparator()
        self.menuSAM_model.addAction(self.actionRemote_SAM)

        self.menuTools.clear()
        for action in (
            self.actionConverter,
            self.actionVideo_to_frames,
            self.actionAuto_segment_with_bounding_box,
            self.actionAnno_validator,
            self.actionProcess_EXIF_tag,
        ):
            self.menuTools.addAction(action)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionPlugins)

        self.menuAbout.clear()
        for action in (
            self.actionLanguage,
            self.actionShortcut,
            self.actionDocs,
            self.actionCreate_a_desktop_shortcut,
        ):
            self.menuAbout.addAction(action)
        self.menuAbout.addSeparator()
        self.menuAbout.addAction(self.actionAbout)

        self.menubar.clear()
        for menu in (
            self.menuFile,
            self.menuNavigate,
            self.menuEdit,
            self.menuTemporal,
            self.menuTracking,
            self.menuView,
            self.menuSAM_model,
            self.menuTools,
            self.menuAbout,
        ):
            self.menubar.addAction(menu.menuAction())

    def _create_temporal_project_docks(self):
        self.temporal_dock = QtWidgets.QDockWidget("时序标注", self)
        self.temporal_dock.setObjectName("temporalAnnotationDock")
        self.temporal_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        temporal_placeholder = QtWidgets.QWidget(self.temporal_dock)
        placeholder_layout = QtWidgets.QVBoxLayout(temporal_placeholder)
        placeholder_layout.setContentsMargins(22, 22, 22, 22)
        placeholder_layout.addStretch(1)
        self.temporal_placeholder_title = QtWidgets.QLabel(
            "时序标注与视频追踪", temporal_placeholder
        )
        self.temporal_placeholder_title.setAlignment(QtCore.Qt.AlignCenter)
        self.temporal_placeholder_title.setStyleSheet("font-weight: 600;")
        placeholder_layout.addWidget(self.temporal_placeholder_title)
        self.temporal_placeholder_text = QtWidgets.QLabel(
            "打开视频或已有 IVAS 视频项目后，可在同一 Case 中编辑 "
            "Phase / Task、标注视频帧并执行事务式追踪。",
            temporal_placeholder,
        )
        self.temporal_placeholder_text.setWordWrap(True)
        self.temporal_placeholder_text.setAlignment(QtCore.Qt.AlignCenter)
        placeholder_layout.addWidget(self.temporal_placeholder_text)
        self.temporal_placeholder_open_video = QtWidgets.QPushButton(
            "打开视频…", temporal_placeholder
        )
        self.temporal_placeholder_open_video.setObjectName("primaryAction")
        self.temporal_placeholder_open_video.clicked.connect(
            self.actionOpen_video_workspace.trigger
        )
        placeholder_layout.addWidget(self.temporal_placeholder_open_video)
        self.temporal_placeholder_open_project = QtWidgets.QPushButton(
            "打开已有视频项目…", temporal_placeholder
        )
        self.temporal_placeholder_open_project.clicked.connect(
            self.actionOpen_video_project_manifest.trigger
        )
        placeholder_layout.addWidget(self.temporal_placeholder_open_project)
        placeholder_layout.addStretch(1)
        self.temporal_dock.setWidget(temporal_placeholder)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.temporal_dock)

        self.timeline_dock = QtWidgets.QDockWidget("视频时间轴", self)
        self.timeline_dock.setObjectName("videoTimelineDock")
        self.timeline_dock.setAllowedAreas(
            QtCore.Qt.TopDockWidgetArea | QtCore.Qt.BottomDockWidgetArea
        )
        timeline_placeholder = QtWidgets.QLabel(
            "视频时间轴会在打开视频 Case 后显示。", self.timeline_dock
        )
        timeline_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.timeline_dock.setWidget(timeline_placeholder)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.timeline_dock)
        self.timeline_dock.hide()

    def _create_shared_video_transport_toolbar(self):
        """Create the contextual host for the video workspace's one transport."""

        self.video_transport_toolbar = QtWidgets.QToolBar("视频播放控制", self)
        self.video_transport_toolbar.setObjectName("videoTransportToolBar")
        self.video_transport_toolbar.setAccessibleName("共享视频播放控制")
        self.video_transport_toolbar.setMovable(False)
        self.video_transport_toolbar.setFloatable(False)
        self.video_transport_toolbar.setAllowedAreas(
            QtCore.Qt.TopToolBarArea | QtCore.Qt.BottomToolBarArea
        )
        self.video_transport_widget_action = None
        self.addToolBar(QtCore.Qt.BottomToolBarArea, self.video_transport_toolbar)
        self.video_transport_toolbar.hide()

    def _set_shared_video_transport_visible(self, visible):
        """Show the hosted transport at its real two-row size, or collapse it."""

        visible = bool(visible)
        action = self.video_transport_widget_action
        if visible and action is not None:
            action.setVisible(True)
            transport = self.video_transport_toolbar.widgetForAction(action)
            if transport is not None:
                transport.setVisible(True)
                self.video_transport_toolbar.setMinimumHeight(
                    transport.layout().sizeHint().height() + 10
                )
            self.video_transport_toolbar.setVisible(True)
            self.video_transport_toolbar.updateGeometry()
            return
        self.video_transport_toolbar.setVisible(False)

    def _update_transport_height(self):
        workspace = self.video_annotation_workspace
        if workspace is not None:
            self.video_transport_toolbar.setMinimumHeight(
                workspace.transport_widget.layout().sizeHint().height() + 10
            )
            self.video_transport_toolbar.updateGeometry()

    @staticmethod
    def _video_shortcut_text_input_has_focus():
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

    def _bind_unified_video_shortcuts(self):
        """Bind project transport/temporal keys at the application-shell level."""

        bindings = (
            ("Space", "toggle_playback"),
            ("Left", "step_frames", -1),
            ("Right", "step_frames", 1),
            ("I", "mark_start"),
            ("O", "mark_end"),
            ("Ctrl+Z", "undo"),
            ("Ctrl+Shift+Z", "redo"),
        )
        self._video_application_shortcuts = []
        for index, binding in enumerate(bindings):
            sequence, method_name, *arguments = binding
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setObjectName("unifiedVideoShortcut{}".format(index))
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            shortcut.activated.connect(
                lambda method_name=method_name, arguments=tuple(arguments):
                self._activate_unified_video_shortcut(method_name, *arguments)
            )
            shortcut.setEnabled(False)
            self._video_application_shortcuts.append(shortcut)
        self._suspended_bitmap_shortcut = None
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.focusChanged.connect(self._update_unified_video_shortcuts)

    def _video_shortcut_context_active(self):
        workspace = self.video_annotation_workspace
        if workspace is None or workspace.store is None:
            return False
        if self.workspace_stack.currentWidget() is workspace:
            return True
        if not self._video_frame_belongs_to_active_project():
            return False
        focused = QtWidgets.QApplication.focusWidget()
        return any(
            focused is panel or (focused is not None and panel.isAncestorOf(focused))
            for panel in (self.temporal_dock, self.timeline_dock,
                          self.video_transport_toolbar, self.toolBar_2)
        )

    def _video_shortcut_window_is_workbench(self, window):
        if window in (None, self):
            return True
        return bool(
            isinstance(window, (QtWidgets.QDockWidget, QtWidgets.QToolBar))
            and window.parentWidget() is self
        )

    def _update_unified_video_shortcuts(self, *_args):
        active = self._video_shortcut_context_active()
        focused = QtWidgets.QApplication.focusWidget()
        focus_window = focused.window() if focused is not None else None
        usable = bool(
            active
            and self._video_shortcut_window_is_workbench(focus_window)
            and not self._video_shortcut_text_input_has_focus()
        )
        for shortcut in getattr(self, "_video_application_shortcuts", ()):
            shortcut.setEnabled(usable)

        # Space is also the standalone image bitmap shortcut.  Suspend only
        # that key binding while a video Case owns transport, leaving the
        # bitmap action itself enabled on an extracted frame.
        if active:
            if self._suspended_bitmap_shortcut is None:
                self._suspended_bitmap_shortcut = self.actionBit_map.shortcut()
            self.actionBit_map.setShortcut(QtGui.QKeySequence())
        elif self._suspended_bitmap_shortcut is not None:
            self.actionBit_map.setShortcut(self._suspended_bitmap_shortcut)
            self._suspended_bitmap_shortcut = None

    def _activate_unified_video_shortcut(self, method_name, *arguments):
        if (
            not self._video_shortcut_context_active()
            or self._video_shortcut_text_input_has_focus()
            or not self._video_shortcut_window_is_workbench(
                QtWidgets.QApplication.activeWindow()
            )
        ):
            return
        self._invoke_video_command(method_name, *arguments)

    def _make_tool_menu(self, text, actions, object_name=""):
        button = QtWidgets.QToolButton(self)
        button.setText(text)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setFocusPolicy(QtCore.Qt.StrongFocus)
        if object_name:
            button.setObjectName(object_name)
        menu = QtWidgets.QMenu(button)
        for action in actions:
            if action is None:
                menu.addSeparator()
            else:
                menu.addAction(action)
        button.setMenu(menu)
        return button

    def _make_action_button(self, action, object_name=""):
        button = QtWidgets.QToolButton(self)
        button.setDefaultAction(action)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        button.setFocusPolicy(QtCore.Qt.StrongFocus)
        if object_name:
            button.setObjectName(object_name)
        return button

    def _configure_compact_toolbars(self):
        """Expose one case workflow, grouped by semantic separators."""

        self.toolBar.clear()
        self.toolBar_2.clear()
        self.toolBar.setMovable(False)
        self.toolBar_2.setMovable(False)

        self.mode_tools_button = self._make_tool_menu(
            "Mode: Frame",
            [self.actionShow_video_workspace, self.actionShow_image_workspace],
            "modeToolMenu",
        )
        self.mode_tools_button.setAccessibleName("Mode: Frame / Video")
        self.toolBar.addWidget(self.mode_tools_button)
        self.project_tools_button = self._make_tool_menu(
            "项目",
            [
                self.actionLabel_dir,
                self.actionVideo_output_root,
                self.actionVideo_relink,
            ],
            "primaryToolMenu",
        )
        self.project_tools_button.setAccessibleName("项目目录与媒体关联")
        self.menuProjectLocations = self.project_tools_button.menu()
        self.menuProjectLocations.setObjectName("menuProjectLocations")
        self.workspace_tools_button = self.mode_tools_button
        self.toolBar.addWidget(self.project_tools_button)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionPrev_image)
        self.toolBar.addAction(self.actionNext_image)
        self.annotation_tools_button = self._make_tool_menu(
            "帧标注",
            [
                self.actionSegment_anything_point,
                self.actionSegment_anything_box,
                self.actionPolygon,
                self.actionRepaint,
            ],
            "primaryToolMenu",
        )
        self.annotation_tools_button.setAccessibleName("标注工具菜单")
        self.toolBar.addWidget(self.annotation_tools_button)
        self.toolBar.addAction(self.actionFinish)
        self.toolBar.addAction(self.actionCancel)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()

        self.annotate_video_frame_button = self._make_action_button(
            self.actionAnnotate_current_video_frame, "annotateVideoFrameButton"
        )
        self.return_to_video_button = self._make_action_button(
            self.actionReturn_to_video, "returnToVideoButton"
        )
        self.annotate_video_frame_widget_action = self.toolBar.addWidget(
            self.annotate_video_frame_button
        )
        self.return_to_video_widget_action = self.toolBar.addWidget(
            self.return_to_video_button
        )
        self.toolBar.addAction(self.actionFit_window)

        # Retain the retired direct-write tracking menu object for plugin/API
        # compatibility, but never add it to a visible toolbar.
        self.tracking_tools_button = self._make_tool_menu(
            "旧追踪",
            [
                self.actionVideo_segment_once,
                self.actionVideo_segment_five_times,
                self.actionVideo_segment,
            ],
        )
        self.tracking_tools_button.setAccessibleName("视频追踪范围菜单")
        self.tracking_tools_button.hide()

        self.image_more_button = self._make_tool_menu(
            "帧工具",
            [
                self.actionBackspace,
                self.actionEdit,
                self.actionCopy,
                self.actionDelete,
                None,
                self.actionTo_top,
                self.actionTo_bottom,
                None,
                self.actionUnion,
                self.actionSubtract,
                self.actionIntersect,
                self.actionExclude,
                None,
                self.actionZoom_in,
                self.actionZoom_out,
                self.actionBit_map,
                self.actionVisible,
                self.actionScene_shot,
                self.actionWindow_shot,
            ],
        )
        saturation_action = QtWidgets.QWidgetAction(self.image_more_button.menu())
        saturation_panel = QtWidgets.QWidget(self.image_more_button.menu())
        saturation_layout = QtWidgets.QHBoxLayout(saturation_panel)
        saturation_layout.setContentsMargins(10, 4, 10, 4)
        self.saturation_menu_label = QtWidgets.QLabel("饱和度", saturation_panel)
        saturation_layout.addWidget(self.saturation_menu_label)
        saturation_layout.addWidget(self.image_saturation, 1)
        saturation_action.setDefaultWidget(saturation_panel)
        self.image_more_button.menu().addSeparator()
        self.image_more_button.menu().addAction(saturation_action)
        self.toolBar.addWidget(self.image_more_button)

        self.temporal_toolbar_label = QtWidgets.QLabel("时序：", self)
        self.temporal_toolbar_label.setObjectName("temporalToolbarLabel")
        self.temporal_toolbar_label.setAccessibleName("Phase 和 Task 时序标注")
        self.toolBar_2.addWidget(self.temporal_toolbar_label)
        self.toolBar_2.addWidget(self.temporal_lane_combo)
        self.toolBar_2.addWidget(self.temporal_label_combo)
        self.toolBar_2.addAction(self.actionTemporal_mark_start)
        self.toolBar_2.addAction(self.actionTemporal_mark_end)
        self.toolBar_2.addSeparator()

        self.temporal_tools_button = self._make_tool_menu(
            "时序工具",
            [
                self.actionTemporal_add_point,
                self.actionTemporal_capture_keyframe,
                self.actionTemporal_add_rating,
                None,
                self.actionVideo_seek_backward_5s,
                self.actionVideo_seek_backward_1s,
                self.actionVideo_seek_forward_1s,
                self.actionVideo_seek_forward_5s,
                None,
                self.actionTimeline_zoom_in,
                self.actionTimeline_zoom_out,
                self.actionTimeline_reset_zoom,
                None,
                self.actionTemporal_templates,
                self.actionTemporal_import_templates,
                None,
                self.actionTemporal_edit,
                self.actionTemporal_delete,
                self.actionTemporal_discard_pending,
                self.actionTemporal_undo,
                self.actionTemporal_redo,
                None,
                self.actionTemporal_validate,
                self.actionTemporal_import,
                self.actionTemporal_export,
                self.actionVideo_rebuild_index,
                self.actionVideo_cancel_task,
            ],
            "primaryToolMenu",
        )
        self.temporal_tools_button.setAccessibleName("时序标注完整功能菜单")
        self.toolBar_2.addWidget(self.temporal_tools_button)
        self.transaction_tracking_button = self._make_tool_menu(
            "事务追踪",
            [
                self.actionTracking_start,
                self.actionTracking_review,
                self.actionTracking_rollback,
            ],
            "primaryToolMenu",
        )
        self.transaction_tracking_button.setAccessibleName("事务化视频追踪菜单")
        self.toolBar_2.addWidget(self.transaction_tracking_button)
        self.toolBar_2.addSeparator()
        self.toolBar_2.addAction(self.actionModel_manage)
        self.application_more_button = self._make_tool_menu(
            "应用",
            [
                self.actionRemote_SAM,
                self.actionPlugins,
                self.actionSetting,
                self.actionShortcut,
                self.actionLanguage,
                None,
                self.actionDocs,
                self.actionAbout,
            ],
        )
        self.application_more_button.setAccessibleName("应用菜单")
        self.toolBar_2.addWidget(self.application_more_button)
        self.toolBar_2.setMaximumWidth(16777215)

        self._image_workspace_actions = tuple(
            action
            for action in (
                self.actionLabel_dir,
                self.actionPrev_image,
                self.actionNext_image,
                self.actionSegment_anything_point,
                self.actionSegment_anything_box,
                self.actionPolygon,
                self.actionRepaint,
                self.actionVideo_segment,
                self.actionVideo_segment_once,
                self.actionVideo_segment_five_times,
                self.actionBackspace,
                self.actionFinish,
                self.actionCancel,
                self.actionEdit,
                self.actionCopy,
                self.actionSave,
                self.actionDelete,
                self.actionTo_top,
                self.actionTo_bottom,
                self.actionUnion,
                self.actionSubtract,
                self.actionIntersect,
                self.actionExclude,
                self.actionZoom_in,
                self.actionZoom_out,
                self.actionFit_window,
                self.actionBit_map,
                self.actionVisible,
                self.actionScene_shot,
                self.actionPrev_group,
                self.actionNext_group,
            )
        )
        self._image_action_state = None

    def _configure_canvas_first_layout(self):
        """Keep the canvas dominant while retaining every inspector on demand."""

        self._inspector_docks = (
            self.categories_dock,
            self.annos_dock,
            self.files_dock,
            self.info_dock,
            self.text_prompt_dock,
            self.visual_prompt_dock,
        )
        self._default_inspector_docks = frozenset(
            (self.categories_dock, self.annos_dock, self.files_dock)
        )
        for dock in self._inspector_docks:
            content = dock.widget()
            scroll = scrollable_content(content, dock)
            dock.setWidget(scroll)

        self.panels_menu = QtWidgets.QMenu("Panels", self.menuView)
        self.panels_menu.setObjectName("menuPanels")
        for dock in self._inspector_docks:
            self.panels_menu.addAction(dock.toggleViewAction())
        self.panels_menu.addAction(self.temporal_dock.toggleViewAction())
        self.panels_menu.addAction(self.timeline_dock.toggleViewAction())
        self.panels_menu.addSeparator()
        self.panels_menu.addAction(self.toolBar.toggleViewAction())
        self.panels_menu.addAction(self.toolBar_2.toggleViewAction())
        self.panels_menu.addAction(self.video_transport_toolbar.toggleViewAction())

        self.actionReset_layout = QtWidgets.QAction("Reset layout", self)
        self.actionReset_layout.setObjectName("actionReset_layout")
        self.actionReset_layout.triggered.connect(self._reset_workspace_layout)

        view_anchor = self.menuView.actions()[0] if self.menuView.actions() else None
        if view_anchor is None:
            self.menuView.addMenu(self.panels_menu)
            self.menuView.addAction(self.actionReset_layout)
            self.menuView.addSeparator()
        else:
            self.menuView.insertMenu(view_anchor, self.panels_menu)
            self.menuView.insertAction(view_anchor, self.actionReset_layout)

        # These actions remain alive for plugin/API compatibility, but the old
        # direct-write tracking workflow must not appear in an active menu.
        for action in (
            self.actionVideo_segment,
            self.actionVideo_segment_once,
            self.actionVideo_segment_five_times,
        ):
            self.menuEdit.removeAction(action)
        self._remove_redundant_menu_separators(self.menuEdit)
        self._reset_workspace_layout()

    @staticmethod
    def _remove_redundant_menu_separators(menu):
        previous_was_separator = True
        for action in tuple(menu.actions()):
            if action.isSeparator() and previous_was_separator:
                menu.removeAction(action)
                continue
            previous_was_separator = action.isSeparator()
        actions = menu.actions()
        if actions and actions[-1].isSeparator():
            menu.removeAction(actions[-1])

    def _reset_workspace_layout(self):
        """Restore the unified two-toolbar shell and contextual project panels."""

        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar_2)
        # Keep the semantic groups on two explicit rows.  Without a toolbar
        # break Qt packs both bars side by side, which pushes the Phase/Task
        # controls and tracking menu behind the overflow chevron on common
        # laptop widths.
        self.insertToolBarBreak(self.toolBar_2)
        self.toolBar.show()
        self.toolBar_2.setMaximumWidth(16777215)
        self.toolBar_2.show()
        self.addToolBar(
            QtCore.Qt.BottomToolBarArea, self.video_transport_toolbar
        )

        for dock in self._inspector_docks:
            dock.setFloating(False)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        for dock in self._inspector_docks[1:]:
            self.tabifyDockWidget(self.categories_dock, dock)
        self.temporal_dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.temporal_dock)
        self.tabifyDockWidget(self.categories_dock, self.temporal_dock)
        self.timeline_dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.timeline_dock)
        self.setTabPosition(
            QtCore.Qt.RightDockWidgetArea, QtWidgets.QTabWidget.North
        )

        desired_visibility = {
            dock: dock in self._default_inspector_docks
            for dock in self._inspector_docks
        }
        in_video_workspace = (
            self.video_annotation_workspace is not None
            and self.workspace_stack.currentWidget() is self.video_annotation_workspace
        )
        has_video_project = bool(
            self.video_annotation_workspace is not None
            and self.video_annotation_workspace.store is not None
        )
        video_frame_context = bool(
            self._video_frame_belongs_to_active_project()
        )
        if in_video_workspace:
            self._image_workspace_dock_visibility = desired_visibility
            for dock in self._inspector_docks:
                dock.hide()
            self.temporal_dock.show()
            self.timeline_dock.show()
            self._set_shared_video_transport_visible(True)
        else:
            self._image_workspace_dock_visibility = None
            for dock, should_show in desired_visibility.items():
                dock.setVisible(should_show)
            # Temporal work is a peer capability, not a hidden mode.  Keep its
            # tab discoverable even for a newly opened image-only window; its
            # cold-start panel provides the two project entry points.
            self.temporal_dock.show()
            self.timeline_dock.setVisible(video_frame_context)
            self._set_shared_video_transport_visible(video_frame_context)
            self.categories_dock.raise_()

        self.resizeDocks([self.categories_dock], [360], QtCore.Qt.Horizontal)
        self._configure_dock_tabs()
        if has_video_project:
            self.resizeDocks([self.timeline_dock], [132], QtCore.Qt.Vertical)

    def _configure_dock_tabs(self):
        # Qt owns these tab bars. Keep full-size labels and expose scroll arrows
        # instead of shrinking every inspector name to a few characters.
        for bar in self.findChildren(QtWidgets.QTabBar, options=QtCore.Qt.FindDirectChildrenOnly):
            if bar.objectName() != "workbenchDockTabs":
                bar.setObjectName("workbenchDockTabs")
                bar.style().unpolish(bar)
                bar.style().polish(bar)
            bar.setExpanding(False)
            bar.setUsesScrollButtons(True)
            bar.setElideMode(QtCore.Qt.ElideNone)
            for button in bar.findChildren(QtWidgets.QToolButton):
                if button.objectName() != "workbenchDockTabScrollButton":
                    button.setObjectName("workbenchDockTabScrollButton")
                    # Re-polish children too: their initial toolbar styling
                    # predates the name assigned to Qt's internal tab bar.
                    button.style().unpolish(button)
                    button.style().polish(button)
                    button.setFixedWidth(28)
            for index in range(bar.count()):
                bar.setTabToolTip(index, bar.tabText(index))

    def event(self, event):
        handled = super().event(event)
        if event.type() in (QtCore.QEvent.LayoutRequest, QtCore.QEvent.Show):
            self._configure_dock_tabs()
        return handled

    def _translate_canvas_first_layout(self, language):
        is_zh = language == "zh"
        self.panels_menu.setTitle("面板" if is_zh else "Panels")
        self.actionReset_layout.setText("重置布局" if is_zh else "Reset layout")
        self.actionReset_layout.setStatusTip(
            "恢复简洁的画布优先布局。"
            if is_zh
            else "Restore the compact canvas-first layout."
        )
        self.toolBar.setWindowTitle("标注工具栏" if is_zh else "Annotation toolbar")
        self.toolBar_2.setWindowTitle(
            "时序与项目工具栏" if is_zh else "Temporal and project toolbar"
        )
        self.video_transport_toolbar.setWindowTitle(
            "视频播放控制" if is_zh else "Video transport"
        )
        self.video_transport_toolbar.setAccessibleName(
            "共享视频播放控制" if is_zh else "Shared video transport"
        )
        self.temporal_dock.setWindowTitle("视频标注" if is_zh else "Video annotation")
        self.timeline_dock.setWindowTitle(
            "视频时间轴" if is_zh else "Video timeline"
        )
        placeholder_title = getattr(self, "temporal_placeholder_title", None)
        if placeholder_title is not None:
            placeholder_title.setText(
                "时序标注与视频追踪"
                if is_zh
                else "Video annotation and video tracking"
            )
            self.temporal_placeholder_text.setText(
                "打开视频或已有 IVAS 视频项目后，可在同一 Case 中编辑 "
                "Phase / Task、标注视频帧并执行事务式追踪。"
                if is_zh
                else "Open a video or an existing IVAS project to edit Phase / "
                "Task intervals, annotate video frames, and run transactional "
                "tracking in the same case."
            )
            self.temporal_placeholder_open_video.setText(
                "打开视频…" if is_zh else "Open video…"
            )
            self.temporal_placeholder_open_project.setText(
                "打开已有视频项目…"
                if is_zh
                else "Open existing video project…"
            )

    def _enter_video_action_scope(self):
        """Disable image-canvas shortcuts while the temporal workspace is active."""

        if self._image_action_state is None:
            self._image_action_state = {
                action: action.isEnabled() for action in self._image_workspace_actions
            }
        for action in self._image_workspace_actions:
            action.setEnabled(False)

    def _leave_video_action_scope(self):
        if self._image_action_state is None:
            return
        previous = self._image_action_state
        self._image_action_state = None
        for action, was_enabled in previous.items():
            action.setEnabled(was_enabled)

    def _apply_frame_action_constraints(self):
        """Apply display/editor restrictions after scene signals and async updates."""
        if not hasattr(self, "_image_workspace_actions"):
            return
        if self.annotation_mode != "Frame":
            for action in self._image_workspace_actions:
                if action is not self.actionSave:
                    action.setEnabled(False)
            self.annos_dock_widget.setEnabled(False)
            return
        editable = self.scene._frame_is_editable()
        self.annos_dock_widget.setEnabled(editable and self.scene.mode in (STATUSMode.VIEW, STATUSMode.EDIT))
        if not editable:
            for action in (
                self.actionPolygon, self.actionRepaint, self.actionSegment_anything_point,
                self.actionSegment_anything_box, self.actionEdit, self.actionCopy,
                self.actionDelete, self.actionTo_top, self.actionTo_bottom,
                self.actionUnion, self.actionSubtract, self.actionIntersect, self.actionExclude,
                self.actionBackspace, self.actionFinish, self.actionCancel, self.actionVisible,
            ):
                action.setEnabled(False)
            self.annos_dock_widget.setEnabled(False)
        if self.scene.image_item is None or not self.can_be_annotated:
            self.actionSave.setEnabled(False)
        index = self.current_index
        navigating = index is not None and self.scene.mode == STATUSMode.VIEW
        self.actionPrev_image.setEnabled(navigating and index > 0)
        self.actionNext_image.setEnabled(navigating and index < len(self.files_list) - 1)
        self._update_tracking_action_safety()

    def _update_tracking_action_safety(self):
        """Retire the legacy direct-write tracker in favour of staged tracking."""

        tracking_actions = (
            self.actionVideo_segment,
            self.actionVideo_segment_once,
            self.actionVideo_segment_five_times,
        )
        for action in tracking_actions:
            action.setEnabled(False)
            action.setToolTip("旧直接写入追踪已停用；请使用统一视频工作台的事务追踪。")
        if hasattr(self, "tracking_tools_button"):
            self.tracking_tools_button.setEnabled(False)
            self.tracking_tools_button.setVisible(False)

    def statusbar_change_status(self, is_message: bool = True):
        self.labelGPUResource.setVisible(is_message)
        self.labelData.setVisible(is_message)
        self.labelCoord.setVisible(is_message)
        self.modeState.setVisible(is_message)
        self.progressbar.setVisible(not is_message)

    def translate(self, language: str = "zh"):
        self._ui_language = language
        if language == "zh":
            self.trans.load(os.path.join(IVAS_ROOT, "ui/zh_CN"))
        else:
            self.trans.load(os.path.join(IVAS_ROOT, "ui/en"))
        language_icon = QtGui.QIcon()
        icon_path = (
            ":/icon/icons/中文_chinese.svg"
            if language == "en"
            else ":/icon/icons/英文_english.svg"
        )
        language_icon.addPixmap(
            QtGui.QPixmap(icon_path), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off
        )
        self.actionLanguage.setIcon(language_icon)
        language_tool_tip = "中文" if language == "en" else "English"
        self.actionLanguage.setToolTip(language_tool_tip)
        language_status_tip = (
            "切换中文界面" if language == "en" else "Switch to English"
        )
        self.actionLanguage.setStatusTip(language_status_tip)

        _app = QtWidgets.QApplication.instance()
        _app.installTranslator(self.trans)
        self.retranslateUi(self)
        self._translate_canvas_first_layout(language)
        # Retranslate all child widgets
        _widget_names = (
            "info_dock_widget", "annos_dock_widget", "files_dock_widget",
            "text_prompt_dock_widget", "visual_prompt_dock_widget",
            "category_edit_widget", "categories_dock_widget",
            "category_setting_dialog", "model_manager_dialog",
            "remote_sam_dialog", "plugin_manager_dialog",
            "about_dialog", "shortcut_dialog", "Converter_dialog",
            "video2frames_dialog", "auto_segment_dialog",
            "annos_validator_dialog", "process_exif_dialog",
            "setting_dialog",
        )
        for _name in _widget_names:
            _widget = getattr(self, _name, None)
            if _widget is not None:
                _widget.retranslateUi(_widget)

        # 手动添加翻译 ------
        _translate = QtCore.QCoreApplication.translate
        self.image_saturation.setStatusTip(
            _translate("MainWindow", "Image saturation.")
        )
        self.image_saturation.setToolTip(_translate("MainWindow", "Image saturation"))

        self.categories_dock_widget.pushButton_group_mode.setStatusTip(
            _translate("MainWindow", "Group id auto add 1 when add a new polygon.")
        )
        self.modeState.setStatusTip(
            "当前媒体视图；图像、时序和追踪命令属于同一个 Case。"
            if language == "zh"
            else "Current media view; frame, temporal and tracking commands share one case."
        )
        self.text_prompt_dock_widget.pushButton_predict_for_single_category.setText(
            "分割" if language == "zh" else "Segment"
        )
        self.text_prompt_dock_widget.pushButton_predict_for_single_category.setToolTip(
            "对当前类别执行文本引导分割。"
            if language == "zh"
            else "Run text-guided segmentation for the selected category."
        )
        is_zh = language == "zh"
        self.categories_dock_widget.pushButton_segment.setText("分割 (Segment)" if is_zh else "Segment")
        self.categories_dock_widget.pushButton_segment.setToolTip(
            "使用当前类别名直接进行文本分割。" if is_zh
            else "Segment the selected category with its category name."
        )
        self.text_prompt_dock_widget.checkBox_use_selected_category.setText(
            "跟随当前类别" if is_zh else "Use selected category"
        )
        self.text_prompt_dock_widget.lineEdit_prompt_text_for_single_category.setPlaceholderText(
            "选择类别或手动输入英文提示词。" if is_zh
            else "Choose a category or enter an English prompt."
        )
        def set_action_text(action, chinese, english):
            action.setText(chinese if is_zh else english)

        self.menuFile.setTitle("文件" if is_zh else "File")
        self.menuNavigate.setTitle("导航" if is_zh else "Navigate")
        self.menuEdit.setTitle("帧标注" if is_zh else "Frame annotation")
        self.menuTemporal.setTitle("视频标注" if is_zh else "Video annotation")
        self.menuTemporalSeek.setTitle("播放定位" if is_zh else "Playhead navigation")
        self.menuTimelineZoom.setTitle("时间轴缩放" if is_zh else "Timeline zoom")
        self.menuTracking.setTitle("事务追踪" if is_zh else "Transactional tracking")
        self.menuView.setTitle("视图" if is_zh else "View")
        self.menuProjectLocations.setTitle(
            "项目位置" if is_zh else "Project locations"
        )
        self.menuTemporalImport.setTitle("导入" if is_zh else "Import")
        self.menuTemporalExport.setTitle("导出" if is_zh else "Export")
        self.menuDisplay.setTitle("显示" if is_zh else "Display")
        self.menuZoom.setTitle("缩放" if is_zh else "Zoom")
        self.menuCapture.setTitle("截图" if is_zh else "Capture")

        set_action_text(self.actionImages_dir, "打开图像目录…", "Open image directory…")
        set_action_text(self.actionLabel_dir, "设置帧标签目录…", "Set frame label directory…")
        set_action_text(self.actionSave, "保存当前工作", "Save current work")
        self.actionSave.setStatusTip(
            "保存当前帧标注或正在预览的视频项目。"
            if is_zh
            else "Save the current frame annotation or active video project."
        )
        set_action_text(self.actionOpen_video_workspace, "打开视频…", "Open video…")
        for action, chinese, english in (
            (self.actionOpen_video_project_manifest, "打开已有 IVAS 视频项目…", "Open existing IVAS video project…"),
            (self.actionVideo_output_root, "设置视频项目输出目录…", "Set video project output directory…"),
            (self.actionVideo_relink, "重新关联视频…", "Relink video…"),
            (self.actionAnnotate_current_video_frame, "标注当前视频帧", "Annotate current video frame"),
            (self.actionReturn_to_video, "返回视频预览", "Return to video preview"),
            (self.actionTemporal_mark_start, "记录区间开始", "Mark interval start"),
            (self.actionTemporal_mark_end, "记录区间结束", "Mark interval end"),
            (self.actionTemporal_discard_pending, "放弃未完成区间", "Discard unfinished interval"),
            (self.actionTemporal_add_point, "新增关键帧 / 点…", "Add key frame / point…"),
            (self.actionTemporal_capture_keyframe, "截取当前帧为关键帧", "Capture current frame as key frame"),
            (self.actionTemporal_add_rating, "新增评分…", "Add rating…"),
            (self.actionTemporal_templates, "管理时序模板…", "Manage temporal templates…"),
            (self.actionTemporal_import_templates, "导入时序模板 TXT…", "Import temporal template TXT…"),
            (self.actionTemporal_edit, "编辑所选时序标注…", "Edit selected video annotation…"),
            (self.actionTemporal_delete, "删除所选时序标注", "Delete selected video annotation"),
            (self.actionTemporal_undo, "撤销时序修改", "Undo temporal change"),
            (self.actionTemporal_redo, "重做时序修改", "Redo temporal change"),
            (self.actionVideo_seek_backward_5s, "后退 5 秒", "Seek backward 5 seconds"),
            (self.actionVideo_seek_backward_1s, "后退 1 秒", "Seek backward 1 second"),
            (self.actionVideo_seek_forward_1s, "前进 1 秒", "Seek forward 1 second"),
            (self.actionVideo_seek_forward_5s, "前进 5 秒", "Seek forward 5 seconds"),
            (self.actionTimeline_zoom_in, "时间轴放大", "Zoom timeline in"),
            (self.actionTimeline_zoom_out, "时间轴缩小", "Zoom timeline out"),
            (self.actionTimeline_reset_zoom, "重置时间轴缩放", "Reset timeline zoom"),
            (self.actionTemporal_validate, "校验与阶段覆盖率…", "Validation and phase coverage…"),
            (self.actionTemporal_import, "导入时序标注 CSV / TXT…", "Import video annotations CSV / TXT…"),
            (self.actionTemporal_export, "导出时序标注 CSV / TXT…", "Export video annotations CSV / TXT…"),
            (self.actionVideo_rebuild_index, "重建精确帧时间索引", "Rebuild exact frame-time index"),
            (self.actionVideo_cancel_task, "取消视频后台任务", "Cancel video background task"),
            (self.actionTracking_start, "开始事务化追踪…", "Start transactional tracking…"),
            (self.actionTracking_review, "审阅 / 提交暂存追踪…", "Review / commit staged tracking…"),
            (self.actionTracking_rollback, "回滚最近追踪提交", "Roll back latest tracking commit"),
        ):
            set_action_text(action, chinese, english)
        for action, chinese, english in (
            (
                self.actionTemporal_discard_pending,
                "清除已记录开始点、但尚未记录结束点的区间。",
                "Clear the interval whose start was marked but whose end is still pending.",
            ),
            (
                self.actionTemporal_capture_keyframe,
                "提取当前位置的精确帧，并保存带图像链接的关键帧标注。",
                "Extract the exact current frame and save a key-frame annotation linked to its image.",
            ),
            (self.actionVideo_seek_backward_5s, "播放位置后退 5 秒。", "Move the playhead backward 5 seconds."),
            (self.actionVideo_seek_backward_1s, "播放位置后退 1 秒。", "Move the playhead backward 1 second."),
            (self.actionVideo_seek_forward_1s, "播放位置前进 1 秒。", "Move the playhead forward 1 second."),
            (self.actionVideo_seek_forward_5s, "播放位置前进 5 秒。", "Move the playhead forward 5 seconds."),
            (self.actionTimeline_zoom_in, "以当前播放位置为中心放大时间轴。", "Zoom the timeline around the playhead."),
            (self.actionTimeline_zoom_out, "以当前播放位置为中心缩小时间轴。", "Zoom the timeline out around the playhead."),
            (self.actionTimeline_reset_zoom, "显示完整视频时间范围。", "Show the complete video time range."),
        ):
            tip = chinese if is_zh else english
            action.setStatusTip(tip)
            action.setToolTip(tip)
        self.actionShow_image_workspace.setText(
            "Frame（帧分割）" if is_zh else "Frame"
        )
        self.actionShow_video_workspace.setText(
            "Video（视频标注）" if is_zh else "Video"
        )
        self.annotation_tools_button.setText("帧标注" if is_zh else "Frame annotate")
        self.image_more_button.setText("帧工具" if is_zh else "Frame tools")
        self.project_tools_button.setText("项目" if is_zh else "Project")
        self.temporal_toolbar_label.setText("视频：" if is_zh else "Video:")
        self.temporal_tools_button.setText("视频工具" if is_zh else "Video tools")
        self.transaction_tracking_button.setText(
            "事务追踪" if is_zh else "Tracking"
        )
        self.application_more_button.setText("应用" if is_zh else "Application")
        self.saturation_menu_label.setText("饱和度" if is_zh else "Saturation")
        in_video_workspace = (
            self.video_annotation_workspace is not None
            and self.workspace_stack.currentWidget() is self.video_annotation_workspace
        )
        video_frame_context = bool(
            self._video_frame_belongs_to_active_project()
        )
        if in_video_workspace:
            state_text = "视频预览" if is_zh else "Video preview"
        elif video_frame_context:
            state_text = "视频帧 + 时序" if is_zh else "Video frame + timeline"
        else:
            state_text = "图像" if is_zh else "Image"
        self.modeState.setText(state_text)
        if self.video_annotation_workspace is not None:
            self.video_annotation_workspace.set_language(language)
        # retranslateUi() restores generated shortcut literals; immediately
        # reapply the user's configured keys (including unified Ctrl+S).
        self.load_actions_shortcut()
        self._sync_video_context_actions()

        # -----------------

    def change_language(self):
        change_to_language = "en" if self.cfg["software"]["language"] == "zh" else "zh"
        self.translate(change_to_language)
        self.cfg["software"]["language"] = change_to_language
        self.save_software_cfg()

    def reload_cfg(self):
        """Reload configuration without treating programmatic UI updates as edits."""

        was_loading = self._loading_config
        self._loading_config = True
        try:
            self._reload_cfg_impl()
        finally:
            self._loading_config = was_loading

    def category_display_name(self, category):
        """Keep the canonical name visible and append a display-only hint."""
        hint = self.category_aliases.get(category)
        return "{} [{}]".format(category, hint) if hint is not None else category

    def _reload_cfg_impl(self):
        # 软件配置
        self.cfg = load_config(self.software_config_file)
        if self.cfg is None:
            self.cfg = {}
        software_cfg = self.cfg.get("software", {})
        self.cfg["software"] = software_cfg

        language = software_cfg.get("language", "en")
        self.cfg["software"]["language"] = language
        self.translate(language)

        auto_save = software_cfg.get("auto_save", False)
        self.cfg["software"]["auto_save"] = auto_save
        self.setting_dialog.checkBox_auto_save.setChecked(auto_save)

        real_time_area = software_cfg.get("real_time_area", False)
        self.cfg["software"]["real_time_area"] = real_time_area
        self.setting_dialog.checkBox_real_time_area.setChecked(real_time_area)

        contour_mode = software_cfg.get("contour_mode", "max_only")
        self.cfg["software"]["contour_mode"] = contour_mode
        self.change_contour_mode(contour_mode)

        contour_method = software_cfg.get("contour_method", "SIMPLE")
        self.cfg["software"]["contour_method"] = contour_method
        self.change_contour_method(contour_method)

        mask_alpha = software_cfg.get("mask_alpha", 0.5)
        self.cfg["software"]["mask_alpha"] = mask_alpha
        self.setting_dialog.horizontalSlider_mask_alpha.setValue(int(mask_alpha * 10))

        polygon_alpha_hover = software_cfg.get("polygon_alpha_hover", 0.6)
        self.cfg["software"]["polygon_alpha_hover"] = polygon_alpha_hover
        self.setting_dialog.horizontalSlider_polygon_alpha_hover.setValue(
            int(polygon_alpha_hover * 10)
        )

        polygon_alpha_no_hover = software_cfg.get("polygon_alpha_no_hover", 0.3)
        self.cfg["software"]["polygon_alpha_no_hover"] = polygon_alpha_no_hover
        self.setting_dialog.horizontalSlider_polygon_alpha_no_hover.setValue(
            int(polygon_alpha_no_hover * 10)
        )

        vertex_size = software_cfg.get("vertex_size", 1)
        self.cfg["software"]["vertex_size"] = int(vertex_size)
        self.setting_dialog.horizontalSlider_vertex_size.setValue(int(vertex_size))

        show_prompt = software_cfg.get("show_prompt", False)
        self.cfg["software"]["show_prompt"] = bool(show_prompt)
        self.setting_dialog.checkBox_show_prompt.setChecked(show_prompt)

        show_edge = software_cfg.get("show_edge", True)
        self.cfg["software"]["show_edge"] = bool(show_edge)
        self.setting_dialog.checkBox_show_edge.setChecked(show_edge)

        use_polydp = software_cfg.get("use_polydp", True)
        self.cfg["software"]["use_polydp"] = bool(use_polydp)
        self.setting_dialog.checkBox_approx_polygon.setChecked(use_polydp)

        invisible_polygon = software_cfg.get("create_mode_invisible_polygon", True)
        self.cfg["software"]["create_mode_invisible_polygon"] = bool(invisible_polygon)
        self.setting_dialog.checkBox_polygon_invisible.setChecked(invisible_polygon)

        if "sam_precision" in software_cfg:
            sam_precision = _normalize_precision_config(software_cfg["sam_precision"])
        elif "use_bfloat16" in software_cfg:
            sam_precision = _normalize_precision_config(
                use_bfloat16=software_cfg["use_bfloat16"]
            )
        else:
            sam_precision = "original"
        self.cfg["software"]["sam_precision"] = sam_precision
        # Compatibility for older plugins/config readers.
        self.cfg["software"]["use_bfloat16"] = sam_precision == "bfloat16"
        precision_index = {
            "original": 0,
            "bfloat16": 1,
            "float16": 2,
            "float32": 3,
        }[sam_precision]
        self.setting_dialog.comboBox_sam_precision.blockSignals(True)
        self.setting_dialog.comboBox_sam_precision.setCurrentIndex(
            precision_index
        )
        self.setting_dialog.comboBox_sam_precision.blockSignals(False)

        use_video_segmentation = software_cfg.get("use_video_segmentation", True)
        self.cfg["software"]["use_video_segmentation"] = bool(use_video_segmentation)
        self.setting_dialog.checkBox_use_video_segmentation.setChecked(use_video_segmentation)

        configured_video_output = str(
            software_cfg.get("video_output_root", "") or ""
        ).strip()
        self.video_output_root = (
            Path(configured_video_output).expanduser().resolve()
            if configured_video_output
            else _default_video_output_root()
        )
        self.cfg["software"]["video_output_root"] = configured_video_output

        # 类别
        self.cfg.update(load_config(self.config_file) or {})
        label_dict_list = self.cfg.get("label", [])
        if (
            len(label_dict_list) < 1
            or label_dict_list[0].get("name", "unknow") != "__background__"
        ):
            label_dict_list.insert(0, {"color": "#000000", "name": "__background__"})

        d = {}
        for label_dict in label_dict_list:
            category = label_dict.get("name", "unknow")
            color = label_dict.get("color", "#000000")
            d[category] = color
        self.category_color_dict = d
        self.cfg["label"] = label_dict_list
        self.category_aliases = load_category_aliases(label_dict_list, self.config_file)
        if self.current_category not in d:
            self.current_category = "__background__"

        self.categories_dock_widget.update_widget()
        self.text_prompt_dock_widget.refresh_categories()

        if self.current_index is not None:
            self.show_image(self.current_index)

        # shortcut
        self.load_actions_shortcut(default=False)

    def load_actions_shortcut(self, default: bool = False):
        """
        Load shortcut

        Arguments:
            default (bool): load default shortcut if True.
        """
        shortcut_cfg = self.cfg["shortcut"]
        for action in shortcut_cfg:

            default_key = shortcut_cfg[action]["default_key"]
            if default:
                shortcut_cfg[action]["key"] = default_key
            key = shortcut_cfg[action]["key"]
            # Migrate IVAS's former single-key image-only default to the
            # conventional shortcut now shared by frame and video projects.
            if action == "actionSave" and key == "S":
                key = "Ctrl+S"
                shortcut_cfg[action]["default_key"] = key
                shortcut_cfg[action]["key"] = key
            qt_action = getattr(self, action, None)
            if not isinstance(qt_action, QtWidgets.QAction):
                continue
            if key is not None:
                qt_action.setShortcut(QtGui.QKeySequence(key))
            else:
                qt_action.setShortcut(QtGui.QKeySequence(0))
            if (qt_action is self.actionBit_map
                    and getattr(self, "_suspended_bitmap_shortcut", None) is not None):
                self._suspended_bitmap_shortcut = qt_action.shortcut()
                qt_action.setShortcut(QtGui.QKeySequence())

    def set_saved_state(self, is_saved: bool):
        self.saved = is_saved
        if not is_saved:
            if self.cfg["software"]["auto_save"]:
                try:
                    self.save()
                except Exception as error:
                    QtWidgets.QMessageBox.critical(self, "Auto-save failed", str(error))
                else:
                    is_saved = True
        self.saved = is_saved
        if self.files_list is not None and self.current_index is not None:

            prefix = "" if is_saved else "*"
            self.setWindowTitle(
                "{}{} — {}".format(prefix, self.current_label.label_path, APP_NAME)
            )

        if not is_saved:
            self.plugin_manager_dialog.trigger_after_annotation_changed()

    def open_dir(self):
        """
        Open image root. Update file list and show the first image.

        .. tip::

            The label root will set same as the image root.

            If an IVAS or legacy category config exists under the label root, load it.
        """
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return

        self.load_image_directory(dir)

    def load_image_directory(
        self,
        directory: str,
        label_directory: str = None,
        selected_filename: str = None,
        video_case: bool = False,
    ) -> bool:
        """Load an image directory, optionally selecting a video-extracted frame."""
        directory = os.path.abspath(os.fspath(directory))
        if not os.path.isdir(directory):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Image directory does not exist: {}".format(directory)
            )
            return False

        suffixs = tuple(
            "{}".format(fmt.data().decode("ascii").lower())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        )
        files = sorted(
            file_name
            for file_name in os.listdir(directory)
            if file_name.lower().endswith(suffixs)
            or file_name.lower().endswith(".dcm")
        )
        if not files:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No supported images found in {}".format(directory)
            )
            return False

        if selected_filename is not None and selected_filename not in files:
            QtWidgets.QMessageBox.warning(self, "Open frame failed", "The requested frame is missing.")
            return False
        selected_path = os.path.join(directory, selected_filename or files[0])
        if not selected_path.lower().endswith(".dcm"):
            try:
                with Image.open(selected_path) as candidate:
                    candidate.verify()
            except Exception as error:
                QtWidgets.QMessageBox.warning(self, "Open image failed", str(error))
                return False

        if not self._confirm_leave_unsaved_image():
            return False
        workspace = self.video_annotation_workspace
        if not video_case and workspace is not None:
            workspace.cancel_frame_extraction()

        # Invalidate old requests without waiting for GPU work on the UI thread.
        if self.use_segment_anything:
            self.seganythread.invalidate()
            if self.use_segment_anything_video:
                self.segany_video.inference_state = {}

        self.files_list.clear()
        self.files_dock_widget.listWidget.clear()

        self.files_list = files

        self.files_dock_widget.update_widget()

        if selected_filename in files:
            self.current_index = files.index(selected_filename)
        else:
            self.current_index = 0

        self.image_root = directory
        self._active_video_frame_case = bool(video_case)
        self._video_frame_project_path = (
            str(workspace.store.path.resolve())
            if video_case and workspace is not None and workspace.store is not None
            else None
        )
        self.actionImages_dir.setStatusTip("Image root: {}".format(self.image_root))

        self.label_root = os.path.abspath(label_directory or directory)
        os.makedirs(self.label_root, exist_ok=True)
        self.actionLabel_dir.setStatusTip("Label root: {}".format(self.label_root))

        self.saved = True

        category_config = _find_category_config(self.label_root)
        if category_config is not None:
            self.config_file = category_config
            self.reload_cfg()

        self.show_image_workspace()
        self.show_image(self.current_index)
        self._update_tracking_action_safety()
        return True

    def _confirm_leave_unsaved_image(self) -> bool:
        if self.scene.mode in (STATUSMode.CREATE, STATUSMode.EDIT, STATUSMode.REPAINT):
            QtWidgets.QMessageBox.information(
                self, "Unfinished frame annotation",
                "Finish or cancel the current shape before switching media or mode.",
            )
            return False
        if self.saved or self.current_label is None:
            return True
        result = QtWidgets.QMessageBox.question(
            self,
            "Unsaved annotation",
            "Save the current image annotation before switching workspace?",
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if result == QtWidgets.QMessageBox.Cancel:
            return False
        if result == QtWidgets.QMessageBox.Save:
            try:
                self.save()
            except Exception as error:
                QtWidgets.QMessageBox.critical(
                    self, "保存失败", "标注未保存：{}".format(error)
                )
                return False
        elif result == QtWidgets.QMessageBox.Discard:
            # Restore the on-disk frame so returning from Video cannot revive
            # a discarded canvas and accidentally save it into another case.
            self.saved = True
            if self.current_index is not None:
                self.show_image(self.current_index, zoomfit=False)
        return True

    def _replace_dock_widget(self, dock, widget):
        previous = dock.widget()
        if previous is widget:
            return
        if previous is not None:
            previous.setParent(None)
            previous.deleteLater()
        dock.setWidget(widget)

    def _attach_video_project_panels(self, workspace):
        editor_panel, timeline = workspace.detach_project_panels()
        transport = workspace.detach_transport()
        workspace.set_local_shortcuts_enabled(False)
        self._replace_dock_widget(self.temporal_dock, scrollable_content(editor_panel, self.temporal_dock))
        # The cold-start call-to-action belongs to the replaced placeholder;
        # clear Python references before a later language switch tries to
        # retranslate Qt objects that have been deleted via deleteLater().
        for attribute_name in (
            "temporal_placeholder_title",
            "temporal_placeholder_text",
            "temporal_placeholder_open_video",
            "temporal_placeholder_open_project",
        ):
            setattr(self, attribute_name, None)
        self._replace_dock_widget(self.timeline_dock, timeline)
        workspace.transport_layout_changed.connect(self._update_transport_height)
        if self.video_transport_widget_action is not None:
            self.video_transport_toolbar.removeAction(
                self.video_transport_widget_action
            )
        self.video_transport_widget_action = self.video_transport_toolbar.addWidget(
            transport
        )
        self.video_transport_widget_action.setVisible(True)
        transport.setVisible(True)
        self.video_transport_toolbar.setMinimumHeight(
            transport.layout().sizeHint().height() + 10
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.temporal_dock)
        self.tabifyDockWidget(self.categories_dock, self.temporal_dock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.timeline_dock)
        self.resizeDocks([self.timeline_dock], [132], QtCore.Qt.Vertical)
        self.temporal_dock.show()
        if self.workspace_stack.currentWidget() is self.view:
            self.categories_dock.raise_()
        self.timeline_dock.hide()
        self._set_shared_video_transport_visible(False)

    def _bind_video_workspace_commands(self, workspace):
        self._video_proxy_targets = {
            self.actionVideo_relink: workspace.relink_media_action,
            self.actionTemporal_add_point: workspace.add_point_action,
            self.actionTemporal_capture_keyframe: workspace.capture_keyframe_action,
            self.actionTemporal_add_rating: workspace.add_rating_action,
            self.actionTemporal_templates: workspace.template_action,
            self.actionTemporal_import_templates: workspace.import_template_action,
            self.actionTemporal_edit: workspace.edit_annotation_action,
            self.actionTemporal_delete: workspace.delete_action,
            self.actionTemporal_discard_pending: workspace.discard_pending_interval_action,
            self.actionTemporal_undo: workspace.undo_action,
            self.actionTemporal_redo: workspace.redo_action,
            self.actionVideo_seek_backward_5s: workspace.seek_backward_5s_action,
            self.actionVideo_seek_backward_1s: workspace.seek_backward_1s_action,
            self.actionVideo_seek_forward_1s: workspace.seek_forward_1s_action,
            self.actionVideo_seek_forward_5s: workspace.seek_forward_5s_action,
            self.actionTimeline_zoom_in: workspace.timeline_zoom_in_action,
            self.actionTimeline_zoom_out: workspace.timeline_zoom_out_action,
            self.actionTimeline_reset_zoom: workspace.timeline_reset_zoom_action,
            self.actionTemporal_validate: workspace.validate_action,
            self.actionTemporal_import: workspace.import_action,
            self.actionTemporal_export: workspace.export_action,
            self.actionVideo_rebuild_index: workspace.rebuild_frame_map_action,
            self.actionVideo_cancel_task: workspace.cancel_worker_action,
            self.actionTracking_start: workspace.start_tracking_action,
            self.actionTracking_review: workspace.review_tracking_action,
            self.actionTracking_rollback: workspace.rollback_tracking_action,
        }
        for target in self._video_proxy_targets.values():
            target.changed.connect(self._sync_video_context_actions)
        workspace.save_project_action.changed.connect(
            self._sync_video_context_actions
        )
        workspace.context_changed.connect(self._sync_video_context_actions)
        workspace.category_combo.currentIndexChanged.connect(
            self._sync_video_context_actions
        )
        workspace.label_combo.currentTextChanged.connect(
            self._sync_video_context_actions
        )
        self._sync_video_context_actions()

    def _video_frame_belongs_to_active_project(self):
        workspace = self.video_annotation_workspace
        return bool(
            self._active_video_frame_case
            and workspace is not None and workspace.store is not None
            and self._video_frame_project_path == str(workspace.store.path.resolve())
        )

    @property
    def annotation_mode(self):
        """The displayed editor is the single source of truth for Mode."""
        return "Video" if (
            self.video_annotation_workspace is not None
            and self.workspace_stack.currentWidget() is self.video_annotation_workspace
        ) else "Frame"

    def _sync_mode_controls(self):
        mode = self.annotation_mode
        self.actionShow_video_workspace.setChecked(mode == "Video")
        self.actionShow_image_workspace.setChecked(mode == "Frame")
        self.mode_tools_button.setText("Mode: {}".format(mode))
        self.mode_tools_button.setAccessibleName("Mode: {}".format(mode))
        self.modeState.setText("Mode: {}".format(mode))
        # Embedded QWidgetActions otherwise shrink below their text width.
        # Let the toolbar overflow move secondary commands into its extension.
        for toolbar in (self.toolBar, self.toolBar_2):
            for action in toolbar.actions():
                button = toolbar.widgetForAction(action)
                if isinstance(button, QtWidgets.QToolButton):
                    button.setMinimumWidth(button.sizeHint().width())

    def _sync_video_context_actions(self, *_args):
        workspace = self.video_annotation_workspace
        has_project = workspace is not None and workspace.store is not None
        in_video_preview = bool(
            has_project and self.workspace_stack.currentWidget() is workspace
        )
        in_image_surface = bool(
            has_project and self.workspace_stack.currentWidget() is self.view
        )
        shared_video_context = bool(
            in_video_preview
            or (in_image_surface and self._video_frame_belongs_to_active_project())
        )
        self._set_shared_video_transport_visible(shared_video_context)
        if hasattr(self, "annotate_video_frame_button"):
            self.annotate_video_frame_widget_action.setVisible(in_video_preview)
            self.return_to_video_widget_action.setVisible(in_image_surface and shared_video_context)
            self.temporal_tools_button.setEnabled(shared_video_context)
            self.transaction_tracking_button.setEnabled(shared_video_context)
        for action in self._video_context_actions:
            action.setEnabled(False)
        self.temporal_lane_combo.setEnabled(shared_video_context)
        self.temporal_label_combo.setEnabled(shared_video_context)
        self.temporal_dock.setEnabled(shared_video_context or not has_project)
        self.actionLabel_dir.setEnabled(
            self.annotation_mode == "Frame" and not self._active_video_frame_case
        )
        self._sync_mode_controls()
        self._apply_frame_action_constraints()
        extracting = bool(workspace and workspace._busy)
        is_zh = getattr(self, "_ui_language", self.cfg["software"]["language"]) == "zh"
        self.actionShow_image_workspace.setEnabled(not extracting)
        self.actionAnnotate_current_video_frame.setText(
            ("正在提取帧…" if is_zh else "Extracting frame…")
            if extracting else
            ("标注当前视频帧" if is_zh else "Annotate current video frame")
        )
        if self.workspace_stack.currentWidget() is workspace:
            self.actionSave.setEnabled(
                bool(has_project and workspace.save_project_action.isEnabled())
            )
        self._update_unified_video_shortcuts()
        if not shared_video_context:
            return

        for proxy, target in getattr(self, "_video_proxy_targets", {}).items():
            proxy.setEnabled(target.isEnabled())
        self.actionAnnotate_current_video_frame.setEnabled(
            workspace.annotate_frame_button.isEnabled()
        )
        self.actionReturn_to_video.setEnabled(
            self.workspace_stack.currentWidget() is not workspace
        )
        pending_interval = workspace._pending_interval is not None
        interval_controls_enabled = workspace.mark_start_button.isEnabled()
        self.actionTemporal_mark_start.setEnabled(
            interval_controls_enabled and not pending_interval
        )
        self.actionTemporal_mark_end.setEnabled(
            interval_controls_enabled and pending_interval
        )

        lane_data = workspace.category_combo.currentData()
        lane_index = self.temporal_lane_combo.findData(lane_data)
        self.temporal_lane_combo.blockSignals(True)
        if lane_index >= 0:
            self.temporal_lane_combo.setCurrentIndex(lane_index)
        self.temporal_lane_combo.blockSignals(False)

        labels = [
            workspace.label_combo.itemText(index)
            for index in range(workspace.label_combo.count())
        ]
        current_label = workspace.label_combo.currentText()
        combo = self.temporal_label_combo
        previous_labels = [combo.itemText(index) for index in range(combo.count())]
        blocker = QtCore.QSignalBlocker(combo)
        if labels != previous_labels:
            combo.clear()
            combo.addItems(labels)
        if combo.currentText() != current_label:
            combo.setCurrentText(current_label)
        del blocker

    def _set_video_temporal_lane(self, _index):
        workspace = self.video_annotation_workspace
        if workspace is None:
            return
        target_index = workspace.category_combo.findData(
            self.temporal_lane_combo.currentData()
        )
        if target_index >= 0 and target_index != workspace.category_combo.currentIndex():
            workspace.category_combo.setCurrentIndex(target_index)

    def _set_video_temporal_label(self, text):
        workspace = self.video_annotation_workspace
        if workspace is not None and workspace.label_combo.currentText() != text:
            workspace.label_combo.setCurrentText(text)

    def _invoke_video_workspace_action(self, action_name):
        workspace = self._ensure_video_workspace()
        if workspace.store is None:
            self.statusbar.showMessage(
                "请先从“文件”菜单打开视频或已有 IVAS 视频项目。", 5000
            )
            return None
        action = getattr(workspace, action_name)
        action.trigger()
        self._sync_video_context_actions()
        return None

    def _invoke_video_command(self, method_name, *args, require_project=True):
        workspace = self._ensure_video_workspace()
        if require_project and workspace.store is None:
            self.statusbar.showMessage(
                "请先从“文件”菜单打开视频或已有 IVAS 视频项目。", 5000
            )
            return None
        result = getattr(workspace, method_name)(*args)
        self._sync_video_context_actions()
        return result

    def _ensure_video_workspace(self):
        if self.video_annotation_workspace is not None:
            return self.video_annotation_workspace
        from IVAS.widgets.video_annotation_workspace import VideoAnnotationWorkspace

        workspace = VideoAnnotationWorkspace(
            output_root=self.video_output_root,
            parent=self,
            tracking_adapter_provider=lambda: (
                self.segany_video
                if self.use_segment_anything_video
                and getattr(self, "segany_video", None) is not None
                else None
            ),
        )
        workspace.set_language(self.cfg["software"]["language"])
        workspace.frame_annotation_requested.connect(
            self.open_video_frame_for_annotation
        )
        workspace.return_to_video_requested.connect(self.show_video_workspace)
        workspace.status_message.connect(self.statusbar.showMessage)
        workspace.output_root_changed.connect(self._remember_video_output_root)
        self.workspace_stack.addWidget(workspace)
        self.video_annotation_workspace = workspace
        self._attach_video_project_panels(workspace)
        self._bind_video_workspace_commands(workspace)
        return workspace

    def _remember_video_output_root(self, output_root):
        """Persist the chosen location for subsequently created video projects."""

        self.video_output_root = Path(output_root).expanduser().resolve()
        self.cfg["software"]["video_output_root"] = str(self.video_output_root)
        self.actionVideo_output_root.setToolTip(str(self.video_output_root))
        self.save_software_cfg()

    def open_video_project_manifest(self):
        """Open an existing video project directly from the unified File menu."""

        if not self._confirm_leave_unsaved_image():
            return
        workspace = self._ensure_video_workspace()
        previous_store = workspace.store
        workspace.open_project_dialog()
        if workspace.store is None or workspace.store is previous_store:
            self._sync_video_context_actions()
            return
        self.active_video_project_path = str(workspace.store.path)
        self._activate_video_workspace()

    def open_video_workspace(self):
        """Choose a video and open its unified temporal/spatial project."""
        if not self._confirm_leave_unsaved_image():
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open video project",
            "",
            "Video (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if not filename:
            return
        try:
            workspace = self._ensure_video_workspace()
            store = workspace.open_video(Path(filename))
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Open video failed", str(error))
            return
        self.active_video_project_path = str(store.path)
        self._activate_video_workspace()

    def show_video_workspace(self):
        if self.annotation_mode == "Video":
            if self.video_annotation_workspace._busy:
                self.video_annotation_workspace.cancel_frame_extraction()
            self._sync_mode_controls()
            return
        if not self._confirm_leave_unsaved_image():
            self._sync_mode_controls()
            return
        self._activate_video_workspace()

    def request_frame_mode(self):
        """Frame within a video means the frame at the current playhead."""
        workspace = self.video_annotation_workspace
        if workspace is not None and workspace._busy:
            self._sync_mode_controls()
            return
        if self.annotation_mode == "Video" and workspace.store is not None:
            if self._confirm_leave_unsaved_image():
                workspace.annotate_current_frame()
            # Extraction is asynchronous: change Mode only after a frame loads.
            self._sync_video_context_actions()
            return
        self.show_image_workspace()

    def _activate_video_workspace(self):
        workspace = self._ensure_video_workspace()
        self.scene.invalidate_mask_prediction()
        self.text_prompt_dock_widget.invalidate_prediction()
        workspace.set_frame_edit_position(None)
        self._enter_video_action_scope()
        if self._image_workspace_dock_visibility is None:
            self._image_workspace_dock_visibility = {
                dock: not dock.isHidden()
                for dock in (
                    self.categories_dock,
                    self.annos_dock,
                    self.files_dock,
                    self.info_dock,
                    self.text_prompt_dock,
                    self.visual_prompt_dock,
                )
            }
        for dock in self._image_workspace_dock_visibility:
            dock.hide()
        self.temporal_dock.show()
        self.timeline_dock.show()
        self._set_shared_video_transport_visible(True)
        self.toolBar.show()
        self.workspace_stack.setCurrentWidget(workspace)
        self.setWindowTitle(
            "{} — {}".format(workspace.video_path.name, APP_NAME)
            if workspace.video_path is not None else APP_NAME
        )
        self.actionShow_video_workspace.setChecked(True)
        self.modeState.setText(
            "视频预览"
            if self.cfg["software"]["language"] == "zh"
            else "Video preview"
        )
        self._sync_video_context_actions()
        self.statusbar.showMessage("统一视频 Case：时序、帧分割与追踪均可用。", 4000)

    def show_image_workspace(self):
        if self.video_annotation_workspace is not None:
            self.video_annotation_workspace.pause()
        self.workspace_stack.setCurrentWidget(self.view)
        if self._image_workspace_dock_visibility is not None:
            for dock, was_visible in self._image_workspace_dock_visibility.items():
                dock.setVisible(was_visible)
            self._image_workspace_dock_visibility = None
        self.toolBar.show()
        self._leave_video_action_scope()
        self._apply_frame_action_constraints()
        self.SeganyEnabled()
        self.actionShow_image_workspace.setChecked(True)
        video_frame_context = bool(
            self._video_frame_belongs_to_active_project()
        )
        self.temporal_dock.show()
        self.timeline_dock.setVisible(video_frame_context)
        self._set_shared_video_transport_visible(video_frame_context)
        if video_frame_context:
            self.tabifyDockWidget(self.categories_dock, self.temporal_dock)
            self.categories_dock.raise_()
        is_zh = self.cfg["software"]["language"] == "zh"
        self.modeState.setText(
            ("视频帧 + 时序" if is_zh else "Video frame + timeline")
            if video_frame_context
            else ("图像" if is_zh else "Image")
        )
        self._update_tracking_action_safety()
        self._sync_video_context_actions()
        if self.current_label is not None:
            self.setWindowTitle("{} — {}".format(self.current_label.label_path, APP_NAME))
        elif self.current_index is not None:
            self.setWindowTitle("{} — {}".format(self.files_list[self.current_index], APP_NAME))
        else:
            self.setWindowTitle(APP_NAME)
        self.view.setFocus(QtCore.Qt.OtherFocusReason)

    def open_video_frame_for_annotation(
        self, image_directory: str, label_directory: str, filename: str
    ) -> None:
        """Switch an extracted video frame into the existing IVAS canvas."""
        image_path = os.path.normcase(os.path.abspath(os.path.join(image_directory, filename)))
        same_frame = (
            self._video_frame_belongs_to_active_project()
            and self.current_label is not None and self.load_finished
            and os.path.normcase(os.path.abspath(os.path.join(self.image_root, self.files_list[self.current_index]))) == image_path
            and os.path.normcase(os.path.abspath(self.label_root)) == os.path.normcase(os.path.abspath(label_directory))
            and self._loaded_frame_signature == self._frame_disk_signature()
        )
        if same_frame:
            self.show_image_workspace()
            self.setWindowTitle("{} — {}".format(self.current_label.label_path, APP_NAME))
            self._pin_video_annotation_frame()
            return
        loaded = self.load_image_directory(
            image_directory,
            label_directory=label_directory,
            selected_filename=filename,
            video_case=True,
        )
        if loaded:
            self._pin_video_annotation_frame()
            project_path = (
                str(self.video_annotation_workspace.store.path)
                if self.video_annotation_workspace is not None
                and self.video_annotation_workspace.store is not None
                else self.active_video_project_path
            )
            self.statusbar.showMessage(
                "Video frame {} is linked to {}".format(
                    filename, project_path or "the active case"
                ),
                6000,
            )

    def _pin_video_annotation_frame(self):
        if not self._video_frame_belongs_to_active_project() or self.current_label is None:
            return
        workspace = self.video_annotation_workspace
        label = Path(self.current_label.label_path).resolve()
        for frame in workspace.store.project["spatial_frames"]:
            if workspace._resolve_project_asset(frame["annotation_path"]) == label:
                workspace.set_frame_edit_position(int(frame["pts_ms"]))
                return

    def _frame_disk_signature(self):
        if self.current_label is None or self.current_index is None:
            return None
        image = Path(self.image_root) / self.files_list[self.current_index]
        signatures = []
        for path in (image, Path(self.current_label.label_path)):
            try:
                stat = path.stat()
                signatures.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            except OSError:
                signatures.append(None)
        return tuple(signatures)

    def save_dir(self):
        """Open label root."""
        if self._active_video_frame_case:
            return
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return
        if not self._confirm_leave_unsaved_image():
            return

        self.label_root = dir
        self.actionLabel_dir.setStatusTip("Label root: {}".format(self.label_root))
        category_config = _find_category_config(dir)
        if category_config is not None:
            self.config_file = category_config
            self.reload_cfg()
        # 刷新图片
        if self.current_index is not None:
            self.show_image(self.current_index)

    def save(self):
        """Save annotations to disk."""
        if self.current_label is None:
            return
        self.current_label.objects.clear()
        for polygon in self.polygons:
            object = polygon.to_object()
            self.current_label.objects.append(object)
        self.current_label.note = self.info_dock_widget.lineEdit_note.text()

        self.plugin_manager_dialog.trigger_before_annotations_save()

        self.current_label.save_annotation()
        if self._video_frame_belongs_to_active_project():
            mark_saved = getattr(
                self.video_annotation_workspace, "mark_frame_annotation_saved", None
            )
            if mark_saved is not None:
                if not mark_saved(self.current_label.label_path):
                    self.saved = False
                    raise RuntimeError("Frame JSON was written, but its video project link could not be saved. Retry Save before leaving this frame.")
        # 保存标注文件的同时保存一份 IVAS 类别配置
        self.save_cfg(os.path.join(self.label_root, CATEGORY_CONFIG_NAME))
        self.set_saved_state(True)
        self._loaded_frame_signature = self._frame_disk_signature()

        self.plugin_manager_dialog.trigger_after_annotations_saved()

    def save_current_work(self):
        """Save whichever part of the unified Case is on the central surface."""

        workspace = self.video_annotation_workspace
        if (
            workspace is not None
            and self.workspace_stack.currentWidget() is workspace
        ):
            return workspace.save_project()
        try:
            return self.save()
        except Exception as error:
            self.saved = False
            QtWidgets.QMessageBox.critical(self, "Save failed", str(error))
            return False

    def update_group_display(self):
        self.categories_dock_widget.lineEdit_currentGroup.setText(
            str(self.current_group)
        )

    def show_image(self, index: int, zoomfit: bool = True):
        """
        Show image and load annotations if exists.

        Arguments:
            index (int): Index of image.
            zoomfit (bool): Zoomfit.
        """
        if not 0 <= index < len(self.files_list):
            return False
        file_path = os.path.join(self.image_root, self.files_list[index])
        try:
            if file_path.lower().endswith(".dcm"):
                from IVAS.utils.dicom import load_dcm_as_image
                image_data = load_dcm_as_image(file_path)
            else:
                with Image.open(file_path) as candidate:
                    candidate.load()
                    image_data = candidate.copy()
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Open image failed", str(error))
            return False
        if not self._confirm_leave_unsaved_image():
            return False

        self.text_prompt_dock_widget.invalidate_prediction()
        self.reset_action()
        self.scene.cancel_draw()
        self.scene.unload_image()
        self.annos_dock_widget.listWidget.clear()
        self.change_bit_map_to_label()
        #
        for label in (
            self.files_dock_widget.label_prev_state,
            self.files_dock_widget.label_current_state,
            self.files_dock_widget.label_next_state,
        ):
            label.setStyleSheet(
                "background-color: {};".format(self.COLOR_STATE_DELETED)
            )

        self.current_label = None
        self.load_finished = False
        self.saved = True
        if not -1 < index < len(self.files_list):
            return
        try:
            self.current_index = index
            file_path = os.path.join(self.image_root, self.files_list[index])

            self.plugin_manager_dialog.trigger_before_image_open(file_path)

            if image_data.mode not in ("L", "P", "RGB", "RGBA"):
                warning_info = (
                    "当前图像不是8位深。PIL获取的mode为：{}".format(image_data.mode)
                    if self.cfg["software"]["language"] == "zh"
                    else "This image is not 8-bit per channel. The mode from PIL is {}".format(image_data.mode)
                )
                QtWidgets.QMessageBox.warning(
                    self, "Warning", warning_info, QtWidgets.QMessageBox.Ok
                )

            png_palette = image_data.getpalette()
            if png_palette is not None and file_path.lower().endswith(".png"):
                self.statusbar.showMessage(
                    "This image might be a label image in VOC format."
                )
                self.can_be_annotated = False
            else:
                self.can_be_annotated = True

            if self.can_be_annotated:
                self.actionPolygon.setEnabled(True)
                self.actionSave.setEnabled(True)
                self.actionBit_map.setEnabled(True)
                self.actionBackspace.setEnabled(True)
                self.actionFinish.setEnabled(True)
                self.actionCancel.setEnabled(True)
                self.actionVisible.setEnabled(True)
            else:
                self.actionPolygon.setEnabled(False)
                self.actionSave.setEnabled(False)
                self.actionBit_map.setEnabled(False)
                self.actionBackspace.setEnabled(False)
                self.actionFinish.setEnabled(False)
                self.actionCancel.setEnabled(False)
                self.actionVisible.setEnabled(False)

            self.scene.load_image(file_path, image=image_data)

            if zoomfit:
                self.view.zoomfit()

            # 判断图像是否旋转
            exif_info = image_data.getexif()
            if exif_info and exif_info.get(274, 1) != 1:
                warning_info = (
                    "这幅图像包含EXIF元数据，且图像的方向已被旋转.\n建议去除EXIF信息后再进行标注\n你可以使用[菜单栏]-[工具]-[处理exif标签]功能处理图像的旋转问题。"
                    if self.cfg["software"]["language"] == "zh"
                    else "This image has EXIF metadata, and the image orientation is rotated.\nSuggest labeling after removing the EXIF metadata.\nYou can use the function of [Process EXIF tag] in [Tools] in [Menu bar] to deal with the problem of images."
                )
                QtWidgets.QMessageBox.warning(
                    self, "Warning", warning_info, QtWidgets.QMessageBox.Ok
                )

            if self.use_segment_anything and self.can_be_annotated:
                self.seganythread.request_image(index)
                self.SeganyEnabled()

            # load label
            if self.can_be_annotated:
                self.current_group = 1
                _, name = os.path.split(file_path)
                label_path = os.path.join(
                    self.label_root, ".".join(name.split(".")[:-1]) + ".json"
                )
                self.current_label = Annotation(file_path, label_path)
                # 载入数据
                self.current_label.load_annotation()

                for object in self.current_label.objects:
                    try:
                        group = int(object.group)
                        # 新增 手动/自动 group选择
                        if self.group_select_mode == "auto":
                            self.current_group = (
                                group + 1
                                if group >= self.current_group
                                else self.current_group
                            )
                        elif self.group_select_mode == "manual":
                            self.current_group = 1
                    except Exception as e:
                        pass
                    if getattr(object, 'is_obb', False):
                        polygon = OBB()
                    else:
                        polygon = Polygon()
                    self._setup_polygon_alpha(polygon)
                    self.scene.addItem(polygon)
                    polygon.load_object(object)
                    self.polygons.append(polygon)
                self.update_group_display()
            if self.current_label is not None:
                self.setWindowTitle("{} — {}".format(self.current_label.label_path, APP_NAME))
            else:
                self.setWindowTitle("{} — {}".format(file_path, APP_NAME))

            self.annos_dock_widget.update_listwidget()
            self.info_dock_widget.update_widget()
            self.files_dock_widget.set_select(index)
            self.files_dock_widget.label_current.setText(
                "{}".format(self.current_index + 1)
            )
            self.load_finished = True
            self._loaded_frame_signature = self._frame_disk_signature()
            if self.annotation_mode == "Frame":
                self._pin_video_annotation_frame()

        except Exception as e:
            print(e)
        finally:
            if self.current_index > 0:
                self.actionPrev_image.setEnabled(True)
            else:
                self.actionPrev_image.setEnabled(False)

            if self.current_index < len(self.files_list) - 1:
                self.actionNext_image.setEnabled(True)
            else:
                self.actionNext_image.setEnabled(False)

            self.plugin_manager_dialog.trigger_after_image_open()
            self._update_tracking_action_safety()

    def prev_image(self):
        """Previous image."""
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index is None:
            return
        current_index = self.current_index - 1
        if current_index < 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "This is the first picture.")
        else:
            self.show_image(current_index)

    def next_image(self):
        """Next image."""
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index is None:
            return
        current_index = self.current_index + 1
        if current_index > len(self.files_list) - 1:
            QtWidgets.QMessageBox.warning(self, "Warning", "This is the last picture.")
        else:
            self.show_image(current_index)

    def jump_to(self, index_or_name: str = None):
        """
        Jump to the image.

        Arguments:
            index_or_name (str): Index or name of the image. Index in (1, len(self.files_list)); name with suffix.
        """
        if index_or_name is None:
            index_or_name = self.files_dock_widget.lineEdit_jump.text()
        if index_or_name:
            if not index_or_name.isdigit():
                if index_or_name in self.files_list:
                    index = self.files_list.index(index_or_name) + 1
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Warning",
                        "Don`t exist image named: {}".format(index_or_name),
                    )
                    self.files_dock_widget.lineEdit_jump.clear()
                    return
            else:
                index = int(index_or_name)
            index = int(index) - 1
            if 0 <= index < len(self.files_list):
                self.show_image(index)
                self.files_dock_widget.lineEdit_jump.clear()
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Warning",
                    "Index must be in [1, {}].".format(len(self.files_list)),
                )
                self.files_dock_widget.lineEdit_jump.clear()
                self.files_dock_widget.lineEdit_jump.clearFocus()
                return

    def category_setting(self):
        dialog = self._ensure_category_setting_dialog()
        dialog.load_cfg()
        dialog.show()

    def _set_map_mode(self, mode):
        if self.annotation_mode != "Frame":
            return False
        if self.scene.mode in (STATUSMode.CREATE, STATUSMode.REPAINT):
            self.statusbar.showMessage("请先完成或取消当前图形，再切换显示模式。", 4000)
            return False
        if mode != MAPMode.LABEL and not self.scene._frame_is_editable() and self.map_mode == MAPMode.LABEL:
            return False
        if self.scene.mode == STATUSMode.EDIT:
            self.scene.cancel_draw()
        if mode != self.map_mode:
            self.scene.invalidate_mask_prediction()
            self.text_prompt_dock_widget.invalidate_prediction()
        self.map_mode = mode
        editable = mode == MAPMode.LABEL
        if mode == MAPMode.INSTANCE and self.instance_cmap is None:
            import imgviz
            self.instance_cmap = imgviz.label_colormap()
        for polygon in self.polygons:
            polygon.setEnabled(editable)
            for vertex in polygon.vertices:
                vertex.setVisible(editable and polygon.isVisible())
            color = QtGui.QColor(self.category_color_dict.get(polygon.category, self.COLOR_DEFAULT))
            if mode == MAPMode.INSTANCE:
                try:
                    group = int(polygon.group)
                except (TypeError, ValueError):
                    group = 0
                rgb = self.instance_cmap[group % len(self.instance_cmap)]
                color = QtGui.QColor(*(int(value) for value in rgb))
            polygon.change_color(color)
            polygon.color.setAlpha(polygon.nohover_alpha if editable else 255)
            polygon.setBrush(polygon.color)
        self.annos_dock_widget.listWidget.setEnabled(editable)
        self.annos_dock_widget.checkBox_visible.setEnabled(editable)
        self.scene.change_mode_to_view()
        icon = {MAPMode.LABEL: "照片_pic.svg", MAPMode.SEMANTIC: "semantic.png",
                MAPMode.INSTANCE: "instance.png"}[mode]
        self.actionBit_map.setIcon(QtGui.QIcon(":/icon/icons/" + icon))
        self._apply_frame_action_constraints()
        return True

    def change_bit_map_to_semantic(self):
        return self._set_map_mode(MAPMode.SEMANTIC)

    def change_bit_map_to_instance(self):
        return self._set_map_mode(MAPMode.INSTANCE)

    def change_bit_map_to_label(self):
        return self._set_map_mode(MAPMode.LABEL)

    def change_bit_map(self):
        """Cycle the display without discarding an unfinished shape."""
        modes = {MAPMode.LABEL: MAPMode.SEMANTIC, MAPMode.SEMANTIC: MAPMode.INSTANCE,
                 MAPMode.INSTANCE: MAPMode.LABEL}
        self._set_map_mode(modes.get(self.map_mode, MAPMode.LABEL))

    def set_labels_visible(self, visible: bool = None):
        """
        Set polygons visible in scene.

        Arguments:
            visible (bool) : True or False
        """
        if visible is None:
            visible = not self.annos_dock_widget.checkBox_visible.isChecked()
        self.annos_dock_widget.set_all_polygon_visible(visible)

    def model_manage(self):
        """Open the model management interface."""
        dialog = self._ensure_model_manager_dialog()
        dialog.update_ui()
        dialog.show()

    def remote_sam(self):
        """Open the remote SAM model interface."""
        self._ensure_remote_sam_dialog().show()

    def change_sam_precision(self, index: int):
        """Change SAM model precision and reload the active checkpoint."""

        precision = ("original", "bfloat16", "float16", "float32")[index]
        if self.cfg["software"].get("sam_precision") == precision:
            return
        self.cfg["software"]["sam_precision"] = precision
        self.cfg["software"]["use_bfloat16"] = precision == "bfloat16"
        self.init_segment_anything()
        self.save_software_cfg()

    def change_bfloat16_state(self, check_state: QtCore.Qt.CheckState):
        """Compatibility adapter for plugins using the former checkbox API."""

        precision = (
            "bfloat16"
            if check_state == QtCore.Qt.CheckState.Checked
            else "float32"
        )
        self.change_sam_precision(
            ("original", "bfloat16", "float16", "float32").index(precision)
        )

    def change_use_video_segmentation_state(self, check_state: QtCore.Qt.CheckState):
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["use_video_segmentation"] = checked
        self.init_segment_anything()
        self.save_software_cfg()

    def change_contour_mode(self, contour_mode: str = "max_only"):
        """
        Change contour mode. It has an effect when converting sam mask to polygons.

        Arguments:
            contour_mode (str) : contour mode. which include 'max_only' 'external' and 'all'.
        """
        if contour_mode == "max_only":
            self.scene.change_contour_mode_to_save_max_only()
        elif contour_mode == "external":
            self.scene.change_contour_mode_to_save_external()
            self.statusbar.showMessage(
                "Save all external contours will bring some noise.", 3000
            )
        elif contour_mode == "all":
            self.scene.change_contour_mode_to_save_all()
            self.statusbar.showMessage(
                "Category of inner contour will be set _background__.", 3000
            )
        else:
            self.scene.change_contour_mode_to_save_max_only()
            self.statusbar.showMessage(
                "The contour mode [{}] not support.".format(contour_mode), 3000
            )

        if contour_mode == "external":
            index = 0
        elif contour_mode == "max_only":
            index = 1
        elif contour_mode == "all":
            index = 2
        else:
            index = 0
        self.setting_dialog.comboBox_contour_mode.setCurrentIndex(index)
        self.cfg["software"]["contour_mode"] = contour_mode
        self.save_software_cfg()

    def change_contour_method(self, contour_method: str = "SIMPLE"):
        if contour_method == "SIMPLE":
            self.scene.change_contour_method_to_simple()
            index = 0
        elif contour_method == "TC89_KCOS":
            self.scene.change_contour_method_to_tc89_kcos()
            index = 1
        elif contour_method == "NONE":
            self.scene.change_contour_method_to_none()
            index = 2
        else:
            self.scene.change_contour_method_to_simple()
            index = 0
        self.setting_dialog.comboBox_contour_method.setCurrentIndex(index)
        self.cfg["software"]["contour_method"] = contour_method
        self.save_software_cfg()

    def change_mask_alpha(self, value: int):
        """
        Change mask alpha.

        Arguments:
            value (int) : mask alpha. [0, 10]
        """
        value = value / 10
        self.scene.mask_alpha = value
        self.scene.update_mask()
        self.cfg["software"]["mask_alpha"] = value
        self.save_software_cfg()
        self.setting_dialog.label_mask_alpha.setText("{}".format(value))

    def change_polygon_alpha_hover(self, value: int):
        value = value / 10
        self.cfg["software"]["polygon_alpha_hover"] = value
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)
        self.setting_dialog.label_polygon_alpha_hover.setText("{}".format(value))

    def change_polygon_alpha_no_hover(self, value: int):
        value = value / 10
        self.cfg["software"]["polygon_alpha_no_hover"] = value
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)
        self.setting_dialog.label_polygon_alpha_no_hover.setText("{}".format(value))

    def change_vertex_size(self, value: int):
        """
        Change vertex size.

        Arguments:
            value (int) : vertex size. [0, 5]
        """
        self.cfg["software"]["vertex_size"] = value
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)
        self.setting_dialog.label_vertex_size.setText("{}".format(value))

    def change_auto_save_state(self, check_state: QtCore.Qt.CheckState):
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["auto_save"] = checked
        self.save_software_cfg()

    def change_real_time_area_state(self, check_state: QtCore.Qt.CheckState):
        """
        Change real time area state.

        .. tip::

            The area of the polygon can be viewed in the edit interface.(double click polygon to open.)
        """
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["real_time_area"] = checked
        self.save_software_cfg()

    def change_edge_state(self, check_state: QtCore.Qt.CheckState):
        """Show edge of polygons."""
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["show_edge"] = checked
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)

    def change_approx_polygon_state(
        self, check_state: QtCore.Qt.CheckState
    ):  # 是否使用多边形拟合，来减少多边形顶点
        """Change polygon state. It has an effect when converting sam mask to polygons, can reduce the number of vertices of the polygon."""
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["use_polydp"] = checked
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)

    def change_create_mode_invisible_polygon_state(
        self, check_state: QtCore.Qt.CheckState
    ):
        """Invisible polygons in the create mode."""
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["create_mode_invisible_polygon"] = checked
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)
        pass

    def change_saturation(self, value: int):  # 调整图像饱和度
        """Only acts on display, don't acts on SAM model."""
        if self.scene.image_data is not None:
            import cv2

            saturation_scale = value / 100.0
            hsv_image = cv2.cvtColor(self.scene.image_data, cv2.COLOR_RGB2HSV)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
            image_hsv = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            height, width, channels = self.scene.image_data.shape
            pixmap = QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    image_hsv.tobytes(),
                    width,
                    height,
                    channels * width,
                    QtGui.QImage.Format_RGB888,
                )
            )
            self.scene.image_item.setPixmap(pixmap)
        else:
            print("Image data not loaded in AnnotationScene")

    def change_prompt_visiable(self, check_state: QtCore.Qt.CheckState):
        """Show SAM propmt point."""
        checked = check_state == QtCore.Qt.CheckState.Checked
        self.cfg["software"]["show_prompt"] = checked
        self.save_software_cfg()
        for item in self.scene.items():
            if isinstance(item, PromptPoint):
                item.setVisible(checked)
        # if self.current_index is not None:
        #     self.show_image(self.current_index)

    def converter(self):
        """Open data converter interface."""
        dialog = self._ensure_converter_dialog()
        current_converter = self.cfg["software"].get("current_converter", "coco")
        if current_converter == "coco":
            current_converter_tab = dialog.tab_COCO
        elif current_converter == "yolo":
            current_converter_tab = dialog.tab_YOLO
        elif current_converter == "labelme":
            current_converter_tab = dialog.tab_LABELME
        elif current_converter == "voc":
            current_converter_tab = dialog.tab_VOC
        elif current_converter == "voc for detection":
            current_converter_tab = dialog.tab_VOC_DETECTION
        else:
            current_converter_tab = dialog.tab_COCO

        dialog.tabWidget.setCurrentWidget(current_converter_tab)

        dialog.show()

    def video2frames(self):
        """Open video to farames interface. Split the video to frames."""
        self._ensure_video2frames_dialog().show()

    def auto_segment(self):
        """Open auto segment interface. Convert object detection data to segment data."""
        if self.use_segment_anything:
            self._ensure_auto_segment_dialog().show()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Select a sam model before auto segment."
            )

    def annos_validator(self):
        self._ensure_annos_validator_dialog().show()

    def process_exif(self):
        """Open process exif interface. Deal with the rotation problem of pictures with EXIF tags."""
        self._ensure_process_exif_dialog().show()

    def shortcut(self):
        """Open shortcut interface."""
        dialog = self._ensure_shortcut_dialog()
        dialog.update_ui()
        dialog.show()

    def about(self):
        """Open about interface."""
        self._ensure_about_dialog().show()

    def setting(self):
        """Open setting interface."""
        # self.setting_dialog.update_ui()
        self.setting_dialog.show()

    def screen_shot(self, type: str = "scene"):
        """
        Screenshot.

        Arguments:
            type (str) : 'scene' or 'window'
        """
        # image_name = "IVAS-{}-{}.png".format(type, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if self.current_index is not None and self.files_list:
            current_image = self.files_list[self.current_index]
        else:
            current_image = "no_image.jpg"
            if type == "scene":
                print("no image saved.")
                return
        image_name = os.path.splitext(current_image)[0] + ".png"
        screen_shot_dir = os.path.join(os.getcwd(), "screen_shots")
        os.makedirs(screen_shot_dir, exist_ok=True)
        save_path = os.path.join(screen_shot_dir, image_name)
        print(f"save path: {save_path}")

        if type == "scene":
            try:
                self.scene.guide_line_x.setVisible(False)
                self.scene.guide_line_y.setVisible(False)
            except:
                pass
            image = QtGui.QImage(
                self.scene.sceneRect().size().toSize(), QtGui.QImage.Format_ARGB32
            )
            painter = QtGui.QPainter(image)
            self.scene.render(painter)
            painter.end()
            image.save(save_path)
            self.statusbar.showMessage(
                "Save scene screenshot to {}".format(save_path), 3000
            )
            try:
                self.scene.guide_line_x.setVisible(True)
                self.scene.guide_line_y.setVisible(True)
            except:
                pass
        elif type == "window":
            image = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
            painter = QtGui.QPainter(image)
            self.render(painter)
            painter.end()
            image.save(save_path)
            self.statusbar.showMessage(
                "Save window screenshot to {}".format(save_path), 3000
            )

        else:
            return

    def save_cfg(self, config_file: str):
        """
        Save categories config to disk.

        Arguments:
            config_file (str) : Path to config file.
        """
        # 只保存类别配置
        cfg = {"label": self.cfg.get("label", [])}
        save_config(cfg, config_file)

    def save_software_cfg(self):
        """
        Save all config to disk.
        """
        if self._loading_config:
            return
        save_config(self.cfg, self.software_config_file)

    def open_docs(self):
        """Open the bundled IVAS documentation without relying on a stale URL."""
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(IVAS_ROOT).parent))
        filename = (
            "README-cn.md"
            if self.cfg["software"]["language"] == "zh"
            else "README.md"
        )
        documentation = bundle_root / filename
        if documentation.is_file():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(documentation.resolve()))
            )
        else:
            self.statusbar.showMessage(
                "Bundled documentation was not found: {}".format(documentation),
                6000,
            )

    def exit(self):
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        if self._shutdown_started:
            a0.accept()
            return
        if not self._confirm_leave_unsaved_image():
            a0.ignore()
            return
        mask_worker = self.scene._mask_worker
        if mask_worker is not None and mask_worker.isRunning():
            self.scene.invalidate_mask_prediction()
            mask_worker.finished.connect(self.close)
            self.statusbar.showMessage("正在结束分割任务，完成后关闭窗口。")
            a0.ignore()
            return
        if self.video_annotation_workspace is not None:
            confirm_pending = getattr(
                self.video_annotation_workspace, "confirm_discard_pending", None
            )
            if confirm_pending is not None and not confirm_pending(self):
                a0.ignore()
                return
            if not self.video_annotation_workspace.shutdown():
                a0.ignore()
                return
        self._shutdown_started = True
        self._version_check_timer.stop()
        version_thread = getattr(self, "check_latest_version_thread", None)
        if version_thread is not None and version_thread.isRunning():
            version_thread.requestInterruption()
            version_thread.wait(3500)
        self._stop_gpu_monitor()
        self.save_cfg(self.config_file)
        self.save_software_cfg()
        self.plugin_manager_dialog.trigger_application_shutdown()
        a0.accept()

    def latest_version_tip(self, is_latest_version: bool, latest_version: str):
        """
        Check whether it is the latest version.

        Arguments:
            is_latest_version (bool): Whether it is the latest version.
            latest_version (str): The latest version.
        """
        if not is_latest_version:
            title = APP_NAME
            if self.cfg["software"]["language"] == "zh":
                text = "检测到新版本 {}，请通过当前软件发布渠道更新。".format(
                    latest_version
                )
            else:
                text = (
                    "Version {} is available. Update it through the channel "
                    "that supplied this IVAS build."
                ).format(latest_version)

            QtWidgets.QMessageBox.information(self, title, text)

    @staticmethod
    def create_desktop_shortcut():
        import sys

        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        python_path = sys.executable
        pkg_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        main_path = os.path.join(pkg_root, "IVAS/main.py")
        icon_path = os.path.join(pkg_root, "logo", "logo.png")

        osplatform = platform.system()
        if osplatform == "Windows":
            try:
                with open(os.path.join(desktop_path, "IVAS.bat"), "w") as f:
                    f.write(
                        """@echo off
set PROJECT_ROOT="{}"
cd /d %PROJECT_ROOT%
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

{} {}
                    """.format(
                            pkg_root, python_path, main_path
                        )
                    )
            except Exception as e:
                print("Create app shortcut error: {}".format(e))

        elif osplatform == "Linux":
            try:
                with open(
                    os.path.join(
                        os.path.expanduser("~"),
                        ".local/share/applications/IVAS.desktop",
                    ),
                    "w",
                ) as f:
                    f.write(
                        """[Desktop Entry]
Name=IVAS
Type=Application
Comment=IVAS visual annotation toolkit
Exec=env PYTHONPATH={} {} {}
Terminal=false
Icon={}
Categories=Development;System;
                    """.format(
                            pkg_root, python_path, main_path, icon_path
                        )
                    )
            except Exception as e:
                print("Create app shortcut error: {}".format(e))

        elif osplatform == "Darwin":
            pass
        else:
            pass

    def init_connect(self):
        self.actionOpen_video_workspace.triggered.connect(self.open_video_workspace)
        self.actionShow_image_workspace.triggered.connect(self.request_frame_mode)
        self.actionShow_video_workspace.triggered.connect(self.show_video_workspace)
        self.actionImages_dir.triggered.connect(self.open_dir)
        self.actionLabel_dir.triggered.connect(self.save_dir)
        self.actionVideo_segment.triggered.connect(
            functools.partial(self.seg_video_start, None)
        )
        self.actionVideo_segment_once.triggered.connect(
            functools.partial(self.seg_video_start, 1)
        )
        self.actionVideo_segment_five_times.triggered.connect(
            functools.partial(self.seg_video_start, 5)
        )

        self.actionPrev_image.triggered.connect(self.prev_image)
        self.actionNext_image.triggered.connect(self.next_image)
        self.actionSetting.triggered.connect(self.setting)
        self.actionExit.triggered.connect(self.exit)

        self.actionSegment_anything_point.triggered.connect(
            self.scene.start_segment_anything
        )
        self.actionSegment_anything_box.triggered.connect(
            self.scene.start_segment_anything_box
        )
        self.actionPolygon.triggered.connect(self.scene.start_draw_polygon)
        self.actionRepaint.triggered.connect(self.scene.change_mode_to_repaint)
        self.actionCancel.triggered.connect(self.scene.cancel_draw)
        self.actionBackspace.triggered.connect(self.scene.backspace)
        self.actionFinish.triggered.connect(self.scene.finish_draw)
        self.actionEdit.triggered.connect(self.scene.edit_polygon)
        self.actionDelete.triggered.connect(self.scene.delete_selected_graph)
        self.actionSave.triggered.connect(self.save_current_work)
        self.actionTo_top.triggered.connect(self.scene.move_polygon_to_top)
        self.actionTo_bottom.triggered.connect(self.scene.move_polygon_to_bottom)
        self.actionCopy.triggered.connect(self.scene.copy_item)
        self.actionUnion.triggered.connect(self.scene.polygons_union)
        self.actionSubtract.triggered.connect(self.scene.polygons_difference)
        self.actionIntersect.triggered.connect(self.scene.polygons_intersection)
        self.actionExclude.triggered.connect(self.scene.polygons_symmetric_difference)

        self.actionPrev_group.triggered.connect(self.annos_dock_widget.go_to_prev_group)
        self.actionNext_group.triggered.connect(self.annos_dock_widget.go_to_next_group)
        self.actionZoom_in.triggered.connect(self.view.zoom_in)
        self.actionZoom_out.triggered.connect(self.view.zoom_out)
        self.actionFit_window.triggered.connect(self.view.zoomfit)
        self.actionBit_map.triggered.connect(self.change_bit_map)
        self.actionVisible.triggered.connect(
            functools.partial(self.set_labels_visible, None)
        )
        self.actionScene_shot.triggered.connect(
            functools.partial(self.screen_shot, "scene")
        )
        self.actionWindow_shot.triggered.connect(
            functools.partial(self.screen_shot, "window")
        )

        self.actionModel_manage.triggered.connect(self.model_manage)
        self.actionModel_manage.setStatusTip(CHECKPOINT_PATH)

        self.actionRemote_SAM.triggered.connect(self.remote_sam)

        self.actionPlugins.triggered.connect(self.plugin_manager_dialog.show)
        self.actionConverter.triggered.connect(self.converter)
        self.actionVideo_to_frames.triggered.connect(self.video2frames)
        self.actionAuto_segment_with_bounding_box.triggered.connect(self.auto_segment)
        self.actionAnno_validator.triggered.connect(self.annos_validator)
        self.actionProcess_EXIF_tag.triggered.connect(self.process_exif)

        self.actionShortcut.triggered.connect(self.shortcut)
        self.actionAbout.triggered.connect(self.about)

        self.actionLanguage.triggered.connect(self.change_language)
        self.actionDocs.triggered.connect(self.open_docs)
        self.actionCreate_a_desktop_shortcut.triggered.connect(
            self.create_desktop_shortcut
        )

        self.annos_dock_widget.listWidget.doubleClicked.connect(self.scene.edit_polygon)

    def reset_action(self):
        self.actionPrev_image.setEnabled(False)
        self.actionNext_image.setEnabled(False)
        self.actionSegment_anything_point.setEnabled(False)
        self.actionSegment_anything_box.setEnabled(False)
        self.actionPolygon.setEnabled(False)
        self.actionRepaint.setEnabled(False)
        self.actionVideo_segment.setEnabled(False)
        self.actionVideo_segment_once.setEnabled(False)
        self.actionVideo_segment_five_times.setEnabled(False)
        self.actionEdit.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionTo_top.setEnabled(False)
        self.actionTo_bottom.setEnabled(False)
        self.actionBit_map.setChecked(False)
        self.actionBit_map.setEnabled(False)
        self.actionCopy.setEnabled(False)
        self.actionUnion.setEnabled(False)
        self.actionSubtract.setEnabled(False)
        self.actionIntersect.setEnabled(False)
        self.actionExclude.setEnabled(False)
        self._update_tracking_action_safety()
