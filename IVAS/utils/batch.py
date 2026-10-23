"""Widget-free batch operations. Each input is validated independently."""

import io
import json
from pathlib import Path

import cv2
from PIL import Image, ImageOps
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from IVAS.annotation import Object
from IVAS.utils.persistence import atomic_write_bytes


def process_exif(image_root, save_root, apply_exif, cancelled, progress):
    source, destination = Path(image_root), Path(save_root)
    if not image_root or not save_root or not source.is_dir() or not destination.is_dir():
        raise ValueError("Select valid input and output directories")
    files = sorted(path for path in source.iterdir() if path.is_file())
    errors = 0
    for index, path in enumerate(files, 1):
        if cancelled.is_set():
            break
        try:
            with Image.open(path) as image:
                image.load()
                corrected = ImageOps.exif_transpose(image) if apply_exif else image.copy()
                # Strip the rotation tag when explicitly removing EXIF too.
                corrected.info.pop("exif", None)
                buffer = io.BytesIO()
                corrected.save(buffer, format=image.format)
            if cancelled.is_set():
                break
            atomic_write_bytes(destination / path.name, buffer.getvalue())
            message = "Saved: {}".format(path.name)
        except Exception as error:
            errors += 1
            message = "Error: {}: {}".format(path.name, error)
        progress(index, len(files), message)
    return "{} error(s)".format(errors)


def video_to_frames(video_path, save_root, step, cancelled, progress):
    source, destination = Path(video_path), Path(save_root)
    if not source.is_file() or not destination.is_dir() or step <= 0:
        raise ValueError("Select a video, an output directory and a positive frame step")
    capture = cv2.VideoCapture(str(source))
    errors = 0
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open video")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = range(0, count, step)
        for position, index in enumerate(indices, 1):
            if cancelled.is_set():
                break
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError("Frame could not be decoded")
                ok, data = cv2.imencode(".jpg", frame)
                if not ok:
                    raise ValueError("Frame could not be encoded")
                output = destination / "{}_{:012d}.jpg".format(source.stem, index)
                if cancelled.is_set():
                    break
                atomic_write_bytes(output, data.tobytes())
                message = "Saved: {}".format(output.name)
            except Exception as error:
                errors += 1
                message = "Error: frame {}: {}".format(index, error)
            progress(position, len(indices), message)
    finally:
        capture.release()
    return "{} error(s)".format(errors)


def validate_annotations(root, cancelled, progress):
    files = sorted(Path(root).glob("*.json"))
    errors = 0
    for index, path in enumerate(files, 1):
        if cancelled.is_set():
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            width, height = data["info"]["width"], data["info"]["height"]
            for number, raw in enumerate(data["objects"]):
                if cancelled.is_set():
                    break
                try:
                    obj = Object.from_dict(raw)
                    polygon = Polygon(obj.segmentation)
                    if not polygon.is_valid:
                        raise ValueError(explain_validity(polygon))
                    left, top, right, bottom = polygon.bounds
                    if left < 0 or top < 0 or right > width or bottom > height:
                        raise ValueError("Polygon is outside the image")
                except Exception as error:
                    errors += 1
                    progress(index, len(files), "Error: {} object {}: {}".format(path.name, number, error))
        except Exception as error:
            errors += 1
            progress(index, len(files), "Error: {}: {}".format(path.name, error))
        progress(index, len(files), "Checked: {}".format(path.name))
    return "{} error(s)".format(errors)
