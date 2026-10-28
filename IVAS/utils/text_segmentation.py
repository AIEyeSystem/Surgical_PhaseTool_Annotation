"""Folder text segmentation using captured inputs, without accessing Qt widgets."""

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from IVAS.annotation import Annotation, Object
from IVAS.branding import SUPPORTED_ANNOTATION_FORMATS
from IVAS.configs import CONTOURMode
from IVAS.utils.masks import mask_to_polygons
from IVAS.utils.persistence import project_write_lock

# Poll interval while another job (usually the image encoder) owns the model.
LOCK_POLL_SECONDS = 0.05

_MESSAGES = {
    "en": {
        "waiting": "{name} - waiting for the SAM model (busy with another job, {seconds}s)",
        "segmenting": "{name} - segmenting",
    },
    "zh": {
        "waiting": "{name} - 等待 SAM 模型（正被其他任务占用，{seconds} 秒）",
        "segmenting": "{name} - 正在分割",
    },
}


def _append_masks(annotation, masks, category, contour_options):
    masks = np.asarray(masks)
    if masks.size == 0:
        return 0
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (annotation.height, annotation.width):
        raise ValueError("SAM masks do not match the image dimensions")

    group = max((obj.group for obj in annotation.objects), default=0) + 1
    layer = max((obj.layer for obj in annotation.objects), default=0) + 1
    created = 0
    seen = set()
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        fingerprint = np.packbits(mask).tobytes()
        if not mask.any() or fingerprint in seen:
            continue
        seen.add(fingerprint)
        contours, hierarchy = mask_to_polygons(mask, *contour_options)
        for index, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            inner = (contour_options[0] == CONTOURMode.SAVE_ALL
                     and hierarchy is not None and hierarchy[0][index][3] != -1)
            points = np.maximum(contour.reshape(-1, 2), 0.1).tolist()
            xs, ys = zip(*points)
            annotation.objects.append(Object(
                "__background__" if inner else category, 0 if inner else group,
                points, float(cv2.contourArea(np.asarray(points, dtype=np.float32))),
                layer, [min(xs), min(ys), max(xs), max(ys)],
                source="sam", review_status="draft",
            ))
            created += 1
            layer += 1
        group += 1
    return created


def label_has_category(label_path, category):
    """Return whether a saved label already contains ``category``.

    Reads only the JSON. SegMiss used to build an ``Annotation`` for every
    image first, which decodes the whole image just to learn its size; on a
    folder of large frames that meant thousands of full decodes before the
    first missing image was reached.
    """
    path = Path(label_path)
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("info"), dict):
        raise ValueError("Annotation must contain an info object")
    if dataset["info"].get("description", "") not in SUPPORTED_ANNOTATION_FORMATS:
        raise ValueError("The file {} is not an IVAS annotation JSON".format(path))
    objects = dataset.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Annotation objects must be a list")
    return any(isinstance(obj, dict) and str(obj.get("category", "unknown")) == category
               for obj in objects)


@contextmanager
def model_lease(model, cancelled, waiting=None, poll=LOCK_POLL_SECONDS):
    """Hold the model's inference lock, giving up as soon as ``cancelled`` is set.

    Yields ``True`` when the lock is held, ``False`` when the job was cancelled
    while another thread (the image encoder, an interactive mask job, ...)
    still owned the model. ``waiting(seconds)`` is called once per elapsed
    second so the UI can show that the job is alive but queued.
    """
    lock = getattr(model, "inference_lock", None)
    if lock is None:
        yield not cancelled.is_set()
        return
    acquired = lock.acquire(blocking=False)
    started = time.monotonic()
    reported = None
    while not acquired and not cancelled.is_set():
        seconds = int(time.monotonic() - started)
        if waiting is not None and seconds != reported:
            reported = seconds
            waiting(seconds)
        acquired = lock.acquire(timeout=poll)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


def _load_rgb(image_path):
    if str(image_path).lower().endswith(".dcm"):
        from IVAS.utils.dicom import load_dcm_as_image
        return load_dcm_as_image(str(image_path)).convert("RGB")
    with Image.open(image_path) as source:
        return source.convert("RGB")


def segment_folder_with_text_prompt(model, image_root, label_root, filenames,
                                    category, prompt, missing_only, contour_options,
                                    cancelled, progress, project_path=None,
                                    language="en"):
    """Append predictions atomically; SegMiss tests canonical annotation categories.

    Existing objects are kept, including manually reviewed objects of the same
    category. A cancelled in-flight image is never saved. A failed image leaves
    its label untouched and does not prevent later images from being processed.
    Images processed without any detection are listed in ``empty_files``.
    """
    if model.model_source not in ("sam3", "sam3.1"):
        raise ValueError("Only SAM3/SAM3.1 text models are supported")
    if not category or category == "__background__" or not prompt.strip():
        raise ValueError("Select a foreground category and a non-empty prompt")
    messages = _MESSAGES.get(language, _MESSAGES["en"])
    filenames = tuple(filenames)
    total = len(filenames)
    report = dict(total=total, processed=0, skipped=0, empty=0,
                  objects=0, errors=[], written_paths=[], empty_files=[], cancelled=False)
    for position, filename in enumerate(filenames, 1):
        if cancelled.is_set():
            break
        image_path = Path(image_root) / filename
        label_path = Path(label_root) / (image_path.stem + ".json")
        progress(position - 1, total, filename)
        try:
            # SegMiss: decide from the JSON alone, before any image decode.
            if missing_only and label_has_category(label_path, category):
                report["skipped"] += 1
                continue
            annotation = Annotation(str(image_path), str(label_path)).load_annotation()
            image = _load_rgb(image_path)
            if cancelled.is_set():
                break

            def waiting(seconds, name=filename, done=position - 1):
                progress(done, total, messages["waiting"].format(name=name, seconds=seconds))

            # Encoding and prediction must share the same image-state lease.
            # The lease is cancellable: the encoder thread may own the model.
            with model_lease(model, cancelled, waiting) as leased:
                if not leased or cancelled.is_set():
                    break
                progress(position - 1, total, messages["segmenting"].format(name=filename))
                if model.model_source == "sam3.1":
                    model.set_image(np.asarray(image))
                    if cancelled.is_set():
                        break
                    masks, _scores = model.predict_with_text_prompt(prompt)
                else:
                    masks, _scores = model.predictor.predict_with_text_prompt(image, prompt)
            if cancelled.is_set():
                break
            # Reload before merging so updates made during inference survive.
            with project_write_lock(project_path) if project_path is not None else nullcontext():
                annotation.load_annotation()
                if missing_only and any(obj.category == category for obj in annotation.objects):
                    report["skipped"] += 1
                    continue
                created = _append_masks(annotation, masks, category, contour_options)
                if cancelled.is_set():
                    break
                if created or not label_path.exists():
                    annotation.save_annotation()
                    report["written_paths"].append(str(label_path))
                report["processed"] += 1
                report["objects"] += created
                if created == 0:
                    report["empty"] += 1
                    report["empty_files"].append(filename)
        except Exception as error:
            report["errors"].append("{}: {}".format(filename, error))
        finally:
            progress(position, total, filename)
    report["cancelled"] = cancelled.is_set()
    return report
