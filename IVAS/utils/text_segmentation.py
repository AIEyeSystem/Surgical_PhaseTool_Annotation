"""Folder text segmentation using captured inputs, without accessing Qt widgets."""

from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from IVAS.annotation import Annotation, Object
from IVAS.configs import CONTOURMode
from IVAS.utils.masks import mask_to_polygons
from IVAS.utils.persistence import project_write_lock


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


def segment_folder_with_text_prompt(model, image_root, label_root, filenames,
                                    category, prompt, missing_only, contour_options,
                                    cancelled, progress, project_path=None):
    """Append predictions atomically; SegMiss tests canonical annotation categories.

    Existing objects are kept, including manually reviewed objects of the same
    category. A cancelled in-flight image is never saved. A failed image leaves
    its label untouched and does not prevent later images from being processed.
    """
    if model.model_source not in ("sam3", "sam3.1"):
        raise ValueError("Only SAM3/SAM3.1 text models are supported")
    if not category or category == "__background__" or not prompt.strip():
        raise ValueError("Select a foreground category and a non-empty prompt")
    filenames = tuple(filenames)
    report = dict(total=len(filenames), processed=0, skipped=0, empty=0,
                  objects=0, errors=[], written_paths=[], cancelled=False)
    for position, filename in enumerate(filenames, 1):
        if cancelled.is_set():
            break
        image_path = Path(image_root) / filename
        label_path = Path(label_root) / (image_path.stem + ".json")
        progress(position - 1, len(filenames), filename)
        try:
            annotation = Annotation(str(image_path), str(label_path)).load_annotation()
            if missing_only and any(obj.category == category for obj in annotation.objects):
                report["skipped"] += 1
                continue
            if str(image_path).lower().endswith(".dcm"):
                from IVAS.utils.dicom import load_dcm_as_image
                image = load_dcm_as_image(str(image_path)).convert("RGB")
            else:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
            # Encoding and prediction must share the same image-state lease.
            with getattr(model, "inference_lock", nullcontext()):
                if cancelled.is_set():
                    break
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
                report["empty"] += int(created == 0)
        except Exception as error:
            report["errors"].append("{}: {}".format(filename, error))
        finally:
            progress(position, len(filenames), filename)
    report["cancelled"] = cancelled.is_set()
    return report
