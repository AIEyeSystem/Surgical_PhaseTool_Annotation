# -*- coding: utf-8 -*-
"""Point exemplars for SAM3 visual (concept) prompting.

SAM3's concept detector (PCS) is prompted by text and/or *box* exemplars; it
does not use points: ``Sam3Image.forward`` warns "Point prompts are ignored in
PCS" (sam3/model/sam3_image.py) and the upstream ``Sam3Processor`` only exposes
``add_geometric_prompt(box, label)``. Points are a visual-segmentation (PVS)
input handled by the SAM2-style instance head.

A clicked point therefore becomes an exemplar in two steps:

1. resolve the click to the instance under it with the interactive point
   predictor (the same path as the canvas point tool), and
2. use that instance's bounding box as a positive/negative exemplar box.

Feeding a tiny box around the click instead would be cheaper but wrong: box
exemplars encode extent (ROI-pooled features and box width/height), so a 2px
box asks for tiny objects.

Post-processing keeps the points meaningful even when the detector disagrees:
a mask containing a negative point is dropped, and a positive point whose
object was not detected contributes its own instance mask.
"""

import numpy as np


def to_numpy(value):
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().float().cpu().numpy() if value.is_floating_point() else detach().cpu().numpy()
    return np.asarray(value)


def as_mask_stack(masks, height, width):
    """Return masks as an (N, H, W) bool array (N may be 0)."""
    masks = to_numpy(masks)
    if masks is None or masks.size == 0:
        return np.zeros((0, height, width), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError("Visual prompt masks must have shape HxW, NxHxW or Nx1xHxW")
    return masks.astype(bool, copy=False)


def mask_bbox(mask):
    """Tight [xmin, ymin, xmax, ymax] box of a mask (pixel edges), or None if empty."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[0]
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def contains_point(mask, point):
    height, width = mask.shape[-2:]
    x = min(max(int(round(point[0])), 0), width - 1)
    y = min(max(int(round(point[1])), 0), height - 1)
    return bool(mask[y, x])


def point_exemplars(predict_instance, points, labels):
    """Resolve clicks to instances and exemplar boxes.

    ``predict_instance(x, y)`` returns ``(mask, score)`` for the object under
    the click (always queried as a *positive* click: the point's own label only
    decides the exemplar polarity). Returns ``(boxes, box_labels, instances)``
    where ``instances`` is a list of ``(mask_or_None, score, point, label)``.
    """
    if len(points) != len(labels):
        raise ValueError("points and point labels must have the same length")
    boxes, box_labels, instances = [], [], []
    for point, label in zip(points, labels):
        x, y = float(point[0]), float(point[1])
        mask, score = predict_instance(x, y)
        mask = None if mask is None else np.asarray(mask, dtype=bool)
        if mask is not None and mask.ndim == 3:
            mask = mask[0]
        box = None if mask is None else mask_bbox(mask)
        if box is None:
            mask = None
        else:
            boxes.append(box)
            box_labels.append(bool(label))
        instances.append((mask, float(score), (x, y), bool(label)))
    return boxes, box_labels, instances


def merge_point_results(masks, scores, instances):
    """Apply point semantics on top of the concept detector output.

    * drop every mask that contains a negative point;
    * add the clicked instance for each positive point that no kept mask
      covers (unless that instance itself contains a negative point).
    """
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(to_numpy(scores) if scores is not None else [], dtype=np.float32).reshape(-1)
    if scores.size != masks.shape[0]:
        scores = np.ones((masks.shape[0],), dtype=np.float32)
    negatives = [point for _, _, point, label in instances if not label]
    keep = [index for index, mask in enumerate(masks)
            if not any(contains_point(mask, point) for point in negatives)]
    kept_masks = [masks[index] for index in keep]
    kept_scores = [float(scores[index]) for index in keep]
    for mask, score, point, label in instances:
        if not label or mask is None:
            continue
        if any(contains_point(existing, point) for existing in kept_masks):
            continue
        if any(contains_point(mask, negative) for negative in negatives):
            continue
        kept_masks.append(mask)
        kept_scores.append(score)
    height, width = masks.shape[-2:]
    if not kept_masks:
        return np.zeros((0, height, width), dtype=bool), np.zeros((0,), dtype=np.float32)
    return np.stack(kept_masks).astype(bool), np.asarray(kept_scores, dtype=np.float32)
