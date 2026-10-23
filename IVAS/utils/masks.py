"""Shared mask conversion using immutable options, without widget access."""

import cv2
import numpy as np
from IVAS.configs import CONTOURMethod, CONTOURMode


def mask_to_polygons(mask, contour_mode=CONTOURMode.SAVE_MAX_ONLY,
                     contour_method=CONTOURMethod.SIMPLE, simplify=True):
    mask = np.asarray(mask)
    height, width = mask.shape[-2:]
    mask = (mask.reshape(height, width) > 0).astype(np.uint8) * 255
    method = {CONTOURMethod.SIMPLE: cv2.CHAIN_APPROX_SIMPLE,
              CONTOURMethod.TC89_KCOS: cv2.CHAIN_APPROX_TC89_KCOS,
              CONTOURMethod.NONE: cv2.CHAIN_APPROX_NONE}.get(contour_method, cv2.CHAIN_APPROX_SIMPLE)
    retrieval = cv2.RETR_CCOMP if contour_mode == CONTOURMode.SAVE_ALL else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(mask, retrieval, method)
    if contours and contour_mode == CONTOURMode.SAVE_MAX_ONLY:
        contours = [max(contours, key=cv2.contourArea)]
    if simplify:
        contours = [cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
                    for contour in contours]
    return contours, hierarchy
