# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

from .model_builder import (
    build_sam3_image_model,
    build_sam3_video_model,
    build_sam3_1_image_model,
    build_sam3_1_video_model,
)
from .model.sam3_image_processor import Sam3Processor
from typing import Optional
import os
import threading

import torch
import numpy as np
from torchvision.transforms import v2
from PIL import Image
from ..precision import autocast_context, infer_floating_dtype


bpe_path = os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz")

def build_sam3(checkpoint):
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    return model

def build_sam3_video(checkpoint):
    model = build_sam3_video_model(
        checkpoint_path=checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    return predictor

def build_sam3_1(checkpoint, precision="original"):
    model = build_sam3_1_image_model(
        checkpoint_path=checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False,
        precision=precision,
    )
    return model

def build_sam3_1_video(checkpoint):
    model = build_sam3_1_video_model(
        checkpoint_path=checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False,
    )
    return model

sam_model_registry = {
    "sam3": build_sam3,
    "sam3_video": build_sam3_video,
    "sam3.1": build_sam3_1,
    "sam3.1_video": build_sam3_1_video,
}


class Sam3Predictor:
    def __init__(self, model, resolution=1008, confidence_threshold=0.5):
        self.model = model
        self.device = model.device
        self._sam3_processor = Sam3Processor(model, resolution, self.device, confidence_threshold)

    def _transforms(self, image):
        image = v2.functional.to_image(image).to(self.model.device)
        image = self._sam3_processor.transform(image).unsqueeze(0)
        return image

    @torch.inference_mode()
    def encode(self, image):
        _orig_hw = tuple(image.shape[:2])
        image = self._transforms(image)

        backbone_out = self.model.backbone.forward_image(image)
        sam2_backbone_out = backbone_out["sam2_backbone_out"]
        sam2_backbone_out["backbone_fpn"][0] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        )
        _, vision_feats, _, _ = (
            self.model.inst_interactive_predictor.model._prepare_backbone_features(
                backbone_out["sam2_backbone_out"]
            )
        )
        vision_feats[-1] = (
                vision_feats[-1] + self.model.inst_interactive_predictor.model.no_mem_embed
        )
        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(
                vision_feats[::-1],
                self.model.inst_interactive_predictor._bb_feat_sizes[::-1]
            )
                ][::-1]
        _features = {
            "image_embed": feats[-1],
            "high_res_feats": tuple(feats[:-1]),
        }
        return _features, _orig_hw

    def set_image(self, image: np.ndarray):
        _features, _orig_hw = self.encode(image)
        self.model.inst_interactive_predictor._is_image_set = True
        self.model.inst_interactive_predictor._orig_hw = [_orig_hw]
        self.model.inst_interactive_predictor._features = _features

    def reset_image(self):
        self.model.inst_interactive_predictor._is_image_set = False
        self.model.inst_interactive_predictor._orig_hw = None
        self.model.inst_interactive_predictor._features = None

    def predict(self,
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = True,
                return_logits: bool = False,
                normalize_coords=True,
                ):
        masks, scores, logits  = self.model.inst_interactive_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )
        return masks, scores, logits

    def predict_with_text_prompt(self, image: Image, prompt:str):
        inference_state = self._sam3_processor.set_image(image)
        self._sam3_processor.reset_all_prompts(inference_state)
        inference_state = self._sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = inference_state["masks"]
        masks = masks.squeeze(1).cpu().numpy()
        scores = inference_state["scores"]
        return masks, scores

    def predict_with_visual_prompt(self, image: Image.Image, boxes: list, labels: list):
        # [xmin ymin, xmax, ymax] to [cx, cy, w, h]
        boxes = [[(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
                      for xmin, ymin, xmax, ymax in boxes]
        # normalize
        height, width = image.height, image.width
        boxes = [[cx / width, cy / height, w / width, h / height] for cx, cy, w, h in boxes]

        inference_state = self._sam3_processor.set_image(image)
        self._sam3_processor.reset_all_prompts(inference_state)

        for box, label in zip(boxes, labels):
            inference_state = self._sam3_processor.add_geometric_prompt(state=inference_state, box=box, label=label)

        masks = inference_state["masks"]
        masks = masks.squeeze(1).cpu().numpy()
        scores = inference_state["scores"]
        return masks, scores


class Sam31Predictor:
    """Interactive predictor for SAM 3.1 (multiplex) models.

    SAM 3.1 does not ship the SAM2-style instance-interactive head, so point and
    box prompts are routed through the grounding path (find stage + geometric
    prompt) instead.
    """

    _PROMPT_OUTPUT_KEYS = (
        "geometric_prompt",
        "boxes",
        "masks",
        "masks_logits",
        "scores",
    )

    def __init__(
        self, model, resolution=1008, confidence_threshold=0.5, lock=None
    ):
        self.model = model
        self.device = model.device
        self.model_dtype = infer_floating_dtype(model)
        self._lock = lock or threading.RLock()
        self._sam3_processor = Sam3Processor(
            model, resolution, self.device, confidence_threshold
        )
        self._cached_state = None
        self._state = {}

    @classmethod
    def _copy_state_for_prompt(cls, state):
        work = dict(state)
        if isinstance(work.get("backbone_out"), dict):
            work["backbone_out"] = dict(work["backbone_out"])
        for key in cls._PROMPT_OUTPUT_KEYS:
            work.pop(key, None)
        return work

    @staticmethod
    def _empty_outputs(height, width):
        return (
            np.empty((0, height, width), dtype=bool),
            np.empty((0,), dtype=np.float32),
            np.empty((0, height, width), dtype=np.float32),
        )

    @staticmethod
    def _state_outputs(state, height, width):
        masks = state.get("masks")
        scores = state.get("scores")
        logits = state.get("masks_logits")
        if masks is None or scores is None:
            return Sam31Predictor._empty_outputs(height, width)
        masks = masks.detach().squeeze(1).cpu().numpy().astype(bool, copy=False)
        scores = scores.detach().cpu().float().numpy()
        if logits is None:
            logits = masks.astype(np.float32, copy=False)
        else:
            logits = logits.detach().squeeze(1).cpu().float().numpy()
        return masks, scores, logits

    @torch.inference_mode()
    def set_image(self, image: np.ndarray):
        self.encode(image)

    @torch.inference_mode()
    def encode(self, image: np.ndarray):
        """Encode an image and cache the grounding state.

        SAM 3.1 keeps the whole grounding state (backbone outputs) in memory,
        so this mirrors `set_image`; the returned tuple has the same shape as
        Sam3Predictor.encode (features, orig_hw).
        """
        with self._lock, autocast_context(self.device, self.model_dtype):
            state = self._sam3_processor.set_image(image, {})
            self._cached_state = state
            self._state = self._copy_state_for_prompt(state)
            original_size = (
                state["original_height"],
                state["original_width"],
            )
            return state, original_size

    def set_cached_state(self, state):
        if not isinstance(state, dict) or "backbone_out" not in state:
            raise ValueError("Invalid SAM 3.1 cached image state.")
        with self._lock:
            self._cached_state = state
            self._state = self._copy_state_for_prompt(state)

    def reset_image(self):
        with self._lock:
            self._cached_state = None
            self._state = {}

    @torch.inference_mode()
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
        text_prompt: Optional[str] = None,
    ):
        with self._lock, autocast_context(self.device, self.model_dtype):
            if self._cached_state is None:
                raise RuntimeError("set_image must be called before predict")
            state = self._copy_state_for_prompt(self._cached_state)
            img_h = state["original_height"]
            img_w = state["original_width"]

            if text_prompt is not None and text_prompt.strip():
                state = self._sam3_processor.set_text_prompt(
                    state=state, prompt=text_prompt.strip()
                )
            if point_coords is not None and len(point_coords) > 0:
                if point_labels is None:
                    raise ValueError("point_labels are required with point_coords")
                points = np.asarray(point_coords, dtype=np.float64).reshape(-1, 2)
                labels = np.asarray(point_labels).reshape(-1)
                if len(points) != len(labels):
                    raise ValueError("point_coords and point_labels must match")
                for (x, y), label in zip(points, labels):
                    point = [x, y]
                    if normalize_coords:
                        point = [x / img_w, y / img_h]
                    state = self._sam3_processor.add_point_prompt(
                        state=state, point=point, label=bool(label)
                    )
            if box is not None:
                xmin, ymin, xmax, ymax = np.asarray(box).reshape(-1)[:4]
                if normalize_coords:
                    box_prompt = [
                        (xmin + xmax) / 2 / img_w,
                        (ymin + ymax) / 2 / img_h,
                        (xmax - xmin) / img_w,
                        (ymax - ymin) / img_h,
                    ]
                else:
                    box_prompt = [
                        (xmin + xmax) / 2,
                        (ymin + ymax) / 2,
                        xmax - xmin,
                        ymax - ymin,
                    ]
                state = self._sam3_processor.add_geometric_prompt(
                    state=state, box=box_prompt, label=True
                )

            self._state = state
            masks, scores, logits = self._state_outputs(state, img_h, img_w)
            if masks.shape[0] == 0:
                return (
                    np.zeros((1, img_h, img_w), dtype=bool),
                    np.zeros((1,), dtype=np.float32),
                    np.zeros((1, img_h, img_w), dtype=np.float32),
                )
            if masks.shape[0] > 1:
                best = int(np.argmax(scores))
                masks = masks[best : best + 1]
                scores = scores[best : best + 1]
                logits = logits[best : best + 1]
            return masks, scores, logits

    @torch.inference_mode()
    def predict_with_text_prompt(self, image=None, prompt=None):
        if prompt is None and isinstance(image, str):
            prompt, image = image, None
        prompt = (prompt or "").strip()
        with self._lock, autocast_context(self.device, self.model_dtype):
            if image is not None:
                base_state = self._sam3_processor.set_image(image, {})
            else:
                base_state = self._cached_state
            if base_state is None:
                raise RuntimeError("set_image must be called before text prediction")
            height = base_state["original_height"]
            width = base_state["original_width"]
            if not prompt:
                masks, scores, _ = self._empty_outputs(height, width)
                return masks, scores
            state = self._copy_state_for_prompt(base_state)
            state = self._sam3_processor.set_text_prompt(
                state=state, prompt=prompt
            )
            masks, scores, _ = self._state_outputs(state, height, width)
            return masks, scores

    def predict_with_text_prompts(self, prompts):
        with self._lock:
            return [
                self.predict_with_text_prompt(prompt=prompt)
                for prompt in prompts
            ]

    @torch.inference_mode()
    def predict_with_visual_prompt(self, image, boxes: list, labels: list):
        if len(boxes) != len(labels):
            raise ValueError("boxes and labels must have the same length")
        with self._lock, autocast_context(self.device, self.model_dtype):
            if image is None:
                base_state = self._cached_state
            else:
                base_state = self._sam3_processor.set_image(image, {})
            if base_state is None:
                raise RuntimeError("set_image must be called before visual prediction")
            height = base_state["original_height"]
            width = base_state["original_width"]
            if not boxes:
                masks, scores, _ = self._empty_outputs(height, width)
                return masks, scores

            state = self._copy_state_for_prompt(base_state)
            for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
                box = [
                    (xmin + xmax) / 2 / width,
                    (ymin + ymax) / 2 / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                ]
                state = self._sam3_processor.add_geometric_prompt(
                    state=state, box=box, label=bool(label)
                )
            masks, scores, _ = self._state_outputs(state, height, width)
            return masks, scores
