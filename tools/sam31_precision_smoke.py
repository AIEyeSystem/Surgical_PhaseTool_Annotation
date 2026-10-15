#!/usr/bin/env python
"""Run one SAM 3.1 image/text precision smoke and save machine-checkable metrics."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IVAS.segment_any.precision import normalize_precision, precision_to_dtype
from IVAS.segment_any.segment_any import SegAny


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--precision",
        required=True,
        choices=("original", "bfloat16", "float16", "float32"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    started = time.time()
    requested = normalize_precision(args.precision)
    metrics = {
        "status": "failed",
        "precision": {"requested": requested, "effective": None},
        "checks": {"all_required": False},
        "outputs": {"mask_count": 0, "nonempty_mask_count": 0},
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "gpu_memory": None,
    }
    segany = None
    try:
        checkpoint = Path(args.checkpoint).resolve()
        image_path = Path(args.image).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
        if not image_path.is_file():
            raise FileNotFoundError("Image not found: {}".format(image_path))
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this precision smoke.")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            metrics["environment"]["cuda_device"] = torch.cuda.get_device_name(0)

        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        segany = SegAny(str(checkpoint), precision=requested)
        if segany.model_source != "sam3.1":
            raise RuntimeError(
                "Expected SAM 3.1, got {!r}.".format(segany.model_source)
            )
        segany.set_image(image)
        masks, scores = segany.predict_with_text_prompt(args.prompt)
        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.ndim != 3:
            raise RuntimeError("Expected masks [N,H,W], got {}.".format(masks.shape))
        if scores.ndim != 1 or scores.shape[0] != masks.shape[0]:
            raise RuntimeError(
                "Scores {} do not match masks {}.".format(scores.shape, masks.shape)
            )

        expected_dtype = precision_to_dtype(requested)
        dtype_matches = expected_dtype is None or segany.model_dtype == expected_dtype
        nonempty = int(np.count_nonzero(masks.reshape(masks.shape[0], -1).any(axis=1)))
        finite_scores = bool(np.isfinite(scores).all())
        all_required = bool(dtype_matches and finite_scores and nonempty > 0)
        if torch.cuda.is_available():
            metrics["gpu_memory"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        metrics.update(
            {
                "status": "passed" if all_required else "failed",
                "precision": {
                    "requested": requested,
                    "effective": _dtype_name(segany.model_dtype),
                    "explicit_dtype_matches": bool(dtype_matches),
                },
                "checks": {
                    "all_required": all_required,
                    "scores_finite": finite_scores,
                    "model_source_sam31": True,
                },
                "inputs": {
                    "checkpoint": str(checkpoint),
                    "image": str(image_path),
                    "image_shape": list(image.shape),
                    "prompt": args.prompt,
                },
                "outputs": {
                    "mask_count": int(masks.shape[0]),
                    "nonempty_mask_count": nonempty,
                    "score_count": int(scores.shape[0]),
                    "max_score": float(scores.max()) if scores.size else None,
                },
            }
        )
    except Exception as error:
        metrics["error"] = "{}: {}".format(type(error).__name__, error)
    finally:
        if segany is not None:
            segany.release()
        metrics["duration_seconds"] = round(time.time() - started, 3)
        _write_json_atomic(args.output, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if metrics["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
