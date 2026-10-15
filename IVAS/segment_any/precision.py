# -*- coding: utf-8 -*-
"""Precision helpers shared by local Segment Anything integrations.

The legacy application exposed a boolean ``use_bfloat16`` switch.  SAM 3.1
checkpoints are also distributed or converted as FP16 and BF16 weights, so a
boolean is no longer sufficient.  This module keeps the compatibility mapping
in one place and, importantly, leaves complex RoPE buffers complex when model
weights are converted.
"""

from contextlib import nullcontext
from typing import Optional, Union

import torch


PRECISION_ORIGINAL = "original"
PRECISION_BFLOAT16 = "bfloat16"
PRECISION_FLOAT16 = "float16"
PRECISION_FLOAT32 = "float32"
SUPPORTED_PRECISIONS = (
    PRECISION_ORIGINAL,
    PRECISION_BFLOAT16,
    PRECISION_FLOAT16,
    PRECISION_FLOAT32,
)

_ALIASES = {
    "auto": PRECISION_ORIGINAL,
    "checkpoint": PRECISION_ORIGINAL,
    "native": PRECISION_ORIGINAL,
    "original": PRECISION_ORIGINAL,
    "bf": PRECISION_BFLOAT16,
    "bf16": PRECISION_BFLOAT16,
    "bfloat16": PRECISION_BFLOAT16,
    "fp": PRECISION_FLOAT16,
    "fp16": PRECISION_FLOAT16,
    "float16": PRECISION_FLOAT16,
    "half": PRECISION_FLOAT16,
    "fp32": PRECISION_FLOAT32,
    "float": PRECISION_FLOAT32,
    "float32": PRECISION_FLOAT32,
}

_DTYPE_TO_PRECISION = {
    torch.bfloat16: PRECISION_BFLOAT16,
    torch.float16: PRECISION_FLOAT16,
    torch.float32: PRECISION_FLOAT32,
}

_PRECISION_TO_DTYPE = {
    PRECISION_BFLOAT16: torch.bfloat16,
    PRECISION_FLOAT16: torch.float16,
    PRECISION_FLOAT32: torch.float32,
}


def normalize_precision(
    precision: Optional[Union[str, torch.dtype]] = None,
    use_bfloat16: Optional[bool] = None,
) -> str:
    """Return a canonical precision name.

    ``use_bfloat16`` is accepted only as a legacy fallback.  An explicit
    ``precision`` always wins.
    """

    if isinstance(precision, torch.dtype):
        try:
            return _DTYPE_TO_PRECISION[precision]
        except KeyError as exc:
            raise ValueError("Unsupported floating dtype: {}".format(precision)) from exc

    if precision is None or str(precision).strip() == "":
        if use_bfloat16 is None:
            return PRECISION_ORIGINAL
        return PRECISION_BFLOAT16 if bool(use_bfloat16) else PRECISION_FLOAT32

    value = str(precision).strip().lower().replace("torch.", "")
    try:
        return _ALIASES[value]
    except KeyError as exc:
        raise ValueError(
            "Unsupported SAM precision {!r}; expected one of {}.".format(
                precision, ", ".join(SUPPORTED_PRECISIONS)
            )
        ) from exc


def precision_to_dtype(precision: Union[str, torch.dtype]) -> Optional[torch.dtype]:
    """Map a precision mode to a dtype; ``original`` intentionally maps to None."""

    mode = normalize_precision(precision)
    return _PRECISION_TO_DTYPE.get(mode)


def infer_floating_dtype(module: torch.nn.Module) -> torch.dtype:
    """Infer the dominant real floating dtype by tensor element count."""

    counts = {}
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    for tensor in tensors:
        if tensor.is_floating_point():
            counts[tensor.dtype] = counts.get(tensor.dtype, 0) + tensor.numel()
    if not counts:
        return torch.float32
    return max(counts.items(), key=lambda item: item[1])[0]


def cast_floating_tensors(
    module: torch.nn.Module, dtype: torch.dtype
) -> torch.nn.Module:
    """Cast floating parameters/buffers while preserving integer and complex data.

    ``Module.to(dtype=...)`` also casts complex buffers to a real dtype.  SAM
    3.1 stores complex64 RoPE tables, so that normal conversion would discard
    their imaginary component.  Updating registered tensors explicitly avoids
    that corruption.
    """

    if dtype not in _DTYPE_TO_PRECISION:
        raise ValueError("Unsupported target dtype: {}".format(dtype))

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)
                if parameter.grad is not None and parameter.grad.is_floating_point():
                    parameter.grad.data = parameter.grad.data.to(dtype=dtype)

        for child in module.modules():
            for name, buffer in tuple(child._buffers.items()):
                if (
                    buffer is not None
                    and buffer.is_floating_point()
                    and buffer.dtype != dtype
                ):
                    child._buffers[name] = buffer.to(dtype=dtype)
    return module


def apply_precision(
    module: torch.nn.Module, precision: Union[str, torch.dtype]
) -> torch.dtype:
    """Apply a precision policy and return the effective floating dtype.

    ``original`` preserves the tensors exactly as loaded and only reports the
    checkpoint's dominant real floating dtype. Explicit modes normalize all
    real floating tensors while complex RoPE buffers remain unchanged.
    """

    mode = normalize_precision(precision)
    target_dtype = precision_to_dtype(mode)
    if target_dtype is None:
        checkpoint_dtype = getattr(module, "_checkpoint_dtype", None)
        return (
            checkpoint_dtype
            if checkpoint_dtype in _DTYPE_TO_PRECISION
            else infer_floating_dtype(module)
        )
    cast_floating_tensors(module, target_dtype)
    return target_dtype


def autocast_context(device: Union[str, torch.device], dtype: torch.dtype):
    """Return an enabled autocast context only for supported device/dtype pairs."""

    device_type = torch.device(device).type
    enabled = (
        device_type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    ) or (device_type == "cpu" and dtype == torch.bfloat16)
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


def precision_display_name(precision: Union[str, torch.dtype]) -> str:
    """Human-readable precision label used by the settings UI."""

    mode = normalize_precision(precision)
    return {
        PRECISION_ORIGINAL: "Original checkpoint dtype",
        PRECISION_BFLOAT16: "BFloat16 (BF16)",
        PRECISION_FLOAT16: "Float16 (FP16)",
        PRECISION_FLOAT32: "Float32 (FP32)",
    }[mode]
