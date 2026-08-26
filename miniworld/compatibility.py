"""Hardware compatibility policy for MiniWorld execution backends."""

from __future__ import annotations

import importlib.util
from typing import Literal, Optional, Tuple

import torch

AttentionBackend = Literal["auto", "sdpa", "flash"]
ResolvedAttentionBackend = Literal["sdpa", "flash"]
TrainingPrecision = Literal["no", "fp16", "bf16"]
SamplePrecision = Literal["auto", "fp16", "bf16", "fp32"]


def flash_attention_available() -> bool:
    """Return whether the optional FlashAttention package is discoverable."""
    return importlib.util.find_spec("flash_attn") is not None


def _format_capability(capability: Optional[Tuple[int, int]]) -> str:
    if capability is None:
        return "unknown"
    return f"{capability[0]}.{capability[1]}"


def resolve_attention_backend(
    requested: AttentionBackend,
    *,
    cuda_available: bool,
    capability: Optional[Tuple[int, int]],
    flash_available: bool,
) -> ResolvedAttentionBackend:
    """Resolve an attention request from explicit hardware/package facts."""
    if requested not in ("auto", "sdpa", "flash"):
        raise ValueError(f"Unknown attention backend: {requested!r}")
    if requested == "sdpa":
        return "sdpa"
    if requested == "auto":
        if cuda_available and capability is not None and capability >= (8, 0) and flash_available:
            return "flash"
        return "sdpa"

    if not cuda_available:
        detected = "CUDA unavailable"
    elif capability is None or capability < (8, 0):
        detected = f"detected CUDA capability {_format_capability(capability)}"
    elif not flash_available:
        detected = "FlashAttention package unavailable"
    else:
        return "flash"
    raise RuntimeError(
        f"FlashAttention was requested but is unsupported ({detected}); use sdpa instead."
    )


def resolve_training_dtype(precision: TrainingPrecision) -> torch.dtype:
    """Map a training precision option to its torch dtype."""
    dtypes = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[precision]
    except KeyError as exc:
        raise ValueError(f"Unknown training precision: {precision!r}") from exc


def resolve_sample_precision(
    requested: SamplePrecision,
    *,
    cuda_available: bool,
    capability: Optional[Tuple[int, int]],
) -> Tuple[torch.dtype, bool]:
    """Return the sampling dtype and whether CUDA autocast should be active."""
    if requested not in ("auto", "fp16", "bf16", "fp32"):
        raise ValueError(f"Unknown sample precision: {requested!r}")
    if requested == "auto":
        if not cuda_available:
            return torch.float32, False
        if capability is not None and capability >= (8, 0):
            return torch.bfloat16, True
        return torch.float16, True
    if requested == "fp32":
        return torch.float32, False
    if not cuda_available:
        raise RuntimeError(
            f"CPU sampling does not support requested {requested}; use fp32 instead."
        )
    dtype = torch.float16 if requested == "fp16" else torch.bfloat16
    return dtype, True
