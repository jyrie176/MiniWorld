"""WAN2.2 VAE loading, encode/decode, and streaming decode helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.distributed import get_rank

from miniworld.vae.wan22_vae import Wan2_2_VAE


def is_main_process() -> bool:
    """Return whether the current process should print user-facing logs."""
    return int(os.environ.get("RANK", "0")) == 0


def print0(message: str) -> None:
    """Print only from rank 0."""
    if is_main_process():
        print(message, flush=True)


def get_rank_id() -> int:
    """Return distributed rank, or 0 when torch.distributed is inactive."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(get_rank())
    return 0


def _get_arg(args: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def load_wan22_vae(args: Mapping[str, Any] | object) -> Wan2_2_VAE:
    """Load a frozen WAN2.2 VAE on the current CUDA rank."""
    checkpoint = _get_arg(args, "vae_checkpoint")
    if checkpoint is None:
        raise ValueError("vae_checkpoint is required")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"WAN2.2 VAE checkpoint not found: {checkpoint_path}")

    device = torch.device(f"cuda:{get_rank_id()}" if torch.cuda.is_available() else "cpu")
    vae = Wan2_2_VAE(vae_pth=os.fspath(checkpoint_path), device=device)
    vae.model.requires_grad_(False)
    vae.model.eval()
    print0(f"WAN2.2 VAE parameters: {sum(p.numel() for p in vae.model.parameters()):,}")
    return vae


def vae_encode(vae: Wan2_2_VAE, video: torch.Tensor) -> torch.Tensor:
    """Encode RGB video in ``[-1, 1]`` to WAN2.2 latents."""
    total_frames = video.shape[2]
    target_frames = ((total_frames - 1) // 4) * 4 + 1
    return vae.encode(video[:, :, :target_frames])


@torch.no_grad()
def vae_decode(vae: Wan2_2_VAE, latents: torch.Tensor) -> torch.Tensor:
    """Decode WAN2.2 latents to RGB video in ``[-1, 1]``."""
    return vae.decode(latents)


class StreamingVAEDecoder:
    """Causal streaming WAN2.2 decode session."""

    def __init__(self, vae: Wan2_2_VAE) -> None:
        self.vae = vae
        self._active = False

    def begin(self) -> None:
        self.vae.decode_stream_begin()
        self._active = True

    def step(self, latents_chunk: torch.Tensor) -> torch.Tensor:
        """Decode one latent chunk to RGB frames."""
        if not self._active:
            raise RuntimeError("StreamingVAEDecoder.begin() must be called before step()")
        return self.vae.decode_stream_step(latents_chunk)

    def end(self) -> None:
        if self._active:
            self.vae.decode_stream_end()
            self._active = False

    def decode_all(self, latents: torch.Tensor) -> torch.Tensor:
        """Stream-decode a complete latent tensor."""
        self.begin()
        try:
            return self.step(latents)
        finally:
            self.end()

