"""Pure metrics for stochastic-rollout uncertainty evaluation."""

from __future__ import annotations

from itertools import combinations

import torch


def _validate_finite(tensor: torch.Tensor, *, ndim: int, name: str) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def future_rgb_blocks(num_latent_frames: int, num_rgb_frames: int) -> list[slice]:
    """Map future latent positions to four-frame RGB blocks."""
    future_latents = int(num_latent_frames) - 1
    future_rgb = int(num_rgb_frames) - 1
    if future_latents <= 0 or future_rgb != 4 * future_latents:
        raise ValueError(
            "expected one context frame and four RGB frames per future latent"
        )
    return [
        slice(1 + 4 * index, 1 + 4 * (index + 1))
        for index in range(future_latents)
    ]


def latent_population_variance(ensemble: torch.Tensor) -> torch.Tensor:
    """Return mean population variance per latent time step for ``(K,C,T,H,W)``."""
    _validate_finite(ensemble, ndim=5, name="latent ensemble")
    if ensemble.shape[0] < 2:
        raise ValueError("latent ensemble requires at least two members")
    return ensemble.float().var(dim=0, correction=0).mean(dim=(0, 2, 3))


def rgb_pairwise_disagreement(
    ensemble: torch.Tensor, blocks: list[slice]
) -> torch.Tensor:
    """Return mean absolute RGB disagreement per temporal block."""
    _validate_finite(ensemble, ndim=5, name="RGB ensemble")
    if ensemble.shape[0] < 2:
        raise ValueError("RGB ensemble requires at least two members")
    values = []
    for block in blocks:
        pair_values = [
            (ensemble[left, block].float() - ensemble[right, block].float())
            .abs()
            .mean()
            for left, right in combinations(range(ensemble.shape[0]), 2)
        ]
        values.append(torch.stack(pair_values).mean())
    return torch.stack(values)


def rgb_memberwise_mae(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    blocks: list[slice],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ensemble-mean and per-member RGB MAE for each temporal block."""
    _validate_finite(ensemble, ndim=5, name="RGB ensemble")
    _validate_finite(target, ndim=4, name="RGB target")
    if ensemble.shape[1:] != target.shape:
        raise ValueError(
            "RGB ensemble members and target must have identical (T,H,W,C) shapes"
        )
    per_member = torch.stack(
        [
            torch.stack(
                [
                    (member[block].float() - target[block].float()).abs().mean()
                    for block in blocks
                ]
            )
            for member in ensemble
        ]
    )
    return per_member.mean(dim=0), per_member
