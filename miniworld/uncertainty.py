"""Pure metrics for stochastic-rollout uncertainty evaluation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
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


def _validate_vectors(*vectors: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(vector, dtype=np.float64) for vector in vectors)
    if not arrays or any(array.ndim != 1 for array in arrays):
        raise ValueError("correlation inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1 or len(arrays[0]) < 2:
        raise ValueError("correlation inputs must have equal length of at least two")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("correlation inputs contain NaN or Inf")
    return arrays


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson correlation, or ``None`` for a constant vector."""
    x_array, y_array = _validate_vectors(x, y)
    if np.ptp(x_array) == 0.0 or np.ptp(y_array) == 0.0:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Spearman correlation with average ranks for ties."""
    x_array, y_array = _validate_vectors(x, y)
    return pearson_correlation(_average_ranks(x_array), _average_ranks(y_array))


def horizon_conditioned_spearman(
    x: np.ndarray,
    y: np.ndarray,
    horizons: np.ndarray,
) -> float | None:
    """Rank within each horizon before pooling and correlating the ranks."""
    x_array, y_array, horizon_array = _validate_vectors(x, y, horizons)
    x_ranks = np.empty_like(x_array)
    y_ranks = np.empty_like(y_array)
    for horizon in np.unique(horizon_array):
        mask = horizon_array == horizon
        x_ranks[mask] = _average_ranks(x_array[mask])
        y_ranks[mask] = _average_ranks(y_array[mask])
    return pearson_correlation(x_ranks, y_ranks)


def equal_count_bins(
    uncertainty: np.ndarray,
    error: np.ndarray,
    bins: int = 4,
) -> list[dict[str, float | int]]:
    """Sort by uncertainty and summarize stable, exhaustive equal-count bins."""
    uncertainty_array, error_array = _validate_vectors(uncertainty, error)
    if bins <= 0 or bins > len(uncertainty_array):
        raise ValueError("bins must be between one and the observation count")
    order = np.argsort(uncertainty_array, kind="mergesort")
    summaries = []
    for bin_index, indices in enumerate(np.array_split(order, bins), start=1):
        bin_uncertainty = uncertainty_array[indices]
        bin_error = error_array[indices]
        summaries.append(
            {
                "bin": bin_index,
                "count": int(len(indices)),
                "mean_uncertainty": float(bin_uncertainty.mean()),
                "min_uncertainty": float(bin_uncertainty.min()),
                "max_uncertainty": float(bin_uncertainty.max()),
                "mean_error": float(bin_error.mean()),
                "min_error": float(bin_error.min()),
                "max_error": float(bin_error.max()),
            }
        )
    return summaries
