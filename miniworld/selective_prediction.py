"""Leakage-safe metrics for uncertainty-aware selective prediction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.7)


def _normalized_rows(rows: Iterable[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        uncertainty = float(row["uncertainty"])
        error = float(row["error"])
        if not math.isfinite(uncertainty) or not math.isfinite(error):
            raise ValueError("uncertainty and error values must be finite")
        if error < 0.0:
            raise ValueError("error values must be non-negative")
        normalized.append(
            {
                "episode": int(row["episode"]),
                "uncertainty": uncertainty,
                "error": error,
            }
        )
    if not normalized:
        raise ValueError("at least one prediction row is required")
    return normalized


def _risk_points(ordered: Sequence[dict]) -> list[dict]:
    total = len(ordered)
    cumulative = 0.0
    points = []
    for retained, row in enumerate(ordered, start=1):
        cumulative += row["error"]
        points.append(
            {
                "retained_count": retained,
                "coverage": retained / total,
                "risk": cumulative / retained,
                "threshold": row["uncertainty"],
            }
        )
    return points


def _operating_point(ordered: Sequence[dict], coverage: float) -> dict:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("target coverage must lie in (0, 1]")
    retained_count = max(1, math.ceil(coverage * len(ordered)))
    retained = ordered[:retained_count]
    errors = np.asarray([row["error"] for row in retained], dtype=np.float64)
    return {
        "target_coverage": coverage,
        "realized_coverage": retained_count / len(ordered),
        "retained_count": retained_count,
        "rejected_count": len(ordered) - retained_count,
        "threshold": retained[-1]["uncertainty"],
        "mean_error": float(errors.mean()),
        "p90_error": float(np.percentile(errors, 90)),
        "worst_error": float(errors.max()),
    }


def risk_coverage_curve(
    rows: Iterable[dict], *, target_coverages: Sequence[float] = DEFAULT_COVERAGES
) -> dict:
    """Rank predictions by uncertainty and summarize selective risk."""
    normalized = _normalized_rows(rows)
    ordered = sorted(normalized, key=lambda row: (row["uncertainty"], row["episode"]))
    curve = _risk_points(ordered)
    oracle = _risk_points(
        sorted(normalized, key=lambda row: (row["error"], row["episode"]))
    )
    overall_risk = float(np.mean([row["error"] for row in normalized]))
    return {
        "curve": curve,
        "aurc": float(np.mean([point["risk"] for point in curve])),
        "random_aurc": overall_risk,
        "oracle_aurc": float(np.mean([point["risk"] for point in oracle])),
        "operating_points": [
            _operating_point(ordered, coverage) for coverage in target_coverages
        ],
    }


def loeo_selective_metrics(
    rows: Iterable[dict], *, target_coverage: float
) -> dict:
    """Fit a quantile threshold outside each held-out episode and aggregate it."""
    normalized = _normalized_rows(rows)
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target coverage must lie in (0, 1]")
    episodes = sorted({row["episode"] for row in normalized})
    if len(episodes) < 2:
        raise ValueError("LOEO requires at least two episodes")

    folds = []
    retained_rows = []
    for held_out in episodes:
        training = [row for row in normalized if row["episode"] != held_out]
        held_rows = [row for row in normalized if row["episode"] == held_out]
        threshold = (
            math.inf
            if target_coverage == 1.0
            else float(
                np.quantile(
                    [row["uncertainty"] for row in training],
                    target_coverage,
                    method="higher",
                )
            )
        )
        retained = [row for row in held_rows if row["uncertainty"] <= threshold]
        retained_rows.extend(retained)
        folds.append(
            {
                "held_out_episode": held_out,
                "threshold": threshold,
                "threshold_source_episodes": sorted(
                    {row["episode"] for row in training}
                ),
                "retained_count": len(retained),
                "rejected_count": len(held_rows) - len(retained),
                "mean_error": (
                    float(np.mean([row["error"] for row in retained]))
                    if retained
                    else None
                ),
            }
        )

    errors = np.asarray([row["error"] for row in retained_rows], dtype=np.float64)
    return {
        "target_coverage": target_coverage,
        "realized_coverage": len(retained_rows) / len(normalized),
        "retained_count": len(retained_rows),
        "rejected_count": len(normalized) - len(retained_rows),
        "mean_error": float(errors.mean()) if len(errors) else None,
        "p90_error": float(np.percentile(errors, 90)) if len(errors) else None,
        "worst_error": float(errors.max()) if len(errors) else None,
        "folds": folds,
    }
