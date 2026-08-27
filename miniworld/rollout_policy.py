"""Pure policy and accounting primitives for uncertainty-aware rollout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


class Decision(str, Enum):
    CONTINUE = "continue"
    REQUEST_OBSERVATION = "request_observation"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class PolicyTrace:
    decisions: tuple[Decision, ...]
    retained_horizon: int
    generated_horizon: int
    requested_observation_at: int | None


@dataclass(frozen=True)
class EpisodePolicyResult:
    episode: int
    trace: PolicyTrace
    retained_error_numerator: float
    retained_count: int
    discarded_error_numerator: float
    discarded_count: int
    per_seed_retained_error: tuple[float, ...]
    retained_step_errors: tuple[float, ...]


def _validate_horizons(h_min: int, h_max: int, available: int) -> None:
    if not 1 <= h_min <= h_max <= available:
        raise ValueError("horizons must satisfy 1 <= h_min <= h_max <= available")


def _validate_uncertainty(
    uncertainty: Sequence[float], tau: float, h_min: int, h_max: int
) -> tuple[float, ...]:
    values = tuple(float(value) for value in uncertainty)
    if len(values) != 5:
        raise ValueError("uncertainty must contain exactly five future steps")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("uncertainty values must be finite")
    if not math.isfinite(tau):
        raise ValueError("threshold must be finite")
    _validate_horizons(h_min, h_max, len(values))
    return values


def fixed_policy(
    horizon: int, *, h_min: int = 1, h_max: int = 5
) -> PolicyTrace:
    _validate_horizons(h_min, h_max, 5)
    if not h_min <= horizon <= h_max:
        raise ValueError("fixed horizon must lie between h_min and h_max")
    decisions = (Decision.CONTINUE,) * (horizon - 1) + (Decision.TERMINATE,)
    return PolicyTrace(decisions, horizon, horizon, None)


def _trace_from_trigger(trigger_step: int | None, h_max: int) -> PolicyTrace:
    if trigger_step is None:
        decisions = (Decision.CONTINUE,) * (h_max - 1) + (Decision.TERMINATE,)
        return PolicyTrace(decisions, h_max, h_max, None)
    decisions = (Decision.CONTINUE,) * (trigger_step - 1) + (
        Decision.REQUEST_OBSERVATION,
    )
    return PolicyTrace(
        decisions,
        max(1, trigger_step - 1),
        trigger_step,
        trigger_step,
    )


def threshold_policy(
    uncertainty: Sequence[float],
    tau: float,
    *,
    h_min: int = 1,
    h_max: int = 5,
) -> PolicyTrace:
    values = _validate_uncertainty(uncertainty, tau, h_min, h_max)
    trigger = next(
        (
            step
            for step, value in enumerate(values[:h_max], start=1)
            if step >= h_min and value > tau
        ),
        None,
    )
    return _trace_from_trigger(trigger, h_max)


def smoothed_hysteretic_policy(
    uncertainty: Sequence[float],
    tau: float,
    *,
    alpha: float = 0.5,
    consecutive: int = 2,
    h_min: int = 1,
    h_max: int = 5,
) -> PolicyTrace:
    values = _validate_uncertainty(uncertainty, tau, h_min, h_max)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    if consecutive < 1:
        raise ValueError("consecutive must be positive")

    ema = values[0]
    run_length = 0
    trigger = None
    for step, value in enumerate(values[:h_max], start=1):
        if step > 1:
            ema = alpha * value + (1.0 - alpha) * ema
        run_length = run_length + 1 if step >= h_min and ema > tau else 0
        if run_length >= consecutive:
            trigger = step
            break
    return _trace_from_trigger(trigger, h_max)


def score_policy_trace(
    episode: int,
    trace: PolicyTrace,
    step_errors: Sequence[float],
    per_seed_step_errors: np.ndarray,
) -> EpisodePolicyResult:
    errors = np.asarray(step_errors, dtype=np.float64)
    seed_errors = np.asarray(per_seed_step_errors, dtype=np.float64)
    if errors.shape != (5,) or seed_errors.ndim != 2 or seed_errors.shape[1] != 5:
        raise ValueError("errors must contain five ordered future steps")
    if seed_errors.shape[0] < 1:
        raise ValueError("at least one seed error row is required")
    if not np.isfinite(errors).all() or not np.isfinite(seed_errors).all():
        raise ValueError("errors must be finite")
    retained = trace.retained_horizon
    if not 1 <= retained <= 5:
        raise ValueError("trace retained horizon must lie in [1, 5]")
    retained_errors = errors[:retained]
    discarded_errors = errors[retained:]
    return EpisodePolicyResult(
        episode=int(episode),
        trace=trace,
        retained_error_numerator=float(retained_errors.sum()),
        retained_count=retained,
        discarded_error_numerator=float(discarded_errors.sum()),
        discarded_count=5 - retained,
        per_seed_retained_error=tuple(
            float(value) for value in seed_errors[:, :retained].mean(axis=1)
        ),
        retained_step_errors=tuple(float(value) for value in retained_errors),
    )


def aggregate_policy_results(
    results: Sequence[EpisodePolicyResult], *, high_error_cutoff: float
) -> dict[str, float | int]:
    if not results:
        raise ValueError("at least one episode result is required")
    if not math.isfinite(high_error_cutoff):
        raise ValueError("high_error_cutoff must be finite")
    retained_count = sum(result.retained_count for result in results)
    generated_count = sum(result.trace.generated_horizon for result in results)
    retained_numerator = sum(
        result.retained_error_numerator for result in results
    )
    retained_values = [
        value for result in results for value in result.retained_step_errors
    ]
    episode_errors = [
        result.retained_error_numerator / result.retained_count for result in results
    ]
    total_steps = len(results) * 5
    return {
        "episode_count": len(results),
        "retained_count": retained_count,
        "generated_count": generated_count,
        "retained_error_numerator": retained_numerator,
        "retained_rgb_mae": retained_numerator / retained_count,
        "coverage": retained_count / total_steps,
        "generated_coverage": generated_count / total_steps,
        "mean_retained_horizon": retained_count / len(results),
        "mean_generated_horizon": generated_count / len(results),
        "p90_episode_error": float(np.percentile(episode_errors, 90)),
        "high_error_retained_fraction": sum(
            value >= high_error_cutoff for value in retained_values
        )
        / retained_count,
    }


def matched_fixed_baseline(
    rows: Sequence[Mapping[str, float]], mean_horizon: float
) -> dict[str, object]:
    if not math.isfinite(mean_horizon) or not 1.0 <= mean_horizon <= 5.0:
        raise ValueError("mean_horizon must lie in [1, 5]")
    by_episode: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        episode = int(row["episode"])
        step = int(row["future_latent_step"])
        error = float(row["error_rgb"])
        if step in by_episode[episode]:
            raise ValueError("duplicate episode-step row")
        if not math.isfinite(error):
            raise ValueError("error_rgb must be finite")
        by_episode[episode][step] = error
    if not by_episode or any(set(values) != set(range(1, 6)) for values in by_episode.values()):
        raise ValueError("each episode must contain exactly five future steps")

    lower = int(math.floor(mean_horizon))
    upper = int(math.ceil(mean_horizon))
    upper_weight = mean_horizon - lower
    episode_numerators = {}
    episode_errors = {}
    for episode, step_map in sorted(by_episode.items()):
        lower_numerator = sum(step_map[step] for step in range(1, lower + 1))
        upper_numerator = sum(step_map[step] for step in range(1, upper + 1))
        expected_numerator = (
            (1.0 - upper_weight) * lower_numerator
            + upper_weight * upper_numerator
        )
        episode_numerators[episode] = expected_numerator
        episode_errors[episode] = expected_numerator / mean_horizon
    mean_numerator = sum(episode_numerators.values()) / len(episode_numerators)
    return {
        "mean_horizon": mean_horizon,
        "coverage": mean_horizon / 5.0,
        "retained_rgb_mae": mean_numerator / mean_horizon,
        "p90_episode_error": float(np.percentile(list(episode_errors.values()), 90)),
        "episode_errors": episode_errors,
    }
