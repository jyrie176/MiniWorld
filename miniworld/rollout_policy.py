"""Pure policy and accounting primitives for uncertainty-aware rollout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence


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
