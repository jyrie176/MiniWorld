import pytest
import numpy as np

from miniworld.rollout_policy import (
    Decision,
    aggregate_policy_results,
    fixed_policy,
    matched_fixed_baseline,
    score_policy_trace,
    smoothed_hysteretic_policy,
    threshold_policy,
)


def test_threshold_trigger_discards_triggering_step_but_counts_generated_work():
    trace = threshold_policy([0.1, 0.7, 0.2, 0.1, 0.1], tau=0.5)

    assert trace.decisions == (Decision.CONTINUE, Decision.REQUEST_OBSERVATION)
    assert trace.retained_horizon == 1
    assert trace.generated_horizon == 2
    assert trace.requested_observation_at == 2


def test_threshold_without_trigger_terminates_at_maximum():
    trace = threshold_policy([0.1] * 5, tau=0.5)

    assert trace.decisions[-1] is Decision.TERMINATE
    assert trace.retained_horizon == trace.generated_horizon == 5
    assert trace.requested_observation_at is None


def test_smoothed_policy_requires_two_consecutive_exceedances():
    trace = smoothed_hysteretic_policy([0.1, 0.9, 0.1, 0.9, 0.9], tau=0.45)

    assert trace.requested_observation_at == 5
    assert trace.retained_horizon == 4
    assert trace.generated_horizon == 5


@pytest.mark.parametrize(
    "values", [[0.1] * 4, [0.1, float("nan"), 0.1, 0.1, 0.1]]
)
def test_policy_rejects_wrong_length_or_nonfinite_uncertainty(values):
    with pytest.raises(ValueError):
        threshold_policy(values, tau=0.5)


def test_score_trace_separates_retained_and_discarded_error():
    trace = fixed_policy(2)
    result = score_policy_trace(
        1064,
        trace,
        [1.0, 3.0, 10.0, 20.0, 30.0],
        np.array(
            [[1.0, 3.0, 10.0, 20.0, 30.0], [2.0, 4.0, 12.0, 22.0, 32.0]]
        ),
    )

    assert result.retained_error_numerator == 4.0
    assert result.retained_count == 2
    assert result.discarded_error_numerator == 60.0
    assert result.discarded_count == 3
    np.testing.assert_allclose(result.per_seed_retained_error, [2.0, 3.0])


def test_aggregate_divides_total_numerator_by_total_retained_steps():
    short = score_policy_trace(
        1064, fixed_policy(1), [10.0, 0.0, 0.0, 0.0, 0.0], np.ones((2, 5))
    )
    long = score_policy_trace(
        1065, fixed_policy(5), [2.0] * 5, np.full((2, 5), 2.0)
    )

    aggregate = aggregate_policy_results([short, long], high_error_cutoff=5.0)

    assert aggregate["retained_rgb_mae"] == pytest.approx(20.0 / 6.0)
    assert aggregate["coverage"] == pytest.approx(0.6)
    assert aggregate["mean_retained_horizon"] == pytest.approx(3.0)


def test_matched_fixed_baseline_interpolates_expected_numerators():
    rows = [
        {"episode": 1064, "future_latent_step": step, "error_rgb": error}
        for step, error in enumerate([1, 2, 3, 4, 5], start=1)
    ] + [
        {"episode": 1065, "future_latent_step": step, "error_rgb": error}
        for step, error in enumerate([2, 4, 6, 8, 10], start=1)
    ]

    result = matched_fixed_baseline(rows, mean_horizon=3.5)

    assert result["mean_horizon"] == pytest.approx(3.5)
    assert result["coverage"] == pytest.approx(0.7)
    assert result["retained_rgb_mae"] == pytest.approx(12.0 / 3.5)
