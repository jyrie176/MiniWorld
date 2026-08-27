import pytest

from miniworld.rollout_policy import (
    Decision,
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
