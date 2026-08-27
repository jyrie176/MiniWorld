import pytest
import numpy as np

from miniworld.rollout_policy import (
    ChunkLayout,
    Decision,
    aggregate_policy_results,
    build_chunk_layout,
    chunk_aligned_smoothed_hysteretic_policy,
    chunk_aligned_threshold_policy,
    choose_operating_point,
    evaluate_offline_gate,
    fixed_policy,
    matched_fixed_baseline,
    run_loeo,
    score_policy_trace,
    smoothed_hysteretic_policy,
    threshold_policy,
    threshold_candidates,
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


def test_threshold_candidates_use_only_training_values():
    candidates = threshold_candidates([0.1, 0.2, 0.3])

    assert 99.0 not in candidates
    assert candidates[0] < 0.1 and candidates[-1] > 0.3


def test_operating_point_prefers_lower_error_then_coverage_then_higher_tau():
    candidates = [
        {"tau": 0.6, "coverage": 0.8, "retained_rgb_mae": 2.0},
        {"tau": 0.8, "coverage": 0.7, "retained_rgb_mae": 1.0},
        {"tau": 0.7, "coverage": 0.8, "retained_rgb_mae": 2.0},
        {"tau": 0.5, "coverage": 0.9, "retained_rgb_mae": 2.0},
    ]

    selected = choose_operating_point(candidates, target_coverage=0.8)

    assert selected["retained_rgb_mae"] == pytest.approx(2.0)
    assert selected["coverage"] == pytest.approx(0.9)
    assert selected["tau"] == pytest.approx(0.5)


def test_loeo_never_passes_held_out_uncertainty_to_candidate_builder():
    rows = []
    for episode in range(1064, 1080):
        for step in range(1, 6):
            rows.append(
                {
                    "episode": episode,
                    "future_latent_step": step,
                    "uncertainty_latent": float(episode * 10 + step),
                    "error_rgb": float(step),
                    **{f"error_seed_{index}": float(step) for index in range(4)},
                }
            )

    folds, _ = run_loeo(rows, "threshold")

    for fold in folds:
        held_out = fold["held_out_episode"]
        assert held_out not in fold["candidate_source_episodes"]
        held_out_values = {float(held_out * 10 + step) for step in range(1, 6)}
        assert held_out_values.isdisjoint(fold["threshold_candidates"])


def test_gate_requires_nine_episode_wins_and_tail_bound():
    adaptive = {"coverage": 0.8, "retained_rgb_mae": 4.0, "p90_episode_error": 6.0}
    fixed = {"retained_rgb_mae": 4.1, "p90_episode_error": 6.05}

    gate = evaluate_offline_gate(adaptive, fixed, [-0.1] * 9 + [0.1] * 7)

    assert gate == {
        "coverage_at_least_0_80": True,
        "retained_error_below_matched_fixed": True,
        "episode_wins_at_least_9_of_16": True,
        "p90_not_worse_by_more_than_0_10": True,
        "passed": True,
    }


def test_official_chunk_layout_completes_future_at_one_three_five():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)

    assert layout.completion_boundaries == (1, 3, 5)


def test_layout_keeps_partial_final_chunk():
    layout = build_chunk_layout(history_len=2, future_horizon=5, chunk_size=3)

    assert layout.completion_boundaries == (1, 4, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_len": 0, "future_horizon": 5, "chunk_size": 2},
        {"history_len": 1, "future_horizon": 0, "chunk_size": 2},
        {"history_len": 1, "future_horizon": 5, "chunk_size": 0},
    ],
)
def test_chunk_layout_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        build_chunk_layout(**kwargs)


def test_step_four_trigger_waits_for_chunk_five_and_retains_three():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    trace = chunk_aligned_smoothed_hysteretic_policy(
        [0.1, 0.1, 0.9, 0.9, 0.1], tau=0.45, layout=layout
    )

    assert trace.requested_observation_at == 4
    assert trace.generated_horizon == 5
    assert trace.retained_horizon == 3
    assert trace.decisions[-1] is Decision.REQUEST_OBSERVATION


def test_trigger_inside_middle_chunk_emits_after_boundary_three():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    trace = chunk_aligned_threshold_policy(
        [0.1, 0.9, 0.1, 0.1, 0.1], tau=0.5, layout=layout
    )

    assert trace.requested_observation_at == 2
    assert trace.generated_horizon == 3
    assert trace.retained_horizon == 1


def test_chunk_policy_rejects_uncertainty_length_different_from_layout():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)

    with pytest.raises(ValueError, match="future_horizon"):
        chunk_aligned_threshold_policy([0.1] * 4, tau=0.5, layout=layout)
