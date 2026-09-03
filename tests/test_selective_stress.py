import torch

from scripts.evaluate_selective_stress import (
    _pair_metrics,
    apply_common_brightness,
    apply_independent_noise,
    stress_seed,
)


def test_common_brightness_preserves_member_difference_without_clipping():
    ensemble = torch.tensor([[[[[40.0]]]], [[[[60.0]]]]])

    shifted = apply_common_brightness(ensemble, delta=20.0)

    torch.testing.assert_close(shifted[1] - shifted[0], ensemble[1] - ensemble[0])


def test_independent_noise_is_reconstructible_and_member_specific():
    ensemble = torch.full((2, 3, 2, 2, 1), 128.0)

    first = apply_independent_noise(ensemble, sigma=8.0, seed=123)
    second = apply_independent_noise(ensemble, sigma=8.0, seed=123)

    torch.testing.assert_close(first, second)
    assert not torch.equal(first[0], first[1])


def test_stress_seed_is_stable_and_changes_by_episode_pair_and_level():
    base = stress_seed(1064, (0, 1), 8)

    assert base == stress_seed(1064, (0, 1), 8)
    assert base != stress_seed(1065, (0, 1), 8)
    assert base != stress_seed(1064, (0, 2), 8)
    assert base != stress_seed(1064, (0, 1), 16)


def test_pair_metrics_reports_gain_against_its_own_error_scale():
    rows = []
    for episode in range(1064, 1080):
        for step in range(1, 6):
            value = float(step + (episode - 1064) / 100.0)
            rows.append(
                {
                    "episode": episode,
                    "uncertainty_rgb": value,
                    "error_rgb": value,
                }
            )

    metrics = _pair_metrics(rows)

    assert metrics["aurc_gain_vs_random"] > 0
    assert metrics["loeo80_error_reduction"] > 0
