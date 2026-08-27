import numpy as np
import pytest
import torch

from miniworld.uncertainty import (
    equal_count_bins,
    future_rgb_blocks,
    horizon_conditioned_spearman,
    latent_population_variance,
    pearson_correlation,
    rgb_memberwise_mae,
    rgb_pairwise_disagreement,
    spearman_correlation,
)


def test_future_rgb_blocks_excludes_context():
    assert future_rgb_blocks(6, 21) == [
        slice(1, 5),
        slice(5, 9),
        slice(9, 13),
        slice(13, 17),
        slice(17, 21),
    ]


def test_latent_population_variance():
    ensemble = torch.tensor([0.0, 2.0]).reshape(2, 1, 1, 1, 1)

    result = latent_population_variance(ensemble)

    torch.testing.assert_close(result, torch.tensor([1.0]))


def test_rgb_disagreement_and_memberwise_error():
    ensemble = torch.tensor([[0.0, 2.0], [2.0, 4.0]]).reshape(2, 2, 1, 1, 1)
    target = torch.tensor([1.0, 1.0]).reshape(2, 1, 1, 1)
    blocks = [slice(0, 2)]

    disagreement = rgb_pairwise_disagreement(ensemble, blocks)
    mean_error, per_seed = rgb_memberwise_mae(ensemble, target, blocks)

    torch.testing.assert_close(disagreement, torch.tensor([2.0]))
    torch.testing.assert_close(mean_error, torch.tensor([1.5]))
    torch.testing.assert_close(per_seed, torch.tensor([[1.0], [2.0]]))


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 3),
        torch.full((2, 1, 2, 1, 1), float("nan")),
    ],
)
def test_latent_metric_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        latent_population_variance(bad)


def test_temporal_mapping_rejects_wrong_rgb_length():
    with pytest.raises(ValueError, match="four RGB frames"):
        future_rgb_blocks(6, 20)


def test_correlations_and_average_tie_ranks():
    tied = np.array([1.0, 2.0, 2.0, 4.0])
    ordered = np.array([1.0, 2.0, 3.0, 4.0])

    assert pearson_correlation(ordered, ordered) == pytest.approx(1.0)
    assert spearman_correlation(tied, ordered) == pytest.approx(
        0.9486832980505139
    )


def test_constant_correlation_is_undefined():
    assert pearson_correlation(np.ones(4), np.arange(4)) is None
    assert spearman_correlation(np.ones(4), np.arange(4)) is None


def test_horizon_conditioning_removes_between_horizon_effect():
    horizon = np.array([1, 1, 2, 2])
    uncertainty = np.array([1.0, 2.0, 10.0, 9.0])
    error = np.array([1.0, 2.0, 9.0, 10.0])

    result = horizon_conditioned_spearman(uncertainty, error, horizon)

    assert result == pytest.approx(0.0)


def test_equal_count_bins_preserve_every_observation():
    rows = equal_count_bins(np.arange(10.0), np.arange(10.0) ** 2, bins=4)

    assert sum(row["count"] for row in rows) == 10
    assert rows[-1]["mean_error"] > rows[0]["mean_error"]
