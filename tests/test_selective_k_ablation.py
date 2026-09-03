import pytest

from scripts.evaluate_selective_k_ablation import (
    attach_subset_errors,
    member_subsets,
    summarize_k_ablation,
)


def _rows(scale, *, include_seeds=False):
    rows = []
    for episode in range(1064, 1080):
        for step in range(1, 6):
            error = float(step + episode % 2)
            row = {
                "episode": episode,
                "future_latent_step": step,
                "uncertainty_rgb": scale * error,
                "error_rgb": error,
            }
            if include_seeds:
                row.update(
                    {f"error_seed_{seed}": error + seed for seed in range(4)}
                )
            rows.append(row)
    return rows


def test_member_subsets_enumerates_all_pairs_without_cherry_picking():
    assert member_subsets(4, 2) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_summary_reports_k1_cost_floor_all_k2_pairs_and_k4():
    pair_rows = {
        pair: _rows(0.1 + index * 0.01)
        for index, pair in enumerate(member_subsets(4, 2))
    }

    summary = summarize_k_ablation(
        k4_rows=_rows(0.2, include_seeds=True), k2_rows_by_pair=pair_rows
    )

    assert summary["k1"]["sampling_cost_x"] == 1
    assert summary["k1"]["uncertainty_available"] is False
    assert summary["k2"]["sampling_cost_x"] == 2
    assert summary["k2"]["pair_count"] == 6
    assert len(summary["k2"]["pairs"]) == 6
    assert summary["k4"]["sampling_cost_x"] == 4
    assert summary["k4"]["loeo_80"]["target_coverage"] == pytest.approx(0.8)
    assert summary["k4"]["loeo_80"]["realized_coverage"] >= 0.8


def test_summary_rejects_missing_k2_pair():
    pairs = member_subsets(4, 2)
    with pytest.raises(ValueError, match="all six"):
        summarize_k_ablation(
            k4_rows=_rows(0.2, include_seeds=True),
            k2_rows_by_pair={pair: _rows(0.1) for pair in pairs[:-1]},
        )


def test_subset_error_is_mean_of_only_the_selected_members():
    source = _rows(0.2, include_seeds=True)
    uncertainty_rows = [
        {
            "episode": row["episode"],
            "future_latent_step": row["future_latent_step"],
            "uncertainty_rgb": 0.5,
        }
        for row in source
    ]

    rows = attach_subset_errors(uncertainty_rows, source, members=(1, 3))

    assert rows[0]["error_rgb"] == pytest.approx(
        (source[0]["error_seed_1"] + source[0]["error_seed_3"]) / 2
    )
