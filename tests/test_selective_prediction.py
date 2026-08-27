import pytest

from miniworld.selective_prediction import (
    loeo_selective_metrics,
    risk_coverage_curve,
)


def test_risk_coverage_orders_low_uncertainty_first_and_computes_aurc():
    rows = [
        {"episode": 1, "uncertainty": 0.30, "error": 9.0},
        {"episode": 2, "uncertainty": 0.10, "error": 1.0},
        {"episode": 3, "uncertainty": 0.20, "error": 2.0},
    ]

    result = risk_coverage_curve(rows)

    assert [point["risk"] for point in result["curve"]] == pytest.approx(
        [1.0, 1.5, 4.0]
    )
    assert result["aurc"] == pytest.approx((1.0 + 1.5 + 4.0) / 3.0)
    assert result["random_aurc"] == pytest.approx(4.0)
    assert result["oracle_aurc"] == pytest.approx(result["aurc"])


def test_risk_coverage_reports_requested_operating_points_and_tail_risk():
    rows = [
        {"episode": index, "uncertainty": float(index), "error": float(index)}
        for index in range(1, 11)
    ]

    result = risk_coverage_curve(rows, target_coverages=(1.0, 0.8))

    full, selective = result["operating_points"]
    assert full["retained_count"] == 10
    assert full["mean_error"] == pytest.approx(5.5)
    assert full["worst_error"] == pytest.approx(10.0)
    assert selective["retained_count"] == 8
    assert selective["mean_error"] == pytest.approx(4.5)
    assert selective["p90_error"] == pytest.approx(7.3)


def test_loeo_threshold_is_fit_without_held_out_episode():
    rows = []
    for episode in range(4):
        rows.extend(
            {
                "episode": episode,
                "uncertainty": float(episode * 10 + step),
                "error": float(step),
            }
            for step in range(1, 6)
        )

    result = loeo_selective_metrics(rows, target_coverage=0.8)

    assert len(result["folds"]) == 4
    for fold in result["folds"]:
        assert fold["held_out_episode"] not in fold["threshold_source_episodes"]
        assert fold["retained_count"] + fold["rejected_count"] == 5
    assert result["retained_count"] + result["rejected_count"] == 20


def test_selective_metrics_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        risk_coverage_curve(
            [{"episode": 1, "uncertainty": float("nan"), "error": 1.0}]
        )


def test_loeo_full_coverage_never_rejects_held_out_outlier():
    rows = [
        {"episode": 1, "uncertainty": 1.0, "error": 1.0},
        {"episode": 2, "uncertainty": 2.0, "error": 2.0},
        {"episode": 3, "uncertainty": 100.0, "error": 3.0},
    ]

    result = loeo_selective_metrics(rows, target_coverage=1.0)

    assert result["realized_coverage"] == 1.0
    assert result["rejected_count"] == 0
