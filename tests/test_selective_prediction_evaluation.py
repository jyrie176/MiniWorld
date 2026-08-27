import json

from scripts.evaluate_selective_prediction import (
    evaluate_selective_prediction,
    write_outputs,
)


def _formal_rows():
    rows = []
    for episode in range(1064, 1080):
        for step in range(1, 6):
            error = float(step + episode % 3)
            rows.append(
                {
                    "episode": episode,
                    "future_latent_step": step,
                    "uncertainty_latent": error / 10.0,
                    "uncertainty_rgb": error / 5.0,
                    "error_rgb": error,
                    **{f"error_seed_{seed}": error for seed in range(4)},
                }
            )
    return rows


def test_evaluator_compares_both_signals_without_ground_truth_leakage():
    result = evaluate_selective_prediction(
        _formal_rows(), source_identity={"sha256": "abc"}
    )

    assert set(result["summary"]["signals"]) == {
        "latent_disagreement",
        "rgb_disagreement",
    }
    for signal in result["summary"]["signals"].values():
        assert signal["aurc"] < signal["random_aurc"]
        assert signal["oracle_gap"] == 0.0
        assert set(signal["loeo"]) == {"1.0", "0.9", "0.8", "0.7"}
    assert "error_seed_0" not in json.dumps(result["summary"])


def test_writer_emits_reconstructible_artifacts(tmp_path):
    result = evaluate_selective_prediction(
        _formal_rows(), source_identity={"sha256": "abc"}
    )

    write_outputs(tmp_path, result)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "loeo_folds.csv",
        "report.md",
        "risk_coverage_curve.csv",
        "selective_prediction_summary.json",
    ]
    saved = json.loads(
        (tmp_path / "selective_prediction_summary.json").read_text()
    )
    assert saved["source"]["sha256"] == "abc"
    assert "SELECTIVE PREDICTION" in (tmp_path / "report.md").read_text()
