import json

import pytest

from scripts.evaluate_adaptive_rollout_chunk_aligned import (
    build_parser,
    build_report,
    evaluate_chunk_aligned,
    validate_cli_identity,
    write_chunk_aligned_outputs,
)


def _formal_literal_rows():
    return [
        {
            "episode": episode,
            "future_latent_step": step,
            "uncertainty_latent": uncertainty,
            "uncertainty_rgb": float(step) / 100.0,
            "error_rgb": float(step),
            **{f"error_seed_{seed}": float(step) for seed in range(4)},
        }
        for episode in range(1064, 1080)
        for step, uncertainty in enumerate(
            [0.01, 0.01, 0.01, 0.04, 0.04], start=1
        )
    ]


def test_previous_policy_recomputes_to_full_completed_generation():
    previous = {
        "adaptive": {
            "smoothed_hysteretic": {
                "loeo": {"generated_coverage": 0.9375},
                "deployment_threshold": 0.020729146897792816,
            }
        }
    }

    result = evaluate_chunk_aligned(
        _formal_literal_rows(), {"code_commit": "abc"}, previous
    )

    correction = result["summary"]["correction"]
    assert correction["previous_idealized_generated_coverage"] == 0.9375
    assert correction["previous_chunk_completed_generated_coverage"] == 1.0
    assert correction["previous_online_authorization_superseded"] is True


def test_chunk_writer_emits_exact_output_set(tmp_path):
    result = {
        "summary": {"online_authorized": False},
        "decisions": [{"episode": 1064, "policy": "threshold"}],
        "curve": [{"policy": "threshold", "tau": 0.1}],
        "folds": [{"held_out_episode": 1064, "tau": 0.1}],
        "costs": [{"policy": "threshold", "completed_k4_member_steps": 20}],
        "counterexamples": [
            {"episode": 1064, "reason": "no complete chunk avoided"}
        ],
        "report": "CHUNK-ALIGNED FAIL: ONLINE NOT AUTHORIZED\n",
    }

    write_chunk_aligned_outputs(tmp_path / "out", result, overwrite=False)

    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "adaptive_rollout_summary.json",
        "chunk_costs.csv",
        "counterexamples.csv",
        "loeo_folds.csv",
        "policy_curve.csv",
        "policy_decisions.csv",
        "report.md",
    ]
    saved = json.loads(
        (tmp_path / "out/adaptive_rollout_summary.json").read_text()
    )
    assert saved["online_authorized"] is False


def test_cli_requires_concrete_code_commit():
    args = build_parser().parse_args(
        [
            "--metrics_csv",
            "metrics.csv",
            "--correlation_summary",
            "summary.json",
            "--sampling_manifest",
            "sampling_manifest.json",
            "--source_archive",
            "source.tar.gz",
            "--previous_summary",
            "previous.json",
            "--code_commit",
            "unknown",
            "--output_dir",
            "out",
        ]
    )

    with pytest.raises(ValueError, match="concrete code commit"):
        validate_cli_identity(args)


def test_report_authorizes_only_a_passing_latent_policy():
    report = build_report(
        {
            "online_authorized": False,
            "adaptive": {
                "threshold": {"gate": {"passed": False}},
                "smoothed_hysteretic": {"gate": {"passed": False}},
            },
        }
    )

    assert report.startswith("CHUNK-ALIGNED FAIL: ONLINE NOT AUTHORIZED")
