import csv
import json

import pytest

from scripts.evaluate_adaptive_rollout import (
    load_formal_rows,
    write_offline_outputs,
)


def write_metric_csv(root, *, episodes, steps, seed_columns=4):
    path = root / "metrics.csv"
    fieldnames = ["episode", "step", "latent_variance", "rgb_mae"] + [
        f"error_seed_{seed}" for seed in range(seed_columns)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            for step in steps:
                writer.writerow(
                    {
                        "episode": episode,
                        "step": step,
                        "latent_variance": episode * 10 + step,
                        "rgb_mae": float(step),
                        **{
                            f"error_seed_{seed}": float(step + seed)
                            for seed in range(seed_columns)
                        },
                    }
                )
    return path


def test_loader_rejects_sealed_test_episode(tmp_path):
    csv_path = write_metric_csv(tmp_path, episodes=[1080], steps=range(1, 6))

    with pytest.raises(ValueError, match="sealed test"):
        load_formal_rows(csv_path)


def test_loader_requires_four_seed_error_columns_and_five_steps(tmp_path):
    csv_path = write_metric_csv(tmp_path, episodes=[1064], steps=[1, 2, 3, 4])

    with pytest.raises(ValueError, match="five future steps"):
        load_formal_rows(csv_path)


def test_writer_emits_exact_reconstructible_output_set(tmp_path):
    result = {
        "summary": {
            "source": {"archive_sha256": "abc"},
            "primary_target_coverage": 0.8,
        },
        "decisions": [
            {"episode": 1064, "policy": "threshold", "retained_horizon": 4}
        ],
        "curve": [{"policy": "threshold", "tau": 0.5, "coverage": 0.8}],
        "folds": [{"held_out_episode": 1064, "tau": 0.5}],
        "counterexamples": [
            {"episode": 1064, "reason": "low uncertainty high error"}
        ],
        "report": "OFFLINE PASS: ONLINE AUTHORIZED\n",
    }

    write_offline_outputs(tmp_path / "out", result, overwrite=False)

    output_names = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert output_names == [
        "adaptive_rollout_summary.json",
        "counterexamples.csv",
        "loeo_folds.csv",
        "policy_curve.csv",
        "policy_decisions.csv",
        "report.md",
    ]
    saved = json.loads(
        (tmp_path / "out/adaptive_rollout_summary.json").read_text()
    )
    assert saved["source"]["archive_sha256"] == "abc"
    assert saved["primary_target_coverage"] == 0.8


def test_writer_refuses_nonempty_output_without_overwrite(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("keep me")

    with pytest.raises(FileExistsError, match="non-empty"):
        write_offline_outputs(
            output_dir,
            {
                "summary": {},
                "decisions": [],
                "curve": [],
                "folds": [],
                "counterexamples": [],
                "report": "",
            },
            overwrite=False,
        )
