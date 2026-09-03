import json

import pytest
import torch

from scripts.evaluate_droid_uncertainty import (
    evaluate_ensemble,
    validate_manifests,
    write_evaluation_outputs,
)


def make_manifests(seeds, episodes):
    return [
        {
            "schema_version": 1,
            "dataset": "droid",
            "data_root": "/validation",
            "data_manifest_sha256": "data-hash",
            "episodes": episodes,
            "checkpoint": "/official.pt",
            "checkpoint_sha256": "checkpoint-hash",
            "weights_source": "ema",
            "wm_model": "0.5B",
            "seed": seed,
            "action_variant": "real",
            "git_commit": "commit-id",
            "sampling": {"total_len": 6, "precision": "fp16", "attention_backend": "sdpa"},
            "samples": [
                {
                    "sample_index": index,
                    "episode": episode,
                    "latent_shape": [1, 48, 6, 15, 20],
                    "latent_dtype": "torch.float16",
                }
                for index, episode in enumerate(episodes)
            ],
        }
        for seed in seeds
    ]


def test_manifest_validation_rejects_test_episode():
    manifests = make_manifests(
        seeds=[11, 22, 33, 44], episodes=list(range(1080, 1096))
    )

    with pytest.raises(ValueError, match="sealed test"):
        validate_manifests(manifests)


def test_manifest_validation_rejects_duplicate_seed():
    manifests = make_manifests(
        seeds=[11, 11, 33, 44], episodes=list(range(1064, 1080))
    )

    with pytest.raises(ValueError, match="distinct seeds"):
        validate_manifests(manifests)


def test_manifest_validation_rejects_inconsistent_checkpoint():
    manifests = make_manifests(
        seeds=[11, 22, 33, 44], episodes=list(range(1064, 1080))
    )
    manifests[-1]["checkpoint_sha256"] = "different"

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_manifests(manifests)


def test_manifest_validation_rejects_unknown_git_commit():
    manifests = make_manifests(
        seeds=[11, 22, 33, 44], episodes=list(range(1064, 1080))
    )
    for manifest in manifests:
        manifest["git_commit"] = "unknown"

    with pytest.raises(ValueError, match="git commit"):
        validate_manifests(manifests)


def test_manifest_validation_requires_four_unless_incomplete():
    manifests = make_manifests(seeds=[11, 22], episodes=[1064])

    with pytest.raises(ValueError, match="exactly four"):
        validate_manifests(manifests)

    combined = validate_manifests(manifests, allow_incomplete=True)
    assert combined["incomplete"] is True
    assert combined["k"] == 2


def test_manifest_validation_rejects_missing_sample_record():
    manifests = make_manifests(
        seeds=[11, 22, 33, 44], episodes=list(range(1064, 1080))
    )
    manifests[-1]["samples"].pop()

    with pytest.raises(ValueError, match="sample records"):
        validate_manifests(manifests)


def test_evaluate_ensemble_returns_one_row_per_future_latent():
    latents = torch.stack(
        [torch.zeros(1, 6, 1, 1), torch.full((1, 6, 1, 1), 2.0)]
    )
    rgb = torch.stack(
        [torch.zeros(21, 1, 1, 1), torch.full((21, 1, 1, 1), 2.0)]
    )
    ground_truth = torch.ones(21, 1, 1, 1)

    rows = evaluate_ensemble(latents, rgb, ground_truth)

    assert len(rows) == 5
    assert rows[0] == {
        "future_latent_step": 1,
        "rgb_frame_start": 1,
        "rgb_frame_end": 4,
        "uncertainty_latent": 1.0,
        "uncertainty_rgb": 2.0,
        "error_rgb": 1.0,
        "error_seed_0": 1.0,
        "error_seed_1": 1.0,
    }


def test_evaluate_ensemble_rejects_shape_mismatch_and_nonfinite():
    latents = torch.zeros(2, 1, 6, 1, 1)
    rgb = torch.zeros(2, 21, 1, 1, 1)
    target = torch.zeros(20, 1, 1, 1)

    with pytest.raises(ValueError, match="identical"):
        evaluate_ensemble(latents, rgb, target)

    latents[0, 0, 1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        evaluate_ensemble(latents, rgb, torch.zeros(21, 1, 1, 1))


def test_write_outputs_is_deterministic_and_marks_incomplete(tmp_path):
    rows = [
        {
            "episode": 1064,
            "future_latent_step": step,
            "rgb_frame_start": 1 + 4 * (step - 1),
            "rgb_frame_end": 4 * step,
            "uncertainty_latent": float(step),
            "uncertainty_rgb": float(step + 1),
            "error_rgb": float(step * 2),
            "error_seed_0": float(step),
            "error_seed_1": float(step * 3),
        }
        for step in range(1, 6)
    ]
    combined = {"incomplete": True, "k": 2, "seeds": [11, 22]}

    write_evaluation_outputs(tmp_path, rows, combined, overwrite=False)

    summary = json.loads((tmp_path / "correlation_summary.json").read_text())
    report = (tmp_path / "report.md").read_text()
    assert summary["incomplete"] is True
    assert summary["count_episode_steps"] == 5
    assert summary["estimators"]["latent"]["spearman"] == pytest.approx(1.0)
    assert summary["estimators"]["latent"]["per_episode"]["1064"][
        "spearman"
    ] == pytest.approx(1.0)
    assert "INCOMPLETE INTEGRATION GATE" in report
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "correlation_summary.json",
        "horizon_summary.csv",
        "metrics_per_episode_step.csv",
        "report.md",
        "sampling_manifest_combined.json",
        "uncertainty_bins.csv",
    ]

    with pytest.raises(FileExistsError):
        write_evaluation_outputs(tmp_path, rows, combined, overwrite=False)


def test_main_writes_synthetic_incomplete_evaluation(tmp_path, monkeypatch):
    roots = [tmp_path / "seed_11", tmp_path / "seed_22"]
    manifests = make_manifests(seeds=[11, 22], episodes=[1064])
    for root, manifest, value in zip(roots, manifests, (0.0, 2.0)):
        (root / "latents").mkdir(parents=True)
        (root / "pred").mkdir()
        (root / "sampling_manifest.json").write_text(json.dumps(manifest))
        torch.save(
            torch.full((1, 1, 6, 1, 1), value),
            root / "latents" / "sample_0000.pt",
        )
        (root / "pred" / "sample_0000.mp4").touch()

    class FakeDataset:
        samples = [1064]

        def __init__(self, **kwargs):
            assert kwargs["num_frames"] == 21
            assert kwargs["randomize"] is False

        def __getitem__(self, index):
            assert index == 0
            return {"videos": torch.full((21, 1, 1, 1), -1.0)}

    monkeypatch.setattr(
        "miniworld.data.droid.LeRobotActionDataset", FakeDataset
    )
    monkeypatch.setattr(
        "scripts.evaluate_droid_uncertainty._read_video",
        lambda path: torch.full(
            (21, 1, 1, 1), 0 if "seed_11" in str(path) else 2, dtype=torch.uint8
        ),
    )
    from scripts.evaluate_droid_uncertainty import main

    output = tmp_path / "metrics"
    main(
        [
            "--data_root",
            str(tmp_path / "validation"),
            "--sample_root",
            str(roots[0]),
            "--sample_root",
            str(roots[1]),
            "--output_dir",
            str(output),
            "--allow_incomplete",
        ]
    )

    rows = (output / "metrics_per_episode_step.csv").read_text().splitlines()
    summary = json.loads((output / "correlation_summary.json").read_text())
    assert len(rows) == 6
    assert summary["incomplete"] is True
    assert summary["count_episode_steps"] == 5
