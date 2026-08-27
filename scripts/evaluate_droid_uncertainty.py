#!/usr/bin/env python3
"""Evaluate stochastic DROID rollouts for uncertainty-error correlation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
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


VALIDATION_EPISODES = tuple(range(1064, 1080))
SEALED_TEST_EPISODES = frozenset(range(1080, 1096))


def validate_manifests(
    manifests: list[dict], *, allow_incomplete: bool = False
) -> dict:
    """Validate ensemble identities and return a combined immutable record."""
    if len(manifests) != 4 and not allow_incomplete:
        raise ValueError("formal evaluation requires exactly four manifests")
    if len(manifests) < 2:
        raise ValueError("uncertainty evaluation requires at least two manifests")
    seeds = [int(manifest["seed"]) for manifest in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("sampling manifests must use distinct seeds")

    episodes = [int(value) for value in manifests[0]["episodes"]]
    if any(episode in SEALED_TEST_EPISODES for episode in episodes):
        raise ValueError("sampling manifest references the sealed test split")
    if any(episode not in VALIDATION_EPISODES for episode in episodes):
        raise ValueError("only validation episodes 1064-1079 are allowed")
    if not allow_incomplete and tuple(episodes) != VALIDATION_EPISODES:
        raise ValueError("formal evaluation requires all validation episodes 1064-1079")

    common_keys = (
        "schema_version",
        "dataset",
        "data_root",
        "data_manifest_sha256",
        "episodes",
        "checkpoint",
        "checkpoint_sha256",
        "weights_source",
        "wm_model",
        "action_variant",
        "git_commit",
        "sampling",
    )
    reference = manifests[0]
    for manifest_index, manifest in enumerate(manifests[1:], start=1):
        for key in common_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(
                    f"manifest {manifest_index} disagrees on {key}"
                )
    expected_records = [(index, episode) for index, episode in enumerate(episodes)]
    reference_shapes = [sample.get("latent_shape") for sample in reference["samples"]]
    for manifest_index, manifest in enumerate(manifests):
        records = [
            (int(sample["sample_index"]), int(sample["episode"]))
            for sample in manifest.get("samples", [])
        ]
        if records != expected_records:
            raise ValueError(
                f"manifest {manifest_index} sample records do not match episodes"
            )
        shapes = [sample.get("latent_shape") for sample in manifest["samples"]]
        if shapes != reference_shapes:
            raise ValueError(
                f"manifest {manifest_index} sample records disagree on latent shapes"
            )
    if reference.get("dataset") != "droid":
        raise ValueError("uncertainty evaluator requires DROID manifests")
    if reference.get("action_variant") != "real":
        raise ValueError("formal uncertainty evaluation requires real actions")

    return {
        "schema_version": 1,
        "incomplete": len(manifests) != 4,
        "k": len(manifests),
        "seeds": seeds,
        "episodes": episodes,
        "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "data_root": reference["data_root"],
        "data_manifest_sha256": reference["data_manifest_sha256"],
        "git_commit": reference["git_commit"],
        "sampling": reference["sampling"],
    }


def evaluate_ensemble(
    latents: torch.Tensor,
    rgb: torch.Tensor,
    ground_truth: torch.Tensor,
) -> list[dict[str, float | int]]:
    """Compute five future-step observations for one aligned ensemble."""
    if latents.ndim != 5:
        raise ValueError("latents must have shape (K,C,T,H,W)")
    if rgb.ndim != 5 or ground_truth.ndim != 4:
        raise ValueError("RGB inputs must have shapes (K,T,H,W,C) and (T,H,W,C)")
    if rgb.shape[0] != latents.shape[0]:
        raise ValueError("latent and RGB ensembles must have identical K")
    if rgb.shape[1:] != ground_truth.shape:
        raise ValueError("RGB ensemble members and target must have identical shapes")
    if not torch.isfinite(latents).all() or not torch.isfinite(rgb).all():
        raise ValueError("ensemble contains NaN or Inf")

    blocks = future_rgb_blocks(latents.shape[2], rgb.shape[1])
    latent_uncertainty = latent_population_variance(latents)[1:]
    rgb_uncertainty = rgb_pairwise_disagreement(rgb, blocks)
    mean_error, per_seed_error = rgb_memberwise_mae(rgb, ground_truth, blocks)
    rows = []
    for step_index, block in enumerate(blocks):
        row: dict[str, float | int] = {
            "future_latent_step": step_index + 1,
            "rgb_frame_start": int(block.start),
            "rgb_frame_end": int(block.stop - 1),
            "uncertainty_latent": float(latent_uncertainty[step_index]),
            "uncertainty_rgb": float(rgb_uncertainty[step_index]),
            "error_rgb": float(mean_error[step_index]),
        }
        for seed_index in range(latents.shape[0]):
            row[f"error_seed_{seed_index}"] = float(
                per_seed_error[seed_index, step_index]
            )
        rows.append(row)
    return rows


def _metric_arrays(rows: list[dict], uncertainty_key: str):
    uncertainty = np.asarray([row[uncertainty_key] for row in rows], dtype=np.float64)
    error = np.asarray([row["error_rgb"] for row in rows], dtype=np.float64)
    horizons = np.asarray(
        [row["future_latent_step"] for row in rows], dtype=np.int64
    )
    return uncertainty, error, horizons


def summarize_estimator(rows: list[dict], uncertainty_key: str) -> dict:
    uncertainty, error, horizons = _metric_arrays(rows, uncertainty_key)
    per_horizon = {}
    for horizon in sorted(set(horizons.tolist())):
        mask = horizons == horizon
        per_horizon[str(horizon)] = {
            "count": int(mask.sum()),
            "pearson": pearson_correlation(uncertainty[mask], error[mask])
            if mask.sum() >= 2
            else None,
            "spearman": spearman_correlation(uncertainty[mask], error[mask])
            if mask.sum() >= 2
            else None,
        }
    per_episode = {}
    for episode in sorted({int(row["episode"]) for row in rows}):
        selected = [row for row in rows if int(row["episode"]) == episode]
        episode_uncertainty = np.asarray(
            [row[uncertainty_key] for row in selected], dtype=np.float64
        )
        episode_error = np.asarray(
            [row["error_rgb"] for row in selected], dtype=np.float64
        )
        per_episode[str(episode)] = {
            "count": len(selected),
            "pearson": pearson_correlation(episode_uncertainty, episode_error),
            "spearman": spearman_correlation(episode_uncertainty, episode_error),
        }
    return {
        "pearson": pearson_correlation(uncertainty, error),
        "spearman": spearman_correlation(uncertainty, error),
        "horizon_conditioned_spearman": horizon_conditioned_spearman(
            uncertainty, error, horizons
        ),
        "per_horizon": per_horizon,
        "per_episode": per_episode,
    }


def _signal_gate(rows: list[dict], uncertainty_key: str) -> dict:
    uncertainty, error, _ = _metric_arrays(rows, uncertainty_key)
    bins = equal_count_bins(uncertainty, error, bins=min(4, len(rows)))
    estimator = summarize_estimator(rows, uncertainty_key)
    pooled = estimator["spearman"]
    conditioned = estimator["horizon_conditioned_spearman"]
    top_above_bottom = bins[-1]["mean_error"] > bins[0]["mean_error"]
    passed = (
        pooled is not None
        and pooled >= 0.30
        and conditioned is not None
        and conditioned >= 0.20
        and top_above_bottom
    )
    return {
        "pooled_spearman_at_least_0_30": pooled is not None and pooled >= 0.30,
        "horizon_conditioned_spearman_at_least_0_20": conditioned is not None
        and conditioned >= 0.20,
        "highest_bin_error_above_lowest": bool(top_above_bottom),
        "passed": bool(passed),
    }


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(rows[0])
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_evaluation_outputs(
    output_dir: Path,
    rows: list[dict],
    combined_manifest: dict,
    *,
    overwrite: bool,
) -> dict:
    """Write deterministic machine-readable tables and their Markdown view."""
    if not rows:
        raise ValueError("cannot write an empty uncertainty evaluation")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        rows, key=lambda row: (int(row["episode"]), int(row["future_latent_step"]))
    )
    _write_csv_atomic(output_dir / "metrics_per_episode_step.csv", ordered_rows)

    estimators = {
        "latent": summarize_estimator(ordered_rows, "uncertainty_latent"),
        "rgb": summarize_estimator(ordered_rows, "uncertainty_rgb"),
    }
    summary = {
        "split": "validation",
        "incomplete": bool(combined_manifest["incomplete"]),
        "episodes": sorted({int(row["episode"]) for row in ordered_rows}),
        "count_episode_steps": len(ordered_rows),
        "k": int(combined_manifest["k"]),
        "seeds": combined_manifest["seeds"],
        "estimators": estimators,
        "gate": {
            "latent": _signal_gate(ordered_rows, "uncertainty_latent"),
            "rgb": _signal_gate(ordered_rows, "uncertainty_rgb"),
        },
    }
    _write_json_atomic(output_dir / "correlation_summary.json", summary)
    _write_json_atomic(
        output_dir / "sampling_manifest_combined.json", combined_manifest
    )

    bin_rows = []
    for label, key in (
        ("latent", "uncertainty_latent"),
        ("rgb", "uncertainty_rgb"),
    ):
        uncertainty, error, _ = _metric_arrays(ordered_rows, key)
        for row in equal_count_bins(uncertainty, error, bins=min(4, len(rows))):
            bin_rows.append({"estimator": label, **row})
    _write_csv_atomic(output_dir / "uncertainty_bins.csv", bin_rows)

    horizon_rows = []
    for horizon in sorted({int(row["future_latent_step"]) for row in ordered_rows}):
        selected = [row for row in ordered_rows if row["future_latent_step"] == horizon]
        horizon_rows.append(
            {
                "future_latent_step": horizon,
                "count": len(selected),
                "mean_uncertainty_latent": float(
                    np.mean([row["uncertainty_latent"] for row in selected])
                ),
                "mean_uncertainty_rgb": float(
                    np.mean([row["uncertainty_rgb"] for row in selected])
                ),
                "mean_error_rgb": float(
                    np.mean([row["error_rgb"] for row in selected])
                ),
            }
        )
    _write_csv_atomic(output_dir / "horizon_summary.csv", horizon_rows)

    status = "INCOMPLETE INTEGRATION GATE" if summary["incomplete"] else "FORMAL VALIDATION"
    report = (
        "# MiniWorld uncertainty-error correlation\n\n"
        f"Status: **{status}**\n\n"
        f"Episodes: {summary['episodes']}  \n"
        f"K: {summary['k']}  \n"
        f"Episode-step observations: {summary['count_episode_steps']}\n\n"
        "## Correlations\n\n"
        "| Estimator | Pearson | Spearman | Horizon-conditioned Spearman | Gate |\n"
        "| --- | ---: | ---: | ---: | --- |\n"
    )
    for label in ("latent", "rgb"):
        result = estimators[label]
        report += (
            f"| {label} | {result['pearson']} | {result['spearman']} | "
            f"{result['horizon_conditioned_spearman']} | "
            f"{summary['gate'][label]['passed']} |\n"
        )
    temporary = output_dir / "report.md.tmp"
    temporary.write_text(report)
    temporary.replace(output_dir / "report.md")
    return summary


def _read_video(path: Path) -> torch.Tensor:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0))
    return torch.from_numpy(reader.get_batch(range(len(reader))).asnumpy())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate DROID uncertainty correlation")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sample_root", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: Iterable[str] | None = None) -> None:
    from miniworld.data.droid import LeRobotActionDataset

    args = build_parser().parse_args(argv)
    roots = [Path(value) for value in args.sample_root]
    manifests = [_load_json(root / "sampling_manifest.json") for root in roots]
    combined = validate_manifests(manifests, allow_incomplete=args.allow_incomplete)
    dataset = LeRobotActionDataset(
        root=args.data_root,
        num_frames=21,
        frame_interval=1,
        resize_hw=(240, 320),
        camera_views=["exterior_image_1_left"],
        action_keys=["cartesian_position", "gripper_position"],
        action_norm="q01q99",
        randomize=False,
        color_aug=False,
        require_success=True,
        max_keep=None,
    )
    dataset_indices = {int(episode): index for index, episode in enumerate(dataset.samples)}
    rows = []
    for sample_index, episode in enumerate(combined["episodes"]):
        if episode not in dataset_indices:
            raise ValueError(f"episode {episode} is missing from the evaluation dataset")
        ground_truth = (
            ((dataset[dataset_indices[episode]]["videos"] + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        latent_members = []
        rgb_members = []
        for root in roots:
            latent_path = root / "latents" / f"sample_{sample_index:04d}.pt"
            video_path = root / "pred" / f"sample_{sample_index:04d}.mp4"
            if not latent_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(
                    f"missing ensemble member for episode {episode}: {root}"
                )
            latent = torch.load(latent_path, map_location="cpu", weights_only=True)
            if latent.ndim != 5 or latent.shape[0] != 1:
                raise ValueError(f"unexpected saved latent shape: {tuple(latent.shape)}")
            latent_members.append(latent[0])
            rgb_members.append(_read_video(video_path))
        episode_rows = evaluate_ensemble(
            torch.stack(latent_members), torch.stack(rgb_members), ground_truth
        )
        for row in episode_rows:
            row["episode"] = int(episode)
            for seed_index, seed in enumerate(combined["seeds"]):
                row[f"seed_{seed_index}"] = int(seed)
            rows.append(row)
    summary = write_evaluation_outputs(
        Path(args.output_dir), rows, combined, overwrite=args.overwrite
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
