#!/usr/bin/env python3
"""Compare K=1/2/4 sampling cost and selective-prediction reliability."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

import torch

from miniworld.selective_prediction import loeo_selective_metrics, risk_coverage_curve
from miniworld.uncertainty import (
    future_rgb_blocks,
    latent_population_variance,
    rgb_pairwise_disagreement,
)
from scripts.evaluate_droid_uncertainty import (
    _read_video,
    validate_manifests,
)
from scripts.evaluate_adaptive_rollout import load_formal_rows


VALIDATION_EPISODES = tuple(range(1064, 1080))


def member_subsets(member_count: int, subset_size: int) -> list[tuple[int, ...]]:
    if not 1 <= subset_size <= member_count:
        raise ValueError("subset size must lie between one and member count")
    return list(combinations(range(member_count), subset_size))


def _selective_metrics(rows: list[dict]) -> dict:
    normalized = [
        {
            "episode": row["episode"],
            "uncertainty": row["uncertainty_rgb"],
            "error": row["error_rgb"],
        }
        for row in rows
    ]
    curve = risk_coverage_curve(normalized, target_coverages=(0.8,))
    return {
        "aurc": curve["aurc"],
        "random_aurc": curve["random_aurc"],
        "oracle_aurc": curve["oracle_aurc"],
        "loeo_80": {
            key: value
            for key, value in loeo_selective_metrics(
                normalized, target_coverage=0.8
            ).items()
            if key != "folds"
        },
    }


def summarize_k_ablation(
    *, k4_rows: list[dict], k2_rows_by_pair: dict[tuple[int, ...], list[dict]]
) -> dict:
    expected_pairs = member_subsets(4, 2)
    if sorted(k2_rows_by_pair) != expected_pairs:
        raise ValueError("K=2 evaluation requires all six member pairs")
    if len(k4_rows) != 80 or any(len(rows) != 80 for rows in k2_rows_by_pair.values()):
        raise ValueError("each formal K evaluation requires exactly 80 rows")

    seed_means = [
        mean(float(row[f"error_seed_{seed}"]) for row in k4_rows)
        for seed in range(4)
    ]
    pair_results = []
    for pair in expected_pairs:
        metrics = _selective_metrics(k2_rows_by_pair[pair])
        pair_results.append({"members": list(pair), **metrics})
    k2_aurcs = [result["aurc"] for result in pair_results]
    k2_maes = [result["loeo_80"]["mean_error"] for result in pair_results]
    k2_worst = [result["loeo_80"]["worst_error"] for result in pair_results]
    return {
        "cost_definition": "number of independently sampled rollouts; no wall-time claim",
        "k1": {
            "sampling_cost_x": 1,
            "uncertainty_available": False,
            "full_coverage_mean_mae": mean(seed_means),
            "per_seed_mean_mae": seed_means,
        },
        "k2": {
            "sampling_cost_x": 2,
            "uncertainty_available": True,
            "pair_count": len(pair_results),
            "pairs": pair_results,
            "aurc_mean_min_max": [mean(k2_aurcs), min(k2_aurcs), max(k2_aurcs)],
            "loeo_80_mean_mae_mean_min_max": [
                mean(k2_maes), min(k2_maes), max(k2_maes)
            ],
            "loeo_80_worst_mae_mean_min_max": [
                mean(k2_worst), min(k2_worst), max(k2_worst)
            ],
        },
        "k4": {
            "sampling_cost_x": 4,
            "uncertainty_available": True,
            **_selective_metrics(k4_rows),
        },
    }


def attach_subset_errors(
    uncertainty_rows: list[dict],
    source_rows: list[dict],
    *,
    members: tuple[int, ...],
) -> list[dict]:
    source_by_key = {
        (row["episode"], row["future_latent_step"]): row for row in source_rows
    }
    result = []
    for row in uncertainty_rows:
        key = (row["episode"], row["future_latent_step"])
        source = source_by_key.get(key)
        if source is None:
            raise ValueError("uncertainty rows do not align with source metrics")
        result.append(
            {
                **row,
                "error_rgb": mean(
                    float(source[f"error_seed_{member}"]) for member in members
                ),
            }
        )
    return result


def _assert_k4_reconstruction(rebuilt: list[dict], source: list[dict]) -> None:
    keys = ("uncertainty_latent", "uncertainty_rgb")
    rebuilt = sorted(rebuilt, key=lambda row: (row["episode"], row["future_latent_step"]))
    for actual, expected in zip(rebuilt, source, strict=True):
        if (actual["episode"], actual["future_latent_step"]) != (
            expected["episode"], expected["future_latent_step"]
        ):
            raise ValueError("rebuilt K=4 rows do not align with source metrics")
        for key in keys:
            if abs(float(actual[key]) - float(expected[key])) > 1e-5:
                raise ValueError(f"rebuilt K=4 {key} differs from source metrics")


def rebuild_subset_rows(
    *, sample_roots: list[Path], source_rows: list[dict]
) -> tuple[list[dict], dict[tuple[int, ...], list[dict]], dict]:
    manifests = [json.loads((root / "sampling_manifest.json").read_text()) for root in sample_roots]
    combined = validate_manifests(manifests, allow_incomplete=False)
    if len(sample_roots) != 4 or tuple(combined["episodes"]) != VALIDATION_EPISODES:
        raise ValueError("formal K ablation requires four members and validation 1064-1079")
    reconstructed_k4 = []
    pair_uncertainty_rows = {pair: [] for pair in member_subsets(4, 2)}
    for sample_index, episode in enumerate(VALIDATION_EPISODES):
        latents = []
        rgbs = []
        for root in sample_roots:
            latent = torch.load(root / "latents" / f"sample_{sample_index:04d}.pt", map_location="cpu", weights_only=True)
            latents.append(latent[0])
            rgbs.append(_read_video(root / "pred" / f"sample_{sample_index:04d}.mp4"))
        latent_stack, rgb_stack = torch.stack(latents), torch.stack(rgbs)
        blocks = future_rgb_blocks(latent_stack.shape[2], rgb_stack.shape[1])
        full_latent = latent_population_variance(latent_stack)[1:]
        full_rgb = rgb_pairwise_disagreement(rgb_stack, blocks)
        for step, (latent_value, rgb_value) in enumerate(
            zip(full_latent, full_rgb, strict=True), start=1
        ):
            reconstructed_k4.append(
                {
                    "episode": episode,
                    "future_latent_step": step,
                    "uncertainty_latent": float(latent_value),
                    "uncertainty_rgb": float(rgb_value),
                }
            )
        for pair in pair_uncertainty_rows:
            pair_rgb = rgb_pairwise_disagreement(rgb_stack[list(pair)], blocks)
            pair_uncertainty_rows[pair].extend(
                {
                    "episode": episode,
                    "future_latent_step": step,
                    "uncertainty_rgb": float(value),
                }
                for step, value in enumerate(pair_rgb, start=1)
            )
    _assert_k4_reconstruction(reconstructed_k4, source_rows)
    pair_rows = {
        pair: attach_subset_errors(rows, source_rows, members=pair)
        for pair, rows in pair_uncertainty_rows.items()
    }
    return source_rows, pair_rows, combined


def _render_report(summary: dict) -> str:
    k2 = summary["k2"]
    k4 = summary["k4"]
    return "\n".join(
        [
            "# K=1/2/4 selective-prediction cost/reliability ablation",
            "",
            "Cost is reported as sampled rollout count (1x/2x/4x); the source logs do not support a wall-time claim.",
            "",
            "| K | cost | uncertainty | AURC | LOEO-80 mean MAE | LOEO-80 worst MAE |",
            "| ---: | ---: | --- | ---: | ---: | ---: |",
            f"| 1 | 1x | no | n/a | {summary['k1']['full_coverage_mean_mae']:.4f} (100% only) | n/a |",
            f"| 2 | 2x | yes, six-pair mean | {k2['aurc_mean_min_max'][0]:.4f} | {k2['loeo_80_mean_mae_mean_min_max'][0]:.4f} | {k2['loeo_80_worst_mae_mean_min_max'][0]:.4f} |",
            f"| 4 | 4x | yes | {k4['aurc']:.4f} | {k4['loeo_80']['mean_error']:.4f} | {k4['loeo_80']['worst_error']:.4f} |",
            "",
            "K=2 reports all six seed pairs rather than selecting a favorable pair.",
        ]
    ) + "\n"


def write_outputs(output_dir: Path, summary: dict) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "k_ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pair_rows = []
    for result in summary["k2"]["pairs"]:
        pair_rows.append(
            {
                "members": "|".join(map(str, result["members"])),
                "aurc": result["aurc"],
                **{f"loeo80_{key}": value for key, value in result["loeo_80"].items()},
            }
        )
    with (output_dir / "k2_pair_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    (output_dir / "report.md").write_text(_render_report(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_root", type=Path, action="append", required=True)
    parser.add_argument("--source_metrics_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    source_rows = load_formal_rows(args.source_metrics_csv)
    k4_rows, pair_rows, combined = rebuild_subset_rows(
        sample_roots=args.sample_root, source_rows=source_rows
    )
    summary = summarize_k_ablation(k4_rows=k4_rows, k2_rows_by_pair=pair_rows)
    summary["source"] = combined
    write_outputs(args.output_dir, summary)
    print(_render_report(summary))


if __name__ == "__main__":
    main()
