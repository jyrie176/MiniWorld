#!/usr/bin/env python3
"""Controlled corruption stress test for K=2 selective prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch

from miniworld.selective_prediction import loeo_selective_metrics, risk_coverage_curve
from miniworld.uncertainty import (
    future_rgb_blocks,
    rgb_memberwise_mae,
    rgb_pairwise_disagreement,
)
from scripts.evaluate_droid_uncertainty import _read_video, validate_manifests
from scripts.evaluate_selective_k_ablation import member_subsets


VALIDATION_EPISODES = tuple(range(1064, 1080))


def apply_common_brightness(ensemble: torch.Tensor, *, delta: float) -> torch.Tensor:
    """Apply the same pixel-space shift to every ensemble member."""
    return (ensemble.float() + float(delta)).clamp(0.0, 255.0)


def apply_independent_noise(
    ensemble: torch.Tensor, *, sigma: float, seed: int
) -> torch.Tensor:
    """Apply deterministic independent Gaussian noise to each member."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        ensemble.shape, generator=generator, dtype=torch.float32
    ) * float(sigma)
    return (ensemble.float() + noise).clamp(0.0, 255.0)


def stress_seed(episode: int, pair: tuple[int, int], level: int) -> int:
    """Derive a stable seed without Python's process-randomized hash()."""
    return int(episode) * 10_000 + pair[0] * 1_000 + pair[1] * 100 + int(level)


def _read_ground_truth(path: Path) -> torch.Tensor:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), width=320, height=240)
    if len(reader) < 21:
        raise ValueError(f"ground-truth video has fewer than 21 frames: {path}")
    return torch.from_numpy(reader.get_batch(range(21)).asnumpy())


def _condition_specs() -> list[tuple[str, int]]:
    return [
        ("clean", 0),
        ("common_brightness", 16),
        ("common_brightness", 32),
        ("independent_noise", 8),
        ("independent_noise", 16),
    ]


def _apply_condition(
    ensemble: torch.Tensor,
    *,
    condition: str,
    level: int,
    episode: int,
    pair: tuple[int, int],
) -> torch.Tensor:
    if condition == "clean":
        return ensemble.float()
    if condition == "common_brightness":
        return apply_common_brightness(ensemble, delta=level)
    if condition == "independent_noise":
        return apply_independent_noise(
            ensemble, sigma=level, seed=stress_seed(episode, pair, level)
        )
    raise ValueError(f"unknown stress condition: {condition}")


def _pair_metrics(rows: list[dict]) -> dict:
    normalized = [
        {
            "episode": row["episode"],
            "uncertainty": row["uncertainty_rgb"],
            "error": row["error_rgb"],
        }
        for row in rows
    ]
    curve = risk_coverage_curve(normalized, target_coverages=(0.8,))
    loeo = loeo_selective_metrics(normalized, target_coverage=0.8)
    full_mean_error = mean(row["error_rgb"] for row in rows)
    return {
        "mean_uncertainty": mean(row["uncertainty_rgb"] for row in rows),
        "full_mean_error": full_mean_error,
        "aurc": curve["aurc"],
        "random_aurc": curve["random_aurc"],
        "aurc_gain_vs_random": curve["random_aurc"] - curve["aurc"],
        "loeo80_coverage": loeo["realized_coverage"],
        "loeo80_mean_error": loeo["mean_error"],
        "loeo80_error_reduction": full_mean_error - loeo["mean_error"],
        "loeo80_worst_error": loeo["worst_error"],
    }


def evaluate_stress(
    *, sample_roots: list[Path], data_root: Path, expected_clean_aurc: float
) -> dict:
    manifests = [json.loads((root / "sampling_manifest.json").read_text()) for root in sample_roots]
    combined = validate_manifests(manifests, allow_incomplete=False)
    if len(sample_roots) != 4 or tuple(combined["episodes"]) != VALIDATION_EPISODES:
        raise ValueError("stress evaluation requires four validation ensemble members")
    pairs = member_subsets(4, 2)
    rows_by_condition = {
        spec: {pair: [] for pair in pairs} for spec in _condition_specs()
    }
    blocks = future_rgb_blocks(6, 21)
    video_dir = data_root / "videos/chunk-001/observation.images.exterior_image_1_left"
    for sample_index, episode in enumerate(VALIDATION_EPISODES):
        target = _read_ground_truth(video_dir / f"episode_{episode:06d}.mp4")
        members = torch.stack(
            [
                _read_video(root / "pred" / f"sample_{sample_index:04d}.mp4")
                for root in sample_roots
            ]
        )
        for pair in pairs:
            clean_pair = members[list(pair)]
            for condition, level in _condition_specs():
                stressed = _apply_condition(
                    clean_pair,
                    condition=condition,
                    level=level,
                    episode=episode,
                    pair=pair,
                )
                uncertainty = rgb_pairwise_disagreement(stressed, blocks)
                error, _ = rgb_memberwise_mae(stressed, target, blocks)
                rows_by_condition[(condition, level)][pair].extend(
                    {
                        "episode": episode,
                        "future_latent_step": step,
                        "uncertainty_rgb": float(u),
                        "error_rgb": float(e),
                    }
                    for step, (u, e) in enumerate(
                        zip(uncertainty, error, strict=True), start=1
                    )
                )

    condition_results = []
    pair_results = []
    for condition, level in _condition_specs():
        metrics = []
        for pair in pairs:
            result = _pair_metrics(rows_by_condition[(condition, level)][pair])
            metrics.append(result)
            pair_results.append(
                {"condition": condition, "level": level, "members": list(pair), **result}
            )
        condition_results.append(
            {
                "condition": condition,
                "level": level,
                **{
                    key: mean(result[key] for result in metrics)
                    for key in metrics[0]
                },
            }
        )
    clean = condition_results[0]
    if abs(clean["aurc"] - expected_clean_aurc) > 1e-5:
        raise ValueError("clean reconstruction differs from frozen K=2 ablation")
    for result in condition_results:
        result["uncertainty_ratio_vs_clean"] = (
            result["mean_uncertainty"] / clean["mean_uncertainty"]
        )
        result["error_delta_vs_clean"] = (
            result["full_mean_error"] - clean["full_mean_error"]
        )
    return {
        "source": combined,
        "stress_scope": "controlled output corruption, not real-world OOD",
        "policy": "K=2 RGB disagreement with LOEO 80% target coverage",
        "conditions": condition_results,
        "pairs": pair_results,
    }


def _render_report(result: dict) -> str:
    lines = [
        "# Controlled corruption stress test",
        "",
        "This is an output-corruption stress test, not a real-world OOD benchmark.",
        "",
        "| condition | level | uncertainty ratio | full error delta | AURC gain vs random | LOEO-80 reduction | worst |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["conditions"]:
        lines.append(
            f"| {row['condition']} | {row['level']} | {row['uncertainty_ratio_vs_clean']:.4f} | "
            f"{row['error_delta_vs_clean']:+.4f} | {row['aurc_gain_vs_random']:.4f} | "
            f"{row['loeo80_error_reduction']:.4f} | {row['loeo80_worst_error']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, result: dict) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stress_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for name, rows in (("conditions.csv", result["conditions"]), ("pair_metrics.csv", result["pairs"])):
        serializable = [
            {**row, "members": "|".join(map(str, row["members"]))}
            if "members" in row
            else row
            for row in rows
        ]
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
            writer.writeheader()
            writer.writerows(serializable)
    (output_dir / "report.md").write_text(_render_report(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_root", type=Path, action="append", required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--k_ablation_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.k_ablation_summary.read_text())
    result = evaluate_stress(
        sample_roots=args.sample_root,
        data_root=args.data_root,
        expected_clean_aurc=frozen["k2"]["aurc_mean_min_max"][0],
    )
    write_outputs(args.output_dir, result)
    print(_render_report(result))


if __name__ == "__main__":
    main()
