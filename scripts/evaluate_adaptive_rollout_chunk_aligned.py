#!/usr/bin/env python3
"""Audit adaptive rollout at executable MiniWorld chunk boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from miniworld.rollout_policy import (
    ChunkLayout,
    EpisodePolicyResult,
    aggregate_policy_results,
    build_chunk_layout,
    chunk_aligned_smoothed_hysteretic_policy,
    chunk_aligned_threshold_policy,
    evaluate_chunk_aligned_gate,
    matched_fixed_baseline,
    run_loeo,
    score_policy_trace,
    select_threshold,
)
from scripts.evaluate_adaptive_rollout import (
    EXPECTED_ARCHIVE_SHA256,
    VALIDATION_EPISODES,
    load_formal_rows,
)


OFFICIAL_LAYOUT = build_chunk_layout(
    history_len=1, future_horizon=5, chunk_size=2
)


def _rows_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    return {
        episode: sorted(
            [row for row in rows if int(row["episode"]) == episode],
            key=lambda row: int(row["future_latent_step"]),
        )
        for episode in VALIDATION_EPISODES
    }


def _score_with_threshold(
    rows: list[dict], policy: str, tau: float, layout: ChunkLayout
) -> list[EpisodePolicyResult]:
    results = []
    for episode, episode_rows in _rows_by_episode(rows).items():
        uncertainty = [float(row["uncertainty_latent"]) for row in episode_rows]
        if policy == "threshold":
            trace = chunk_aligned_threshold_policy(
                uncertainty, tau, layout=layout
            )
        elif policy == "smoothed_hysteretic":
            trace = chunk_aligned_smoothed_hysteretic_policy(
                uncertainty, tau, layout=layout
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
        results.append(
            score_policy_trace(
                episode,
                trace,
                [float(row["error_rgb"]) for row in episode_rows],
                np.asarray(
                    [
                        [
                            float(row[f"error_seed_{seed}"])
                            for row in episode_rows
                        ]
                        for seed in range(4)
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return results


def _decision_row(
    policy: str, result: EpisodePolicyResult, *, tau: float
) -> dict:
    completion_boundary = result.trace.generated_horizon
    return {
        "episode": result.episode,
        "policy": policy,
        "tau": tau,
        "completion_boundaries": "1|3|5",
        "decisions": "|".join(decision.value for decision in result.trace.decisions),
        "requested_observation_at": result.trace.requested_observation_at,
        "emitted_at_completion_boundary": completion_boundary
        if result.trace.requested_observation_at is not None
        else None,
        "retained_horizon": result.trace.retained_horizon,
        "generated_horizon": result.trace.generated_horizon,
        "retained_error_numerator": result.retained_error_numerator,
        "retained_count": result.retained_count,
        "discarded_error_numerator": result.discarded_error_numerator,
        "discarded_count": result.discarded_count,
        "per_seed_retained_error": json.dumps(result.per_seed_retained_error),
    }


def _policy_cost(policy: str, metrics: dict) -> dict:
    completed = int(metrics["generated_count"])
    fixed_k4 = 16 * 5 * 4
    fixed_k1 = 16 * 5
    completed_k4 = completed * 4
    return {
        "policy": policy,
        "completed_future_steps": completed,
        "completed_k4_member_steps": completed_k4,
        "fixed_k4_member_steps": fixed_k4,
        "fixed_k1_member_steps": fixed_k1,
        "k4_fraction_of_fixed_k4": completed_k4 / fixed_k4,
        "k4_ratio_to_fixed_k1": completed_k4 / fixed_k1,
        "avoided_complete_future_steps": 16 * 5 - completed,
        "wall_time_claim": False,
    }


def evaluate_chunk_aligned(
    rows: list[dict], source_identity: dict, previous_summary: dict
) -> dict:
    if len(rows) != 80 or sorted({int(row["episode"]) for row in rows}) != list(
        VALIDATION_EPISODES
    ):
        raise ValueError("formal chunk-aligned evaluation requires 80 validation rows")
    high_error_cutoff = float(np.percentile([row["error_rgb"] for row in rows], 75))
    previous_policy = previous_summary["adaptive"]["smoothed_hysteretic"]
    previous_threshold = float(previous_policy["deployment_threshold"])
    previous_replay = _score_with_threshold(
        rows, "smoothed_hysteretic", previous_threshold, OFFICIAL_LAYOUT
    )
    previous_metrics = aggregate_policy_results(
        previous_replay, high_error_cutoff=high_error_cutoff
    )
    if previous_metrics["generated_coverage"] != 1.0:
        raise ValueError(
            "previous frozen policy must replay to full completed generation"
        )

    summary = {
        "source": source_identity,
        "primary_signal": "latent_population_variance",
        "primary_target_coverage": 0.80,
        "execution": {
            "history_len": OFFICIAL_LAYOUT.history_len,
            "future_horizon": OFFICIAL_LAYOUT.future_horizon,
            "df_chunk_size": OFFICIAL_LAYOUT.chunk_size,
            "completion_boundaries": list(OFFICIAL_LAYOUT.completion_boundaries),
            "cost_semantics": "completed_chunks_not_wall_time",
        },
        "correction": {
            "previous_idealized_generated_coverage": float(
                previous_policy["loeo"]["generated_coverage"]
            ),
            "previous_chunk_completed_generated_coverage": float(
                previous_metrics["generated_coverage"]
            ),
            "previous_online_authorization_superseded": True,
            "previous_deployment_threshold": previous_threshold,
        },
        "adaptive": {},
    }
    decisions = []
    curve = []
    folds = []
    costs = []
    counterexamples = []

    for policy in ("threshold", "smoothed_hysteretic"):
        policy_folds, results = run_loeo(
            rows,
            policy,
            execution="chunk",
            layout=OFFICIAL_LAYOUT,
        )
        aggregate = aggregate_policy_results(
            results, high_error_cutoff=high_error_cutoff
        )
        matched = matched_fixed_baseline(rows, aggregate["mean_retained_horizon"])
        adaptive_episode_errors = {
            result.episode: result.retained_error_numerator / result.retained_count
            for result in results
        }
        deltas = [
            adaptive_episode_errors[episode]
            - float(matched["episode_errors"][episode])
            for episode in VALIDATION_EPISODES
        ]
        gate = evaluate_chunk_aligned_gate(aggregate, matched, deltas)
        deployment = select_threshold(
            rows,
            policy,
            execution="chunk",
            layout=OFFICIAL_LAYOUT,
        )
        summary["adaptive"][policy] = {
            "loeo": aggregate,
            "matched_fixed": matched,
            "episode_deltas": dict(zip(VALIDATION_EPISODES, deltas)),
            "gate": gate,
            "deployment_threshold": deployment["tau"],
            "deployment_training_metrics": {
                key: deployment[key]
                for key in (
                    "coverage",
                    "generated_coverage",
                    "retained_rgb_mae",
                    "mean_retained_horizon",
                    "mean_generated_horizon",
                    "request_rate",
                    "avoided_completed_future_steps",
                )
            },
        }
        folds.extend(policy_folds)
        for fold, result in zip(policy_folds, results):
            decisions.append(
                _decision_row(policy, result, tau=float(fold["tau"]))
            )
        costs.append(_policy_cost(policy, aggregate))
        curve.extend(
            {
                "signal": "latent",
                "gating": True,
                "policy": policy,
                **{
                    key: candidate[key]
                    for key in (
                        "tau",
                        "coverage",
                        "generated_coverage",
                        "retained_rgb_mae",
                        "mean_retained_horizon",
                        "mean_generated_horizon",
                        "request_rate",
                        "avoided_completed_future_steps",
                    )
                },
            }
            for candidate in deployment["curve"]
        )
        for episode, delta in zip(VALIDATION_EPISODES, deltas):
            if delta >= 0.0:
                counterexamples.append(
                    {
                        "episode": episode,
                        "policy": policy,
                        "reason": "adaptive_not_better_than_matched_fixed",
                        "adaptive_minus_fixed_rgb_mae": delta,
                    }
                )
        if aggregate["generated_coverage"] == 1.0:
            counterexamples.append(
                {
                    "episode": "all",
                    "policy": policy,
                    "reason": "no_complete_chunk_avoided",
                    "adaptive_minus_fixed_rgb_mae": "",
                }
            )

    if all("uncertainty_rgb" in row for row in rows):
        rgb_rows = [
            {**row, "uncertainty_latent": row["uncertainty_rgb"]} for row in rows
        ]
        for policy in ("threshold", "smoothed_hysteretic"):
            diagnostic = select_threshold(
                rgb_rows,
                policy,
                execution="chunk",
                layout=OFFICIAL_LAYOUT,
            )
            curve.extend(
                {
                    "signal": "rgb_diagnostic_only",
                    "gating": False,
                    "policy": policy,
                    **{
                        key: candidate[key]
                        for key in (
                            "tau",
                            "coverage",
                            "generated_coverage",
                            "retained_rgb_mae",
                            "mean_retained_horizon",
                            "mean_generated_horizon",
                            "request_rate",
                            "avoided_completed_future_steps",
                        )
                    },
                }
                for candidate in diagnostic["curve"]
            )

    passing = [
        (policy, value)
        for policy, value in summary["adaptive"].items()
        if value["gate"]["passed"]
    ]
    if passing:
        selected_policy, selected_value = min(
            passing,
            key=lambda item: (
                item[1]["loeo"]["retained_rgb_mae"],
                item[1]["loeo"]["generated_coverage"],
                -item[1]["loeo"]["coverage"],
                -item[1]["deployment_threshold"],
            ),
        )
        summary["selected_policy"] = selected_policy
        summary["selected_deployment_threshold"] = selected_value[
            "deployment_threshold"
        ]
        summary["online_authorized"] = True
    else:
        summary["selected_policy"] = None
        summary["selected_deployment_threshold"] = None
        summary["online_authorized"] = False

    return {
        "summary": summary,
        "decisions": decisions,
        "curve": curve,
        "folds": folds,
        "costs": costs,
        "counterexamples": counterexamples,
        "report": build_report(summary),
    }


def build_report(summary: dict) -> str:
    status = (
        "CHUNK-ALIGNED PASS: ONLINE AUTHORIZED"
        if summary.get("online_authorized")
        else "CHUNK-ALIGNED FAIL: ONLINE NOT AUTHORIZED"
    )
    lines = [status + "\n"]
    adaptive = summary.get("adaptive", {})
    if adaptive and all("loeo" in value for value in adaptive.values()):
        lines.extend(
            [
                "\n| Policy | Retained coverage | Generated coverage | RGB MAE | Matched fixed | Wins | Gate |\n",
                "|---|---:|---:|---:|---:|---:|---|\n",
            ]
        )
        for policy, value in adaptive.items():
            wins = sum(delta < 0 for delta in value["episode_deltas"].values())
            lines.append(
                f"| {policy} | {value['loeo']['coverage']:.6f} | "
                f"{value['loeo']['generated_coverage']:.6f} | "
                f"{value['loeo']['retained_rgb_mae']:.6f} | "
                f"{value['matched_fixed']['retained_rgb_mae']:.6f} | "
                f"{wins}/16 | {'PASS' if value['gate']['passed'] else 'FAIL'} |\n"
            )
        lines.append(
            "\nGenerated coverage counts completed chunks only; it is not a wall-time claim.\n"
        )
    return "".join(lines)


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if rows:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("", encoding="utf-8")
    temporary.replace(path)


def write_chunk_aligned_outputs(
    output_dir: Path, result: dict, *, overwrite: bool
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError("refusing to write to non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_dir / "policy_decisions.csv", result["decisions"])
    _write_csv_atomic(output_dir / "policy_curve.csv", result["curve"])
    _write_csv_atomic(output_dir / "loeo_folds.csv", result["folds"])
    _write_csv_atomic(output_dir / "chunk_costs.csv", result["costs"])
    _write_csv_atomic(output_dir / "counterexamples.csv", result["counterexamples"])
    summary_path = output_dir / "adaptive_rollout_summary.json"
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(
        json.dumps(result["summary"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)
    report_path = output_dir / "report.md"
    report_tmp = report_path.with_suffix(".md.tmp")
    report_tmp.write_text(result["report"], encoding="utf-8")
    report_tmp.replace(report_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--correlation_summary", type=Path, required=True)
    parser.add_argument("--sampling_manifest", type=Path, required=True)
    parser.add_argument("--source_archive", type=Path, required=True)
    parser.add_argument("--previous_summary", type=Path, required=True)
    parser.add_argument("--code_commit", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_cli_identity(args: argparse.Namespace) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", args.code_commit) is None:
        raise ValueError("a concrete code commit is required")


def _validate_sources(correlation: dict, manifest: dict, archive_sha: str) -> None:
    if correlation.get("incomplete") is not False or manifest.get("incomplete") is not False:
        raise ValueError("source artifacts must be complete")
    if correlation.get("count_episode_steps") != 80:
        raise ValueError("correlation summary must contain 80 episode-steps")
    if correlation.get("k") != 4 or manifest.get("k") != 4:
        raise ValueError("formal evaluation requires K=4")
    if correlation.get("episodes") != list(VALIDATION_EPISODES):
        raise ValueError("correlation summary episodes do not match validation")
    if manifest.get("episodes") != correlation.get("episodes"):
        raise ValueError("sampling manifest episodes disagree with correlation summary")
    if manifest.get("seeds") != correlation.get("seeds"):
        raise ValueError("sampling manifest seeds disagree with correlation summary")
    if correlation.get("gate", {}).get("latent", {}).get("passed") is not True:
        raise ValueError("latent correlation gate did not pass")
    sampling = manifest.get("sampling", {})
    if sampling.get("history_len") != 1 or sampling.get("df_chunk_size") != 2:
        raise ValueError("sampling manifest does not use the official chunk layout")
    if sampling.get("total_len") != 6:
        raise ValueError("sampling manifest must contain five future frames")
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("source archive SHA256 does not match preregistration")


def main() -> None:
    args = build_parser().parse_args()
    validate_cli_identity(args)
    correlation = json.loads(args.correlation_summary.read_text(encoding="utf-8"))
    manifest = json.loads(args.sampling_manifest.read_text(encoding="utf-8"))
    previous = json.loads(args.previous_summary.read_text(encoding="utf-8"))
    archive_sha = _sha256(args.source_archive)
    _validate_sources(correlation, manifest, archive_sha)
    rows = load_formal_rows(args.metrics_csv)
    source_identity = {
        "metrics_csv": str(args.metrics_csv),
        "correlation_summary": str(args.correlation_summary),
        "sampling_manifest": str(args.sampling_manifest),
        "source_archive": str(args.source_archive),
        "source_archive_sha256": archive_sha,
        "previous_summary": str(args.previous_summary),
        "code_commit": args.code_commit,
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "data_root": manifest["data_root"],
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "sampling_git_commit": manifest["git_commit"],
        "seeds": manifest["seeds"],
        "sampling": manifest["sampling"],
    }
    result = evaluate_chunk_aligned(rows, source_identity, previous)
    write_chunk_aligned_outputs(args.output_dir, result, overwrite=args.overwrite)
    print(result["report"], end="")


if __name__ == "__main__":
    main()
