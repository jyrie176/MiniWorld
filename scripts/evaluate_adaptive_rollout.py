#!/usr/bin/env python3
"""Evaluate uncertainty-aware rollout policies without rerunning the model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from miniworld.rollout_policy import (
    EpisodePolicyResult,
    aggregate_policy_results,
    evaluate_offline_gate,
    fixed_policy,
    matched_fixed_baseline,
    run_loeo,
    score_policy_trace,
    select_threshold,
)


VALIDATION_EPISODES = tuple(range(1064, 1080))
SEALED_TEST_EPISODES = frozenset(range(1080, 1096))
EXPECTED_ARCHIVE_SHA256 = (
    "4fe4b2fb4b59dce77199b46fb82109a3d1197378ee6fd1a1b1df1dca2a354d86"
)


def _first(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise ValueError(f"missing required column; expected one of {tuple(names)}")


def load_formal_rows(path: Path) -> list[dict]:
    """Load, normalize, and strictly validate an uncertainty metrics CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("metrics CSV has no header")
        missing_seed_columns = [
            f"error_seed_{seed}"
            for seed in range(4)
            if f"error_seed_{seed}" not in reader.fieldnames
        ]
        if missing_seed_columns:
            raise ValueError("formal evaluation requires four seed error columns")
        normalized = []
        seen = set()
        for raw in reader:
            episode = int(_first(raw, ("episode",)))
            if episode in SEALED_TEST_EPISODES:
                raise ValueError("metrics CSV references the sealed test split")
            if episode not in VALIDATION_EPISODES:
                raise ValueError("only validation episodes 1064-1079 are allowed")
            step = int(_first(raw, ("future_latent_step", "step")))
            if (episode, step) in seen:
                raise ValueError("duplicate episode-step row")
            seen.add((episode, step))
            row = {
                "episode": episode,
                "future_latent_step": step,
                "uncertainty_latent": float(
                    _first(raw, ("uncertainty_latent", "latent_variance"))
                ),
                "error_rgb": float(_first(raw, ("error_rgb", "rgb_mae"))),
                **{
                    f"error_seed_{seed}": float(raw[f"error_seed_{seed}"])
                    for seed in range(4)
                },
            }
            if "uncertainty_rgb" in raw and raw["uncertainty_rgb"] != "":
                row["uncertainty_rgb"] = float(raw["uncertainty_rgb"])
            numeric_values = [
                value for value in row.values() if isinstance(value, float)
            ]
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError("metrics CSV contains NaN or Inf")
            normalized.append(row)

    episodes = sorted({row["episode"] for row in normalized})
    for episode in episodes:
        steps = sorted(
            row["future_latent_step"]
            for row in normalized
            if row["episode"] == episode
        )
        if steps != list(range(1, 6)):
            raise ValueError("each episode must contain five future steps")
    if tuple(episodes) != VALIDATION_EPISODES:
        raise ValueError("formal evaluation requires validation episodes 1064-1079")
    return sorted(
        normalized, key=lambda row: (row["episode"], row["future_latent_step"])
    )


def _rows_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    return {
        episode: [row for row in rows if row["episode"] == episode]
        for episode in VALIDATION_EPISODES
    }


def _score_fixed(rows: list[dict], horizon: int) -> list[EpisodePolicyResult]:
    results = []
    for episode, episode_rows in _rows_by_episode(rows).items():
        results.append(
            score_policy_trace(
                episode,
                fixed_policy(horizon),
                [row["error_rgb"] for row in episode_rows],
                np.asarray(
                    [
                        [row[f"error_seed_{seed}"] for row in episode_rows]
                        for seed in range(4)
                    ]
                ),
            )
        )
    return results


def _decision_row(
    policy: str, result: EpisodePolicyResult, *, tau: float | None = None
) -> dict:
    return {
        "episode": result.episode,
        "policy": policy,
        "tau": tau,
        "decisions": "|".join(decision.value for decision in result.trace.decisions),
        "requested_observation_at": result.trace.requested_observation_at,
        "retained_horizon": result.trace.retained_horizon,
        "generated_horizon": result.trace.generated_horizon,
        "retained_error_numerator": result.retained_error_numerator,
        "retained_count": result.retained_count,
        "discarded_error_numerator": result.discarded_error_numerator,
        "discarded_count": result.discarded_count,
        "per_seed_retained_error": json.dumps(result.per_seed_retained_error),
    }


def evaluate_offline(rows: list[dict], source_identity: dict) -> dict:
    """Run fixed and leakage-safe adaptive evaluation on frozen rows."""
    if len(rows) != 80:
        raise ValueError("formal evaluation requires exactly 80 rows")
    high_error_cutoff = float(np.percentile([row["error_rgb"] for row in rows], 75))
    summary: dict = {
        "source": source_identity,
        "primary_signal": "latent_population_variance",
        "primary_target_coverage": 0.80,
        "validation_episodes": list(VALIDATION_EPISODES),
        "high_error_cutoff": high_error_cutoff,
        "fixed": {},
        "adaptive": {},
    }
    decisions = []
    curve = []
    folds = []
    counterexamples = []

    for horizon in range(1, 6):
        fixed_results = _score_fixed(rows, horizon)
        metrics = aggregate_policy_results(
            fixed_results, high_error_cutoff=high_error_cutoff
        )
        summary["fixed"][str(horizon)] = metrics
        decisions.extend(
            _decision_row(f"fixed_h{horizon}", result) for result in fixed_results
        )

    for policy in ("threshold", "smoothed_hysteretic"):
        policy_folds, results = run_loeo(rows, policy)
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
            - matched["episode_errors"][episode]
            for episode in VALIDATION_EPISODES
        ]
        gate = evaluate_offline_gate(aggregate, matched, deltas)
        deployment = select_threshold(rows, policy)
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
                )
            },
        }
        folds.extend(policy_folds)
        for fold, result in zip(policy_folds, results):
            decisions.append(_decision_row(policy, result, tau=float(fold["tau"])))
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

    if all("uncertainty_rgb" in row for row in rows):
        rgb_rows = [
            {**row, "uncertainty_latent": row["uncertainty_rgb"]} for row in rows
        ]
        for policy in ("threshold", "smoothed_hysteretic"):
            diagnostic = select_threshold(rgb_rows, policy)
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
                        )
                    },
                }
                for candidate in diagnostic["curve"]
            )

    authorized = any(
        policy_summary["gate"]["passed"]
        for policy_summary in summary["adaptive"].values()
    )
    summary["online_authorized"] = authorized
    status = (
        "OFFLINE PASS: ONLINE AUTHORIZED"
        if authorized
        else "OFFLINE FAIL: ONLINE NOT AUTHORIZED"
    )
    report_lines = [
        f"# Adaptive rollout offline evaluation\n\n{status}\n",
        "| Policy | Coverage | Retained RGB MAE | Matched fixed MAE | Wins | Gate |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for policy, policy_summary in summary["adaptive"].items():
        report_lines.append(
            f"| {policy} | {policy_summary['loeo']['coverage']:.6f} | "
            f"{policy_summary['loeo']['retained_rgb_mae']:.6f} | "
            f"{policy_summary['matched_fixed']['retained_rgb_mae']:.6f} | "
            f"{sum(delta < 0 for delta in policy_summary['episode_deltas'].values())}/16 | "
            f"{'PASS' if policy_summary['gate']['passed'] else 'FAIL'} |\n"
        )
    report_lines.append(
        "\nRetained horizon measures trusted predictions; generated horizon measures "
        "actual K=4 compute. RGB uncertainty curves are diagnostic only.\n"
    )
    return {
        "summary": summary,
        "decisions": decisions,
        "curve": curve,
        "folds": folds,
        "counterexamples": counterexamples,
        "report": "".join(report_lines),
    }


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if rows:
        fieldnames = list(rows[0])
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("", encoding="utf-8")
    temporary.replace(path)


def write_offline_outputs(output_dir: Path, result: dict, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError("refusing to write to non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_dir / "policy_decisions.csv", result["decisions"])
    _write_csv_atomic(output_dir / "policy_curve.csv", result["curve"])
    _write_csv_atomic(output_dir / "loeo_folds.csv", result["folds"])
    _write_csv_atomic(output_dir / "counterexamples.csv", result["counterexamples"])
    summary_path = output_dir / "adaptive_rollout_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(result["summary"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    report_path = output_dir / "report.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text(result["report"], encoding="utf-8")
    temporary_report.replace(report_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--correlation_summary", type=Path, required=True)
    parser.add_argument("--source_archive", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    correlation = json.loads(args.correlation_summary.read_text(encoding="utf-8"))
    archive_sha256 = _sha256(args.source_archive)
    if correlation.get("incomplete") is not False:
        raise ValueError("correlation source is incomplete")
    if correlation.get("count_episode_steps") != 80 or correlation.get("k") != 4:
        raise ValueError("correlation source must contain 80 K=4 episode-steps")
    if correlation.get("gate", {}).get("latent", {}).get("passed") is not True:
        raise ValueError("latent uncertainty gate did not pass")
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("source archive SHA256 does not match preregistration")
    rows = load_formal_rows(args.metrics_csv)
    result = evaluate_offline(
        rows,
        {
            "metrics_csv": str(args.metrics_csv),
            "correlation_summary": str(args.correlation_summary),
            "source_archive": str(args.source_archive),
            "archive_sha256": archive_sha256,
            "correlation_estimators": correlation["estimators"],
            "correlation_gate": correlation["gate"],
            "seeds": correlation["seeds"],
        },
    )
    write_offline_outputs(args.output_dir, result, overwrite=args.overwrite)
    print(result["report"], end="")


if __name__ == "__main__":
    main()
