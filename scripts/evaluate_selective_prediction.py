#!/usr/bin/env python3
"""Evaluate uncertainty as a selective-prediction reliability signal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from miniworld.selective_prediction import (
    DEFAULT_COVERAGES,
    loeo_selective_metrics,
    risk_coverage_curve,
)
from scripts.evaluate_adaptive_rollout import load_formal_rows


SIGNALS = {
    "latent_disagreement": "uncertainty_latent",
    "rgb_disagreement": "uncertainty_rgb",
}


def _signal_rows(rows: list[dict], column: str) -> list[dict]:
    if not all(column in row for row in rows):
        raise ValueError(f"formal evaluation requires {column}")
    return [
        {
            "episode": row["episode"],
            "uncertainty": row[column],
            "error": row["error_rgb"],
        }
        for row in rows
    ]


def evaluate_selective_prediction(rows: list[dict], source_identity: dict) -> dict:
    """Compute global risk curves and leakage-safe LOEO operating points."""
    if len(rows) != 80:
        raise ValueError("formal evaluation requires exactly 80 validation rows")
    summary = {
        "source": source_identity,
        "unit": "episode_future_latent_step",
        "interpretation": "reject means request a fresh observation",
        "target_coverages": list(DEFAULT_COVERAGES),
        "signals": {},
    }
    curve_rows = []
    fold_rows = []
    for name, column in SIGNALS.items():
        signal_rows = _signal_rows(rows, column)
        global_metrics = risk_coverage_curve(signal_rows)
        loeo = {}
        for coverage in DEFAULT_COVERAGES:
            metrics = loeo_selective_metrics(
                signal_rows, target_coverage=coverage
            )
            loeo[str(coverage)] = {
                key: value for key, value in metrics.items() if key != "folds"
            }
            for fold in metrics["folds"]:
                fold_rows.append(
                    {
                        "signal": name,
                        "target_coverage": coverage,
                        **fold,
                        "threshold_source_episodes": "|".join(
                            str(value)
                            for value in fold["threshold_source_episodes"]
                        ),
                    }
                )
        summary["signals"][name] = {
            "aurc": global_metrics["aurc"],
            "random_aurc": global_metrics["random_aurc"],
            "oracle_aurc": global_metrics["oracle_aurc"],
            "aurc_improvement_vs_random": (
                global_metrics["random_aurc"] - global_metrics["aurc"]
            ),
            "oracle_gap": global_metrics["aurc"]
            - global_metrics["oracle_aurc"],
            "operating_points": global_metrics["operating_points"],
            "loeo": loeo,
        }
        curve_rows.extend(
            {"signal": name, **point} for point in global_metrics["curve"]
        )
    report = _render_report(summary)
    return {
        "summary": summary,
        "curve": curve_rows,
        "folds": fold_rows,
        "report": report,
    }


def _render_report(summary: dict) -> str:
    lines = [
        "# SELECTIVE PREDICTION reliability evaluation",
        "",
        "A rejected prediction means request a fresh observation; it is not an "
        "online speedup claim. Test episodes 1080-1095 remain sealed.",
        "",
        "| signal | AURC | random | oracle | improvement vs random |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["signals"].items():
        lines.append(
            f"| {name} | {metrics['aurc']:.6f} | "
            f"{metrics['random_aurc']:.6f} | {metrics['oracle_aurc']:.6f} | "
            f"{metrics['aurc_improvement_vs_random']:.6f} |"
        )
    lines.extend(["", "## Leakage-safe LOEO operating points", ""])
    for name, metrics in summary["signals"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| target | realized | mean MAE | P90 | worst |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for target, point in metrics["loeo"].items():
            lines.append(
                f"| {target} | {point['realized_coverage']:.4f} | "
                f"{point['mean_error']:.4f} | {point['p90_error']:.4f} | "
                f"{point['worst_error']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, result: dict, *, overwrite: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selective_prediction_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "risk_coverage_curve.csv", result["curve"])
    _write_csv(output_dir / "loeo_folds.csv", result["folds"])
    (output_dir / "report.md").write_text(result["report"], encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = load_formal_rows(args.metrics_csv)
    result = evaluate_selective_prediction(
        rows,
        source_identity={
            "metrics_csv": str(args.metrics_csv.resolve()),
            "sha256": _sha256(args.metrics_csv),
        },
    )
    write_outputs(args.output_dir, result, overwrite=args.overwrite)
    print(result["report"])


if __name__ == "__main__":
    main()
