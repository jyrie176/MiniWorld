#!/usr/bin/env python3
"""Evaluate deterministic DROID action-ablation sample directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _mae(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().mean())


def compute_episode_metrics(ground_truth, real, zero, reverse):
    """Compute comparable RGB metrics for one episode."""
    videos = (ground_truth, real, zero, reverse)
    if any(video.ndim != 4 for video in videos):
        raise ValueError("videos must have shape (T, H, W, C)")
    if any(video.shape != ground_truth.shape for video in videos[1:]):
        raise ValueError("ground-truth and generated videos must have identical shapes")
    if ground_truth.shape[0] < 2:
        raise ValueError("videos must contain one observed and at least one future frame")

    gt_future = ground_truth[1:]
    real_future = real[1:]
    zero_future = zero[1:]
    reverse_future = reverse[1:]
    persistence = ground_truth[:1].expand_as(gt_future)
    return {
        "mae_real": _mae(real_future, gt_future),
        "final_mae_real": _mae(real_future[-1], gt_future[-1]),
        "mae_zero": _mae(zero_future, gt_future),
        "final_mae_zero": _mae(zero_future[-1], gt_future[-1]),
        "mae_reverse": _mae(reverse_future, gt_future),
        "final_mae_reverse": _mae(reverse_future[-1], gt_future[-1]),
        "output_mae_real_zero": _mae(real_future, zero_future),
        "output_mae_real_reverse": _mae(real_future, reverse_future),
        "mae_persistence": _mae(persistence, gt_future),
        "gt_final_motion_mae_from_frame0": _mae(ground_truth[0], ground_truth[-1]),
    }


def _read_video(path: Path) -> torch.Tensor:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0))
    return torch.from_numpy(reader.get_batch(range(len(reader))).asnumpy())


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _write_comparison(path: Path, videos: list[torch.Tensor], labels: list[str]) -> None:
    from PIL import Image, ImageDraw

    frame_ids = np.linspace(0, videos[0].shape[0] - 1, num=5, dtype=int)
    height, width = videos[0].shape[1:3]
    label_width = 72
    canvas = Image.new("RGB", (label_width + 5 * width, len(videos) * height), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (video, label) in enumerate(zip(videos, labels)):
        draw.text((4, row * height + 4), label, fill="black")
        for column, frame_id in enumerate(frame_ids):
            frame = Image.fromarray(video[frame_id].cpu().numpy())
            canvas.paste(frame, (label_width + column * width, row * height))
    canvas.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate DROID action ablations")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sample_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed_base", type=int, default=20260827)
    return parser


def main() -> None:
    from miniworld.data.droid import LeRobotActionDataset

    args = build_parser().parse_args()
    sample_root = Path(args.sample_root)
    output_dir = Path(args.output_dir or args.sample_root)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    rows = []
    episode_videos = {}
    for index, episode in enumerate(dataset.samples):
        ground_truth = ((dataset[index]["videos"] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        real = _read_video(sample_root / "real" / "pred" / f"sample_{index:04d}.mp4")
        zero = _read_video(sample_root / "zero" / "pred" / f"sample_{index:04d}.mp4")
        reverse = _read_video(sample_root / "shuffle" / "pred" / f"sample_{index:04d}.mp4")
        metrics = compute_episode_metrics(ground_truth, real, zero, reverse)
        metrics.update(
            episode=episode,
            real_better_zero=metrics["mae_real"] < metrics["mae_zero"],
            real_better_reverse=metrics["mae_real"] < metrics["mae_reverse"],
        )
        metrics["real_best"] = metrics["real_better_zero"] and metrics["real_better_reverse"]
        rows.append(metrics)
        episode_videos[episode] = [ground_truth, real, zero, reverse]

    columns = ["episode"] + [key for key in rows[0] if key != "episode"]
    with (output_dir / "metrics_per_episode.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": "validation",
        "episodes": [int(rows[0]["episode"]), int(rows[-1]["episode"])],
        "count": len(rows),
        "checkpoint": "official MiniWorld 0.55B DROID bare state dict",
        "seed_base": args.seed_base,
        "future_frames": 20,
        "rgb_scale": "0-255",
        "zero_note": "zero-valued normalized action, not learned null/unconditional",
        "reverse_note": "CLI variant shuffle deterministically reverses time",
    }
    for variant in ("real", "zero", "reverse"):
        summary[f"mae_{variant}"] = _summary([row[f"mae_{variant}"] for row in rows])
    summary["wins"] = {
        "real_better_zero": sum(row["real_better_zero"] for row in rows),
        "real_better_reverse": sum(row["real_better_reverse"] for row in rows),
        "real_best_both": sum(row["real_best"] for row in rows),
    }
    summary["mean_output_difference"] = {
        "real_vs_zero": float(np.mean([row["output_mae_real_zero"] for row in rows])),
        "real_vs_reverse": float(np.mean([row["output_mae_real_reverse"] for row in rows])),
    }
    summary["mean_delta_vs_real"] = {
        "zero_minus_real": float(np.mean([row["mae_zero"] - row["mae_real"] for row in rows])),
        "reverse_minus_real": float(np.mean([row["mae_reverse"] - row["mae_real"] for row in rows])),
    }
    persistence = [row["mae_persistence"] for row in rows]
    summary["persistence"] = {
        "mean": float(np.mean(persistence)),
        "median": float(np.median(persistence)),
        "real_better_count": sum(row["mae_real"] < row["mae_persistence"] for row in rows),
        "count": len(rows),
        "mean_real_minus_persistence": float(
            np.mean([row["mae_real"] - row["mae_persistence"] for row in rows])
        ),
        "note": "repeat observed RGB frame 0",
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ranked = sorted(rows, key=lambda row: row["mae_real"] - min(row["mae_zero"], row["mae_reverse"]))
    for label, row in (("best-real", ranked[0]), ("worst-real", ranked[-1])):
        episode = int(row["episode"])
        _write_comparison(
            output_dir / f"comparison_{label}_episode_{episode}.png",
            episode_videos[episode],
            ["ground truth", "real", "zero", "reverse"],
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
