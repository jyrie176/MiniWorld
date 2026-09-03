# DROID evaluation splits

These episode-disjoint splits close the MiniWorld-B (0.12B) sanity evaluation
gate and remain fixed for the later 0.55B zero-shot, continued-training, and
adaptive-rollout experiments.

| Split | Episodes | Count | Local dataset |
| --- | --- | ---: | --- |
| train | 0-1063 | 1,064 | `/data/miniworld/datasets/droid-expanded-0-1063` |
| validation | 1064-1079 | 16 | `/data/miniworld/datasets/droid-validation-1064-1079` |
| test | 1080-1095 | 16 | `/data/miniworld/datasets/droid-test-1080-1095` |

Only validation may be used for method selection, thresholds, checkpoint
selection, or debugging model quality. Test is reserved for final reporting;
decoding and schema integrity checks do not count as model evaluation.

All episodes are marked successful and have at least 75 raw frames, exceeding
the current 21-frame short-horizon requirement. Each local split contains the
LeRobot parquet plus only the
`observation.images.exterior_image_1_left` camera video.

## Integrity

Hashes were computed on 2026-08-27. `content digest` is SHA-256 over the sorted
per-file SHA-256 lines for all parquet and video files in the split.

| Split | `episodes.jsonl` SHA-256 | Content digest |
| --- | --- | --- |
| train | `c57b4c83e1134be2a4a7ac23e6959ad1216270d6d5e085cd60090619eb938793` | `2645485be04d479fa3571351462846ffaf801b3b7a16492e3cd71df5ec2737f1` |
| validation | `ac3e1dbff5a22732b54c47834751714f31302bf0aca859b4dc78a2809203ae70` | `aba246351af39e3963f9c2c77f285b8524a3b6792390a6a99728a1bee85f8d00` |
| test | `95018295230086d934e712f00f9f45dee91473a608568353aff8750a54b03e71` | `f339037595c8c1f1f3d4926611a32e29ee783b3af03c05a2e0adb1b4411a5fa6` |

Validation and test each passed a full 16/16 decode and schema check with video
shape `(21, 240, 320, 3)`, action shape `(20, 7)`, and finite video/action
tensors.

The original 64-episode train split remains at
`/data/miniworld/datasets/droid-mini-1000-1063` for historical reproduction.
The expanded train split passed the same full audit on all 1,064 episodes; see
`manifests/droid-train-expanded-0-1063.md`.
