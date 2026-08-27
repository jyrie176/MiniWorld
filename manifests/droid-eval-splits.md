# DROID evaluation splits

These episode-disjoint splits close the MiniWorld-B (0.12B) sanity evaluation
gate and remain fixed for the later 0.55B zero-shot, continued-training, and
adaptive-rollout experiments.

| Split | Episodes | Count | Local dataset |
| --- | --- | ---: | --- |
| train | 1000-1063 | 64 | `/data/miniworld/datasets/droid-mini-1000-1063` |
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
| train | `1e0d7337e25fb4b01e90d212c8d6ec63672dd1c0c90e2d084deddc09db9cd97c` | `44935c545740a67b48c6ed063ce20ff00f489c4fd7e21e31ef2205fbbca8b800` |
| validation | `ac3e1dbff5a22732b54c47834751714f31302bf0aca859b4dc78a2809203ae70` | `aba246351af39e3963f9c2c77f285b8524a3b6792390a6a99728a1bee85f8d00` |
| test | `95018295230086d934e712f00f9f45dee91473a608568353aff8750a54b03e71` | `f339037595c8c1f1f3d4926611a32e29ee783b3af03c05a2e0adb1b4411a5fa6` |

Validation and test each passed a full 16/16 decode and schema check with video
shape `(21, 240, 320, 3)`, action shape `(20, 7)`, and finite video/action
tensors.
