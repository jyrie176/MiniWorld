# Expanded DROID training split

- Source: `GEAR-Dreams/DreamZero-DROID-Data`
- Local dataset: `/data/miniworld/datasets/droid-expanded-0-1063`
- Episodes: 0-1063, contiguous
- Count: 1,064 successful episodes
- Frame-length range: 46-1,481
- Files: 1,064 parquet and 1,064
  `observation.images.exterior_image_1_left` MP4 files
- `episodes.jsonl` SHA-256:
  `c57b4c83e1134be2a4a7ac23e6959ad1216270d6d5e085cd60090619eb938793`
- Sorted parquet/video content digest:
  `2645485be04d479fa3571351462846ffaf801b3b7a16492e3cd71df5ec2737f1`

Audit on 2026-08-27 loaded every episode through
`LeRobotActionDataset(randomize=False)`. All 1,064 returned video tensors with
shape `(21, 240, 320, 3)`, action tensors with shape `(20, 7)`, and finite
values. Validation 1064-1079 and sealed test 1080-1095 are disjoint by episode
index.

Chunk 000 was downloaded directly from the source repository. Episodes
1000-1063 in chunk 001 are hard-linked from the prior audited 64-episode local
dataset; inode equality was verified, so this does not duplicate media bytes.
