# Official MiniWorld-0.55B expanded-data 1k-step baseline

Result date: 2026-08-27

## Question

The 64-episode continued-training run reduced training loss but degraded held-out
quality. This controlled experiment tests whether data repetition was the main
cause by expanding training from 64 to 1,064 DROID episodes while keeping the
official initialization, optimizer, global batch, LR, EMA, precision, and
1,000-step compute budget unchanged. Validation remains episodes 1064-1079;
test episodes 1080-1095 remain sealed.

## Expanded data

- Source: `GEAR-Dreams/DreamZero-DROID-Data`
- Episodes: 0-1063, contiguous, 1,064/1,064 marked successful
- Files: 1,064 parquet files and 1,064
  `observation.images.exterior_image_1_left` videos
- Length: minimum 46 frames, maximum 1,481; all exceed the 21-frame window
- Full audit: all 1,064 samples decoded with video shape `(21, 240, 320, 3)`,
  action shape `(20, 7)`, and finite values
- `episodes.jsonl` SHA-256:
  `c57b4c83e1134be2a4a7ac23e6959ad1216270d6d5e085cd60090619eb938793`
- Parquet/video content digest:
  `2645485be04d479fa3571351462846ffaf801b3b7a16492e3cd71df5ec2737f1`

Only chunk 000 was downloaded. Chunk 001 episodes 1000-1063 are hard-linked
from the previously audited dataset, avoiding duplicate media storage.

## Training setup and stability

- Model: official 0.55B DROID checkpoint, six latent frames
- Runtime: two V100 GPUs, DDP, per-rank batch 1 / global batch 2
- Optimizer: AdamW, LR `2e-5`, EMA `0.9999`, gradient clip `1.0`
- Precision/backend: FP16 GradScaler / PyTorch SDPA
- Budget: 1,000 effective steps, approximately 1.88 dataset passes
- Seed base: `20260827`
- W&B: `csp6jnw6`

One overflow occurred before logged step 90. GradScaler safely skipped the
attempt and reduced the scale from 65,536 to 32,768; no further overflow
occurred. Mean logged loss was 0.1169 through step 531 and 0.1088 from 532 to
1000. Final loss was 0.0847 and steady throughput was 0.85 step/s / 1.70
sample/s. Checkpoints were saved at effective steps 531 and 1000.

## Real-action validation

All evaluations use the fixed 16-episode validation set, identical seeds, 20
sampling steps, CFG 2.0, FP16/SDPA, and a six-frame active stream window.

| Weights | Step 531 | Step 1000 | Official zero-shot | 64-episode step 1000 |
| --- | ---: | ---: | ---: | ---: |
| model | 8.7416 | 6.3781 | **5.4680** | 6.4532 |
| EMA | 5.4804 | 5.5007 | **5.4680** | 5.4759 |

Final expanded-data model improved over the 64-episode model by 0.0751 mean
MAE and on 10/16 paired episodes, but remained 0.9101 worse than the official
checkpoint and beat it on only 1/16 episodes. EMA drifted by +0.0326 and beat
the official checkpoint on 5/16 episodes. Persistence remained 4.4681.

## Final-model action ablation

| Condition | Expanded 1064 | 64 episodes | Official zero-shot |
| --- | ---: | ---: | ---: |
| real | 6.3781 | 6.4532 | 5.4680 |
| zero-valued | 9.2527 | 8.4448 | 9.3679 |
| reversed | 9.0868 | 9.2114 | 9.2408 |

Expanded-data real actions beat zero on 15/16 episodes, reverse on 15/16, and
both on 14/16. These counts exactly restore the official action-control result
and improve over the 64-episode continued model's 13/16, 14/16, and 12/16.
More diverse data therefore protects action generalization even though overall
RGB quality still degrades.

## Checkpoint and decision

The final checkpoint contains 420 model tensors, 420 EMA tensors, optimizer
state, epoch 2, global step 1000, metadata, and GradScaler state. The final
scale is 32,768 with growth tracker 914.

Data scarcity was a real contributor but not the main failure. Expanding from
64 to 1,064 episodes restored action-control consistency and slightly improved
the final model, yet the same all-parameter `LR=2e-5` recipe still damages the
released checkpoint. Do not extend this recipe to 5k. The next controlled
baseline experiment should keep the expanded data and lower adaptation strength
(first choice: LR `2e-6` with the same 1k-step budget). If that also fails,
freeze the official zero-shot checkpoint for the uncertainty mainline.

## Artifacts

```text
/data/miniworld/datasets/droid-expanded-0-1063
/data/miniworld/outputs/droid-official-055b-expanded1064-continued-1k-lf6-ddp2
/data/miniworld/experiments/droid-official-055b-expanded1064-continued-1k-lf6-ddp2/train.log
/data/miniworld/experiments/official-055b-expanded1064-continued-1k-validation
```
