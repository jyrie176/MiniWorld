# Official MiniWorld-0.55B 1k-step continued-training baseline

Result date: 2026-08-27

## Mainline mapping

Phase 4 / E2-E3: measure whether short continued training on the local 64-episode
DROID subset improves the released 0.55B model before freezing the baseline for
uncertainty-aware innovation. Validation episodes 1064-1079 were used for model
selection; test episodes 1080-1095 remained sealed.

## Setup

- Initialization: official `MiniWorld_0_5b_droid.pt`
- Train data: DROID episodes 1000-1063
- Runtime: two V100 GPUs, DDP, FP16 GradScaler, PyTorch SDPA
- Window: six latent frames / 21 RGB frames
- Batch: one per rank, global batch two
- Optimization: 1,000 effective steps, AdamW, LR `2e-5`, EMA `0.9999`,
  gradient clipping `1.0`
- Seed base: `20260827`
- W&B run: `k2y8o0jh`

Checkpoints were saved at effective steps 320, 639, 959, and 1000. The 639 and
959 labels, rather than 640 and 960, reflect one safely skipped FP16 overflow at
attempted step 340; checkpoint cadence is epoch-based while `global_step`
counts successful optimizer updates.

## Training stability

The run exited normally at global step 1000. One non-finite gradient was
detected and skipped at attempted step 340; GradScaler reduced the scale from
65,536 to 32,768, after which no further step was skipped. Final loss was
0.053203, final gradient norm was 0.1861, and steady throughput was about 0.85
step/s or 1.70 sample/s.

| Effective-step interval | Logged finite points | Mean loss | Median loss | Range |
| --- | ---: | ---: | ---: | ---: |
| 1-320 | 32 | 0.1149 | 0.1078 | 0.0505-0.2454 |
| 321-639 | 31 | 0.1437 | 0.1038 | 0.0556-0.4989 |
| 640-959 | 32 | 0.0908 | 0.0811 | 0.0524-0.1810 |
| 960-1000 | 5 | 0.0728 | 0.0802 | 0.0532-0.0894 |

The loss trend is numerically stable but cannot establish generalization.

## Fixed validation results

All runs used real actions, identical per-episode seeds, 20 sampling steps, CFG
2.0, FP16/SDPA, and the same 16 validation episodes. Lower RGB MAE is better.

| Weights | Step 320 | Step 639 | Step 1000 | Official zero-shot |
| --- | ---: | ---: | ---: | ---: |
| model | 6.7049 | 7.2835 | 6.4532 | 5.4680 |
| EMA | 5.4695 | 5.4742 | 5.4759 | 5.4680 |

Against the official checkpoint episode by episode, model weights improved on
0/16, 0/16, and 1/16 episodes at steps 320, 639, and 1000. EMA changed only
slightly: its mean deltas were +0.0015, +0.0062, and +0.0079. Persistence
remained 4.4681; final model beat it on only 1/16 episodes and final EMA on
5/16.

Because final model weights changed materially, their action ablation was
expanded:

| Condition | Mean MAE | Real paired wins |
| --- | ---: | ---: |
| real | 6.4532 | - |
| zero-valued | 8.4448 | 13/16 |
| reversed | 9.2114 | 14/16 |

Real remained best against both on 12/16 episodes. The action path therefore
still functions, but its advantage weakened relative to the official 15/16,
15/16, and 14/16 counts. Final EMA was not expanded to zero/reverse because its
real MAE moved by only +0.0079, below a meaningful change at this validation
size.

## Checkpoint integrity and decision

The final checkpoint is 8,865,354,022 bytes and contains 420 model tensors, 420
EMA tensors, optimizer state, epoch 32, global step 1000, metadata, and
GradScaler state. Metadata records 0.5B, six trained latent frames, action
conditioning dimension 28, FP16, SDPA, and gradient clip 1.0. The saved scaler
is 32,768 with growth tracker 660.

This experiment rejects extending the same `LR=2e-5`, all-parameter,
64-episode recipe directly to 5k steps. Training loss improved while held-out
model quality worsened, which is consistent with small-data overfitting or
catastrophic forgetting. EMA prevented most damage but produced no benefit.
The official zero-shot checkpoint remains the Phase 4 quality baseline. The
next bounded experiment should test a more conservative adaptation recipe
(lower LR and/or partial freezing) before deciding whether continued training
is useful; if it still fails, Phase 4 should freeze the official checkpoint and
move to uncertainty-error correlation.

## Artifacts

```text
/data/miniworld/outputs/droid-official-055b-continued-1k-lf6-ddp2/
/data/miniworld/experiments/droid-official-055b-continued-1k-lf6-ddp2/train.log
/data/miniworld/experiments/official-055b-continued-1k-validation/
```
