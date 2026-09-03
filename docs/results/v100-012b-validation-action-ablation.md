# MiniWorld-B validation action ablation

Result date: 2026-08-27

## Mainline mapping

Phase 3 / E1: close the 0.12B from-scratch sanity gate before moving to the
0.55B continued-training baseline. This is not a final model-quality claim.

## Setup

- Model: MiniWorld-B (0.12B), 123,852,364 trainable parameters
- Training data: DROID episodes 1000-1063
- Validation data: held-out DROID episodes 1064-1079
- Checkpoint: `epoch_0079_step_00005000.pt`, EMA weights
- Context/rollout: 1 observed latent frame, 6 total latent frames (21 RGB frames)
- Precision/backend: FP16 / PyTorch SDPA on one V100
- Seed: `20260827 + sample_index`, identical across action variants
- Metric: mean absolute RGB error on future frames 1-20 in the 0-255 space
- Test episodes 1080-1095 were not evaluated

The three conditions are real normalized actions, zero-valued normalized
actions, and deterministically time-reversed actions. The CLI name `shuffle`
currently implements the reverse condition. Zero is not the model's learned
null/unconditional action.

## Results

| Condition | Mean MAE | Median MAE |
| --- | ---: | ---: |
| real | 12.700 | 12.181 |
| zero-valued | **12.293** | **12.143** |
| reversed | 12.701 | 12.228 |

Paired episode counts:

- real better than zero: 3/16
- real better than reverse: 10/16
- real best against both: 3/16

Mean paired deltas relative to real:

- `zero - real = -0.407` (zero is better on average)
- `reverse - real = +0.002` (effectively tied)

The generated videos do respond to the action tensor: mean output MAE is 2.772
for real versus zero and 1.673 for real versus reverse. Response alone is not
correct control, however, because the real condition is not more accurate.

## Persistence baseline

Repeating the observed first RGB frame gives mean MAE 4.584 and median MAE
3.897, and beats the generated real-action video on all 16 episodes. Pixel MAE
strongly rewards static/sharp predictions and does not measure perceptual video
quality by itself, but this result confirms that the current 0.12B checkpoint
is not a useful held-out predictive baseline.

## Interpretation

The 0.12B run successfully validates the V100 FP16/SDPA training and evaluation
pipeline, but it does not pass the held-out action-conditioning quality gate:

1. Actions change the output, so the conditioning path is active.
2. Correct action timing has no aggregate advantage over reversed timing.
3. Zero-valued normalized actions outperform real actions on this metric.
4. The model underperforms a first-frame persistence baseline on every episode.

The first 21 frames of some episodes contain little visible motion, and RGB MAE
can favor blur or persistence. Future formal evaluation should add multiple
deterministic windows per episode and perceptual/feature metrics. These limits
do not reverse the current conclusion: this checkpoint provides no evidence of
reliable held-out action control.

This is a stop signal for additional unguided 0.12B training. The next work
should diagnose only correctness issues that would also block 0.55B (notably
action/video temporal alignment), finish the DDP/resource gate, and then move
to the pretrained 0.55B zero-shot baseline.

## Artifacts

```text
/data/miniworld/experiments/droid-v100-64ep-5k-validation-action-ablation/
  metrics_summary.json
  metrics_per_episode.csv
  comparison_best-real_episode_1074.png
  comparison_worst-real_episode_1069.png
  ema-real/pred/*.mp4
  ema-zero/pred/*.mp4
  ema-shuffle/pred/*.mp4
```
