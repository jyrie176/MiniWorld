# Official MiniWorld-0.55B validation action ablation

Result date: 2026-08-27

## Mainline mapping

Phase 4 / E2: establish the released 0.55B DROID checkpoint as the zero-shot
baseline before any continued training or uncertainty-aware innovation work.

## Setup

- Checkpoint: upstream `MiniWorld_0_5b_droid.pt`, a bare state dict with
  556,675,120 parameters and a 64-latent-frame RoPE window
- SHA-256: `e4b118befe88cee7338400c5510fdd497212b9b1988034290030b3ed351ced32`
- Validation data: held-out DROID episodes 1064-1079; test episodes 1080-1095
  remain sealed
- Context/rollout: one observed latent frame, six total latent frames (21 RGB
  frames)
- Runtime: V100, FP16 autocast, PyTorch SDPA
- Sampling: 20 outer steps, CFG 2.0, seed `20260827 + sample_index`
- Metric: RGB MAE over future frames 1-20 in the 0-255 space

The official file contains one bare set of weights rather than separate model
and EMA entries. It strictly matches the local 0.5B architecture after adding
bare-state-dict checkpoint compatibility.

## Results

| Condition | Mean MAE | Median MAE |
| --- | ---: | ---: |
| real | **5.468** | **5.393** |
| zero-valued | 9.368 | 8.442 |
| reversed | 9.241 | 8.068 |

Paired validation counts:

- real better than zero: 15/16
- real better than reverse: 15/16
- real best against both: 14/16

Mean paired deltas are `zero - real = +3.900` and `reverse - real = +3.773`.
Mean output MAE is 6.982 for real versus zero and 6.936 for real versus
reverse. Correct actions therefore affect the output and provide a strong,
mostly consistent accuracy advantage.

## Persistence baseline and interpretation

Repeating the observed first frame gives mean MAE 4.468 and median MAE 3.765;
real-action generation beats it on 4/16 episodes. Pixel MAE favors static
predictions over plausible motion, especially in low-motion clips, so this is
not evidence that persistence is a better world model. It is nevertheless a
real limitation: short-horizon pixel prediction has not yet beaten the strong
static baseline in aggregate. Continued-training experiments must preserve the
clear action-control advantage while measuring perceptual and motion-aware
quality in addition to RGB MAE.

The contrast with the 0.12B run is decisive. The same validation split,
conditioning construction, temporal alignment, seeds, and sampling settings
now make real actions win consistently. The earlier failure was therefore not
evidence of a broken action pipeline; it was specific to the 64-episode,
5k-step 0.12B checkpoint.

## Runtime and artifacts

Mean generation time per 21-frame sample was 6.34 seconds (real), 6.31 seconds
(zero), and 6.36 seconds (reverse), with one V100 per condition.

```text
/data/miniworld/experiments/official-055b-validation-action-ablation/
  metrics_summary.json
  metrics_per_episode.csv
  comparison_best-real_episode_1075.png
  comparison_worst-real_episode_1069.png
  real/pred/*.mp4
  zero/pred/*.mp4
  shuffle/pred/*.mp4
```
