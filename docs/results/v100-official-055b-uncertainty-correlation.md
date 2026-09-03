# Official MiniWorld-0.55B uncertainty-error correlation

Result date: 2026-08-27

## Mainline decision

This is the first formal Phase 5 experiment. It tests the prerequisite for
uncertainty-aware adaptive rollout: stochastic prediction disagreement should
be positively associated with future prediction error. It uses the frozen
official 0.55B checkpoint and validation episodes 1064-1079. Test episodes
1080-1095 remain sealed.

Both pre-registered estimators pass the initial signal gate. This authorizes a
separate adaptive-rollout design; it does not establish that adaptive stopping
improves a quality-coverage or quality-cost frontier.

## Setup and integrity

- Checkpoint: official `MiniWorld_0_5b_droid.pt`
- Checkpoint SHA256: `e4b118befe88cee7338400c5510fdd497212b9b1988034290030b3ed351ced32`
- Validation manifest SHA256: `ac3e1dbff5a22732b54c47834751714f31302bf0aca859b4dc78a2809203ae70`
- Code identity in sampling manifests: `f5ccd34a4108523d1bbd72e325cf79ebf307742e`
- Seeds: 20260827, 20260828, 20260829, 20260830
- Runtime: V100, FP16 autocast, PyTorch SDPA
- Conditioning: real actions
- Rollout: one context plus five future latent steps, decoding to 21 RGB frames
- Sampling: 20 effective steps, CFG 2.0
- Observations: 16 episodes x 5 future steps = 80

Each of four seed directories contains 16 MP4 files and 16 latent tensors.
All 64 tensors are finite with shape `(1,48,6,15,20)`, and all manifests agree
on data, checkpoint, configuration, episode order, and Git identity. The final
latent buffer is float32 even though DiT computation uses FP16 autocast; this is
recorded rather than silently cast. The evaluator reports `incomplete=false`.

## Metric definitions

Primary latent uncertainty is population variance across the four generated
latents, averaged over channel and spatial dimensions for each future latent
step. RGB uncertainty is mean pairwise absolute disagreement across all six
sample pairs in the corresponding four-frame block. Target error is mean
member-wise RGB MAE to ground truth on the 0-255 scale; predictions are not
averaged before scoring.

Context frame zero is excluded. Future latent steps 1-5 align respectively to
RGB blocks 1-4, 5-8, 9-12, 13-16, and 17-20.

## Correlation result

| Estimator | Pearson | Spearman | Horizon-conditioned Spearman | Gate |
| --- | ---: | ---: | ---: | --- |
| latent variance | 0.7692 | 0.7171 | 0.5818 | **pass** |
| RGB disagreement | 0.9335 | 0.9335 | 0.9041 | **pass** |

The pre-registered gate requires pooled Spearman at least 0.30,
horizon-conditioned Spearman at least 0.20, and higher mean error in the top
uncertainty quartile than the bottom quartile. Both estimators satisfy all
three conditions. Controlling horizon is important: the result is not solely
explained by both uncertainty and error increasing later in the rollout.

Per-horizon latent Spearman values for steps 1-5 are 0.4500, 0.4912, 0.5794,
0.7441, and 0.6441. RGB values are 0.8941, 0.9588, 0.9618, 0.8912, and 0.8147.
The association is positive at every fixed horizon.

## Uncertainty bins and horizon curve

| Estimator | Uncertainty quartile | Mean uncertainty | Mean RGB error |
| --- | ---: | ---: | ---: |
| latent | 1 | 0.01252 | 3.8699 |
| latent | 2 | 0.01672 | 4.3641 |
| latent | 3 | 0.02201 | 5.5107 |
| latent | 4 | 0.03322 | 8.1770 |
| RGB | 1 | 1.6431 | 3.1822 |
| RGB | 2 | 2.3778 | 4.4973 |
| RGB | 3 | 3.4856 | 5.9369 |
| RGB | 4 | 5.4454 | 8.3052 |

Both estimators show monotonic error growth across four equal-count bins.

| Future latent step | Mean latent uncertainty | Mean RGB disagreement | Mean RGB error |
| ---: | ---: | ---: | ---: |
| 1 | 0.01533 | 1.6780 | 3.3325 |
| 2 | 0.01498 | 2.5103 | 4.2696 |
| 3 | 0.02249 | 3.2748 | 5.0201 |
| 4 | 0.02444 | 4.0417 | 6.9261 |
| 5 | 0.02834 | 4.6852 | 7.8537 |

## Counterexamples and limitations

The relationship is not perfect. Latent variance substantially under-ranks
error for episode 1078 steps 4-5 and episode 1070 step 4. Episode 1078 has
per-episode latent Spearman -0.30, while the other 15 episodes are positive.
Latent variance also over-ranks uncertainty for episode 1065 step 3 and episode
1072 step 1 despite relatively low RGB error. RGB disagreement has smaller but
real rank errors, including under-ranking episode 1072 step 5.

RGB disagreement and member-wise RGB error share the decoded output domain and
both respond strongly to scene motion, so its very high correlation should not
be interpreted as calibrated probability or causality. Latent variance is the
cleaner primary signal and still passes after horizon conditioning. The
validation set has only 16 episodes; thresholds must remain validation-only,
and no generalization claim is made before the final sealed test.

## Conclusion and next step

Sampling disagreement contains actionable error-ranking information under the
frozen official 0.55B baseline. Phase 5 may proceed to a new design comparing
fixed horizon, single-threshold adaptive horizon, and smoothed/hysteretic
adaptive horizon under explicit quality-coverage and quality-cost curves.

The next experiment must not merely show lower error after predicting fewer
frames. It must report average horizon, coverage, retained error, early-stop
rate, and the cost of `K=4` sampling under a fair comparison. Thresholds are
selected on validation only; test remains sealed until the complete method is
frozen.

## Artifacts

```text
/data/miniworld/experiments/official-055b-uncertainty-correlation-k4
  seed_20260827/{pred,latents,sampling_manifest.json,sample.log}
  seed_20260828/{pred,latents,sampling_manifest.json,sample.log}
  seed_20260829/{pred,latents,sampling_manifest.json,sample.log}
  seed_20260830/{pred,latents,sampling_manifest.json,sample.log}
  metrics/{metrics_per_episode_step.csv,correlation_summary.json,
           uncertainty_bins.csv,horizon_summary.csv,
           sampling_manifest_combined.json,report.md}
```
