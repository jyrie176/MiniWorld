# MiniWorld Uncertainty–Error Correlation Design

Date: 2026-08-27

## 1. Objective and Scope

Phase 5 starts by testing the prerequisite for uncertainty-aware adaptive
rollout: whether disagreement among stochastic predictions is associated with
future prediction error. This first subproject is an offline, fixed-horizon
validation experiment. It does not change model weights, stop a rollout early,
select a final test result, or access test episodes 1080–1095.

The experiment uses the frozen official MiniWorld 0.55B DROID checkpoint and
validation episodes 1064–1079. Each episode is sampled `K=4` times with the
same observation and real actions but different deterministic seeds.

Success means producing a reproducible dataset and complete correlation
analysis. A positive adaptive-rollout claim is not required: weak or unstable
correlation is a valid result and must be reported as such.

## 2. Alternatives Considered

### Recommended: latent disagreement plus RGB control metric

Compute uncertainty from both generated latent variance and decoded RGB
disagreement, then correlate both against decoded RGB error. Latent
disagreement is the primary estimator because it measures model-output
variation before VAE decode; RGB disagreement is an interpretable control.
This requires retaining generated latent tensors but no new learned model.

### RGB-only disagreement

This could reuse MP4 outputs with almost no sampling changes, but video
encoding and VAE decoding become part of the uncertainty signal. It is kept as
a control, not the sole estimator.

### Learned or perceptual uncertainty

A learned confidence head or external perceptual encoder could be more
task-aligned, but would introduce training data, calibration leakage, extra
dependencies, and attribution questions before the basic hypothesis is known
to hold. It is deferred until simple sampling disagreement is evaluated.

## 3. Architecture

The implementation has three isolated responsibilities.

1. `miniworld/uncertainty.py` contains pure tensor/statistical primitives. It
   accepts aligned ensembles and returns per-future-step latent variance, RGB
   pairwise disagreement, RGB prediction error, Pearson correlation, Spearman
   correlation, and quantile-bin summaries. It performs no model loading or
   filesystem traversal.
2. The existing sampling path gains an opt-in latent export. Normal sampling
   behavior remains unchanged. When enabled, it saves the final latent tensor
   associated with each generated video, using a deterministic sample name.
3. `scripts/evaluate_droid_uncertainty.py` validates and joins `K` sampling
   directories, loads validation ground truth, calls the pure metric module,
   and writes auditable tabular and summary artifacts.

This separation allows statistical correctness to be tested on small CPU
tensors without loading the 0.55B model or VAE.

## 4. Sampling and Reproducibility

The frozen sampling configuration must match the official 0.55B real-action
baseline except for repeated seeds. The four seed bases are fixed in the run
manifest before evaluation. Each seed directory contains:

```text
seed_<seed>/
  pred/sample_0000.mp4
  latents/sample_0000.pt
  ...
  sampling_manifest.json
```

The manifest records checkpoint path and SHA256, data manifest/hash, episode
IDs, action variant, precision, attention backend, all rollout/sampler
arguments, seed, software commit, and latent tensor shape/dtype. The evaluator
rejects missing samples, duplicate seeds, inconsistent episode lists,
inconsistent shapes, non-finite tensors, or manifests that disagree on any
setting other than seed.

Latent files contain only detached CPU tensors needed for the experiment; they
must never contain model, optimizer, or input-dataset state.

## 5. Time Alignment and Metric Definitions

The current DROID baseline uses six latent frames: one observed context latent
and five future latent steps. It decodes to 21 RGB frames: frame 0 is observed
context and frames 1–20 are future targets.

For future latent index `j` in `1..5`, its corresponding RGB target block is:

```text
j=1 -> RGB frames 1..4
j=2 -> RGB frames 5..8
...
j=5 -> RGB frames 17..20
```

Context latent/frame zero is excluded from every correlation.

For an aligned generated latent ensemble
`Z[K,C,T,H,W]`, primary uncertainty at future step `j` is the population
variance across `K`, averaged across channel and spatial dimensions:

```text
u_latent[j] = mean_C,H,W(var_K(Z[:,:,j,:,:], correction=0))
```

For decoded RGB ensemble `X[K,F,H,W,C]`, the control uncertainty is mean
pairwise absolute disagreement over all unordered sample pairs and all pixels
in the four-frame block:

```text
u_rgb[j] = mean_{a<b,pixels} |X[a,block(j)] - X[b,block(j)]|
```

The target error is the mean member-wise RGB MAE to ground truth, rather than
the error of an averaged video, so ensemble averaging cannot create an
artificially smooth prediction:

```text
error_rgb[j] = mean_{k,pixels} |X[k,block(j)] - GT[block(j)]|
```

RGB values are evaluated on the existing 0–255 scale. Latent uncertainty is
reported in raw units and also normalized within the complete validation run
for threshold visualization; correlations use raw values because Pearson is
scale invariant and Spearman is rank based.

## 6. Statistical Analysis

The primary analysis pools the 16 episodes × 5 future latent steps into 80
observations. For both latent and RGB uncertainty it reports:

- Pearson correlation with `error_rgb`;
- Spearman rank correlation with `error_rgb`, using average ranks for ties;
- correlation separately at each future step across 16 episodes;
- a horizon-conditioned Spearman value obtained by average-ranking
  uncertainty and error within each future step before pooling those ranks;
- correlation separately within each episode across five future steps,
  clearly labeled exploratory because each group is small;
- equal-count uncertainty quantiles with count, mean uncertainty, mean error,
  and error range;
- the error curve and uncertainty curve by future step.

No p-value or confidence interval is used as a binary proof of success in this
small initial validation set. The decision considers sign consistency,
magnitude, monotonic quantile behavior, horizon stratification, and failure
cases. Constant inputs produce a recorded undefined correlation rather than a
fabricated zero.

## 7. Outputs

The evaluator writes:

```text
metrics_per_episode_step.csv
correlation_summary.json
uncertainty_bins.csv
horizon_summary.csv
sampling_manifest_combined.json
report.md
```

Each row of `metrics_per_episode_step.csv` includes episode ID, future latent
step, RGB frame range, both uncertainty values, target error, per-seed errors,
and the four seeds. JSON stores definitions, counts, correlations, checkpoint
identity, data identity, and warnings. `report.md` is generated from the same
tables so every displayed number remains machine-traceable.

W&B logging is optional mirroring, not the source of truth. Local CSV/JSON and
the immutable manifests remain sufficient to reconstruct all claims.

## 8. Error Handling and Integrity Rules

- Test episode IDs 1080–1095 are rejected explicitly, not merely omitted by
  convention.
- The evaluator requires exactly the declared `K`; partial ensembles fail.
- All prediction/ground-truth lengths and spatial shapes must align before any
  metric is computed.
- MP4 decoding is converted to uint8 and checked for the expected 21 frames.
- Latents are loaded on CPU and checked for finite values and the expected six
  temporal positions.
- Existing outputs are not silently overwritten unless an explicit overwrite
  option is given.
- Any excluded episode or failed seed makes the complete formal run fail; a
  partial exploratory run must carry a visible `incomplete=true` marker.

## 9. Testing and Verification

Development follows test-driven implementation. CPU tests cover:

- latent population variance on a hand-computed ensemble;
- RGB pairwise disagreement and member-wise target MAE;
- six-latent/21-RGB temporal alignment and exclusion of context;
- Pearson and tied-rank Spearman against known examples;
- constant-input undefined correlation handling;
- quantile bins retaining every observation exactly once;
- parser defaults (`K=4`) and manifest compatibility validation;
- rejection of test episode IDs, missing seeds, shape mismatch, and NaN/Inf;
- opt-in latent export naming without changing default sampling outputs.

The full existing non-CUDA/non-FlashAttention suite must remain green. A small
synthetic end-to-end evaluator fixture verifies deterministic CSV/JSON output.
The first GPU gate samples one validation episode with two seeds to verify
model/VAE integration and latent-video alignment. Only after that gate passes
does the formal 16-episode, four-seed validation run begin.

## 10. Phase Gate and Next Decision

After the formal run, results are recorded before designing a threshold policy.
The pre-registered gate is evaluated first for primary latent disagreement and
then reported separately for the RGB control estimator. An estimator passes
the initial signal gate only when all three conditions hold:

1. pooled Spearman correlation is at least `0.30`;
2. horizon-conditioned Spearman correlation is at least `0.20`, so a result
   caused only by both quantities increasing with horizon does not pass;
3. mean RGB error in the highest uncertainty quartile is greater than in the
   lowest uncertainty quartile.

Passing this gate authorizes design of the fixed-vs-threshold-vs-smoothed
adaptive rollout experiment; it does not by itself establish adaptive rollout
effectiveness.

If the signal is weak, contradictory, or explained only by horizon, the next
step is a bounded diagnostic comparison of estimator definitions and
horizon-conditioned normalization—not threshold tuning. If reasonable simple
estimators remain uninformative, the project reports the negative result and
does not claim adaptive rollout effectiveness.
