# Uncertainty-Aware Adaptive Rollout Design

Date: 2026-08-27

## 1. Objective and Scope

Phase 5 has established that stochastic disagreement predicts future RGB
error for the frozen official MiniWorld 0.55B checkpoint. This subproject tests
whether that signal can select a useful rollout horizon and, only if the
offline quality-coverage gate passes, implements real online early stopping.

The core method is inference-only. It does not train or modify MiniWorld. The
validation split remains episodes 1064-1079; test episodes 1080-1095 remain
sealed until the complete method and thresholds are frozen.

The method exposes three decisions:

```text
CONTINUE              keep generating the current rollout
REQUEST_OBSERVATION   end this rollout and ask an outer system for new state
TERMINATE             stop because the configured horizon or task ends
```

MiniWorld currently has no interactive robot, simulator, or planner. In the
formal offline evaluation, `REQUEST_OBSERVATION` therefore marks the boundary
after which predictions are not trusted. It must not be described as an
executed physical observation or replanning action.

## 2. Chosen Two-Stage Approach

### Stage A: offline policy selection

Reuse the completed four-seed validation ensemble and simulate stopping rules
over its five future latent steps. This stage selects estimator, threshold,
and smoothing behavior without new model sampling. It must establish a fair
quality-coverage advantage before online inference is modified.

### Stage B: online stopping

After Stage A freezes a policy, integrate chunk-level uncertainty decisions
into streaming inference. Run a one-episode equivalence and resource gate
before expanding online measurement. Online work is cancelled if Stage A does
not pass.

This is preferred over offline-only evaluation, which cannot demonstrate real
compute savings, and over immediate online implementation, which would mix
threshold selection with a costly inference refactor.

## 3. Policy Inputs and Decisions

The primary signal and every formal gate use latent population variance across
`K=4` stochastic members. RGB disagreement remains a diagnostic/oracle control
because it has stronger observed correlation but requires decode and shares
its output domain with RGB error; an RGB-only pass cannot authorize online
implementation when the latent policy fails.

For future latent step `t`, the policy receives:

```text
episode identifier
t in {1,2,3,4,5}
raw latent uncertainty u_t
previous policy state
H_min=1 and H_max=5
```

It returns a decision plus diagnostics. A trigger at step `t` means the system
had to generate step `t` to measure its uncertainty, but step `t` is not
considered trusted coverage. The retained horizon is `max(H_min, t-1)`. If no
trigger occurs, all five steps are retained and the terminal decision is
`TERMINATE` at `H_max`.

This distinction is recorded separately:

- `generated_horizon`: work performed before the decision;
- `retained_horizon`: predictions accepted as usable;
- `requested_observation_at`: first triggering step, or null.

## 4. Strategies

### Fixed horizon

Evaluate fixed retained horizons 1, 2, 3, 4, and 5. This is both the standard
baseline and the source for matched-coverage interpolation.

### Single threshold

At each step, emit `REQUEST_OBSERVATION` when:

```text
u_t > tau
```

Otherwise emit `CONTINUE` until `H_max`.

### Smoothed/hysteretic threshold

Use a fixed exponential smoother:

```text
s_1 = u_1
s_t = 0.5 * u_t + 0.5 * s_(t-1)
```

Emit `REQUEST_OBSERVATION` only after `s_t > tau` on two consecutive future
steps. The constants `alpha=0.5` and `consecutive=2` are fixed before the
formal sweep and are not tuned per episode.

## 5. Threshold Selection Without Direct Validation Reuse

Threshold evaluation uses leave-one-episode-out (LOEO):

1. hold out one of the 16 validation episodes;
2. build candidate thresholds from the other 15 episodes;
3. choose the threshold on those 15 episodes;
4. apply it unchanged to the held-out episode;
5. repeat for all 16 episodes and aggregate held-out decisions.

Candidate thresholds are the sorted unique primary uncertainties plus boundary
values that retain none or all. If there are more than 101 unique values, use
the 101 empirical quantiles at probabilities `0.00, 0.01, ..., 1.00`, deduplicate
them, and add the two boundary values. No threshold is created from a held-out
episode.

The primary operating point requires mean retained coverage at least `0.80`.
Among candidates satisfying it, choose the lowest retained RGB MAE. Ties are
resolved by higher coverage, then the higher threshold. The complete
20%-100% quality-coverage curve is reported; 80% is the frozen primary point,
not the only displayed point.

After LOEO evaluation, select one deployment threshold from all 16 validation
episodes with the same rule. That value is frozen before any final test run.

## 6. Metrics and Fair Baselines

For every policy configuration report:

- mean and distribution of retained horizon;
- mean generated horizon;
- retained coverage: total retained future steps divided by `16*5`;
- retained member-wise RGB MAE;
- discarded-region RGB MAE;
- early/request-observation rate;
- fraction of high-error steps retained, where high error means above the 75th
  percentile RGB error computed from that LOEO training fold;
- median, 90th percentile, and worst episode retained error;
- per-episode horizon and failure examples;
- threshold and policy state trace.

### Matched-coverage fixed baseline

An adaptive mean horizon can be fractional. For mean horizon `h=m+p`, where
`m=floor(h)` and `0<=p<1`, compute the analytic expected fixed baseline as a
mixture of fixed `m` and fixed `m+1` with weights `1-p` and `p`. Aggregate
retained error by expected error numerator divided by expected retained-step
count. This matches coverage without assigning longer fixed rollouts to
specific episodes after seeing their errors. At the endpoints `h=1` and `h=5`,
use the corresponding pure fixed baseline without interpolation.

The main offline gate requires, on LOEO held-out predictions:

1. mean retained coverage at least `0.80`;
2. retained RGB MAE strictly below the matched-coverage fixed baseline;
3. positive improvement on at least 9 of 16 held-out episodes;
4. 90th-percentile retained episode error no worse than the matched fixed
   baseline by more than `0.10` RGB MAE;
5. at least one of single-threshold or smoothed/hysteretic policies satisfies
   all four conditions.

No relative-improvement percentage is pre-required because the validation set
is small; the exact absolute and relative differences are reported.

## 7. Cost Accounting

Two comparisons are mandatory.

### Method-internal

Compare fixed `K=4` with adaptive `K=4`. Report generated latent-member steps,
DiT/chunk work, decoded frames, peak memory, and wall time. This answers whether
stopping saves work once ensemble uncertainty is already required.

### Deployment baseline

Compare adaptive `K=4` with ordinary fixed `K=1`. Report the total cost ratio.
Adaptive rollout cannot be called more efficient than standard inference if
four-member sampling costs more despite stopping early.

Generated horizon, not retained horizon, drives compute accounting because the
triggering step must be generated before its uncertainty is known.

## 8. Offline Components and Outputs

Add a pure policy module with no model or filesystem dependencies. It consumes
ordered uncertainty traces and returns typed decision traces. Add an offline
evaluator that reads the existing `metrics_per_episode_step.csv`, performs
LOEO selection, computes fixed/adaptive curves, and writes:

```text
policy_decisions.csv
policy_curve.csv
loeo_folds.csv
adaptive_rollout_summary.json
counterexamples.csv
report.md
```

Every displayed value is reconstructible from CSV/JSON. Output records the
source correlation archive SHA, checkpoint/data identities, code commit,
policy constants, threshold candidates, tie-breaking rule, and gate result.

## 9. Online Architecture and Fallbacks

The preferred online implementation represents the four stochastic members
as a batch sharing observation and action conditioning but using distinct
noise. When a generated latent chunk completes:

1. collect the four member latents for the newly completed future step(s);
2. compute latent population variance;
3. update policy state;
4. emit `REQUEST_OBSERVATION` or `CONTINUE`;
5. on request, stop subsequent diffusion/chunk work and decode only retained
   output.

The one-episode online gate measures whether batch `K=4` fits a 32 GB V100. If
it OOMs, the next approved fallback is two GPUs with `K=2` per GPU and
synchronized chunk decisions. Four complete sequential rollouts followed by a
decision are not an online fallback because they cannot save generation work.

Online integration must preserve streaming cache/RoPE correctness and expose a
callback or policy interface rather than embedding threshold logic directly
inside the denoising schedule.

## 10. Online Gate

Use validation episode 1064 and the frozen deployment threshold. Compare a
full fixed rollout and online adaptive rollout under identical four noise
tensors. The gate requires:

- identical retained latent prefix before the decision within recorded FP16
  tolerance;
- the same request-observation step as offline replay;
- no NaN/Inf, cache, RoPE, or streaming decode error;
- fewer generated post-trigger steps than fixed `K=4`;
- peak memory and synchronized wall-time measurements;
- explicit cost comparison with fixed `K=1` and fixed `K=4`.

Only after this passes may online cost measurement expand across validation.

## 11. Re-observation Semantics and Oracle Extension

`REQUEST_OBSERVATION` is a real systems interface, not a claim that the current
repository can acquire a robot observation. A future controller may respond by
capturing new state, replanning, or terminating. Without such an environment,
the formal adaptive-horizon metric ends the current rollout at that boundary.

After the adaptive-horizon gate passes, an optional Phase 6 experiment may use
the recorded DROID future frame at a request boundary as a new context and
restart prediction. It must be labeled:

```text
oracle re-observation / teacher-forced re-anchoring
```

Compare no re-observation, fixed-interval oracle re-observation,
uncertainty-triggered oracle re-observation, and error-oracle triggering. This
measures an upper bound on recoverability, not deployable robot performance.
It does not block the core adaptive-horizon result.

## 12. Error Handling and Test Protection

- Reject test episode IDs 1080-1095 in all threshold-selection paths.
- Reject missing/duplicate episode-step rows, horizons outside 1-5, non-finite
  metrics, source identity mismatch, and threshold folds that inspect held-out
  uncertainty.
- Constant uncertainty produces an explicit degenerate policy result.
- Never overwrite a formal output directory without an explicit flag.
- Unit tests use hand-calculated policy traces, coverage/error numerators,
  analytic fixed mixtures, LOEO isolation, tie breaking, and gate boundaries.
- Integration tests verify deterministic output reconstruction from a
  synthetic CSV.
- Existing 108-test CPU suite must remain green before any GPU run.

## 13. Decision After Stage A

If neither adaptive policy passes the offline gate, stop before online model
changes and report the negative quality-coverage result. If a policy passes,
freeze estimator, constants, and deployment threshold, then implement Stage B.

Passing Stage B authorizes the final method-freeze review. It does not
automatically authorize opening the test split: all ablations required for the
complete method must be frozen first.
