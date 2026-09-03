# Chunk-Aligned Adaptive Rollout Stage A′ Design

Date: 2026-08-27

## 1. Purpose and Correction

The first adaptive-rollout offline evaluation established a quality result:
latent uncertainty can identify a retained prefix whose RGB error is lower than
an analytically matched fixed-horizon baseline. It did not establish executable
compute savings.

The frozen official 0.55B configuration uses `history_len=1`,
`df_chunk_size=2`, and five future latent frames. Streaming inference therefore
finishes future frames at boundaries 1, 3, and 5:

```text
chunk 0: history frame 0 + future frame 1
chunk 1: future frames 2 and 3
chunk 2: future frames 4 and 5
```

The previous evaluator counted a trigger at future step 4 as generated horizon
4. In the executable model, future step 5 finishes in the same chunk, so the
actual completed generated horizon is 5. Every trigger selected by the frozen
EMA/hysteresis policy occurs at step 4 or 5. Its chunk-completed generated
coverage is therefore 1.0, not 0.9375.

This correction narrows the prior conclusion: the policy passed a retained
quality gate, but online compute-saving authorization is withdrawn until this
Stage A′ passes. The test split remains sealed.

## 2. Scope

Stage A′ reuses the frozen 80-row validation correlation table. It does not
train, sample, decode, modify the checkpoint, or inspect test episodes
1080-1095. It changes only policy simulation, cost accounting, threshold
selection, and the experiment record.

Online model integration remains out of scope. It receives authorization only
if a chunk-executable policy passes all quality conditions and has completed
generated coverage strictly below 1.0.

## 3. Executable Completion Model

For `history_len=1`, `chunk_size=2`, and `H_max=5`, the ordered future
completion boundaries are:

```text
(1, 3, 5)
```

The policy still updates EMA state sequentially for every newly completed
future uncertainty value. It may observe step 2 and then step 3 when chunk 1
finishes, but it may emit only one external decision after the whole chunk is
complete.

If any step inside a newly completed chunk satisfies the two-consecutive
exceedance rule:

- `requested_observation_at` records the first triggering future step;
- `generated_horizon` is the end of that completed chunk;
- `retained_horizon` is the previous completed future boundary, with a minimum
  of 1;
- later chunks are not counted as completed generation.

The resulting mapping is:

| First trigger | Generated horizon | Retained horizon |
| ---: | ---: | ---: |
| step 1 | 1 | 1 |
| step 2 or 3 | 3 | 1 |
| step 4 or 5 | 5 | 3 |
| no trigger | 5 | 5 |

This table records completed chunks, not exact DiT FLOPs. With
`stream_inflight_chunks=8`, later chunks may already have received partial
denoising work before the leading chunk completes. Therefore completed
generated coverage below 1.0 is necessary but not sufficient evidence of
wall-time savings. Only the later online gate may make a measured efficiency
claim.

## 4. Strategies and Frozen Inputs

The primary strategy remains latent population variance across `K=4` members
with EMA alpha `0.5` and two consecutive exceedances. Single threshold remains
a comparison. RGB disagreement remains diagnostic and cannot pass the formal
gate.

Candidate thresholds, target retained coverage `0.80`, tie breaking, and LOEO
fold construction remain unchanged. The threshold is reselected because the
executable retained/generated horizons change the objective and feasibility of
each candidate. This is a documented correction before test access, not a
post-test adjustment.

Constant uncertainty, non-finite values, missing steps, duplicate rows,
unsupported completion layouts, and sealed test episodes are rejected.

## 5. Quality and Cost Metrics

For each LOEO-held-out policy report:

- retained and completed-generated horizon per episode;
- retained and completed-generated coverage;
- retained/discarded RGB error numerators and counts;
- retained RGB MAE;
- high-error retained fraction;
- median, p90, and worst episode retained error;
- request rate and first triggering step;
- completed chunks and avoided complete chunks;
- threshold, raw uncertainty, EMA state, and decision trace.

The matched-quality baseline continues to use analytic interpolation between
fixed horizons 1-5. A fixed rollout knows its horizon before generation and may
request a final partial chunk, while an adaptive rollout can react only after a
chunk finishes. The report must show this asymmetry explicitly rather than
rounding the fixed baseline to adaptive boundaries.

Two cost views are reported:

1. completed future latent-member steps: adaptive `K=4` versus fixed `K=4`;
2. the same proxy versus ordinary fixed `K=1`.

Neither proxy is called wall-time speedup.

## 6. Stage A′ Gate

A policy passes only when its 16 LOEO-held-out results satisfy all five:

1. mean retained coverage is at least `0.80`;
2. retained RGB MAE is strictly below the matched fixed baseline;
3. adaptive retained error improves on at least 9 of 16 episodes;
4. adaptive p90 episode error is no worse than matched fixed by more than
   `0.10` RGB MAE;
5. completed generated coverage is strictly below `1.0`.

At least one latent policy must satisfy all five. RGB diagnostic results cannot
authorize online work.

If more than one latent policy passes, select the one with lowest retained RGB
MAE, then lower completed generated coverage, then higher retained coverage,
then higher threshold.

After LOEO, choose one deployment threshold from all 16 validation episodes
using the same target and tie-breaking rules. Freeze estimator, alpha,
hysteresis, chunk layout, and threshold before any online run.

## 7. Outputs and Identity

Write a new output directory; do not overwrite the earlier result:

```text
/data/miniworld/experiments/official-055b-adaptive-rollout-chunk-aligned
```

Required artifacts:

```text
policy_decisions.csv
policy_curve.csv
loeo_folds.csv
chunk_costs.csv
adaptive_rollout_summary.json
counterexamples.csv
report.md
```

Record source CSV/summary/archive paths, source archive SHA256, code commit,
checkpoint/data identities, `history_len=1`, `df_chunk_size=2`, completion
boundaries `(1,3,5)`, policy constants, candidate thresholds, fold identities,
tie-breaking rule, all five gate booleans, and the correction relationship to
the previous report.

The earlier result directory remains immutable. Its project report and journal
entry receive an addendum stating that 0.9375 was an idealized per-frame proxy,
the executable completed-chunk coverage is 1.0, and online authorization is
superseded by Stage A′.

## 8. Verification and Decision

Unit tests use hand-calculated layouts and must prove:

- completion boundaries for odd history and partial final chunks;
- step-4 trigger maps to generated 5 and retained 3;
- the original frozen EMA policy maps to generated coverage 1.0;
- LOEO candidate isolation remains intact;
- a quality pass with generated coverage 1.0 fails the fifth gate;
- malformed layouts and sealed episodes are rejected.

An independent reconstruction reads only the new CSV files and frozen source
CSV, recomputes every numerator/count, matched baseline, coverage, episode win,
p90, and gate, and agrees with JSON within the existing tolerances.

If Stage A′ passes, write a separate online implementation plan. The online gate
uses episode 1064 as a no-trigger prefix-equivalence control and the lowest
validation episode ID whose frozen deployment trace avoids at least one complete
chunk as the trigger/cost case. Choosing the trigger case by episode ID and
trace, not prediction error, prevents favorable-error cherry-picking.

If Stage A′ fails, do not modify online inference. Record the negative result
and separately decide whether changing `df_chunk_size` deserves a new baseline
subproject. A `df_chunk_size=1` experiment is not an automatic fallback because
it changes official inference behavior and scheduling.
