# Adaptive Rollout Offline Policy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine with leave-one-episode-out validation whether latent-uncertainty threshold policies improve retained RGB error over an analytically matched fixed-horizon baseline at at least 80% coverage.

**Architecture:** Add pure decision and accounting primitives in `miniworld/rollout_policy.py`, then add an offline evaluator that consumes the already-audited 80-row uncertainty CSV. The evaluator selects thresholds strictly inside each LOEO training fold, writes reconstructible policy curves and decisions, and stops the project before online inference changes if the pre-registered gate fails.

**Tech Stack:** Python 3.11, dataclasses/enums, NumPy, CSV/JSON, pytest, Docker. No model, VAE, CUDA, or new dependency is required for offline selection.

**Spec:** `docs/superpowers/specs/2026-08-27-adaptive-rollout-design.md`

## Global Constraints

- Use only validation episodes 1064-1079 and reject sealed test episodes 1080-1095.
- Consume the frozen official 0.55B `K=4` correlation artifacts; do not resample or train.
- Formal policy gates use latent population variance only; RGB disagreement is diagnostic and cannot rescue a latent failure.
- Use `H_min=1`, `H_max=5`, primary retained coverage at least `0.80`, EMA alpha `0.5`, and hysteresis count `2`.
- A trigger at generated step `t` retains through `max(1,t-1)` and records generated horizon `t`.
- Select thresholds with LOEO; no candidate or statistic used for selection may inspect the held-out episode.
- Compare against analytic matched-coverage fixed mixtures and report both retained and generated horizons.
- Stop before online inference implementation if neither adaptive policy passes all offline gate conditions.
- Preserve local CSV/JSON as the source of truth and keep the full existing CPU suite green.

---

### Task 1: Pure policy decision traces

**Files:**
- Create: `miniworld/rollout_policy.py`
- Create: `tests/test_rollout_policy.py`

**Interfaces:**
- Produces: `Decision(str, Enum)` with `CONTINUE`, `REQUEST_OBSERVATION`, `TERMINATE`.
- Produces: immutable `PolicyTrace` with `decisions`, `retained_horizon`, `generated_horizon`, and `requested_observation_at`.
- Produces: `fixed_policy(horizon: int, *, h_min: int = 1, h_max: int = 5) -> PolicyTrace`.
- Produces: `threshold_policy(uncertainty: Sequence[float], tau: float, *, h_min: int = 1, h_max: int = 5) -> PolicyTrace`.
- Produces: `smoothed_hysteretic_policy(uncertainty: Sequence[float], tau: float, *, alpha: float = 0.5, consecutive: int = 2, h_min: int = 1, h_max: int = 5) -> PolicyTrace`.

- [ ] **Step 1: Write failing decision-semantics tests**

```python
def test_threshold_trigger_discards_triggering_step_but_counts_generated_work():
    trace = threshold_policy([0.1, 0.7, 0.2, 0.1, 0.1], tau=0.5)
    assert trace.decisions == (Decision.CONTINUE, Decision.REQUEST_OBSERVATION)
    assert trace.retained_horizon == 1
    assert trace.generated_horizon == 2
    assert trace.requested_observation_at == 2

def test_threshold_without_trigger_terminates_at_maximum():
    trace = threshold_policy([0.1] * 5, tau=0.5)
    assert trace.decisions[-1] is Decision.TERMINATE
    assert trace.retained_horizon == trace.generated_horizon == 5
    assert trace.requested_observation_at is None
```

- [ ] **Step 2: Run Task 1 tests and verify import failure**

Run: `pytest -q tests/test_rollout_policy.py`

Expected: collection fails because `miniworld.rollout_policy` does not exist.

- [ ] **Step 3: Implement immutable trace types and fixed/single-threshold policies**

```python
class Decision(str, Enum):
    CONTINUE = "continue"
    REQUEST_OBSERVATION = "request_observation"
    TERMINATE = "terminate"

@dataclass(frozen=True)
class PolicyTrace:
    decisions: tuple[Decision, ...]
    retained_horizon: int
    generated_horizon: int
    requested_observation_at: int | None
```

Validate exactly five finite uncertainty values, finite threshold, and `1 <= h_min <= h_max <= len(uncertainty)`. Generate decisions in order and apply the trigger-step retention rule exactly.

- [ ] **Step 4: Write failing smoothing and validation tests**

```python
def test_smoothed_policy_requires_two_consecutive_exceedances():
    trace = smoothed_hysteretic_policy([0.1, 0.9, 0.1, 0.9, 0.9], tau=0.45)
    assert trace.requested_observation_at == 5
    assert trace.retained_horizon == 4
    assert trace.generated_horizon == 5

@pytest.mark.parametrize("values", [[0.1] * 4, [0.1, float("nan"), 0.1, 0.1, 0.1]])
def test_policy_rejects_wrong_length_or_nonfinite_uncertainty(values):
    with pytest.raises(ValueError):
        threshold_policy(values, tau=0.5)
```

- [ ] **Step 5: Implement smoothing, run Task 1 tests, and commit**

Run: `pytest -q tests/test_rollout_policy.py`

Expected: all policy trace tests pass.

```bash
git add miniworld/rollout_policy.py tests/test_rollout_policy.py
git commit -m "feat: add adaptive rollout policy traces"
```

---

### Task 2: Retained-quality accounting and matched fixed baselines

**Files:**
- Modify: `miniworld/rollout_policy.py`
- Modify: `tests/test_rollout_policy.py`

**Interfaces:**
- Produces: immutable `EpisodePolicyResult` with episode, trace, retained error numerator/count, discarded numerator/count, and per-seed retained errors.
- Produces: `score_policy_trace(episode: int, trace: PolicyTrace, step_errors: Sequence[float], per_seed_step_errors: np.ndarray) -> EpisodePolicyResult`.
- Produces: `aggregate_policy_results(results: Sequence[EpisodePolicyResult], *, high_error_cutoff: float) -> dict[str, float | int]`.
- Produces: `matched_fixed_baseline(rows: Sequence[Mapping[str, float]], mean_horizon: float) -> dict[str, float]`.

- [ ] **Step 1: Write failing hand-calculated retained/discarded tests**

```python
def test_score_trace_separates_retained_and_discarded_error():
    trace = fixed_policy(2)
    result = score_policy_trace(
        1064,
        trace,
        [1.0, 3.0, 10.0, 20.0, 30.0],
        np.array([[1.0, 3.0, 10.0, 20.0, 30.0], [2.0, 4.0, 12.0, 22.0, 32.0]]),
    )
    assert result.retained_error_numerator == 4.0
    assert result.retained_count == 2
    assert result.discarded_error_numerator == 60.0
    assert result.discarded_count == 3
    np.testing.assert_allclose(result.per_seed_retained_error, [2.0, 3.0])
```

- [ ] **Step 2: Run focused scoring test and verify missing API failure**

Run: `pytest -q tests/test_rollout_policy.py -k 'score or matched or aggregate'`

- [ ] **Step 3: Implement numerator-first aggregation**

Do not average per-episode means to obtain the global MAE. Sum error numerators and retained-step counts first, then divide. Compute coverage from total retained steps over `episode_count * 5`; generated coverage is computed separately.

- [ ] **Step 4: Write failing analytic mixture test**

```python
def test_matched_fixed_baseline_interpolates_expected_numerators():
    rows = [
        {"episode": 1064, "future_latent_step": step, "error_rgb": error}
        for step, error in enumerate([1, 2, 3, 4, 5], start=1)
    ] + [
        {"episode": 1065, "future_latent_step": step, "error_rgb": error}
        for step, error in enumerate([2, 4, 6, 8, 10], start=1)
    ]
    result = matched_fixed_baseline(rows, mean_horizon=3.5)
    assert result["mean_horizon"] == pytest.approx(3.5)
    assert result["coverage"] == pytest.approx(0.7)
    assert result["retained_rgb_mae"] == pytest.approx(12.0 / 3.5)
```

The hand calculation is fixed-3 expected numerator `(6+12)/2=9` plus
half of the fourth-step mean `0.5*(4+8)/2=3`, giving `12` over expected
horizon `3.5`.

- [ ] **Step 5: Implement endpoint-safe fixed interpolation, run, and commit**

Handle mean horizons exactly 1 and 5 as pure fixed baselines. Reject values outside `[1,5]`.

Run: `pytest -q tests/test_rollout_policy.py`

```bash
git add miniworld/rollout_policy.py tests/test_rollout_policy.py
git commit -m "feat: score adaptive rollout coverage and quality"
```

---

### Task 3: LOEO threshold selection and pre-registered gate

**Files:**
- Modify: `miniworld/rollout_policy.py`
- Modify: `tests/test_rollout_policy.py`

**Interfaces:**
- Produces: `threshold_candidates(training_uncertainty: Sequence[float], *, max_candidates: int = 103) -> np.ndarray`.
- Produces: `choose_operating_point(candidates: Sequence[Mapping], *, target_coverage: float = 0.80) -> dict`.
- Produces: `select_threshold(training_rows: Sequence[Mapping], policy: str, *, target_coverage: float = 0.80) -> dict`.
- Produces: `run_loeo(rows: Sequence[Mapping], policy: str) -> tuple[list[dict], list[EpisodePolicyResult]]`.
- Produces: `evaluate_offline_gate(adaptive: dict, matched_fixed: dict, episode_deltas: Sequence[float]) -> dict[str, bool]`.

- [ ] **Step 1: Write failing candidate-isolation and tie-break tests**

```python
def test_threshold_candidates_use_only_training_values():
    candidates = threshold_candidates([0.1, 0.2, 0.3])
    assert 99.0 not in candidates
    assert candidates[0] < 0.1 and candidates[-1] > 0.3

def test_operating_point_prefers_lower_error_then_coverage_then_higher_tau():
    candidates = [
        {"tau": 0.6, "coverage": 0.8, "retained_rgb_mae": 2.0},
        {"tau": 0.8, "coverage": 0.7, "retained_rgb_mae": 1.0},
        {"tau": 0.7, "coverage": 0.8, "retained_rgb_mae": 2.0},
        {"tau": 0.5, "coverage": 0.9, "retained_rgb_mae": 2.0},
    ]
    selected = choose_operating_point(candidates, target_coverage=0.8)
    assert selected["retained_rgb_mae"] == pytest.approx(2.0)
    assert selected["coverage"] == pytest.approx(0.9)
    assert selected["tau"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run selection tests and verify missing API failures**

Run: `pytest -q tests/test_rollout_policy.py -k 'candidate or select or loeo or gate'`

- [ ] **Step 3: Implement deterministic candidates and selection**

Use unique sorted values when at most 101 exist. Otherwise use empirical quantiles `np.linspace(0,1,101)`, deduplicate, and add `np.nextafter(min,-inf)` and `np.nextafter(max,+inf)`. Evaluate every candidate through the policy and scoring APIs. Filter for coverage at least 0.80, then sort by `(retained_rgb_mae, -coverage, -tau)`.

- [ ] **Step 4: Write failing LOEO leakage and gate-boundary tests**

```python
def test_loeo_never_passes_held_out_uncertainty_to_candidate_builder(monkeypatch):
    rows = []
    for episode in range(1064, 1080):
        for step in range(1, 6):
            rows.append({
                "episode": episode,
                "future_latent_step": step,
                "uncertainty_latent": float(episode * 10 + step),
                "error_rgb": float(step),
                **{f"error_seed_{index}": float(step) for index in range(4)},
            })
    folds, _ = run_loeo(rows, "threshold")
    for fold in folds:
        held_out = fold["held_out_episode"]
        assert held_out not in fold["candidate_source_episodes"]
        held_out_values = {float(held_out * 10 + step) for step in range(1, 6)}
        assert held_out_values.isdisjoint(fold["threshold_candidates"])

def test_gate_requires_nine_episode_wins_and_tail_bound():
    adaptive = {"coverage": 0.8, "retained_rgb_mae": 4.0, "p90_episode_error": 6.0}
    fixed = {"retained_rgb_mae": 4.1, "p90_episode_error": 6.05}
    gate = evaluate_offline_gate(adaptive, fixed, [-0.1] * 9 + [0.1] * 7)
    assert gate == {
        "coverage_at_least_0_80": True,
        "retained_error_below_matched_fixed": True,
        "episode_wins_at_least_9_of_16": True,
        "p90_not_worse_by_more_than_0_10": True,
        "passed": True,
    }
```

- [ ] **Step 5: Implement LOEO/gate, run all policy tests, and commit**

Require exactly episodes 1064-1079 for formal LOEO and five ordered steps per episode. The high-error cutoff in each fold is the training fold's 75th error percentile.

Run: `pytest -q tests/test_rollout_policy.py`

```bash
git add miniworld/rollout_policy.py tests/test_rollout_policy.py
git commit -m "feat: select adaptive rollout thresholds with LOEO"
```

---

### Task 4: Offline evaluator and reconstructible artifacts

**Files:**
- Create: `scripts/evaluate_adaptive_rollout.py`
- Create: `tests/test_adaptive_rollout_evaluation.py`

**Interfaces:**
- Consumes: `--metrics_csv`, `--correlation_summary`, `--source_archive`, `--output_dir`, and `--overwrite`.
- Produces: `policy_decisions.csv`, `policy_curve.csv`, `loeo_folds.csv`, `adaptive_rollout_summary.json`, `counterexamples.csv`, and `report.md`.
- Produces: `load_formal_rows(path: Path) -> list[dict]`, `evaluate_offline(rows: list[dict], source_identity: dict) -> dict`, and `write_offline_outputs(output_dir: Path, result: dict, *, overwrite: bool) -> None` for tests.

- [ ] **Step 1: Write failing parser and sealed-split integrity tests**

```python
def write_metric_csv(root, *, episodes, steps, seed_columns=4):
    path = root / "metrics.csv"
    fieldnames = ["episode", "step", "latent_variance", "rgb_mae"] + [
        f"error_seed_{seed}" for seed in range(seed_columns)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            for step in steps:
                writer.writerow({
                    "episode": episode,
                    "step": step,
                    "latent_variance": episode * 10 + step,
                    "rgb_mae": float(step),
                    **{f"error_seed_{seed}": float(step + seed) for seed in range(seed_columns)},
                })
    return path

def test_loader_rejects_sealed_test_episode(tmp_path):
    csv_path = write_metric_csv(tmp_path, episodes=[1080], steps=range(1, 6))
    with pytest.raises(ValueError, match="sealed test"):
        load_formal_rows(csv_path)

def test_loader_requires_four_seed_error_columns_and_five_steps(tmp_path):
    csv_path = write_metric_csv(tmp_path, episodes=[1064], steps=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="five future steps"):
        load_formal_rows(csv_path)
```

- [ ] **Step 2: Run evaluator tests and verify import failure**

Run: `pytest -q tests/test_adaptive_rollout_evaluation.py`

- [ ] **Step 3: Implement strict loading and offline evaluation**

Parse all numerics explicitly, reject duplicate `(episode,step)`, non-finite values, missing `error_seed_0..3`, and any episode outside validation. Run fixed horizons 1-5 plus both adaptive policies. Build LOEO held-out decisions and full-validation deployment thresholds through Task 3 APIs. RGB policy curves are diagnostic and labeled non-gating.

- [ ] **Step 4: Write failing synthetic end-to-end output test**

```python
def test_writer_emits_exact_reconstructible_output_set(tmp_path):
    result = {
        "summary": {"source": {"archive_sha256": "abc"}, "primary_target_coverage": 0.8},
        "decisions": [{"episode": 1064, "policy": "threshold", "retained_horizon": 4}],
        "curve": [{"policy": "threshold", "tau": 0.5, "coverage": 0.8}],
        "folds": [{"held_out_episode": 1064, "tau": 0.5}],
        "counterexamples": [{"episode": 1064, "reason": "low uncertainty high error"}],
        "report": "OFFLINE PASS: ONLINE AUTHORIZED\n",
    }
    write_offline_outputs(tmp_path / "out", result, overwrite=False)
    output_names = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert output_names == [
        "adaptive_rollout_summary.json", "counterexamples.csv", "loeo_folds.csv",
        "policy_curve.csv", "policy_decisions.csv", "report.md",
    ]
    saved = json.loads((tmp_path / "out/adaptive_rollout_summary.json").read_text())
    assert saved["source"]["archive_sha256"] == "abc"
    assert saved["primary_target_coverage"] == 0.8
```

- [ ] **Step 5: Implement atomic writers/report, run, and commit**

The report states `OFFLINE PASS: ONLINE AUTHORIZED` only when the latent single-threshold or latent smoothed/hysteretic gate passes. Otherwise state `OFFLINE FAIL: ONLINE NOT AUTHORIZED`. Refuse non-empty output without `--overwrite`.

Run: `pytest -q tests/test_rollout_policy.py tests/test_adaptive_rollout_evaluation.py tests/test_uncertainty_evaluation.py`

```bash
git add scripts/evaluate_adaptive_rollout.py tests/test_adaptive_rollout_evaluation.py
git commit -m "feat: evaluate adaptive rollout offline"
```

---

### Task 5: Formal offline run, decision, and handoff

**Files:**
- Create: `docs/results/v100-official-055b-adaptive-rollout-offline.md`
- Modify: `docs/experiment_journal_zh.md`

**Interfaces:**
- Consumes the audited formal correlation CSV/JSON and archive.
- Produces an immutable offline policy result, a result archive, and either authorization or rejection for a separate online implementation plan.

- [ ] **Step 1: Preflight source identities**

Verify the correlation source contains 80 rows, episodes 1064-1079, steps 1-5, four seed error columns, `incomplete=false`, latent gate passed, and archive SHA `4fe4b2fb4b59dce77199b46fb82109a3d1197378ee6fd1a1b1df1dca2a354d86`. Record the current code commit and require a new output directory.

- [ ] **Step 2: Run the offline evaluator once**

```bash
python scripts/evaluate_adaptive_rollout.py \
  --metrics_csv /data/miniworld/experiments/official-055b-uncertainty-correlation-k4/metrics/metrics_per_episode_step.csv \
  --correlation_summary /data/miniworld/experiments/official-055b-uncertainty-correlation-k4/metrics/correlation_summary.json \
  --source_archive /data/miniworld/exports/miniworld-official-055b-uncertainty-correlation-k4.tar.gz \
  --output_dir /data/miniworld/experiments/official-055b-adaptive-rollout-offline
```

- [ ] **Step 3: Independently reconstruct the decision**

From `policy_decisions.csv`, recompute each policy's retained numerator/count, mean retained/generated horizons, coverage, episode wins, p90, matched-fixed interpolation, and all four gate booleans. Require exact agreement with JSON within `1e-12` for correlations/ratios and `1e-9` for RGB MAE.

- [ ] **Step 4: Record the result and package without source videos/checkpoints**

The report and journal must state the selected LOEO result, full-validation deployment threshold, curve, matched baseline, failure episodes, generated-versus-retained cost distinction, and whether online work is authorized. Package only offline CSV/JSON/report/log plus project reports, because the source K=4 archive already contains predictions. Record archive SHA256.

- [ ] **Step 5: Run fresh verification and commit**

Run in the established PyTorch container:

```bash
PYTHONPATH=. pytest -q -m "not cuda and not flash_attn"
git diff --check
git status --short
```

If the offline latent gate passes, stop after committing and write a separate online implementation plan from the approved spec. If it fails, record that online modification is not authorized and do not create an online plan.

```bash
git add miniworld/rollout_policy.py scripts/evaluate_adaptive_rollout.py tests/test_rollout_policy.py tests/test_adaptive_rollout_evaluation.py
git add -f docs/experiment_journal_zh.md docs/results/v100-official-055b-adaptive-rollout-offline.md
git commit -m "exp: evaluate adaptive rollout offline"
```
