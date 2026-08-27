# Chunk-Aligned Adaptive Rollout Stage A′ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate adaptive rollout at the official model's executable chunk-completion boundaries and authorize online early stopping only when quality passes and completed generated coverage is below 100%.

**Architecture:** Extend the pure rollout-policy module with an explicit chunk layout and chunk-aligned traces, then reuse the existing scoring and LOEO machinery through an execution-mode parameter. Add a separate Stage A′ evaluator and output directory so the earlier result stays immutable, independently reconstruct the new decision, and add correction notes to the project record.

**Tech Stack:** Python 3.11, dataclasses/enums, NumPy, CSV/JSON, pytest, Docker. No CUDA, model loading, sampling, decoding, training, or new dependency is required.

**Spec:** `docs/superpowers/specs/2026-08-27-chunk-aligned-adaptive-rollout-design.md`

## Global Constraints

- Consume only the frozen 80-row validation artifact for episodes 1064-1079; reject sealed test episodes 1080-1095.
- Keep `history_len=1`, `df_chunk_size=2`, `H_min=1`, `H_max=5`, latent `K=4`, EMA alpha `0.5`, and hysteresis count `2` fixed.
- Formal selection uses latent population variance; RGB disagreement is diagnostic only.
- Evaluate policy state sequentially for every completed future frame, but emit an external decision only after its whole model chunk completes.
- For the official layout, completion boundaries are exactly `(1,3,5)`.
- Keep the previous formal result directory and archive immutable; corrections go in project docs and a new Stage A′ result directory.
- Online work requires all four existing quality gates plus completed generated coverage strictly below `1.0`.
- Completed-chunk cost is a proxy, not a wall-time or FLOP speedup claim.
- Preserve local CSV/JSON as the source of truth and keep the full non-CUDA/non-FlashAttention suite green.

---

### Task 1: Executable chunk layouts and policy traces

**Files:**
- Modify: `miniworld/rollout_policy.py`
- Modify: `tests/test_rollout_policy.py`

**Interfaces:**
- Produces: immutable `ChunkLayout(history_len: int, future_horizon: int, chunk_size: int, completion_boundaries: tuple[int, ...])`.
- Produces: `build_chunk_layout(*, history_len: int, future_horizon: int, chunk_size: int) -> ChunkLayout`.
- Produces: `chunk_aligned_threshold_policy(uncertainty: Sequence[float], tau: float, *, layout: ChunkLayout) -> PolicyTrace`.
- Produces: `chunk_aligned_smoothed_hysteretic_policy(uncertainty: Sequence[float], tau: float, *, layout: ChunkLayout, alpha: float = 0.5, consecutive: int = 2) -> PolicyTrace`.

- [ ] **Step 1: Write failing hand-calculated layout tests**

```python
def test_official_chunk_layout_completes_future_at_one_three_five():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    assert layout.completion_boundaries == (1, 3, 5)

def test_layout_keeps_partial_final_chunk():
    layout = build_chunk_layout(history_len=2, future_horizon=5, chunk_size=3)
    assert layout.completion_boundaries == (1, 4, 5)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_len": 0, "future_horizon": 5, "chunk_size": 2},
        {"history_len": 1, "future_horizon": 0, "chunk_size": 2},
        {"history_len": 1, "future_horizon": 5, "chunk_size": 0},
    ],
)
def test_chunk_layout_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        build_chunk_layout(**kwargs)
```

- [ ] **Step 2: Run the layout tests and verify the missing API failure**

Run: `pytest -q tests/test_rollout_policy.py -k 'chunk_layout or layout_'`

Expected: collection fails because `ChunkLayout` and `build_chunk_layout` do not exist.

- [ ] **Step 3: Implement layout construction from global model chunks**

Compute global chunk ends with `range(chunk_size, history_len + future_horizon, chunk_size)`, append the partial final end when necessary, subtract `history_len`, discard non-positive values, clamp to `future_horizon`, and deduplicate while preserving order. Validate that the last completion boundary equals `future_horizon`.

```python
@dataclass(frozen=True)
class ChunkLayout:
    history_len: int
    future_horizon: int
    chunk_size: int
    completion_boundaries: tuple[int, ...]

def build_chunk_layout(*, history_len: int, future_horizon: int, chunk_size: int) -> ChunkLayout:
    if min(history_len, future_horizon, chunk_size) <= 0:
        raise ValueError("history_len, future_horizon, and chunk_size must be positive")
    total = history_len + future_horizon
    global_ends = list(range(chunk_size, total, chunk_size)) + [total]
    boundaries = tuple(dict.fromkeys(
        min(future_horizon, end - history_len)
        for end in global_ends
        if end > history_len
    ))
    if not boundaries or boundaries[-1] != future_horizon:
        raise ValueError("chunk layout does not cover the future horizon")
    return ChunkLayout(history_len, future_horizon, chunk_size, boundaries)
```

- [ ] **Step 4: Write failing chunk-aligned trigger tests**

```python
def test_step_four_trigger_waits_for_chunk_five_and_retains_three():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    trace = chunk_aligned_smoothed_hysteretic_policy(
        [0.1, 0.1, 0.9, 0.9, 0.1], tau=0.45, layout=layout
    )
    assert trace.requested_observation_at == 4
    assert trace.generated_horizon == 5
    assert trace.retained_horizon == 3
    assert trace.decisions[-1] is Decision.REQUEST_OBSERVATION

def test_trigger_inside_middle_chunk_emits_after_boundary_three():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    trace = chunk_aligned_threshold_policy(
        [0.1, 0.9, 0.1, 0.1, 0.1], tau=0.5, layout=layout
    )
    assert trace.requested_observation_at == 2
    assert trace.generated_horizon == 3
    assert trace.retained_horizon == 1

def test_chunk_policy_rejects_uncertainty_length_different_from_layout():
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    with pytest.raises(ValueError, match="future_horizon"):
        chunk_aligned_threshold_policy([0.1] * 4, tau=0.5, layout=layout)
```

The first test produces EMA values `0.1, 0.1, 0.5, 0.7, ...`, so step 4 is the second consecutive exceedance; both steps 4 and 5 belong to the final chunk, so generated horizon is 5. The middle-chunk test proves `requested_observation_at` and emission boundary are distinct.

- [ ] **Step 5: Implement chunk-aligned state evaluation**

Evaluate raw/EMA state one future step at a time. Track the first triggering step within the current completed chunk, but return only after processing through that chunk's completion boundary. Retain the preceding boundary, using 1 when there is no earlier boundary. Preserve the original `PolicyTrace.decisions` convention: one decision per inspected future step, with the last inspected decision replaced by `REQUEST_OBSERVATION`; do not add a second decision for the boundary.

- [ ] **Step 6: Run all policy tests and commit**

Run: `pytest -q tests/test_rollout_policy.py`

```bash
git add miniworld/rollout_policy.py tests/test_rollout_policy.py
git commit -m "feat: model executable rollout chunk boundaries"
```

---

### Task 2: Chunk-aware LOEO selection and fifth gate

**Files:**
- Modify: `miniworld/rollout_policy.py`
- Modify: `tests/test_rollout_policy.py`

**Interfaces:**
- Modifies: `select_threshold(..., execution: str = "frame", layout: ChunkLayout | None = None) -> dict[str, object]`.
- Modifies: `run_loeo(..., execution: str = "frame", layout: ChunkLayout | None = None) -> tuple[list[dict], list[EpisodePolicyResult]]`.
- Produces: `evaluate_chunk_aligned_gate(adaptive: Mapping[str, float], matched_fixed: Mapping[str, float], episode_deltas: Sequence[float]) -> dict[str, bool]`.
- Preserves: default `execution="frame"` behavior and every existing Stage A test.

- [ ] **Step 1: Write failing execution-mode and regression tests**

```python
def _literal_training_rows():
    return [
        {
            "episode": episode,
            "future_latent_step": step,
            "uncertainty_latent": episode * 0.001 + step * 0.01,
            "error_rgb": float(step),
            **{f"error_seed_{seed}": float(step) for seed in range(4)},
        }
        for episode in range(1064, 1079)
        for step in range(1, 6)
    ]

def test_chunk_execution_scores_completed_boundary_not_trigger_step():
    rows = []
    for episode in range(1064, 1080):
        for step, uncertainty in enumerate([0.1, 0.1, 0.1, 0.9, 0.9], start=1):
            rows.append({
                "episode": episode,
                "future_latent_step": step,
                "uncertainty_latent": uncertainty,
                "error_rgb": float(step),
                **{f"error_seed_{seed}": float(step) for seed in range(4)},
            })
    layout = build_chunk_layout(history_len=1, future_horizon=5, chunk_size=2)
    _, results = run_loeo(rows, "smoothed_hysteretic", execution="chunk", layout=layout)
    assert {result.trace.generated_horizon for result in results} == {5}
    assert {result.trace.retained_horizon for result in results} <= {3, 5}

def test_frame_execution_remains_the_default():
    selected = select_threshold(_literal_training_rows(), "threshold")
    explicit = select_threshold(_literal_training_rows(), "threshold", execution="frame")
    assert selected["tau"] == explicit["tau"]
    assert selected["coverage"] == explicit["coverage"]
```

Do not call production row builders from this literal fixture.

- [ ] **Step 2: Run focused tests and verify the unexpected-keyword failure**

Run: `pytest -q tests/test_rollout_policy.py -k 'chunk_execution or frame_execution'`

- [ ] **Step 3: Route scoring through explicit execution mode**

Add `execution` and `layout` to the internal episode scorer. Accept only `"frame"` and `"chunk"`. Require `layout` for chunk execution and reject it for frame execution. Route threshold and smoothed/hysteretic names to the matching frame or chunk functions. Include `execution`, layout fields, and completion boundaries in every selection and fold record.

- [ ] **Step 4: Write failing fifth-gate boundary tests**

```python
def test_chunk_gate_rejects_quality_pass_without_avoided_complete_chunk():
    adaptive = {
        "coverage": 0.8,
        "generated_coverage": 1.0,
        "retained_rgb_mae": 4.0,
        "p90_episode_error": 6.0,
    }
    fixed = {"retained_rgb_mae": 4.1, "p90_episode_error": 6.05}
    gate = evaluate_chunk_aligned_gate(adaptive, fixed, [-0.1] * 9 + [0.1] * 7)
    assert gate["quality_gate_passed"] is True
    assert gate["completed_generated_coverage_below_1_00"] is False
    assert gate["passed"] is False

def test_chunk_gate_passes_when_quality_and_cost_both_pass():
    adaptive = {
        "coverage": 0.8,
        "generated_coverage": 0.9,
        "retained_rgb_mae": 4.0,
        "p90_episode_error": 6.0,
    }
    fixed = {"retained_rgb_mae": 4.1, "p90_episode_error": 6.05}
    gate = evaluate_chunk_aligned_gate(adaptive, fixed, [-0.1] * 9 + [0.1] * 7)
    assert gate["completed_generated_coverage_below_1_00"] is True
    assert gate["passed"] is True
```

- [ ] **Step 5: Implement the composed gate and cost fields**

Call `evaluate_offline_gate` for the four quality checks, expose them unchanged, add `quality_gate_passed`, add the strict completed-coverage check, and define top-level `passed` as their conjunction. Extend aggregation with `median_episode_error`, `worst_episode_error`, `request_rate`, and `avoided_completed_future_steps = episode_count * 5 - generated_count`. Existing keys and numeric definitions must not change.

- [ ] **Step 6: Run all policy and earlier evaluator tests and commit**

Run: `pytest -q tests/test_rollout_policy.py tests/test_adaptive_rollout_evaluation.py`

```bash
git add miniworld/rollout_policy.py tests/test_rollout_policy.py
git commit -m "feat: select chunk-aligned rollout policies"
```

---

### Task 3: Separate Stage A′ evaluator and reconstructible outputs

**Files:**
- Create: `scripts/evaluate_adaptive_rollout_chunk_aligned.py`
- Create: `tests/test_chunk_aligned_rollout_evaluation.py`

**Interfaces:**
- Consumes: `--metrics_csv`, `--correlation_summary`, `--sampling_manifest`, `--source_archive`, `--previous_summary`, `--code_commit`, `--output_dir`, and `--overwrite`.
- Reuses: `load_formal_rows` and source SHA constant from `scripts/evaluate_adaptive_rollout.py`.
- Produces: `evaluate_chunk_aligned(rows: list[dict], source_identity: dict, previous_summary: dict) -> dict`.
- Produces: `write_chunk_aligned_outputs(output_dir: Path, result: dict, *, overwrite: bool) -> None`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `validate_cli_identity(args: argparse.Namespace) -> None`, and `build_report(summary: dict) -> str`.
- Writes exactly: `policy_decisions.csv`, `policy_curve.csv`, `loeo_folds.csv`, `chunk_costs.csv`, `adaptive_rollout_summary.json`, `counterexamples.csv`, and `report.md`.

- [ ] **Step 1: Write failing correction and output-contract tests**

```python
def _formal_literal_rows():
    return [
        {
            "episode": episode,
            "future_latent_step": step,
            "uncertainty_latent": uncertainty,
            "uncertainty_rgb": float(step) / 100.0,
            "error_rgb": float(step),
            **{f"error_seed_{seed}": float(step) for seed in range(4)},
        }
        for episode in range(1064, 1080)
        for step, uncertainty in enumerate(
            [0.01, 0.01, 0.01, 0.04, 0.04], start=1
        )
    ]

def test_previous_policy_recomputes_to_full_completed_generation():
    previous = {
        "adaptive": {
            "smoothed_hysteretic": {
                "loeo": {"generated_coverage": 0.9375},
                "deployment_threshold": 0.020729146897792816,
            }
        }
    }
    rows = _formal_literal_rows()
    result = evaluate_chunk_aligned(rows, {"code_commit": "abc"}, previous)
    correction = result["summary"]["correction"]
    assert correction["previous_idealized_generated_coverage"] == 0.9375
    assert correction["previous_chunk_completed_generated_coverage"] == 1.0
    assert correction["previous_online_authorization_superseded"] is True

def test_chunk_writer_emits_exact_output_set(tmp_path):
    result = {
        "summary": {"online_authorized": False},
        "decisions": [{"episode": 1064, "policy": "threshold"}],
        "curve": [{"policy": "threshold", "tau": 0.1}],
        "folds": [{"held_out_episode": 1064, "tau": 0.1}],
        "costs": [{"policy": "threshold", "completed_k4_member_steps": 20}],
        "counterexamples": [{"episode": 1064, "reason": "no complete chunk avoided"}],
        "report": "CHUNK-ALIGNED FAIL: ONLINE NOT AUTHORIZED\n",
    }
    write_chunk_aligned_outputs(tmp_path / "out", result, overwrite=False)
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "adaptive_rollout_summary.json", "chunk_costs.csv", "counterexamples.csv",
        "loeo_folds.csv", "policy_curve.csv", "policy_decisions.csv", "report.md",
    ]
```

This fixture makes the previous threshold trigger only inside the final chunk and is independent of production helpers.

- [ ] **Step 2: Run evaluator tests and verify import failure**

Run: `pytest -q tests/test_chunk_aligned_rollout_evaluation.py`

- [ ] **Step 3: Implement strict evaluation and correction replay**

Use `ChunkLayout(1,5,2,(1,3,5))` from `build_chunk_layout`, and run both latent policies through chunk execution LOEO. Recompute the previous frozen deployment policy with threshold `0.020729146897792816`; require its completed generated coverage to equal `1.0` on the formal data or reject the run. Compute matched fixed results and all five gate booleans. Select the online candidate only among passing latent policies using `(retained_rgb_mae, generated_coverage, -coverage, -tau)`.

For each policy, `chunk_costs.csv` must include:

```text
completed_future_steps
completed_k4_member_steps = completed_future_steps * 4
fixed_k4_member_steps = 16 * 5 * 4
fixed_k1_member_steps = 16 * 5
k4_fraction_of_fixed_k4
k4_ratio_to_fixed_k1
avoided_complete_future_steps
wall_time_claim = false
```

Record raw/EMA trace fields, first trigger, completion boundary, retained/generated horizons, numerators/counts, request rate, median/p90/worst errors, deployment threshold, fold candidates and source episodes. RGB curves must be labeled `gating=false`.

- [ ] **Step 4: Write failing source-identity and status tests**

```python
def test_cli_requires_concrete_code_commit():
    args = build_parser().parse_args([
        "--metrics_csv", "metrics.csv",
        "--correlation_summary", "summary.json",
        "--sampling_manifest", "sampling_manifest.json",
        "--source_archive", "source.tar.gz",
        "--previous_summary", "previous.json",
        "--code_commit", "unknown",
        "--output_dir", "out",
    ])
    with pytest.raises(ValueError, match="concrete code commit"):
        validate_cli_identity(args)

def test_report_authorizes_only_a_passing_latent_policy():
    report = build_report({
        "online_authorized": False,
        "adaptive": {
            "threshold": {"gate": {"passed": False}},
            "smoothed_hysteretic": {"gate": {"passed": False}},
        },
    })
    assert report.startswith("CHUNK-ALIGNED FAIL: ONLINE NOT AUTHORIZED")
```

- [ ] **Step 5: Implement CLI identity validation, atomic writers, and report**

Require a 40-character lowercase hexadecimal `--code_commit`; validate the frozen archive SHA, 80 rows, K=4, `incomplete=false`, latent correlation gate pass, and exact validation episode list. Load the combined sampling manifest, require its `k`, seeds, episodes, and sampling configuration to match the correlation summary, and record its checkpoint SHA, data-manifest SHA, sampling Git commit, and `df_chunk_size=2`. Refuse a non-empty output directory unless `--overwrite`. The status is `CHUNK-ALIGNED PASS: ONLINE AUTHORIZED` only when a latent policy passes all five gates; otherwise use the exact failure string from the test.

- [ ] **Step 6: Run all related tests and commit**

Run: `pytest -q tests/test_rollout_policy.py tests/test_adaptive_rollout_evaluation.py tests/test_chunk_aligned_rollout_evaluation.py tests/test_uncertainty_evaluation.py`

```bash
git add scripts/evaluate_adaptive_rollout_chunk_aligned.py tests/test_chunk_aligned_rollout_evaluation.py
git commit -m "feat: audit adaptive rollout at chunk boundaries"
```

---

### Task 4: Formal Stage A′ run, independent reconstruction, and correction record

**Files:**
- Create: `docs/results/v100-official-055b-adaptive-rollout-chunk-aligned.md`
- Modify: `docs/results/v100-official-055b-adaptive-rollout-offline.md`
- Modify: `docs/experiment_journal_zh.md`

**Interfaces:**
- Consumes the frozen correlation CSV/JSON/archive and immutable previous Stage A summary.
- Produces a new Stage A′ result directory and archive, plus a corrected online-authorization decision.

- [ ] **Step 1: Preflight immutable identities and new output path**

Verify:

```text
metrics rows = 80 plus one CSV header
episodes = 1064-1079
future steps = 1-5
K = 4
correlation incomplete = false
latent correlation gate = true
source archive SHA256 = 4fe4b2fb4b59dce77199b46fb82109a3d1197378ee6fd1a1b1df1dca2a354d86
combined sampling manifest = /data/miniworld/experiments/official-055b-uncertainty-correlation-k4/metrics/sampling_manifest_combined.json
checkpoint/data/seeds/episodes/sampling identities agree between manifest and summary
previous summary path = /data/miniworld/experiments/official-055b-adaptive-rollout-offline/adaptive_rollout_summary.json
new output path does not exist
code commit is concrete
```

- [ ] **Step 2: Run the formal evaluator once**

```bash
python scripts/evaluate_adaptive_rollout_chunk_aligned.py \
  --metrics_csv /data/miniworld/experiments/official-055b-uncertainty-correlation-k4/metrics/metrics_per_episode_step.csv \
  --correlation_summary /data/miniworld/experiments/official-055b-uncertainty-correlation-k4/metrics/correlation_summary.json \
  --sampling_manifest /data/miniworld/experiments/official-055b-uncertainty-correlation-k4/metrics/sampling_manifest_combined.json \
  --source_archive /data/miniworld/exports/miniworld-official-055b-uncertainty-correlation-k4.tar.gz \
  --previous_summary /data/miniworld/experiments/official-055b-adaptive-rollout-offline/adaptive_rollout_summary.json \
  --code_commit "$(git rev-parse HEAD)" \
  --output_dir /data/miniworld/experiments/official-055b-adaptive-rollout-chunk-aligned
```

- [ ] **Step 3: Independently reconstruct the decision without importing project evaluators**

Write a one-use verifier under `/tmp` using only Python standard-library `csv`, `json`, `math`, and `pathlib`. From `policy_decisions.csv` and the frozen source CSV, recompute per policy:

```text
retained/generated numerator and count
retained/generated coverage
mean retained/generated horizon
request rate
median/p90/worst episode error
matched fixed expected numerator and MAE
16 episode deltas and win count
all five gate booleans
K=4 completed-member steps and K=1 ratio
```

Require agreement with JSON/CSV within `1e-12` for ratios/count-derived values and `1e-9` for RGB MAE. Require the previous frozen policy replay to produce completed generated coverage exactly `1.0`.

- [ ] **Step 4: Record the corrected result and package lightweight artifacts**

The new report and journal entry must state why Stage A was insufficient, the `(1,3,5)` boundaries, LOEO thresholds/results, matched baseline, failure episodes, completed-chunk costs, lack of a wall-time claim, and the final online authorization. Add an explicit correction box to the earlier Stage A report: its retained-quality result remains valid, but `generated_coverage=0.9375` was an idealized frame proxy and its online authorization is superseded.

Package the seven Stage A′ artifacts plus the new/corrected project reports, excluding source videos, latents, and checkpoints:

```text
/data/miniworld/exports/miniworld-official-055b-adaptive-rollout-chunk-aligned.tar.gz
```

Record its SHA256 and size after packaging.

- [ ] **Step 5: Run fresh verification and commit**

Run in the established isolated PyTorch test container:

```bash
PYTHONPATH=. pytest -q -m "not cuda and not flash_attn"
git diff --check
git status --short
```

If Stage A′ passes, stop after committing and write a separate online implementation plan. If it fails, explicitly keep online model code unchanged and do not create an online plan.

```bash
git add miniworld/rollout_policy.py scripts/evaluate_adaptive_rollout_chunk_aligned.py tests/test_rollout_policy.py tests/test_chunk_aligned_rollout_evaluation.py
git add -f docs/experiment_journal_zh.md docs/results/v100-official-055b-adaptive-rollout-offline.md docs/results/v100-official-055b-adaptive-rollout-chunk-aligned.md
git commit -m "exp: audit adaptive rollout at chunk boundaries"
```
