# Uncertainty–Error Correlation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible validation-only evaluator that measures whether `K=4` stochastic MiniWorld predictions disagree where future RGB prediction error is high.

**Architecture:** Add pure CPU tensor/statistics functions in `miniworld/uncertainty.py`, add opt-in latent and sampling-manifest export to the existing sampler, and add one offline DROID ensemble evaluator. Keep model sampling separate from statistics so metric tests need no model, VAE, or GPU.

**Tech Stack:** Python 3.11, PyTorch 2.4, NumPy, pandas, decord, pytest, Bash, Docker, V100 FP16 with PyTorch SDPA.

**Spec:** `docs/superpowers/specs/2026-08-27-uncertainty-error-correlation-design.md`

## Global Constraints

- Freeze the official `MiniWorld_0_5b_droid.pt`; do not train or modify weights.
- Use validation episodes 1064–1079 only; reject test episodes 1080–1095.
- Formal sampling uses four distinct deterministic seeds and real actions.
- Six latent frames map to 21 RGB frames; exclude context index zero and map each future latent to four RGB frames.
- Primary uncertainty is latent population variance; RGB pairwise disagreement is the control.
- Target error is mean member-wise RGB MAE on the 0–255 scale.
- Do not silently accept missing seeds, episodes, files, shape mismatch, or non-finite values.
- Local CSV/JSON/manifests are the source of truth; W&B is optional mirroring.
- Follow TDD and keep every existing non-CUDA/non-FlashAttention test passing.

---

### Task 1: Pure ensemble metrics and temporal alignment

**Files:**
- Create: `miniworld/uncertainty.py`
- Create: `tests/test_uncertainty.py`

**Interfaces:**
- Produces: `future_rgb_blocks(num_latent_frames: int, num_rgb_frames: int) -> list[slice]`
- Produces: `latent_population_variance(ensemble: torch.Tensor) -> torch.Tensor`
- Produces: `rgb_pairwise_disagreement(ensemble: torch.Tensor, blocks: list[slice]) -> torch.Tensor`
- Produces: `rgb_memberwise_mae(ensemble: torch.Tensor, target: torch.Tensor, blocks: list[slice]) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing hand-calculated metric tests**

```python
def test_future_rgb_blocks_excludes_context():
    assert future_rgb_blocks(6, 21) == [slice(1, 5), slice(5, 9), slice(9, 13), slice(13, 17), slice(17, 21)]

def test_latent_population_variance():
    z = torch.tensor([0.0, 2.0]).reshape(2, 1, 1, 1, 1)
    torch.testing.assert_close(latent_population_variance(z), torch.tensor([1.0]))

def test_rgb_disagreement_and_memberwise_error():
    x = torch.tensor([[0.0, 2.0], [2.0, 4.0]]).reshape(2, 2, 1, 1, 1)
    gt = torch.tensor([1.0, 1.0]).reshape(2, 1, 1, 1)
    blocks = [slice(0, 2)]
    torch.testing.assert_close(rgb_pairwise_disagreement(x, blocks), torch.tensor([2.0]))
    mean_error, per_seed = rgb_memberwise_mae(x, gt, blocks)
    torch.testing.assert_close(mean_error, torch.tensor([1.5]))
    torch.testing.assert_close(per_seed, torch.tensor([[0.5], [2.5]]))
```

- [ ] **Step 2: Run the focused tests and confirm missing imports fail**

Run: `pytest -q tests/test_uncertainty.py`

Expected: collection fails because `miniworld.uncertainty` does not exist.

- [ ] **Step 3: Implement shape validation and the minimal metric functions**

```python
def future_rgb_blocks(num_latent_frames: int, num_rgb_frames: int) -> list[slice]:
    future_latents = num_latent_frames - 1
    future_rgb = num_rgb_frames - 1
    if future_latents <= 0 or future_rgb != 4 * future_latents:
        raise ValueError("expected one context frame and four RGB frames per future latent")
    return [slice(1 + 4 * j, 1 + 4 * (j + 1)) for j in range(future_latents)]

def latent_population_variance(ensemble: torch.Tensor) -> torch.Tensor:
    _validate_finite(ensemble, ndim=5, name="latent ensemble")
    return ensemble.float().var(dim=0, correction=0).mean(dim=(0, 2, 3))
```

Implement RGB pairwise disagreement over every `a < b`, and return member-wise as well as mean error without averaging generated videos.

- [ ] **Step 4: Add rejection tests for invalid dimensions, temporal mapping, and NaN**

```python
@pytest.mark.parametrize("bad", [torch.zeros(2, 3), torch.full((2, 1, 2, 1, 1), float("nan"))])
def test_latent_metric_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        latent_population_variance(bad)

def test_temporal_mapping_rejects_wrong_rgb_length():
    with pytest.raises(ValueError, match="four RGB frames"):
        future_rgb_blocks(6, 20)
```

- [ ] **Step 5: Run and commit Task 1**

Run: `pytest -q tests/test_uncertainty.py`

Expected: all Task 1 tests pass.

```bash
git add miniworld/uncertainty.py tests/test_uncertainty.py
git commit -m "feat: add uncertainty ensemble metrics"
```

---

### Task 2: Correlations, horizon conditioning, and quantile bins

**Files:**
- Modify: `miniworld/uncertainty.py`
- Modify: `tests/test_uncertainty.py`

**Interfaces:**
- Consumes: one-dimensional finite NumPy arrays produced from Task 1 tensors.
- Produces: `pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None`
- Produces: `spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None`
- Produces: `horizon_conditioned_spearman(x: np.ndarray, y: np.ndarray, horizons: np.ndarray) -> float | None`
- Produces: `equal_count_bins(x: np.ndarray, y: np.ndarray, bins: int = 4) -> list[dict[str, float | int]]`

- [ ] **Step 1: Write failing exact statistical tests**

```python
def test_correlations_and_average_tie_ranks():
    x = np.array([1.0, 2.0, 2.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_correlation(y, y) == pytest.approx(1.0)
    assert spearman_correlation(x, y) == pytest.approx(0.9486832980505139)

def test_constant_correlation_is_undefined():
    assert pearson_correlation(np.ones(4), np.arange(4)) is None
    assert spearman_correlation(np.ones(4), np.arange(4)) is None

def test_horizon_conditioning_removes_between_horizon_effect():
    horizon = np.array([1, 1, 2, 2])
    x = np.array([1.0, 2.0, 10.0, 9.0])
    y = np.array([1.0, 2.0, 9.0, 10.0])
    assert horizon_conditioned_spearman(x, y, horizon) == pytest.approx(0.0)
```

- [ ] **Step 2: Run the focused tests and confirm undefined functions fail**

Run: `pytest -q tests/test_uncertainty.py -k 'correlation or horizon or bins'`

Expected: failures name the new undefined functions.

- [ ] **Step 3: Implement average ranks, correlations, and stable equal-count bins**

```python
def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks
```

Use NumPy covariance after rejecting constant vectors. For horizon conditioning, rank independently inside each horizon and correlate the pooled within-horizon ranks. Use a stable sort and `np.array_split` so each observation appears in exactly one bin.

- [ ] **Step 4: Add bin conservation and validation tests**

```python
def test_equal_count_bins_preserve_every_observation():
    rows = equal_count_bins(np.arange(10.0), np.arange(10.0) ** 2, bins=4)
    assert sum(row["count"] for row in rows) == 10
    assert rows[-1]["mean_error"] > rows[0]["mean_error"]
```

- [ ] **Step 5: Run and commit Task 2**

Run: `pytest -q tests/test_uncertainty.py`

```bash
git add miniworld/uncertainty.py tests/test_uncertainty.py
git commit -m "feat: add uncertainty correlation statistics"
```

---

### Task 3: Opt-in latent and sampling-manifest export

**Files:**
- Modify: `miniworld/sample.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces CLI flag: `--save_latents` with default `False`.
- Produces: `save_sample_latents(root: Path, index: int, latents: torch.Tensor) -> Path`
- Produces: `sampling_manifest.json` containing seed, checkpoint, weights source, action variant, sampler configuration, and per-sample latent metadata.

- [ ] **Step 1: Write failing parser and serialization tests**

```python
def test_sampling_parser_accepts_opt_in_latent_export():
    args = build_sample_parser().parse_args(_required_sample_args() + ["--save_latents"])
    assert args.save_latents is True

def test_save_sample_latents_uses_cpu_tensor_and_stable_name(tmp_path):
    value = torch.ones(1, 2, 3, 4, 5)
    path = save_sample_latents(tmp_path, 7, value)
    assert path == tmp_path / "latents" / "sample_0007.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved.device.type == "cpu"
    torch.testing.assert_close(saved, value)
```

- [ ] **Step 2: Run tests and confirm the new parser/helper tests fail**

Run: `pytest -q tests/test_cli.py -k 'latent_export or sample_latents'`

- [ ] **Step 3: Add the flag and save only the final detached tensor**

```python
def save_sample_latents(root: Path, index: int, latents: torch.Tensor) -> Path:
    path = root / "latents" / f"sample_{index:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents.detach().cpu(), path)
    return path
```

Call it after generation only when `args.save_latents and not args.benchmark_no_save`. Accumulate sample metadata and atomically write `sampling_manifest.json` after the loop. Preserve the existing default output exactly when the flag is absent.

- [ ] **Step 4: Test manifest content and default non-export behavior**

Use a pure `build_sampling_manifest(args, samples, identities)` helper in
`sample.py`; assert that it records `seed`, checkpoint path/SHA256,
`weights_source`, data-root manifest SHA256, ordered episode IDs,
`action_variant`, `total_len`, sampler arguments, precision/backend requests,
Git commit, and tensor shape/dtype, while `save_latents=False` creates no latent
directory. Add a streaming `sha256_file(path, chunk_bytes=8*1024*1024)` helper
so checkpoint hashing does not allocate checkpoint-sized memory.

- [ ] **Step 5: Run and commit Task 3**

Run: `pytest -q tests/test_cli.py`

```bash
git add miniworld/sample.py tests/test_cli.py
git commit -m "feat: export sampling latents and manifest"
```

---

### Task 4: Offline DROID ensemble evaluator

**Files:**
- Create: `scripts/evaluate_droid_uncertainty.py`
- Create: `tests/test_uncertainty_evaluation.py`

**Interfaces:**
- Consumes: `--data_root`, repeated `--sample_root` values, `--output_dir`,
  `--overwrite`, and `--allow_incomplete` (integration gate only).
- Produces: `metrics_per_episode_step.csv`, `correlation_summary.json`, `uncertainty_bins.csv`, `horizon_summary.csv`, `sampling_manifest_combined.json`, and `report.md`.
- Produces: `validate_manifests(manifests: list[dict]) -> dict` and `evaluate_ensemble(latents, rgb, ground_truth) -> list[dict]` for focused tests.

- [ ] **Step 1: Write failing manifest-integrity tests**

```python
def test_manifest_validation_rejects_test_episode():
    manifests = make_manifests(seeds=[11, 22, 33, 44], episodes=list(range(1080, 1096)))
    with pytest.raises(ValueError, match="sealed test"):
        validate_manifests(manifests)

def test_manifest_validation_rejects_duplicate_seed():
    manifests = make_manifests(seeds=[11, 11, 33, 44], episodes=list(range(1064, 1080)))
    with pytest.raises(ValueError, match="distinct seeds"):
        validate_manifests(manifests)
```

Also test mismatched checkpoint/configuration, fewer than four roots without
`--allow_incomplete`, incomplete output missing its visible marker, missing
episode, shape mismatch, and non-finite latent values.

- [ ] **Step 2: Run the evaluator tests and confirm import failure**

Run: `pytest -q tests/test_uncertainty_evaluation.py`

- [ ] **Step 3: Implement strict loading, per-step rows, and deterministic outputs**

Use `LeRobotActionDataset` with the same deterministic 21-frame configuration as the action-ablation evaluator. Load each MP4 as uint8, each latent as a CPU tensor, stack seed dimension first, compute the five rows per episode through `miniworld.uncertainty`, and sort rows by `(episode, future_latent_step)` before writing.

```python
summary = {
    "split": "validation",
    "episodes": [1064, 1079],
    "count_episode_steps": 80,
    "k": 4,
    "estimators": {
        "latent": summarize_estimator(rows, "uncertainty_latent"),
        "rgb": summarize_estimator(rows, "uncertainty_rgb"),
    },
    "gate": evaluate_signal_gate(rows),
}
```

Represent undefined correlations as JSON `null`. Write files through temporary siblings followed by `Path.replace`; refuse a non-empty output directory without `--overwrite`.

- [ ] **Step 4: Add a synthetic end-to-end fixture**

Construct two small episodes with four saved seeds and a monkeypatched video reader/dataset. Run `main()` and assert deterministic row ordering, exact output filenames, 80-row logic generalized to fixture size, JSON `null` handling, and that every report number comes from the JSON/CSV values.

- [ ] **Step 5: Run and commit Task 4**

Run: `pytest -q tests/test_uncertainty.py tests/test_uncertainty_evaluation.py tests/test_action_ablation_evaluation.py`

```bash
git add scripts/evaluate_droid_uncertainty.py tests/test_uncertainty_evaluation.py
git commit -m "feat: evaluate DROID uncertainty correlation"
```

---

### Task 5: Launcher, full regression, and one-episode GPU gate

**Files:**
- Create: `scripts/evaluate_droid_uncertainty.sh`
- Modify: `tests/test_launchers.py`
- Modify: `docs/experiment_journal_zh.md`

**Interfaces:**
- Consumes environment variables `DATA_ROOT`, `CKPT`, `VAE_CKPT`, `OUTPUT_ROOT`, `GPU`, and optional comma-separated `SEEDS` defaulting to four fixed values.
- Produces four seed directories and invokes the offline evaluator only after all sampling commands succeed.

- [ ] **Step 1: Write a failing launcher rendering test**

Assert the launcher uses official 0.55B, real actions, `TOTAL_LEN=6`, `SAMPLE_NUM_VIDEOS=16`, FP16, SDPA, `--save_latents`, four distinct seeds, and never references episode IDs 1080–1095.

- [ ] **Step 2: Implement the fail-fast launcher**

```bash
IFS=',' read -r -a SEED_VALUES <<< "${SEEDS:-20260827,20260828,20260829,20260830}"
[[ "${#SEED_VALUES[@]}" -eq 4 ]] || { echo "SEEDS must contain exactly four values" >&2; exit 2; }
for seed in "${SEED_VALUES[@]}"; do
  SAMPLE_DIR="${OUTPUT_ROOT}/seed_${seed}" SEED="${seed}" SAVE_LATENTS=1 \
    bash scripts/sample_droid.sh
done
python scripts/evaluate_droid_uncertainty.py \
  --data_root "${DATA_ROOT}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[0]}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[1]}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[2]}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[3]}" \
  --output_dir "${OUTPUT_ROOT}/metrics"
```

Add `SEED="${SEED:-0}"`, `ACTION_VARIANT="${ACTION_VARIANT:-real}"`, and
`SAVE_LATENTS="${SAVE_LATENTS:-0}"` near the other launcher defaults. Forward
`--seed "${SEED}"`, `--action_variant "${ACTION_VARIANT}"`, and conditionally
append `--save_latents` through a Bash array when `SAVE_LATENTS=1`. Continue to
use the existing `TOTAL_LEN` and `SAMPLE_NUM_VIDEOS` environment overrides, so
formal runs set them to `6` and `16`, while the GPU gate sets them to `6` and
`1`.

- [ ] **Step 3: Run the complete CPU regression suite**

Run in the established PyTorch container:

```bash
PYTHONPATH=. pytest -q -m "not cuda and not flash_attn"
```

Expected: all tests pass, with only CUDA/FlashAttention tests deselected.

- [ ] **Step 4: Run the one-episode/two-seed GPU integration gate**

Use official checkpoint, validation episode 1064 only, `K=2`, FP16, SDPA, and a new output directory under `/data/miniworld/experiments`. Verify:

```text
2/2 sampling processes exit 0
2 MP4 files contain exactly 21 RGB frames
2 latent files are finite and shaped (1,48,6,15,20)
manifest checkpoint/config fields match except seed
evaluator produces exactly 5 episode-step rows in explicitly incomplete mode
```

Do not start the formal run if any gate condition fails.

- [ ] **Step 5: Record the GPU gate and commit Task 5**

Append the question, reason, exact configuration, result paths, observed shapes, failures, decision, and next step to `docs/experiment_journal_zh.md`.

```bash
git add scripts/evaluate_droid_uncertainty.sh scripts/sample_droid.sh tests/test_launchers.py
git add -f docs/experiment_journal_zh.md
git commit -m "exp: validate uncertainty sampling gate"
```

---

### Task 6: Formal K=4 validation run and Phase 5 decision

**Files:**
- Create: `docs/results/v100-official-055b-uncertainty-correlation.md`
- Modify: `docs/experiment_journal_zh.md`

**Interfaces:**
- Consumes the verified launcher and frozen assets.
- Produces formal 16-episode × 4-seed samples, 80 episode-step rows, correlation tables, a results archive without checkpoints, and a documented gate decision.

- [ ] **Step 1: Preflight immutable identities and storage**

Record checkpoint SHA256, validation manifest SHA256, Git commit, four seeds, free disk, and the exact command before sampling. Confirm sample roots do not already contain partial formal results.

- [ ] **Step 2: Run the four-seed validation sampler**

Run on V100 with FP16/SDPA and monitor exit codes, GPU memory, wall time, and per-seed completion. Do not substitute continued-training weights or access the test split.

- [ ] **Step 3: Run the evaluator and verify artifacts**

Assert 16 unique episode IDs, five rows per episode, four distinct seeds, 80 total rows, finite metric columns, and exact agreement between JSON summary values and recomputation from CSV.

- [ ] **Step 4: Apply the pre-registered gate without changing thresholds**

For latent primary and RGB control, report pooled Pearson/Spearman, horizon-conditioned Spearman, per-horizon values, per-episode exploratory values, quartile errors, and pass/fail for the `0.30 / 0.20 / top>bottom` gate. Inspect and record high-uncertainty/low-error and low-uncertainty/high-error counterexamples.

- [ ] **Step 5: Document, package, verify, and commit**

Write the independent report and journal entry. Package logs, manifests, CSV/JSON, report, and representative videos without checkpoint files. Record archive SHA256. Run:

```bash
PYTHONPATH=. pytest -q -m "not cuda and not flash_attn"
git diff --check
git status --short
```

If the gate passes, the next task is a new approved design for fixed-vs-threshold-vs-smoothed adaptive rollout. If it fails, design only the bounded estimator/horizon diagnostic specified in the design document.

```bash
git add miniworld/uncertainty.py miniworld/sample.py scripts/evaluate_droid_uncertainty.py scripts/evaluate_droid_uncertainty.sh scripts/sample_droid.sh tests/test_uncertainty.py tests/test_uncertainty_evaluation.py tests/test_cli.py tests/test_launchers.py
git add -f docs/experiment_journal_zh.md docs/results/v100-official-055b-uncertainty-correlation.md
git commit -m "exp: evaluate uncertainty error correlation"
```
