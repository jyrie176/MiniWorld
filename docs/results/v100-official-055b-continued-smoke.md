# Official MiniWorld-0.55B continued-training smoke test

Result date: 2026-08-27

## Mainline mapping

Phase 4 / E2-E3-E8: verify that the released DROID checkpoint can initialize a
V100 FP16 training run, survive optimization and checkpoint resume, and retain
its zero-shot action-control baseline before scaling to DDP or a longer budget.

## Checkpoint compatibility

The upstream checkpoint is a bare state dict. The training loader previously
looked only for wrapped `model` or `ema_model` entries and would silently load
zero pretrained tensors. A regression test reproduced the missing-key failure;
the loader now treats a bare tensor mapping as the source for both model and
EMA initialization.

For a six-latent-frame training window, initialization reported only these
expected missing buffers:

```text
net.feat_rope.freqs_cos
net.feat_rope.freqs_sin
```

They are window-sized RoPE tables rebuilt for 6 rather than the official
64-latent-frame window. All learned parameters were inherited.

## Training setup

- Data: train episodes 1000-1063; validation episodes 1064-1079
- Model: 0.5B, 553,910,284 trainable parameters at the six-frame window
- Initialization: official `MiniWorld_0_5b_droid.pt` into model and EMA
- Runtime: one V100, FP16 GradScaler, SDPA, AdamW
- Window/batch: 6 latent frames (21 RGB frames), batch size 1
- Optimization: 20 effective steps, LR `2e-5`, EMA `0.9999`, grad clip `1.0`
- Seed: `20260827`
- W&B: `9gwiazi7`

The 20 recorded losses had mean 0.1362, median 0.1048, minimum 0.0573, and
maximum 0.3060. The final loss was 0.0573. No step was skipped, the loss scale
remained 65536, and the largest pre-clipping gradient norm was 4.4625. Steady
throughput was 0.925 sample/s. Observed device memory during training was
17,557 MiB on a 32,768 MiB V100.

## Checkpoint and resume

The step-20 checkpoint is about 8.87GB and contains 420 model tensors, 420 EMA
tensors, optimizer state, epoch/global step, metadata, and GradScaler state.
Resume loaded all keys, restored step 20 and scaler growth tracker 20, completed
step 21 without a skip, and saved tracker 21. This closes the single-GPU
checkpoint-resume gate.

## Validation before and after 20 steps

Both evaluations use EMA, validation episodes 1064-1079, identical per-sample
seeds, 20 sampling steps, CFG 2.0, FP16, and SDPA.

| Condition | Zero-shot mean MAE | Step-20 mean MAE | Delta |
| --- | ---: | ---: | ---: |
| real | 5.468005 | 5.467836 | -0.000170 |
| zero | 9.367881 | 9.372440 | +0.004559 |
| reverse | 9.240826 | 9.243434 | +0.002608 |

Action-control wins remain unchanged: real beats zero on 15/16 episodes,
reverse on 15/16, and both on 14/16. Persistence remains 4.468. With EMA decay
0.9999, 20 updates move EMA only minimally, so this result is a no-regression
and pipeline check, not evidence of quality improvement.

## Artifacts and decision

```text
/data/miniworld/outputs/droid-official-055b-continued-smoke-lf6-20step/
/data/miniworld/experiments/droid-official-055b-continued-smoke-lf6-20step/
/data/miniworld/experiments/official-055b-continued-step20-validation-action-ablation/
```

The next mainline task is an equivalent-global-batch two-GPU DDP/resource gate.
Only after sampler, synchronization, throughput, memory, and resume behavior
pass should a longer continued-training budget be selected.
