# Official MiniWorld-0.55B two-GPU DDP/resource gate

Result date: 2026-08-27

## Mainline mapping

Phase 4 / E2-E3-E8: compare one- and two-GPU training at the same global batch,
verify rank sharding, synchronized optimization, resource scaling, checkpoint
save/resume, and telemetry before selecting a longer continued-training budget.

## Setup

Both runs use the official DROID state dict, train episodes 1000-1063, six
latent frames, FP16 GradScaler, SDPA, AdamW, LR `2e-5`, EMA `0.9999`, and seed
base `20260827`.

| Run | GPUs | Per-rank batch | Global batch | Steps |
| --- | ---: | ---: | ---: | ---: |
| single reference | 1 | 2 | 2 | 10 |
| DDP comparison | 2 | 1 | 2 | 10 |

The 64-episode `DistributedSampler` gives each rank 32 indices, with zero
overlap and a 64-index union for an epoch.

## Telemetry correctness fix

The initial DDP run trained correctly, but printed only rank 0's local loss.
That value cannot be compared with a single-rank global-batch loss. A failing
regression test was added, and training telemetry now all-reduces the scalar
loss and divides by world size at every log point. The fixed DDP run below is
the source for loss comparisons; the initial run remains useful for resource
and resume evidence.

## Results

| Metric | Single GPU, GB2 | Two GPUs, GB2 |
| --- | ---: | ---: |
| mean loss, 10 points | 0.1154 | 0.1037 |
| median loss | 0.1015 | 0.1014 |
| loss range | 0.0519-0.2001 | 0.0535-0.1650 |
| steady step/s | 0.516 | 0.860 |
| steady sample/s | 1.031 | 1.720 |
| observed memory | 18,169 MiB | 19,611 MiB per GPU |
| skipped steps | 0 | 0 |
| loss scale | 65,536 | 65,536 |

The two-GPU speedup is 1.67x with approximately 83.4% scaling efficiency.
Losses are not expected to match point-by-point because the samplers and
per-rank random streams differ, but their mean, median, and range show no
optimization or synchronization anomaly. The extra per-GPU memory relative to
single-GPU batch 2 is consistent with DDP reducer/communication state and still
leaves about 13GB on each 32GB V100.

## Resume and checkpoint

The DDP checkpoint loaded with all keys matching and resumed from step 10 to
step 15. Both ranks continued at 0.86 step/s without skipped steps. During
active computation, GPU 0 and GPU 1 each used 19,611 MiB and showed 100%/95%
utilization. The final checkpoint records:

- epoch 2, global step 15;
- 420 model and 420 EMA tensors;
- 415 optimizer states;
- GradScaler scale 65,536 and growth tracker 15;
- the expected 0.5B/FP16/SDPA/six-frame metadata.

## W&B and artifacts

- Single reference: `fnqph9f6`
- Initial DDP/resource and resume run: `kmta7zr2`
- Fixed global-loss DDP comparison: `mmweo7q6`

```text
/data/miniworld/experiments/droid-official-055b-ddp-gate-single-gb2/
/data/miniworld/experiments/droid-official-055b-ddp-gate-dual-gb2/
/data/miniworld/experiments/droid-official-055b-ddp-gate-dual-gb2-global-loss/
/data/miniworld/outputs/droid-official-055b-ddp-gate-single-gb2/
/data/miniworld/outputs/droid-official-055b-ddp-gate-dual-gb2/
/data/miniworld/outputs/droid-official-055b-ddp-gate-dual-gb2-global-loss/
```

The DDP/resource gate passes. The next mainline decision is the bounded
continued-training baseline budget and checkpoint/validation schedule; it is
not yet the uncertainty-aware innovation phase.
