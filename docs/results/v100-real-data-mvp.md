# V100 Real-Data Training MVP

Result date: 2026-08-26

## Scope

This MVP validates MiniWorld-B training on one Tesla V100-SXM2-32GB with real
DROID samples and the official Wan2.2 VAE. It is an execution and numerical
stability check, not a model-quality evaluation.

## Assets

- VAE: `Wan2.2_VAE.pth`, 2,818,839,170 bytes
- VAE SHA-256: `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`
- Data: DROID episodes 1000 through 1015, success-only
- Camera: `observation.images.exterior_image_1_left`
- Local mini dataset size: 11 MB (16 parquet files and 16 videos)
- Auditable episode selection: `manifests/droid-mini-1000-1015/episodes.jsonl`

## Configuration

- GPU: Tesla V100-SXM2-32GB, capability 7.0
- Container: `pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime`
- Model: MiniWorld-B, 123,852,364 trainable parameters
- VAE: Wan2.2, 704,688,668 frozen parameters
- Precision: FP16 autocast with dynamic GradScaler
- Attention: `auto` resolved to PyTorch SDPA
- Optimizer: AdamW, learning rate 2e-5
- Gradient clipping: 1.0
- Batch size: 1
- Video: 21 RGB frames at 240x320, encoded to 6 latent frames
- Action conditioning: 7 normalized DROID action dimensions, conditioning
  dimension 28
- Diffusion forcing chunk size: 2

## Results

- 100 optimizer steps completed on one V100.
- No optimizer step was skipped because of non-finite FP16 gradients.
- Dynamic loss scale remained at 65,536 through step 100.
- Logged loss was 1.936367 at step 10 and 1.490966 at step 90. The stochastic
  step-100 loss was 1.700915; these values are execution telemetry, not a
  convergence or quality claim.
- Epoch and final checkpoints containing model, EMA, optimizer, and GradScaler
  state were written successfully.
- Resume loaded the step-100 checkpoint with all model keys matching, restored
  training state, completed step 101 without a skipped update, and saved a new
  checkpoint.

## Data Decode Check

The deterministic dataset smoke check returned finite tensors with:

- video shape `(21, 240, 320, 3)` and range `[-1, 1]`
- action shape `(20, 7)` and range `[-1, 1]`
- all 16 selected episodes passing the minimum-frame filter

## Remaining Limits

This run uses a deliberately small, single-view data subset. It does not
measure held-out reconstruction or generation quality, and it does not validate
multi-GPU scaling. Those are separate experiment gates.
