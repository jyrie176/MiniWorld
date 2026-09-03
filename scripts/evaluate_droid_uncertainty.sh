#!/usr/bin/env bash
# Generate a four-seed validation ensemble and evaluate uncertainty correlation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to validation episodes 1064-1079}"
CKPT="${CKPT:?Set CKPT to the official MiniWorld 0.55B checkpoint}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to Wan2.2_VAE.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new experiment directory}"
GPU="${GPU:-0}"
if [[ -z "${MINIWORLD_GIT_COMMIT:-}" ]]; then
  MINIWORLD_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null)" || {
    echo "Set MINIWORLD_GIT_COMMIT when the linked worktree Git metadata is not mounted" >&2
    exit 2
  }
fi
export MINIWORLD_GIT_COMMIT

IFS=',' read -r -a SEED_VALUES <<< "${SEEDS:-20260827,20260828,20260829,20260830}"
if [[ "${#SEED_VALUES[@]}" -ne 4 ]]; then
  echo "SEEDS must contain exactly four values" >&2
  exit 2
fi

for seed in "${SEED_VALUES[@]}"; do
  MODEL="0.5B" \
  DATA_ROOT="${DATA_ROOT}" \
  CKPT="${CKPT}" \
  VAE_CKPT="${VAE_CKPT}" \
  SAMPLE_DIR="${OUTPUT_ROOT}/seed_${seed}" \
  GPU="${GPU}" \
  PRECISION="fp16" \
  ATTENTION_BACKEND="sdpa" \
  SEED="${seed}" \
  ACTION_VARIANT="real" \
  SAVE_LATENTS="1" \
  TOTAL_LEN="6" \
  NUM_SAMPLING_STEPS="20" \
  SAMPLE_NUM_VIDEOS="16" \
    bash scripts/sample_droid.sh
done

python scripts/evaluate_droid_uncertainty.py \
  --data_root "${DATA_ROOT}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[0]}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[1]}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[2]}" \
  --sample_root "${OUTPUT_ROOT}/seed_${SEED_VALUES[3]}" \
  --output_dir "${OUTPUT_ROOT}/metrics"
