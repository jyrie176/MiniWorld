#!/usr/bin/env bash
# Stream-sample a DROID MiniWorld checkpoint.
# Default model is 1B. Set MODEL=0.5B or MODEL=3B to switch model scale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL="${MODEL:-1B}" # choices: B, L, 0.5B, 1B, 3B
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the LeRobot DROID root}"
CKPT="${CKPT:?Set CKPT to a MiniWorld checkpoint}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to Wan2.2_VAE.pth}"
SAMPLE_DIR="${SAMPLE_DIR:-${REPO_DIR}/samples/droid_${MODEL}}"
GPU="${GPU:-0}"
PRECISION="${PRECISION:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
SEED="${SEED:-0}"
ACTION_VARIANT="${ACTION_VARIANT:-real}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"

SAMPLE_EXTRA_ARGS=()
if [[ "${SAVE_LATENTS}" == "1" ]]; then
  SAMPLE_EXTRA_ARGS+=(--save_latents)
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m miniworld.sample \
  --dataset droid \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${CKPT}" \
  --vae_checkpoint "${VAE_CKPT}" \
  --sample_dir "${SAMPLE_DIR}" \
  --wm_model "${MODEL}" \
  --precision "${PRECISION}" \
  --attention_backend "${ATTENTION_BACKEND}" \
  --seed "${SEED}" \
  --action_variant "${ACTION_VARIANT}" \
  --total_len "${TOTAL_LEN:-64}" \
  --df_chunk_size 2 \
  --df_ardiff_step "${DF_ARDIFF_STEP:-5}" \
  --stream_inflight_chunks "${STREAM_INFLIGHT_CHUNKS:-8}" \
  --stream_max_cache_chunks "${STREAM_MAX_CACHE_CHUNKS:-24}" \
  --stream_sink_size "${STREAM_SINK_SIZE:-1}" \
  --cfg_scale "${CFG_SCALE:-2.0}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-100}" \
  --sample_num_videos "${SAMPLE_NUM_VIDEOS:-50}" \
  "${SAMPLE_EXTRA_ARGS[@]}"
