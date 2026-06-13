#!/usr/bin/env bash
# Launch the pinned vLLM server with the CANONICAL benchmarking flags
# (prefix-caching OFF, logging OFF, fixed dtype/seed/util). Use on a single-GPU
# Linux box with the NVIDIA Container Toolkit.
#
#   HF_TOKEN=hf_... ./scripts/serve_vllm.sh
#
# Env overrides: MODEL, SERVED_NAME, PORT, MAX_MODEL_LEN, GPU_MEM_UTIL, IMAGE, QUANTIZATION
set -euo pipefail

# Auto-load a local .env (gitignored) if present, so HF_TOKEN "just works" here.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

IMAGE="${IMAGE:-vllm/vllm-openai:v0.11.0}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_NAME="${SERVED_NAME:-llama3.1-8b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token that has accepted the Llama-3.1 license}"

EXTRA=()
[ -n "${QUANTIZATION:-}" ] && EXTRA+=(--quantization "${QUANTIZATION}")

exec docker run --rm -it \
  --runtime nvidia --gpus all --ipc=host \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN}" --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  -p "${PORT}:${PORT}" \
  "${IMAGE}" \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --seed 0 \
  --no-enable-prefix-caching \
  --no-enable-log-requests \
  --host 0.0.0.0 --port "${PORT}" \
  "${EXTRA[@]}"
