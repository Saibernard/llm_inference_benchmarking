#!/usr/bin/env bash
# Reference ORACLE: run vLLM's official `vllm bench serve` against the SAME server
# our custom load-gen hits, with matched params, so we can prove our numbers match.
# --random-range-ratio 0.0 => EXACT fixed input/output lengths.
#
#   ./scripts/run_bench_serve_oracle.sh   (run inside/next to the vLLM env)
#
# Env overrides: BASE_URL, MODEL, INPUT_LEN, OUTPUT_LEN, NUM_PROMPTS, REQUEST_RATE,
#                MAX_CONCURRENCY, RESULT_DIR
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-llama3.1-8b}"                                  # name sent in API requests (server's --served-model-name)
TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}"    # real HF repo id: bench loads the tokenizer LOCALLY
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-16}"
RESULT_DIR="${RESULT_DIR:-results/oracle}"
mkdir -p "${RESULT_DIR}"

ARGS=(
  bench serve
  --backend openai
  --endpoint /v1/completions
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --base-url "${BASE_URL}"
  --dataset-name random
  --random-input-len "${INPUT_LEN}"
  --random-output-len "${OUTPUT_LEN}"
  --random-range-ratio 0.0
  --ignore-eos
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --seed 0
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,95,99
  --save-result --result-dir "${RESULT_DIR}"
)
[ -n "${MAX_CONCURRENCY:-}" ] && ARGS+=(--max-concurrency "${MAX_CONCURRENCY}")

echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
