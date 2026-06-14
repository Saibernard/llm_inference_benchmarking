#!/usr/bin/env bash
# One command to run the whole AWS validation: build the harness, start the
# Dockerized vLLM server, run the request-rate sweep, then the
# concurrency/prompt-length/output-length sweep, then the cross-check against
# vLLM's official benchmark, and tear down. Results land in ./results.
#
# On the GPU box, with a .env file containing HF_TOKEN:
#   bash scripts/run_aws.sh        (or: sudo bash scripts/run_aws.sh)
set -uo pipefail

echo "=== building the harness image ==="
docker compose build harness

echo "=== starting the vLLM server (detached) ==="
docker compose up -d vllm

echo "=== waiting for vLLM to be healthy (first launch downloads ~16GB, give it several minutes) ==="
ready=""
for _ in $(seq 1 180); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then ready=1; echo "vLLM ready"; break; fi
  sleep 5
done
if [ -z "${ready}" ]; then
  echo "vLLM did not become healthy. Logs:"; docker compose logs --tail 40 vllm; exit 1
fi

echo "=== sweep 1/2: open-loop request-rate sweep (the headline knee) ==="
docker compose run --rm harness run --config configs/aws.yaml

echo "=== sweep 2/2: closed-loop concurrency x prompt-length x output-length ==="
docker compose run --rm harness run --config configs/aws_dimensions.yaml

echo "=== cross-check vs vLLM's official benchmark (rate 16) ==="
docker compose exec -T vllm vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --model llama3.1-8b --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000 --dataset-name random \
  --random-input-len 512 --random-output-len 128 --random-range-ratio 0.0 \
  --ignore-eos --num-prompts 200 --request-rate 16 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  || echo "(cross-check step failed, non-fatal — your sweep results are still saved)"

echo "=== tearing down vLLM ==="
docker compose down

echo "Done. Results are in ./results — scp them back to your Mac."
