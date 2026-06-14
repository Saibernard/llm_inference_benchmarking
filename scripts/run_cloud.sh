#!/usr/bin/env bash
# All-in-one cloud GPU run. On an Ubuntu GPU box (Brev, AWS, etc.) it:
#   1. makes sure Docker + the NVIDIA container runtime are installed,
#   2. starts the Dockerized vLLM server,
#   3. runs the request-rate sweep AND the concurrency/prompt/output sweep,
#   4. runs the cross-check against vLLM's official benchmark,
#   5. if GH_TOKEN is set, publishes the curated results to GitHub.
#
# Needs a .env in the repo root with HF_TOKEN (required) and GH_TOKEN (optional, to publish):
#   bash scripts/run_cloud.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- load .env ----
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${HF_TOKEN:?set HF_TOKEN in .env (your Hugging Face token)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ---- pick docker / sudo docker ----
SUDO=""
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then SUDO="sudo"; fi
fi
DOCKER="${SUDO} docker"
DC="${SUDO} docker compose"

# ---- ensure Docker ----
if ! command -v docker >/dev/null 2>&1; then
  echo "=== installing Docker ==="
  curl -fsSL https://get.docker.com | ${SUDO} sh
fi

# ---- ensure the NVIDIA container runtime ----
if ! ${DOCKER} run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "=== installing NVIDIA Container Toolkit ==="
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | ${SUDO} gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  ${SUDO} apt-get update -qq && ${SUDO} apt-get install -y -qq nvidia-container-toolkit
  ${SUDO} nvidia-ctk runtime configure --runtime=docker
  ${SUDO} systemctl restart docker 2>/dev/null || ${SUDO} service docker restart 2>/dev/null || true
fi
if ! ${DOCKER} run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "Docker still cannot see the GPU. Output below — paste it and stop:"
  ${DOCKER} run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  exit 1
fi
echo "Docker can see the GPU."

# ---- start vLLM ----
echo "=== building harness + starting vLLM ==="
${DC} build harness
${DC} up -d vllm
echo "=== waiting for vLLM to be healthy (first launch downloads ~16GB) ==="
ready=""
for _ in $(seq 1 180); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then ready=1; echo "vLLM ready"; break; fi
  sleep 5
done
[ -n "$ready" ] || { echo "vLLM not healthy. Logs:"; ${DC} logs --tail 40 vllm; exit 1; }

# ---- the two sweeps ----
echo "=== sweep 1/2: open-loop request-rate sweep ==="
${DC} run --rm harness run --config configs/aws.yaml
echo "=== sweep 2/2: closed-loop concurrency x prompt-length x output-length ==="
${DC} run --rm harness run --config configs/aws_dimensions.yaml

# ---- cross-check (captured to a file) ----
echo "=== cross-check vs vLLM's official benchmark (rate 16) ==="
${DC} exec -T vllm vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --model llama3.1-8b --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000 --dataset-name random \
  --random-input-len 512 --random-output-len 128 --random-range-ratio 0.0 \
  --ignore-eos --num-prompts 200 --request-rate 16 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  2>&1 | tee results/oracle_crosscheck.txt || echo "(cross-check failed, non-fatal)"

echo "=== tearing down vLLM ==="
${DC} down

# ---- publish curated results to GitHub ----
if [ -n "${GH_TOKEN:-}" ]; then
  echo "=== publishing results to GitHub ==="
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  DEST="results_published/cloud_${STAMP}"
  mkdir -p "$DEST"
  for run in results/*/; do
    [ -f "${run}summary.csv" ] || continue
    name=$(basename "$run")
    mkdir -p "$DEST/$name/plots"
    cp -f "${run}summary.csv" "${run}summary.json" "${run}run_manifest.json" "$DEST/$name/" 2>/dev/null || true
    cp -f "${run}plots/"*.png "$DEST/$name/plots/" 2>/dev/null || true
  done
  cp -f results/oracle_crosscheck.txt "$DEST/" 2>/dev/null || true
  git config user.name "Saibernard Yogendran"
  git config user.email "bernie97@seas.upenn.edu"
  git remote set-url origin "https://${GH_TOKEN}@github.com/Saibernard/llm_inference_benchmarking.git"
  git fetch -q origin && git rebase -q origin/master 2>/dev/null || { git rebase --abort 2>/dev/null || true; }
  git add results_published/
  git commit -q -m "results: cloud A100 run ${STAMP} (Dockerized, all sweep dims + cross-check)" || echo "(nothing new to commit)"
  if git push origin master; then
    echo "Results pushed to GitHub: ${DEST}"
  else
    echo "Push failed — check the GH_TOKEN has repo write scope. Results are saved in ./results."
  fi
else
  echo "GH_TOKEN not set, skipping GitHub publish. Results are in ./results."
fi
echo "DONE."
