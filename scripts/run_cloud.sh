#!/usr/bin/env bash
# All-in-one cloud GPU run, hardened so it does NOT waste GPU money. On an Ubuntu GPU box it:
#   1. installs Docker + the NVIDIA container runtime if missing,
#   2. FAILS FAST (before the 16GB download) if the GPU or HF token is wrong,
#   3. starts the Dockerized vLLM server (bounded wait, no infinite hang),
#   4. confirms the harness container actually sees the GPU (so telemetry is real, not synthetic),
#   5. runs the request-rate sweep AND the concurrency/prompt/output sweep (each hard-capped),
#   6. runs the cross-check against vLLM's official benchmark,
#   7. if GH_TOKEN is set, publishes curated results to GitHub (never hangs on a credential prompt),
#   8. ALWAYS tears down the server and prints a loud "terminate the instance" reminder, on every exit.
#
# Needs a .env in the repo root with HF_TOKEN (required) and GH_TOKEN (optional, to publish):
#   bash scripts/run_cloud.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- load .env, strip stray CR (Windows line endings), require HF_TOKEN ----
[ -f .env ] && { set -a; . ./.env; set +a; }
HF_TOKEN="${HF_TOKEN:-}"; HF_TOKEN="${HF_TOKEN%$'\r'}"
GH_TOKEN="${GH_TOKEN:-}"; GH_TOKEN="${GH_TOKEN%$'\r'}"
: "${HF_TOKEN:?set HF_TOKEN in .env (your Hugging Face token)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
mkdir -p results   # bind-mount target + cross-check log dir must exist on the host

# ---- docker / sudo detection (re-runnable: call again after installing docker) ----
detect_docker() {
  SUDO=""
  if ! docker info >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then SUDO="sudo"; fi
  fi
  DOCKER="${SUDO} docker"; DC="${SUDO} docker compose"
}
detect_docker

# ---- timeout helpers (so nothing hangs forever and burns money) ----
# dto: docker-with-timeout. sudo is OUTSIDE timeout so the kill signal reaches the container, not sudo.
dto() { local t="$1"; shift; if command -v timeout >/dev/null 2>&1; then ${SUDO} timeout -k 30s "$t" "$@"; else ${SUDO} "$@"; fi; }
# to: plain timeout (no sudo) for git etc.
to()  { local t="$1"; shift; if command -v timeout >/dev/null 2>&1; then timeout -k 15s "$t" "$@"; else "$@"; fi; }

# ---- safety net: on ANY exit, tear the server down + show cost + remind to terminate ----
START=$(date +%s); RATE_PER_HR="${RATE_PER_HR:-1.63}"
cleanup() {
  ${DC} down >/dev/null 2>&1 || true
  local min cost; min=$(( ($(date +%s) - START) / 60 ))
  cost=$(awk "BEGIN{printf \"%.2f\", ${RATE_PER_HR}*${min}/60}")
  echo ""
  echo "================================================================"
  echo "  wall-time: ${min} min   (~\$${cost} of GPU time at \$${RATE_PER_HR}/hr)"
  echo "  >>> NOW TERMINATE THIS INSTANCE IN THE BREV CONSOLE <<<"
  echo "  The box keeps billing until you terminate it."
  echo "================================================================"
}
trap cleanup EXIT

# ---- ensure Docker, then re-detect sudo and bail clearly if still unusable ----
if ! command -v docker >/dev/null 2>&1; then
  echo "=== installing Docker ==="; curl -fsSL https://get.docker.com | ${SUDO} sh
fi
detect_docker
if ! ${DOCKER} info >/dev/null 2>&1; then
  echo "Docker isn't usable. Likely a permissions issue — run:  sudo usermod -aG docker \$USER  (then re-login)"
  echo "or use a root / passwordless-sudo box. Aborting."; exit 1
fi

# ---- ensure the NVIDIA container runtime ----
if ! dto 5m docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "=== installing NVIDIA Container Toolkit ==="
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | ${SUDO} gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  ${SUDO} apt-get update -qq && ${SUDO} apt-get install -y -qq nvidia-container-toolkit
  ${SUDO} nvidia-ctk runtime configure --runtime=docker
  ${SUDO} systemctl restart docker 2>/dev/null || ${SUDO} service docker restart 2>/dev/null || true
fi

# ---- FAIL FAST #1: GPU must be visible to Docker (before any download) ----
if ! dto 5m docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "Docker cannot see the GPU. Output (paste this and stop):"
  ${DOCKER} run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi; exit 1
fi
echo "GPU visible to Docker."

# ---- FAIL FAST #2: HF token can reach the gated model (before the 16GB download) ----
echo "=== preflight: HF token access to the model ==="
hf_code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${HF_TOKEN}" \
  https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct 2>/dev/null || echo "000")
if [ "$hf_code" != "200" ]; then
  echo "HF token cannot access meta-llama/Llama-3.1-8B-Instruct (HTTP ${hf_code})."
  echo "Check the token and that this account accepted the Llama-3.1 license. Aborting before any download."; exit 1
fi
echo "HF token OK."

# ---- build harness + start vLLM (clean leftovers; bounded waits) ----
${DC} down >/dev/null 2>&1 || true
echo "=== building harness ==="; dto 20m docker compose build harness
echo "=== starting vLLM ===";    dto 25m docker compose up -d vllm
echo "=== waiting for vLLM to be healthy (first launch downloads ~16GB; capped at 15 min) ==="
ready=""
for _ in $(seq 1 180); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then ready=1; echo "vLLM ready"; break; fi
  sleep 5
done
[ -n "$ready" ] || { echo "vLLM not healthy within 15 min. Logs:"; ${DC} logs --tail 40 vllm; exit 1; }

# ---- resolve vLLM's network + the harness image, so we can run the harness via PLAIN
#      `docker run --gpus all` instead of `docker compose run`. On this box compose run
#      neither passes the GPU (telemetry went synthetic) nor exits cleanly (it hung for the
#      full 45m cap, which then tore vLLM down and broke the next sweep). Plain `docker run
#      --gpus all` is exactly what the fail-fast GPU check already proved works here, and
#      --rm with no TTY exits the instant the sweep ends; vLLM is left untouched throughout. ----
PROJECT=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]//g')
VLLM_CID=$(${DC} ps -q vllm 2>/dev/null | head -1)
NET=$(${DOCKER} inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' "$VLLM_CID" 2>/dev/null | head -1)
NET="${NET:-${PROJECT}_default}"; IMG="${PROJECT}-harness"
echo "harness image: ${IMG}   vLLM network: ${NET}"

# ---- confirm the harness image sees the GPU via docker run --gpus all (real telemetry) ----
echo "=== preflight: harness GPU access (docker run --gpus all) ==="
if dto 2m docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --entrypoint nvidia-smi "$IMG" -L >/dev/null 2>&1; then
  echo "harness sees the GPU — telemetry will be REAL."
else
  echo "WARNING: harness can't access the GPU even via docker run --gpus all — telemetry will be SYNTHETIC."
  echo "(Not fatal: vLLM still has the GPU and the serving metrics are real.)"
fi

# ---- the two sweeps (hard-capped; plain docker run can't hang on a TTY or recreate vLLM) ----
echo "=== sweep 1/2: open-loop request-rate sweep ==="
dto 45m docker run --rm --gpus all --network "$NET" -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD/results:/app/results" -v "$PWD/configs:/app/configs" "$IMG" \
  run --config configs/aws.yaml --base-url http://vllm:8000 \
  || echo "(rate sweep timed out/errored — continuing)"
echo "=== sweep 2/2: closed-loop concurrency x prompt-length x output-length ==="
dto 45m docker run --rm --gpus all --network "$NET" -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD/results:/app/results" -v "$PWD/configs:/app/configs" "$IMG" \
  run --config configs/aws_dimensions.yaml --base-url http://vllm:8000 \
  || echo "(dimensions sweep timed out/errored — continuing)"

# ---- cross-check (captured to a file) ----
echo "=== cross-check vs vLLM's official benchmark (rate 16) ==="
dto 15m docker compose exec -T vllm vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --model llama3.1-8b --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000 --dataset-name random \
  --random-input-len 512 --random-output-len 128 --random-range-ratio 0.0 \
  --ignore-eos --num-prompts 200 --request-rate 16 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  2>&1 | tee results/oracle_crosscheck.txt || echo "(cross-check failed, non-fatal)"

# ---- publish curated results to GitHub (only if there ARE results; never hang on a prompt) ----
if [ -n "${GH_TOKEN}" ]; then
  echo "=== publishing results to GitHub ==="
  export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true
  STAMP=$(date -u +%Y%m%dT%H%M%SZ); DEST="results_published/cloud_${STAMP}"; mkdir -p "$DEST"; collected=0
  for run in results/*/; do
    [ -f "${run}summary.csv" ] || continue
    # copy the WHOLE run dir: summary.csv/json + run_manifest.json + plots/ + the RAW
    # per-request JSONL and per-cell telemetry.csv, so any metric can be recomputed later.
    cp -r "${run%/}" "$DEST/"
    collected=$((collected+1))
  done
  cp -f results/oracle_crosscheck.txt "$DEST/" 2>/dev/null || true
  if [ "$collected" -eq 0 ]; then
    echo "No sweep results were produced — NOT publishing an empty run. Check the logs above."
  else
    git config user.name "Saibernard Yogendran"; git config user.email "bernie97@seas.upenn.edu"
    git remote set-url origin "https://${GH_TOKEN}@github.com/Saibernard/llm_inference_benchmarking.git"
    to 2m git fetch -q origin && (git rebase -q origin/master 2>/dev/null || git rebase --abort 2>/dev/null || true)
    git add results_published/
    git commit -q -m "results: cloud A100 run ${STAMP} (${collected} sweeps + cross-check)" || echo "(nothing new to commit)"
    if to 3m git push origin master; then echo "Results pushed to GitHub: ${DEST}"
    else echo "Push failed (token scope? network?) — results are saved in ./results."; fi
  fi
else
  echo "GH_TOKEN not set — skipping GitHub publish. Results are in ./results."
fi
echo "RUN COMPLETE."
