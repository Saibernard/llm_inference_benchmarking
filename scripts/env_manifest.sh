#!/usr/bin/env bash
# Capture the environment that makes a run reproducible. Pipe into your results
# dir:  ./scripts/env_manifest.sh > results/<run_id>/env.txt
set -uo pipefail

echo "# gpubench environment manifest"
echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname: $(hostname)"
echo "python: $(python --version 2>&1)"
echo "platform: $(uname -srm)"

echo "## GPU (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
  nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/gpu /'
else
  echo "nvidia-smi: NOT FOUND (no NVIDIA GPU on this host)"
fi

echo "## CUDA / vLLM"
python - <<'PY' 2>/dev/null || echo "torch/vllm not importable here"
try:
    import torch; print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as e: print("torch: n/a", e)
try:
    import vllm; print("vllm:", vllm.__version__)
except Exception as e: print("vllm: n/a", e)
PY
