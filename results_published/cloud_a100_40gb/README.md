# Cloud A100 run (single A100 40 GB SXM4)

The full Dockerized stack run end to end on a cloud A100, with real `nvidia-smi`
telemetry. Two sweeps against Llama-3.1-8B (bf16, vLLM 0.11):

- `2026-06-15T01-14-49Z_372763/` — open-loop request-rate sweep: 2 to 32 req/s,
  512-token prompts, 128-token outputs, 200 requests per cell. This is the one with
  the saturation knee (~1,650 output tok/s @ ~92% GPU util, ~360 W, KV cache ~77%).
- `2026-06-15T01-19-20Z_393d65/` — closed-loop sweep over concurrency (4 / 16 / 64),
  prompt length (512 / 1024) and output length (128 / 256), 60 requests per cell.

Each run dir has `summary.csv` (aggregated metrics), `summary.json`, `run_manifest.json`
(versions, GPU, the full config), `plots/`, and under `configs/<cell>/` the raw
`requests.jsonl` (per-request timings) and `telemetry.csv` (per-sample GPU util / power /
HBM / KV occupancy) so any metric can be recomputed from the raw data.

Every `run_manifest.json` here records `telemetry_backend: nvidia-smi` and
`telemetry_synthetic: false`, i.e. the GPU numbers are measured, not simulated.
