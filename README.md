# llm_inference_benchmarking — `gpubench`

A **single-GPU LLM inference benchmarking harness** for [vLLM](https://docs.vllm.ai).
It drives a vLLM OpenAI-compatible server with a *coordinated-omission-correct*
load generator, correlates serving latency with GPU telemetry, finds the
**latency–throughput knee**, and cross-checks every number against vLLM's own
`vllm bench serve` as a reference oracle.

It is built as a **measurement instrument**: the design assumes that *wrong
numbers are worse than no numbers*, so the subtle, credibility-defining details
(coordinated omission, the exact TPOT formula, the current vLLM `/metrics` names,
the GQA KV-cache math) are gotten right and pinned by tests.

> New to LLM serving? Read [`TEACHING.md`](TEACHING.md) first — it explains the
> whole thing from zero using one running analogy (a restaurant kitchen).

![GPU saturation example](docs/sample_gpu_saturation.png)

---

## What it measures

Per sweep cell it reports, over a steady-state window:

- **TTFT** — time to first token (prefill + queueing)
- **TPOT / ITL** — time per output token / inter-token latency (decode)
- **E2E latency** — full request latency, with **P50 / P95 / P99**
- **Throughput** — output tok/s, total tok/s, requests/s (window-based)
- **Goodput** — throughput of requests meeting an SLO (the number that matters)
- **GPU telemetry** — utilization, HBM used, power draw, KV-cache occupancy
- **Failures** — by class (timeout, HTTP error, truncated stream, …)

It sweeps **request rate**, **concurrency**, **prompt length**, and **output
length**, and finds the **saturation knee** where throughput plateaus while P99
latency climbs.

---

## Why it's credible (the engineering, not the buzzwords)

- **Coordinated-omission correct.** The open-loop generator pre-schedules
  *absolute* arrival times (a Poisson process) and fires without waiting for prior
  responses, recording *intended* vs *actual* send time. A slow server can never
  throttle the offered load and hide tail latency — the classic home-grown
  benchmark bug.
- **Cross-checked against an oracle.** The same server is hit by vLLM's official
  `vllm bench serve` with matched params; our numbers must match it *and* the
  server's own `/metrics` histograms. Three independent measurements agreeing is
  the validation gate.
- **Statistically honest.** Window-based throughput (never sum-of-per-request
  rates); `TPOT = (E2E − TTFT)/(output_tokens − 1)`; percentiles via
  `numpy.percentile` with a minimum-sample-size guard (no fabricated P99s);
  failures excluded from latency but tracked separately; goodput is a strict SLO
  conjunction.
- **Reproducible.** Pinned vLLM version, seeded RNG, and an environment manifest
  (GPU, driver, CUDA, model) written into every run. Raw per-request data is kept
  as JSONL so any metric can be recomputed.
- **Telemetry done right.** vLLM's V1 `/metrics` names (`vllm:kv_cache_usage_perc`,
  `vllm:inter_token_latency_seconds`) with legacy fallback; monotonic-clock
  alignment of GPU samples to load windows; honest `synthetic` flag when no GPU.

The design and these decisions were produced and **adversarially reviewed by a
multi-agent workflow** against current vLLM / NVIDIA / AWS docs before
implementation.

---

## Architecture

```
┌──────────────────────── orchestrator ────────────────────────┐
│  sweep matrix · monotonic fences · writes JSONL/CSV/manifest  │
└───┬───────────────────────┬───────────────────────┬──────────┘
    │                       │                       │
┌───▼────┐   HTTP/SSE   ┌───▼─────────┐       ┌─────▼──────────┐
│ vLLM   │◄─────────────│  loadgen    │       │  telemetry      │
│ server │  /v1/        │ open+closed │       │ NVML/nvidia-smi │
│  (or   │  completions │ loop, CO-   │       │ + vLLM /metrics │
│  mock) │  /tokenize   │ correct     │       │ (monotonic)     │
│/metrics│              └─────────────┘       └────────────────┘
└────────┘                     │                       │
    └──────────────────────────▼───────────────────────┘
                       ┌────────────────┐
                       │  metrics (the  │  ← single aggregation path
                       │     ruler)     │
                       └───────┬────────┘
                       ┌───────▼────────┐
                       │   reporter     │ → summary.csv/json + 4 plots
                       └────────────────┘
```

---

## Quickstart

### macOS / any laptop — no GPU (mock server)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

gpubench serve-mock --port 8137 &            # GPU-free vLLM-shaped server
gpubench run --config configs/smoke.yaml --base-url http://127.0.0.1:8137
# -> results/<run_id>/summary.csv + plots/
```

The mock streams fake tokens with configurable TTFT/ITL and a saturation curve,
so the *entire* measurement + plotting pipeline is exercised offline. Or just:
`./scripts/run_mock.sh configs/mock.yaml`.

### Google Colab Pro — real GPU, first true numbers

Open [`notebooks/colab_validation.ipynb`](notebooks/colab_validation.ipynb):
set a `HF_TOKEN` secret (with the Llama-3.1 license accepted), pick a GPU runtime,
Run All. It installs vLLM natively (no Docker), runs the open-loop sweep, and
cross-checks against `vllm bench serve`.

### AWS (or any single-GPU Linux box) — full Dockerized run

```bash
export HF_TOKEN=hf_...          # account must have accepted the Llama-3.1 license
docker compose up --build       # vLLM server + harness; results land in ./results
```

Pick an instance with `gpubench plan` (Llama-3.1-8B memory math):

```bash
gpubench plan                   # table for 16/24/40/48/80 GB
gpubench plan --gpu-mem 24 --ctx 4096
```

---

## Interpreting the output

`results/<run_id>/` contains:

- `summary.csv` / `summary.json` — one row per sweep cell (all latencies in **ms**)
- `configs/<cell>/requests.jsonl` — raw per-request truth (recompute anything)
- `configs/<cell>/telemetry.csv` — time-aligned GPU + vLLM `/metrics`
- `run_manifest.json` — versions, seeds, GPU, full resolved config
- `plots/` — four charts:

| Plot | What it proves |
|---|---|
| `pareto_knee.png` | Output tok/s vs P99 latency with the **knee** marked — your max sustainable load. |
| `ttft_tpot_vs_load.png` | Splits latency into **TTFT (queue/prefill)** vs **TPOT (decode)** so a regression points at the right fix. |
| `gpu_saturation.png` | Util / KV-cache / power vs load + a time series — *why* it saturated (often KV-cache hitting ~100% before compute). |
| `goodput_vs_load.png` | Raw throughput vs **goodput**; the shaded gap is work that breached the SLO and is useless. |

---

## Repo layout

```
src/gpubench/
  schema.py        canonical RequestRecord + vLLM metric-name constants + summary columns
  config.py        typed configs, pinned vLLM version, Llama-3.1-8B memory math, YAML loader
  metrics.py       the single "ruler": finalize + aggregate (TTFT/TPOT/ITL/throughput/goodput)
  loadgen.py       coordinated-omission-correct async open/closed-loop generator + SSE parse
  telemetry.py     GPU backends (NVML/nvidia-smi/synthetic) + vLLM /metrics scraper + knee signals
  serving.py       vLLM launch + `vllm bench serve` oracle command builders
  mock_server.py   GPU-free vLLM-shaped server (the macOS dev + test fixture)
  orchestrator.py  the hub: drives the sweep, writes all on-disk artifacts
  reporter.py      summary (seconds -> ms here) + knee detection + the four plots
  cli.py           gpubench serve-mock | run | report | plan | crosscheck
configs/           mock · smoke · colab · aws (one YAML per platform, sectioned)
scripts/           serve_vllm.sh · run_bench_serve_oracle.sh · env_manifest.sh · run_mock.sh
notebooks/         colab_validation.ipynb
tests/             metric math, SSE parse, Poisson arrivals, GQA memory, telemetry, serve flags
```

Run the tests: `pip install -e . pytest pytest-asyncio && pytest -q`.

---

## License

MIT. Model weights are **not** included; Llama-3.1-8B is gated — accept Meta's
license on Hugging Face and supply your own `HF_TOKEN`.
