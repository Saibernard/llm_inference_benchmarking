# Cloud A100 run (single A100 40 GB SXM4, 400 W limit)

The full Dockerized stack run end to end on **NVIDIA Brev**, with real `nvidia-smi`
telemetry. Llama-3.1-8B (bf16), vLLM 0.11, prefix caching off, `ignore_eos` on, so every
request pays a genuine prefill and emits exactly `output_len` tokens.

> **Provenance, stated precisely.** The `run_manifest.json` files say `"platform": "aws"`.
> That is the **config filename** (`configs/aws.yaml`), not the host — these runs were on
> NVIDIA Brev. The manifest does not record GPU name, driver or CUDA version (a gap worth
> fixing; `scripts/env_manifest.sh` collects them but is not wired into the run). The card
> is nonetheless pinned by the telemetry itself:
>
> | `telemetry.csv` field | value | implies |
> |---|---|---|
> | `mem_total_bytes` | 41,875,931,136 (39.0 GiB) | A100 **40 GB** |
> | `power_limit_w` | 400 | **SXM4** (the PCIe 40 GB part is 250 W) |
> | `sm_clock_mhz` (max) | 1410 | A100 boost clock |
> | `mem_clock_mhz` (max) | 1215 | HBM2e ⇒ 1215 × 2 × 5120 bit ÷ 8 = **1,555 GB/s** |
>
> Those last two are what the roofline analysis is built on (1,555 GB/s, 312 bf16 TFLOP/s),
> so the hardware constants are verifiable from the shipped data, not assumed.

Three sweeps:

- **`2026-06-15T01-14-49Z_372763/`** — open-loop request-rate sweep: 2 → 32 req/s,
  512-token prompts, 128-token outputs, 200 requests per cell.
- **`2026-06-14T23-55-58Z_3f697b/`** — an **independent repeat of the identical open-loop
  sweep**, ~75 minutes earlier on the same box. This is the reproducibility evidence: the two
  runs agree on output throughput to within **0.8% at every load level** and **0.2% at the
  knee** (1,649 vs 1,647 output tok/s), with GPU util, power and KV occupancy matching
  within a point or two.
- **`2026-06-15T01-19-20Z_393d65/`** — closed-loop sweep over concurrency (4 / 16 / 64),
  prompt length (512 / 1024) and output length (128 / 256), 60 requests per cell.

Each run dir has `summary.csv` (aggregated metrics, latencies in ms), `summary.json`,
`run_manifest.json` (versions, resolved config, telemetry backend), `plots/`, and under
`configs/<cell>/` the raw `requests.jsonl` (per-request timings, including every token's
emission timestamp) and `telemetry.csv` (per-sample GPU util / power / HBM / memory-bus
activity / KV occupancy / vLLM's running+waiting queue depths) — so any metric can be
recomputed from the raw data.

Every `run_manifest.json` records `telemetry_backend: nvidia-smi` and
`telemetry_synthetic: false`: the GPU numbers are measured, not simulated.

## What the numbers say

Raw ceiling ~16.6 req/s (~2,125 output tok/s). Useful capacity under an SLO of
TTFT ≤ 1s **and** TPOT ≤ 50ms peaks around **7.2 req/s** — less than half — and collapses
to ~0.5% SLO attainment by 32 req/s.

**The bottleneck is prefill compute, not decode bandwidth.** This corrects the project's
original claim. Run `scripts/analyze_bottleneck.py` against either open-loop run:

- **Decode *cannot* be the compute wall — VRAM proves it.** KV traffic scales with the batch
  and refuses to be amortized, so decode's arithmetic intensity stops climbing. And KV must
  *fit*: ~16 GB of the 40 GB card is stored weights, so even handing every remaining byte
  (~24 GB) to KV allows at most **~317 sequences** at a 576-token context (75 MB each). Feed
  that ceiling back in and decode's intensity tops out at **~125 FLOP/byte**, against an A100
  ridge of **~201**. At the observed batch (~145) it is **85**.
  *(The batch→∞ asymptote is ~203, marginally* above *the ridge — so the asymptote alone
  settles nothing. VRAM is what binds.)*
- **Power: the original argument was backwards.** The "~350 W, below the 400 W limit" figure
  is the whole-window mean, diluted by idle ramp and drain (it is what `summary.csv`
  reports). Over rows with GPU util ≥90%, rate 32 averages **384.6 W** of the 400 W cap
  (repeat run: 376.3 W). Memory-bound kernels draw *less* power; dense GEMM pins a power
  limit. High power does not identify the limiting unit on its own — but it is not the
  signature of a GPU idling on memory.
- **Prefill owns the clock.** Forward passes carrying a prompt are ~30% of passes but occupy
  **~65% of the GPU's clock**, at ~62% of the A100's bf16 peak. Prompt-bearing intervals
  average **4.3×** the duration of pure-decode intervals *and contain fewer decode
  emissions*, so a bigger decode batch does not explain the gap. With 512-in/128-out there
  are four prompt tokens for every generated token.

Decode *is* bandwidth-bound in isolation (at rate 2, TPOT of 15 ms implies ~1.07 TB/s of
weight streaming, ~69% of spec). It just isn't what runs out under load.

**What this does NOT claim.** FLOP/s and GB/s are *modeled* from token counts, not measured:
`nvidia-smi` has no DRAM-throughput counter (its memory-utilization figure is a duty cycle),
and the traffic model omits activations and KV writes, so it is a lower bound — and a lower
bound below peak cannot rule out saturation within decode. Neither roof is actually *hit*
(prefill ~62% of compute peak, decode ~55% of bandwidth peak). What is established is that
prompt-bearing passes own two-thirds of the clock, which contradicts the decode-only story.
Apportioning the remainder needs phase-separated DCGM/Nsight counters or a
prefill-only vs decode-only ablation.

## Two things to know before quoting these numbers

1. **TTFT p99 at rate 8 in run `372763` reads 1,529 ms — a streaming-path disruption of
   UNRESOLVED cause.** Seven consecutive requests (idx 21–27) took 1.3–2.0 s to first token,
   and the 21 requests already streaming at that moment all show SSE chunk coalescing
   (`tokens_chunks_mismatch: true`) in a staircase ordered by start time — 122, 121, 76, 75
   … 10, 4 chunks instead of 128. Something disrupted the streaming path for everything in
   flight at once.

   **I cannot attribute it, and two earlier write-ups of this were overconfident** (first
   "server stall", then "client event-loop stall"). `schedule_delay` (~1 ms) proves the
   request coroutines fired on time, but it is stamped *before* HTTP connection acquisition,
   so it does not prove the bytes left on time. And chunk coalescing is something the
   *server* does when its output queue backs up — which a slow reader, transport
   backpressure, or a stalled server output loop all produce identically.

   Related instrument defect found while checking: the telemetry sampler (configured at
   200 ms) has a **median gap of 3.35 s at rate 4** and 8.5 s worst case, and is *healthiest*
   at the highest load (median 0.26 s at rate 32) — most likely `nvidia-smi` responding
   slowly on an idling GPU. That is the opposite of what a load-induced client stall would
   predict, and it is why the low-rate telemetry rows are thin.

   The 7 requests are treated as **contaminated measurements**. They are the entire reason
   rate-8 SLO attainment is 96.5% rather than 100% (`7.4204 × 193/200 = 7.1607`); the repeat
   run `3f697b` has no disruption and measures that cell at **100% attainment, goodput 7.42
   req/s**. Quote useful capacity as **~7.2–7.4 req/s**, never a false-precision 7.16. The
   data is left in place, unfixed, on purpose.

2. **`ttft_p99` / `tpot_p99` / `e2e_p99` are blank in the closed-loop `summary.csv`.**
   That is deliberate, not a bug: 60 requests per cell is below the minimum-sample-size
   guard for a P99, so the code returns NaN rather than a fabricated number.
