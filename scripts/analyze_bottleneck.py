#!/usr/bin/env python3
"""Where does the GPU's time actually go?

Reconstructs vLLM's forward passes from the per-token emission timestamps in a published
open-loop run, then splits the wall clock between passes that carried a prefill and passes
that were pure decode.

This exists because the first version of this project concluded the A100 saturated because
decode was memory-bandwidth-bound. That was wrong, and this script is the arithmetic that
overturned it. The workload is 512-in/128-out -- four prompt tokens for every generated
token -- and at saturation prefill-carrying passes occupy ~65% of the clock, while modeled
HBM traffic never exceeds ~28% of peak and the card sits at ~97% of its power cap.

WHAT IS MEASURED vs MODELED
---------------------------
Measured (from the published artifacts):
  * per-token emission timestamps  -> forward-pass structure, ITLs
  * nvidia-smi util / power / memory-bus-activity
  * vLLM /metrics num_running (batch), num_waiting, kv_cache_usage
Modeled (derived, NOT measured -- there is no DCGM/Nsight counter in this dataset;
`dram_active_frac` and `tensor_active_frac` are empty for the nvidia-smi backend):
  * FLOP/s      -- from token counts x model geometry
  * GB/s        -- weight reads + KV reads ONLY. Omits activations, KV writes and kernel
                   workspace, so it is a LOWER BOUND on real DRAM traffic (which makes any
                   derived arithmetic intensity an UPPER bound).

FLOP MODEL (the subtle bit)
---------------------------
Llama-3.1-8B is 8.03B params, but 1.05B of those are the embedding table (525M) and the LM
head (525M). The embedding is a *gather* -- zero FLOPs. The LM head runs only on sampled
positions, not on every prompt token. So per-token cost is 2 x BODY params (6.98B), not
2 x 8.03B. Using the naive 8.03B overstates FLOPs by 15% and inflates the apparent MFU.

ATTRIBUTION
-----------
The interval between emission cluster i and i+1 is the time spent COMPUTING i+1, so it is
charged to i+1. (Charging it to i inflates early-step FLOP/s past hardware peak -- a useful
check that the attribution is right.)

CAVEAT
------
This sweep is a finite fill-and-drain burst, not a stationary overload: at rate 32 all 200
requests are sent by t~6.2s of a ~12s window, and the running batch reaches the whole
corpus. Directions are robust; exact figures are soft.

Usage:  python scripts/analyze_bottleneck.py [run_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# --- hardware -------------------------------------------------------------------
A100_PEAK_TFLOPS = 312e12   # bf16 dense tensor core
A100_PEAK_BW     = 1555e9   # HBM2e
A100_POWER_CAP   = 400.0
A100_RIDGE       = A100_PEAK_TFLOPS / A100_PEAK_BW   # ~201 FLOP/byte

# --- Llama-3.1-8B ---------------------------------------------------------------
P_TOTAL   = 8.03e9
VOCAB, H  = 128256, 4096
LAYERS    = 32
P_EMBED   = VOCAB * H                      # GATHERED, not streamed, and 0 FLOPs
P_LMHEAD  = VOCAB * H                      # runs on every SAMPLED position (i.e. every decode token)
P_BODY    = P_TOTAL - P_EMBED - P_LMHEAD   # ~6.98B transformer body

# What a decode token costs in FLOPs: body + LM head (the LM head projects to vocab for
# every generated token -- it is NOT a prefill-only cost).
FLOPS_PER_DECODE_TOKEN = 2 * (P_BODY + P_LMHEAD)
# What a PROMPT token costs in FLOPs: body only (the LM head runs once per sequence, on the
# last position, not on all 512 prompt tokens).
FLOPS_PER_PROMPT_TOKEN = 2 * P_BODY

# What actually streams from HBM each forward pass: body + LM head weights. The embedding
# table is gathered (a handful of rows), not streamed.
WEIGHT_BYTES  = 2 * (P_BODY + P_LMHEAD)    # ~15.0 GB in bf16
KV_PER_TOKEN  = 131072                     # 128 KiB: 2 * 32 layers * 8 KV heads (GQA) * 128 * 2B


def attn_flops_per_decode_token(ctx: float) -> float:
    """QK^T and (softmax)V for one query token against `ctx` cached keys/values."""
    return 4 * LAYERS * H * ctx


STEP_GAP_S = 0.006   # token emissions closer than this belong to one forward pass

DEFAULT_RUN = os.path.join(
    os.path.dirname(__file__), "..",
    "results_published/cloud_a100_40gb/2026-06-15T01-14-49Z_372763",
)


def decode_intensity(batch: float, ctx: float) -> float:
    """Arithmetic intensity (FLOP/byte) of a pure-decode step.

        AI(B) = B * [2*(P_body + P_lmhead) + attn(ctx)] / [2*(P_body + P_lmhead) + B*ctx*KV]

    Weights are read once per step and amortized across the batch; KV must be read PER
    SEQUENCE. So intensity does not keep climbing with the batch -- KV traffic grows with it
    and refuses to be amortized.

    Why decode cannot be the compute wall ON THIS CARD
    --------------------------------------------------
    Note the batch->infinity asymptote is ~203 FLOP/byte at ctx=576, which is marginally
    ABOVE the A100's ~201 ridge. So the asymptote ALONE does not settle it -- an earlier
    version of this analysis wrongly claimed it did.

    What settles it is VRAM. KV must fit. Even unrealistically handing every byte of the
    40 GB card that is not stored weights (~16.1 GB) to KV leaves ~23.9 GB, and at a
    576-token context one sequence's KV is ~75 MB -- so at most ~317 sequences fit, giving
    an intensity of only ~125. Against a ridge of ~201, decode cannot reach the compute roof
    at ANY batch this card can physically hold. At the batch actually observed here (~145)
    the model gives ~85.

    So whatever pins the tensor cores, it is not decode.
    """
    flops = batch * (FLOPS_PER_DECODE_TOKEN + attn_flops_per_decode_token(ctx))
    byts = WEIGHT_BYTES + batch * ctx * KV_PER_TOKEN
    return flops / byts


def analyze_cell(cell: str) -> dict:
    prompt_len = int(cell.split("pin")[1].split("-")[0])
    output_len = int(cell.split("pout")[1])
    rate = float(os.path.basename(cell).split("rate")[1].split("-")[0])
    ctx = prompt_len + output_len / 2

    reqs = [json.loads(line) for line in open(f"{cell}/requests.jsonl")]
    reqs = [r for r in reqs if r["status"] == "success" and not r["is_warmup"]]

    # --- telemetry. Two windows, both reported, because they differ and it matters:
    #     "all"  = whole cell (what summary.csv averages -- diluted by ramp + drain)
    #     "busy" = only samples where the GPU was actually active (util >= 90)
    tel = pd.read_csv(f"{cell}/telemetry.csv").dropna(subset=["util_gpu_pct"])
    busy = tel[tel.util_gpu_pct >= 90]
    if busy.empty:
        busy = tel

    t0 = min(r["actual_send_ts"] for r in reqs)
    t1 = max(r["last_token_ts"] for r in reqs)

    # --- reconstruct forward passes from token emissions
    events = [(ts, j == 0)
              for r in reqs
              for j, ts in enumerate(r["token_timestamps"])]
    events.sort()
    sid_of = np.concatenate([[0], np.cumsum(np.diff([e[0] for e in events]) > STEP_GAP_S)])

    steps: dict[int, dict] = {}
    for (ts, is_first), sid in zip(events, sid_of):
        s = steps.setdefault(sid, {"t0": ts, "decode": 0, "prefills": 0})
        s["decode"] += 1
        s["prefills"] += int(is_first)

    sids = sorted(steps)
    dur = np.array([steps[b]["t0"] - steps[a]["t0"] for a, b in zip(sids[:-1], sids[1:])])
    pf = np.array([steps[b]["prefills"] for b in sids[1:]])
    dec = np.array([steps[b]["decode"] for b in sids[1:]])

    mixed = pf > 0                      # forward passes carrying at least one prefill
    pure = ~mixed
    clock = dur.sum()

    def flops_of(pf_n, dec_n):
        """Prompt tokens pay body FLOPs; decode tokens pay body + LM head (+ attention)."""
        return (pf_n * prompt_len * FLOPS_PER_PROMPT_TOKEN
                + dec_n * (FLOPS_PER_DECODE_TOKEN + attn_flops_per_decode_token(ctx)))

    mixed_tflops = flops_of(pf[mixed], dec[mixed]).sum() / dur[mixed].sum() if mixed.any() else np.nan
    pure_tflops = flops_of(0, dec[pure]).sum() / dur[pure].sum() if pure.any() else np.nan

    # bytes moved by a pure-decode step: weights once + KV per resident sequence
    pure_batch = dec[pure].mean() if pure.any() else np.nan
    pure_bytes = WEIGHT_BYTES + pure_batch * ctx * KV_PER_TOKEN
    pure_gbs = pure_bytes / (dur[pure].mean()) / 1e9 if pure.any() else np.nan

    achieved = len(reqs) / (t1 - t0)
    decode_tps = achieved * output_len

    return dict(
        rate=rate,
        achieved=round(achieved, 2),
        batch_meas=round(busy["num_running"].mean(), 1),      # vLLM /metrics, GPU-busy samples
        batch_step=round(pure_batch, 1),                      # from reconstructed decode steps
        waiting=round(busy["num_waiting"].mean(), 1),
        kv_max_pct=round(100 * tel["kv_cache_perc"].max(), 1),
        # --- power: report BOTH windows, because the difference is the original mistake
        power_all_w=round(tel["power_w"].mean(), 1),          # what summary.csv reports
        power_busy_w=round(busy["power_w"].mean(), 1),
        power_busy_pct_cap=round(100 * busy["power_w"].mean() / A100_POWER_CAP, 1),
        # --- memory-bus activity: conditioning changes the trend, so label it
        membus_all_pct=round(tel["mem_bus_busy_pct"].mean(), 1),
        membus_busy_pct=round(busy["mem_bus_busy_pct"].mean(), 1),
        # --- the clock split (the actual finding)
        mixed_pct_steps=round(100 * mixed.mean(), 1),
        mixed_pct_clock=round(100 * dur[mixed].sum() / clock, 1),
        mixed_tflops=round(mixed_tflops / 1e12, 1),
        mixed_mfu_pct=round(100 * mixed_tflops / A100_PEAK_TFLOPS, 1),
        # --- pure-decode steps: where they actually sit on the roofline
        decode_gbs=round(pure_gbs),
        decode_pct_bw=round(100 * pure_gbs * 1e9 / A100_PEAK_BW, 1),
        decode_AI=round(decode_intensity(pure_batch, ctx), 1),
    )


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    cells = sorted(glob.glob(f"{run}/configs/*rate*"),
                   key=lambda c: float(os.path.basename(c).split("rate")[1].split("-")[0]))
    if not cells:
        sys.exit(f"no open-loop rate cells under {run}")

    df = pd.DataFrame([analyze_cell(c) for c in cells])
    pd.set_option("display.width", 250)

    ctx = 576.0
    kv_max_gb = 40.0 - 2 * P_TOTAL / 1e9        # every byte that is not stored weights
    batch_cap = kv_max_gb * 1e9 / (ctx * KV_PER_TOKEN)
    print(f"\nRun: {os.path.basename(run.rstrip('/'))}")
    print(f"A100 ridge point        : {A100_RIDGE:.0f} FLOP/byte")
    print(f"decode AI asymptote     : {decode_intensity(1e9, ctx):.0f} FLOP/byte "
          f"(batch->inf) -- NOT below the ridge, so the asymptote alone settles nothing")
    print(f"but VRAM caps the batch : <= ~{batch_cap:.0f} seqs (all {kv_max_gb:.1f} GB of "
          f"non-weight memory as KV, ctx {ctx:.0f})")
    print(f"=> decode AI <= ~{decode_intensity(batch_cap, ctx):.0f} FLOP/byte at ANY batch "
          f"this card can hold. Decode cannot reach the compute roof.\n")
    print(df.to_string(index=False))

    print(f"""
How to read this
----------------
decode_AI / decode_pct_bw
    Pure-decode steps sit far to the LEFT of the ridge ({A100_RIDGE:.0f} FLOP/byte) and never
    approach peak bandwidth. Decode is memory-bound -- and it is structurally incapable of
    becoming compute-bound here, because KV traffic grows with the batch AND the KV budget
    caps the batch (see decode_intensity docstring). It is not what runs out; it is what
    gets crowded out.

mixed_pct_clock / mixed_mfu_pct
    Forward passes carrying a prompt. At saturation these are ~30% of passes but eat ~2/3 of
    the wall clock, at the highest MFU anything in this run achieves. THIS is the ceiling.

power_all_w vs power_busy_w
    The original (wrong) diagnosis quoted the whole-window mean -- diluted by idle ramp and
    drain -- and concluded "power is below TDP, so we're memory-bound". Over GPU-busy samples
    the card is near its 400 W cap. Memory-bound kernels draw LESS power; dense GEMM pins a
    power limit. The telemetry was saying "compute" all along.

membus_all_pct vs membus_busy_pct
    Conditioning matters and reverses the trend, so both are shown. The unconditional mean
    RISES with load only because the card is idle most of the window at low rates. Over
    GPU-busy samples it FALLS -- when the GPU is actually working, the memory bus is least
    busy at the highest load.

Not proven
----------
Neither roof is actually hit (prefill steps reach ~60-65% of compute peak; decode steps
~55% of bandwidth peak), so this does NOT establish a hard tensor-core roof as opposed to a
mixed scheduler/kernel-efficiency ceiling. What it does establish is that the HBM-bandwidth
story is wrong and that prefill owns the clock. Settling the rest needs phase-separated
DCGM/Nsight counters, or a prefill-only vs decode-only ablation.
""")


if __name__ == "__main__":
    main()
