"""The harness's single ruler: per-request finalization + sweep-cell aggregation.

Every number the harness reports is computed here, once, over raw per-request
samples. The reporter calls into this module rather than reimplementing it, so
the custom load generator and the `vllm bench serve` oracle are aggregated by
identical code.

Correctness decisions (verified against vLLM main `vllm/benchmarks/serve.py`):
  * TPOT = (last_token - first_token) / (output_tokens - 1), guarded output>1.
  * ITL = per-chunk gaps, POOLED across requests (token-weighted). Reported
    alongside request-weighted mean-TPOT — they diverge only at aggregation.
  * Throughput is window-based: numerator over SUCCESSFUL requests, denominator
    is the harness-level perf_counter window passed in (NOT last-minus-first).
  * Percentiles via numpy.percentile(method='linear'); NaN below a min sample
    size (a P99 from <100 samples is fabricated) — a deliberate, documented
    divergence from the oracle, which always interpolates.
  * Goodput is a strict conjunction; a single-token success contributes TPOT=0
    so it passes any TPOT bound (matches vLLM `all_tpots`). Comparison is
    non-strict (value <= bound).
  * Failed/timeout/truncated requests are excluded from latency/throughput and
    tracked in a per-class error table.

Latency anchor: headline TTFT/E2E use the COORDINATED-OMISSION-CORRECTED
timestamp (relative to the intended send time). Under open-loop saturation this
reads higher than service time — that is the honest, user-felt latency and the
whole reason the load generator pre-schedules arrivals. For closed loop the two
anchors are equal by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .schema import RequestRecord, RequestStatus, SLO, TokenSource, LoadMode

MS = 1000.0


# --------------------------------------------------------------------------- #
# Per-request finalization
# --------------------------------------------------------------------------- #
def compute_itls(token_timestamps: Sequence[float]) -> list[float]:
    """Per-chunk gaps between consecutive token-bearing chunks.

    token_timestamps[0] is the first token (its gap from send is TTFT, never an
    ITL). So ITLs are the diffs, length == len(token_timestamps) - 1.
    """
    return [b - a for a, b in zip(token_timestamps, token_timestamps[1:])]


def tpot_per_request(first_token_ts: float, last_token_ts: float, output_tokens: int) -> float | None:
    """Exact vLLM formula: (latency - ttft)/(output_len-1) == decode span/(n-1).

    None for output_tokens <= 1 (undefined decode speed); such requests must be
    excluded from the TPOT percentile/mean sample set.
    """
    if output_tokens is None or output_tokens <= 1:
        return None
    return (last_token_ts - first_token_ts) / (output_tokens - 1)


def finalize_record(rec: RequestRecord, tokenizer=None) -> RequestRecord:
    """Fill derived fields from raw timestamps. Failures get None derived fields
    (never a poison finite value) so they can't leak into percentile arrays."""
    if rec.status != RequestStatus.SUCCESS.value:
        return rec

    # Missing first/last token on a "success" is actually a failure.
    if rec.first_token_ts is None or rec.last_token_ts is None:
        rec.status = RequestStatus.MISSING_USAGE.value
        rec.error = rec.error or "no tokens received on a success-coded record"
        return rec

    # Resolve output token count: server usage preferred, else client tokenizer.
    if rec.output_tokens_server is not None:
        rec.output_tokens = rec.output_tokens_server
        rec.token_source = TokenSource.SERVER_USAGE.value
    elif tokenizer is not None and rec.generated_text:
        rec.output_tokens_client = len(tokenizer(rec.generated_text, add_special_tokens=False).input_ids)
        rec.output_tokens = rec.output_tokens_client
        rec.token_source = TokenSource.CLIENT_TOKENIZER.value
    else:
        # Stream finished but token count unknown -> not a trustworthy success.
        rec.status = RequestStatus.MISSING_USAGE.value
        rec.error = rec.error or "no usage chunk and no client tokenizer"
        return rec

    if rec.output_tokens is None or rec.output_tokens < 1:
        rec.status = RequestStatus.EMPTY_OUTPUT.value
        return rec

    # Prompt tokens: server usage, else the count the loadgen targeted (never None).
    rec.prompt_tokens = rec.prompt_tokens_server if rec.prompt_tokens_server is not None else rec.target_prompt_tokens

    # Latency anchors. service = from POST issue; corrected = from intended arrival.
    rec.ttft_service_s = rec.first_token_ts - rec.actual_send_ts
    rec.ttft_corrected_s = rec.first_token_ts - rec.intended_send_ts
    rec.e2el_service_s = rec.last_token_ts - rec.actual_send_ts
    rec.e2el_corrected_s = rec.last_token_ts - rec.intended_send_ts
    rec.schedule_delay_s = rec.actual_send_ts - rec.intended_send_ts
    rec.admission_delay_s = (rec.sem_acquired_ts - rec.actual_send_ts) if rec.sem_acquired_ts else 0.0

    rec.itls_s = compute_itls(rec.token_timestamps)
    rec.tpot_s = tpot_per_request(rec.first_token_ts, rec.last_token_ts, rec.output_tokens)
    rec.normalized_e2el_s = rec.e2el_service_s / rec.output_tokens
    # vLLM does not guarantee one token per SSE chunk; flag if they disagree.
    rec.tokens_chunks_mismatch = (rec.n_stream_chunks != rec.output_tokens)
    return rec


# --------------------------------------------------------------------------- #
# Percentiles + summaries (raw samples only; never pre-aggregated values)
# --------------------------------------------------------------------------- #
def _default_min_n(q: float) -> int:
    if q >= 99:
        return 100
    if q >= 95:
        return 20
    return 2


def percentile(samples: Sequence[float], q: float, min_n: int | None = None, method: str = "linear") -> float:
    arr = np.asarray([s for s in samples if s is not None and not np.isnan(s)], dtype=float)
    threshold = _default_min_n(q) if min_n is None else min_n
    if arr.size < threshold:
        return float("nan")
    return float(np.percentile(arr, q, method=method))


def summarize_metric(samples: Sequence[float], percentiles: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    arr = np.asarray([s for s in samples if s is not None and not np.isnan(s)], dtype=float)
    out: dict[str, float] = {"n": float(arr.size)}
    if arr.size == 0:
        out.update({"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")})
        for q in percentiles:
            out[f"p{int(q)}"] = float("nan")
        return out
    out.update({
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
    })
    for q in percentiles:
        out[f"p{int(q)}"] = percentile(arr, q)
    return out


# --------------------------------------------------------------------------- #
# Goodput
# --------------------------------------------------------------------------- #
def is_good_request(rec: RequestRecord, slo: SLO, anchor: str = "corrected") -> bool:
    """True iff success and every SET SLO bound holds (non-strict).

    Single-token outputs have undefined TPOT; per vLLM they count as TPOT=0 so
    they trivially satisfy any TPOT bound.
    """
    if not rec.success:
        return False
    ttft_s = rec.ttft_corrected_s if anchor == "corrected" else rec.ttft_service_s
    e2el_s = rec.e2el_corrected_s if anchor == "corrected" else rec.e2el_service_s
    eps = 1e-6  # 1 ns: absorb FP error from timestamp math * 1000 so value==bound stays GOOD
    if slo.ttft_ms is not None and (ttft_s is None or ttft_s * MS > slo.ttft_ms + eps):
        return False
    if slo.tpot_ms is not None:
        tpot_ms = 0.0 if rec.tpot_s is None else rec.tpot_s * MS  # single-token => 0 (oracle parity)
        if tpot_ms > slo.tpot_ms + eps:
            return False
    if slo.e2el_ms is not None and (e2el_s is None or e2el_s * MS > slo.e2el_ms + eps):
        return False
    return True


def compute_goodput(records: Sequence[RequestRecord], slo: SLO, window_dur_s: float, anchor: str = "corrected") -> dict[str, float]:
    success = [r for r in records if r.success]
    good = sum(1 for r in success if is_good_request(r, slo, anchor))
    return {
        "good_completed": float(good),
        "goodput_req_per_s": good / window_dur_s if window_dur_s > 0 else float("nan"),
        # DistServe-style attainment fraction (NOT a field the vLLM oracle emits).
        "slo_attainment": good / len(success) if success else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Per-cell aggregation
# --------------------------------------------------------------------------- #
@dataclass
class RunSummary:
    """One sweep cell, all latencies in SECONDS (reporter converts to ms)."""
    config_id: str
    mode: str
    n_total: int
    n_success: int
    n_failed: int
    error_breakdown: dict[str, int]
    window_dur_s: float
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    ttft: dict[str, float] = field(default_factory=dict)
    tpot: dict[str, float] = field(default_factory=dict)
    itl: dict[str, float] = field(default_factory=dict)
    e2el: dict[str, float] = field(default_factory=dict)
    normalized_e2el: dict[str, float] = field(default_factory=dict)
    mean_itl_token_weighted_s: float = float("nan")
    mean_tpot_request_weighted_s: float = float("nan")
    goodput: dict[str, float] | None = None
    n_token_chunk_mismatch: int = 0


def aggregate_run(
    records: Sequence[RequestRecord],
    window_dur_s: float,
    slo: SLO | None = None,
    drop_warmup: bool = True,
    percentiles: Sequence[float] = (50, 95, 99),
    anchor: str = "corrected",
) -> RunSummary:
    """Aggregate one sweep cell. `window_dur_s` is the harness-level perf_counter
    window measured by the orchestrator around send+gather (mirrors vLLM's
    benchmark_duration) — NOT derived from request timestamps."""
    pool = [r for r in records if not (drop_warmup and r.is_warmup)]
    success = [r for r in pool if r.success]
    failed = [r for r in pool if not r.success]

    error_breakdown: dict[str, int] = {}
    for r in failed:
        error_breakdown[r.status] = error_breakdown.get(r.status, 0) + 1

    ttft_samples = [(r.ttft_corrected_s if anchor == "corrected" else r.ttft_service_s) for r in success]
    e2e_samples = [(r.e2el_corrected_s if anchor == "corrected" else r.e2el_service_s) for r in success]
    tpot_samples = [r.tpot_s for r in success if r.tpot_s is not None]
    norm_samples = [r.normalized_e2el_s for r in success]
    pooled_itls = [g for r in success for g in r.itls_s]  # token-weighted

    out_tokens = sum(r.output_tokens or 0 for r in success)
    in_tokens = sum(r.prompt_tokens or 0 for r in success)
    w = window_dur_s if window_dur_s > 0 else float("nan")

    summary = RunSummary(
        config_id=success[0].config_id if success else (pool[0].config_id if pool else "unknown"),
        mode=success[0].mode if success else (pool[0].mode if pool else LoadMode.OPEN.value),
        n_total=len(pool), n_success=len(success), n_failed=len(failed),
        error_breakdown=error_breakdown, window_dur_s=window_dur_s,
        request_throughput=len(success) / w,
        output_throughput=out_tokens / w,
        total_token_throughput=(in_tokens + out_tokens) / w,
        ttft=summarize_metric(ttft_samples, percentiles),
        tpot=summarize_metric(tpot_samples, percentiles),
        itl=summarize_metric(pooled_itls, percentiles),
        e2el=summarize_metric(e2e_samples, percentiles),
        normalized_e2el=summarize_metric(norm_samples, percentiles),
        mean_itl_token_weighted_s=float(np.mean(pooled_itls)) if pooled_itls else float("nan"),
        mean_tpot_request_weighted_s=float(np.mean(tpot_samples)) if tpot_samples else float("nan"),
        n_token_chunk_mismatch=sum(1 for r in success if r.tokens_chunks_mismatch),
    )
    if slo is not None and not slo.is_empty():
        summary.goodput = compute_goodput(success, slo, window_dur_s, anchor)
    return summary


def combine_runs(*record_groups: Sequence[RequestRecord], window_dur_s: float, **kw) -> RunSummary:
    """Merge groups by RE-POOLING raw records then aggregating once. Never average
    precomputed percentiles (P99-of-P99s is not a P99)."""
    pooled: list[RequestRecord] = []
    for g in record_groups:
        pooled.extend(g)
    return aggregate_run(pooled, window_dur_s=window_dur_s, **kw)
