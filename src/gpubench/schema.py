"""Canonical data contracts shared by every gpubench subsystem.

This module exists because the design review found the SAME concept
(`RequestRecord`) defined three incompatible ways across subsystems. Everything
now imports the ONE definition here, so serving/loadgen/metrics/telemetry/
reporter cannot drift apart at the seams.

It also centralizes the vLLM Prometheus metric names. vLLM's V1 engine renamed
several of these; using a stale name causes a *silent* all-NaN column rather
than an error, so the names live in exactly one place with a documented
new->legacy fallback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# vLLM Prometheus /metrics names (V1 canonical, with legacy fallbacks).
# Verified against https://docs.vllm.ai/en/stable/design/metrics/
# --------------------------------------------------------------------------- #
class MetricNames:
    # gauges
    NUM_REQUESTS_RUNNING = "vllm:num_requests_running"
    NUM_REQUESTS_WAITING = "vllm:num_requests_waiting"
    # KV-cache occupancy: V1 name first, then the pre-V1 name.
    KV_CACHE_USAGE = "vllm:kv_cache_usage_perc"
    KV_CACHE_USAGE_LEGACY = "vllm:gpu_cache_usage_perc"  # <= v0.6, do NOT lead with this
    # counters (cumulative since server start -> use end-minus-start deltas)
    PROMPT_TOKENS_TOTAL = "vllm:prompt_tokens_total"
    GENERATION_TOKENS_TOTAL = "vllm:generation_tokens_total"
    PREFIX_CACHE_QUERIES = "vllm:prefix_cache_queries_total"
    PREFIX_CACHE_HITS = "vllm:prefix_cache_hits_total"
    REQUEST_SUCCESS_TOTAL = "vllm:request_success_total"
    # histograms
    TTFT_SECONDS = "vllm:time_to_first_token_seconds"
    # This IS TPOT/ITL in V1. There is NO vllm:time_per_output_token_seconds in V1.
    ITL_SECONDS = "vllm:inter_token_latency_seconds"
    ITL_SECONDS_LEGACY = "vllm:time_per_output_token_seconds"  # pre-V1 fallback only
    E2E_SECONDS = "vllm:e2e_request_latency_seconds"

    # (current_name, legacy_name) pairs the scraper tries in order.
    FALLBACKS: dict[str, str] = {
        KV_CACHE_USAGE: KV_CACHE_USAGE_LEGACY,
        ITL_SECONDS: ITL_SECONDS_LEGACY,
    }


# --------------------------------------------------------------------------- #
# Request outcome taxonomy. Only SUCCESS records ever enter latency/throughput
# aggregation; everything else is counted in a separate per-class error table.
# --------------------------------------------------------------------------- #
class RequestStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    CONNECTION_ERROR = "connection_error"
    TRUNCATED_STREAM = "truncated_stream"  # connection dropped before [DONE]
    MISSING_USAGE = "missing_usage"        # stream ended but no usage chunk -> token count unknown
    EMPTY_OUTPUT = "empty_output"          # 0 generated tokens


class TokenSource(str, Enum):
    SERVER_USAGE = "server_usage"          # authoritative: usage chunk from the server
    CLIENT_TOKENIZER = "client_tokenizer"  # fallback: re-tokenize generated_text client-side


class LoadMode(str, Enum):
    OPEN = "open"                # Poisson arrivals at a fixed QPS (reveals the knee)
    CLOSED = "closed"            # N fixed concurrent workers (max throughput @ concurrency N)
    MAX_THROUGHPUT = "max_throughput"  # fire-all + client semaphore cap


@dataclass
class SLO:
    """Conjunctive service-level objective (all set bounds must hold).

    Mirrors vLLM `--goodput KEY:VALUE` (ms). Only the bounds that are set are
    checked. Comparison is NON-strict (value <= bound), matching vLLM's `s >= r`,
    so a request exactly at the bound counts as good.
    """
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2el_ms: float | None = None

    def is_empty(self) -> bool:
        return self.ttft_ms is None and self.tpot_ms is None and self.e2el_ms is None


# --------------------------------------------------------------------------- #
# THE canonical per-request record. Raw timestamps (perf_counter seconds) are
# the truth; derived fields are filled exactly once by metrics.finalize_record.
# Coordinated-omission audit trail: intended vs actual vs admission timestamps.
# --------------------------------------------------------------------------- #
@dataclass
class RequestRecord:
    # identity / sweep tags
    request_id: str
    config_id: str                       # groups a sweep cell; percentiles never cross this
    mode: str = LoadMode.OPEN.value
    sweep_point: float = 0.0             # the QPS (open) or concurrency (closed) of this run
    target_prompt_tokens: int = 0
    target_output_tokens: int = 0
    is_warmup: bool = False

    # --- scheduling / coordinated-omission timestamps (perf_counter seconds) ---
    intended_send_ts: float = 0.0        # pre-scheduled absolute deadline (open loop)
    actual_send_ts: float = 0.0          # stamped right before the POST, BEFORE the semaphore
    sem_acquired_ts: float = 0.0         # stamped AFTER admission (client concurrency cap)
    first_token_ts: float | None = None  # None until first token-bearing chunk arrives
    last_token_ts: float | None = None
    token_timestamps: list[float] = field(default_factory=list)  # one per token-bearing SSE chunk
    n_stream_chunks: int = 0             # token-bearing chunk count (may != output_tokens)
    wall_send_epoch: float | None = None  # wall clock at send, for telemetry correlation only

    # --- server-reported results ---
    prompt_tokens_server: int | None = None
    output_tokens_server: int | None = None
    output_tokens_client: int | None = None   # filled only if server usage absent
    token_source: str | None = None           # TokenSource value actually used
    generated_text: str = ""
    finish_reason: str | None = None

    # --- outcome ---
    status: str = RequestStatus.SUCCESS.value
    error: str | None = None
    status_code: int | None = None

    # --- derived (filled by metrics.finalize_record; NaN/None for failures) ---
    ttft_service_s: float | None = None       # first_token - actual_send (oracle-comparable)
    ttft_corrected_s: float | None = None      # first_token - intended_send (user-felt; CO-corrected)
    e2el_service_s: float | None = None
    e2el_corrected_s: float | None = None
    tpot_s: float | None = None                # (last_token - first_token)/(output_tokens-1); None if <=1
    itls_s: list[float] = field(default_factory=list)  # per-chunk gaps (token-weighted)
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    normalized_e2el_s: float | None = None
    schedule_delay_s: float | None = None      # actual_send - intended_send (client dispatch lag)
    admission_delay_s: float | None = None      # sem_acquired - actual_send (client concurrency wait)
    tokens_chunks_mismatch: bool = False        # True if server streamed multi-token chunks

    @property
    def success(self) -> bool:
        return self.status == RequestStatus.SUCCESS.value

    # ---- serialization (enums already stored as str values) ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        def clean(v):
            if isinstance(v, float):
                return None if (math.isnan(v) or math.isinf(v)) else v  # spec-valid JSON (no NaN/Infinity literals)
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v
        return json.dumps({k: clean(v) for k, v in self.to_dict().items()})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RequestRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Deterministic sweep-cell id. One file per config_id under results/<run>/configs/.
# --------------------------------------------------------------------------- #
def make_config_id(
    mode: str, rate: float | None, concurrency: int | None, prompt_len: int, output_len: int
) -> str:
    rate_str = "inf" if (rate is not None and math.isinf(rate)) else ("na" if rate is None else f"{rate:g}")
    conc_str = "na" if concurrency is None else str(int(concurrency))
    return f"{mode}-rate{rate_str}-conc{conc_str}-pin{prompt_len}-pout{output_len}"


# --------------------------------------------------------------------------- #
# Canonical summary column order (one row per sweep cell). Latencies are in
# MILLISECONDS here (the reporter converts from the seconds the metrics module
# computes). Keeping ms matches vLLM bench serve, eliminating 1000x mistakes.
# --------------------------------------------------------------------------- #
SUMMARY_COLUMNS: list[str] = [
    # config dims
    "run_id", "config_id", "model", "mode", "request_rate", "concurrency",
    "prompt_len", "output_len",
    # latency (ms)
    "ttft_p50", "ttft_p95", "ttft_p99", "ttft_mean",
    "tpot_p50", "tpot_p95", "tpot_p99", "tpot_mean",
    "itl_p50", "itl_p95", "itl_p99", "itl_mean",
    "e2e_p50", "e2e_p95", "e2e_p99", "e2e_mean",
    # throughput
    "output_tps", "total_tps", "req_per_s",
    # reliability
    "goodput", "slo_attainment", "success_rate", "n_requests", "n_failed",
    # gpu / server telemetry
    "gpu_util_mean", "gpu_util_max", "mem_used_max_mib",
    "power_mean_w", "power_max_w", "kv_cache_max",
    # derived flags
    "offered_qps", "achieved_qps", "saturation_flag", "bench_duration_s",
]


@dataclass
class KneeResult:
    knee_index: int | None = None
    knee_config_id: str | None = None
    knee_output_tps: float | None = None
    knee_e2e_p99_ms: float | None = None
    max_sustainable_qps: float | None = None
    basis: str | None = None  # 'kneedle' | 'chord' | 'util' | 'kv' | 'none'


def nan() -> float:
    return float("nan")


def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)
