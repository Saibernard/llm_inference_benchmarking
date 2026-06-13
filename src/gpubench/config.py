"""Typed configuration + Llama-3.1-8B memory math + YAML loader.

One place owns: the pinned vLLM version, how the server is launched
(``ServeConfig``), the sweep matrix (``SweepConfig``), telemetry/report options,
and the SLO. ``load_config`` reads a single per-platform YAML with sections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .schema import SLO, LoadMode


# --------------------------------------------------------------------------- #
# Single source of truth for the pinned vLLM version. Consumed by ServeConfig,
# the docker-compose image tag, the Colab pip pin, and the run manifest.
# A benchmark is only reproducible if the engine binary is fixed; verify the
# tag exists on Docker Hub / PyPI before a real run and bump in ONE place.
# --------------------------------------------------------------------------- #
DEFAULT_VLLM_VERSION = "0.11.0"
VLLM_IMAGE = f"vllm/vllm-openai:v{DEFAULT_VLLM_VERSION}"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SERVED_MODEL_NAME = "llama3.1-8b"

# Verified from the Llama-3.1-8B-Instruct config.json. Grouped-Query Attention:
# 32 query heads but only 8 KV heads -> KV cache is 4x smaller than naive.
LLAMA31_8B = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,   # GQA — the field the KV-cache formula MUST use
    "head_dim": 128,
    "hidden_size": 4096,
    "max_position_embeddings": 131072,
    "vocab_size": 128256,
    "torch_dtype_bytes": 2,     # bf16
    "weights_gb": 16.1,         # ~8.03B params * 2 bytes
}


def kv_cache_bytes_per_token(
    num_layers: int = LLAMA31_8B["num_hidden_layers"],
    num_kv_heads: int = LLAMA31_8B["num_key_value_heads"],
    head_dim: int = LLAMA31_8B["head_dim"],
    dtype_bytes: int = LLAMA31_8B["torch_dtype_bytes"],
) -> int:
    """KV bytes per token = 2 (K and V) * layers * KV_heads * head_dim * dtype_bytes.

    For Llama-3.1-8B at bf16 this is 2*32*8*128*2 = 131072 bytes = 128 KiB/token.
    Using num_attention_heads (32) here instead of num_key_value_heads (8) is the
    classic 4x overestimate — don't.
    """
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def kv_capacity_estimate(
    gpu_mem_gb: float,
    weights_gb: float = LLAMA31_8B["weights_gb"],
    overhead_gb: float = 3.0,
    ctx_len: int = 4096,
) -> dict[str, float]:
    """Map a GPU memory size + context length to a feasible concurrency grid.

    Approximate (vLLM PagedAttention block sizing + chunked prefill change the
    exact fit). Use for INSTANCE SELECTION, not exact provisioning — let vLLM
    auto-size KV blocks via --gpu-memory-utilization and read the real count
    from startup logs.
    """
    kv_budget_gb = max(0.0, gpu_mem_gb - weights_gb - overhead_gb)
    bytes_per_tok = kv_cache_bytes_per_token()
    total_token_slots = kv_budget_gb * (1024 ** 3) / bytes_per_tok
    return {
        "gpu_mem_gb": gpu_mem_gb,
        "weights_gb": weights_gb,
        "overhead_gb": overhead_gb,
        "kv_budget_gb": round(kv_budget_gb, 2),
        "bytes_per_token": bytes_per_tok,
        "total_token_slots": int(total_token_slots),
        "example_concurrency_at_ctx": int(total_token_slots // max(1, ctx_len)),
        "ctx_len": ctx_len,
    }


# --------------------------------------------------------------------------- #
# How the vLLM server is launched. Recorded verbatim into every report.
# --------------------------------------------------------------------------- #
@dataclass
class ServeConfig:
    model: str = MODEL_ID
    served_model_name: str = SERVED_MODEL_NAME
    max_model_len: int | None = 8192
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int | None = None             # caps server admission (the 'concurrency' knob)
    max_num_batched_tokens: int | None = None   # trades TTFT vs ITL
    enforce_eager: bool = False                 # True = no CUDA graph warmup variance, ~10-20% slower
    quantization: str | None = None             # e.g. 'awq' for 16GB GPUs
    enable_prefix_caching: bool = False         # OFF for canonical runs (else fake-low TTFT)
    seed: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    vllm_version: str = DEFAULT_VLLM_VERSION

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class BenchServeConfig:
    """Mirror of a sweep point for the `vllm bench serve` reference-oracle run."""
    dataset_name: str = "random"
    random_input_len: int = 512
    random_output_len: int = 128
    random_range_ratio: float = 0.0   # 0.0 => EXACT fixed lengths
    num_prompts: int = 200
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    seed: int = 0
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,95,99"
    result_dir: str = "results/oracle"


# --------------------------------------------------------------------------- #
# The sweep matrix. A single engine drives both an open-loop QPS sweep and a
# closed-loop concurrency sweep.
# --------------------------------------------------------------------------- #
@dataclass
class SweepConfig:
    mode: str = LoadMode.OPEN.value
    request_rates: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 16.0])
    concurrencies: list[int] = field(default_factory=lambda: [1, 4, 16, 32, 64])
    prompt_lens: list[int] = field(default_factory=lambda: [512])
    output_lens: list[int] = field(default_factory=lambda: [128])
    num_requests: int = 200          # measurement-window requests per cell
    warmup_requests: int = 16        # discarded (primes caches / CUDA graphs)
    burstiness: float = 1.0          # 1.0 = Poisson; <1 bursty; >1 uniform
    max_concurrency: int | None = None  # client cap for open loop (None = uncapped, CO-correct)
    seed: int = 0
    temperature: float = 0.0
    request_timeout_s: float = 120.0
    cooldown_s: float = 2.0          # idle gap between sweep cells (let KV cache drain)


@dataclass
class TelemetryConfig:
    enabled: bool = True
    backend: str = "auto"            # auto | nvml | dcgm | nvidia-smi | synthetic
    interval_ms: int = 100
    gpu_index: int = 0
    enable_dcgm: bool = False
    scrape_vllm_metrics: bool = True
    metrics_url: str | None = None   # defaults to {base_url}/metrics
    guard_ms: int = 200              # window edge guard band
    tdp_frac_compute: float = 0.9
    dram_frac_decode: float = 0.5


@dataclass
class ReportConfig:
    make_plots: bool = True
    percentiles: list[float] = field(default_factory=lambda: [50.0, 95.0, 99.0])
    knee_min_throughput_gain: float = 0.05   # <5% tps gain => plateau
    knee_util_threshold: float = 95.0
    knee_kv_threshold: float = 0.95


@dataclass
class GpubenchConfig:
    """Top-level resolved config for one sweep run."""
    base_url: str = "http://127.0.0.1:8000"
    model: str = SERVED_MODEL_NAME
    platform: str = "mock"           # mock | colab | aws
    results_dir: str = "results"
    server: ServeConfig = field(default_factory=ServeConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    slo: SLO = field(default_factory=lambda: SLO(ttft_ms=500.0, tpot_ms=50.0))

    @property
    def metrics_url(self) -> str:
        return self.telemetry.metrics_url or f"{self.base_url.rstrip('/')}/metrics"


# --------------------------------------------------------------------------- #
# YAML loading — tolerant of partial sections; unknown keys are ignored with
# a note so a typo doesn't silently misconfigure a run.
# --------------------------------------------------------------------------- #
def _build(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    unknown = set(data) - known
    if unknown:
        print(f"[gpubench.config] warning: ignoring unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**kwargs)


def load_config(path: str | Path) -> GpubenchConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = GpubenchConfig(
        base_url=raw.get("base_url", GpubenchConfig.base_url),
        model=raw.get("model", GpubenchConfig.model),
        platform=raw.get("platform", "mock"),
        results_dir=raw.get("results_dir", "results"),
        server=_build(ServeConfig, raw.get("server")),
        sweep=_build(SweepConfig, raw.get("sweep")),
        telemetry=_build(TelemetryConfig, raw.get("telemetry")),
        report=_build(ReportConfig, raw.get("report")),
        slo=_build(SLO, raw.get("slo")),
    )
    # Normalize "inf" strings in request_rates to float('inf').
    cfg.sweep.request_rates = [
        float("inf") if (isinstance(r, str) and r.lower() == "inf") else float(r)
        for r in cfg.sweep.request_rates
    ]
    return cfg
