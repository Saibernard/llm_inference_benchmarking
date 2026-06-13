"""Build the vLLM server launch command and the `vllm bench serve` oracle command.

These are the canonical, reproducible invocations. The deploy scripts and the
Colab notebook call build_vllm_serve_cmd(ServeConfig) instead of hardcoding argv,
so the same flags (prefix-caching OFF, logging OFF, fixed dtype/seed/util) apply
everywhere — the completeness audit flagged hardcoded launches drifting from the
canonical ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ServeConfig, BenchServeConfig


def build_vllm_serve_cmd(cfg: ServeConfig) -> list[str]:
    argv = [
        "vllm", "serve", cfg.model,
        "--served-model-name", cfg.served_model_name,
        "--dtype", cfg.dtype,
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--seed", str(cfg.seed),
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--no-enable-log-requests",            # logging off (the old --disable-log-requests is deprecated)
    ]
    if cfg.max_model_len is not None:
        argv += ["--max-model-len", str(cfg.max_model_len)]
    if cfg.max_num_seqs is not None:
        argv += ["--max-num-seqs", str(cfg.max_num_seqs)]
    if cfg.max_num_batched_tokens is not None:
        argv += ["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)]
    if cfg.enforce_eager:
        argv += ["--enforce-eager"]
    if cfg.quantization:
        argv += ["--quantization", cfg.quantization]
    if not cfg.enable_prefix_caching:
        argv += ["--no-enable-prefix-caching"]  # else repeated prompts fake low TTFT
    return argv


def build_vllm_bench_serve_cmd(cfg: BenchServeConfig, model: str, base_url: str) -> list[str]:
    """Reference-oracle invocation. --random-range-ratio 0.0 => EXACT fixed lengths."""
    argv = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--endpoint", "/v1/completions",
        "--model", model,
        "--base-url", base_url,
        "--dataset-name", cfg.dataset_name,
        "--random-input-len", str(cfg.random_input_len),
        "--random-output-len", str(cfg.random_output_len),
        "--random-range-ratio", str(cfg.random_range_ratio),
        "--ignore-eos",
        "--num-prompts", str(cfg.num_prompts),
        "--seed", str(cfg.seed),
        "--percentile-metrics", cfg.percentile_metrics,
        "--metric-percentiles", cfg.metric_percentiles,
        "--save-result", "--result-dir", cfg.result_dir,
    ]
    argv += ["--request-rate", "inf" if cfg.request_rate == float("inf") else str(cfg.request_rate)]
    if cfg.max_concurrency is not None:
        argv += ["--max-concurrency", str(cfg.max_concurrency)]
    return argv


def parse_bench_serve_json(path: str | Path) -> dict[str, float]:
    """Normalize the JSON `vllm bench serve --save-result` writes into the harness
    comparison schema (ms latencies, tok/s) for side-by-side diffing."""
    raw = json.loads(Path(path).read_text())
    keys = [
        "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
    ]
    return {k: raw[k] for k in keys if k in raw}
