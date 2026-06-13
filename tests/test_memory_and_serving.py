"""Llama-3.1-8B memory math (GQA trap) + canonical serve/oracle command flags."""
from __future__ import annotations

from gpubench.config import (ServeConfig, BenchServeConfig, kv_cache_bytes_per_token,
                             kv_capacity_estimate)
from gpubench.serving import build_vllm_serve_cmd, build_vllm_bench_serve_cmd


def test_kv_cache_uses_8_kv_heads_not_32():
    # 2 * 32 layers * 8 KV heads * 128 head_dim * 2 bytes = 131072 (128 KiB)/token.
    assert kv_cache_bytes_per_token() == 131072
    # Using 32 attention heads would 4x it — the classic bug.
    assert kv_cache_bytes_per_token(num_kv_heads=32) == 131072 * 4


def test_kv_capacity_more_memory_more_slots():
    small = kv_capacity_estimate(24)
    big = kv_capacity_estimate(48)
    assert big["total_token_slots"] > small["total_token_slots"]
    assert small["kv_budget_gb"] > 0


def test_serve_cmd_has_canonical_benchmark_flags():
    argv = build_vllm_serve_cmd(ServeConfig(max_num_seqs=64))
    assert "--no-enable-prefix-caching" in argv   # else fake-low TTFT
    assert "--no-enable-log-requests" in argv
    assert "bfloat16" in argv and "--seed" in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "64"


def test_bench_serve_cmd_fixed_lengths_and_ignore_eos():
    argv = build_vllm_bench_serve_cmd(BenchServeConfig(), "llama3.1-8b", "http://x:8000")
    assert "--ignore-eos" in argv
    assert argv[argv.index("--random-range-ratio") + 1] == "0.0"   # EXACT lengths
    assert "--save-result" in argv
