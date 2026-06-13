"""Telemetry: Prometheus parsing, metric-name fallback, phase classifier, alignment."""
from __future__ import annotations

import pandas as pd

from gpubench import telemetry as T


PROM = """\
# HELP vllm:num_requests_running running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama3.1-8b"} 7
vllm:num_requests_waiting{model_name="llama3.1-8b"} 2
vllm:kv_cache_usage_perc{model_name="llama3.1-8b"} 0.83
"""

PROM_LEGACY = 'vllm:gpu_cache_usage_perc{model_name="x"} 0.41\n'


def test_parse_and_snapshot_prefers_v1_name():
    fam = T.parse_prom_text(PROM)
    snap = T.vllm_snapshot(fam)
    assert snap["num_running"] == 7 and snap["num_waiting"] == 2
    assert abs(snap["kv_cache_perc"] - 0.83) < 1e-9


def test_snapshot_falls_back_to_legacy_kv_name():
    snap = T.vllm_snapshot(T.parse_prom_text(PROM_LEGACY))
    assert abs(snap["kv_cache_perc"] - 0.41) < 1e-9


def test_parse_labels_handles_commas_in_quotes():
    fam = T.parse_prom_text('vllm:x{a="p,q,r",b="z"} 1.0\n')
    labels = fam["vllm:x"][0][0]
    assert labels["a"] == "p,q,r" and labels["b"] == "z"


def test_classify_phase_signatures():
    # Compute-bound: power near TDP, dram present.
    assert T.classify_phase({"power_w": 290, "power_limit_w": 300, "dram_active_frac": 0.3,
                             "util_gpu_pct": 99}) == "prefill_compute_bound"
    # Memory-bound: dram high, power below TDP.
    assert T.classify_phase({"power_w": 150, "power_limit_w": 300, "dram_active_frac": 0.7,
                             "util_gpu_pct": 60}) == "decode_memory_bound"
    # NVML-only (no dram): unknown discriminator -> proxy/unknown, never a false claim.
    assert T.classify_phase({"util_gpu_pct": 50}) in ("unknown", "idle", "decode_memory_bound")


def test_synthetic_backend_flagged_and_deterministic():
    a = T.SyntheticBackend(); a.open()
    b = T.SyntheticBackend(); b.open()
    assert a.synthetic is True
    assert a.sample()["util_gpu_pct"] == b.sample()["util_gpu_pct"]  # deterministic


def test_align_to_window_masks_by_monotonic_clock():
    df = pd.DataFrame({"t_mono_ns": [10, 20, 30, 40, 50], "util_gpu_pct": [1, 2, 3, 4, 5]})
    out = T.align_to_window(df, 20, 40)
    assert list(out["util_gpu_pct"]) == [2, 3, 4]
