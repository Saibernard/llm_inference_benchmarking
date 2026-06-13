"""Reporter: knee detection + seconds->ms conversion boundary."""
from __future__ import annotations

import math

import pandas as pd

from gpubench import reporter as R
from gpubench.metrics import RunSummary
from gpubench.schema import SLO


def test_chord_knee_finds_elbow_and_guards_short_input():
    # Clear elbow at index 3 (latency stays flat then shoots up).
    idx = R._chord_knee([1, 2, 3, 4, 5], [1.0, 1.05, 1.1, 4.0, 9.0])
    assert idx is not None and 2 <= idx <= 4
    assert R._chord_knee([1, 2], [1, 2]) is None        # < 3 points
    assert R._chord_knee([1, 2, 3], [float("nan")] * 3) is None  # all-NaN -> no knee (was the ptp crash path)


def test_find_knee_on_closed_sweep():
    df = pd.DataFrame({
        "config_id": [f"c{c}" for c in (1, 4, 16, 64)],
        "mode": ["closed"] * 4,
        "concurrency": [1, 4, 16, 64],
        "output_tps": [100, 380, 1480, 1250],
        "e2e_p99": [700, 710, 690, 4900],
        "achieved_qps": [1.5, 6.0, 23.0, 19.5],
        "gpu_util_mean": [60, 70, 82, 85],
        "kv_cache_max": [0.06, 0.25, 1.0, 1.0],
    })
    knee = R.find_knee(df, "concurrency")
    assert knee.knee_config_id is not None
    assert knee.basis in ("chord", "kneedle", "util", "kv")


def test_seconds_to_ms_conversion_happens_once(monkeypatch):
    # aggregate_config must multiply metrics' second-valued summary by 1000 exactly once.
    s = RunSummary(config_id="c", mode="closed", n_total=120, n_success=120, n_failed=0,
                   error_breakdown={}, window_dur_s=10.0, request_throughput=12.0,
                   output_throughput=384.0, total_token_throughput=1536.0,
                   ttft={"p50": 0.05, "p95": 0.06, "p99": 0.07, "mean": 0.05},
                   tpot={"p50": 0.01, "p95": 0.012, "p99": 0.013, "mean": 0.011},
                   itl={"p50": 0.01, "p95": 0.012, "p99": 0.013, "mean": 0.011},
                   e2el={"p50": 0.7, "p95": 0.71, "p99": 0.72, "mean": 0.7},
                   normalized_e2el={"p50": 0.02}, goodput={"goodput_req_per_s": 12.0, "slo_attainment": 1.0})
    monkeypatch.setattr(R.M, "aggregate_run", lambda *a, **k: s)
    cfg_data = {"records": [], "telemetry": pd.DataFrame(),
                "meta": {"window_dur_s": 10.0, "request_rate": None, "concurrency": 16,
                         "prompt_len": 256, "output_len": 64, "model": "llama3.1-8b"}}
    row = R.aggregate_config(cfg_data, "run1", SLO(ttft_ms=500), percentiles=(50, 95, 99))
    assert math.isclose(row["ttft_p50"], 50.0)   # 0.05 s -> 50 ms
    assert math.isclose(row["e2e_p99"], 720.0)   # 0.72 s -> 720 ms
    assert math.isclose(row["output_tps"], 384.0)  # throughput NOT scaled
