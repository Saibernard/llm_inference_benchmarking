"""Correctness tests for the ruler. Each pins a specific audit fix."""
from __future__ import annotations

import math

from gpubench import metrics as M
from gpubench.schema import SLO, RequestStatus
from conftest import make_success_record, make_failed_record


def test_tpot_formula_and_off_by_one():
    # 11 tokens, 10ms apart => decode span 100ms over (11-1) gaps => 10ms TPOT.
    rec = make_success_record(output_tokens=11, ttft_s=0.05, itl_s=0.010)
    M.finalize_record(rec)
    assert math.isclose(rec.tpot_s, 0.010, rel_tol=1e-6)
    assert len(rec.itls_s) == 10  # output_tokens - 1


def test_tpot_none_for_single_token():
    rec = make_success_record(output_tokens=1, ttft_s=0.05, itl_s=0.0)
    M.finalize_record(rec)
    assert rec.tpot_s is None
    assert rec.itls_s == []


def test_itl_token_weighted_vs_tpot_request_weighted_diverge():
    # Anyscale-style example under the rigorous n-1 model:
    #   A = 100 tokens @ 10ms, B = 2 tokens @ 50ms
    #   request-weighted mean TPOT = (10 + 50)/2 = 30ms
    #   token-weighted mean ITL   = (99*10 + 1*50)/100 = 10.4ms
    a = make_success_record(output_tokens=100, ttft_s=0.05, itl_s=0.010)
    b = make_success_record(output_tokens=2, ttft_s=0.05, itl_s=0.050)
    M.finalize_record(a)
    M.finalize_record(b)
    s = M.aggregate_run([a, b], window_dur_s=2.0)
    assert math.isclose(s.mean_tpot_request_weighted_s, 0.030, rel_tol=1e-6)
    assert math.isclose(s.mean_itl_token_weighted_s, 0.0104, rel_tol=1e-3)
    assert s.mean_tpot_request_weighted_s > 2.5 * s.mean_itl_token_weighted_s


def test_throughput_is_window_based_and_excludes_failures():
    a = make_success_record(output_tokens=100, ttft_s=0.05, itl_s=0.010)
    b = make_success_record(output_tokens=2, ttft_s=0.05, itl_s=0.050)
    fail = make_failed_record(RequestStatus.TIMEOUT)
    M.finalize_record(a); M.finalize_record(b)
    s = M.aggregate_run([a, b, fail], window_dur_s=2.0)
    assert s.n_failed == 1 and s.n_success == 2
    assert math.isclose(s.output_throughput, (100 + 2) / 2.0)   # tokens over WINDOW
    assert math.isclose(s.request_throughput, 2 / 2.0)          # failures excluded
    assert s.error_breakdown.get(RequestStatus.TIMEOUT.value) == 1


def test_percentile_min_n_returns_nan():
    assert math.isnan(M.percentile([1, 2, 3], 99))          # < 100 samples
    assert not math.isnan(M.percentile(list(range(200)), 99))


def test_goodput_strict_and_boundary_and_single_token_tpot_zero():
    slo = SLO(ttft_ms=100, tpot_ms=50)
    good = make_success_record(output_tokens=10, ttft_s=0.05, itl_s=0.010)        # tpot 10ms ok
    boundary = make_success_record(output_tokens=2, ttft_s=0.05, itl_s=0.050)     # tpot == 50ms (non-strict)
    one_tok = make_success_record(output_tokens=1, ttft_s=0.05, itl_s=0.0)        # tpot None -> 0 -> passes
    breach = make_success_record(output_tokens=10, ttft_s=0.05, itl_s=0.080)      # tpot 80ms > 50ms
    for r in (good, boundary, one_tok, breach):
        M.finalize_record(r)
    assert M.is_good_request(good, slo)
    assert M.is_good_request(boundary, slo)     # value == bound is GOOD
    assert M.is_good_request(one_tok, slo)      # single-token TPOT treated as 0
    assert not M.is_good_request(breach, slo)


def test_combine_reaggregates_not_averages_percentiles():
    g1 = [make_success_record(output_tokens=5, ttft_s=0.05, itl_s=0.01) for _ in range(60)]
    g2 = [make_success_record(output_tokens=5, ttft_s=0.20, itl_s=0.01) for _ in range(60)]
    for r in g1 + g2:
        M.finalize_record(r)
    combined = M.combine_runs(g1, g2, window_dur_s=10.0)
    # 120 pooled TTFT samples -> P99 is defined (>= 100) and reflects the slow group.
    assert combined.n_success == 120
    assert combined.ttft["p99"] > 0.05
