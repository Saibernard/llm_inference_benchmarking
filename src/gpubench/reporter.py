"""Reporting: raw JSONL -> one-row-per-cell summary (CSV+JSON) -> plots.

The reporter is a pure consumer and the ONLY place aggregation happens for the
report: it calls metrics.aggregate_run (the single ruler) rather than recomputing
percentiles, and it is the ONE place seconds become milliseconds (so a 1000x unit
bug can't hide). Plots are headless (Agg) so they render on Colab/AWS.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402
from .schema import SLO, SUMMARY_COLUMNS, KneeResult, LoadMode, RequestRecord  # noqa: E402
from .telemetry import summarize_window  # noqa: E402

MS = 1000.0


# --------------------------------------------------------------------------- #
# Loading a run from disk
# --------------------------------------------------------------------------- #
def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    configs: dict[str, dict] = {}
    for cdir in sorted((run_dir / "configs").glob("*")):
        if not cdir.is_dir():
            continue
        records = []
        rp = cdir / "requests.jsonl"
        if rp.exists():
            for line in rp.read_text().splitlines():
                if line.strip():
                    rec = RequestRecord.from_dict(json.loads(line))
                    M.finalize_record(rec)  # idempotent; guarantees derived fields
                    records.append(rec)
        try:
            tel = pd.read_csv(cdir / "telemetry.csv") if (cdir / "telemetry.csv").exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:  # sampler collected 0 rows on a very short cell
            tel = pd.DataFrame()
        meta = json.loads((cdir / "config_meta.json").read_text())
        configs[cdir.name] = {"records": records, "telemetry": tel, "meta": meta}
    return {"manifest": manifest, "configs": configs, "run_dir": run_dir}


def _num_rate(v) -> float:
    if v in ("inf", None):
        return float("inf") if v == "inf" else float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Aggregate one cell into a summary row (seconds -> milliseconds HERE)
# --------------------------------------------------------------------------- #
def aggregate_config(cfg_data: dict, run_id: str, slo: SLO, percentiles=(50, 95, 99)) -> dict:
    records, telemetry, meta = cfg_data["records"], cfg_data["telemetry"], cfg_data["meta"]
    _w = meta.get("window_dur_s")
    window = float(_w) if _w is not None else float("nan")  # .get(...) or nan would corrupt a legit 0.0
    s = M.aggregate_run(records, window_dur_s=window, slo=slo, percentiles=percentiles)
    gpu = summarize_window(telemetry) if len(telemetry) else {}

    def lat(d: dict, key: str) -> float:
        v = d.get(key, float("nan"))
        return v * MS if v is not None else float("nan")

    rate = _num_rate(meta.get("request_rate"))
    conc = meta.get("concurrency")
    gp = s.goodput or {}
    row = {
        "run_id": run_id, "config_id": s.config_id, "model": meta.get("model"), "mode": s.mode,
        "request_rate": rate, "concurrency": conc,
        "prompt_len": meta.get("prompt_len"), "output_len": meta.get("output_len"),
        "ttft_p50": lat(s.ttft, "p50"), "ttft_p95": lat(s.ttft, "p95"), "ttft_p99": lat(s.ttft, "p99"), "ttft_mean": lat(s.ttft, "mean"),
        "tpot_p50": lat(s.tpot, "p50"), "tpot_p95": lat(s.tpot, "p95"), "tpot_p99": lat(s.tpot, "p99"), "tpot_mean": lat(s.tpot, "mean"),
        "itl_p50": lat(s.itl, "p50"), "itl_p95": lat(s.itl, "p95"), "itl_p99": lat(s.itl, "p99"), "itl_mean": lat(s.itl, "mean"),
        "e2e_p50": lat(s.e2el, "p50"), "e2e_p95": lat(s.e2el, "p95"), "e2e_p99": lat(s.e2el, "p99"), "e2e_mean": lat(s.e2el, "mean"),
        "output_tps": s.output_throughput, "total_tps": s.total_token_throughput, "req_per_s": s.request_throughput,
        "goodput": gp.get("goodput_req_per_s", float("nan")), "slo_attainment": gp.get("slo_attainment", float("nan")),
        "success_rate": (s.n_success / s.n_total) if s.n_total else float("nan"),
        "n_requests": s.n_total, "n_failed": s.n_failed,
        "gpu_util_mean": gpu.get("gpu_util_mean", float("nan")), "gpu_util_max": gpu.get("gpu_util_max", float("nan")),
        "mem_used_max_mib": gpu.get("mem_used_max_mib", float("nan")),
        "power_mean_w": gpu.get("power_mean_w", float("nan")), "power_max_w": gpu.get("power_max_w", float("nan")),
        "kv_cache_max": gpu.get("kv_cache_max", float("nan")),
        "offered_qps": rate if s.mode != LoadMode.CLOSED.value else float("nan"),
        "achieved_qps": s.request_throughput,
        "saturation_flag": False, "bench_duration_s": window,
    }
    return {k: row.get(k) for k in SUMMARY_COLUMNS}


# --------------------------------------------------------------------------- #
# Knee / saturation detection
# --------------------------------------------------------------------------- #
def _chord_knee(x, y) -> int | None:
    """Pure-numpy elbow: point of max perpendicular distance to the first-last chord."""
    import numpy as np
    if len(x) < 3:
        return None
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    xn = (x - x.min()) / (np.ptp(x) or 1)   # np.ptp, not x.ptp() (removed in NumPy 2.0)
    yn = (y - y.min()) / (np.ptp(y) or 1)
    x1, y1, x2, y2 = xn[0], yn[0], xn[-1], yn[-1]
    denom = math.hypot(x2 - x1, y2 - y1) or 1
    dist = abs((y2 - y1) * xn - (x2 - x1) * yn + x2 * y1 - y2 * x1) / denom
    return int(dist.argmax())


def find_knee(df: pd.DataFrame, load_col: str, util_thr: float = 95.0, kv_thr: float = 0.95) -> KneeResult:
    if df.empty or load_col not in df:
        return KneeResult(basis="none")
    d = df.sort_values(load_col).reset_index(drop=True)
    idx = None
    basis = "chord"
    try:
        from kneed import KneeLocator
        kl = KneeLocator(d["output_tps"], d["e2e_p99"], curve="concave", direction="increasing")
        if kl.knee is not None:
            idx = int((d["output_tps"] - kl.knee).abs().idxmin())
            basis = "kneedle"
    except Exception:
        idx = None
    if idx is None:
        idx = _chord_knee(d[load_col].tolist(), d["e2e_p99"].tolist())
    if idx is None:
        # fall back to first cell breaching a physical saturation signal
        for i, r in d.iterrows():
            if (r.get("gpu_util_mean") or 0) >= util_thr:
                idx, basis = i, "util"
                break
            if (r.get("kv_cache_max") or 0) >= kv_thr:
                idx, basis = i, "kv"
                break
    if idx is None:
        return KneeResult(basis="none")
    row = d.iloc[idx]
    return KneeResult(
        knee_index=int(idx), knee_config_id=row["config_id"],
        knee_output_tps=float(row["output_tps"]), knee_e2e_p99_ms=float(row["e2e_p99"]),
        max_sustainable_qps=float(row.get("achieved_qps", float("nan"))), basis=basis,
    )


def build_summary(run: dict, slo: SLO, percentiles=(50, 95, 99)) -> tuple[pd.DataFrame, KneeResult]:
    run_id = run["manifest"]["run_id"]
    rows = [aggregate_config(cd, run_id, slo, percentiles) for cd in run["configs"].values()]
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    load_col = "concurrency" if (df["mode"] == LoadMode.CLOSED.value).all() else "request_rate"
    knee = find_knee(df, load_col)
    if knee.knee_config_id is not None:
        df.loc[df["config_id"] == knee.knee_config_id, "saturation_flag"] = True
    return df, knee


def write_summary(df: pd.DataFrame, run: dict, knee: KneeResult) -> tuple[Path, Path]:
    out = run["run_dir"]
    csv_path = out / "summary.csv"
    json_path = out / "summary.json"
    df.to_csv(csv_path, index=False)
    payload = {
        "run_manifest": run["manifest"],
        "knee": asdict(knee),
        "rows": json.loads(df.to_json(orient="records")),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _load_axis(df: pd.DataFrame) -> tuple[str, str]:
    if (df["mode"] == LoadMode.CLOSED.value).all():
        return "concurrency", "Concurrency (workers)"
    return "request_rate", "Offered load (req/s)"


def _stamp(fig, run_id: str):
    fig.text(0.99, 0.01, f"gpubench · {run_id}", ha="right", va="bottom", fontsize=7, color="gray")


def plot_pareto_knee(df: pd.DataFrame, knee: KneeResult, manifest: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    d = df.sort_values("output_tps")
    ax.plot(d["output_tps"], d["e2e_p99"], "-o", color="#2b6cb0", label="P99 E2E latency")
    if knee.knee_output_tps is not None:
        ax.scatter([knee.knee_output_tps], [knee.knee_e2e_p99_ms], color="crimson", zorder=5, s=90)
        ax.annotate(f"KNEE ({knee.basis})\n{knee.knee_output_tps:.0f} tok/s @ {knee.knee_e2e_p99_ms:.0f} ms",
                    (knee.knee_output_tps, knee.knee_e2e_p99_ms), textcoords="offset points", xytext=(12, -28),
                    fontsize=9, bbox=dict(boxstyle="round", fc="#fff5f5", ec="crimson"))
    ax.set_xlabel("Output throughput (tokens/s)")
    ax.set_ylabel("End-to-end P99 latency (ms)")
    ax.set_title(f"Latency–throughput knee · {manifest.get('model')} · {manifest.get('telemetry_backend')}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _stamp(fig, manifest["run_id"])
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency_vs_load(df: pd.DataFrame, manifest: dict, out: Path) -> Path:
    xcol, xlabel = _load_axis(df)
    d = df.sort_values(xcol)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for q, c in [("p50", "#90cdf4"), ("p95", "#3182ce"), ("p99", "#1a365d")]:
        a1.plot(d[xcol], d[f"ttft_{q}"], "-o", color=c, label=f"TTFT {q.upper()}")
        a2.plot(d[xcol], d[f"tpot_{q}"], "-o", color=c, label=f"TPOT {q.upper()}")
    a1.set_ylabel("TTFT (ms)"); a1.set_title("Time to first token (prefill/queue) vs load"); a1.grid(True, alpha=0.3); a1.legend()
    a2.set_ylabel("TPOT (ms)"); a2.set_title("Time per output token (decode) vs load"); a2.set_xlabel(xlabel); a2.grid(True, alpha=0.3); a2.legend()
    _stamp(fig, manifest["run_id"])
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gpu_saturation(df: pd.DataFrame, rep_tel: pd.DataFrame, manifest: dict, out: Path) -> Path:
    xcol, xlabel = _load_axis(df)
    d = df.sort_values(xcol)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(d[xcol], d["gpu_util_mean"], "-o", color="#2f855a", label="GPU util mean (%)")
    a1.plot(d[xcol], d["kv_cache_max"] * 100, "-s", color="#b7791f", label="KV-cache max (%)")
    a1b = a1.twinx()
    a1b.plot(d[xcol], d["power_mean_w"], "-^", color="#c53030", label="Power mean (W)")
    a1.set_xlabel(xlabel); a1.set_ylabel("Utilization / KV-cache (%)"); a1b.set_ylabel("Power (W)")
    a1.set_title("GPU saturation vs load"); a1.grid(True, alpha=0.3)
    lines = a1.get_lines() + a1b.get_lines()
    a1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    if len(rep_tel) and "t_mono_ns" in rep_tel:
        t = (rep_tel["t_mono_ns"] - rep_tel["t_mono_ns"].iloc[0]) / 1e9
        if "util_gpu_pct" in rep_tel:
            a2.plot(t, rep_tel["util_gpu_pct"], color="#2f855a", label="GPU util (%)")
        if "kv_cache_perc" in rep_tel:
            a2.plot(t, rep_tel["kv_cache_perc"] * 100, color="#b7791f", label="KV-cache (%)")
        a2.set_title("Representative run (time series)")
        a2.set_xlabel("Time (s)"); a2.set_ylabel("%"); a2.grid(True, alpha=0.3); a2.legend(fontsize=8)
    else:
        a2.text(0.5, 0.5, "no telemetry", ha="center")
    if "synthetic" in rep_tel and len(rep_tel) and rep_tel["synthetic"].any():
        a2.text(0.5, 0.95, "SYNTHETIC telemetry (no GPU)", ha="center", va="top", color="crimson", transform=a2.transAxes)
    _stamp(fig, manifest["run_id"])
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_goodput_vs_load(df: pd.DataFrame, manifest: dict, out: Path) -> Path:
    xcol, xlabel = _load_axis(df)
    d = df.sort_values(xcol)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(d[xcol], d["req_per_s"], "-o", color="#718096", label="Raw throughput (req/s)")
    ax.plot(d[xcol], d["goodput"], "-o", color="#2f855a", label="Goodput (SLO-meeting req/s)")
    ax.fill_between(d[xcol], d["goodput"], d["req_per_s"], color="crimson", alpha=0.12, label="Wasted (SLO-violating)")
    if d["goodput"].notna().any():
        peak = d.loc[d["goodput"].idxmax()]
        ax.annotate(f"peak goodput\n{peak['goodput']:.2f} req/s", (peak[xcol], peak["goodput"]),
                    textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel("Requests/s")
    ax.set_title("Goodput vs raw throughput (the gap is work nobody can use)")
    ax.grid(True, alpha=0.3); ax.legend()
    _stamp(fig, manifest["run_id"])
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_report(run_dir: str | Path, slo: SLO | None = None, make_plots: bool = True) -> dict:
    run = load_run(run_dir)
    if slo is None:
        s = run["manifest"].get("slo", {}) or {}
        slo = SLO(ttft_ms=s.get("ttft_ms"), tpot_ms=s.get("tpot_ms"), e2el_ms=s.get("e2el_ms"))
    df, knee = build_summary(run, slo)
    csv_path, json_path = write_summary(df, run, knee)
    paths = {"summary_csv": csv_path, "summary_json": json_path, "plots": []}
    if make_plots and not df.empty:
        plots = run["run_dir"] / "plots"
        rep_id = knee.knee_config_id or df.iloc[0]["config_id"]
        rep_tel = run["configs"].get(rep_id, {}).get("telemetry", pd.DataFrame())
        paths["plots"] = [
            plot_pareto_knee(df, knee, run["manifest"], plots / "pareto_knee.png"),
            plot_latency_vs_load(df, run["manifest"], plots / "ttft_tpot_vs_load.png"),
            plot_gpu_saturation(df, rep_tel, run["manifest"], plots / "gpu_saturation.png"),
            plot_goodput_vs_load(df, run["manifest"], plots / "goodput_vs_load.png"),
        ]
    paths["knee"] = knee
    paths["summary_df"] = df
    return paths
