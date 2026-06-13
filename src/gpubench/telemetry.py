"""GPU + vLLM /metrics telemetry, time-aligned to load windows.

Owns BOTH halves of server-side telemetry (the completeness audit said the
Prometheus scraper must have exactly one home, and this is it):
  * GPU hardware counters via a pluggable backend (NVML primary, nvidia-smi
    fallback, Synthetic no-op for macOS where there is no NVIDIA GPU).
  * vLLM Prometheus /metrics (kv_cache_usage, running/waiting) scraped on the
    same tick and merged into the same rows.

A single background daemon thread samples every `interval_s`, stamping each row
with time.monotonic_ns() so windows can be selected immune to wall-clock jumps.
Synthetic rows are flagged `synthetic=True` so fake telemetry can NEVER be
mistaken for a measurement.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

import pandas as pd

from .schema import MetricNames


# --------------------------------------------------------------------------- #
# Prometheus text exposition parsing + a normalized vLLM snapshot
# --------------------------------------------------------------------------- #
def _parse_labels(s: str) -> dict[str, str]:
    """Quote-aware label parse: a quoted value may itself contain commas/escapes."""
    out: dict[str, str] = {}
    i, n = 0, len(s)
    while i < n:
        eq = s.find("=", i)
        if eq == -1:
            break
        key = s[i:eq].strip()
        j = eq + 1
        if j < n and s[j] == '"':
            buf, k = [], j + 1
            while k < n and s[k] != '"':
                if s[k] == "\\" and k + 1 < n:
                    buf.append(s[k + 1]); k += 2
                else:
                    buf.append(s[k]); k += 1
            out[key] = "".join(buf)
            i = k + 1
            while i < n and s[i] in ", ":
                i += 1
        else:
            comma = s.find(",", j)
            comma = n if comma == -1 else comma
            out[key] = s[j:comma].strip().strip('"')
            i = comma + 1
    return out


def parse_prom_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Group Prometheus exposition lines by metric name -> list of (labels, value).

    Histograms appear as separate families: name_bucket / name_sum / name_count.
    """
    families: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name = line[: line.index("{")]
                inner = line[line.index("{") + 1 : line.index("}")]
                value = line[line.index("}") + 1 :].strip().split()[0]
                labels = _parse_labels(inner)
            else:
                name, value = line.split()[:2]
                labels = {}
            families.setdefault(name, []).append((labels, float(value)))
        except (ValueError, IndexError):
            continue
    return families


def vllm_snapshot(families: dict[str, list[tuple[dict[str, str], float]]]) -> dict[str, float]:
    """Pull the gauges we correlate with GPU telemetry, trying V1 names then legacy."""
    def g(*names: str) -> float:
        for n in names:
            if families.get(n):
                return families[n][0][1]
        return float("nan")

    return {
        "kv_cache_perc": g(MetricNames.KV_CACHE_USAGE, MetricNames.KV_CACHE_USAGE_LEGACY),
        "num_running": g(MetricNames.NUM_REQUESTS_RUNNING),
        "num_waiting": g(MetricNames.NUM_REQUESTS_WAITING),
    }


# --------------------------------------------------------------------------- #
# GPU backends
# --------------------------------------------------------------------------- #
_GPU_FIELDS = [
    "util_gpu_pct", "mem_bus_busy_pct", "mem_used_bytes", "mem_total_bytes",
    "power_w", "power_limit_w", "sm_clock_mhz", "mem_clock_mhz", "temp_c",
    "sm_active_frac", "dram_active_frac", "tensor_active_frac",
]


def _empty_gpu_row() -> dict[str, float]:
    return {k: float("nan") for k in _GPU_FIELDS}


class NvmlBackend:
    name = "nvml"
    synthetic = False

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._nvml = None
        self._h = None

    @staticmethod
    def available() -> bool:
        try:
            import pynvml  # noqa: F401
            return True
        except Exception:
            return False

    def open(self) -> None:
        import pynvml
        self._nvml = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def sample(self) -> dict[str, float]:
        p, h = self._nvml, self._h
        row = _empty_gpu_row()
        try:
            util = p.nvmlDeviceGetUtilizationRates(h)
            mem = p.nvmlDeviceGetMemoryInfo(h)
            row["util_gpu_pct"] = float(util.gpu)
            row["mem_bus_busy_pct"] = float(util.memory)
            row["mem_used_bytes"] = float(mem.used)
            row["mem_total_bytes"] = float(mem.total)
            row["power_w"] = p.nvmlDeviceGetPowerUsage(h) / 1000.0
            try:
                row["power_limit_w"] = p.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                pass
            row["sm_clock_mhz"] = float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_SM))
            row["mem_clock_mhz"] = float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_MEM))
            row["temp_c"] = float(p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        return row

    def close(self) -> None:
        try:
            if self._nvml:
                self._nvml.nvmlShutdown()
        except Exception:
            pass


_SMI_QUERY = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
              "power.draw,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu")


class NvidiaSmiBackend:
    name = "nvidia-smi"
    synthetic = False

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    def open(self) -> None:
        pass

    def sample(self) -> dict[str, float]:
        row = _empty_gpu_row()
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()[0]
            vals = [v.strip() for v in out.split(",")]
            def f(x: str) -> float:
                return float("nan") if x in ("[N/A]", "[Not Supported]", "") else float(x)
            keys = ["util_gpu_pct", "mem_bus_busy_pct", "mem_used_mib", "mem_total_mib",
                    "power_w", "power_limit_w", "sm_clock_mhz", "mem_clock_mhz", "temp_c"]
            parsed = dict(zip(keys, [f(v) for v in vals]))
            row.update({k: parsed.get(k, float("nan")) for k in row})
            row["mem_used_bytes"] = parsed.get("mem_used_mib", float("nan")) * 1024 * 1024
            row["mem_total_bytes"] = parsed.get("mem_total_mib", float("nan")) * 1024 * 1024
        except Exception:
            pass
        return row

    def close(self) -> None:
        pass


class SyntheticBackend:
    """macOS / no-GPU no-op. Emits plausible, deterministic rows flagged synthetic
    so the full pipeline (and unit tests) run without hardware."""
    name = "synthetic"
    synthetic = True

    def __init__(self, gpu_index: int = 0):
        self._i = 0

    @staticmethod
    def available() -> bool:
        return True

    def open(self) -> None:
        pass

    def sample(self) -> dict[str, float]:
        i = self._i
        self._i += 1
        wave = 0.5 * (1 + math.sin(i / 7.0))
        row = _empty_gpu_row()
        row.update({
            "util_gpu_pct": round(40 + 50 * wave, 1),
            "mem_bus_busy_pct": round(30 + 40 * wave, 1),
            "mem_used_bytes": (8.0 + 6.0 * wave) * (1024 ** 3),
            "mem_total_bytes": 24.0 * (1024 ** 3),
            "power_w": round(120 + 120 * wave, 1),
            "power_limit_w": 300.0,
            "sm_clock_mhz": 1500.0,
            "mem_clock_mhz": 6000.0,
            "temp_c": round(45 + 25 * wave, 1),
        })
        return row

    def close(self) -> None:
        pass


def select_backend(prefer: str = "auto", gpu_index: int = 0):
    order = {
        "nvml": [NvmlBackend], "nvidia-smi": [NvidiaSmiBackend],
        "synthetic": [SyntheticBackend],
        "auto": [NvmlBackend, NvidiaSmiBackend, SyntheticBackend],
    }.get(prefer, [SyntheticBackend])
    for cls in order:
        if cls.available():
            b = cls(gpu_index)
            if cls is SyntheticBackend and prefer not in ("synthetic", "auto"):
                print(f"[gpubench.telemetry] WARNING: '{prefer}' unavailable -> SYNTHETIC telemetry (not real).")
            return b
    return SyntheticBackend(gpu_index)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class TelemetrySampler:
    def __init__(self, backend, interval_s: float = 0.1, metrics_url: str | None = None,
                 clock: Callable[[], int] = time.monotonic_ns):
        self.backend = backend
        self.interval_s = interval_s
        self.metrics_url = metrics_url
        self.clock = clock
        self._rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._http = None

    def _loop(self) -> None:
        if self.metrics_url:
            import httpx
            self._http = httpx.Client(timeout=2.0)
        while not self._stop.is_set():
            row: dict[str, Any] = {"t_mono_ns": self.clock(), "t_wall": time.time(),
                                   "backend": self.backend.name, "synthetic": self.backend.synthetic}
            row.update(self.backend.sample())
            if self.metrics_url:
                try:
                    fam = parse_prom_text(self._http.get(self.metrics_url).text)
                    row.update(vllm_snapshot(fam))
                except Exception:
                    row.update({"kv_cache_perc": float("nan"), "num_running": float("nan"),
                                "num_waiting": float("nan")})
            self._rows.append(row)
            self._stop.wait(self.interval_s)

    def start(self) -> "TelemetrySampler":
        self.backend.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> pd.DataFrame:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.backend.close()
        if self._http:
            self._http.close()
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# --------------------------------------------------------------------------- #
# Window alignment + per-window aggregates + phase classification
# --------------------------------------------------------------------------- #
def align_to_window(df: pd.DataFrame, t_start_ns: int, t_stop_ns: int, guard_ns: int = 0) -> pd.DataFrame:
    if df.empty or "t_mono_ns" not in df:
        return df
    lo, hi = t_start_ns - guard_ns, t_stop_ns + guard_ns
    return df[(df["t_mono_ns"] >= lo) & (df["t_mono_ns"] <= hi)].copy()


def summarize_window(df: pd.DataFrame) -> dict[str, float]:
    def stat(col: str, fn) -> float:
        return float(fn(df[col])) if (col in df and len(df) and df[col].notna().any()) else float("nan")
    mem_used_max = stat("mem_used_bytes", lambda s: s.max())
    return {
        "gpu_util_mean": stat("util_gpu_pct", lambda s: s.mean()),
        "gpu_util_max": stat("util_gpu_pct", lambda s: s.max()),
        "mem_used_max_mib": mem_used_max / (1024 * 1024) if not math.isnan(mem_used_max) else float("nan"),
        "power_mean_w": stat("power_w", lambda s: s.mean()),
        "power_max_w": stat("power_w", lambda s: s.max()),
        "kv_cache_max": stat("kv_cache_perc", lambda s: s.max()),
        "n_samples": float(len(df)),
        "synthetic": bool(df["synthetic"].any()) if "synthetic" in df else False,
    }


def classify_phase(row: dict[str, float], tdp_frac_compute: float = 0.9,
                   dram_frac_decode: float = 0.5) -> str:
    """Heuristic prefill(compute-bound) vs decode(memory-bound) label. Returns
    'unknown' when the discriminating DCGM fields are absent (NVML-only)."""
    power = row.get("power_w", float("nan"))
    limit = row.get("power_limit_w", float("nan"))
    dram = row.get("dram_active_frac", float("nan"))
    util = row.get("util_gpu_pct", float("nan"))
    power_frac = power / limit if (limit and not math.isnan(limit) and limit > 0) else float("nan")
    if not math.isnan(dram):
        if not math.isnan(power_frac) and power_frac >= tdp_frac_compute:
            return "prefill_compute_bound"
        if dram >= dram_frac_decode:
            return "decode_memory_bound"
        return "mixed"
    # NVML-only: weak proxy via power vs TDP
    if not math.isnan(power_frac):
        if power_frac >= tdp_frac_compute:
            return "prefill_compute_bound"
        if not math.isnan(util) and util > 20:
            return "decode_memory_bound"
        return "idle"
    return "unknown"
