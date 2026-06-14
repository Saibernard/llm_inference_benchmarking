"""Sweep orchestrator — the integration hub.

For each sweep cell it: builds an exact-token prompt, brackets the load run with
monotonic-clock fences, samples GPU + vLLM /metrics telemetry over that window,
finalizes the per-request records, and writes the on-disk artifacts the reporter
consumes (requests.jsonl, telemetry.csv, config_meta.json) plus a top-level
run_manifest.json. Aggregation is NOT done here — that is the reporter calling
into metrics, so there is a single ruler.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import __version__
from .config import GpubenchConfig
from .loadgen import CellPlan, make_async_client, make_prompt_with_token_count, run_cell
from .metrics import finalize_record
from .schema import LoadMode, make_config_id
from .telemetry import TelemetrySampler, align_to_window, select_backend


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def _json_safe(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return "inf" if o > 0 else ("-inf" if o < 0 else None)
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def build_cells(cfg: GpubenchConfig) -> list[dict]:
    s = cfg.sweep
    cells: list[dict] = []
    for pin in s.prompt_lens:
        for pout in s.output_lens:
            if s.mode in (LoadMode.OPEN.value, LoadMode.MAX_THROUGHPUT.value):
                for rate in s.request_rates:
                    cells.append({
                        "config_id": make_config_id(s.mode, rate, s.max_concurrency, pin, pout),
                        "mode": s.mode, "request_rate": rate, "concurrency": s.max_concurrency,
                        "prompt_len": pin, "output_len": pout, "sweep_point": rate,
                    })
            else:  # closed
                for c in s.concurrencies:
                    cells.append({
                        "config_id": make_config_id(LoadMode.CLOSED.value, None, c, pin, pout),
                        "mode": LoadMode.CLOSED.value, "request_rate": None, "concurrency": c,
                        "prompt_len": pin, "output_len": pout, "sweep_point": float(c),
                    })
    return cells


async def _wait_health(client: httpx.AsyncClient, base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"{base_url.rstrip('/')}/health"
    last = None
    while time.monotonic() < deadline:
        try:
            r = await client.get(url, timeout=3.0)
            if r.status_code == 200:
                return
        except Exception as e:
            last = e
        await asyncio.sleep(1.0)
    raise RuntimeError(f"server at {base_url} not healthy within {timeout_s}s (last: {last})")


def _raise_fd_limit(target: int = 8192) -> None:
    """A large open-loop connection pool needs file descriptors; raise the soft limit."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, target), hard), hard))
    except Exception:
        pass


def _client_concurrency(cfg: GpubenchConfig) -> int:
    # Open-loop in-flight count is EMERGENT and spikes far above the nominal rate under
    # saturation (e.g. rate 16 with multi-second latency -> ~140 concurrent). If the httpx
    # pool is smaller than that peak, requests queue CLIENT-side and TTFT is inflated (a
    # cross-check vs `vllm bench serve` revealed exactly this). Size it generously so the
    # client never becomes the bottleneck.
    if cfg.sweep.mode in (LoadMode.OPEN.value, LoadMode.MAX_THROUGHPUT.value):
        return 1024
    candidates = [cfg.sweep.max_concurrency or 0]
    if cfg.sweep.concurrencies:
        candidates.append(max(cfg.sweep.concurrencies))
    candidates.append(64)
    return max(candidates)


async def _run_all(cfg: GpubenchConfig, cells: list[dict], run_dir: Path) -> None:
    client = make_async_client(_client_concurrency(cfg), cfg.sweep.request_timeout_s)
    try:
        await _wait_health(client, cfg.base_url)
        prompt_cache: dict[int, tuple[str, int]] = {}
        for idx, cell in enumerate(cells, 1):
            pin = cell["prompt_len"]
            if pin not in prompt_cache:
                prompt_cache[pin] = await make_prompt_with_token_count(client, cfg.base_url, cfg.model, pin)
            prompt, realized_pin = prompt_cache[pin]

            plan = CellPlan(
                config_id=cell["config_id"], mode=cell["mode"], sweep_point=cell["sweep_point"],
                prompt=prompt, prompt_len=pin, output_len=cell["output_len"],
                n_requests=cfg.sweep.num_requests, warmup=cfg.sweep.warmup_requests,
                request_rate=cell["request_rate"], concurrency=cell["concurrency"],
                burstiness=cfg.sweep.burstiness, seed=cfg.sweep.seed, temperature=cfg.sweep.temperature,
                max_concurrency=cfg.sweep.max_concurrency, timeout_s=cfg.sweep.request_timeout_s,
                model=cfg.model, base_url=cfg.base_url,
            )

            sampler = None
            if cfg.telemetry.enabled:
                sampler = TelemetrySampler(
                    select_backend(cfg.telemetry.backend, cfg.telemetry.gpu_index),
                    interval_s=cfg.telemetry.interval_ms / 1000.0,
                    metrics_url=cfg.metrics_url if cfg.telemetry.scrape_vllm_metrics else None,
                )

            print(f"[{idx}/{len(cells)}] {cell['config_id']} ...", flush=True)
            t_start = time.monotonic_ns()
            if sampler:
                sampler.start()
            records, window_dur_s = await run_cell(client, plan)
            t_stop = time.monotonic_ns()
            tdf = sampler.stop() if sampler else None
            if tdf is not None:
                tdf = align_to_window(tdf, t_start, t_stop, cfg.telemetry.guard_ms * 1_000_000)

            for r in records:
                finalize_record(r)

            cdir = run_dir / "configs" / cell["config_id"]
            cdir.mkdir(parents=True, exist_ok=True)
            with (cdir / "requests.jsonl").open("w") as f:
                for r in records:
                    f.write(r.to_jsonl() + "\n")
            if tdf is not None:
                tdf.to_csv(cdir / "telemetry.csv", index=False)
            meta = {
                "config_id": cell["config_id"], "mode": cell["mode"], "model": cfg.model,
                "request_rate": _json_safe(cell["request_rate"]), "concurrency": cell["concurrency"],
                "prompt_len": pin, "realized_prompt_tokens": realized_pin, "output_len": cell["output_len"],
                "sweep_point": _json_safe(cell["sweep_point"]), "window_dur_s": window_dur_s,
                "t_start_mono_ns": t_start, "t_stop_mono_ns": t_stop,
                "n_records": len(records),
            }
            (cdir / "config_meta.json").write_text(json.dumps(meta, indent=2))

            n_ok = sum(1 for r in records if r.success)
            print(f"      done: {n_ok}/{len(records)} ok, window={window_dur_s:.1f}s", flush=True)
            await asyncio.sleep(cfg.sweep.cooldown_s)
    finally:
        await client.aclose()


def run(cfg: GpubenchConfig) -> Path:
    _raise_fd_limit()
    run_id = _run_id()
    run_dir = Path(cfg.results_dir) / run_id
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    backend = select_backend(cfg.telemetry.backend, cfg.telemetry.gpu_index)
    started = datetime.now(timezone.utc).isoformat()
    cells = build_cells(cfg)
    print(f"Run {run_id}: {len(cells)} sweep cells -> {run_dir}", flush=True)

    asyncio.run(_run_all(cfg, cells, run_dir))

    manifest = {
        "run_id": run_id,
        "gpubench_version": __version__,
        "vllm_version": cfg.server.vllm_version,
        "model": cfg.model,
        "platform": cfg.platform,
        "base_url": cfg.base_url,
        "hostname": socket.gethostname(),
        "telemetry_backend": backend.name,
        "telemetry_synthetic": backend.synthetic,
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "config": _json_safe(asdict(cfg)),
        "slo": _json_safe(asdict(cfg.slo)),
        "config_ids": [c["config_id"] for c in cells],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote run_manifest.json. Run dir: {run_dir}", flush=True)
    return run_dir
