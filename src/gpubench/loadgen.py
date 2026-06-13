"""Coordinated-omission-correct async load generator for vLLM /v1/completions.

One asyncio engine drives both an OPEN loop (Poisson arrivals at a fixed QPS)
and a CLOSED loop (N fixed concurrent workers). The open loop is the part most
home-grown benchmarks get wrong: it pre-schedules ABSOLUTE send deadlines and
fires each request with create_task WITHOUT awaiting the previous response, so a
slow server can never throttle the offered load and hide tail latency
(coordinated omission). Each record stores intended vs actual send time so the
client-side backlog is auditable.

Audit fixes baked in:
  * actual_send_ts is stamped BEFORE the (optional) client semaphore; admission
    delay = sem_acquired_ts - actual_send_ts is separate and visible. Open-loop
    QPS runs default to NO client cap so offered load is never clipped.
  * TTFT = the first SSE chunk with a non-empty `choices` list, regardless of
    text content (matches vLLM; gating on non-empty text inflates TTFT).
  * SSE 'data:' parsed with prefix-strip (handles 'data:' and 'data: ').
  * usage read with explicit None checks (never `x or y`, which masks a real 0).
  * A stream that ends without a usage chunk is a failure (MISSING_USAGE), so a
    truncated stream can't enter the good set with 0 tokens.
  * Arrival law == vLLM oracle: gamma(shape=burstiness, scale=1/(rate*burstiness)),
    cumulative, rescaled to n/rate (burstiness=1 => exponential => Poisson).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass

import httpx
import numpy as np

from .schema import RequestRecord, RequestStatus, LoadMode


# --------------------------------------------------------------------------- #
# Request payload + prompt construction
# --------------------------------------------------------------------------- #
def build_completion_payload(model: str, prompt: str, output_len: int, *, temperature: float = 0.0,
                             seed: int | None = None) -> dict:
    """Exact /v1/completions body. ignore_eos + min_tokens==max_tokens==N pin the
    decode length so TPOT/throughput measure the hardware, not the model's mood."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "ignore_eos": True,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


async def tokenize_count(client: httpx.AsyncClient, base_url: str, model: str, prompt: str,
                         add_special_tokens: bool = True) -> int:
    """Authoritative server-side token count via vLLM /tokenize (TokenizeResponse.count)."""
    r = await client.post(f"{base_url.rstrip('/')}/tokenize",
                          json={"model": model, "prompt": prompt, "add_special_tokens": add_special_tokens})
    r.raise_for_status()
    return int(r.json()["count"])


async def make_prompt_with_token_count(client: httpx.AsyncClient, base_url: str, model: str,
                                       target_tokens: int, filler: str = " the", max_iters: int = 16
                                       ) -> tuple[str, int]:
    """Build a prompt whose SERVER-tokenized length == target_tokens.

    BPE means tokens != words and the model prepends BOS/special tokens, so we
    converge against the tokenizer's actual count instead of guessing.
    """
    n = max(1, target_tokens)
    count = n
    for _ in range(max_iters):
        prompt = (filler * n).strip()
        count = await tokenize_count(client, base_url, model, prompt)
        if count == target_tokens:
            return prompt, count
        n = max(1, n + (target_tokens - count))
    return (filler * n).strip(), count


# --------------------------------------------------------------------------- #
# SSE streaming + a single request's lifecycle
# --------------------------------------------------------------------------- #
async def _parse_sse(resp: httpx.Response, record: RequestRecord) -> bool:
    """Timestamp TTFT/ITL from the stream. Returns True iff we saw 'data: [DONE]'."""
    saw_done = False
    async for line in resp.aiter_lines():
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].lstrip()  # handles 'data:' and 'data: '
        if data == "[DONE]":
            saw_done = True
            break
        obj = json.loads(data)
        now = time.perf_counter()
        choices = obj.get("choices")
        if choices:  # a non-empty choices list => a token-bearing chunk (TTFT on the first)
            if record.first_token_ts is None:
                record.first_token_ts = now
            record.token_timestamps.append(now)
            record.last_token_ts = now
            record.n_stream_chunks += 1
            ch = choices[0]
            text = ch.get("text")
            if text is None:  # tolerate chat-style delta shape too
                text = (ch.get("delta") or {}).get("content")
            if text:
                record.generated_text += text
            if ch.get("finish_reason") is not None:
                record.finish_reason = ch["finish_reason"]
        usage = obj.get("usage")
        if usage:
            ct = usage.get("completion_tokens")
            if ct is not None:
                record.output_tokens_server = ct
            pt = usage.get("prompt_tokens")
            if pt is not None:
                record.prompt_tokens_server = pt
    return saw_done


async def _do_stream(client: httpx.AsyncClient, url: str, payload: dict, record: RequestRecord,
                     timeout: float) -> None:
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "ignore")[:300]
                record.status = RequestStatus.HTTP_ERROR.value
                record.status_code = resp.status_code
                record.error = body
                return
            record.status_code = 200
            saw_done = await _parse_sse(resp, record)
        if record.first_token_ts is None:
            record.status = RequestStatus.EMPTY_OUTPUT.value if saw_done else RequestStatus.TRUNCATED_STREAM.value
            record.error = record.error or "no tokens received"
        elif not saw_done:
            record.status = RequestStatus.TRUNCATED_STREAM.value
        elif record.output_tokens_server is None:
            record.status = RequestStatus.MISSING_USAGE.value
            record.error = "stream ended without a usage chunk"
        else:
            record.status = RequestStatus.SUCCESS.value
    except httpx.TimeoutException as e:
        record.status = RequestStatus.TIMEOUT.value
        record.error = f"timeout: {e}"
    except httpx.HTTPError as e:
        record.status = RequestStatus.CONNECTION_ERROR.value
        record.error = f"{type(e).__name__}: {e}"
    except Exception as e:  # never let one bad request abort the run
        record.status = RequestStatus.CONNECTION_ERROR.value
        record.error = f"{type(e).__name__}: {e}"


async def send_one(client: httpx.AsyncClient, url: str, payload: dict, record: RequestRecord,
                   semaphore: asyncio.Semaphore | None, timeout: float, sync_intended: bool = False
                   ) -> RequestRecord:
    """Issue one streaming request. actual_send_ts is stamped BEFORE the semaphore
    so a client-side admission wait is attributed to admission_delay, not service."""
    record.actual_send_ts = time.perf_counter()
    record.wall_send_epoch = time.time()
    if sync_intended:  # closed loop: arrival == issue, so corrected == service
        record.intended_send_ts = record.actual_send_ts
    if semaphore is not None:
        async with semaphore:
            record.sem_acquired_ts = time.perf_counter()
            await _do_stream(client, url, payload, record, timeout)
    else:
        await _do_stream(client, url, payload, record, timeout)
    return record


# --------------------------------------------------------------------------- #
# Arrival schedule (open loop)
# --------------------------------------------------------------------------- #
def generate_arrival_schedule(n: int, rate: float, burstiness: float = 1.0, seed: int | None = None,
                              rescale: bool = True) -> list[float]:
    """Cumulative relative send offsets (seconds) from an anchor. rate==inf => all
    at t=0. burstiness=1 => exponential interarrivals => Poisson (vLLM-identical)."""
    if n <= 0:
        return []
    if math.isinf(rate) or rate <= 0:  # rate<=0 => fire all at t=0 (same as inf); avoids div-by-zero
        return [0.0] * n
    rng = np.random.default_rng(seed)
    gaps = rng.gamma(shape=burstiness, scale=1.0 / (rate * burstiness), size=n)
    cum = np.cumsum(gaps)
    if rescale and cum[-1] > 0:
        cum = cum * ((n / rate) / cum[-1])
    return cum.tolist()


# --------------------------------------------------------------------------- #
# One sweep cell
# --------------------------------------------------------------------------- #
@dataclass
class CellPlan:
    config_id: str
    mode: str
    sweep_point: float
    prompt: str
    prompt_len: int
    output_len: int
    n_requests: int
    warmup: int
    request_rate: float | None
    concurrency: int | None
    burstiness: float
    seed: int
    temperature: float
    max_concurrency: int | None
    timeout_s: float
    model: str
    base_url: str


async def _fire_schedule(client, url, payload_base, n, rate, burstiness, seed, sem, plan: CellPlan,
                         is_warmup: bool, prefix: str) -> list[RequestRecord]:
    offsets = generate_arrival_schedule(n, rate, burstiness, seed)
    anchor = time.perf_counter()
    records: list[RequestRecord] = []
    tasks: list[asyncio.Task] = []
    for i in range(n):
        deadline = anchor + offsets[i]
        rec = RequestRecord(
            request_id=f"{prefix}-{i}", config_id=plan.config_id, mode=plan.mode,
            sweep_point=plan.sweep_point, target_prompt_tokens=plan.prompt_len,
            target_output_tokens=plan.output_len, is_warmup=is_warmup, intended_send_ts=deadline,
        )
        records.append(rec)
        now = time.perf_counter()
        if deadline > now:
            await asyncio.sleep(deadline - now)
        tasks.append(asyncio.create_task(send_one(client, url, dict(payload_base), rec, sem, plan.timeout_s)))
    if tasks:
        await asyncio.gather(*tasks)
    return records


async def _run_workers(client, url, payload_base, n_workers, n, plan: CellPlan, is_warmup: bool,
                       prefix: str) -> list[RequestRecord]:
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        q.put_nowait(i)
    records: list[RequestRecord] = []

    async def worker():
        while True:
            try:
                i = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            rec = RequestRecord(
                request_id=f"{prefix}-{i}", config_id=plan.config_id, mode=plan.mode,
                sweep_point=plan.sweep_point, target_prompt_tokens=plan.prompt_len,
                target_output_tokens=plan.output_len, is_warmup=is_warmup,
            )
            await send_one(client, url, dict(payload_base), rec, None, plan.timeout_s, sync_intended=True)
            records.append(rec)

    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(max(1, n_workers))])
    return records


async def run_cell(client: httpx.AsyncClient, plan: CellPlan) -> tuple[list[RequestRecord], float]:
    """Run one sweep cell: warmup (discarded) then the measurement window.

    Returns (measurement_records, window_dur_s) where window_dur_s is a single
    perf_counter delta around the measurement send+gather (vLLM's benchmark_duration).
    """
    url = f"{plan.base_url.rstrip('/')}/v1/completions"
    payload_base = build_completion_payload(plan.model, plan.prompt, plan.output_len,
                                            temperature=plan.temperature, seed=plan.seed)

    if plan.mode in (LoadMode.OPEN.value, LoadMode.MAX_THROUGHPUT.value):
        rate = float("inf") if plan.mode == LoadMode.MAX_THROUGHPUT.value else float(plan.request_rate)
        sem = asyncio.Semaphore(plan.max_concurrency) if plan.max_concurrency else None
        if plan.warmup:
            await _fire_schedule(client, url, payload_base, plan.warmup, rate, plan.burstiness,
                                 plan.seed, sem, plan, is_warmup=True, prefix=f"{plan.config_id}-warm")
        t0 = time.perf_counter()
        recs = await _fire_schedule(client, url, payload_base, plan.n_requests, rate, plan.burstiness,
                                    plan.seed + 1, sem, plan, is_warmup=False, prefix=plan.config_id)
        return recs, time.perf_counter() - t0

    # CLOSED loop
    n_workers = int(plan.concurrency or 1)
    if plan.warmup:
        await _run_workers(client, url, payload_base, n_workers, plan.warmup, plan,
                           is_warmup=True, prefix=f"{plan.config_id}-warm")
    t0 = time.perf_counter()
    recs = await _run_workers(client, url, payload_base, n_workers, plan.n_requests, plan,
                              is_warmup=False, prefix=plan.config_id)
    return recs, time.perf_counter() - t0


def make_async_client(max_concurrency: int, timeout_s: float) -> httpx.AsyncClient:
    """Shared client. Limits sized to concurrency so httpx's default 100-cap can't
    become a hidden second throttle; HTTP/1.1 (vLLM doesn't need HTTP/2)."""
    headroom = max(32, max_concurrency + 16)
    return httpx.AsyncClient(
        http2=False,
        limits=httpx.Limits(max_connections=headroom, max_keepalive_connections=headroom),
        timeout=httpx.Timeout(timeout_s, connect=10.0),
    )
