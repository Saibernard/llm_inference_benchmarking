"""A GPU-free, vLLM-shaped OpenAI server for developing/testing on macOS.

It streams fake tokens with a configurable TTFT delay + per-token ITL delay, a
simple concurrency-saturation model (so the latency-throughput KNEE is visible
without a GPU), and a /metrics endpoint emitting the EXACT vLLM V1 metric names.
This is the offline contract the real server must match, and the unit-test
fixture for the load generator and reporter.

Endpoints: /v1/completions (text SSE), /v1/chat/completions, /tokenize,
/v1/models, /metrics, /health.

Config via env: MOCK_TTFT_MS, MOCK_ITL_MS, MOCK_JITTER (0-1), MOCK_MAX_CONCURRENCY.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse

from .config import SERVED_MODEL_NAME

app = FastAPI(title="gpubench-mock-vllm")

TTFT_MS = float(os.environ.get("MOCK_TTFT_MS", "40"))
ITL_MS = float(os.environ.get("MOCK_ITL_MS", "8"))
JITTER = float(os.environ.get("MOCK_JITTER", "0.15"))
MAX_CONC = int(os.environ.get("MOCK_MAX_CONCURRENCY", "32"))

# --- shared state for the saturation model + /metrics ---
_state = {"active": 0, "prompt_tokens_total": 0, "generation_tokens_total": 0}
_seed = {"n": 0}


def _jittered(ms: float) -> float:
    # deterministic pseudo-jitter (no RNG, so tests are reproducible)
    _seed["n"] += 1
    wob = JITTER * ((hash(_seed["n"]) % 1000) / 1000.0 - 0.5)  # symmetric +/- about 0
    return ms * (1.0 + wob) / 1000.0


def _saturation_factor() -> float:
    """Past MAX_CONC in-flight requests, every token slows down — this is what
    produces a visible knee on the latency-throughput plot."""
    over = max(0, _state["active"] - MAX_CONC)
    return 1.0 + (over / MAX_CONC) * 1.5


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


async def _completion_stream(prompt_tokens: int, output_len: int, chat: bool, model: str):
    _state["active"] += 1
    try:
        factor = _saturation_factor()
        await asyncio.sleep(_jittered(TTFT_MS) * factor)  # prefill -> TTFT
        created = int(time.time())
        rid = f"cmpl-mock-{_seed['n']}"
        for i in range(output_len):
            if i > 0:
                await asyncio.sleep(_jittered(ITL_MS) * _saturation_factor())  # decode -> ITL
            if chat:
                delta = {"role": "assistant"} if i == 0 else {}
                delta["content"] = "tok "
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": created,
                         "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            else:
                chunk = {"id": rid, "object": "text_completion", "created": created, "model": model,
                         "choices": [{"index": 0, "text": "tok ", "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        _state["generation_tokens_total"] += output_len
        _state["prompt_tokens_total"] += prompt_tokens
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": output_len,
                 "total_tokens": prompt_tokens + output_len}
        final = {"id": rid, "object": "text_completion", "created": created, "model": model,
                 "choices": [], "usage": usage}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        _state["active"] -= 1


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    prompt_tokens = _count_tokens(prompt if isinstance(prompt, str) else " ".join(prompt))
    output_len = int(body.get("max_tokens", 16))
    model = body.get("model", SERVED_MODEL_NAME)
    if not body.get("stream"):
        return JSONResponse({"id": "cmpl-mock", "object": "text_completion", "model": model,
                             "choices": [{"index": 0, "text": "tok " * output_len, "finish_reason": "length"}],
                             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": output_len,
                                       "total_tokens": prompt_tokens + output_len}})
    return StreamingResponse(_completion_stream(prompt_tokens, output_len, chat=False, model=model),
                             media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    prompt_tokens = _count_tokens(" ".join(m.get("content", "") for m in msgs)) if msgs else 1
    output_len = int(body.get("max_tokens", 16))
    model = body.get("model", SERVED_MODEL_NAME)
    return StreamingResponse(_completion_stream(prompt_tokens, output_len, chat=True, model=model),
                             media_type="text/event-stream")


@app.post("/tokenize")
async def tokenize(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    count = _count_tokens(prompt)
    return {"count": count, "max_model_len": 131072, "tokens": list(range(count)), "token_strs": None}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model"}]}


@app.get("/health")
async def health():
    return PlainTextResponse("OK")


@app.get("/metrics")
async def metrics():
    active = _state["active"]
    kv = min(1.0, active / max(1, MAX_CONC))
    waiting = max(0, active - MAX_CONC)
    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running.",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="{SERVED_MODEL_NAME}"}} {min(active, MAX_CONC)}',
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{model_name="{SERVED_MODEL_NAME}"}} {waiting}',
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f'vllm:kv_cache_usage_perc{{model_name="{SERVED_MODEL_NAME}"}} {kv:.4f}',
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{model_name="{SERVED_MODEL_NAME}"}} {_state["prompt_tokens_total"]}',
        "# TYPE vllm:generation_tokens_total counter",
        f'vllm:generation_tokens_total{{model_name="{SERVED_MODEL_NAME}"}} {_state["generation_tokens_total"]}',
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
