"""Load generator: payload shape, Poisson arrivals, SSE parsing."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gpubench import loadgen as L
from gpubench.schema import RequestRecord


def test_payload_pins_output_length_and_usage():
    p = L.build_completion_payload("m", "hello", output_len=64, seed=7)
    assert p["max_tokens"] == 64 and p["min_tokens"] == 64 and p["ignore_eos"] is True
    assert p["stream"] is True and p["stream_options"]["include_usage"] is True
    assert p["seed"] == 7


def test_arrival_schedule_poisson_mean_and_inf():
    sched = L.generate_arrival_schedule(2000, rate=10.0, burstiness=1.0, seed=0)
    gaps = np.diff([0.0] + sched)
    assert math.isclose(gaps.mean(), 0.1, rel_tol=0.1)        # mean interarrival ~ 1/rate
    assert math.isclose(sched[-1], 2000 / 10.0, rel_tol=0.05)  # rescaled to n/rate
    assert L.generate_arrival_schedule(5, rate=float("inf")) == [0.0] * 5  # fire-all
    assert L.generate_arrival_schedule(4, rate=0.0) == [0.0] * 4           # rate<=0 must not div-by-zero


def test_record_jsonl_is_spec_valid_with_nonfinite():
    from gpubench.schema import RequestRecord
    r = RequestRecord(request_id="r", config_id="c", sweep_point=float("inf"))
    r.normalized_e2el_s = float("nan")
    s = r.to_jsonl()
    assert "NaN" not in s and "Infinity" not in s   # spec-valid JSON
    json.loads(s)                                    # parses


class _FakeStream:
    """Minimal stand-in for httpx streaming response."""
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b""


@pytest.mark.asyncio
async def test_sse_parse_ttft_itl_and_usage():
    tokens = [
        'data: {"choices":[{"text":"a","finish_reason":null}]}',
        'data: {"choices":[{"text":"b","finish_reason":null}]}',
        'data: {"choices":[{"text":"c","finish_reason":null}]}',
        ': keep-alive comment',  # must be skipped
        'data:{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}',  # no space after data:
        "data: [DONE]",
    ]
    rec = RequestRecord(request_id="r", config_id="c")
    saw_done = await L._parse_sse(_FakeStream(tokens), rec)
    assert saw_done is True
    assert rec.first_token_ts is not None
    assert rec.n_stream_chunks == 3                  # 3 token-bearing chunks, usage chunk NOT counted
    assert rec.output_tokens_server == 3 and rec.prompt_tokens_server == 12
    assert rec.generated_text == "abc"
