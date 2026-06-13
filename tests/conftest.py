"""Shared test fixtures/helpers."""
from __future__ import annotations

from gpubench.schema import RequestRecord, RequestStatus


def make_success_record(output_tokens: int, ttft_s: float, itl_s: float, *, config_id: str = "c",
                        send_ts: float = 0.0) -> RequestRecord:
    """Build a finalized-ready SUCCESS record with `output_tokens` evenly spaced at `itl_s`."""
    first = send_ts + ttft_s
    ts = [first + i * itl_s for i in range(output_tokens)]
    return RequestRecord(
        request_id="r", config_id=config_id, mode="closed",
        target_prompt_tokens=128, target_output_tokens=output_tokens,
        intended_send_ts=send_ts, actual_send_ts=send_ts,
        first_token_ts=first, last_token_ts=ts[-1] if ts else None,
        token_timestamps=ts, n_stream_chunks=output_tokens,
        prompt_tokens_server=128, output_tokens_server=output_tokens,
        status=RequestStatus.SUCCESS.value,
    )


def make_failed_record(status: RequestStatus = RequestStatus.TIMEOUT) -> RequestRecord:
    return RequestRecord(request_id="f", config_id="c", status=status.value, error="boom",
                         actual_send_ts=0.0, intended_send_ts=0.0)
