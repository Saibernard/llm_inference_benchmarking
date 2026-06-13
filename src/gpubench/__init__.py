"""gpubench — a single-GPU LLM inference benchmarking harness for vLLM.

A *measurement instrument*: it drives a vLLM OpenAI-compatible server with a
coordinated-omission-correct load generator, correlates serving latency with
GPU telemetry, and reports the latency-throughput knee — cross-checked against
vLLM's own `vllm bench serve` as a reference oracle.
"""

__version__ = "0.1.0"
