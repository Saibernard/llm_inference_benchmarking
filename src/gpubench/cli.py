"""gpubench command-line interface.

  gpubench serve-mock                 # GPU-free mock vLLM server (for macOS dev)
  gpubench run --config configs/mock.yaml   # drive a sweep, write results
  gpubench report results/<run_id>    # (re)build summary + plots from raw data
  gpubench plan --gpu-mem 24          # Llama-3.1-8B memory/capacity math
  gpubench crosscheck --config ...    # print the `vllm bench serve` oracle command
"""

from __future__ import annotations

import argparse
import os
import sys


def _cmd_serve_mock(args) -> int:
    os.environ["MOCK_TTFT_MS"] = str(args.ttft_ms)
    os.environ["MOCK_ITL_MS"] = str(args.itl_ms)
    os.environ["MOCK_JITTER"] = str(args.jitter)
    os.environ["MOCK_MAX_CONCURRENCY"] = str(args.max_concurrency)
    from .mock_server import serve
    print(f"mock vLLM on http://{args.host}:{args.port}  (ttft={args.ttft_ms}ms itl={args.itl_ms}ms "
          f"max_conc={args.max_concurrency})")
    serve(host=args.host, port=args.port)
    return 0


def _cmd_run(args) -> int:
    from .config import load_config
    from . import orchestrator, reporter
    cfg = load_config(args.config)
    if args.base_url:
        cfg.base_url = args.base_url
    if args.platform:
        cfg.platform = args.platform
    run_dir = orchestrator.run(cfg)
    if not args.no_report:
        paths = reporter.generate_report(run_dir, slo=cfg.slo, make_plots=cfg.report.make_plots)
        print(f"\nSummary: {paths['summary_csv']}")
        for p in paths.get("plots", []):
            print(f"Plot:    {p}")
        k = paths["knee"]
        if k.knee_config_id:
            print(f"Knee:    {k.knee_output_tps:.0f} tok/s @ {k.knee_e2e_p99_ms:.0f} ms P99 "
                  f"(basis={k.basis}, cell={k.knee_config_id})")
    print(f"\nRun dir: {run_dir}")
    return 0


def _cmd_report(args) -> int:
    from . import reporter
    paths = reporter.generate_report(args.run_dir, make_plots=not args.no_plots)
    print(f"Summary: {paths['summary_csv']}")
    for p in paths.get("plots", []):
        print(f"Plot:    {p}")
    return 0


def _cmd_plan(args) -> int:
    from .config import kv_capacity_estimate, kv_cache_bytes_per_token
    print(f"Llama-3.1-8B KV cache: {kv_cache_bytes_per_token()} bytes/token "
          f"({kv_cache_bytes_per_token()/1024:.0f} KiB) — uses 8 KV heads (GQA), not 32.\n")
    mems = [args.gpu_mem] if args.gpu_mem else [16, 24, 40, 48, 80]
    print(f"{'GPU(GB)':>8} {'KVbudget(GB)':>13} {'token-slots':>12} {f'conc@{args.ctx}ctx':>14}")
    for m in mems:
        e = kv_capacity_estimate(m, ctx_len=args.ctx)
        print(f"{m:>8} {e['kv_budget_gb']:>13} {e['total_token_slots']:>12} {e['example_concurrency_at_ctx']:>14}")
    return 0


def _cmd_crosscheck(args) -> int:
    from .config import load_config, BenchServeConfig
    from .serving import build_vllm_bench_serve_cmd, build_vllm_serve_cmd
    cfg = load_config(args.config)
    print("# 1) Launch vLLM (GPU box):")
    print("  " + " ".join(build_vllm_serve_cmd(cfg.server)) + "\n")
    pin = cfg.sweep.prompt_lens[0]
    pout = cfg.sweep.output_lens[0]
    bench = BenchServeConfig(random_input_len=pin, random_output_len=pout,
                             num_prompts=cfg.sweep.num_requests,
                             request_rate=cfg.sweep.request_rates[0] if cfg.sweep.request_rates else float("inf"))
    print("# 2) Reference oracle to cross-check our load-gen:")
    print("  " + " ".join(build_vllm_bench_serve_cmd(bench, cfg.model, cfg.base_url)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpubench", description="Single-GPU LLM inference benchmarking harness for vLLM")
    sub = p.add_subparsers(dest="cmd", required=True)

    sm = sub.add_parser("serve-mock", help="run the GPU-free mock vLLM server")
    sm.add_argument("--host", default="127.0.0.1")
    sm.add_argument("--port", type=int, default=8000)
    sm.add_argument("--ttft-ms", type=float, default=40)
    sm.add_argument("--itl-ms", type=float, default=8)
    sm.add_argument("--jitter", type=float, default=0.15)
    sm.add_argument("--max-concurrency", type=int, default=32)
    sm.set_defaults(func=_cmd_serve_mock)

    r = sub.add_parser("run", help="run a benchmark sweep")
    r.add_argument("--config", required=True)
    r.add_argument("--base-url", default=None)
    r.add_argument("--platform", default=None)
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=_cmd_run)

    rp = sub.add_parser("report", help="(re)build summary + plots from a run dir")
    rp.add_argument("run_dir")
    rp.add_argument("--no-plots", action="store_true")
    rp.set_defaults(func=_cmd_report)

    pl = sub.add_parser("plan", help="Llama-3.1-8B memory/capacity math")
    pl.add_argument("--gpu-mem", type=float, default=None, help="GPU memory in GB (omit for a table)")
    pl.add_argument("--ctx", type=int, default=4096)
    pl.set_defaults(func=_cmd_plan)

    cc = sub.add_parser("crosscheck", help="print vLLM serve + bench-serve oracle commands")
    cc.add_argument("--config", required=True)
    cc.set_defaults(func=_cmd_crosscheck)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
