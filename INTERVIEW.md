# Interview prep

A study guide for talking about this project. It covers what it is, what each
resume line actually means, the concepts behind them, and the questions people
tend to ask. Read [TEACHING.md](TEACHING.md) alongside it for the from-scratch
explanations.

## What the project is, in one breath

It's a tool that stress-tests a language model running on one GPU and tells you
how much you can get out of that GPU before it falls apart, and why.

Why it matters: running LLMs is expensive, and the cost comes down to how many
requests or tokens you can squeeze out of each GPU. Lots of people can call an
API. Far fewer can look at a model on a GPU and say "it serves about this many
users, at this latency, and here's where and why it breaks." That's what an
inference/serving engineer does, and that's what this shows.

The goal: take one model (Llama-3.1-8B) and one GPU, and produce a defensible
answer to "how many requests per second can it serve, at what latency, and where
and why does it saturate."

How it works end to end: run vLLM, which serves the model behind an OpenAI-style
API. Hit it with a load generator at different load levels, measure latency and
throughput, watch the GPU with nvidia-smi at the same time, and produce reports
and charts. Then check the numbers against vLLM's own official benchmark so you
know they're right.

## The pitch (say this first)

"I built a benchmarking harness for LLM inference on a single GPU. It runs a vLLM
server, drives it with a load generator I wrote, watches the card with nvidia-smi,
and works out how hard you can push one GPU before latency falls apart. The part
I'm proudest of: I cross-checked my numbers against vLLM's official tool, which
caught a real bug in my own load generator, and once I fixed it everything lined
up to within a few percent."

---

## Resume line 1: Dockerized single-GPU vLLM harness, OpenAI-compatible API

Be ready to define every term, because that's where questions come from.

- **vLLM** is a high-performance inference engine. Two key tricks: *continuous
  batching*, where it slots a new request into the running batch token by token
  instead of waiting for the batch to finish, and *PagedAttention*, where it
  stores each request's attention cache (the KV cache) in non-contiguous pages
  like OS virtual memory so it barely wastes GPU memory. Those two are why one
  GPU can serve many users at once.
- **OpenAI-compatible API** means vLLM speaks the same HTTP endpoints as OpenAI
  (`/v1/completions`, `/v1/chat/completions`), so any OpenAI client works and the
  benchmark talks to it over that standard interface.
- **Single-GPU** is deliberate: one GPU removes multi-GPU variables (tensor
  parallelism, etc.) so the measurement is clean and reproducible.
- **Dockerized** means the server runs from the official vLLM image, pinned to a
  version, so the benchmark is reproducible and portable.
- **Harness** is the whole rig: load generator, telemetry sampler, the
  orchestrator that runs the sweep, and the reporter.

Likely questions: How does vLLM serve many requests on one GPU? (continuous
batching + KV cache). Why single GPU? (isolate variables, reproducible, and it's
the cost-per-GPU question). Why pin the version? (a benchmark you can't reproduce
is just an anecdote).

## Resume line 2: sweeps over rate/concurrency/prompt/output, measuring TTFT, TPOT/ITL, throughput, P95/P99, failures

Two halves: what you vary, and what you measure.

What you vary (the knobs):
- **Request rate** is requests per second arriving (open-loop, models real traffic).
- **Concurrency** is how many are in flight at once (closed-loop, models a fixed
  worker pool).
- **Prompt length** is the input size, drives prefill cost and so TTFT.
- **Output length** is how many tokens get generated, drives decode cost and so
  total time.

What you measure (define each):
- **TTFT** (time to first token): how long until the model starts responding.
  Dominated by prefill (processing the whole prompt) plus any queue wait.
- **TPOT / ITL** (time per output token / inter-token latency): how fast tokens
  come after the first. Dominated by decode.
- **Throughput**: tokens/sec and requests/sec the whole server delivers.
- **P95/P99 latency**: the tail. You report the tail, not the average, because the
  average hides the unlucky requests and the tail is what breaks SLAs.
- **Failures**: timeouts, errors, truncated streams, tracked separately so a fast
  error doesn't sneak into the latency numbers.

The deep concepts behind this point:
- **The two phases.** Prefill is compute-bound: a big matrix multiply over all the
  prompt tokens at once, sets TTFT. Decode is memory-bandwidth-bound: one token at
  a time, reloading the weights from memory every step, sets TPOT. Same GPU, two
  different bottlenecks. This is the single most important idea in the field.
- **Open vs closed loop.** Open loop holds arrival rate fixed regardless of whether
  the server keeps up, so it exposes overload and queueing. Closed loop holds
  concurrency fixed, so it shows max sustainable parallelism. The harness does both.
- **Coordinated omission.** A naive load generator slows its own sending when the
  server gets slow, so it never records how late requests really got, and the
  numbers come out flatteringly wrong. I avoid it by deciding every request's send
  time in advance and firing on schedule no matter what.
- **Goodput**: throughput counting only the requests that met an SLO. It's the
  number that actually matters in production, because tokens delivered too late
  are useless.

## Resume line 3: nvidia-smi GPU telemetry correlated with serving performance

What it is: while the load runs, a background thread runs nvidia-smi every couple
hundred milliseconds and records GPU utilization, HBM memory used, and power draw,
timestamped so it lines up with each load level.

Why it matters: the latency numbers tell you *that* it slowed down. The telemetry
tells you *why*. Correlating the two is the diagnosis, and it's what turns "I ran a
benchmark" into "I found the bottleneck."

Concepts to have ready:
- **GPU utilization is misleading.** It only means a kernel was running during the
  sample, not that the whole chip was busy. You can read 90% while most of the
  silicon idles. So you pair it with power-versus-limit and memory.
- **HBM and the KV cache.** The KV cache is per-request attention state that lives
  in GPU memory and grows with context length and concurrency. When it fills, the
  server can't admit more requests, so memory usage shows you a wall.
- **Saturation** is the point where more load stops buying throughput and only
  adds latency. The knee.

In my A100 run this is what let me make the call: util ~89%, but power held around
350W (below the card's limit) and the KV cache never filled past 76%. Util high,
power below max, memory not full. That combination is the signature of being
memory-bandwidth bound on decode, not compute bound and not out of memory.

## Resume line 4: CSV/JSON reports and plots, latency-throughput tradeoffs and saturation

What it does: every request is logged as raw JSONL, aggregated into a
one-row-per-config summary (CSV + JSON), and four charts get rendered: the
latency-throughput knee, TTFT/TPOT vs load, GPU saturation vs load, and goodput
vs load.

The idea: the latency-throughput tradeoff. More load means more throughput but
worse latency, up to a knee. Left of the knee throughput is basically free, right
of it you pay latency for no throughput. The optimal serving config sits just
below the knee, and the number you optimize is goodput, not raw throughput.

Keeping raw data separate from summaries matters: any metric can be recomputed,
and the custom and official numbers get aggregated by the same code, which keeps
the comparison honest.

---

## The strongest story (lead with this)

I built my own load generator, got numbers that looked totally plausible, then
cross-checked against vLLM's official benchmark. The check revealed my TTFT was
about five times too high. It wasn't the server, it was my own client's connection
pool being too small, so requests queued on my side before they ever reached vLLM.
I found the real cause by recording when I *intended* to send versus when I
*actually* sent, fixed the pool, and everything converged to within about 1 to 6
percent of the reference. The corrected data even flipped my diagnosis from "it's
the queue" to "memory-bandwidth-bound decode."

Built it, verified it against a trusted reference, found my own mistake, fixed it,
re-validated. That's what real measurement work looks like, and it's far more
convincing than numbers that just happened to look fine.

## The result in one breath

On a single A100, Llama-3.1-8B (512-token prompts, 128-token outputs) tops out
around 16 requests/sec raw, but useful capacity under an SLO is about 7, the
bottleneck is memory-bandwidth-bound decode rather than queueing or KV-cache, and
all of it is cross-validated against vLLM's official tool to within ~1 to 6%.

## Questions they'll probably ask, with short answers

- **Walk me through a request.** Prefill processes the whole prompt and produces
  the first token (TTFT), then decode generates the rest one token at a time
  (TPOT), all batched with other in-flight requests.
- **How do you know your numbers are right?** Cross-checked against vLLM's official
  benchmark, agreed within ~5%, and the check caught a bug in my own load generator.
- **What's the bottleneck and how would you fix it?** Memory-bandwidth-bound
  decode. The fix is admission control, more replicas, or a higher-bandwidth GPU,
  not more compute. The telemetry is what tells you that.
- **Why percentiles instead of averages?** The tail is what breaks SLAs, and
  averages hide it.
- **What's coordinated omission?** A load generator that slows its own sending when
  the server is slow, so it never records the real tail. I avoid it with
  pre-scheduled send times.
- **TTFT vs TPOT, and why care?** First-token vs between-token latency. They
  bottleneck on different GPU resources (compute for prefill, memory bandwidth for
  decode), so a regression in one points at a different fix than the other.
- **Why single GPU?** Isolate variables, reproducible, and it's the unit-economics
  question (cost per million tokens).
- **What surprised you?** That the client side can be the bottleneck, not the
  server, which is exactly what the cross-check caught.
