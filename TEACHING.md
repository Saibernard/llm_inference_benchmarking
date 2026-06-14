# Teaching guide: explaining this project from zero

This is a script you can narrate to someone who has never touched LLM serving —
and the thing that locks the concepts into *your* head. Teaching it is the
fastest way to find the holes in your own understanding (the Feynman technique).

Read it top to bottom; the order is the order that builds understanding.

---

## 0. The one-liner

> "I built a stress-testing rig for AI models. You give it one GPU and a language
> model, and it tells you exactly how many users you can serve, how fast, and the
> precise point where it falls apart — with charts, not guesses."

Why anyone cares:

> "Anyone can *call* an LLM API. Almost nobody can tell you what's happening on
> the GPU underneath, or what it'd cost to run your own. That's the expensive
> skill — and this project makes it measurable."

---

## 1. The master analogy: a restaurant kitchen

Use this for the whole explanation. Every term maps to one thing in the kitchen.

| Kitchen | The real thing |
|---|---|
| The kitchen (one chef station) | The **GPU** |
| The chef's skill / recipe | The **model** (Llama-3.1-8B) |
| Customers placing orders | **Requests** |
| Reading the whole order ticket | **Prefill** (processing the prompt) |
| Plating the dish one bite at a time | **Decode** (generating tokens) |
| Cooking many orders on one griddle | **Continuous batching** |
| The chef's per-dish scratch notes | **KV cache** (uses GPU memory) |
| Kitchen slammed, everyone waits | **Saturation** |

---

## 2. The #1 insight: every response has two phases with opposite bottlenecks

1. **Prefill** — the chef reads your *entire* order at once. All-at-once, works
   the chef's *hands* hard. Sets **TTFT** (time to first token).
2. **Decode** — the chef plates *one bite at a time*, running to the pantry before
   each bite. Sets **TPOT / ITL** (time per output token / inter-token latency).

In real terms:
- **Prefill is compute-bound** — big dense matrix multiplies over all prompt
  tokens, so the GPU's math units (tensor cores) are the bottleneck; power sits
  near the chip's max (TDP).
- **Decode is memory-bandwidth-bound** — it generates one token per forward pass,
  which means streaming the *entire* model's weights out of GPU memory for just a
  couple of FLOPs per byte. The math units sit idle waiting on memory, so power
  stays *below* max. (This is the "roofline": you need ~200+ FLOPs per byte to
  keep tensor cores fed; decode delivers ~1.)

> If you can say *"a long prompt hurts TTFT, a long answer hurts total time via
> TPOT, and they're bottlenecked by completely different parts of the GPU"* — you
> already understand more than most people who use these models daily.

And the kicker: **batching helps decode** because many requests reuse the same
weights loaded once, raising FLOPs-per-byte back toward the compute regime.

---

## 3. What the customer actually feels (the metrics)

Frame each metric as a feeling, then name it:

- **TTFT** — "how long until the first bite arrives." The silence before the
  model starts typing. *Includes* time waiting in the kitchen's queue.
- **TPOT / ITL** — "once it starts, how snappy is each next bite." Smooth stream
  vs stutter.
- **End-to-end latency** — "order to last bite."
- **Throughput** — "how many bites the *whole kitchen* puts out per second across
  *all* customers." Productivity, not one diner's experience.

The two "pro" metrics that earn respect:

- **Percentiles (P95/P99), not averages** — *"don't tell me the average wait, tell
  me the wait for the unluckiest 1 in 100 customers — that's the one who tweets
  about you."* Tails break SLAs; averages hide them.
- **Goodput** — *"don't count dishes served, count dishes served* within the
  promised time*. A kitchen flinging out cold food fast isn't serving anyone."*
  Goodput = throughput that met the SLA. It's the number a product owner cares
  about.

A subtle one worth knowing (it makes you sound sharp):
**mean-ITL and mean-TPOT are the same per request but diverge when you average
across requests.** Pooled ITL is *token-weighted* (a 500-token reply contributes
499 samples and dominates); mean-TPOT is *request-weighted* (every request counts
once). A "100 tokens @ 10ms" vs "2 tokens @ 50ms" pair gives ~30ms
request-weighted but ~10ms token-weighted — a 3× gap. Always say which you mean.

---

## 4. Why one GPU serves many users — and where it breaks

> "The chef doesn't cook orders one at a time. He cooks ~30 on one griddle at
> once, and vLLM can *slide a new order onto the griddle mid-cook* instead of
> waiting for the batch to finish — that's **continuous batching**."

The cost: every in-progress dish needs scratch notes — the **KV cache** — and
those notes pile up on the counter (GPU memory). Run out of counter, kitchen
jams. For Llama-3.1-8B the KV cache is **128 KiB per token** (and note it uses
**8 key/value heads, not 32** — the model shares KV across query heads via
"grouped-query attention," so it's 4× smaller than a naive estimate; getting that
factor wrong is the classic capacity-planning bug).

---

## 5. The experiment (the sweep)

> "To find where the kitchen breaks, I don't ask once — I run a grid of stress
> tests, turning four knobs."

- **Request rate** (customers/sec) · **Concurrency** (served at once) ·
  **Prompt length** (order ticket size) · **Output length** (dish complexity)

For each combo we record every metric above **and film the GPU** (utilization,
memory, power, KV-cache occupancy).

**The headline result — the "knee":**
> "Push more requests: throughput climbs, climbs… then flattens. Past that point,
> more customers don't get more food out — they all just wait longer. That bend is
> the **saturation knee**, and finding it = the exact best operating point for
> this GPU + model."

The chart — throughput vs P99 latency, knee circled, GPU util/KV-cache overlaid —
is the money shot.

---

## 6. The gotcha that separates pros from tutorial-followers: coordinated omission

This is the part to *really* own — a sharp interviewer will probe it.

> "Imagine you only start your stopwatch *when a customer sits down*. If there's a
> line out the door, everyone waiting outside is never timed — so your stats say
> 'everyone served in 2 minutes!', a flat-out lie. Naive load tests do exactly
> this: when the server slows down, they slow down *sending* requests, accidentally
> hiding the slowness."

The fix this project uses: **decide every request's send time in advance and stick
to it**, no matter how backed up the kitchen is. The load generator pre-schedules
arrival times (a Poisson process) and fires on schedule without waiting for prior
replies. It records both *when it meant to send* and *when it actually sent* — the
gap is the backlog a naive test throws away. Get this right and the numbers are
honest; get it wrong and they're fiction.

Sibling concept — **open vs closed loop:**
- **Open loop** = a faucet dripping at a fixed rate regardless of the drain.
  Models real traffic; if the server clogs, you *see* the backup. Use for QPS
  sweeps and finding the knee.
- **Closed loop** = N workers, each grabbing a new task only after finishing the
  last. Models a fixed worker pool; can never show overload because it
  self-throttles. Use for "max throughput at concurrency N."

---

## 7. Reading the GPU's vital signs (telemetry)

> "While stress-testing, I strap a heart-rate monitor on the GPU."

- **Utilization %** — but beware: nvidia-smi "GPU-Util" only means "≥1 kernel was
  running," *not* "the silicon is busy." You can read 100% while 90% of cores
  idle. So back it up with power-vs-TDP and (on datacenter GPUs) DCGM's real
  occupancy/bandwidth counters.
- **HBM memory used** — how full the counter is (KV-cache pressure).
- **Power draw (W) vs TDP** — the tell: prefill pulls power *near* max
  (compute-bound); decode sits *below* max (memory-bound). You can literally see
  the two phases in the power trace.
- **KV-cache occupancy** (from vLLM's own `/metrics`) — when it approaches 100%
  and the wait-queue grows, that's *why* latency exploded — often before raw
  compute even maxes out.

---

## 8. The journey: laptop → Colab → AWS (and why)

> "I can't run this on my Mac — no NVIDIA GPU. So I built it in three hops, same
> code the whole way; only the kitchen gets more real."

| Stage | Hardware | Docker? | Purpose |
|---|---|---|---|
| **Mac** | no NVIDIA GPU | n/a | Write everything; test against a *mock* GPU server so the timing/metrics/plot code is proven before spending a cent. |
| **Colab Pro** | real GPU (T4/L4/A100) | ❌ (already a container) | First *true* numbers cheaply; run vLLM natively (pip, not Docker). |
| **AWS** | single cloud GPU | ✅ | The full **Dockerized**, reproducible benchmark for the polished results. |

All three speak the *identical* OpenAI API + expose the *same* `/metrics` names,
so the benchmarking code is written once and never knows which platform it's on.

---

## 9. Why the numbers are trustworthy (the part that makes it an *instrument*)

- **Cross-checked against an oracle.** We wrote our own load generator to
  understand every metric — but a self-built ruler can be confidently wrong. So we
  point vLLM's official `vllm bench serve` at the *same* server and demand our
  numbers match it *and* the server's own `/metrics` histograms. Three independent
  measurements agreeing = trust; a disagreement = a bug finder.
- **Pinned + seeded + manifested.** Fixed vLLM version, fixed seeds, and an
  environment manifest (GPU, driver, CUDA, model SHA) in every report, so any run
  is reproducible six months later. A benchmark you can't reproduce is an anecdote.
- **Raw data is the truth, summaries are derived.** Every request is logged
  per-line (JSONL); any percentile can be recomputed from it. Aggregation lives in
  exactly one place so the custom and oracle paths are compared apples-to-apples.

---

## The 5 things to make them remember

1. **Two phases:** prefill (think) sets first-token speed; decode (run to pantry)
   sets between-token speed — and they bottleneck on different GPU resources.
2. **One GPU serves many** via continuous batching; the wall is KV-cache memory.
3. **Latency vs throughput is a tradeoff**, and the **knee** is the sweet spot.
4. **Percentiles & goodput**, not averages — the unlucky customer is what matters.
5. **Honest measurement is hard** — coordinated omission will lie to you if you
   let it.

---

## Questions your listener will ask (be ready — this is where you grow)

- *"Why not a bigger GPU?"* → Cost. The point is squeezing max value from one GPU
  — the $/million-tokens story.
- *"Why measure if vLLM prints numbers?"* → To understand each one, and to
  cross-check mine against the official tool. Trust, but verify.
- *"Most important chart?"* → Throughput vs P99 latency with the knee marked —
  "how hard can I push before users suffer."
- *"Where does the GPU run out first?"* → Usually KV-cache memory (counter space),
  which caps concurrency — visible as KV-cache% → ~100% and the wait-queue growing.
- *"What's coordinated omission?"* → (See §6 — if you can explain the
  stopwatch-at-the-door story, you've got it.)

---

# Part II — what I actually found running it for real (and the lessons that only show up then)

Everything above is the theory. Here's what a real A100 run taught me — and the most
valuable lesson is one I didn't expect.

## The result (single A100, Llama-3.1-8B, 512-token in / 128-token out)

- **Throughput ceiling ≈ 13 req/s (~1,600 output tok/s).** Below ~8 req/s the server
  keeps up (achieved ≈ offered). Push harder and **achieved throughput plateaus** — you
  offer 16/24/32 req/s but only ~13 come out. That plateau is the GPU's max useful rate
  for this model/config.
- **The bottleneck was decode, not the queue.** Time-to-first-token stayed low (~160 ms);
  it was **TPOT** (time per output token, ~48 ms) that crept past the 50 ms SLO. The GPU
  telemetry agreed: utilization ~90% but **power sat below peak** and **KV-cache only ~32%
  full** — the classic fingerprint of a **memory-bandwidth-bound decode**, not a compute
  or memory-capacity wall.
- **Goodput ≠ throughput.** Raw throughput kept inching up while *useful* throughput
  (requests meeting the SLO) fell off — the gap is work nobody can use.

## The lesson that matters most: cross-checking caught MY OWN bug

This is the part to really teach — it's what separates a measurement *engineer* from
someone who ran a script.

I built my own load generator, got numbers, and they looked totally plausible — TTFT
"exploding" to ~2.4 s under load. Great story… except it was **wrong**. When I pointed
vLLM's *official* benchmark at the same server:
- throughput, TPOT, and end-to-end latency **matched mine within ~10%** (good — the core
  was sound);
- but **my TTFT was 4.7× higher** than the official tool's (746 ms vs 159 ms).

A self-built ruler can be *confidently* wrong. The only reason I caught it is that I
cross-checked against a trusted reference. **That is the entire point of "trust, but
verify."**

## How I found the cause (a sharp distinction worth teaching)

Two failure modes inflate latency and look identical on a chart, but they're different:
- **Coordinated omission:** your load generator falls behind and sends *late*, so it never
  records how late requests really were.
- **A client-side bottleneck:** you send *on time*, but your own client (connection pool,
  event loop) stalls *after* sending.

The way to tell them apart: **record when you *intended* to send vs when you *actually*
sent.** In my data that gap was **1.5 ms** — so I'd sent on time; it was **not**
coordinated omission. The real culprit was my HTTP **connection pool (80)** being smaller
than the **peak in-flight requests (141)** — so ~60 requests queued *inside my own client*,
inflating TTFT (and, via token buffering, deflating TPOT). I sized the pool to the real
peak and the numbers converged.

**The kicker:** the bug had pointed me at the *wrong diagnosis* — "blame the queue." The
corrected data pointed at **decode / memory-bandwidth**, which is exactly what the GPU
telemetry had been saying all along. The cross-check didn't just fix a number; it fixed my
*understanding*.

## The unglamorous half is the real job

Half of "inference engineering" is fighting the environment, and I hit all of it:
- **Gated model access** — accept the license, mint a token, mind which account it's tied to.
- **Reproducibility is not optional** — an *unpinned* dependency (a `transformers` version
  too new for the pinned vLLM) silently broke the whole run. That one crash is *why* you
  pin versions and write an environment manifest.
- **GPU runtimes reset and wipe state**; editable installs don't load mid-kernel; a model's
  *served name* isn't its *HuggingFace repo id*. None of this is in a tutorial — it's the job.

## If you remember ONE thing

> A benchmark is a **measurement instrument**, and **wrong numbers are worse than no
> numbers.** So you make the math correct, you preserve the raw data, you label anything
> synthetic — and above all you **cross-check against an independent reference**, because
> the most dangerous result is the plausible-looking one that's quietly wrong.
