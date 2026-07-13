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
- **Prefill tends to be compute-bound** — big dense matrix multiplies over all
  prompt tokens, so the GPU's math units (tensor cores) are the bottleneck.
- **Decode at low batch is memory-bandwidth-bound** — it generates one token per
  forward pass, which means streaming the *entire* model's weights out of GPU
  memory for just a couple of FLOPs per byte. The math units sit idle waiting on
  memory. (This is the "roofline": an A100 needs ~200 FLOPs per byte to keep its
  tensor cores fed; decode at batch 1 delivers ~1.)

> If you can say *"a long prompt hurts TTFT, a long answer hurts total time via
> TPOT, and they're bottlenecked by completely different parts of the GPU"* — you
> already understand more than most people who use these models daily.

And the kicker: **batching helps decode** because many requests reuse the same
weights loaded once, raising FLOPs-per-byte back toward the compute regime.

> ### ⚠️ Two traps hiding in the paragraph you just read
>
> **1. "Decode is memory-bound" is a statement about the IDLE GPU.** Batching raises
> decode's arithmetic intensity, so the sentence stops being the whole story exactly
> when the machine gets busy — which is when you care. (It still doesn't reach the
> ridge on an A100; see Part II for why, and why that turns out to matter.)
>
> **2. "Compute-bound ⇒ power near TDP, memory-bound ⇒ power below TDP" is a real
> physical tendency, and I still managed to reason backwards from it.** I read a
> whole-window mean power of ~350 W (diluted by idle ramp and drain), called it
> "below the 400 W cap", and concluded memory-bound. Over the samples where the GPU
> was actually *busy*, the card was at ~385 W. Always ask what window your average
> is over — and remember that power tells you the GPU is working hard, not *which
> unit* is the limit.
>
> Part II is the story of walking into both.

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

## 8. The journey: laptop → Colab → cloud GPU (and why)

> "I can't run this on my Mac — no NVIDIA GPU. So I built it in three hops, same
> code the whole way; only the kitchen gets more real."

| Stage | Hardware | Docker? | Purpose |
|---|---|---|---|
| **Mac** | no NVIDIA GPU | n/a | Write everything; test against a *mock* GPU server so the timing/metrics/plot code is proven before spending a cent. |
| **Colab Pro** | real GPU (T4/L4/A100) | ❌ (already a container) | First *true* numbers cheaply; run vLLM natively (pip, not Docker). |
| **Cloud GPU** (the published run used **NVIDIA Brev**) | single A100 40GB SXM4 | ✅ | The full **Dockerized**, reproducible benchmark for the polished results. |

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

## The result (single A100 40 GB, Llama-3.1-8B, 512-token in / 128-token out)

- **Highest observed throughput ≈ 16.6 req/s (~2,125 output tok/s).** Below ~8 req/s the
  server keeps up (achieved ≈ offered). Push harder and the gain collapses: from 24 to 32
  req/s offered, achieved throughput rises only 5.6% while latency degrades badly.
- **Goodput ≠ throughput.** Under an SLO of TTFT ≤ 1s and TPOT ≤ 50ms, useful capacity
  peaks around **7.2 req/s** — less than half the raw number — and falls to 0.5% SLO
  attainment by 32 req/s. Raw throughput keeps inching up while *useful* throughput
  collapses. The gap is work nobody can use.
- **TPOT is what breaks the SLO.** At 16 and 24 req/s, *every single* SLO failure is a TPOT
  failure; TTFT is still fine. But see below — TPOT is the victim, not the culprit.

## ⚠️ A correction: the bottleneck is prefill compute, not decode bandwidth

**This guide used to say the ceiling was memory-bandwidth-bound decode, citing "GPU util
~90%, power below peak, KV never full." That was wrong, and the published data refutes
it.** Keeping the mistake here on purpose, because how it happened is the lesson.

Run `scripts/analyze_bottleneck.py` against either published open-loop run:

- **Decode *cannot* be the compute wall — and VRAM is what proves it.** This is the cleanest
  argument in the project, and it reuses the 128 KiB/token number from §4.

  Decode's arithmetic intensity is *not* just the batch size; that only holds while weight
  traffic dominates. Weights are read once per step and amortized across the batch, but
  **every sequence's KV must be read separately**, so KV traffic grows with the batch and
  refuses to be amortized — intensity stops climbing.

  Now the part that binds: **KV has to fit in VRAM.** A 40 GB A100 stores ~16 GB of weights.
  Even handing *every* remaining byte (~24 GB) to KV allows at most **~317 sequences** at a
  576-token context, since one sequence's KV costs 75 MB. Feed that ceiling back into the
  formula and decode's intensity tops out at **~125 FLOP/byte** — against an A100 ridge of
  **~201**. At the batch actually observed (~145) it sits at **85**.

  So decode cannot reach the compute roof at any batch this card can physically hold.
  **Whatever pins the tensor cores, it isn't decode.**

  *(Careful: the batch→∞ asymptote is ~203, which is marginally* above *the ridge. So the
  asymptote alone proves nothing — an earlier version of this doc claimed it did and was
  wrong. It is the VRAM cap, not the asymptote, that settles it.)*

- **The power argument was backwards.** The "below peak" figure averaged in the idle ramp
  and drain — it is what `summary.csv` reports, and it is what fooled me. Over rows with GPU
  util ≥90%, rate 32 averages **384.6 W** of the 400 W cap. Memory-bound kernels draw *less*
  power — it is dense GEMM that pins a power cap. (High power alone does not identify the
  limiting unit, but it is not what a GPU idling on memory looks like.)
- **The workload is 512-in / 128-out: four prompt tokens for every generated token.** At
  saturation, forward passes carrying a prompt are ~30% of passes but occupy **~65% of the
  GPU's clock**, at ~62% of the card's bf16 peak. Prompt-bearing intervals average **4.3×**
  the duration of pure-decode intervals *and carry fewer decode emissions*, so a bigger
  decode batch does not explain the gap.

So TPOT degrades not because HBM saturates, but because decode steps are sharing forward
passes with other people's prompts. **Decode is the casualty, not the cause.**

What survives: **decode really is memory-bandwidth-bound in isolation.** At rate 2 the
batch is ~3 and TPOT is 15 ms, implying ~1.07 TB/s of weight streaming — squarely
memory-bound. The error was **scope**: that describes the *idle* GPU. Under load, batching
amortizes the weight read across the batch, and prefill compute takes over.

The general lesson, which is the whole point of §6 below: *the plausible story that
matches what everyone already says is the dangerous one.* "Decode is memory-bound" is true
and famous, and I reached for it instead of doing the arithmetic on my own workload.

**The obvious objection, which you should raise yourself:** prefill passes only hit ~62% of
compute peak and decode passes ~57% of bandwidth peak, so *neither roof is actually hit* —
isn't the real limiter just kernel/scheduler overhead? Partly, yes: there is real headroom
in both. But the claim isn't that prefill *saturates* the tensor cores; it's that prefill
**owns the clock**. 30% of passes consume two-thirds of the wall time, and a pass takes 4×
longer precisely when a prompt is in it. Scheduler overhead doesn't care whether a prompt is
in the batch. Compute does.

(Honest limits: those FLOP/s and GB/s are **modeled** from token counts, not measured —
`nvidia-smi` gives no DRAM-active counter. And the sweep is a fill-and-drain transient, not
a steady overload: at rate 32 all 200 requests are sent by t≈6.2s of a ~12s window. The
directions are solid; the exact numbers are soft. Cleanly separating a compute *roof* from a
mixed scheduler/kernel ceiling needs DCGM counters or a prefill-only/decode-only ablation.)

## The third problem: knowing where the evidence stops

TTFT p99 at rate 8 reads 1,529 ms — seven consecutive requests taking ~2 s to first token.

My first explanation: they were *sent* on schedule, so it must be a **server** stall. My
second: it must be a **client** event-loop stall. **Both were overconfident**, and that is
the lesson worth more than either answer.

What the data does say: the 21 requests *already streaming* at that moment all show their 128
tokens arriving in fewer and fewer SSE chunks (122, 121, 76 … 10, 4), in a staircase ordered
by start time. The harness flagged every one with `tokens_chunks_mismatch`. Something
disrupted the streaming path for everything in flight at once.

What the data does **not** say is which side caused it. `schedule_delay` proves the request
coroutines fired on time — but it is stamped *before* the HTTP client acquires a connection,
so it does not prove the bytes left on time. And chunk coalescing is something the *server*
does when its output queue backs up, which a slow reader, transport backpressure, or a
stalled server output loop all produce identically.

The general point, and it is the same one as §6: **`schedule_delay` rules out late SENDS. It
says nothing about the READ path.** Coordinated omission, a client send stall, and a client
*read* stall look identical on a latency chart, and only the first two are what that metric
tests. Naming the boundary of what you can conclude is part of the job.

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
