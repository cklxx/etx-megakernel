# The dependency-gap ledger: every wait on the critical path, priced and classified (2026-10-11)

Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X VF (gfx942), ROCm 7.14.0
(`results_qwen300_2026-10-11/record.log:7`). The analysed trace is the delivered one:
`results_qwen300_2026-10-11/tr_off.bin` (sha256 `5e3a64b30f5b...`), 60,362 tasks on 304 workers,
32/32 tokens equal to HF greedy, device span **4303.5 us**, device median **4.299 ms**, host wall median
**4.471 ms**, 233 tok/s (`results_qwen300_2026-10-11/tr_off.log:5,26,464,466`). CPU only; no GPU ran for
this document.

Labels: **[M]** measured, with a `file:line` source. **[E]** estimate, with its arithmetic. **[U]** unknown;
an [U] contributes nothing to any total. Paths starting with `results_` are under `~/code/etx/` and outside
git. `account.txt` below means `docs/dep-gap-ledger-2026-10-11/account.txt`, the output of
`examples/vllm_llm/xcd_account.py` on the delivered trace.

## 0. Sources, and errors in the brief

**Plan.** The session fetch has no `plan.json` (the known `tools/session.sh` exclusion,
`docs/XCD-TIMELINE-2026-10-11.md:209`). This ledger uses
`results_vlbest2_2026-10-10/b_all/plan.json` (sha256 `b32c3299d416...`), with the same check the XCD
timeline used: every one of the 60,362 trace rows has the type, coordinates and worker the plan assigns to
that task, **0 mismatches [M]**, and the traced build reports the same 60,362 tasks and 436 events
(`tr_off.log:5`; `off_compile.log:2-3`). Unlike the XCD timeline's trace, this trace and the 4.299/4.471 ms
timing are the **same build in the same process**, so the earlier session's build caveat
(`docs/XCD-TIMELINE-2026-10-11.md:30`) does not apply.

**Trace overhead.** No untraced run of this build exists: the paired rounds after the trace were NOT-RUN
(`results_qwen300_2026-10-11/job2_rerun.log:71-87`). The previous trace's overhead measured +0.008 ms
device, 0.2% (`docs/XCD-TIMELINE-2026-10-11.md:26`); treating that as the overhead here is [E].

**Byte model.** The [E] byte/hop split of bodies reuses `docs/xcd-timeline-2026-10-11/bytes.json`
(context 1055; the session packed maxlen 1072, `results_qwen300_2026-10-11/pack.log:2`). No gap number in
this ledger uses bytes; every gap is a difference of two trace stamps.

**Errors in the brief.**
1. The brief's wait taxonomy (span 4412.1 us, handoff 653.1 us, machinery 1551.3 + 233.8 us, host 0.143 ms)
   is measured on `results_tiles/tr_all.bin`, a different build from a different session
   (`docs/XCD-TIMELINE-2026-10-11.md:19-26,88-98`), not on the delivered trace. On the delivered trace the
   same classes are: span 4303.5 us, all critical-path gaps **871.0 us [M]**, of which the
   handoff-after-producer pool is `871.0 - 231.2 = 639.8 us [M]` and the busy-consumer arrive-and-fetch part
   is **231.2 us [M]** (`account.txt:55,80`). The brief's **653.1** is the other trace's `886.9 - 233.8`
   (`docs/xcd-timeline-2026-10-11/account.txt:55,80`). This ledger sums to the delivered trace's 871.0 and
   reports the 639.8 correspondence explicitly; nothing is unattributed.
2. The delivered file changed under the brief. An earlier copy of `tr_off.log` reported span 4295.4, median
   4.297, wall 4.474 (the brief's numbers); the file fetched at 16:00 (`job2_rerun.log:68-70`) reports
   4303.5 / 4.299 / 4.471 (`tr_off.log:26,464,466`). The brief's per-layer body table matches the earlier
   copy. Recomputed from the current log (48 layers, us/step): moe_up **761.4**, moe_down **442.8**, qkv
   **433.6**, topk **383.5**, attn **254.4**, reduce **250.5**, o_proj **248.6**, prep **181.3**, router
   **137.0**; total **3093.0 = 71.9%** of the span (brief: 3087.0). All within 1%; no conclusion moves.
3. "Host work outside the span 0.143 ms" is `tr_all`'s. On the delivered run, wall minus device is
   `4.471 - 4.299 = 0.172 ms [E]` (`tr_off.log:464`), traced, with no untraced pair this session.

The plan rows that touch this pool are unchanged and consistent with this trace: attention-boundary fusion
**0-0.096 ms gross, net [U]** and queue/dispatcher **0 credited** (`docs/PLAN-300TOKS-2026-10-11.md:129,134`;
queue gap on the path measures 0.0 us, `account.txt:55`).

## 1. What one gap is made of

A critical-path gap runs from the critical producer's body-done stamp to the consumer's deps-ready stamp.
The trace stamps three times per task: taken (`etx/codegen/kernel.py:264`), deps ready (`kernel.py:314`,
immediately after the phase-1 barrier at `kernel.py:313`), body done (`kernel.py:333`, after the per-wave
store drain at `kernel.py:324-326` and the barrier at `kernel.py:332`). The producer's **arrive is emitted
after the done stamp** (`kernel.py:335-359`), so it is untimed and lies inside the consumer's gap
(`docs/XCD-TIMELINE-2026-10-11.md:73,204`).

The consumer's wait, all in generated code shared by every row of section 2:

- **Flag poll**: thread 0 of wave 0 loops on the event word with an agent-scope atomic load, `s_sleep(8)`
  backoff, abort check every 256 polls (`etx/runtime/include/etx/primitives.h:19-25`;
  `etx/machine/arch/gfx942.yaml:38,45,92`), emitted per grid at `kernel.py:290-292` from
  `_wait_code` (`kernel.py:77-94`).
- **Acquire**: on poll success, an L1/L2 invalidate (`kernel.py:301-307`; `buffer_inv sc1`,
  `gfx942.yaml:32,44`); coherent tiles skip it (`kernel.py:296-299`).
- **Touch**: in grids whose plan type carries `touch` (qkv, o_proj, router, lm_head), waves 1+ pull the
  task's first weight lines into L2 while wave 0 waits, and the barrier at `kernel.py:313` waits for those
  loads (`kernel.py:283-289`; `docs/XCD-TIMELINE-2026-10-11.md:124`).
- **Barrier**: one `__syncthreads` releases all waves to the body (`kernel.py:313`).

The producer's side of the gap: the release fence (`kernel.py:345-351`) and the event arrive
(`kernel.py:97-122`). A DOMAIN event is one agent-scope `fetch_sub` (`primitives.h:26-28`;
`gfx942.yaml:39`). A DEVICE event with several producers per domain runs the last-arriver flush: per-domain
sub-counter `atomicAdd`, then the last arriver does the global `fetch_sub`, plus an L2 write-back unless the
producers are coherent (`primitives.h:42-58`, `_nf` at `primitives.h:64-70`). Measured one-way signal
latencies: 642 ns same-XCD, 742 ns cross-XCD (`gfx942.yaml:69-70`); the repository's handoff unit price is
1.00 us/edge (`docs/BUDGET-2026-10-01.md:161`).

**Late edges** add, before the consumer is even taken, the consumer worker's own previous task: its phase-3
arrive (`kernel.py:335-359`) and the static-queue fetch with its broadcast barrier
(`kernel.py:443-448`). This "before taken" part is measured per edge (`account.txt:60-80`).

Event identities come from the plan's event table (`results_vlbest2_2026-10-10/b_all/plan.json`, `events`
list): per-layer events are `E_{9L+1}`..`E_{9L+9}` for layer L (layer 0: `E_0`..`E_8`).

## 2. The ledger: every gap on the critical path

The critical path is 436 tasks; its bodies and gaps sum exactly to the span (`account.txt:40`). All 435
edge gaps are dependency gaps; the queue gap is 0.0 us (`account.txt:55,80`). One row per edge class,
aggregated over the 48 layers; sums are [M] (`account.txt:62-79`). "Late" = the consumer's worker was still
busy when the producer finished. "bt" = before-taken sum, the part of the row's gaps spent in the consumer
worker's previous arrive + queue fetch [M].

| # | Producer -> consumer | Event (scope, producers) | Edges (late) | Sum us | Med us | bt us | Consumer waits for | Class |
|---|---|---|---|---|---|---|---|---|
| 1 | moe_up -> moe_down, same XCD | E_8+9L, DOMAIN, 24/word | 48 (48) | **126.8** | 2.60 | 95.7 | XCD-local fold flag: 24 arrives, then poll + acquire + barrier; own previous arrive+fetch first | (ii)+(iii) |
| 2 | moe_down -> qkv, cross | E_9+9L, DEVICE, 256 | 29 (29) | **111.9** | 3.78 | 44.9 | layer-boundary flag (all 256 producers), last-arriver flush, cross-XCD visibility, poll, touch, barrier | (ii)+(iii) |
| 3 | o_proj -> router, cross | E_5+9L, DEVICE, 256 | 33 (18) | **91.2** | 2.67 | 33.9 | same shape as row 2 | (ii)+(iii) |
| 4 | reduce -> o_proj, cross | E_4+9L, DEVICE, 4 | 44 (0) | **83.8** | 1.86 | 0 | one word fed by the 4 reduce partitions; cross-XCD flush, poll, touch, barrier | (iii) |
| 5 | qkv -> prep, cross | E_1+9L, DEVICE, 256 | 44 (0) | **77.4** | 1.78 | 0 | all-256-producer flag, flush, cross visibility, poll, barrier | (iii) |
| 6 | moe_down -> qkv, same XCD | E_9+9L, DEVICE, 256 | 18 (18) | **71.7** | 4.12 | 31.5 | as row 2, producer and consumer on one XCD | (ii)+(iii) |
| 7 | router -> topk, cross | E_6+9L, DEVICE, 256 | 38 (0) | **59.1** | 1.53 | 0 | all-256-producer flag into the 8 topk replicas | (iii) |
| 8 | attn -> reduce, same XCD | E_3+9L, DOMAIN, 5/word | 48 (0) | **57.7** | 1.20 | 0 | per-head word, 5 partition arrives; poll + barrier | **(i)** |
| 9 | prep -> attn, same XCD | E_2+9L, DOMAIN, 1/word | 48 (0) | **52.4** | 1.08 | 0 | per-head word, the single prep owner's arrive; poll + barrier | **(i)** |
| 10 | topk -> moe_up, same XCD | E_7+9L, DOMAIN, 1/word | 48 (0) | **51.2** | 1.08 | 0 | the XCD's own topk replica's arrive; poll + barrier | (iii) |
| 11 | o_proj -> router, same XCD | E_5+9L, DEVICE, 256 | 15 (13) | **44.7** | 3.28 | 24.6 | as row 3 | (ii)+(iii) |
| 12 | router -> topk, same XCD | E_6+9L, DEVICE, 256 | 10 (0) | 14.9 | 1.40 | 0 | as row 7 | (iii) |
| 13 | qkv -> prep, same XCD | E_1+9L, DEVICE, 256 | 4 (0) | 7.4 | 1.96 | 0 | as row 5 | (iii) |
| 14 | reduce -> o_proj, same XCD | E_4+9L, DEVICE, 4 | 4 (0) | 7.3 | 1.88 | 0 | as row 4 | (iii) |
| 15 | prologue -> in_norm, cross | E_pro, DEVICE, 1 | 1 (0) | 4.1 | 4.06 | 0 | the step's first flag; includes the launch-adjacent skew | (iii) |
| 16 | in_norm -> qkv, same XCD | E_0, DOMAIN, 1/word | 1 (0) | 3.6 | 3.56 | 0 | per-XCD word | (iii) |
| 17 | moe_down -> lm_head, cross | E_432, DEVICE, 256 | 1 (1) | 2.6 | 2.55 | 0.6 | as row 2 | (ii)+(iii) |
| 18 | lm_head -> argmax, cross | E_433, DEVICE, 256 | 1 (0) | 1.5 | 1.50 | 0 | all-256-producer flag | (iii) |

**Sum: 869.0 us over 435 edges [M]** (`account.txt:80`; unrounded values), plus the 2.0 us window-start gap
of the prologue (`account.txt:54`, the prologue's own "dep gap"), equals the path's total gap time
**871.0 us [M]** (`account.txt:55`). Of that, the before-taken part is **231.2 us on 127 late edges [M]**
(`account.txt:80`), so the handoff pool in the brief's sense is `871.0 - 231.2 = 639.8 us [M]` — the
delivered trace's counterpart of the brief's 653.1 us (section 0, error 1). The rows above attribute 100%
of the gap time; nothing is unattributed.

Consistency checks: the free same-XCD rows (8, 9, 10) sit at 1.08-1.20 us median against the 1.00 us/edge
unit price (`docs/BUDGET-2026-10-01.md:161`) and the 1.12 us fold measurement
(`results_2026-09-27/vlnext/overlap.log:9`). The two attention-boundary edges measure
`52.4 + 57.7 = 110.1 us [M]` against the plan's gross allowance of `2 x 48 x 1.00 = 96 us`
(`docs/PLAN-300TOKS-2026-10-11.md:82,129`).

## 3. Classification, and what the pool can carry

Classes per the brief: **(i)** removable by protocol, **(ii)** movable only, **(iii)** irreducible.

**(i) Removable by protocol: rows 8 and 9 only. Sum 110.1 us = 0.110 ms gross [M], net [U].**
The one-head owner protocol of `docs/PLAN-300TOKS-2026-10-11.md:64-72` joins prep, the five attention
partitions and the reduce of one head into a single task chain with a mid-body `prep_ready` signal and a
last-arriver-elected reduce. For the path through the owner, the two boundary events, their arrives, the
polls, the acquires and one queue fetch per edge disappear. What does **not** disappear: the store drain
and the publication for the four non-owner partitions, which reappear inside the fused body
(`primitives.h:42-58` still runs; the partials must be visible before the reduce,
`docs/PLAN-300TOKS-2026-10-11.md:68`). Hence gross [M], net [U], exactly as the plan registers
(`docs/PLAN-300TOKS-2026-10-11.md:82`).

**(ii) Movable only: the 231.2 us before-taken [M]** inside rows 1, 2, 3, 6, 11, 17 (95.7 + 44.9 + 33.9 +
31.5 + 24.6 + 0.6). This is the consumer worker's own previous task's arrive and its next static-queue
fetch (`kernel.py:335-359,443-448`). The work is real and reappears wherever the previous task runs. It
cannot be placed off the critical path at these boundaries: the consumer grids are as wide as the producer
grids (256 tasks = 32 per XCD against 38 workers; moe_up 192 = 24 per XCD), the next phase gates on **every**
consumer (event counts [256] and [24] in section 2), and only 6 of 38 workers per XCD are free after a
32-task phase (14 after moe_up's 24) [E arithmetic from the grid sizes, `tr_off.log:30-38`, and
`off_compile.log:2-3`]. So at least 18-26 consumers per XCD must start behind a producer on the same
worker, and the path, needing the last consumer, always pays one busy worker's arrive+fetch. 13 of the 48
moe_up -> moe_down critical consumers even sit on the critical producer's own worker (`account.txt:62`).
The measured attempt to move the fetch earlier, BAR2, made the step **+0.095 ms** slower
(`docs/BUDGET-2026-10-01.md:257`; mechanism `kernel.py:315-331`): the cost reappeared, larger.

**(iii) Irreducible for this schedule: the rest, 871.0 - 110.1 - 231.2 = 529.7 us [E by difference].**
The free-consumer rows (4, 5, 7, 10, 12-16, 18) and the after-take part of the late rows
(`448.9 - 231.2 = 217.7 us [E]`) are the price of a true dependency: the payload must exist and be visible
before the consumer runs, and its production is already paid in the producer's body, not in the gap. The
gap is the minimum notification: one (DOMAIN) or two (DEVICE last-arriver flush) serialized fabric round
trips at 642/742 ns one-way (`gfx942.yaml:69-70`), the poll's backoff granularity (`gfx942.yaml:92`), the
acquire and the barrier. Boundary removal is not available for these rows: qkv -> prep fusion would repeat
prep per KV head and miscompiles (`docs/PLAN-300TOKS-2026-10-11.md:62`); reduce -> o_proj needs all four
partitions (event counts [4]); router -> topk needs all 256 router rows before any selection (counts [256]);
topk -> moe_up is already the XCD-local replica at the 1.08 us floor. The cross-XCD premium over 1.00
(medians 1.40-1.96) is consistent with the second fabric hop of the flush, but the split between that hop,
the poll granularity and the touch barrier is **[U]** — the trace cannot separate them (section 5).

**Can the pool carry the two numbers?**
- Against the plan's **0.096 ms** fusion row (`docs/PLAN-300TOKS-2026-10-11.md:129`): **yes, gross.** The
  trace measures the two edges at 110.1 us, 15% above the 96 us allowance. Whether it carries the row's
  *net* is exactly the registered [U]; the plan's threshold is 1.042 us/layer net
  (`docs/PLAN-300TOKS-2026-10-11.md:84`).
- Against the **0.324 ms** the whole-target decision still needs
  (`docs/PLAN-300TOKS-2026-10-11.md:136,146`): **no.** The honest removable pool is 0.110 ms gross with an
  unmeasured net — a third of 0.324 even at 100% net. Pretending the movable 0.231 ms vanished too gives
  `0.110 + 0.231 = 0.341 ms [E]`, barely above 0.324, and the premise is false: that part reappears by
  construction (work conservation, section 6) and by measurement (BAR2, +0.095 ms). The dependency-gap
  pool cannot carry the 0.324 ms, and no honest accounting of it gets close.

## 4. Per-phase attribution

Each gap is charged to the **consumer** phase (the wait ends when that phase's critical task starts).
Source: `account.txt:41-54` [M]. "Late share" is the fraction of the phase's edges whose consumer was still
busy.

| Consumer phase | Gap us [M] | From edges | Late | Where the wait sits |
|---|---|---|---|---|
| qkv | **187.1** | moe_down 183.6, in_norm 3.6 | 47/48 | at the handoff, before the body; mostly before the take |
| router | **135.8** | o_proj 135.8 | 31/48 | at the handoff; mixed free/busy |
| moe_down | **126.8** | moe_up 126.8 | 48/48 | at the handoff; mostly before the take |
| o_proj | **91.1** | reduce 91.1 | 0/48 | at the handoff, consumer already waiting |
| prep | **84.8** | qkv 84.8 | 0/48 | at the handoff, consumer already waiting |
| topk | **74.0** | router 74.0 | 0/48 | at the handoff, consumer already waiting |
| reduce | **57.7** | attn 57.7 | 0/48 | at the handoff, consumer already waiting |
| attn | **52.4** | prep 52.4 | 0/48 | at the handoff, consumer already waiting |
| moe_up | **51.2** | topk 51.2 | 0/48 | at the handoff, consumer already waiting |
| in_norm | 4.1 | prologue 4.1 | 0/1 | step start |
| lm_head | 2.6 | moe_down 2.6 | 1/1 | at the handoff |
| prologue | 2.0 | window start | — | launch skew |
| argmax | 1.5 | lm_head 1.5 | 0/1 | at the handoff |
| **sum** | **871.0** | | 127/435 | |

No gap sits **inside** a dependent chain. The trace stamps only taken / deps-ready / body-done
(`kernel.py:264,314,333`); a body is one unbroken interval, and the dependent-chain cost (the acquired-load
round trips) is in-body machinery, priced only by the [E] hop model (`examples/vllm_llm/xcd_account.py:41-56`,
0.50 us per coherent trip from `docs/FLOORS-2026-10-08.md:76-88`).

**The prep/reduce warning is confirmed.** The earlier evidence prices prep's and reduce's excess as in-body
dependent-chain cost, with the two boundary transitions priced separately
(`docs/PLAN-300TOKS-2026-10-11.md:130`). On the delivered trace: prep's on-path time is
`84.8 boundary + 184.4 body` (49.0 hops + 135.5 machinery [E]) = 269.2 us, **68.5% in-body**; reduce's is
`57.7 boundary + 254.9 body` (130.0 hops + 124.3 machinery [E]) = 312.6 us, **81.5% in-body**
(`account.txt:48,50`). Their boundary prices are ordinary cross-XCD handoffs (medians 1.78 and 1.86,
rows 4-5), not excess. The reduce/attn hop term is 5 acquired round trips per task
(`xcd_account.py:46-47`). So the excess over the byte floor sits in the body's dependent chain; shortening
the boundary does nothing for it.

## 5. The deciding measurement

The largest [U] rows: **(a)** the net of the 110.1 us fusion gross (does the plan's 0.096 ms row survive?);
**(b)** the after-take composition of the 448.9 us busy pool — how much is the exposed touch
(`kernel.py:287-289`) versus an already-satisfied poll; **(c)** the free cross-XCD premium over 1.00 us —
the second flush hop, the poll backoff granularity, or the touch barrier.

**The existing trace cannot resolve any of them.** Three stamps per task (`kernel.py:264,314,333`); the
producer's arrive is emitted after the done stamp (`kernel.py:335-359`), so its duration is inside the gap,
unmeasured; the wait loop leaves no record of when the poll succeeded (`primitives.h:19-25`); nothing
inside phase 1 is stamped, so the touch is invisible. This is the tooling gap already registered
(`docs/XCD-TIMELINE-2026-10-11.md:204-206`, items 2-3).

**The measurement that turns (b) and (c) into [M]** — one short GPU session, everything CPU-preparable in
advance, per `AGENTS.md` section 4 (one job file, full logs, deadlines):

1. Build the touch-off variant and trace it exactly as the delivered trace was made:
   `ETX_HIPCC_FLAGS="-DETX_NO_TOUCH" bash examples/vllm_llm/build.sh /root/llm/qwen3-30b-a3b`
   (`examples/vllm_llm/build.sh:3`; the touch compiles out at `kernel.py:287`), then
   `stdbuf -oL -eL .venv/bin/python -m examples.vllm_llm.import_certificate --run-build <build> --model
   /root/llm/qwen3-30b-a3b -- --repeat 1` with `ETX_TRACE=1 ETX_TRACE_RAW=<bin>`
   (the delivered command: `results_qwen300_2026-10-11/job2_rerun.log:68`; the env pattern:
   `examples/sessions/qwen300.py:397`). Rerun `examples.vllm_llm.xcd_account` on both traces and diff
   rows 1-3, 6, 11.
2. Add an **arrive stamp** (end of phase 3, one more int64 per trace row; the row is 9 int64,
   `etx/tools/timeline.py:33`) and a **poll-success stamp** in the wait loop, and trace the stock build.
   This splits every row of section 2 into producer arrive / visibility+poll / acquire+barrier.

Expected VM time: setup 483 s [M] (`results_tiles/status.json`, cited at
`docs/XCD-TIMELINE-2026-10-11.md:174`); two device builds about 92 s each, in parallel [M] (same source);
two traced runs about 60-75 s each [E] (the tr_off job finished within its 180 s deadline after one 15 s
progress line, `job2_rerun.log:68-70`; the timed loop itself is 1025 x 4.5 ms, the rest is model load,
`tr_off.log:6-24`). Total about 13-15 min on a staged VM; provisioning and transfer [U]. **Budget: 20 min =
20/60 x $1.99 = $0.66 [E].**

Thresholds, registered now: pursue the touch question only if NO_TOUCH shrinks the five busy rows by at
least 0.5 us per edge median (their pooled stake is `448.9 - 231.2 = 217.7 us [E]` at most); if the stamp
split puts arrive+visibility at 80% or more of the free rows' medians, the free rows are at the protocol
floor and (c) closes.

For **(a)**, the net of the fusion gross, the measurement is the plan's tranche-A head-local protocol
microbenchmark: **15 min = $0.4975 [E]** (`docs/PLAN-300TOKS-2026-10-11.md:144`), threshold 1.042 us/layer
net (`:84`). It needs the CPU-prepared prototype first (8-16 h [E], `:155`); it does not exist today, so it
is not part of the session above.

## 6. The counter-argument: the gaps are structural for this schedule

The strongest case that none of the 871.0 us can be removed, only relabelled:

1. **The chain pays the boundary count, not an above-floor price.** Every free-consumer edge measures
   1.08-1.20 us same-XCD and 1.40-1.96 cross-XCD (section 2), against the measured 1.00 us/edge floor
   (`docs/BUDGET-2026-10-01.md:161`) and fabric one-ways of 642/742 ns (`gfx942.yaml:69-70`). Nine
   boundaries per layer times 48 layers at that floor is most of the 871.0 us no matter what protocol moves
   the bits. Fewer boundaries means fusing bodies, and fusion deletes only the flag round trip: the drain
   and the publication survive inside the fused body for every non-owner consumer
   (`docs/PLAN-300TOKS-2026-10-11.md:66-68`).
2. **The busy half is work conservation under full-barrier events.** Consumer grids are as wide as producer
   grids, every consumer gates the next phase (event counts [256], [24]), and only 6-14 of 38 workers per
   XCD are free at a boundary (section 3, (ii)). Some consumer on the path must start behind a producer on
   the same worker; 13 of 48 critical moe_up -> moe_down consumers sit on the critical producer's own
   worker (`account.txt:62`). The one measured attempt to move the fetch earlier cost +0.095 ms
   (`docs/BUDGET-2026-10-01.md:257`).
3. **The escapes are redesigns, not protocol changes.** Narrowing a consumer grid to the idle pool
   lengthens its bodies proportionally (fewer workers stream the same bytes per phase). Fusing the wide
   pairs moves the wait mid-body: the MoE fold is an XCD-local all-to-all of 24 producers (event E_8,
   counts [24]), so a fused moe_up+moe_down task still polls the same flag between its halves.

The evidence that would settle it: the section-5 session — if the stamp split shows arrive+visibility
dominating the free rows and NO_TOUCH moves the busy rows by less than 0.5 us/edge, the gaps are structural
at this plan's granularity; a same-bodies resimulated schedule (`etx/sim/xcd.py:1-20` exists for exactly
this what-if and runs on CPU) showing no legal queue layout shortens the layer period; and the fusion
microbenchmark's net, where a result near zero closes the last removable row.

## 7. Reproduce

Inputs [M]:

| File | SHA256 |
|---|---|
| `results_qwen300_2026-10-11/tr_off.bin` | `5e3a64b30f5b596ddcec0a7b255e1fe8a3b3b26e5420cd64daf203ed73e7fb7e` |
| `results_qwen300_2026-10-11/tr_off.log` | `df022f2c9e53a0c1785779d6086f7080e64e54b9a959630c8e12cbcda13d7e4f` |
| `results_vlbest2_2026-10-10/b_all/plan.json` | `b32c3299d4165ba8d68c7b04461d9ed050818d397c7d99082928922836f9f315` |

```sh
A=docs/dep-gap-ledger-2026-10-11
P=~/code/etx/results_vlbest2_2026-10-10/b_all/plan.json
T=~/code/etx/results_qwen300_2026-10-11/tr_off.bin
# row check: every trace row matches the plan's type/coord/worker (section 0), 0/60362 mismatches
.venv/bin/python -m examples.vllm_llm.xcd_account $P $T --bytes docs/xcd-timeline-2026-10-11/bytes.json --out $A
```

Artefacts: `account.txt` (per-worker classes, critical-path account, the edge table of section 2, layer-24
table), `account.json`, `xcd_layer24.svg`, `xcd_step.svg` and the two series JSONs. Validation:
`.venv/bin/python -m pytest -q tests/test_xcd_account.py tests/test_timeline.py` (8 passed).
