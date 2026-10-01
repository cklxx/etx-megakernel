# Split-phase scheduling: sharing the weight stream without lockstep (design, 2026-10-01)

Round 33 ended the batching line with a wall: every batched design loses to replication, because batching
oversubscribes the workers (384 MoE tasks on 304) and destroys the ~12 µs cross-request stagger in which each
request fills the other's stalls (`results_2026-10-01/round33/FINDING.md`). Yet the batched dense phases
themselves *win* per phase (qkv 20.9 µs against 25.9 for the replicated pair), because the weights stream once.
The prize is therefore a design that streams each phase's weights once while letting the per-request halves start
and finish at different times. This document designs that, does the arithmetic, and names the deciding
experiments. **No code is changed by this document.**

## 1. The physical mechanism: the second request's reads can hit L2

Facts (docs/BUDGET-2026-10-01.md, RESULTS-2026-10-01.md, round 15/23/30 findings):

- L2 is 4 MB **per XCD** and is exactly what `ETX_VL_COH` stopped invalidating; a warm L2-resident phase runs at
  11+ TB/s against ~2.5 TB/s streamed (hw_bench `restart`).
- The dense `domain_map` sends task coordinate *i* to the same XCD for every request, so **request 1's task *i*
  reads the same weight rows, on the same XCD, as request 0's task *i***. Per layer per XCD the slices are:

  | phase | MB/XCD | fits 4 MB L2? |
  |---|---|---|
  | qkv | 2.50 | yes |
  | o_proj | 2.00 | yes |
  | router | 0.06 | yes |
  | moe_down | 3.00 | yes (marginal) |
  | moe_up | 6.00 | **no** — cyclic stream larger than the cache gives ~0 hits under LRU |
  | lm_head | 78 | no |

- The replicated graph already alternates the requests' launches (`model.py:186-189, 426`), so each worker's
  static queue holds request 0's phase-*n* task immediately followed by request 1's. The measured stagger between
  the halves is ~12 µs, during which that XCD streams almost nothing else (the attention block is tiny and the
  next heavy phase waits on its own event). The lines plausibly survive.
- **The one thing standing directly in the way is our own round-15 optimisation**: `ETX_DSL_NT` marks every DSL
  weight load non-temporal (`etx/ops/gemv.py:102-108`), precisely so the lines do *not* linger. At B=1 that is
  right (+0.106 ms: weights must not evict the MALL-resident KV cache). At B≥2 it forbids the only mechanism by
  which the replica could reuse the stream.

So the design space is not a new scheduler. Replication already gives adjacency-in-time, same-XCD placement, and
per-request stagger. What it does not give is (a) line retention (NT suppresses it) and (b) same-worker
alignment (the round-robin counter in `p5_queues.py:74-76` is keyed by domain and never resets, so request 1's
task *i* lands ~32 workers offset from request 0's — same XCD, different CU; L2 is shared per XCD so this costs
nothing for L2 reuse, but alignment would also give L1/K$ locality and perfectly serialised pair ordering).

## 2. Options

### Option A — replication + per-phase NT policy (no scheduling change at all)

Keep `REQUESTS=N` exactly as shipped. Change only the cache policy: weight loads are non-temporal **iff** the
phase's per-XCD slice exceeds L2 or B=1 (`nt := B == 1 or slice_bytes/XCD > L2`). qkv, o_proj, router and
moe_down keep their lines; moe_up and lm_head stay NT (no reuse is possible there anyway, and they are the
phases whose traffic most threatens the KV cache in the MALL).

- Mechanism required: nothing new. The 12 µs stagger *is* the split-phase: halves share the stream through L2
  and still start/finish apart.
- Oversubscription: impossible — task counts are the replicated ones (max phase 256 < 304 workers at B=2;
  at B=4 the per-request phases still interleave at ≤256).
- Risk: un-NT'd dense weights (2.84 GB/token of the 6.08) re-enter the MALL and may evict the 2×104 MB KV
  cache, taxing `attn` (which runs above HBM rate only because the cache is MALL-resident, worth ≤ ~0.2 ms).
  The deciding experiment measures the *net* automatically.

### Option B — Option A + worker alignment

Reset the round-robin counter per grid (one line in `p5_queues.py`) so request r's task *i* lands on the same
worker for every r. The pair becomes strictly back-to-back in one queue: request 1's half starts the moment its
event fires with the lines still warmest, and K$/L1 locality is added on top of L2. Slightly tightens the
stagger (the pair can no longer drift apart by more than the event skew); the stall-filling overlap between
*different* phases of the two requests is untouched, which is where the measured 12 µs of mutual stall-filling
actually lives (r1_attn waits while r0's MoE streams — different phases, different workers).

### Option C — a split-phase IR construct (only if A/B underdeliver)

One phase, R column groups; group r's tasks wait on request r's event; groups share the worker set without
exceeding it (p2 derives one event set per group — already expressible, since the per-request events exist;
p5 orders group 0's tasks then group r's on the same workers — which is exactly what A+B already produce under
replication; the emitter needs nothing, the per-task wait is unchanged). C's only addition over A+B is *forcing*
adjacency when the natural stagger exceeds L2 lifetime, at the cost of coupling the halves' schedules. Build it
only if the measured reuse under A/B is partial and the traces show eviction between the halves.

## 3. The arithmetic and the honest ceiling

Reusable fraction: qkv + o_proj + router + moe_down = 7.56 MB/XCD/layer → **2.84 GB of the 6.08 GB/token**.
At B=2, request 1's HBM traffic falls from 6.08 to ~3.24 GB; step traffic 12.16 → 9.32 GB. At the ~3.5 TB/s
achievable rate that is **−0.81 ms on a 6.90 ms replicated step → ceiling ≈ 6.1 ms** (vLLM: 5.555). Partial MALL
hits for moe_up and the removed second-stream latency could take a little more; KV-cache eviction could give
back ~0.2. So:

- **B=2: ~6.0–6.3 against 6.90 today.** Clearly below today, clearly not past vLLM.
- **B=4: each extra request saves the same 2.84 GB** (requests 1–3 all read lines request 0 warmed, within the
  same window): 24.3 → 15.8 GB, ~−2.4 ms on 12.49 → **~10.1 ms, ratio to vLLM 1.80× → ~1.45×.** The curve bends;
  it does not cross. Crossing needs the moe_up slice to fit reuse too, i.e. fewer-larger expert tasks are not
  enough — it exceeds L2 per XCD by construction at top-8; only a B-column batched MoE shares it, and round 33
  measured why that loses under lockstep.

Per the directive's bar — "if the ceiling is not clearly below 6.9, do not build" — the ceiling **is** clearly
below 6.9, from a mechanism measured on this machine, at near-zero implementation cost for Options A/B.

## 4. Deciding experiments, ranked by cost

1. **NT off at B=2, zero code (~20 s GPU).** Build the replicated B=2 exactly as today but `ETX_VL_NT=0`
   (the env escape hatch exists). If L2 reuse is real, the step improves immediately — today's queues already
   put the halves on the same XCD ~12 µs apart. This one run tests the entire mechanism *including* the
   KV-eviction risk, since it reports the net. Expected if the mechanism holds: 6.90 → ≤ 6.4 even with NT lost
   on every phase; a null result here kills the design before any code is written.
2. **Per-phase NT policy (Option A proper, ~20 lines, manifest-routed):** NT only for moe_up and lm_head.
   Isolates the KV-cache protection from the reuse. Compare against experiment 1 to price each side.
3. **Worker alignment (Option B, one line in p5):** rr counter keyed by (domain, grid-role) or reset per grid.
   Measures the same-worker increment and any lost-drift cost.
4. **A trace of the winning variant** to confirm where the reuse landed (request-1 bodies of qkv/o_proj/
   moe_down should fall toward their warm rates; moe_up must not move).
5. Only if 1–3 show partial reuse with eviction between halves: Option C.

## 5. Recommendation

Do not build a scheduler. **Option A (per-phase NT policy over today's replication), preceded by the free
experiment 1**, then Option B's one-line alignment. The split-phase construct (C) stays unbuilt unless the
traces show the natural stagger exceeding line lifetime. The expected landing zone is ~6.0–6.3 at B=2 and
~10 at B=4 — a real bend from a cache policy and a queue key, with the lockstep failure modes (oversubscription,
lost stagger) structurally impossible because the task structure is untouched. The B-crossing of vLLM remains
closed off by moe_up's slice exceeding L2; that is the honest residual wall, and it is a property of top-8 over
128 experts at this hidden size, not of ETX.

## Addendum, same day: the deciding experiment refuted the design

Replicated B=2 with `ETX_VL_NT=0` against NT on, three interleaved rounds each, all 32/32 on both requests
(both builds identical resources; the box tree's spill count differed from round 30's — both arms shared it,
so the paired comparison stands):

| | wall ms/step |
|---|---|
| NT on (shipped) | 6.875 / 6.879 / 6.883 |
| NT off | 6.916 / 6.921 / 6.924 |

Removing the non-temporal hint costs 0.04 ms rather than recovering any of the projected 0.81. The missing term
in this document's arithmetic: request 1's replica task waits ~12 µs for its own chain's event, and during that
wait the XCD's other 37 CUs keep streaming request 0's subsequent phases — roughly 5.3 MB through the shared
4 MB L2 — so the lines are evicted before the replica reads them. Cross-request L2 reuse needs the halves within
about a microsecond, which neither today's queues nor the Option A/B orderings provide (the wait, not the queue
position, sets the distance). The options below are therefore not worth building as stated; any revival must
first show a mechanism that closes the halves to ~1 µs without recreating round 33's lockstep. `ETX_VL_NT`
stays on unconditionally: it wins at B=1 (+0.106) and at B=2 (+0.04).
