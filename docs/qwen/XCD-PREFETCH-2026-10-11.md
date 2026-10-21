# Cross-layer pipeline and prefetch: the experiment that decides it (2026-10-11)

The owner's mechanism: while one phase runs, an XCD that would otherwise be idle loads the next layer's or next phase's weights early, so the consumer finds them resident; with enough XCDs, hold two phases' weights and switch between them. This document prices that mechanism from the existing trace and designs the smallest session that decides whether it buys time.

CPU-only. No GPU ran for this document and nothing here is a performance result. Labels: **[M]** measured with a `file:line` source, **[E]** estimate with its arithmetic, **[U]** unknown; a `[U]` contributes nothing to any total. bf16 only; no precision change is proposed. The corrected timing model this builds on is `docs/qwen/XCD-TIMELINE-2026-10-11.md` section 8; its artefacts are `docs/xcd-timeline-2026-10-11/` (`account.txt`, `latency.txt`, `bandwidth.txt` and the JSON beside them). Paths starting `results_` are under `~/code/etx/`, outside git. `fleet/` means `~/code/fleet-mi300x/`.

The idle account this starts from [E, arithmetic in section 8.2 of the corrected report]: XCD 0-3 busy 83.72%, XCD 4-7 busy 62.16% (`docs/xcd-timeline-2026-10-11/account.txt:15-22` [M]), device idle 27.06% of the span = **1194 us** of 8-XCD-equivalent time per step, against the **1138 us** wall gap to 300 tok/s (`docs/qwen/PLAN-300TOKS-2026-10-11.md:15`). The mechanism is the proposed way to spend that idle time.

## 1. The capacity account

Per-layer bytes, from the per-phase byte spec divided by 48 (`bandwidth.json`; the text table rounds, `bandwidth.txt:3-15`) [E]:

| Phase | MB per layer | Per consuming XCD | Fits one XCD's 4 MB L2? |
|---|---|---|---|
| qkv | 20.98 | 2.62 MB (8 XCDs) | yes, 66% of it |
| prep | 0.003 | negligible | yes |
| attn (KV read) | 2.16 | 0.54 MB (4 XCDs) | yes |
| reduce | 0.05 | negligible | yes |
| o_proj | 16.78 | 2.10 MB (8 XCDs) | yes, 52% |
| router | 0.53 | 0.13 MB (rows on XCD 0-3) | yes |
| topk | 0.003 | negligible | yes |
| moe_up | 50.33 (selected experts) | 6.29 MB (8 XCDs) | **no** |
| moe_down | 25.17 (selected experts) | 3.15 MB (8 XCDs) | yes, 79% |
| **one whole layer** | **115.996** | 14.5 MB per XCD | **no** |

The layer total 115.996 MB is the timeline report's own layer-24 figure (`docs/qwen/XCD-TIMELINE-2026-10-11.md:56`); the brief's `6.1904 / 48 = 129.0 MB` [E] additionally charges each layer a 1/48 share of lm_head and the singletons. lm_head itself is 622.3 MB once per step (`bandwidth.txt:12`), 2.43 times the MALL: it can never be cache-resident and is excluded from every window below.

**L2 (4 MB per XCD, private).** A prefetch lands in the L2 of the XCD that issues it, so an L2 window only works when prefetcher and consumer are the same XCD. What fits per XCD: qkv, o_proj, moe_down and everything smaller; moe_up (6.29 MB) and a whole layer (14.5 MB) do not. Fleet hit this cliff on the same GPU: making 113 MB of weights cacheable pushed the working set past the 4 MB per-XCD L2 and ran 18% slower (fleet `docs/STATUS.md:81,233`).

**MALL (256 MB, shared by all XCDs).** A prefetch issued by any XCD is visible to every consumer. The resident set competing for it is the KV cache, 103.7 MB per step at context 1024 (`docs/xcd-timeline-2026-10-11/bytes.json`; the KV residency is worth at most 15.4 us, `docs/qwen/XCD-TIMELINE-2026-10-11.md` section 3). Headroom: `256 - 103.7 = 152.3 MB` [E].

- One whole layer: 116.0 MB. Fits, with KV, at `219.7 / 256 = 86%` full [E].
- Two layers: 232.0 MB. Does not fit with KV resident; fits only by evicting the KV cache [E].
- Two phases held at once, the owner's switch-between-phases form: the largest pair, moe_up + moe_down, is `50.33 + 25.17 = 75.5 MB` [E]. Fits easily, with KV.

**The window is one layer, and what governs it is the 256 MB MALL minus the 104 MB resident KV cache.** In L2 the window is at most one moe_down-sized phase per XCD, governed by the 4 MB cliff. One phase-specific limit governs separately: moe_up's *selected* weights are 50.33 MB, but which experts are selected is only known when topk finishes, 1.08 us (median) before moe_up starts (`account.txt:72`). Prefetching all 128 experts instead of the 8 selected is `50.33 x 128 / 8 = 805.3 MB` per layer [E] (`examples/llm/configs/qwen3-30b-a3b.json:1`: 128 experts, 8 per token), 5.3 times the MALL headroom. moe_up is therefore not prefetchable in practice; its window collapses to the 1.08 us topk-to-moe_up gap.

## 2. The traffic account

A prefetch reads the line from HBM once; the consumer then reads it from cache. **The HBM byte count is unchanged: 6.1904 GB per step either way.** The win is exposure time only — the load's latency moves off the critical path into idle time, and the consumer's stream may run at the cache rate instead of the HBM rate — and only if the prefetched line survives until the consumer reads it.

What the repository knows:

- The rate difference is measured, and it is small. `bench/overlap/overlap_bench.hip`'s stream mode on this GPU (`results_2026-09-27/vlnext/overlap.log:2-8`) [M]: at 104 MB, cold 3.43 TB/s, warm 3.66, from the Infinity Cache 3.74 (+9.0%); at 192 MB, cold 3.61, IC 4.23 (+17.2%); at 17.3, 34.6 and 64 MB the cache-resident pass is **slower** than cold (2.32/1.72, 3.03/2.45, 3.30/3.20). Residency helps only at layer-size windows, and there by about 9%.
- Load *latency* does not depend on the level under load: the round trip is 0.73 / 0.71 / 0.69 us at L2 / MALL / HBM with 303 CUs streaming (`docs/qwen/MODEL-vllm-kernels.md:321`) [M]. Queueing sets it, not the cache level. So caching cannot shorten the 534.4 us of exposed load round trips on the critical path (`account.txt:55`), including the 138.0 us of hops before the first weight load in the streaming phases.
- Today's consumer weight loads are non-temporal by default (`examples/vllm_llm/build.sh:52-55`, `etx/ops/gemv.py:124-126`): they are chosen so the 6.08 GB weight stream does not evict the MALL-resident KV cache. Whether an NT consumer load can even *hit* a prefetched MALL line is [U]. If it cannot, the consumer phases must switch to cached loads, which is exactly the change fleet measured at 18% slower when applied broadly (fleet `docs/STATUS.md:81`).
- What the repo does not know [U]: the MALL's eviction policy and any MALL hit counter. The timeline report's section 3 records that no data set has a MALL counter and that TCC counters cannot distinguish an Infinity Cache hit from an HBM read; the L2 is 4 MB per XCD. Eviction survival is therefore decidable today only by timing and by DRAM re-read counts, not by a hit counter.

**The eviction risk, stated plainly:** if a prefetched line is evicted before the consumer reads it, the line is read from DRAM twice — once by the prefetcher, once by the consumer — and the step is slower, not faster, by the prefetch's own traffic. The experiment detects this two ways: (a) `rocprofv3`'s `TCC_EA0_RDREQ_DRAM_sum` per dispatch, compared against the control arm; the baseline is 48.9-50.9 M requests at 119.5 B per request, one fetch per weight byte (`results_2026-10-01/round15/cache.txt:3-5`, `ntc.txt:1-3`, `results_2026-10-01/round15/FINDING.md:55-56`) [M], so any excess over the known 6.19 GB is re-reads; (b) the pass-C rate in the survival arm of the microbenchmark (section 5), which falls back to the cold rate if the window was evicted.

## 3. The independence account

**Compute: no pair of phases is independent.** The decoder layer is a serial chain, qkv -> prep -> attn -> reduce -> o_proj -> router -> topk -> moe_up -> moe_down -> next layer's qkv (the layer order of `account.txt:82-143`), and every consecutive pair carries a measured dependency edge in the trace (`account.txt:62-79`). The brief's example needs a correction here: layer N's `attn` and `reduce` are indeed dependent, but layer N's `moe_down` and layer N+1's `qkv` are **not** independent either — the trace measures 47 such edges and 190.2 us of gaps on them (`account.txt:63,65`) [M], because qkv_{N+1} reads the residual stream that moe_down_N produces (the input/residual norm is folded into qkv). What is independent of everything is the **weight stream**: weights are constants, so any phase's weights can be loaded during any earlier phase, subject to capacity (section 1) and to the addresses being known (moe_up/moe_down only after topk).

The pairs worth overlapping, per layer, with the resource each fills:

| # | Prefetch during | Prefetch target | MB/layer | Window us/layer | Time needed at spare rate [E] | Resource filled |
|---|---|---|---|---|---|---|
| P1 | prep+attn+reduce_N on XCD 0-3 | o_proj_N + router_N | 17.3 | 20.94 (`docs/qwen/XCD-TIMELINE-2026-10-11.md:59`) | 17.3 MB / 2 TB/s = 8.7 us | **idle XCDs 4-7** (no body for ~21 us of every 89 us layer) plus their idle HBM share; attn runs at 17.1% of its own ceiling |
| P2 | moe_up_N | moe_down_N (selected experts; known since topk_N) | 25.2 | 18.3 (`account.txt` layer table: 2176.4-2194.7) | 25.2 / 1.4 = 18.0 us | **unused HBM bandwidth**: moe_up runs at 65.9% of its 4.12 TB/s own ceiling, spare ~1.4 TB/s |
| P3 | moe_down_N | qkv_{N+1} (+ prep/attn inputs) | 21.0 | 12.1 | 21.0 / 2.0 = 10.5 us | **unused HBM bandwidth**: moe_down runs at 47.0% of its 3.88 TB/s own ceiling, spare ~2.0 TB/s |

(P4, cross-step: prefetch layer-0 qkv during lm_head/argmax is worth at most 0.4 us [E] — lm_head already runs at 90.5% of its own ceiling, and the window is one layer in 48. Not pursued.)

P2 and P3 are tight by construction: the prefetch nearly fills the window's spare bandwidth, so any interference with the running phase eats the gain directly. P1 has slack (8.7 us of work in a 20.94 us window) and uses XCDs that are otherwise completely idle; it is the cleanest test of the mechanism. Unused L2/MALL capacity is the enabler for all three, not the resource being filled; the resources being filled are idle XCDs (P1) and idle HBM bandwidth (P2, P3). The "67.4% of the span is not byte time" figure (`docs/qwen/XCD-TIMELINE-2026-10-11.md` summary) is why the HBM has room in P2/P3 at all.

## 4. The expected saving, bounded

The saving, if everything works, is the consumer phases' byte time re-priced at the cache-resident rate. Critical-path byte times [E]: o_proj 187.3, moe_down 280.9, qkv 234.1 us (`account.txt:43,44,47`), sum 702.3 us. The rate deltas are the measured ones of section 2:

- **Upper bound: 120.8 us [E]** = `702.3 x 0.172`, using the most favourable measured residency delta (+17.2% at 192 MB, `overlap.log:8`). That is 2.7% of the 4412.1 us span and 10.6% of the 1138 us gap.
- **Central estimate: 63.2 us [E]** = `702.3 x 0.090`, at the layer-size point (+9.0% at 104 MB, `overlap.log:6`).
- **Size-matched estimate: about 0 [E].** At the actual per-layer window sizes (17-51 MB) the measured cache-resident passes were slower than cold (`overlap.log:3-5`). If the bench's size dependence is real and not a ramp artefact, the mechanism gains nothing at these windows. Arm A of section 5 re-measures exactly these sizes.
- **Hops: ~0 [E from M].** The 138.0 us of hops before the first weight load are load round trips; under load a round trip costs the same at every cache level (`docs/qwen/MODEL-vllm-kernels.md:321`), so residency does not shorten them.

Why the realised value will be lower than even 63 us: (a) the +9% was measured with no co-running consumer stream; during P2/P3 the prefetch competes with the running phase for the same HBM, and the measured price of filling idle bandwidth next to a latency chain is a doubling of its round trip, 1.52 -> 3.08 us (`overlap.log:15-16`, split placement, background 4.31 TB/s) — the latency phases carry 396.4 us of exposed hops [E], so even a 10% slowdown there (~40 us) eats most of the central estimate; (b) the MALL window sits at 86% occupancy with the KV cache, so some eviction is likely and every evicted line is read twice (section 2); (c) if NT consumer loads cannot hit the prefetched lines [U], the consumers must go cacheable, risking the fleet 18% L2-cliff loss; (d) fleet already measured this mechanism's two direct ancestors on the same GPU: idle-worker prefetch into the Infinity Cache gave no gain, and prefetch across the event wait was 4% slower (fleet `docs/STATUS.md:72,30`).

Net: the bound is 0-121 us, central 63 us [E]. Even the bound closes at most ~11% of the remaining gap. This mechanism, if it works, is a KEEP-sized improvement (the bar is 0.030 ms), not a target-closing one.

## 5. The experiment

Preference order per the brief: the existing trace first (done — sections 1-4 are that analysis), then the existing probe `bench/overlap/overlap_bench.hip`, then new device code only where unavoidable. The `examples/vllm_llm/bf16_300/` probes (merged at `a4ca716`) cover fusion and timestamp attribution, not memory residency; they are not the right tool for this question. `overlap_bench.hip` is: its stream mode already measures cold/warm/MALL rates and its fold mode already measures the interference price of filling idle bandwidth.

**Phase 1: the deciding microbenchmark session (existing bench + one new bench mode).**

New code, stated exactly: a `survive` mode in `bench/overlap/overlap_bench.hip` (host loop plus one kernel variant, ~60 lines, reusing `k_stream`). Pass A reads buffer B with cacheable loads (the prefetch); pass B streams a disturber of D MB with NT loads (what the step's own weight traffic does between prefetch and use); pass C re-reads B, once with cacheable loads and once with `__builtin_nontemporal_load` (the NT-consumer variant). This is bench code, locally compilable with the ROCm container check before any provisioning; it is not model device code.

Arms:
- A: existing `stream` cold/warm/mall at the model's window sizes: 17, 25, 51, 116 MB (o_proj, moe_down, moe_up, whole layer), median of 5, as before.
- B: new `survive`, B ∈ {17, 25, 51, 116} MB x D ∈ {0, 64, 128, 256} MB, median of 5, cacheable and NT pass C.
- C: existing `fold`, split placement, foreground sized to the attention shape (5 workgroups per XCD on 4 XCDs), background depth ∈ {0, 1, 2, 4} (0.67-1.71 TB/s covers the P1-P3 prefetch rates).
- D: `rocprofv3 --list-avail` (seconds: is there any MALL/data-fabric counter), then one counter pass over arm B's 116 MB configurations: `TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_DRAM_sum` plus any MALL counter found. One pass took 14 s in `results_tiles` (`status.json`).

Pass rule, registered before the run. Proceed to phase 2 only if all hold:
- S1 (survival): at B = 116 MB, D = 128 MB (one layer of NT traffic between prefetch and use), pass-C rate >= 90% of the D = 0 rate.
- S2 (no re-reads): DRAM read bytes for A+C at D = 128 MB within 5% of D = 0. Excess is eviction; an evicted prefetch makes the step slower (section 2).
- S3 (benefit exists at a real window): primed-vs-cold gain >= 5% at at least one of the three phase-window sizes.
- S4 (NT visibility): the NT pass C keeps at least half of the cacheable pass-C gain. If S4 fails, phase 2 must switch consumer loads to cached, and the fleet 18% loss becomes a priced risk against the KEEP bar.
- I1 (interference): the fold median at background depth 2 grows <= 25% over depth 0. If it grows more, subtract `396.4 us x (growth - 1)` [E] from the section-4 bound; stop if the remainder is below the 0.030 ms KEEP bar.

If S1 or S2 fails, the mechanism is closed: idle-time prefetch does not survive one layer of the step's own traffic, and no model build can fix that.

**Phase 2, conditional on phase 1: the model arm.** New device code is unavoidable here, exactly this and no more:

1. A prefetch task kind: a plan grid whose body issues cacheable reads over an address range and publishes nothing, added in `examples/vllm_llm/model.py`'s plan construction (static mode, so placement is exact). Instances: on XCD 4-7 spanning prep/attn/reduce_N (target o_proj_N + router_N); on all XCDs after topk_N's event, overlapping moe_up_N (target moe_down_N's selected experts — the in-edge on topk is what makes the addresses known); on spare bandwidth during moe_down_N (target qkv_{N+1}). The static-queue ordering rules of `docs/qwen/PLAN-300TOKS-2026-10-11.md` section 2 apply: a resident prefetcher must never sit ahead of a blocked consumer on the same worker.
2. Only if S4 failed: a per-phase NT override. Today NT is global (`ETX_DSL_NT`, `examples/vllm_llm/build.sh:52-55`, `etx/ops/gemv.py:124-126`); add a per-type mask so the three consumer phases issue cached loads. This changes build identity and needs the full correctness gates.
3. Measurement, all existing: `ETX_TRACE_RAW` on both arms analysed by `examples/vllm_llm/xcd_account.py` (per-phase windows, idle fill, edge table), five pinned interleaved wall pairs by the `pairs.py` convention, `rocprofv3` DRAM-read counter pass on both arms, `host --check` on every imported launch and 32/32 tokens against HF greedy.

KEEP rule, registered now, mirroring `docs/qwen/PLAN-300TOKS-2026-10-11.md`'s: KEEP iff every one of the five interleaved pairs is faster, the median gain is >= 0.030 ms and at least twice the spread, 32/32 tokens equal, `host --check` OK for every imported launch, and per-step DRAM read bytes <= 1.02x the control arm (the eviction gate). On a miss the variant stays off and the result is recorded as neutral/slower, not reinterpreted. Prediction from section 4 [E]: the gain lands between 0 and 0.121 ms, central 0.063 ms — so the honest expectation is a marginal pass or a fail, which is exactly why the rule is written before the run.

**Wall time and cost at $1.99/hour [E]:**
- Phase 1: setup 8 min (measured 483 s in `results_tiles/status.json`) [M] + bench build 2 min [E] + arms A-D 10 min [E] = **20 min, $0.66** [E].
- Phase 2 (only on a phase-1 pass): two builds 3 min [E] + five pairs and traces 8 min [E] + checks and counters 4 min [E] = **15 min, $0.50** [E].
- Session cap with both tranches: **40 min, $1.33** [E], with the stop decision at 20 min. One committed `gpu_jobs.py` job file, full unfiltered logs, pinned pairs, deadlines and silence probes per `AGENTS.md` section 4; the VM is deleted at completion.

## 6. The counter-argument, and what settles it

The strongest case that the 1194 us of idle time is not usable:

1. **The idle is a symptom, not a resource.** A worker idles because its next task's inputs do not exist yet. Section 3 shows there is no independent compute to give it: every consecutive phase pair is dependency-linked in the trace, including the pair the brief calls independent (moe_down_N -> qkv_{N+1}, 47 measured edges, `account.txt:63,65`). The only thing that can move into the idle time is weight traffic, and weight traffic's value is bounded by the cache-rate delta, measured at +9% at layer size and negative at window size (`overlap.log:3-8`).
2. **Filling the idle ports has a measured price on the work that matters.** The fold benchmark on this GPU: the latency chain's round trip went 1.52 -> 3.08 us when a background stream filled the idle bandwidth (`overlap.log:15-16`). The latency phases carry 396.4 us of exposed hops [E]; a slowdown there is paid on the critical path, while the prefetch benefit is bounded at 63-121 us [E]. The mechanism can be net negative even when it works.
3. **It has been tried, twice, on this GPU.** Fleet's idle-worker prefetch of routing-independent weights into the Infinity Cache: no gain, 4.84 vs 4.80 ms, with the reason recorded — the per-CU streaming rate (~30 GB/s) caps what idle workers can pull, and the phases that would benefit are latency-bound, not byte-bound (fleet `docs/STATUS.md:72`). Cross-task prefetch across the event wait: 4% slower (fleet `docs/STATUS.md:30`). And ETX's own touch hook is already the within-phase version of this mechanism — waves 1-15 pull the task's first weights into L2 during the dependency wait (`etx/codegen/kernel.py:283-290`) — and its value after `ETX_VL_COH` was never isolated (`docs/core/OPTIMISATIONS.md:27`).
4. **Caching cannot shorten the latency that dominates.** The loaded round trip is level-independent (`docs/qwen/MODEL-vllm-kernels.md:321`), so residency touches only bandwidth exposure, which section 4 bounds at 2.7% of the span in the most favourable case.

The evidence that settles it, in order: the existing `-DETX_NO_TOUCH` A/B (a 20-second existing-switch probe that prices the mechanism's already-shipped within-phase form, `docs/core/OPTIMISATIONS.md:27`); phase 1's S1/S2 (does a prefetched window survive the step's own NT traffic at all) and S3/S4/I1 (is there a benefit, and what does it cost the latency chain); then phase 2's KEEP vote. If S1 or S2 fails, the idle time is provably not usable by this mechanism at this model shape, and the counter-argument holds. If phase 1 passes and phase 2 still misses KEEP, the idle time is usable in principle but worth less than 0.030 ms — and the 300 tok/s question must be answered elsewhere.

## 7. What this document could not determine

- The MALL's eviction policy and any MALL hit counter [U] — the single largest unknown; phase 1 arm D checks whether the VF exposes a counter at all.
- Whether non-temporal consumer loads can hit a prefetched MALL line [U] — decides whether phase 2 needs the per-phase cached-load change.
- The interference price at the exact prefetch rates P1-P3 would use [U] — the existing fold data is at full-tilt background rates.
- Per-phase L2 hit rates on the current build [U], as recorded in the timeline report's section 3.
- The trace is one build of one session; the idle shares are measured on it and transferred, not re-measured, to the anchor (the basis caveat of section 8.2 of the corrected report).
