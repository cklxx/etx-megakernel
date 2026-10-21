# Per-XCD timeline of one decode step: Qwen3-30B-A3B, bf16, batch 1, context 1024, MI300X (2026-10-11)

This report models where each of the 8 XCDs spends one ETX megakernel decode step.
It uses data that already exists. No GPU ran for this report.

Labels: **[M]** measured, with a `file:line` source. **[E]** estimate, with its arithmetic. **[U]** unknown.
Paths that start with `results_` are under `~/code/etx/` and are outside git. Paths under `docs/xcd-timeline-2026-10-11/` are the artefacts of this report.

## Summary

**Corrections of 2026-10-11, second pass, are in section 8.** They replace the headline metric (item 1) and re-base every per-phase bandwidth share on the phase's own ceiling. The rest of this report is unchanged.

1. **Useful fraction: 32.6% [E].** The step moves 6.1904 GB [E] in a device span of 4412.1 us [M]. The denominator is the span times the achievable 4.3 TB/s: `6.1904 / (4412.1e-6 x 4.3e3) = 6.1904 / 18.972 = 0.326`. Against the 5.3 TB/s peak, the fraction is `6.1904 / 23.384 = 26.5% [E]`. **[Corrected as a headline by section 8.2: this stays as the byte-utility of the span, but the headline is the idle worker time, 1194 us of 8-XCD-equivalent time against a 1138 us gap to 300 tok/s. The per-phase bandwidth shares derived from this denominator are corrected in section 8.1.]**
2. **Wait taxonomy on the critical path, 4412.1 us:** (a) bytes 1974.0 us, 44.7% [E]. (b) dependency handoff 653.1 us, 14.8% [M]. (c) machinery 1785.1 us, 40.5% [E]: 1551.3 us body residual [E] and 233.8 us arrive and queue fetch [M]. (d) launch 0 us inside the span [M]. Host work outside the span is 0.143 ms [E]. Section 2 has the derivation.
3. **L2 and Infinity Cache: probably not a material factor [E].** The weights are 23.75 times the 256 MB Infinity Cache. They stream from memory once per step. The KV cache is the only large data set that can stay in a cache from one step to the next. The largest saving from it is about 15 us [E]. The current build has no cache counters. One measurement can close the question: about 20 minutes, `20/60 x 1.99 = $0.66 [E]`. Section 3 has the details.
4. **Ranked reasons for the remaining 67.4%:** (1) bodies of the latency phases, 1304.4 us, 29.6% [E]. (2) dependency gaps, 886.9 us, 20.1% [M]. (3) streaming phases below full rate, 781.3 us, 17.7% [E]. **[Item (3)'s denominator is corrected in section 8.1: against their own ceilings the streaming phases run at 39.7 to 90.5%, not 37.5 to 90.3%. The 781.3 us time stands; what changes is that it is not unused device bandwidth.]**
5. **Not determined [U]:** the split of each body into bytes and machinery, the split of each handoff gap into arrive, visibility, poll, touch and acquire, per-phase L2 hit rates on the current build, and any Infinity Cache hit count. Section 5 lists the tool changes that would measure them.

## 0. Sources and errors in the brief

**Trace.** The only full raw trace of the current Qwen plan is `results_tiles/tr_all.bin` [M]. It records 60,362 tasks on 304 workers, 8 XCDs x 38 (`results_tiles/tr_all.log:3`). The step is correct: 32/32 tokens equal to HF greedy (`results_tiles/tr_all.log:4`). The GPU was an AMD Instinct MI300X VF with 304 CUs and 8 XCCs (`results_tiles/pmc_all/pmc_agent_info.csv:3`). The ROCm version of that session is [U]. The build flags were `-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH` (`results_tiles/build_all.log:2`). `ETX_VL_NT` was not set, so the non-temporal weight loads were on by default (`examples/vllm_llm/build.sh:55`).

**Plan.** `results_tiles` has no `plan.json`. This report uses `results_vlbest2_2026-10-10/b_all/plan.json`. That choice has three checks:
1. The plan text of both builds is identical. `results_tiles/vlplan_all.txt` and `results_vlbest2_2026-10-10/b_all/vl_plan.txt` have the same SHA256, `046be64d2fd14fe0...` [M].
2. Each of the 60,362 trace rows has the type, coordinates and worker that the plan assigns to that task. There are 0 mismatches [M].
3. Every task executed. Every grid is in static mode, and its domain is the physical XCD from `etx_amdgcn_xcc_id()` [M].

**Trace overhead.** The same build ran untraced in the same session. The device medians were 4.393, 4.405 and 4.416 ms. The wall medians were 4.533, 4.548 and 4.555 ms (`results_tiles/ab1_all_x.log:6`, `ab2_all_x.log:6`, `ab3_all_x.log:6`). The traced run had a 4.413 ms device median and a 4.594 ms wall median (`results_tiles/tr_all.log:443`). The trace adds `4.413 - 4.405 = 0.008 ms` device time [E]. That is 0.2%, so the traced step represents this build.

**Errors in the brief.**
1. The anchor session `results_vlbest2_2026-10-10` has no raw trace. It has only device and wall times: 4.340 ms device and 4.471 ms wall (`results_vlbest2_2026-10-10/etx_r1.log:2118`).
2. The traced build is not the timed anchor build. The DSL tile source differs: the anchor has `readfirstlane` on `d.K` and inactive `ETX_TOPK_KEY32` blocks. The spills differ: 9 VGPRs in the traced build (`results_tiles/build_all.log:4`), 10 in the anchor (`results_vlbest2_2026-10-10/b_all/hipcc.log:11`). The rules forbid comparing absolute milliseconds across sessions. Thus this report compares no time of the trace with the anchor's times.
3. The brief treats the L2 and the Infinity Cache as one cache. They are two levels. Each XCD has a private 4 MB L2. The 256 MB Infinity Cache, also called the MALL, is on the memory side and all XCDs share it (`docs/qwen/BUDGET-2026-10-01.md:144`, `~/code/fleet-mi300x/docs/STATUS.md:81,233`).
4. The `wf/300max` worktree no longer exists. Its work is merged into `main` at `8f44dcc`. This report reads `docs/qwen/PLAN-300TOKS-2026-10-11.md` from `main`.

**Byte model.** The bytes come from `python -m examples.vllm_llm.bw_spec` at 1,055 positions with `--argmax-write-next 1`. The step total is 6.1904 GB [E], the same value as `docs/qwen/FLOORS-2026-10-08.md:93`. It counts each weight once, plus the KV cache and the activations.

## 1. Timeline

### 1.1 What each XCD does in one layer

A layer has 9 phases. The table shows layer 24, which starts 2119.84 us into the step. Window means first task ready to last task done, over all XCDs that run the phase. Gap means the previous phase's last done to this phase's first ready. Source: `docs/xcd-timeline-2026-10-11/account.txt:82-143` [M]. The window and gap columns are differences of those values [E].

| Phase | XCDs | Tasks per XCD | Window us | Gap before us | MB | Task body us, per XCD mean |
|---|---|---|---|---|---|---|
| qkv | 0-7 | 32 | 11.78 | see below | 20.976 | 8.54-9.52 |
| prep | 0-3 | 1 | 4.19 | 1.31 | 0.003 | 3.72-4.00 |
| attn | 0-3 | 5 | 6.24 | 0.64 | 2.161 | 4.97-5.43 |
| reduce | 0-3 | 1 | 6.48 | 0.36 | 0.050 | 5.24-5.44 |
| o_proj | 0-7 | 32 | 5.82 | 1.72 | 16.777 | 5.10-5.33 |
| router | 0-7 | 32 | 6.20 | 2.06 | 0.528 | 4.90-5.51 on XCD 0-3, 0.29-0.30 on XCD 4-7 |
| topk | 0-7 | 1 replica | 8.07 | 1.33 | 0.003 | 7.40-8.04 |
| moe_up | 0-7 | 24 | 18.56 | 0.33 | 50.332 | 16.99-17.38 |
| moe_down | 0-7 | 32 | 12.41 | 0.39 | 25.166 | 9.56-10.21 |

The layer period is `2209.20 - 2119.84 = 89.36 us` [E], from the first qkv_24 ready to the first qkv_25 ready. The gap from moe_down_24 to qkv_25 is 1.47 us. The 9 windows sum to 79.75 us and the 9 gaps, with that last one, sum to 9.61 us. The layer moves 115.996 MB. At 4.3 TB/s that takes `115.996 / 4.3 = 26.98 us` [E], or 30.2% of the period.

Three facts explain the per-XCD shape:
1. **prep, attn and reduce run only on XCD 0-3.** Their grids have 4, 20 and 4 tasks (`results_vlbest2_2026-10-10/b_all/plan.json`, `types` and `descs`) [M]. For about 21 us of each 89 us layer, XCD 4-7 run no task body at all. The sum is `1.31 + 4.19 + 0.64 + 6.24 + 0.36 + 6.48 + 1.72 = 20.94 us` [E].
2. **The router's real rows are all on XCD 0-3.** The router has M = 128 rows on a grid of 256 tasks. That gives `ceil(128 / 256) = 1` row per task (`examples/vllm_llm/model.py:289-291,343`). Tasks 0-127 have the rows and are on XCD 0-3. Tasks 128-255 are on XCD 4-7 and have no row. Their bodies take 0.28 us, against 5.20 us on XCD 0-3 (median over all layers) [M]. The router finish skew between XCDs is about 6 us in every layer (`docs/xcd-timeline-2026-10-11/report.txt:459,468,477`).
3. **The streaming phases use 24 or 32 of the 38 workers of each XCD.** qkv, o_proj, router and moe_down have 256 tasks, moe_up has 192 (`docs/xcd-timeline-2026-10-11/latency.txt:5-13`). No worker gets two tasks of one grid in the whole step [M].

### 1.2 Per-XCD shares over the whole step

Each worker's step is cut into disjoint intervals. The classes of one worker add up to the span. The tool is `examples/vllm_llm/xcd_account.py`. Source: `docs/xcd-timeline-2026-10-11/account.txt:13-22`.

| XCD | body: bytes [E] | body: machinery [E] | dep [M] | handoff [M] | between [M] | none [M] | at least one body runs [M] | mean workers in a body [M] |
|---|---|---|---|---|---|---|---|---|
| 0-3 | 28.24% | 14.40-14.61% | 42.72-42.90% | 7.19-7.37% | 6.10-6.38% | 0.75-0.87% | 83.4-84.0% | 16.20-16.28 of 38 |
| 4-7 | 26.89-26.90% | 10.32-10.55% | 48.75-48.96% | 6.92-7.07% | 5.91-6.06% | 0.85-0.87% | 62.1-62.3% | 14.15-14.23 of 38 |

Class definitions, all per worker:
- **dep**: the worker took the task, and the last producer's body still runs.
- **handoff**: from the last producer's body done, or from the take if that is later, to deps ready. This is the producer's arrive, event visibility, the poll, the touch, the acquire, the context setup and one barrier.
- **between**: from the previous task's body done to the take of the next task. This is the untimed arrive of the previous task and the static-queue fetch. The mean is `81840.9 / 60058 = 1.36 us` per task [E] (`account.txt:5`).
- **bytes**: the phase's bytes at 4.3 TB/s plus the exposed load round trips from the ISA hop table, capped at the body (`docs/qwen/FLOORS-2026-10-08.md:84-88,127-140`). This is an estimate, because nothing inside a body is stamped.
- **machinery**: the rest of the body.
- **none**: after the worker's last task.

The trace stamps three times per task: taken, deps ready and body done (`etx/codegen/kernel.py:264,314,333`). The arrive after the body is not stamped. The table is consistent with `report.txt:443-450`: body 42.6-42.9% and 37.2-37.4%, wait 50.1-50.2% and 55.7-56.0%.

XCD 4-7 have no body at all for 37.7-37.9% of the step. XCD 0-3 have none for 16.0-16.6% [E, 100% minus the "at least one body" column]. XCD 0-3 carry the attention phases and the router rows. All 8 XCDs move almost the same bytes: 787.0-787.4 MB on XCD 0-3, 760.5 MB on XCD 4-7 (`report.txt:443-450`).

Figures: `docs/xcd-timeline-2026-10-11/xcd_layer24.svg` shows layer 24 in 0.25 us bins. `xcd_step.svg` shows the whole step in 5 us bins. Each figure has one row per XCD and shows how many of its 38 workers are in each class. `step.html` is the per-worker timeline of `etx.tools.timeline`, with the critical path outlined.

## 2. Wait taxonomy

### 2.1 On the critical path

The critical path is a chain of 436 tasks. Its bodies and gaps sum exactly to the span: `3525.2 + 886.9 = 4412.1 us` (`report.txt:888`) [M]. All 435 gaps are dependency gaps. The queue gap is 0.0 us. The table cuts the same chain into the four classes of the brief (`account.txt:40-80`).

| Class | Part | us | % of span | Label and arithmetic |
|---|---|---|---|---|
| (a) bytes | phase bytes at 4.3 TB/s | 1439.6 | 32.6 | [E] 6.1904 GB / 4.3 TB/s. Each phase has one task on the path, so the path carries each phase's full floor once. |
| (a) bytes | exposed load round trips | 534.4 | 12.1 | [E] ISA hops x 0.50 us coherent or 0.104 us cached (`docs/qwen/FLOORS-2026-10-08.md:76-88,127-140`). The hop counts are from the round11 ISA, not this build. |
| (b) dependency | handoff after the producer, consumer's worker free | 653.1 | 14.8 | [M] `886.9 - 233.8` |
| (c) machinery | body above (a) | 1551.3 | 35.2 | [E] measured body minus the (a) estimate. 643.3 us is in the streaming phases and 908.0 us in the latency phases (`account.txt:56`). |
| (c) machinery | consumer's worker still in its previous task's arrive and queue fetch | 233.8 | 5.3 | [M] 138 of 435 edges (`account.txt:80`) |
| (d) launch | inside the span | 0 | 0 | [M] one launch per step, queue gap 0.0 |
| | **sum** | **4412.2** | **100** | rounding |

**(d) outside the span.** Host work per step is the wall median minus the device median of the untraced runs. That is `4.548 - 4.405 = 0.143 ms` [E], or 3.1% of the 4.548 ms wall. In the anchor session it is `4.471 - 4.340 = 0.131 ms` (`results_vlbest2_2026-10-10/etx_r1.log:2118`).

**Weakest number.** The split between (a) and (c) is an estimate. Under load a round trip costs 0.69 to 0.73 us, not 0.50 us (`docs/qwen/MODEL-vllm-kernels.md:321`). At that price the coherent hops would cost up to `0.73 / 0.50 = 1.46` times more [E], and (c) would shrink by the same amount. Only intra-body stamps can settle the split (section 5).

### 2.2 The dependency gaps by edge

Source: `account.txt:59-80` [M]. A gap runs from the critical producer's body done to the consumer's deps ready. "Late" means the consumer's worker took the task only after the producer finished, because the worker was still busy.

| Edge | XCD | Edges | Producers | Sum us | Median us | Late edges | Same worker | Median before the take us |
|---|---|---|---|---|---|---|---|---|
| moe_up -> moe_down | same | 48 | 24 | 130.5 | 2.72 | 46 | 15 | 2.08 |
| moe_down -> qkv | cross | 25 | 256 | 97.9 | 3.81 | 25 | 0 | 0.88 |
| o_proj -> router | cross | 33 | 256 | 96.0 | 2.99 | 29 | 0 | 1.47 |
| moe_down -> qkv | same | 22 | 256 | 92.3 | 4.28 | 22 | 16 | 1.96 |
| reduce -> o_proj | cross | 44 | 4 | 88.7 | 2.04 | 0 | 0 | 0 |
| router -> topk | cross | 47 | 256 | 69.6 | 1.44 | 0 | 0 | 0 |
| qkv -> prep | cross | 42 | 256 | 68.9 | 1.64 | 0 | 0 | 0 |
| attn -> reduce | same | 48 | 5 | 53.3 | 1.08 | 0 | 0 | 0 |
| o_proj -> router | same | 15 | 256 | 51.8 | 3.40 | 15 | 13 | 2.08 |
| prep -> attn | same | 48 | 1 | 51.4 | 1.00 | 0 | 0 | 0 |
| topk -> moe_up | same | 48 | 1 | 50.4 | 1.08 | 0 | 0 | 0 |

The seven smaller edges sum to 34.0 us. Two groups stand out:
1. **Free consumer, 1.0 to 2.4 us per edge.** In every edge without a late take, the consumer already waits in phase 1 when the producer finishes. The gap is the producer's arrive, visibility, the poll and the consumer's acquire and barrier. The narrow edges on one XCD, prep -> attn, attn -> reduce and topk -> moe_up, cost 1.00 to 1.08 us median. The edges that cross XCDs into prep, topk and o_proj cost 1.44 to 2.04 us median. This agrees with the 1.00 us per edge in `docs/qwen/BUDGET-2026-10-01.md:161`. It is also near the 1.12 us of one fold read by 8 workgroups (`results_2026-09-27/vlnext/overlap.log:9`).
2. **Busy consumer, 2.7 to 4.3 us per edge.** In moe_up -> moe_down, moe_down -> qkv and o_proj -> router, the consumer's worker still runs a task of the producer grid. This is true for all 138 late edges [M]. Then the gap holds that worker's own arrive and queue fetch: 0.9 to 2.1 us median. After the take it holds the consumer's whole phase 1. For qkv, o_proj, router and lm_head, phase 1 includes the touch hook. Waves 1 to 15 read the task's first weights into L2, and the barrier waits for those loads (`etx/codegen/kernel.py:283-290`). The plan types of these grids carry `touch`. With no wait to hide it, the touch is probably exposed [U]. These five rows sum to 468.5 us, 53% of all gaps [E].

**Check against the 300 tok/s ledger.** The ledger grants the attention boundary `2 x 48 x 1.00 us = 0.096 ms` gross (`docs/qwen/PLAN-300TOKS-2026-10-11.md:82,129`). The trace measures the two edges on the path at `51.4 + 53.3 = 104.7 us` [M]. So the gross allowance matches the trace. The net value is still [U], as the ledger says.

### 2.3 On the whole machine

Worker-time is 304 workers x 4412.1 us = 1,341,278 worker-us. Source: `account.txt:3-11`.

| Class | Share of worker-time | Taxonomy |
|---|---|---|
| bytes | 27.57% [E] | (a) |
| dep + handoff | 45.85% + 7.15% = 53.00% [M] | (b) |
| machinery | 12.46% [E] | (c) |
| between + start | 6.10% + 0.03% = 6.13% [M] | (d), task fetch in place of a kernel launch |
| none | 0.85% [M] | queue empty |

The machine view and the path view answer different questions. On the machine, most workers wait for producers most of the time. On the path, waits are only 20.1% of the span. Most machine waits are workers with nothing to do while a narrow phase runs. They are not on the critical path.

## 3. L2 and Infinity Cache account

**Question: is either cache a factor in the step time?** Answer: probably not [E]. The current build has no cache measurement [U].

**What the step reads [E].**
- The weights are 6.08 GB per step and each weight is read once (`etx/ops/gemv.py:124`). That is `6.08 / 0.256 = 23.75` times the Infinity Cache, and `6.08 / 0.032 = 190` times all 8 L2s together. So no weight can stay in either cache from one step to the next. Each step must stream the weights from HBM.
- The KV cache read is `108032 B x 20 tasks x 48 layers = 103.7 MB` (`docs/xcd-timeline-2026-10-11/bytes.json`). It fits in the 256 MB Infinity Cache. The non-temporal weight loads keep the weights from evicting it (`docs/qwen/BUDGET-2026-10-01.md:144-146,345`).
- The activations per phase are small. The largest input is the 2048-wide hidden vector and the expert rows of a fold.

**What was measured [M], on an older build.** Round 15 measured the L2 per megakernel dispatch, averaged over 1,055 positions (`results_2026-10-01/round15/cache.txt:3-5`, `ntc.txt:1-3`):

| Counter, per dispatch | Without NT | With NT |
|---|---|---|
| `TCC_HIT_sum` | 23,838,256 | 25,925,273 |
| `TCC_MISS_sum` | 50,927,283 | 48,961,909 |
| L2 hit rate | 31.9% | 34.6% |
| `TCC_EA0_RDREQ_DRAM_sum` | 50,867,626 | 48,907,452 |

The round-15 finding: 50.87 M requests for 6.08 GB is 119.5 B per request. So the L2 fetches each weight byte once, with no read amplification (`results_2026-10-01/round15/FINDING.md:55-56`). NT saved 0.106 ms on that build (`FINDING.md:72`) and is on by default now (`examples/vllm_llm/build.sh:52-55`). That was an older build in another session, so its counters describe the current build only approximately. The unfused path has a 26.9% L2 hit rate (`docs/qwen/BUDGET-2026-10-01.md:229`).

**Upper bounds for the remaining cache levers [E].**
1. **KV in the Infinity Cache.** The attention bytes at 4.3 TB/s take 24.1 us on the critical path (`account.txt:48`). At the Infinity Cache rate of 11.9 TB/s (`docs/core/SURVEY-2026-09-29.md:17`) they take `24.1 x 4.3 / 11.9 = 8.7 us`. The largest saving is `24.1 - 8.7 = 15.4 us`, 0.35% of the span. The streaming benchmark shows a smaller effect at 104 MB: 3.43 TB/s cold, 3.66 warm and 3.74 from the Infinity Cache (`results_2026-09-27/vlnext/overlap.log:6`). attn is latency-bound: its phase reaches 7.8% of the bandwidth (`bandwidth.txt:9`) **[17.1% of its own 3.67-XCD ceiling, section 8.1; the latency-bound conclusion is unchanged]**.
2. **The latency phases' loads.** Their activation loads are coherent, so they go past the XCD's L2 by design (`etx/codegen/kernel.py:296-299`). Cross-XCD data must take that path, so a higher L2 hit rate cannot shorten these hops. The round trip under load is 0.69 to 0.73 us at every level, L2, MALL or HBM (`docs/qwen/MODEL-vllm-kernels.md:321`). Queueing sets it, not the cache level.
3. **Cacheable weights.** fleet made the router and kv_b weights cacheable on the same GPU. The result was 18% slower, because the working set passed the 4 MB per-XCD L2 cliff (`~/code/fleet-mi300x/docs/STATUS.md:81,233`). fleet's idle-worker prefetch into the Infinity Cache gave no gain (`STATUS.md:72`).

**What is unknown [U].** The current build's hit rates. The per-phase hit rates: rocprofv3 counts per dispatch, and one dispatch is a whole step. Any Infinity Cache hit count: TCC counters cannot tell an Infinity Cache hit from an HBM read, and no data set has a MALL counter. The `results_tiles` PMC run has only instruction counts, and rocprofv3 crashed at teardown after it wrote them (`results_tiles/pmc_all.log:2-3,6`).

**Proposed measurement: about 20 minutes, `20/60 x 1.99 = $0.66 [E]`.**
1. Run `rocprofv3 --list-avail` on the VM to find any MALL or data-fabric counter. This takes seconds.
2. On the current build, run one counter pass over 64 steps, as `results_tiles/pmc_all.log:2` did: `TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_DRAM_sum`, plus any MALL counter from step 1. One pass took 14 s in `results_tiles` (`status.json`).
3. Build `ETX_VL_NT=0` beside the default build and run three interleaved wall pairs. Each run takes about 23 s (`status.json`).

The setup before the first measurement took 483 s in `results_tiles`: apt 21, etx_env 3, llm_env 98, dl 70, prep 199 and build 92 s (`results_tiles/status.json`). VM provisioning and transfer are [U].

Thresholds, registered before the run:
- The cache is a factor only if the NT pair moves the wall by at least 0.030 ms, the repository's KEEP cut (`docs/qwen/PLAN-300TOKS-2026-10-11.md:157`), or if L2 misses per step exceed `6.1904 GB / 119.5 B = 51.8 M` [E] by more than 10%. A miss count above that means read amplification.
- If neither holds, close the cache question for this plan and do not fund cache work.

## 4. Utilisation and ranked reasons

**The number: 32.6% [E]** of the achievable 4.3 TB/s over the traced device span. Arithmetic: `6.1904 GB / (4412.1 us x 4.3 TB/s) = 0.326`. `etx.tools.timeline` prints the same value (`docs/xcd-timeline-2026-10-11/bandwidth.txt:16`). **[Second pass, section 8: this is the byte-utility of the span, not the headline. The headline is the idle worker time of section 8.2, and the per-phase shares below are re-based in section 8.1.]**

Other denominators give these values [E]:
- the untraced device median of the same build, 4.405 ms: `6.1904 / (4.405 x 4.3) = 32.7%`.
- the anchor build's device median, 4.340 ms: `6.1904 / (4.340 x 4.3) = 33.2%`. Its wall median, 4.471 ms: `32.2%`. `examples/sessions/qwen300.py:441` uses this wall basis.
- the 5.3 TB/s peak over the traced span: `26.5%`. fleet's DeepSeek megakernel reaches 25.9% of peak on the same GPU (`~/code/fleet-mi300x/docs/STATUS.md:90`). That is a different model, so it is context only.

The remaining 67.4% of the span is 2972.5 us. Three disjoint terms explain it. With the 1439.6 us floor they sum to the span: `1439.6 + 1304.4 + 886.9 + 781.3 = 4412.2 us` [E].

1. **Bodies of the latency phases: 1304.4 us, 29.6% [E].** topk, router, prep, attn, reduce, argmax, in_norm and the prologue have critical bodies of 371.6, 240.6, 184.7, 264.3, 258.4, 9.4, 3.8 and 2.2 us (`latency.txt:3-15`). Together they move 0.13 GB, 2.1% of the bytes (`bandwidth.txt:3-15`). The term is their exposed hops, 396.4 us [E], plus their residual machinery, 908.0 us [E] (`account.txt:42-57`). During these phases the machine draws less than 25% of the bandwidth. The low-bandwidth time by phase is topk 401.5, router 328.0, attn 310.0, reduce 308.5 and prep 225.0 us (`report.txt:1328-1338`). The router's 128 tasks with rows and the 4-, 20- and 4-task attention grids keep 300 workers idle, as section 1 shows.
2. **Dependency gaps: 886.9 us, 20.1% [M].** The path has 435 handoffs, a mean of `886.9 / 435 = 2.04 us` [E]. The busy-consumer edges after the wide phases cost the most (section 2.2). In 138 edges the consumer's worker was still busy with its previous task. That part alone is 233.8 us [M].
3. **Streaming phases below full rate: 781.3 us, 17.7% [E].** moe_up, moe_down, qkv, o_proj and lm_head have critical bodies of 835.8, 474.6, 445.2, 277.0 and 157.4 us (`latency.txt:5-14`). Their floors are 561.8, 280.9, 234.1, 187.3 and 144.7 us (`account.txt:42-54`). The phases reach 63.1%, 42.5%, 37.5%, 50.0% and 90.3% of the bandwidth (`bandwidth.txt:4-12`) **[of the whole-device 4.3 TB/s; against their own ceilings they reach 65.9%, 47.0%, 39.7%, 52.4% and 90.5%, section 8.1]**. The excess is 138.0 us of hops before the first weight load and 643.3 us of residual [E]. The residual holds the norm fold, the x staging into LDS, the reductions and the store drains. Per-task start costs matter because each task streams only 0.07 to 0.26 MB [E]: 16.777 MB / 256 for o_proj, 50.332 MB / 192 for moe_up.

**Host work** adds 0.143 ms outside the span [E] (section 2.1).

**Context from the instruction counters.** Over 64 steps the build executed 2.429e10 SALU and 1.755e10 VALU instructions (`results_tiles/pmc_all.log:6`) [M]. The ratio is 1.38 [E]. The event polls run on the scalar unit, so a part of the SALU count is probably the wait loops [U]. Round 15 measured `SQ_WAIT_ANY / SQ_WAVE_CYCLES = 303839357.7 / 353764411.8 = 85.9%` and zero memory-unit stalls (`results_2026-10-01/round15/util.txt:2,4-5`). Waves wait most of the time, and the memory pipeline is not saturated.

## 5. Tooling gap

These changes would turn the [E] and [U] values above into measurements. They are in order of value.

1. **Intra-body stamps.** The bytes and machinery split of every body is an estimate (section 2.1). An optional second trace level could stamp, in the DSL tiles, the first weight load issued, the last weight load returned and the start of the store drain. The imported attention and reduce bodies cannot carry stamps without a change to the imported code [U].
2. **An arrive stamp.** The arrive after `tr[2]` is not stamped (`etx/codegen/kernel.py:333-335`). It is inside the 233.8 us of late edges and inside every gap. One more int64 per trace row would measure it. The row is now 9 int64 (`etx/tools/timeline.py:71-78`, 72 B per task).
3. **A poll-success stamp in phase 1.** It would split each gap into visibility and the rest: touch, acquire, context setup and the barrier. It would show whether the touch is exposed in the busy-consumer edges. A cheaper first test exists: an A/B of the `-DETX_NO_TOUCH` build (`etx/codegen/kernel.py:287`). `docs/core/OPTIMISATIONS.md:27` says the touch was never isolated after `ETX_VL_COH`.
4. **Per-phase counters.** rocprofv3 reports one value per dispatch, and one dispatch is a whole step. No tool gives L2 hits per phase. rocprofv3 thread trace could time each wave's instructions. Its support on this VF is [U].
5. **An Infinity Cache counter.** None of the data sets has one. Section 3 step 1 checks whether the VF exposes one.
6. **Fixed in this report:** `etx.tools.timeline` grouped the low-bandwidth time by phase kind with a regex that kept the `_x` suffix of replicated grids. Each `topk_N_x` was a separate kind, so topk dropped out of the list. topk is in fact the largest low-bandwidth kind at 401.5 us. The fix is at `etx/tools/timeline.py:296`. The same suffix also gave each topk layer its own colour in the HTML page (`timeline.py:497-499`). A test covers the grouping (`tests/test_timeline.py`).
7. **Session fetch.** `tools/session.sh:89` excludes `b_*/`, so a fetched session has no `plan.json` and no `vl_plan.txt`. This report had to borrow the plan of another session and check it row by row. A copy of both files to the session root, as `qwen300.py` already does for `compile.log`, `hipcc.log` and `compiler_version.txt`, would remove that step.
8. **Trace every timed build.** The anchor session timed a build that it never traced. A session that traces the build it times removes the build caveat of section 0. qwen300 does this (section 6).

The new tool `examples/vllm_llm/xcd_account.py` adds the per-worker class partition, the critical-path account in the four classes, the edge table, the layer table, the series and the SVG figures. `tests/test_xcd_account.py` checks on a simulated DeepSeek-V2-Lite schedule that every worker's classes cover the span exactly once and that the path account sums to the span.

## 6. What the qwen300 session adds

`examples/sessions/qwen300.py` is on `main` (merge `9b7f16a`). The brief calls the session running. Its results are not in `~/code/etx/` at the time of writing.

Retry policy (2026-10-11): before rerunning a step, archive its stale log under `log_history/<log stem>/attempt<N>/<filename>` in the same session directory; the canonical log holds the latest full, unfiltered run. This also covers retained compiler logs and `compiler_version.txt`. Certificate and routing directories use `mkdir(exist_ok=True)`, preserving existing contents without deleting or borrowing another step's data. Commands, deadlines, timeouts, ordering and pass/fail rules are unchanged. See `docs/qwen/KEY32-MEASURED-2026-10-11.md` for the retry regression evidence and measured KEY32 decision.

**It adds:**
1. **Two traces of the timed builds.** Job 2 builds KEY32 off and on, checks that both have the same plan (`qwen300.py:377-378`) and traces each to `tr_off.bin` and `tr_on.bin` (`qwen300.py:397`). The same session times both builds in five pinned pairs. So the account of this report can run on the build whose wall time the session reports. That removes error 2 of section 0.
2. **A direct test of the top-k body.** It prints the body change on minus off for topk, attn, reduce, prep, moe_up and moe_down (`qwen300.py:399-405`). topk is the largest latency phase on the path: 371.6 us of body [M].
3. **A MoE comparison.** Job 1 prices vLLM's AITER MoE per layer. In this trace the MoE costs `(835.8 + 474.6) / 48 = 27.30 us` of path body per layer, or `(886.2 + 605.2) / 48 = 31.07 us` with its gaps [E] (`account.txt:42,44`). That trace is not the qwen300 build.

**It does not add:** cache counters, intra-body stamps, an arrive stamp or a poll stamp. After qwen300, sections 3 and 5 stay open.

**Predictions, registered before the qwen300 results [E]:**
1. `tr_off.bin` has the same task layout as `tr_all.bin`. The row check of section 0 confirms it.
2. The `tr_off.bin` span is within 1% of the session's untraced device median for the off build. The trace overhead here was 0.2%.
3. The four path classes of `tr_off.bin` stay within 2 percentage points of section 2.1. A larger change means that the account depends on the build.
4. The KEY32 estimate is at most `(60 - 6) x 0.056 x 48 = 145.2 us` (`docs/qwen/PLAN-300TOKS-2026-10-11.md:53`). So the topk path body in `tr_on.bin` is at least `371.6 - 145.2 = 226.4 us` if the off build matches this trace. topk stays a latency phase.

**Analysis after the fetch.** session.sh does not fetch the plan (section 5, item 7). Run the row check of section 0 against `results_vlbest2_2026-10-10/b_all/plan.json`. If it passes, run the commands of section 7 with `tr_off.bin` and `tr_on.bin`.

## 7. Reproduce

Inputs [M]:

| File | SHA256 |
|---|---|
| `results_vlbest2_2026-10-10/b_all/plan.json` | `b32c3299d4165ba8d68c7b04461d9ed050818d397c7d99082928922836f9f315` |
| `results_tiles/tr_all.bin` (also in `results_tiles.tgz`) | `44e589fd599a94c142424064ee3ec28175dc0e5297bb9e8d3b97d4ca6f476a9c` |
| `results_tiles/vlplan_all.txt` = `results_vlbest2_2026-10-10/b_all/vl_plan.txt` | `046be64d2fd14fe0b82af1e05c0c0c44049d530345b884401de67a1c03be13c1` |

Commands, from the repository root, with `P` the plan and `T` the trace:

```sh
A=docs/xcd-timeline-2026-10-11
P=~/code/etx/results_vlbest2_2026-10-10/b_all/plan.json
T=~/code/etx/results_tiles/tr_all.bin
.venv/bin/python -m examples.vllm_llm.bw_spec $P --config examples/llm/configs/qwen3-30b-a3b.json --context 1055 --argmax-write-next 1 > $A/bytes.json
.venv/bin/python -m etx.tools.timeline $P $T --bytes $A/bytes.json --html $A/step.html --json $A/step.json > $A/report.txt
.venv/bin/python -m etx.tools.timeline $P $T --bytes $A/bytes.json --bandwidth-report --json $A/bandwidth.json > $A/bandwidth.txt
.venv/bin/python -m etx.tools.timeline $P $T --latency-report --json $A/latency.json > $A/latency.txt
.venv/bin/python -m examples.vllm_llm.xcd_account $P $T --bytes $A/bytes.json --out $A
```

Artefacts in `docs/xcd-timeline-2026-10-11/`:

| File | Content |
|---|---|
| `bytes.json` | per-type byte spec, 6.1904 GB per step |
| `report.txt`, `step.json` | timeline report: per type, per XCD, finish skew, critical path, memory demand |
| `step.html` | per-worker timeline page, grouped by XCD, with the critical path |
| `bandwidth.txt`, `bandwidth.json` | phase time against the byte floor |
| `latency.txt`, `latency.json` | phase bodies, waits, live XCDs, dependency gaps, path terms |
| `account.txt`, `account.json` | per-worker class partition, per XCD and phase kind, path account, edge table, layer-24 table |
| `xcd_layer24.svg`, `xcd_series_layer24.json` | workers per XCD by class in layer 24, 0.25 us bins |
| `xcd_step.svg`, `xcd_series_step.json` | the same for the whole step, 5 us bins |

Validation: `.venv/bin/python -m pytest -q tests/test_xcd_account.py tests/test_timeline.py`.

## 8. Corrections of 2026-10-11, second pass

Two choices in this report were wrong. This section corrects both. Nothing above is deleted; the affected lines carry a bracketed pointer here. Both corrections are CPU-only recomputations of the same artefacts; no GPU ran for them.

### 8.1 The denominator: each phase has its own ceiling

The per-phase bandwidth shares above divide each phase's achieved GB/s by the whole device's 4.3 TB/s. No phase runs on all eight XCDs. `latency.txt` already measures the per-phase `XCDs live` column (`docs/xcd-timeline-2026-10-11/latency.txt:5-14`). One XCD's share of the achievable 4.3 TB/s is `4.3 / 8 = 0.5375 TB/s` [E], so a phase's own ceiling is `XCDs live x 0.5375 TB/s`, and its share of that ceiling is the only bandwidth number that means anything for it. A phase on four XCDs could never have reached 4.3 TB/s; comparing it to 4.3 TB/s makes latency-bound phases look like they waste bandwidth they never had.

Corrected table, recomputed from the JSON artefacts at full precision (`latency.json`, `bandwidth.json`; the rounded text columns are `latency.txt:5-15` and `bandwidth.txt:3-15`). "Ceiling" is `XCDs live x 0.5375 TB/s` [E]. "Own share" is `achieved GB/s / ceiling` [E]. The old column, the share of the whole-device 4.3 TB/s, is kept so the size of the correction is visible.

| Phase | XCDs live [M] | Own ceiling TB/s [E] | Achieved GB/s [M] | **Share of own ceiling [E]** | Old: % of 4.3 TB/s [E] |
|---|---|---|---|---|---|
| lm_head | 7.98 | 4.288 | 3881.6 | **90.5%** | 90.3% |
| moe_up | 7.67 | 4.120 | 2713.6 | **65.9%** | 63.1% |
| o_proj | 7.63 | 4.104 | 2152.0 | **52.4%** | 50.0% |
| moe_down | 7.23 | 3.884 | 1826.8 | **47.0%** | 42.5% |
| qkv | 7.57 | 4.067 | 1612.9 | **39.7%** | 37.5% |
| attn | 3.67 | 1.971 | 337.1 | **17.1%** | 7.8% |
| argmax | 1.00 | 0.538 | 32.2 | **6.0%** | 0.7% |
| router | 4.19 | 2.255 | 69.3 | **3.1%** | 1.6% |
| reduce | 3.52 | 1.894 | 6.4 | **0.3%** | 0.1% |
| prep | 3.60 | 1.937 | 0.5 | **0.03%** | 0.01% |
| topk, in_norm, prologue | 7.54, 7.03, 1.00 | 4.053, 3.778, 0.538 | 0.3, 0.9, 0.5 | **0.01%, 0.02%, 0.09%** | 0.01%, 0.02%, 0.01% |

Two properties of this table:

1. The ceiling is still generous to the narrow phases. `XCDs live` counts an XCD as live when one task of the phase runs on it. attn runs 5 tasks per XCD on 4 XCDs and prep and reduce 1 (`account.txt:82-143`, section 1.1), so only a handful of each live XCD's 38 workers issue loads. The own-ceiling shares of the narrow phases are therefore upper bounds; their true ceilings are lower.
2. Consistency consequence [E]: the 1439.6 us floor of section 2 prices every phase's bytes at 4.3 TB/s. Re-based on the own ceilings, the critical-path byte floor becomes `1551.8 us` [E]: the per-phase floors change from 234.1/24.1/0.6/187.3/5.9/561.8/280.9/144.7 us to 247.6/52.6/1.3/196.2/11.3/586.4/311.1/145.1 us for qkv/attn/reduce/o_proj/router/moe_up/moe_down/lm_head, an increase of 112.2 us. Section 2 is not rewritten; the 1439.6 us value stands as the whole-device pricing, and this note is the per-phase-ceiling pricing. Both are estimates; the split into bytes and machinery stays the weakest number either way.

### 8.2 The headline: idle worker time, not bandwidth

The old headline, "achieved 32.6% of the bandwidth", prices the wrong resource. HBM is idle most of the step: known bytes cover only 32.6% of the span's byte capacity. The workers are not idle. The scarce-looking resource, and the one the 300 tok/s question turns on, is worker time. From the per-XCD `any body` shares (`docs/xcd-timeline-2026-10-11/account.txt:15-22`) [M]:

- XCD 0-3 average busy: `(83.75 + 83.43 + 83.96 + 83.72) / 4 = 83.72%` [E].
- XCD 4-7 average busy: `(62.13 + 62.31 + 62.09 + 62.11) / 4 = 62.16%` [E].
- Device busy: `(4 x 83.715 + 4 x 62.16) / 8 = 72.94%`; idle `27.06%` of the span [E].
- In 8-XCD-equivalent time: `0.2706 x 4412.1 = 1194 us` [E] per step.

Side by side with the gap to the target:

| Quantity | Value | Basis |
|---|---|---|
| Idle worker time per step | **1194 us** [E] of 8-XCD-equivalent time | traced build, device span 4412.1 us, `account.txt:15-22` |
| Gap to 300 tok/s | **1138 us** [E] | anchor wall 4.471 ms vs 3.333333 ms (`docs/qwen/PLAN-300TOKS-2026-10-11.md:15`) |
| Same idle share on the anchor's device median | 1174 us [E] | `0.2706 x 4340`; the anchor was never traced, so this transfers the share, it does not measure it |
| Gap after the measured KEY32 cut | 846.7 us [E] | `4.180 - 3.3333` (`docs/qwen/KEY32-MEASURED-2026-10-11.md:66-67`) |

Basis caveat: the 1194 us is measured on the traced build's device span; the 1138 us gap is on the anchor build's wall time. The rules forbid comparing absolute milliseconds across sessions and these are different builds, so the side-by-side is indicative, not a same-build identity. It is the right comparison of kind: both numbers are per step, and the traced build's device span differs from the anchor's by 1.7%.

One precision about what "idle" means here. `any body` marks an XCD busy when at least one of its 38 workers is in a task body. At worker granularity the busy share is lower: bodies cover 40.03% of worker-time (bytes 27.57% + machinery 12.46%, `account.txt:8-9`), and 53.00% of worker-time is dependency wait (`account.txt:6-7`). A worker in a dependency wait has already taken its next task and still occupies its CU; the 27.06% counts XCDs with no body at all, where all 38 workers are waiting or done. Both numbers matter below: the first says how much work could in principle be re-issued, the second says how much of the machine has nothing issued at all.

### 8.3 What changes and what does not

The device has approximately as much idle XCD time, 1194 us [E], as the whole remaining gap to 300 tok/s, 1138 us [E] (and more than the 846.7 us post-KEY32 gap). So the question "can this reach 300 tok/s" is not "is there more bandwidth" — HBM already stands idle for 67.4% of the span — but "is there independent work to put in the idle time". The experiment design for that question is `docs/qwen/XCD-PREFETCH-2026-10-11.md`.

Rows of this report whose conclusion changes:

1. **Summary item 1 and section 4's headline.** "32.6% of the bandwidth" stops being the headline. It remains as the byte-utility of the span. The headline is the 1194 us idle worker time against the 1138 us gap.
2. **Summary item 4, rank 3, and section 4 item 3: "streaming phases below full rate".** The 781.3 us excess stands as a time, but "below full rate" was judged against a ceiling no phase can use. Against their own ceilings the streaming phases run at 39.7 to 90.5%. The excess is machinery and hops sitting on top of the byte time, not device bandwidth left on the table; recovering it is a body-cost problem, not a bandwidth problem.
3. **Section 3 item 1's attn number.** attn reaches 17.1% of its own ceiling, not 7.8% of the device's. The conclusion (latency-bound, the cache lever is worth at most 15.4 us) is unchanged.
4. **The floor readings.** The 1439.6 us floor and the "67.4% is not byte time" figure price bytes at the whole-device rate; the per-phase-ceiling pricing is 1551.8 us [E] (section 8.1). Any plan that adds the old floor to per-phase ceilings double-counts the difference.

Rows that do not change: the whole timeline of section 1 [M]; the wait taxonomy of section 2, which is built from measured times, not from the bandwidth denominator; the dependency edge table; the L2/MALL account of section 3 apart from the one annotated number; the tooling gap of section 5; and sections 6 and 7. In particular the 886.9 us of dependency gaps and the 1304.4 us of latency-phase bodies, the two largest ranked terms, are untouched by both corrections.
