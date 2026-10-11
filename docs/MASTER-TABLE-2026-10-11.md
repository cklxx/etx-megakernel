# Master decomposition table and optimisation checklist, 2026-10-11

Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X VF (gfx942), ROCm 7.14.0, the kept KEY32-on build
(`3288aa5c`, 32/32 tokens, 97/97 import checks). CPU only: no GPU ran for this document and no number in it is
a new measurement of the device. Every [M] figure is a difference of stamps in an existing trace or a line of
an existing log.

## 0. Basis, labels, sources

**Labels.** **[M]** measured, with a `file:line`. **[E]** estimate, with its arithmetic or model. **[R]**
residual: what is left of an [M] interval after an [E] model is subtracted; it is named, never hidden. **[U]**
unknown; an [U] adds nothing to any total.

**The step.** The trace is `results_qwen300_2026-10-11/tr_on.bin`, the KEY32-on traced run (paths starting
`results_` are under `~/code/etx/`, outside git). Plan: `results_vlbest2_2026-10-10/b_all/plan.json`, row-checked
against this trace with 0 of 60,362 mismatches (`docs/review-pool-analysis-2026-10-11/rebase.txt:1-2`). Byte
spec: `docs/xcd-timeline-2026-10-11/bytes.json`. All decomposition numbers come from one generator,
`docs/master-table-2026-10-11/master.py`, whose output is `master.txt` (cited **MT:line**). The static ISA
census of the DSL tiles is `isa.py` → `isa.txt` (cited **IS:line**).

**Short citations.** ACC = `docs/review-pool-analysis-2026-10-11/account_on.txt`; RB = `.../rebase.txt`;
R = `results_qwen300_2026-10-11/`; K = `etx/codegen/kernel.py`; PR = `etx/runtime/include/etx/primitives.h`;
Y = `etx/machine/arch/gfx942.yaml`; GV = `etx/ops/gemv.py`; AP = `etx/ops/attnprep.py`; TU =
`etx/runtime/include/etx/tile_util.h`; IM = `etx/runtime/include/etx/import.h`; B = the local copy of the build,
`build/qwen300-rocm-on/` (untracked; `dsl_tiles.hip` and `imp_tiles.hip` match the certificate's hashes);
MOE, DEP, XT, PF, REV, PLAN = `docs/MOE-BODY-ANATOMY-`, `DEP-GAP-LEDGER-`, `XCD-TIMELINE-`, `XCD-PREFETCH-`,
`REVIEW-POOL-ANALYSIS-`, `PLAN-300TOKS-2026-10-11.md`.

**Money.** $1.99 per GPU hour; 1 GPU minute = $0.0332.

**What one trace row can and cannot see.** Each task carries three stamps: taken (start of `etx_run_task`,
K:264), deps-ready (after phase 1's wait, acquire, touch and barrier, K:314) and done (after the body, the
all-wave drain and the barrier, K:333). The producer's arrive (K:335-358) and the next queue fetch (K:436-448)
happen after done and are invisible: they sit inside the next gap. Nothing inside a body is stamped. Every
"L3" split of a body below is therefore source and ISA, not time, unless it says [M].

[Part 3 summary tables are in section 1 (added after Parts 1 and 2).]

## 2. Part 1: the atomic decomposition

### 2.1 L0: the step

| Quantity | Value | Source |
|---|---:|---|
| Untraced fused host wall, median of the on-arm rounds | **4.180 ms** [M] | R/job2_run4.log:144 |
| Untraced device time, same rounds | 4.045 ms [M] | R/job2_run4.log:145 |
| Host excess (wall − device) | 0.135 ms [M] | difference of the two |
| Traced device median / traced wall | 4.067 / 4.241 ms [M] | R/tr_on.log:464,466 |
| Trace overhead on the device | 4.067 − 4.045 = **0.022 ms** [M, across runs] | the two lines above |
| Span of the analysed step (= critical path) | **4067.3 us** [M], 60,362 tasks, 436 on the path | MT:1, R/tr_on.log:26 |
| 300 tok/s target | 3.3333 ms wall | 1000/300 |
| **Need** | 4.180 − 3.333 = **0.847 ms** of wall [E] | |
| Device-to-wall transfer of a device cut | 1:1 [M once]: KEY32 cut device 4.279 → 4.045 (0.234) and wall 4.410 → 4.180 (0.230) | R/job2_run4.log:138-145 |

The 0.847 ms need is wall time. Section 1 converts device-trace cuts to wall at 1:1, on the one measured
precedent above.

### 2.2 L1: the 13 roles on the critical path

Every row is the role's share of the 436-task critical path: the bodies of its path tasks plus the gaps charged
to it as the consumer. Columns 3-6 split the body. Bytes = path bytes at 4.3 TB/s, capped at the body; hops =
exposed load round trips from the round11 hop model (`examples/vllm_llm/xcd_account.py:36-56`); machinery =
body − bytes − hops [R].

| Role | Tasks/step, XCDs | Path body [M] | Bytes [E] | Hops [E] | Machinery [R] | Gap [M] | Path total | Per layer: body / gap |
|---|---|---:|---:|---:|---:|---:|---:|---|
| prologue | 1, XCD 0 | 3.9 | 0.0 | 0.4 | 3.5 | 2.6 | 6.5 | once |
| in_norm | 8, 0-7 | 3.7 | 0.0 | 3.6 | 0.1 | 4.4 | 8.1 | once |
| qkv | 12,288, 0-7 | 433.2 | 234.1 | 20.0 | 179.0 | 183.2 | 616.3 | 9.03 / 3.82 |
| prep | 192, 0-3 | 194.8 | 0.0 | 49.0 | 145.8 | 81.1 | 275.9 | 4.06 / 1.69 |
| attn | 960, 0-3 | 259.7 | 24.1 | 130.0 | 105.6 | 52.5 | 312.2 | 5.41 / 1.09 |
| reduce | 192, 0-3 | 254.3 | 0.6 | 130.0 | 123.7 | 56.6 | 310.8 | 5.30 / 1.18 |
| o_proj | 12,288, 0-7 | 263.5 | 187.3 | 20.0 | 56.2 | 96.7 | 360.2 | 5.49 / 2.01 |
| router | 12,288, 0-7 | 265.4 | 5.9 | 49.0 | **210.5** | 129.2 | 394.6 | 5.53 / 2.69 |
| topk | 384, 0-7 | 146.3 | 0.0 | 34.0 | 112.3 | 74.6 | 220.9 | 3.05 / 1.55 |
| moe_up | 9,216, 0-7 | 770.3 | 561.8 | 49.0 | 159.5 | 52.2 | 822.5 | 16.05 / 1.09 |
| moe_down | 12,288, 0-7 | 435.4 | 280.9 | 49.0 | 105.5 | 139.1 | 574.4 | 9.07 / 2.90 |
| lm_head | 256, 0-7 | 151.9 | 144.7 | 0.0 | 7.2 | 1.7 | 153.6 | once |
| argmax | 1, XCD 0 | 9.5 | 0.1 | 0.6 | 8.8 | 1.7 | 11.2 | once |
| **Total** | 60,362 | **3191.8** | **1439.6** | **534.4** | **1217.8** | **875.5** | **4067.3** | |

Source MT:4-19; the totals equal ACC:41-57. The body + gap column closes on the span exactly (MT:19).

**Gap fate** (MT:21-38). The 875.5 us of gaps split, per edge, by when the consumer's worker took its task:

| Gap class | us | Edges | Mean us/edge | What is inside |
|---|---:|---:|---:|---|
| Window start | 2.6 | — | — | launch skew before the prologue |
| Before-taken (consumer worker still busy) [M] | 231.3 | 123 late | 1.88 | the worker's own previous arrive (K:335-358) and queue fetch (K:436-448) |
| Late, after the take [R] | 171.4 | same 123 | 1.39 | the rest of a late edge: wait, touch, acquire, phase-1 barrier |
| Free consumer, into non-touch grids [M] | 325.4 | 243 | 1.34 | producer arrive + visibility + poll + acquire + barrier, no touch |
| Free consumer, into touch grids [R] | 144.7 | 69 | 2.10 | the same plus a possibly exposed touch (K:284-289) |

Late edges sit on three boundaries only: moe_down → qkv (46), o_proj → router (30), moe_up → moe_down (47)
(MT:25,30,33).

### 2.3 L2 and L3: inside each role

Common protocol atoms first; every task of every role runs them. Then one table per role. "Per task" means
one task of the grid; the role's path cost is the slowest-path task per layer.

**P. Task protocol (every task; K and PR).**

| ID | Atom | Source | Cost basis |
|---|---|---|---|
| P1 | Queue fetch: thread 0 loads `static_queue[cursor]` and the descriptor, then `__syncthreads` | K:436-448 | inside before-taken [M] |
| P2 | Stamp `tr[0]` (trace builds only) | K:264 | part of the 0.022 ms trace overhead [M] |
| P3 | Pre-wait argument touch: `etx_imp_touch`/`etx_pre_*`, 112-192 B of scalar loads into the K$ | K:279-282, IM:112-117 | [U], overlaps the wait |
| P4 | Touch hook (qkv, o_proj, router, lm_head only): waves 1.. load the task's first weight trip (32 KB, 4 KB router) into L2 | K:283-289, GV:672-695 | [R], free-into-touch and late after-take rows |
| P5 | Wait: thread 0 polls the event with an agent atomic load, `s_sleep 8` between polls | PR:19-25, Y:38,92 | in the gap [M] |
| P6 | Acquire: `buffer_inv` for imported grids (in_norm, attn, reduce, argmax); none for coherent DSL grids | K:296-307, Y:37 | 44-181 ns [E, Y:73-74] |
| P7 | Phase-1 barrier, then stamp `tr[1]` | K:313-314 | bare 0.0612 us [E, MOE:65] |
| P8 | Body call (the role tables below) | K:323 | body [M] |
| P9 | All-wave drain `ETX_RELEASE_DOMAIN`, barrier, stamp `tr[2]` | K:324-333 | inside the body interval [M] |
| P10 | Arrive. DEVICE events: last-arriver flush with compile-time share 32 (qkv, o_proj, router, moe_down) or share 1 with an L2 write-back per task (reduce). DOMAIN events: release + atomic add (prep, attn, topk, moe_up) | K:335-358, PR:35-45 | inside the next gap [U] |
| P11 | Visibility and poll latency of the arrive: same-XCD 642 ns, cross-XCD 742 ns one-way; DEVICE flag 916-951 ns | Y:69-72 | in the gap; the measured floor is 1.00 us/edge [M, `docs/BUDGET-2026-10-01.md:161`] |

**qkv** (`etx_tile_dsl_gemv_fold`, layer 0 `etx_tile_x_dsl_gemv`; 256 tasks; M=5120, K=2048, 20 rows/task).
Path per layer: body 9.03 = bytes 4.88 [E] + hops 0.42 [E] + machinery 3.73 [R]; gap 3.82, of which 1.55
before-taken [M] and 2.12 after the take [R]. The path task is never block 0 (0/48, MT:67).

| ID | Atom | Source | Per task |
|---|---|---|---|
| QK1 | Arguments: scalar block read, early return for empty range | GV:851-860, 468-481 | ~30 scalar loads, K$-warm after P3 |
| QK2 | Norm fold, sum: 256 threads load the residual and the 8 moe_down slot rows (R=8) together, bf16-round the sum, write LDS, accumulate squares | GV:503-531 | 36,864 B coherent (18 × 8 B per thread) |
| QK3 | Block sum: wave sum + LDS + 2 barriers | GV:554, TU:123-131 | 2 barriers |
| QK4 | Norm weights (cached 16 B per thread), scale, bf16 round, LDS write, barrier | GV:557-580, 617 | 4,096 B cached; 1 barrier |
| QK5 | Block 0 only: store the new residual | GV:532 | 4,096 B coherent |
| QK6 | Weight stream: 16 waves × (2 trips of U=2 + tail of 1) NT 16-B loads | GV:247-257, 129-133 | 81,920 B NT |
| QK7 | Dot: 8 LDS reads + 8 FMA per vector | GV:140-146, 218-226 | 20 rows × 2048 |
| QK8 | Row flush: wave sum (WSUM: 1 bpermute, 1 swizzle, 4 DPP) + lane-0 LDS add | GV:205-214; IS:22-38 | 20 flushes |
| QK9 | Partial clear + barrier; epilogue sums 16 partials per row; barrier; 20 × 2-B coherent stores; barrier | GV:185-186, 266-277 | 3 barriers, 40 B |
| QK10 | Static ISA: 2,174 instructions, 33 vector loads, 11 barriers, 58 VGPR, no spill (local compiler); runtime divisions `sb/spr`, `(S+nw-1)/nw` | IS:6,14,22-38 | [E] |

All 256 tasks recompute the full RMSNorm (QK2-QK4); only block 0 stores it (QK5). Bytes [E] count the weights
once; Q2's 36 KB per task is reread by every task (9.4 MB/layer of coherent reads, `256 × 36,864`) [E].

**prep** (`etx_tile_dsl_attnprep`; 4 tasks on XCD 0-3, one per KV head). Path per layer: body 4.06 = hops 1.02
[E] + machinery 3.04 [R]; gap 1.69, all free [M]. Bytes are nil (2.8 KB stored per task).

| ID | Atom | Source | Per task |
|---|---|---|---|
| PP1 | Arguments (scalar) | AP:86-99 | 112 B |
| PP2 | Waves 0-7 (q heads) and 8 (k head): 2 coherent 8-B loads per active lane (lanes 0-15) | AP:100-116 | 2,304 B coherent |
| PP3 | RMS: wave sum (shuffles) and `rsqrtf` | AP:124-125 | 1 wave sum per wave |
| PP4 | Position: lane-0 coherent scalar load → cos/sin address | AP:78-84, 126 | 1 dependent coherent trip |
| PP5 | Norm weights and cos/sin: cached 2-B loads, issued after R4 | AP:127-140 | 8 + 8 per lane |
| PP6 | RoPE math, pack, store q/k in place | AP:143-145 | 2,304 B coherent |
| PP7 | k and v waves: slot_mapping coherent scalar load, then `slot / BS`, `slot % BS` (64-bit, runtime BS) | AP:141, 154; B/dsl_tiles.hip:204,214 | 1 dependent coherent trip |
| PP8 | The 64-bit divide: ~150 static instructions per site, but a 32-bit bypass branch skips them when the high words are zero, as for slot < 2^32 | IS:59-60; ISA `v_cmp_ne_u32 vcc, 0, ...` before the long path | [E] small |
| PP9 | K cache: 2 × 8-B coherent stores per lane; V cache: wave 9, 2 loads + 8 strided 2-B stores per lane | AP:146-167 | 256 + 256 B; 128 scalar requests |
| PP10 | Occupancy: 10 of 16 waves, 16 of 64 lanes active; no barrier, no LDS | AP:100-111, 22 | [M from source] |

**attn** (`paged_attention_ll4mi_QKV_mfma16_kernel` imported, 20 tasks on XCD 0-3 = 4 KV heads × 5 partitions of
256 tokens). Path per layer: body 5.41 = bytes 0.50 + hops 2.71 [E] + machinery 2.20 [R]; gap 1.09, all free.

| ID | Atom | Source (B = build copy of vLLM source) | Per task |
|---|---|---|---|
| AT1 | Wrapper: one 256-thread slot; waves 4-15 idle but take every barrier | B/imp_tiles.hip:10-19; `etx/importer/adapter.py:53-104` | 12 idle waves |
| AT2 | seq_len load → early exit test | B/imp/attention.hip:360-367 | dependent trip 1 |
| AT3 | Block-table loads (4 per lane) | :429-437 | dependent trip 2 |
| AT4 | Q load to LDS, barrier, LDS to registers | :440-487 | 2 KB; 1 barrier |
| AT5 | K fetch, 16 × 16-B loads per lane | :495-513 | 64 KB (full partition) |
| AT6 | QK: 32 MFMA 16×16×16 per wave | :596-651 | 128 MFMA |
| AT7 | Softmax: max/exp shuffles, LDS exchange, 3 barriers, logits to LDS | :668-766 | 3 barriers |
| AT8 | V block table + V fetch, 16 × 16-B loads per lane | :535-577 | 64 KB |
| AT9 | PV: 32 MFMA per wave; output via LDS (2 barriers); 2 KB `tmp_out` + 64 B stats stores | :769-908 | 2 barriers |
| AT10 | Exit: counted-barrier rounds end (LDS atomics + ≥1 `s_barrier`); every imported barrier is counted (fence + `s_barrier` + fence + LDS add) | IM:56-88 | 7 + ≥1 barriers |
| AT11 | Partition 4 holds only 1-48 tokens, but its loads are clamped and still issued | :433-435, 545-547 | [E] ~1-3 distinct KV blocks |

**reduce** (`paged_attention_ll4mi_reduce_kernel` imported; 4 tasks on XCD 0-3, 8 slots × 128 threads = 8 q heads
per task). Path per layer: body 5.30 = hops 2.71 [E] + machinery 2.58 [R]; gap 1.18, all free.

| ID | Atom | Source | Per task |
|---|---|---|---|
| RD1 | Wrapper: 8 slots, all 16 waves busy | B/imp_tiles.hip:63-72 | — |
| RD2 | seq_len load, number of partitions (5) | B/imp/attention.hip:1440-1441 | dependent trip 1 |
| RD3 | Warp 0: `max_logits` loads + 6 shuffles; `exp_sums` loads, `expf`, 6 shuffles, LDS | :1448-1513 | trips 2-3 |
| RD4 | `tmp_out` loads, 16 × 2 B per thread | :1515-1538 | 10 KB; trip 4 |
| RD5 | Barrier, 16 FMA with the LDS sums, divide, 2-B output stores | :1539-1617 | 1 barrier; 2 KB |
| RD6 | Exit rounds (as AT10); arrive with a per-task L2 write-back (share 1, P10) | IM:73-88 | ≥1 barrier |

**o_proj** (`etx_tile_dsl_gemv`; 256 tasks; M=2048, K=4096, 8 rows/task). Path per layer: body 5.49 = bytes
3.90 + hops 0.42 [E] + machinery 1.17 [R]; gap 2.01, all free, into a touch grid, behind a DEVICE event with
only 4 producers.

| ID | Atom | Source | Per task |
|---|---|---|---|
| OP1 | Stage x (attention output) to LDS: 512 threads × 2 coherent 8-B loads, barrier | GV:393-407 | 8,192 B coherent; 1 barrier |
| OP2 | Weight stream: 16 waves × 2 trips of U=2 NT loads | GV:247-257 | 65,536 B NT |
| OP3 | Dot, 8 row flushes, partial clear/gather, 8 × 2-B stores | GV:185-277 | 3 barriers |
| OP4 | Static ISA identical to qkv's (the fold is a runtime mode of one body) | IS:6-7 | [E] |

**router** (`etx_tile_dsl_gemv_fold`; 256 tasks, M=128, K=2048, 1 row/task). Path per layer: body 5.53 = bytes
0.12 + hops 1.02 [E] + **machinery 4.39 [R]**, the largest machinery residual per layer of any role; gap 2.69
(1.06 before-taken [M], 0.81 after the take [R], 0.83 free into a touch grid [R]).

| ID | Atom | Source | Per task |
|---|---|---|---|
| RT1 | Tasks 128-255 (XCD 4-7) have no row and return at once | GV:703; XT:60 | 0.28 us [M, XT:60] |
| RT2 | Norm fold on tasks 0-127: residual + o_proj output (R=1), 4 coherent 8-B loads per thread, sum, LDS | GV:503-531 | 8,192 B coherent |
| RT3 | Block sum (2 barriers), cached norm weights (issued after the sum), scale, LDS, barrier | GV:554-617 | 4,096 B cached; 3 barriers |
| RT4 | Block 0 only: store resid2 and the normed copy `ox` | GV:532, 581 | 8,192 B coherent |
| RT5 | Weights: waves 0-3 issue one NT 16-B load each; waves 4-15 idle | GV:247-257 | 4,096 B |
| RT6 | One row: dot, flush, clear/gather (3 barriers), one 2-B store | GV:185-277 | 3 barriers |

Working tasks have a median body of 5.44 us; block 0, with its 8 KB of extra stores, 5.84 us, yet it is never
the path task (0/48) [M, MT:72]. So the router's 5.5 us is the latency of one task's serial chain (RT2 → RT3 → RT5
→ RT6 → P9), not its bytes and not the block-0 stores.

**topk** (`etx_tile_rep_dsl_topk`, KEY32; 8 replicas, one per XCD, wave 0 only). Path per layer: body 3.05 =
hops 0.71 [E] + machinery 2.34 [R]; gap 1.55, all free.

| ID | Atom | Source | Per task |
|---|---|---|---|
| TK1 | Arguments through generic (non-scalar) pointer loads; waves 1-15 return | B/dsl_tiles.hip:1432-1440 | 64 B |
| TK2 | 2 coherent 2-B logit loads per lane | B/dsl_tiles.hip:1448-1454 | 256 B; 1 trip |
| TK3 | 32-bit keys; round 0 DPP wave max (6 DPP + readlane) | TU:39-46, 59-67, 84-90 | — |
| TK4 | Softmax: 2 `expf`, butterfly wave sum (6 `ds_bpermute`), divide | TU:91-97, 19-22 | — |
| TK5 | 7 more rounds of DPP wave max + clear | TU:98-109 | 7 wave maxes |
| TK6 | Renormalise, lanes 0-7 store ids, slots and weights (coherent) | TU:110; B/dsl_tiles.hip:1487-1494 | 96 B |
| TK7 | Static ISA: 706 instructions, 54 waits, no body barrier; the dead sparsemixer branch (mode 2) accounts for the largest dsl_tiles.hip lines | IS:9, 73-89 | [E] |

**moe_up** (`etx_tile_x_dsl_moe_up_fold`; 192 tasks, 24 per XCD; gate and up rows as one stream, GU1). Path per
layer: body 16.05 = bytes 11.70 + hops 1.02 [E] + machinery 3.32 [R]; gap 1.09, all free. **moe_down**
(`etx_tile_x_dsl_moe_down_fold`; 256 tasks). Path per layer: body 9.07 = bytes 5.85 + hops 1.02 [E] + machinery
2.20 [R]; gap 2.90, of which 2.21 before-taken [M]. Their atoms are MOE's A-U, unchanged here (MOE:56-80):

| ID | Atom (MOE:58-80) | Bytes or count per task, up / down [E] |
|---|---|---|
| M.A | Arguments and valid-block guard | 160 B argument block |
| M.B | Slot, row and request range (runtime `blk/tpe`) | — |
| M.C | LDS layout, expert id coherent load, expert base | 4 B; 1 dependent coherent trip |
| M.D | Stage activation to LDS + barrier | 4,096 / 1,536 B coherent |
| M.E | Path selection (GU1 on up, EPL=4 on down) | — |
| M.F | Segment/range setup | — |
| M.G | Zero the 16×64 partial matrix (93.75% never written) | 4,096 B LDS |
| M.H | Clear barrier | 1 barrier |
| M.I | Base, range start, accumulator init | — |
| M.J | Weight stream (NT) | 262,144 / 98,304 B |
| M.K | Consume control: next-row test, segment walk | 256 / 192 segments |
| M.L | Row-transition flush control | 64 flushes |
| M.M1 | Wave sum per flush (WSUM) | 64 |
| M.M2 | Lane-0 partial read-add-write | 512 B LDS |
| M.N | Unpack, LDS x reads, dot | 524,288 / 196,608 B LDS reads |
| M.O | Pipeline tail and final guard | — |
| M.P | All-wave partial join barrier | 1 barrier |
| M.Q | Wave-0 gather of 16 partials | 4,096 B LDS |
| M.R | Round to bf16 (up) / routing weight, round, store (down) | 64 / 128 B coherent out |
| M.S | Second helper barrier (arena reuse) | 1 barrier |
| M.T | SiLU × up, round, store (up) | 64 B coherent out |
| M.U | Publication drain + barrier (= P9) | 16 waves |

Static ISA: moe_up 3,858 instructions, 18 barriers, 56 VGPR; moe_down 1,873, 9 barriers, 40 VGPR (IS:10-11,
18-19).

**lm_head** (`etx_tile_dsl_gemv_fold`, 256 tasks, 594 rows each, once per step). Body 151.9 = bytes 144.7 [E] +
machinery 7.2 [R]. Atoms: **LM1** norm fold recompute in every task (as QK2-QK4, 40 KB per task); **LM2** 2.43 MB
NT weights per task in 10 chunks of 64 rows; **LM3** 33 barriers and 1,188 B of stores per task (GV:715-723).

**argmax** (`vl_argmax`, one 1024-thread workgroup). Body 9.5 = bytes 0.1 + hops 0.6 [E] + machinery 8.8 [R].
Atoms: **AX1** 16-17 × 16-B loads per thread over 303,872 B of logits (`examples/vllm_llm/vl_tiles.hip:52-68`);
**AX2** LDS tree of 10 halvings, 11 barriers (:74-82); **AX3** thread 0 writes the token, the next input and the
time (:83-88).

**prologue** (`vl_prologue`, one workgroup). Body 3.9 [M]. Atoms: **PL1** token time and position stores by
thread 0; **PL2** embedding row copy of 4 KB to each residual replica (`examples/vllm_llm/vl_tiles.hip:28-42`;
replica count [U]).

**in_norm** (`vllm::rms_norm_kernel` imported, 8 replicas, one block of 256 threads each). Body 3.7 [M]. Atom
**IN1**: 1 × 16-B input load per thread, CUB block reduce (1 barrier), `rsqrtf` + barrier, reload input and
weight, 16-B store (`B/imp/layernorm_kernels.hip:64-98`).

### 2.4 The three summing views and how they reconcile

Each view must sum to the span, 4067.3 us. Only the first is a closure of measured intervals; the other two
close because one of their rows is defined as a remainder, and that row is named.

| View | Rows | Sum | Residual row |
|---|---|---:|---|
| **Body** (MT:4-19) | path body 3191.8 [M] + gap 875.5 [M] | 4067.3 | none: both are stamp differences |
| **Wait** (MT:4-19, 21-38) | bytes 1439.6 [E] + hops 534.4 [E] + machinery 1217.8 [R] + before-taken 231.3 [M] + after-take 171.4 [R] + free 470.1 [M] + window 2.6 [M] | 4067.3 | machinery 1217.8 (body minus model) and after-take 171.4 (gap minus before-taken) |
| **Ledger** = REV's rebase (MT:40-53) | 48 × round11 floors 2255.0 [E] + allowances B 782.9 [E] + correction −15.2 [R] + non-layer bodies 169.1 [M] + gaps 875.5 [M] | 4067.3 | correction −15.2 |

**Ledger to body.** B is RB's rule, 48 × (critical body − floor), and reproduces REV's 0.783 ms (RB:5-14). REV's
"critical body" is the mean over layers 1-45 of the last-finishing task's body (`etx/tools/floors.py`
`round11_from_trace`), not the body of the task on the critical path. The difference, summed over the nine
layer roles, is **−15.2 us** [R] (MT:42-51): the path is slightly shorter than 48 × the layer-1-45 critical
bodies, mostly in moe_up (−8.4), topk (−4.5) and prep (−3.7). The correction is below every bar in this
document, so REV's allowances stand on the path basis within 0.015 ms.

**Wait to ledger.** The ledger's 2255.0 us of floors is not the wait view's 1439.6 + 534.4 = 1974.0 us of
bytes and hops: the round11 floors (`docs/FLOORS-2026-10-08.md:239-247`) include serial instruction and barrier
time that the wait view leaves in machinery. Per layer role, floor − (bytes + hops) is the part of machinery
that the floors already call irreducible; the rest of machinery is B.

**Known, estimated, residual** (MT:77-96). [M] 556.7 us (13.7%): before-taken waits and free notification into
non-touch grids, the intervals whose content has one cause. [E] 1974.0 us (48.5%): the byte and hop models.
[R] 1536.6 us (37.8%): machinery, late after-take, free gaps into touch grids and the window start. The [R]
share is what the existing trace cannot split.
