# vLLM's decode kernels on MI300X, in vLLM's form and fused by ETX: operation model, calibration, XCD simulation

Status: 2026-09-28, after session moe_gate in results_2026-09-28.
The approach replaces “measure, then guess” with “model, then measure what the model cannot explain”.
Each number has one label: measured, fitted, or predicted.
A fitted number uses a free parameter from data.
A prediction uses the model without a parameter fitted to that number.

Tools:

- `examples/vllm_llm/ops/hip.md` and `ops/fused_moe.md` list operations with source and assembly line references.
- `isa_build.sh` + `isa_inventory.py` compare instruction counts for vLLM and imported forms with the ROCm 7.2.4 compiler.
- `kmodel.py` contains the cost model and its fit.
- `etx/sim/xcd.py` + `xcd_run.py` simulate the XCDs.
- `measured.py` stores all measured per-role values in one JSON.
- `compare.py` produces the ledgers.

## 1. The model architecture: one Qwen3-30B-A3B decode layer (batch 1, context 1024, bf16)

| launch (vLLM order) | kernel | grid x block | read per launch | work |
|---|---|---|---|---|
| post_norm | fused_add_rms_norm | 1 x 1024 (256 with data) | 12 KB | residual add + RMS norm |
| router | wvSplitK YTILE 1, 128 x 2048 | 304 x 1024 (32 with rows) | 0.5 MB | expert logits |
| topk | topkGating 8 of 128, softmax, renormalise | 1 x 256 (16 lanes work) | 256 B | expert ids, weights |
| (fill) | torch fill of num_tokens_post_padded | 1 x 256 | - | a constant at batch 1: not in ETX's graph |
| moe_up | Triton fused_moe, gate_up of 8 experts | 192 x 256 | 50 MB | one real row of the 16-row tile |
| moe_act | silu_and_mul, 8 rows | 8 x 96 | 24 KB | expf + IEEE divide per element |
| moe_down | Triton fused_moe, down x routing weight | 256 x 256 | 25 MB | |
| moe_sum | moe_sum, top-8 | 1 x 256 | 32 KB | |
| (copy) | copyBuffer into the custom op's output | 1 x 512 | - | aliased in ETX |
| in_norm | fused_add_rms_norm | 1 x 1024 | 12 KB | |
| qkv | wvSplitK 5120 x 2048 | 304 x 1024 (285 with rows) | 21 MB | |
| q/k norm, rope, cache | rms_norm_3d x2, rotary neox, reshape_and_cache | small | 40 KB | |
| attn, reduce | paged attention ll4mi mfma16 GQA 8, reduction | 20 x 256, 32 x 128 | 2.2 MB (KV) | |
| o_proj | wvSplitK 2048 x 4096 | 304 x 1024 (256 with rows) | 17 MB | |

116 MB per layer, 5.57 GB per token plus 0.62 GB lm_head: **bandwidth floor 1.44 ms** at 4.3 TB/s; vLLM 4.90 ms.

## 2. Hardware parameters

| parameter | value | status | source |
|---|---|---|---|
| hardware barrier `__syncthreads` | 77 ns | measured | hw_bench wg (results_2026-09-27d) |
| LDS cross-lane step `ds_bpermute` | 56 ns | measured | hw_bench wg |
| DPP step | 4 ns | assumed (a few cycles) | |
| software slot barrier (ETX import) | 0.26 us | predicted from its instructions (3-5 dependent LDS round trips + poll) | ops/fused_moe.md 4 |
| clock under load | 1.98 GHz | measured | hw_bench clock |
| kernel-boundary cost F (dispatch, argument fetch, end release) | 2.5 us | fitted to vLLM's launches (the empty-kernel boundary alone is 1.6 us) | kmodel fit |
| idle dependent round trip of fresh data t_v | 0.2 us | fitted | kmodel fit |
| issue cost per instruction on a wave's critical path t_i | 1.5 ns | fitted | kmodel fit |
| scalar round trip t_s | 0 (absorbed in F) | fitted, not identifiable | kmodel fit |
| effective stream bandwidth | 5.2 TB/s | fitted (at the grid's edge<br> includes overlap the model does not describe) | kmodel fit |
| ETX unfused: extra front per launch A_unf | 0.65 us | fitted to ETX's unfused launches without software barriers | kmodel fit |
| ETX task: front of every imported kernel A_task | 2.1 us | fitted to ETX's small fused tasks | kmodel fit |
| handoff XCD-local / device-wide | 1.0 / 1.3 us | measured (hw_bench handoff, padded counters) | used unchanged by the simulator |
| wake-up spread / per-task stream-rate spread | 0.6 us / 3% | fitted to the measured ramps and tails | simulator |

Every kernel's stream follows Little's law.
A trip's memory time is the loaded latency max(t_v, bytes in flight on the GPU / bandwidth).
The trip count is bytes per block / bytes in flight per CU.
Work outside the memory overlap adds to each trip.
For fused_moe, the LDS write / barrier / read costs 0.33 us.

## 3. The kernels as pipelines (ops/hip.md section 13, ops/fused_moe.md)

At batch 1 every kernel is a dependent chain. What decides each one's time:

| kernel | chain per block | stream | bound (vLLM form) |
|---|---|---|---|
| fused_add_rms_norm | 2 S, 2 V, 10 LDS steps, 2 barriers<br> the weight load waits for barrier 2 | - | latency |
| wvSplitK (qkv / o_proj / router) | 4 S<br> the whole activation into LDS<br> barrier<br> then weight trips, each waiting for its own 4 x 16 B loads (not pipelined)<br> 8-16 DPP | 16 KB (4 waves) or 36 KB (9 waves) in flight per CU | bandwidth (qkv, o_proj), latency (router: 32 CUs) |
| rms_norm_3d, rope | 4 S + 2 V + 2 barriers<br> rope 5 V of 2 B scalar loads | - | latency |
| reshape_and_cache | 9 V: the V copy is 8 serial 2 B load / store rounds | - | latency |
| paged attention mfma16 | 4 S + 7 V (2 scratch reloads), 7 barriers, 8 of 16 MFMA rows real | 128 KB KV per block | latency at 1024 tokens |
| reduce | 3 S + 1 V, 12 LDS steps, 1 barrier | - | latency |
| topkGating | 2 S + 5 V, 40 dependent LDS steps, ~1500 instructions, 16 lanes | - | issue / latency |
| silu (MoE) | 2 S + 1 V, ~340 VALU (expf + IEEE divide) | - | issue |
| moe_sum | 2 S + 1 V, 8 x 16 B loads | - | latency |
| fused_moe gate_up / down | 2 S + 2-3 V prologue<br> K tiles of 16 KB, 2 barriers each (31 / 11 per program), the next tile's loads issued after this tile's LDS work | 16 KB in flight per CU | bandwidth + 0.33 us exposed per tile |

The imported form in `isa_inventory.py` has the same instructions, with two exceptions.
For blocks smaller than the 1024-thread workgroup, each barrier becomes a software slot barrier.
This affects attention, reduce, rms_norm_3d, and fused_moe.
The fused_moe prologue also has 5-6 dependent round trips instead of 2.

## 4. The three forms: predicted against measured (us)

vLLM fitted; both ETX columns predicted except the two front constants (A_unf, A_task). MoE:

| role | vLLM pred / meas | ETX unfused pred / meas | ETX task body pred / meas |
|---|---|---|---|
| in_norm | 3.99 / 3.68 | 4.64 / 4.33 | 3.59 / 2.94 |
| qkv | 7.24 / 6.84 | 7.89 / 8.38 | 6.84 / 7.92 |
| attn | 5.74 / 7.04 | 7.67 / 8.70 | 6.62 / 7.28 |
| reduce | 4.09 / 3.52 | 4.93 / 6.90 | 3.88 / 5.45 |
| o_proj | 6.59 / 6.32 | 7.24 / 7.66 | 6.19 / 7.38 |
| post_norm | 3.99 / 3.56 | 4.64 / 4.61 | 3.59 / 3.66 |
| router | 3.31 / 3.48 | 3.96 / 4.01 | 2.91 / 3.20 |
| topk | 7.99 / 7.49 | 8.64 / 9.98 | 7.59 / 8.80 |
| moe_up | 17.85 / 17.31 | 24.94 / 24.42 | 23.89 / 21.91 |
| moe_act | 3.21 / 3.96 | 3.86 / 4.05 | 2.81 / 2.28 |
| moe_down | 9.92 / 9.37 | 13.36 / 12.71 | 12.31 / 11.58 |
| moe_sum | 3.15 / 3.84 | 3.80 / 4.33 | 2.75 / 3.42 |
| layer (roles with a vLLM counterpart) | 81.9 / 80.8 | 116.1 / 121.4 | |

Qwen2.5-1.5B (the same parameters): vLLM layer 61.6 / 61.6, ETX unfused 70.2 / 70.0 us.

**The imported fused_moe cost has an explanation.**
vLLM's version is bandwidth-bound, with 48 MB at loaded latency.
In ETX, each K tile adds two software barriers to the critical path, or +0.37 us per tile.
This makes the kernel latency-bound.

The predicted increases are gate_up +7.1 us and down +3.4 us, within 0.65 us of both unfused measurements.
No ETX-side parameter was fitted to fused_moe.
A_unf was fitted only to kernels without software barriers.

## 5. The XCD simulation (etx/sim/xcd.py)

The simulation executes the compiled plan task by task.
The plan has 67,994 tasks, each worker's static queue, and each event's producers and scope.
A task first waits for its events: the last producer's end, plus scope-dependent handoff and wake-up spread.
It then executes front, chain, stream, and end.
Little's law distributes stream bandwidth among all active CUs.

The output uses the ETX_TRACE layout.
Thus, the same ledger processes measured and simulated steps.

MoE, ETX_VL_XCD schedule (one selected expert per XCD), per layer (us, layers 2-45):

| | handoffs | ramp | bodies | tails | layer | token |
|---|---|---|---|---|---|---|
| measured | 13.7 | 4.5 | 102.2 | 10.7 | 131.0 | 6.63 ms |
| simulated | | | | | 121.2 | 5.84 ms |

The simulation is 12% optimistic.
Device-wide handoffs measure 1.5-2.1 us against 1.3, including the first read of data from another XCD.
XCD-local handoffs measure 0.3-0.5 us against 1.0.
They are faster than the microbenchmark because one producer and its readers share L2.
The reduce, topk, post_norm, and moe_sum bodies also differ by 1-2 us each. See section 7.

The XCD view of layer 5 shows the dependency stalls.
The attention block combines q/k norm + rope + cache in one task, then uses 5 CUs for attention and 4 for reduction.
About 300 CUs wait during 38 us of the 131 us layer.

The expert path uses 24 gate_up CUs and 32 down CUs per XCD, with one expert per XCD.
An XCD starts the next launch as soon as its expert finishes, without a wait for other XCDs.
This produces negative handoffs in the ledger.
All XCDs meet only at top-k and moe_sum.

## 6. What the model says

Per imported kernel, a fused task costs its front (2.1 us, fitted) plus a handoff (1.0-1.3 us) where vLLM pays one
kernel boundary (2.5 us, fitted): **fusing an imported kernel as it is today costs 0.6-0.9 us more than launching it**,
before any barrier effect. The two fixes and what the simulation predicts (MoE, ms per token; vLLM 4.90):

| | xcd schedule | base schedule |
|---|---|---|
| today (simulated / measured) | 5.84 / 6.63 | 6.14 / 6.76 |
| imported barriers as hardware barriers (rounds for idle waves) | 5.31 | 5.61 |
| + the argument path of every imported kernel 2.1 -> 0.5 us (arguments read before the wait) | 4.08 | 4.37 |

With the 12% the simulation is optimistic by, both fixes land at about 4.1-4.6 ms: at or below vLLM, not far below.

## 7. Not explained, and the measurement for each (next session, one build each)

| residual | size | candidate | measurement |
|---|---|---|---|
| A_task = 2.1 us per imported kernel (chained parts pay it too) | 720 tasks x 2.1 = 1.5 ms per token | the argument chain (args[0] -> argument block -> parameters) before the first load, 5-6 round trips for fused_moe | build with the argument block read before the wait<br> trace: body shrinks by the predicted amount or not |
| software barrier cost | 0.26 us predicted | the slot barrier's LDS round trips | build with hardware barrier rounds<br> fused_moe gate_up task body 21.9 -> ~15 us predicted |
| ETX unfused excess not explained (reduce +2.0, topk +1.3, cache +0.8, silu 1.5B +1.5 us) | 5-7 us per layer | instruction-cache misses (the unfused kernel holds every tile's code), slot packing (reduce: 8 blocks per workgroup) | SQC_ICACHE_MISSES per launch, ETX unfused vs vLLM<br> reduce with slots = 1 |
| device handoff 1.5-2.1 us vs 1.3 | ~4 us per layer | first read of remote-written data (staging) | trace: split ready -> first load |
| vLLM attention 7.0 vs 5.7 us | 1.3 us | the model's attention chain (KV loads at 1024 tokens) | from vLLM's trace at shorter contexts (no GPU needed if the traces exist) |

The inventory raises a scalar-cache correctness question.
Imported kernels use scalar loads for per-step positions, seq_lens, and slot_mapping.
ETX acquires invalidate vector caches, not the scalar cache.

The current host is correct because each step launches its own megakernel, which invalidates the scalar cache at kernel start.
Within that launch, every reader waits transitively for the prologue that writes those scalars.
Thus, each CU first reads them after the write.
Each run decodes 1055 steps with a new position and matches 32/32 tokens.
A stale position would write the wrong KV-cache slot.
This would become a bug if one launch executed several steps.

## 8. What each lever is worth (examples/vllm_llm/whatif_sim.py, simulated; ~12% optimistic today)

The measured layer 5 contains 304 CUs x 130.7 us of CU time.
Tasks occupy 32%, and CUs without a task account for 51%.
The sparse launches have 1-8 tasks: norms, top-k, moe_sum, and attention.
Expert launches use 192 / 256 of 304 CUs.
Handoffs take 10%, and ramps and tails take 7%.

Weight transfer accounts for 31% of CU time.
The layer's 116 MB would take 27 us at 4.3 TB/s.
Idle CUs wait for dependencies instead of data transfer.

| one request | ms per token | tokens/s |
|---|---|---|
| today | 5.84 | 171 |
| hardware barriers for imports | 5.31 | 188 |
| argument path 2.1 -> 0.5 us | 4.61 | 217 |
| both | 4.08 | 245 |
| both + handoffs halved | 3.71 | 270 |
| no argument path, no handoff (bound for these kernels) | 2.95 | 339 |
| vLLM, measured | 4.90 | 204 |

| two requests interleaved, one workgroup per CU (request 1 fills the CUs request 0 leaves idle) | ms per step | tokens/s |
|---|---|---|
| today | 7.38-7.91 | 253-271 (x1.48-1.58) |
| both fixes | 5.28 | 379 (x1.54) |
| vLLM batch 2, measured | 5.72 | 349 |

The argument path is the largest opportunity at 1.2 ms, followed by barriers at 0.5 ms.
Both fixes put one request at vLLM latency: 4.1 simulated, or ~4.6 with simulation bias.
Two interleaved requests reach +9% over vLLM batch 2 with lower per-request latency, within simulation error.

Even with free arguments and handoffs, these kernels take 2.95 ms at batch 1, against a floor of 1.44.
Their dependent round trips and synchronization impose this cost.
Further progress needs different kernels or more work per step, such as more requests or speculative decoding.
Scheduling alone cannot supply it.

## 9. The fixes as built (2026-09-28, local; session examples/sessions/moe_fix.py)

- **Hardware barriers in rounds** (etx/runtime/include/etx/import.h, etx/importer/adapter.py): an imported kernel's
  __syncthreads is the workgroup's `s_barrier`; around the blocks of one launch every wave runs a round protocol: count
  itself done after its block (or at once if it has none), then repeat `s_barrier` until all 16 waves are done. Blocks of
  one kernel reach the same barriers in the same order, so each round is every unfinished wave's next barrier; an early
  `return` inside a kernel joins the rounds, as a terminated wave leaves the hardware barrier count. `-DETX_IMP_SW_BARRIER`
  restores the software slot barrier.
- **Argument path** (import.h `etx_imp_argblock`, codegen `pre_wait` hook): the argument-block pointer is made
  wave-uniform, so the kernel's parameter reads compile to scalar loads (megakernel: 95 -> 2644 `s_load`, that is, they had
  been vector loads through the L1/L2 every acquire invalidates), and wave 0 touches the task's argument block (every part
  of a chained task) into the scalar cache before it waits for the task's events. `-DETX_IMP_VECTOR_ARGS` and
  `-DETX_NO_PRE_WAIT` restore the old path and remove the touch.
- **Two requests** (`ETX_VL_REQUESTS=2`, model.py / host.hip / vl_tiles.hip): every launch twice, alternating in the graph
  and so in every worker's queue; per-request buffers, KV caches, argument blocks and step parameters; request 1 decodes
  the 1024-token prompt rotated by 97 tokens (`examples/llm/prep.py --rotate 97 --golden-only`, HF greedy on the VM).
- ROCm 7.2.4 builds: MoE base 0 spills, xcd 3, xcd two requests 4, xcd one block per slot 0; Qwen2.5-1.5B base 0.

Pre-registered, xcd schedule, ms per token (simulation x 1.135, its bias against today's measurement):

| build | predicted |
|---|---|
| old (all three switches) | 6.63 (today<br> otherwise compare ratios) |
| hardware barriers only | 6.03 |
| argument path only, front 0.3 / 1.0 / 1.5 us | 5.05 / 5.67 / 6.10 |
| both | 4.45 / 5.07 / 5.51 |
| two requests, both | 347 / 304 / 278 tokens/s (vLLM batch 2: 349) |
| fused_moe gate_up task body (trace) | 21.9 -> ~15 us |

The trace of the new build measures the front directly; that number decides which column applies. Beating vLLM at batch
1 (4.90) needs the front at about 0.5 us or less.

## 10. Every parameter from a microbenchmark (pre-registered 2026-09-28, before the session)

Sections 4-9 fit six numbers to launch times (F, t_s, t_v, t_i, BW, A_task) and assume three more (t_swbar from an
instruction count, the handoffs from a benchmark whose protocol is not ETX's). Fitted values can absorb a wrong
structure: t_v = 0.2 us and t_s = 0 are below any physical round trip on MI300X, so the fit has most likely traded
them against F (2.5 us against a measured 1.6 us kernel boundary) and against round-trip counts that overlap in
reality. The session measures each one directly, first, on GPU 0 (`bench/hw/hw_bench.hip` lat / lf / launch / evx,
`bench/hw/imp_bench.hip` in both barrier protocols, with `etx/import.h` itself), and
`examples/vllm_llm/bench_params.py` maps them onto the model (the mapping is in its docstring) and simulates the step
with microbenchmark values only.

| measurement | stands for | fitted / assumed | predicted range | if outside |
|---|---|---|---|---|
| lat vector MALL idle | t_v | 0.2 | 0.25-0.5 us | above 0.3: the V counts of section 3 overcount dependent trips<br> recount from the ISA, do not refit |
| lat vector L1, L2, HBM idle | levels | - | 0.05-0.15, 0.1-0.3, 0.3-0.8 us | |
| loaded / idle, same level | Little's-law latency | - | x1.5-3 at full rate | |
| lat scalar K$ after inv sc1 | the K$ survives ETX's acquire | assumed | < 0.1 us (a K$ hit) | the argument fix of section 9 has no effect: the scalar path is also invalidated |
| lat vector L2 after inv sc1, x2 | A_task, vector build | 2.1 | 0.6-1.2 us | below 1.0: A_task holds more than the argument path (other loads of the adapter) |
| lf, highest rate | BW | 5.2 TB/s | 4.0-4.5 TB/s at >= 4 MB in flight | 5.2 is not physical (it absorbed a missing term) |
| lf 192 WGs x 16 KB (fused_moe) | fused_moe stream rate | 3.2 TB/s (model) | 2.5-3.5 TB/s | the gate_up body cannot reach 15 us |
| launch empty, 304 x 1024 | F | 2.5 | 1.5-1.8 us (sync_bench: 1.6) | |
| launch hops 0 - empty | first load after a boundary | - | 0.3-0.8 us | |
| evx xcd P 37 C 37 median | DOMAIN handoff | 1.0 | 0.3-0.6 us (the trace's 0.3-0.5) | ETX's poll is not what the trace saw |
| evx device P 296 C 296 median | DEVICE handoff | 1.3 | 1.3-2.1 us (the trace's 1.5-2.1) | |
| evx late (event already complete) | per-task wait cost | 0 | 0.1-0.4 us | a term the simulator lacks |
| impbar sw, fused_moe | t_swbar | 0.26 | 0.2-0.3 us | |
| impbar rounds, fused_moe | the round barrier | 0.077 (bare s_barrier) | 0.08-0.12 us | above 0.15: the hardware-barrier gain of section 9 shrinks |
| impbar rounds R 0 | exit protocol per imported task | 0 | < 0.5 us | a per-task cost of the round protocol |

Rule from here: the model runs on microbenchmark values; a term whose microbenchmark-only prediction misses the trace
by more than 15% is re-examined structurally (what it counts, what overlaps), not refitted.

## 11. vLLM's fused q/k norm + rope (ETX_VL_QKNR, pre-registered 2026-09-28)

What it replaces: the attention block's preparation (q_norm 32 blocks, k_norm 4, rope, cache) ran as one chained task
on one CU for 19 us of the 40 us block (trace, layer 5): four argument paths and about 20 dependent round trips in
series. (Why it is serial: model.py links every launch to the previous one in vLLM's order, so q_norm and k_norm,
which write different buffers, do not overlap either; splitting the chain into tasks only adds handoffs, simulated
6.05 against 5.84 ms.) vLLM ships `tensorrt_llm::kernels::fusedQKNormRopeKernel` (csrc/libtorch_stable/
fused_qknorm_rope_kernel.cu, ROCm paths included), enabled by compilation_config pass_config
enable_qk_norm_rope_fusion (off by default; it adds +rotary_embedding). One 32-lane warp per head, 8 per 256-thread
block, 5 blocks for 32 + 4 heads, in place on qkv; per warp: the qkv and position loads, a 5-step reduce, cos/sin
after the position, the store (2 dependent round trips, from the source; the kernel model's `qknr` entry). Imported:
80 B of arguments, no LDS, no barrier, 4 slots per workgroup. The megakernel's registers are unchanged (VGPR 128, the
xcd build's 2 spills). The cache write (9 serial round trips) stays.

Simulated (fitted parameters, ms per token; the fitted set is ~12% optimistic):

| build | today's protocol | with both fixes |
|---|---|---|
| xcd (four launches chained) | 5.840 | 4.078 |
| QKNR=1: fused launch chained with the cache (2 rounds of 4 slots on one CU) | 5.593 | 3.942 |
| QKNR=grid: fused launch on 2 CUs, then the cache | 5.511 | 3.937 |
| QKNR=grid, slots 1 (5 CUs) | 5.529 | 3.955 |

Predictions for the session (moe_xcd_qknr_new = grid, moe_xcd_qknrc_new = chained, both with the fixes): each 3.3-3.5%
faster than moe_xcd_new, grid and chained within 1% of each other; the preparation span in the trace 19 -> 8-11 us.
Correctness: moe_base_qknr --check byte-identical against the original fused kernel, 32/32 tokens (the fused kernel
rounds once where rms_norm + rotary rounded twice, so a token flip is possible; if it happens it is reported, not
hidden). Baseline: vLLM with the same pass on, same image, same VM, beside the default. The rocprof trace must show
fusedQKNormRope launches (48 per step) or the vLLM number does not count as "the same kernels".

### 10.1 Measured (results_2026-09-28b, 1x MI300X enc1-gpuvm002, 16:13 UTC, before any timed run of the session)

| measurement | predicted | measured | |
|---|---|---|---|
| vector round trip L1 / L2 / MALL / HBM, idle | 0.05-0.15 / 0.1-0.3 / 0.25-0.5 / 0.3-0.8 | 0.103 / 0.143 / 0.261 / 0.412 us | in range |
| the same, loaded (303 CUs streaming) | x1.5-3 | 0.73 / 0.71 / 0.69 us (L2 / MALL / HBM) | queueing sets it, not the level |
| scalar K$ hit, and after inv sc1 | < 0.1 | 0.113, 0.119 us | the K$ survives ETX's acquire |
| buffer_inv sc1 alone, idle / loaded | - | 0.19 / 0.63 us | every DEVICE acquire under load pays 0.6 us |
| vector hop after inv sc1, loaded | - | 1.14 us | |
| lf highest rate | 4.0-4.5 TB/s | 3.93 TB/s (18.5 MB in flight) | the fitted 5.2 is not physical |
| lf 192 WGs x 16 KB (fused_moe) | 2.5-3.5 | 3.56 TB/s | |
| kernel boundary, empty / + first load | 1.5-1.8 / +0.3-0.8 | 1.61 / +0.26 us | the fitted F 2.5 absorbed other terms |
| software slot barrier (fused_moe shape) | 0.2-0.3 | 0.282 us | |
| round barrier / exit protocol per task | 0.08-0.12 / < 0.5 | 0.106 / 0.311 us | |
| wait on a complete event (acquire + barrier) | 0.1-0.4 | 0.52 us | a per-task term the simulator lacks |
| evx XCD P 37 C 37 / device 296 / 296 | 0.3-0.6 / 1.3-2.1 | 2.20 / 3.91 us | NOT the trace's handoff: unexplained, not used |

Derived (examples/vllm_llm/bench_params.py): the argument path under load, vector build = one invalidate (0.63) + two
hops that miss it (2 x 0.51) = 1.65 us (fitted A_task 2.1); scalar build = 0.63 + two K$ hits = 0.86 us. So the
argument fix saves ~0.8 us per imported task, not the 1.6-1.8 the section 9 table's best column assumed.

The simulation with microbenchmark values only (t_v, t_s, F, BW, t_swbar, the round barrier, A_task; handoffs from
the trace, 0.4 / 1.8 us): **today's build 6.661 ms against the measured 6.63 (0.5%)**, where the fitted set said
5.84. Its predictions for this session's builds, written before they run:

| build | microbenchmark-only prediction | (fitted x 1.135, section 9) |
|---|---|---|
| old (moe_xcd_old) | 6.66 | 6.63 |
| hardware barriers only (moe_xcd_hw) | 6.15 | 6.03 |
| argument path only (moe_xcd_args) | 6.05 | 5.05-6.10 |
| both (moe_xcd_new) | 5.54 | 4.45-5.51 |

## 12. The DSL structure for the small ops, simulated before building (2026-09-29, local)

ETX_VL_DSL=1 in model.py (simulation only): q/k norm + rope + cache + attention as one tile task per (partition, kv
head), top-k as one tile, moe_sum folded into the next in_norm. The streaming ops stay vLLM's. Costs: kmodel DSL_MOE
(estimated from the chains); parameters from the microbenchmarks, handoffs from the trace; examples/vllm_llm/dsl_ceiling.py.

| plan | simulated | measured | realistic (sim x 1.12) |
|---|---|---|---|
| xcd, four small launches chained | 5.54 | 6.31 | |
| xcd + fused q/k norm + rope (grid) | 5.26 | 5.86 | |
| DSL: fused small ops | 4.55 | - | ~5.1 (vLLM 4.67) |
| DSL, two interleaved requests | 6.16 ms/step, 325 tok/s | - | ~290 tok/s (vLLM batch 2: 360) |

Layer 5 changes from 113 -> 108 -> 92.5 us.
Small-op savings are 15 us per layer, not the manual estimate of 39.
The result fails the 4.4 ms go/no-go threshold.

The 92.5 us layer contains 29.5 us of bytes at 3.93 TB/s and ~28 us of latency-bound tasks.
It also contains ~15 us of handoffs and ~20 us of stream excess inside vLLM's kernels.
fused_moe gate_up takes 21.6 us per task against 12.8 at full rate.
It has 16 KB in flight per program and a 0.33 us gap per trip.
Other pairs are down 11.6 vs 6.4, qkv 7.9 vs 5.3, and o_proj 7.0 vs 4.3.

This excess, ~14 us/layer or ~0.65 ms/token, is the only opportunity that reaches vLLM at batch 1.
It requires DSL-generated GEMV streams instead of imported kernels.

## 13. Direction A: the streaming GEMVs generated by the DSL (2026-09-29, pre-registered)

Decision (the user's, 2026-09-29): direction A, the DSL generates the streaming GEMVs; the small ops follow. The op:
`etx/ops/gemv.py` emits the tiles and the argument layout in the adapter's form (three exports: dsl_gemv, dsl_moe_up
with silu fused into the epilogue, dsl_moe_down with the routing weight); model.py links them with ETX_VL_DSL_GEMV=1;
the host fills their blocks and, with --check-dsl, checks the last layer's outputs against a CPU reference (2 bf16
ulps. The summation order differs from vLLM's kernels, so byte identity is not the criterion). The stream loop is
examples/llm/gemv.h's at 4 (16 B) or 8 (8 B) loads per lane in flight: 64 KB per 16-wave workgroup (16 per lane, the
4-wave design, spilled 359 VGPRs in the 128-register megakernel). ROCm 7.2.4 build: 7 spills (2 before). The ISA of
the standalone tile: 4 global_load_dwordx4 back to back per batch, one 16 B load per thread for the staged x.

Simulated (microbenchmark parameters), ms per token:

| plan | simulated | realistic (x1.11) |
|---|---|---|
| today's best (xcd + fused q/k norm + rope, grid) | 5.26 | measured 5.86 |
| build 1: + DSL GEMVs | 4.58 | ~5.1 |
| build 1 + DSL top-k (etx/ops/topk.py) | 4.42 | ~4.9 |
| build 2: + the attention block by kv head (etx/ops/attnprep.py on XCD h<br> vLLM's attention at one block per task and its reduce on XCD h<br> the session, examples/sessions/dsl_gemv.py) | 4.01 | ~4.45 |
| build 2, two interleaved requests (ETX_VL_REQUESTS=2<br> the session's run_two) | 5.20 ms / step | ~2.89 ms per token, 347 tok/s (vLLM batch 2: 360) |
| + DSL top-k and the moe_sum fold (phase 1) | 4.36 | ~4.85 |
| + attention with its preparation in one task (all of A) | 4.00 | ~4.45 (vLLM 4.67) |

The session uses build 2, predicted at 4.4-4.5 ms against vLLM at 4.67.
Build 2 obtains A's attention benefit without a DSL attention kernel.
Preparation for one kv head combines norm + rope + cache write in a 9-wave DSL task on XCD h.
The head's attention partitions and reduce also use XCD h.

Only qkv -> prep and reduce -> o_proj remain device-wide handoffs, versus five before.
GQA keeps qkv -> prep device-wide because one kv head's q rows span two XCDs at full qkv bandwidth.
The gates are 32/32 tokens and CHECK-DSL OK.
If build 1 exceeds 5.4 ms, the GEMV tiles fail to reach the lf rate.
Compare qkv / o_proj / moe_up / moe_down bodies with 5.3 / 4.3 / 12.8 / 6.4 us to locate the loss.

## 14. Formal checks (2026-09-29, `formal/`, `bash formal/check.sh [old]`)

| what | tool | claim | result |
|---|---|---|---|
| the DSL GEMV stream loop (etx/ops/gemv.py: segments in contiguous per-wave runs, flush on row change, per-wave partials summed) | Lean 4 (core, no Mathlib), `formal/Gemv.lean` | `gemv_correct`: for every row, the sum over waves of the partials equals the sum of the row's segment values, for any spr, S, per >= 1 and any wave count covering S (so the flush bookkeeping and the tail path lose or duplicate nothing) | proved<br> axioms `propext`, `Quot.sound` only |
| the imported-kernel barrier protocol (import.h: counted s_barrier in rounds, exit by round number) | TLA+ / TLC, `formal/ImportRounds.tla` | no deadlock<br> every wave of a use has taken the same number of barriers when it leaves (SameNbarAtBoundary)<br> no wave leaves while another still has block barriers to take (NoOrphanBarrier)<br> termination under weak fairness<br> shared-memory steps interleave freely | NW 3, 2 uses, 0-2 barriers per wave: 271,503 states, no error<br> NW 4: 9,397,101 states, no error |
| the first version of that protocol (leave when done = NW, reset at the next begin: the race the 2026-09-28 review found) | `formal/ImportRoundsOld.tla` | the same claims | TypeOK violated (done exceeds NW: a wave of the next use increments before the reset): the model discriminates |
| the DEVICE event with the last-arriver flush (primitives.h: per-domain sub-counter, the domain's last producer writes back and subtracts the share<br> consumers poll <= 0 then acquire) | `formal/EventFlush.tla` | the word never goes negative<br> it reaches 0 only when every producer's output is visible device-wide<br> a consumer past its wait reads every output (no stale read)<br> one flush per domain<br> termination | 3 domains x 2 producers, 2 consumers: 1,848 states, no error |

What these do not cover: the hardware's memory model beyond "release makes the domain's L2 hold the data, write-back
makes it visible" (the K$ and vector-L1 behaviour was measured, section 10), the fp32 summation order of the GEMV (the
host's --check-dsl tolerance), and the imported kernels themselves (their byte-identity is the --check gate).

## 15. The configuration search (2026-09-29, `examples/vllm_llm/autoconf.py`)

The build switches of model.py (chain, fused q/k norm + rope, DSL GEMVs, DSL top-k, the attention block by kv head) are
searched, not set: every combination is compiled to a plan, every task is costed from the tile its type links
(`kmodel.kern_for_symbol`, so imported kernels and DSL ops mix without a hand-picked model name) and the step is
simulated with the microbenchmark parameters. Of 32 configurations the best is build 2 (4.01 ms simulated; the same
whether the chain switch is on: with the head block there is nothing left to chain), then build 2 without the DSL top-k
(4.17), then the fused-q/k-norm builds with the DSL GEMVs (4.42-4.45). The search is what turns "the user sets six
environment variables" into "the compiler chooses"; slots per import and rows per DSL task are the next dimensions.

## 16. Build 3: the folds (2026-09-29, pre-registered; ETX_VL_DSL_FOLD=1)

The one-block launches between the GEMVs become prologues of their consumers (the design's rematerialisation, section
7.4 of the technical design: redundant parallel work in place of a serial event): every qkv task of layer L > 0 sums the
8 expert rows + the residual and norms them (ln1) itself, task 0 storing the new residual; every router task norms
o + residual (ln2), task 0 storing the residual and the normed vector; lm_head does the same with final_norm; every
expert task recomputes the softmax top-k from the logits (deterministic, lower index on ties) and takes its slot's
expert and weight. Gone per layer: in_norm, post_norm, top-k, moe_sum (4 launches, 4 handoffs, 4 fronts, 3 device-wide
events). The residual alternates between two buffers (router reads X writes Y, qkv of the next layer reads Y writes X, lm_head
reads Y; layer 0 keeps its imported in_norm on X) so readers and the one writer never touch the same one in a step.
Numerics follow vLLM's moe_sum + fused_add_rms_norm (the sum rounded, the add rounded per element, the variance over the
rounded values, out = bf16(z * rsqrt * w)) and topkGating; only the reduction orders differ.

| plan | simulated | realistic (x1.11) | two requests |
|---|---|---|---|
| build 2 | 4.01 | ~4.45 | 2.89 ms / token, 347 tok/s |
| build 3 (this session) | 3.55 | ~3.94 (vLLM 4.67: -16%) | 2.63 ms / token, 380 tok/s (vLLM batch 2: 360) |

Layer 5 in build 3: qkv 9.7, prep 4, attention 6.5, reduce 3.7, o_proj 6.7, router 5.3, gate_up 18.5, down 9.4 us of a
71.6 us layer (build 2: 81.3, today's measured best 108). Batch, extrapolated on the same layer (expert bytes grow with
the distinct experts of B tokens, per-token tasks run in parallel): B = 2: 2.58 ms per token (387 tok/s; vLLM 360),
B = 4: 1.73 (579; vLLM 563), B = 8: 1.22 (818): the margin shrinks with the batch, so batching is not where ETX wins.
Gates as before; ROCm 7.2.4 build: 4 spills (two requests: see the build log).

## 17. Sensitivity of the build-3 prediction; batching against pipelining (2026-09-29, before the measurement)

`examples/vllm_llm/sensitivity.py` repeats build 3 for one and two requests with parameters that the microbenchmarks left uncertain.
A mismatch can then identify a parameter.
The table reports simulated ms per token. Multiply by x1.11 for the realistic figure.

| variant | R = 1 sim | x1.11 | R = 2 sim / token | x1.11 |
|---|---|---|---|---|
| baseline: trace handoffs 0.4 / 1.8, A_task 0.86, wake 0.6, t_v 0.26 (pre-registered) | 3.54 | 3.93 | 2.28 | 2.53 |
| handoffs = evx microbenchmark 2.2 / 3.9 | 4.31 | 4.79 | 2.56 | 2.84 |
| handoffs 0 (a perfect event) | 2.98 | 3.30 | 2.42 | 2.68 |
| argument path A_task x 0.5 / x 1.5 | 3.38 / 3.71 | 3.75 / 4.12 | 2.17 / 2.40 | 2.41 / 2.66 |
| wake-up spread 0.3 / 0.9 us | 3.51 / 3.58 | 3.89 / 3.97 | 2.29 / 2.28 | 2.54 / 2.53 |
| prologue round trip t_v 0.41 (idle HBM) / 0.70 (loaded) | 3.73 / 4.09 | 4.14 / 4.54 | 2.44 / 2.75 | 2.71 / 3.05 |
| everything pessimistic (evx, A x 1.5, t_v 0.7, wake 0.9) | 5.07 | 5.63 | 3.04 | 3.38 |

Reading a miss: 3.9-4.2 ms is the prediction; ~4.5 says the prologue loads see the loaded latency (the GEMV prologues
run while other tasks stream: plausible, and the fix is to issue them before the acquire); ~4.8 says the handoffs are
what evx measured, not what the trace showed; 5.6 says all three. Two requests are less sensitive to everything (the
second request's tasks fill the waits): 2.5-2.7 ms per token in every single-parameter variant, 3.4 at worst. The
two-request figure of section 16 (2.63) came from a cost table that did not parse the `r<k>_` task prefix (fixed in
`xcd_run.cost_fn`; every two-request task had been costed as a fixed small task): 2.53 replaces it as the
pre-registered number. The "hand-offs 0" row is slower than the baseline for two requests: the simulator's static
queue order, not a property of the hardware.

`examples/vllm_llm/batch_pp.py`: batching against a layer pipeline across XCDs, analytic, on the build-3 structure.
Bytes per token 6.08 GB (dense 1.84, 8 experts 3.62, lm_head 0.62): 1.71 ms at the 3.56 TB/s streaming rate; the
step's tax (sim 3.54 - floor 1.55 at 3.93) 1.99 ms. Batch of B on the whole GPU: tax + bytes(B) / BW, the experts the
union of B independent top-8 draws. Pipeline of S stages (8/S XCDs each, 48/S layers, micro-batch m = B/S, S x m
requests in flight): stage step = (tax - 0.4 ms device-wide handoff premium) / S + bytes(m) / S at f x (8/S)/8 of the
GPU's bandwidth, **f assumed** (1.0 / 0.8 / 0.6) until `hw_bench lfx` measures what 1 / 2 / 4 XCDs pull. All x1.11.

| B | batch: ms, tok/s | PP S=8 f=1.0 | PP S=8 f=0.6 | PP S=4 f=1.0 | PP S=2 f=1.0 | PP S=2 f=0.6 |
|---|---|---|---|---|---|---|
| 1 | 4.11, 244 | - | - | - | - | - |
| 2 | 5.16, 387 | - | - | - | 5.56, 360 | 8.09, 247 |
| 4 | 7.09, 564 | - | - | 9.35, 428 | 7.68, 521 | 11.62, 344 |
| 8 | 10.27, 779 | 16.94, 472 | 27.06, 296 | 13.59, 589 | 11.53, 694 | 18.03, 444 |
| 16 | 14.62, 1095 | 25.41, 630 | 41.18, 389 | 21.29, 752 | 17.88, 895 | 28.62, 559 |

(PP columns: latency per token = S x stage step, then tok/s.) With the same requests in flight, batching beats the
pipeline at every f and every S: the pipeline streams the dense weights once per micro-batch instead of once per batch
and only saves the device-wide handoff premium; per-token latency grows with S. Even f = 1 (a stage pulls its full
share of the bandwidth) does not close it. Pipelining is for the multi-GPU case; on one MI300X the order is batching
(B <= 8: 779 tok/s at 10.3 ms), then the tax. The lfx number settles f. It does not change the ranking.

## 18. The touch hook: the dense GEMVs' first weight trip during the wait (2026-09-29, pre-registered; ETX_VL_TOUCH=1)

What every megakernel that beats its launch-boundary baseline does (docs/core/SURVEY-2026-09-29.md: Hazy's instruction gap
10.2 -> 3.4 us from cross-instruction pipelining alone, Cohere's weights prefetched before the activation dependency,
MPK's descriptor prefetch): the weights depend on no event, so their first round trip need not wait for the acquire.

Mechanism (`TileBody.touch`, `etx/codegen/kernel.py`): a task type may name a `touch` device function. In phase 1 of
`etx_run_task`, the waves that are not waiting (threadIdx >= 64; wave 0 polls the event) call it before wave 0's wait,
so its loads complete at the barrier after the switch, which wave 0 reaches after its wait: the loads' latency overlaps
the wait, not the arrive or the body. (Issuing them during the arrive of the previous task, as first planned, would not
overlap the wait: the `__syncthreads` at the end of the loop and at the top of the next iteration fence every
outstanding load, so they would be waited for before the next wait began.) `-DETX_NO_TOUCH` removes it; the unfused
kernel has no wait and no touch.

`etx/ops/gemv.py` (`dsl_touch_plain`): for the dense forms (dsl_gemv, dsl_gemv_fold; not the expert forms, whose rows
depend on the top-k, that is, on the event), waves 1..15 load the first U vectors of their own segment range of the task's
first DSL_RCH-row pass, exactly what `dsl_gemv_rows` loads first: 15 x 64 lanes x 4 x 16 B = 60 KB per workgroup, one
trip, since 38 workgroups x 64 KB already take most of an XCD's 4 MB L2. `asm volatile` keeps the loads. The arg block
comes through the K$ as the tile's does. `ETX_VL_TOUCH=1` in model.py sets the hook on those grids (qkv, o_proj,
router, lm_head, plain and x forms).

Simulation (`etx/sim/xcd.py` `Cost.touch_bytes`, `xcd_run.cost_fn`): the touched bytes (one inflight, 64 KB) stream
from the moment the worker takes the task, before its events fire, sharing the bandwidth with every other stream
(Little's law as before). The body's stream of the rest starts when both the touch and front + chain are done, plus one
L2 round trip (0.143 us) to re-read the touched trip. What the model does not capture: L2 eviction of the touched
lines by the previous task's own stream passing through L2 (38 x 64 KB = 2.4 MB of 4 MB), and a wait shorter than the
touch's own round trip (then the touch delays the barrier by what the body would have paid anyway, +0.143). A first
version that simply dropped the trip from the stream (no bandwidth for the touched bytes) gave 3.13 / 3.47. It is
wrong because the touch of 304 workgroups is 19 MB, 5.5 us of bandwidth, longer than the wait it hides in.

`examples/vllm_llm/sensitivity.py`, build 3 (x1.11 = realistic):

| build 3 | R=1 sim | x1.11 | R=2 per token | x1.11 |
|---|---|---|---|---|
| baseline (section 17) | 3.54 | 3.93 | 2.28 | 2.53 |
| + touch | 3.20 | **3.56** | 2.18 | **2.42** |
| + touch, handoffs = evx 2.2 / 3.9 | 3.97 | 4.41 | 2.39 | 2.65 |

Pre-registered: build 3 with the touch hook ~3.56 ms per token (vLLM 4.67), two requests ~2.42 ms per token (413
tok/s). An analytic bound agrees on the order: per dense GEMV launch the touch fills the idle bandwidth of the wait
(handoff 1.8 + wake ~0.6 us of a 6 us qkv stream), ~2 us per launch, 48 layers x 2 launches + lm_head ~0.2-0.35 ms.
Container build (`rocm_build_check.sh qwen3-30b-a3b xcd_dsl3_touch`, and `xcd_dsl3x2_touch`): 10 VGPR spills, 36 B
scratch per lane, as without the hook. The session (`examples/sessions/dsl_gemv.py`) builds build 3 and its
two-request form with ETX_VL_TOUCH=1; build 2 stays the fallback without it.

## 19. The model was wrong in kind; what replaced it (2026-09-30, after the measurement)

Sections 10-18 modeled each task as a fixed cost plus dependent round trips, then counted events.
The model predicted large gains from folds and fewer events.
Hardware delivered only 0.7%.

| build | predicted (section 17-18) | measured |
|---|---|---|
| build 3 (DSL GEMV + head attention + fold + touch), fused | 3.56 | **5.895** |
| build 2 (no fold, no touch), fused | 4.45 | **5.934** |
| build 3, two requests, per token | 2.42 | **4.49** |

Both correctness gates passed, and the event count fell from 580 to 388.
The implementation matched the design. The cost model measured the wrong quantity.
Two measurements identify the correct quantity.

**A phase has a floor, and it does not depend on the bytes.** `hw_bench stream`, no events and no ETX, just N
workgroups reading B bytes each: 296 workgroups reading 896 B still take 3.2 us. Spans at 296 workgroups,
averaged over two runs, `contig` = one contiguous block per workgroup (what `dsl_gemv_rows` does), `rows` = row r
to worker r % W:

| bytes per workgroup | contig us | rows us | aggregate contig | aggregate rows |
|---|---|---|---|---|
| 896 | 3.30 | 3.16 | 0.87 TB/s | 1.66 |
| 28 672 | 7.05 | 4.86 | 2.10 | 4.56 |
| 51 200 | 10.09 | 6.16 | 2.14 | 4.80 |
| 159 744 | 17.84 | 16.64 | 3.19 | 3.47 |
| 319 488 | 34.43 | 32.16 | 3.01 | 3.24 |
| 1 048 576 | 91.32 | 87.93 | 3.51 | 3.65 |
| 319 488, 8 workgroups | 12.10 | 11.74 | 0.21 | 0.22 |

Two things follow. Achievable bandwidth is not one number: a workgroup reading 28 KB keeps a shallower queue than
one reading 1 MB, and the device only reaches 3.5 TB/s at the deep end. And the `rows` rows below 50 KB per
workgroup imply 4.6-5.8 TB/s, above anything HBM can deliver -- those phases total 8-15 MB and fit in the 256 MB
Infinity Cache. A weight stream (6.2 GB per token) never does, so `etx.sim.phase.agg_bw` clamps to 3.55 TB/s and
the model does not inherit them. **The 30-60% row-mapping gain reported from the raw table is largely a MALL
artifact; between HBM-resident points it is 4-7%.**

**Cost follows phase width, not the number of phases.**
Build 3 has 8 phases per layer and takes 5.895 ms.
The qknr build has 13 and takes 5.875 ms.
The five extra narrow phases, with 8 tasks each, add no measurable cost.

A 304-task phase costs about 10 us even with few bytes.
`router` spreads 0.5 MB over 304 tasks and takes 11.7 us per layer, or 0.56 ms per token.
Thus, each task in an n-task phase receives `(floor + etx) * min(1, n/296)` cost.

### The model (`etx/sim/phase.py`, `examples/vllm_llm/pmodel.py`)

    task = (floor_us + etx_us + fused_us) * min(1, n_tasks / 296)      fixed, paid in parallel across the phase
         + chain_us                                                     known dependent work (attention partitions)
         + bytes_per_task / min(cu_rate, agg_bw(bytes_per_task) / n)    measured rates, HBM-clamped

The hardware floor, `floor_us`, is 3.0 us.
Only `etx_us` at 7.5 us and `fused_us` at 1.8 us are fitted constants, from eight measured builds.
`fused_us` applies only to the megakernel and is **positive**.
Build 3 takes 5.895 ms fused and 5.660 unfused. The qknr pair is 5.875 / 5.623.

An unfused launch also pays a 1.61 us kernel boundary per phase.
Yet fusion costs about 2 us more per phase than separate launches of the same tile.
This contradicts ETX's intended benefit and defines the next target.

### Validation (`python3 -m examples.vllm_llm.validate`)

| build | predicted | measured | error | gate |
|---|---|---|---|---|
| build 3 fused | 5.722 | 5.895 | -2.9% | 10% |
| build 2 fused | 5.929 | 5.934 | -0.1% | 10% |
| build 3 unfused | 5.451 | 5.660 | -3.7% | 10% |
| build 3, two requests (per step) | 9.807 | 8.988 | +9.1% | 10% |
| qknr fused | 6.493 | 5.875 | +10.5% | 15% |
| qknr unfused | 6.338 | 5.623 | +12.7% | 15% |
| moe new | 6.493 | 6.313 | +2.9% | 15% |
| moe old | 6.493 | 6.556 | -1.0% | 15% |

The model cannot distinguish the three 2026-09-28b builds because their phases and widths are identical.
It predicts one result, but they measured 5.875 / 6.313 / 6.556.
Their 0.68 ms spread comes from argument-path implementations, which the new model does not represent.
The result fits the 15% tolerance only because that tolerance is wide.
The two-request build has the worst fit at +9.1%.

### What the corrected model chooses

`autoconf` over the build switches now ranks every one of the 32 configurations between 5.93 and 6.40 ms: chain,
the fused q/k norm + rope, the DSL GEMVs, the DSL top-k and the head attention block all move the step by less than
0.5 ms, and the best is what build 2 and build 3 already are. **The switch space is exhausted.** The lever the
model points at is phase width, which is not a switch today (`examples/vllm_llm/pmodel.py width_sweep`):

| phase | MB per layer | tasks now | us now | best tasks | us | ms per token |
|---|---|---|---|---|---|---|
| qkv | 20.97 | 304 | 21.38 | 88 | 10.44 | 0.525 |
| o_proj | 16.78 | 304 | 20.01 | 90 | 9.05 | 0.526 |
| router | 0.52 | 304 | 12.82 | 19 | 1.58 | 0.540 |
| moe_up | 50.33 | 192 | 24.36 | 91 | 19.65 | 0.226 |
| moe_down | 25.17 | 256 | 20.34 | 87 | 11.89 | 0.406 |
| | | | | | | **2.22** |

Narrowing every wide phase to about 90 tasks is worth about 2.2 ms per token under this model -- far more than any
switch, and the direct consequence of the old cost model having searched with the wrong objective.

The interleaved row mapping (replacing the contiguous `[r0, r1)` of `dsl_gemv_rows`) is worth 0.50 ms on build 3
(5.722 -> 5.221 with everything else equal, 8.8%), not the 30-60% the raw table suggested, and it is the one lever
whose size the model can only bound: the rows points that would settle it below 50 KB per workgroup were the ones
the Infinity Cache contaminated. Not implemented.

## 20. ETX_VL_COH: the acquire stops destroying the cache (2026-09-30, pre-registered)

Section 19 has positive `fused_us`: an in-kernel phase boundary costs more than a kernel launch.
The lowering identifies the mechanism: every task uses `buffer_inv sc1` from gfx942.yaml for its acquire.
This invalidates the whole XCD L2 and makes each phase as cold as a new kernel.

It also cancels the section 18 touch hook.
Waves 1-15 fetch weights into L2 during the wait. Thread 0's acquire then discards them before the body.
ETX's own memory protocol disabled cache continuity across phases, its structural advantage.

The redesign (`ETX_VL_COH=1`, commit dc9105d) splits by data class: DSL tiles read and write ACTIVATIONS with
coherent `sc0 sc1` accesses (weights stay plain cached loads) and skip the acquire inv entirely; imported kernels
(attn, reduce) keep `buffer_inv sc1`. An imported producer feeding a coherent consumer keeps a `buffer_wbl2 sc1`
release (the reduce -> o_proj stale-L2 trap); events whose producers are all coherent arrive without the L2
write-back. TLA+ (`formal/DataVis.tla`): fresh-data guarantee holds for every class x placement combination, and
dropping the write-back on an imported->coherent edge is caught (negative test). Container build: spill 9 (was 10).

**What the model can and cannot say.** The fitted `etx_us + fused_us` (9.3 us per phase width) bundles acquire,
argument path, release and wave start. The inv + write-back share is bounded by measured pieces only: 0.5 us
(idle inv 0.19 + fence 0.18) to 1.5 us (loaded inv 0.63 + a loaded wbl2 arrive). The touch's value under COH is
the sim's overlap of the dense first trips with the wait, which was optimistic once already (section 18), so the
low end is capped. `python3 -m examples.vllm_llm.pmodel <plan> --coh`:

| build 4 = build 3 + COH | model (base 5.722) | anchored to the 5.895 measurement |
|---|---|---|
| overhead share alone | 5.39-5.61 | 5.57-5.79 |
| touch also landing | 4.69-4.91 | **4.9-5.3** (touch capped at 0.3-0.7 ms) |
| two requests, per step | 7.81-9.61 | 7.4-8.8 |

**Readings, pre-registered.** ~5.9: the inv was never the cost; the overhead lives in the argument path or the
scheduling loop (cohload will show coherent ~ inv + plain). 5.6-5.8: the overhead share moved, the touch still
does not land (the trace's dense waits stay exposed; restart says how much was left). 4.9-5.3: both landed.
Below 4.9: the model was conservative; restart and cohload say why. Independent of the number: fused < unfused
flips the fusion penalty, which is the claim ETX is built on; a --check-dsl or token-match failure means the
implementation diverges from the TLA-checked design (the design itself holds). vLLM stands at 4.67.

### Build 5: + ETX_VL_ARGPF + ETX_VL_BAR2 (the other two boundary layers, same session)

The remaining decomposition of the software boundary against the 1.61 us hardware launch: the argument path (a
task fetches its 136 B block through scalar loads at use; the 2026-09-28b builds measured 0.24 ms between
argument-path implementations, all-or-nothing per step) and the loop's four full-workgroup __syncthreads per
task. `ETX_VL_ARGPF=1`: wave 1 pulls the NEXT static task's block (per-type offset + this domain's stride) while
thread 0 runs the arrive; with COH no invalidate follows, so the lines survive into the next task's reads.
`ETX_VL_BAR2=1`: thread 0 advances the queue into a double-buffered shared slot BEFORE the post-drain barrier,
which then doubles as the broadcast; the loop-top and loop-end barriers disappear (two barriers per task, not
four. The write-after-read hazard is closed by the parity slots). Both are emission switches on the graph,
default off; off = byte-identical emission (tests/test_bar2_argpf.py). Container: spill 9, tasks/events
unchanged (66938/388).

The model bounds the pair's savings at 0.10-0.35 ms per step.
The argument path contributes at most its measured 0.24 ms.
Removal of one to two barriers at measured s_swbar 0.106-0.282 us across 384 phases accounts for the rest.
This promises no floor crossing: `pmodel <plan> --all` clamps at 4.6 ms.

| build 5 = build 4 + ARGPF + BAR2 | anchored range |
|---|---|
| overhead share alone | 5.25-5.7 |
| touch also landing | **4.6-5.2** |
| two requests, per step | 7.1-8.7 |

**Readings.** Build 5 vs build 4 (one extra fused run) isolates ARGPF+BAR2 directly: < 0.05 ms apart means the
argument path and barriers were already covered (or never exposed); ~0.24 ms apart means the argument path was
the exposed share, as the 2026-09-28b spread suggested. Build 5 fused < 5.660 (build 3's unfused) flips the
fusion penalty. A BAR2 hang or token mismatch indicts the two-barrier loop's broadcast fold (the parity-slot
argument above), not the coherence design.

## 21. Fixed 2026-10-08 tile profile (CPU validation, 2026-10-09)

The default `2026-09-30` profile is unchanged. On the current
`build/rocm22/qwen3-30b-a3b_default/plan.json` it predicts 5.476 ms and a fusion gain of
-179.5 us, against 4.393 ms device time and +225 us measured gain. Its constants describe
older builds. `--profile 2026-10-08` adds a separate profile; no constant is refitted to
this session. The same kernels in a HIP graph remain the fusion baseline; the system
baseline remains vLLM 0.27.1 default, whose pre-registered verdict is INSUFFICIENT
(see [NVREF](NVREF-2026-10-09.md), including the explicitly post hoc healthy-window estimate).

Conditions: one MI300X VF, Qwen3-30B-A3B bf16, batch 1, prompt 1024, 32 greedy tokens.
Predictions below are CPU simulations, not new GPU measurements. Device references are
`results_nvref_2026-10-09/report.log:9` and `fusion_pairs.txt:2-6`; wall references are
`report.log:7-8`. The validator requires lines 2 onward of the plan's `vl_plan.txt` to
match `results_nvref_2026-10-09/vlplan_all.txt`, otherwise it fails with `plan differs`.

| Quantity (ms) | Model | Measured | Model minus measured | Gate |
|---|---:|---:|---:|---|
| Tiles fused, device | 4.058 | 4.393 | -0.335 (-7.6%) | within 10% |
| Tiles unfused, device | 4.246 | 4.617 | -0.371 (-8.0%) | within 10% |
| Fusion gain, unfused minus fused | 0.188 | 0.225 | -0.037 | absolute error < 0.093 |
| Tiles fused, host wall | 4.198 | 4.526 | -0.328 | information only |
| Tiles unfused, host wall | 4.282 | 4.657 | -0.375 | information only |

The measured gain is the paired statistic, not subtraction of rounded aggregate times.
The eight historical validation rows retain their old profile and are the only rows
included in `--fit`.

### Constants fixed before this validation

- ETX overhead `e = 0`, ignoring `etx_us`, `fused_us` and `coh_save_us`; no touch is applied.
  The existing width-scaled hardware floor remains (`phase.FLOOR_US = 3.0 us`).
- Streaming roles `qkv`, `o_proj`, `router`, `moe_up`, `moe_down`, `lm_head` with positive
  bytes use `phase.BW_HBM = 3.55e6 bytes/us` and `cu_rate = 1e9 bytes/us` (a nonbinding
  per-CU limit). These reuse the existing hardware bandwidth constant, not a new fit.
- Narrow `chain_us`: prep 6.04, reduce 5.58, topk 8.35, prologue 11.3, in_norm 5.6,
  argmax 10.1 us. Sources: round11 critical bodies in `docs/qwen/FLOORS-2026-10-08.md:99,101,104`
  and tail times at lines 116,117,119. These are from another build; the current build
  is predicted out of sample. Attention retains its existing 1.9 us chain plus its
  existing stream cost, not a 1.9 us total body.
- Fused handoffs retain 0.4 us DOMAIN / 1.8 us DEVICE. Unfused uses 1.61 us for both,
  the existing `hw_bench launch` measurement.
- Host excess: fused 140 us, the median of wall minus device median in
  `results_tiles/ab{1,2,3}_all_x.log:8`; unfused 36 us from
  `docs/qwen/VLLM-LADDER-2026-10-01.md:39`. Both predate NVREF. `wall_ms` adds the appropriate
  excess to the simulated step (the legacy `step_us` function returns milliseconds).

### Per-phase check against the current trace

Measured mean bodies below are means over layers 1-45 from `results_tiles/tr_all.bin`,
recomputed by `etx.tools.floors.round11_from_trace`; the readable evidence is
`results_tiles/tr_all.log:7-442`. The trace plan matches the session plan as described
above. These current-build measurements are validation data, not the source of the
narrow-body constants.

| Phase | Model component (us) | Current measured mean body (us) | Component meaning |
|---|---:|---:|---|
| qkv | 8.907 | 8.933 | full floor + role bytes / HBM |
| o_proj | 7.726 | 5.347 | full floor + role bytes / HBM |
| router | 3.148 | 2.760 | full floor + role bytes / HBM |
| moe_up | 17.178 | 16.964 | full floor + role bytes / HBM |
| moe_down | 10.089 | 9.887 | full floor + role bytes / HBM |
| prep | 6.040 | 3.837 | chain only |
| reduce | 5.580 | 5.336 | chain only |
| topk | 8.350 | 7.697 | chain only |
| attn | 1.900 | 5.315 | chain only; stream cost is additional |

This component check is not an additive decomposition of the simulated makespan: the
simulator also accounts for width scaling, placement, contention and handoffs. The
three stream checks required here (qkv, moe_up, moe_down) agree within 3% using the
full-floor expression. The other rows expose limitations rather than new fit targets.

**Unexplained residual:** the fused prediction still has -335 us error (4.058 minus
4.393 ms). Its cause is unknown; no compensating constant is added. Correct fusion
gain sign and a passing tolerance do not establish a causal decomposition or full
predictive accuracy. The wall estimates are informational only.

**Pending GPU measurements:** replicated, interleaved fused/unfused timing with the
same build; byte-identical import checks for all 97 imported instances in that timed
build (the existing check covers only 5, in another build); phase body/start/handoff
traces and host/device excess from the same build to locate the residual; and nvref2
with `--repeats 5` under the unchanged rule to resolve the vLLM system comparison.
None of these GPU measurements was run for this CPU-only implementation.
