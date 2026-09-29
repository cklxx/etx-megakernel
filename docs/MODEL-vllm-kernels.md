# vLLM's decode kernels on MI300X, in vLLM's form and fused by ETX: operation model, calibration, XCD simulation

Status: 2026-09-28. Built after session moe_gate (results_2026-09-28) to replace "measure, then guess" with "model,
then measure what the model cannot explain". Every number is labelled measured, fitted (a free parameter set from
data) or predicted (from the model with no parameter fitted to that number).

Tools: `examples/vllm_llm/ops/hip.md`, `ops/fused_moe.md` (operation inventories, cited to source and assembly lines),
`isa_build.sh` + `isa_inventory.py` (instruction counts, vLLM's form vs the imported form, ROCm 7.2.4 compiler),
`kmodel.py` (the cost model and its fit), `etx/sim/xcd.py` + `xcd_run.py` (the XCD simulation), `measured.py` (every
measured per-role number in one JSON), `compare.py` (ledgers).

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
| effective stream bandwidth | 5.2 TB/s | fitted (at the grid's edge; includes overlap the model does not describe) | kmodel fit |
| ETX unfused: extra front per launch A_unf | 0.65 us | fitted to ETX's unfused launches without software barriers | kmodel fit |
| ETX task: front of every imported kernel A_task | 2.1 us | fitted to ETX's small fused tasks | kmodel fit |
| hand-off XCD-local / device-wide | 1.0 / 1.3 us | measured (hw_bench handoff, padded counters) | used unchanged by the simulator |
| wake-up spread / per-task stream-rate spread | 0.6 us / 3% | fitted to the measured ramps and tails | simulator |

The stream of every kernel follows Little's law: a trip's memory time is the loaded latency max(t_v, bytes in flight on
the GPU / bandwidth), and trips = bytes per block / bytes in flight per CU; work of a trip not overlapped with memory
(fused_moe's LDS write / barrier / read, 0.33 us) adds to every trip.

## 3. The kernels as pipelines (ops/hip.md section 13, ops/fused_moe.md)

At batch 1 every kernel is a dependent chain. What decides each one's time:

| kernel | chain per block | stream | bound (vLLM form) |
|---|---|---|---|
| fused_add_rms_norm | 2 S, 2 V, 10 LDS steps, 2 barriers; the weight load waits for barrier 2 | - | latency |
| wvSplitK (qkv / o_proj / router) | 4 S; the whole activation into LDS; barrier; then weight trips, each waiting for its own 4 x 16 B loads (not pipelined); 8-16 DPP | 16 KB (4 waves) or 36 KB (9 waves) in flight per CU | bandwidth (qkv, o_proj), latency (router: 32 CUs) |
| rms_norm_3d, rope | 4 S + 2 V + 2 barriers; rope 5 V of 2 B scalar loads | - | latency |
| reshape_and_cache | 9 V: the V copy is 8 serial 2 B load / store rounds | - | latency |
| paged attention mfma16 | 4 S + 7 V (2 scratch reloads), 7 barriers, 8 of 16 MFMA rows real | 128 KB KV per block | latency at 1024 tokens |
| reduce | 3 S + 1 V, 12 LDS steps, 1 barrier | - | latency |
| topkGating | 2 S + 5 V, 40 dependent LDS steps, ~1500 instructions, 16 lanes | - | issue / latency |
| silu (MoE) | 2 S + 1 V, ~340 VALU (expf + IEEE divide) | - | issue |
| moe_sum | 2 S + 1 V, 8 x 16 B loads | - | latency |
| fused_moe gate_up / down | 2 S + 2-3 V prologue; K tiles of 16 KB, 2 barriers each (31 / 11 per program), the next tile's loads issued after this tile's LDS work | 16 KB in flight per CU | bandwidth + 0.33 us exposed per tile |

Imported form (`isa_inventory.py`): the same instructions, except that every barrier of a kernel whose block is smaller
than the 1024-thread workgroup (attention, reduce, rms_norm_3d, fused_moe) becomes a software slot barrier, and the
fused_moe prologue chains 5-6 dependent round trips instead of 2.

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

**The imported fused_moe explained:** vLLM's is bandwidth-bound (48 MB at the loaded latency); in ETX's form each K tile
adds two software barriers on its critical path (+0.37 us per tile), which moves it to latency-bound: gate_up +7.1 us,
down +3.4 us, predicted within 0.65 us of both unfused measurements; no ETX-side parameter was fitted to fused_moe
(A_unf was fitted on the kernels without software barriers).

## 5. The XCD simulation (etx/sim/xcd.py)

The compiled plan (67,994 tasks, every worker's static queue, every event with its producers and scope) runs task by
task: wait for its events (last producer's end + the scope's hand-off + wake-up spread), front, chain, stream (Little's
law, shared by every CU streaming at that moment), end. The output has ETX_TRACE's layout, so the measured and the
simulated step go through the same ledger.

MoE, ETX_VL_XCD schedule (one selected expert per XCD), per layer (us, layers 2-45):

| | hand-offs | ramp | bodies | tails | layer | token |
|---|---|---|---|---|---|---|
| measured | 13.7 | 4.5 | 102.2 | 10.7 | 131.0 | 6.63 ms |
| simulated | | | | | 121.2 | 5.84 ms |

The simulation is 12% optimistic. Where the difference sits: device-wide hand-offs measured 1.5-2.1 us against 1.3
(they include the first read of data another XCD wrote), XCD-local hand-offs measured 0.3-0.5 us against 1.0 (faster
than the microbenchmark: one producer, readers on the same L2), and the bodies of reduce, topk, post_norm and moe_sum
(1-2 us each, section 7).

What the XCD view shows (the page's timeline, layer 5): the attention block (q/k norm + rope + cache in one task, then
attention on 5 CUs and the reduction on 4) runs while about 300 CUs wait: 38 us of the 131; the expert path runs 24 (gate_up) and 32 (down) CUs per XCD, each XCD on its own expert, and
an XCD that finishes its expert starts the next launch without waiting for the others (negative hand-offs in the ledger);
top-k and moe_sum are the two points where all XCDs meet.

## 6. What the model says

Per imported kernel, a fused task costs its front (2.1 us, fitted) plus a hand-off (1.0-1.3 us) where vLLM pays one
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
| A_task = 2.1 us per imported kernel (chained parts pay it too) | 720 tasks x 2.1 = 1.5 ms per token | the argument chain (args[0] -> argument block -> parameters) before the first load, 5-6 round trips for fused_moe | build with the argument block read before the wait; trace: body shrinks by the predicted amount or not |
| software barrier cost | 0.26 us predicted | the slot barrier's LDS round trips | build with hardware barrier rounds; fused_moe gate_up task body 21.9 -> ~15 us predicted |
| ETX unfused excess not explained (reduce +2.0, topk +1.3, cache +0.8, silu 1.5B +1.5 us) | 5-7 us per layer | instruction-cache misses (the unfused kernel holds every tile's code), slot packing (reduce: 8 blocks per workgroup) | SQC_ICACHE_MISSES per launch, ETX unfused vs vLLM; reduce with slots = 1 |
| device hand-off 1.5-2.1 us vs 1.3 | ~4 us per layer | first read of remote-written data (staging) | trace: split ready -> first load |
| vLLM attention 7.0 vs 5.7 us | 1.3 us | the model's attention chain (KV loads at 1024 tokens) | from vLLM's trace at shorter contexts (no GPU needed if the traces exist) |

Correctness question raised by the inventory: imported kernels read the per-step scalars (positions, seq_lens,
slot_mapping) with scalar loads, and ETX's acquires invalidate the vector caches, not the scalar cache. Not a bug in the
current host: every step is its own megakernel launch (the kernel start invalidates the scalar cache), and within a launch
every reader waits (transitively) on the prologue that writes them, so a CU's first read of them in a launch comes after
the write. Evidence: every run decodes 1055 steps with a new position each and matches 32/32; a stale position would
write the KV cache at the wrong slot. It would become a bug if one launch ran several steps.

## 8. What each lever is worth (examples/vllm_llm/whatif_sim.py, simulated; ~12% optimistic today)

The CU time of a measured layer (layer 5, 304 CUs x 130.7 us): 32% in tasks, 51% on CUs the launch has no task for
(1-8-task launches: norms, top-k, moe_sum, the attention block; 192 / 256 of 304 CUs in the expert launches), 10% in
hand-offs between launches, 7% in ramps and tails. Weight streaming is 31% of the CU time; the layer's 116 MB would take
27 us at 4.3 TB/s. The empty CUs are waiting on the dependency chain, not moving data.

| one request | ms per token | tokens/s |
|---|---|---|
| today | 5.84 | 171 |
| hardware barriers for imports | 5.31 | 188 |
| argument path 2.1 -> 0.5 us | 4.61 | 217 |
| both | 4.08 | 245 |
| both + hand-offs halved | 3.71 | 270 |
| no argument path, no hand-off (bound for these kernels) | 2.95 | 339 |
| vLLM, measured | 4.90 | 204 |

| two requests interleaved, one workgroup per CU (request 1 fills the CUs request 0 leaves idle) | ms per step | tokens/s |
|---|---|---|
| today | 7.38-7.91 | 253-271 (x1.48-1.58) |
| both fixes | 5.28 | 379 (x1.54) |
| vLLM batch 2, measured | 5.72 | 349 |

Reading: the argument path is the largest single lever (1.2 ms), the barriers second (0.5 ms). With both, one request
lands at vLLM's latency (4.1 simulated, ~4.6 with the simulation's bias) and two interleaved requests at +9% over vLLM's
batch 2 with lower latency per request, inside the simulation's error. Even with free arguments and free hand-offs these
kernels take 2.95 ms at batch 1 (their own dependent round trips and synchronisation; floor 1.44): going further takes
different kernels or more work per step (more requests, speculative decoding), not scheduling.

## 9. The fixes as built (2026-09-28, local; session examples/sessions/moe_fix.py)

- **Hardware barriers in rounds** (etx/runtime/include/etx/import.h, etx/importer/adapter.py): an imported kernel's
  __syncthreads is the workgroup's `s_barrier`; around the blocks of one launch every wave runs a round protocol: count
  itself done after its block (or at once if it has none), then repeat `s_barrier` until all 16 waves are done. Blocks of
  one kernel reach the same barriers in the same order, so each round is every unfinished wave's next barrier; an early
  `return` inside a kernel joins the rounds, as a terminated wave leaves the hardware barrier count. `-DETX_IMP_SW_BARRIER`
  restores the software slot barrier.
- **Argument path** (import.h `etx_imp_argblock`, codegen `pre_wait` hook): the argument-block pointer is made
  wave-uniform, so the kernel's parameter reads compile to scalar loads (megakernel: 95 -> 2644 `s_load`, i.e. they had
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
| old (all three switches) | 6.63 (today; otherwise compare ratios) |
| hardware barriers only | 6.03 |
| argument path only, front 0.3 / 1.0 / 1.5 us | 5.05 / 5.67 / 6.10 |
| both | 4.45 / 5.07 / 5.51 |
| two requests, both | 347 / 304 / 278 tokens/s (vLLM batch 2: 349) |
| fused_moe gate_up task body (trace) | 21.9 -> ~15 us |

The trace of the new build measures the front directly; that number decides which column applies. Beating vLLM at batch
1 (4.90) needs the front at about 0.5 us or less.

## 10. Every parameter from a microbenchmark (pre-registered 2026-09-28, before the session)

Sections 4-9 fit six numbers to launch times (F, t_s, t_v, t_i, BW, A_task) and assume three more (t_swbar from an
instruction count, the hand-offs from a benchmark whose protocol is not ETX's). Fitted values can absorb a wrong
structure: t_v = 0.2 us and t_s = 0 are below any physical round trip on MI300X, so the fit has most likely traded
them against F (2.5 us against a measured 1.6 us kernel boundary) and against round-trip counts that overlap in
reality. The session measures each one directly, first, on GPU 0 (`bench/hw/hw_bench.hip` lat / lf / launch / evx,
`bench/hw/imp_bench.hip` in both barrier protocols, with `etx/import.h` itself), and
`examples/vllm_llm/bench_params.py` maps them onto the model (the mapping is in its docstring) and simulates the step
with microbenchmark values only.

| measurement | stands for | fitted / assumed | predicted range | if outside |
|---|---|---|---|---|
| lat vector MALL idle | t_v | 0.2 | 0.25-0.5 us | above 0.3: the V counts of section 3 overcount dependent trips; recount from the ISA, do not refit |
| lat vector L1, L2, HBM idle | levels | - | 0.05-0.15, 0.1-0.3, 0.3-0.8 us | |
| loaded / idle, same level | Little's-law latency | - | x1.5-3 at full rate | |
| lat scalar K$ after inv sc1 | the K$ survives ETX's acquire | assumed | < 0.1 us (a K$ hit) | the argument fix of section 9 has no effect: the scalar path is also invalidated |
| lat vector L2 after inv sc1, x2 | A_task, vector build | 2.1 | 0.6-1.2 us | below 1.0: A_task holds more than the argument path (other loads of the adapter) |
| lf, highest rate | BW | 5.2 TB/s | 4.0-4.5 TB/s at >= 4 MB in flight | 5.2 is not physical (it absorbed a missing term) |
| lf 192 WGs x 16 KB (fused_moe) | fused_moe stream rate | 3.2 TB/s (model) | 2.5-3.5 TB/s | the gate_up body cannot reach 15 us |
| launch empty, 304 x 1024 | F | 2.5 | 1.5-1.8 us (sync_bench: 1.6) | |
| launch hops 0 - empty | first load after a boundary | - | 0.3-0.8 us | |
| evx xcd P 37 C 37 median | DOMAIN hand-off | 1.0 | 0.3-0.6 us (the trace's 0.3-0.5) | ETX's poll is not what the trace saw |
| evx device P 296 C 296 median | DEVICE hand-off | 1.3 | 1.3-2.1 us (the trace's 1.5-2.1) | |
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
which write different buffers, do not overlap either; splitting the chain into tasks only adds hand-offs, simulated
6.05 against 5.84 ms.) vLLM ships `tensorrt_llm::kernels::fusedQKNormRopeKernel` (csrc/libtorch_stable/
fused_qknorm_rope_kernel.cu, ROCm paths included), enabled by compilation_config pass_config
enable_qk_norm_rope_fusion (off by default; it adds +rotary_embedding). One 32-lane warp per head, 8 per 256-thread
block, 5 blocks for 32 + 4 heads, in place on qkv; per warp: the qkv and position loads, a 5-step reduce, cos/sin
after the position, the store (2 dependent round trips, from the source; the kernel model's `qknr` entry). Imported:
80 B of arguments, no LDS, no barrier, 4 slots per workgroup; the megakernel's registers are unchanged (VGPR 128, the
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
hidden). Baseline: vLLM with the same pass on, same image, same VM, beside the default; the rocprof trace must show
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
| evx XCD P 37 C 37 / device 296 / 296 | 0.3-0.6 / 1.3-2.1 | 2.20 / 3.91 us | NOT the trace's hand-off: unexplained, not used |

Derived (examples/vllm_llm/bench_params.py): the argument path under load, vector build = one invalidate (0.63) + two
hops that miss it (2 x 0.51) = 1.65 us (fitted A_task 2.1); scalar build = 0.63 + two K$ hits = 0.86 us. So the
argument fix saves ~0.8 us per imported task, not the 1.6-1.8 the section 9 table's best column assumed.

The simulation with microbenchmark values only (t_v, t_s, F, BW, t_swbar, the round barrier, A_task; hand-offs from
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
head), top-k as one tile, moe_sum folded into the next in_norm; the streaming ops stay vLLM's. Costs: kmodel DSL_MOE
(estimated from the chains); parameters from the microbenchmarks, hand-offs from the trace; examples/vllm_llm/dsl_ceiling.py.

| plan | simulated | measured | realistic (sim x 1.12) |
|---|---|---|---|
| xcd, four small launches chained | 5.54 | 6.31 | |
| xcd + fused q/k norm + rope (grid) | 5.26 | 5.86 | |
| DSL: fused small ops | 4.55 | - | ~5.1 (vLLM 4.67) |
| DSL, two interleaved requests | 6.16 ms/step, 325 tok/s | - | ~290 tok/s (vLLM batch 2: 360) |

Layer 5: 113 -> 108 -> 92.5 us; the small-op savings are 15 us per layer, not the 39 estimated by hand. The go/no-go
threshold of 4.4 ms is not met. What is left in a layer at 92.5 us: 29.5 us of bytes at 3.93 TB/s, ~28 us of
latency-bound tasks, ~15 us of hand-offs, and ~20 us of streaming excess inside vLLM's own kernels (fused_moe gate_up
21.6 us per task against 12.8 at full rate: 16 KB in flight per program and a 0.33 us gap per trip; down 11.6 vs 6.4;
qkv 7.9 vs 5.3; o_proj 7.0 vs 4.3). That excess (~14 us/layer, ~0.65 ms/token) is the only lever that reaches vLLM at
batch 1, and it means generating the streaming GEMVs in the DSL instead of importing them.

## 13. Direction A: the streaming GEMVs generated by the DSL (2026-09-29, pre-registered)

Decision (the user's, 2026-09-29): direction A, the DSL generates the streaming GEMVs; the small ops follow. The op:
`etx/ops/gemv.py` emits the tiles and the argument layout in the adapter's form (three exports: dsl_gemv, dsl_moe_up
with silu fused into the epilogue, dsl_moe_down with the routing weight); model.py links them with ETX_VL_DSL_GEMV=1;
the host fills their blocks and, with --check-dsl, checks the last layer's outputs against a CPU reference (2 bf16
ulps; the summation order differs from vLLM's kernels, so byte identity is not the criterion). The stream loop is
examples/llm/gemv.h's at 4 (16 B) or 8 (8 B) loads per lane in flight: 64 KB per 16-wave workgroup (16 per lane, the
4-wave design, spilled 359 VGPRs in the 128-register megakernel). ROCm 7.2.4 build: 7 spills (2 before). The ISA of
the standalone tile: 4 global_load_dwordx4 back to back per batch, one 16 B load per thread for the staged x.

Simulated (microbenchmark parameters), ms per token:

| plan | simulated | realistic (x1.11) |
|---|---|---|
| today's best (xcd + fused q/k norm + rope, grid) | 5.26 | measured 5.86 |
| build 1: + DSL GEMVs | 4.58 | ~5.1 |
| build 1 + DSL top-k (etx/ops/topk.py) | 4.42 | ~4.9 |
| build 2: + the attention block by kv head (etx/ops/attnprep.py on XCD h; vLLM's attention at one block per task and its reduce on XCD h; the session, examples/sessions/dsl_gemv.py) | 4.01 | ~4.45 |
| build 2, two interleaved requests (ETX_VL_REQUESTS=2; the session's run_two) | 5.20 ms / step | ~2.89 ms per token, 347 tok/s (vLLM batch 2: 360) |
| + DSL top-k and the moe_sum fold (phase 1) | 4.36 | ~4.85 |
| + attention with its preparation in one task (all of A) | 4.00 | ~4.45 (vLLM 4.67) |

The session's build is build 2, predicted at 4.4-4.5 ms (vLLM 4.67). Build 2 gets A's attention gain without a DSL
attention kernel: the preparation of one kv head (norm + rope + cache write) is a 9-wave DSL task on XCD h, and the
head's attention partitions and reduce sit on XCD h, so qkv -> prep and reduce -> o_proj are the block's only
device-wide hand-offs (five before). GQA's topology (a kv head's q rows span two XCDs at full qkv bandwidth) is why the
qkv -> prep hand-off stays device-wide. Gates: 32/32 tokens, CHECK-DSL OK. If build 1 measures above 5.4 ms the GEMV tiles are not at the lf rate
and the trace (per-task bodies of qkv / o_proj / moe_up / moe_down against 5.3 / 4.3 / 12.8 / 6.4 us) says which.

## 14. Formal checks (2026-09-29, `formal/`, `bash formal/check.sh [old]`)

| what | tool | claim | result |
|---|---|---|---|
| the DSL GEMV stream loop (etx/ops/gemv.py: segments in contiguous per-wave runs, flush on row change, per-wave partials summed) | Lean 4 (core, no Mathlib), `formal/Gemv.lean` | `gemv_correct`: for every row, the sum over waves of the partials equals the sum of the row's segment values, for any spr, S, per >= 1 and any wave count covering S (so the flush bookkeeping and the tail path lose or duplicate nothing) | proved; axioms `propext`, `Quot.sound` only |
| the imported-kernel barrier protocol (import.h: counted s_barrier in rounds, exit by round number) | TLA+ / TLC, `formal/ImportRounds.tla` | no deadlock; every wave of a use has taken the same number of barriers when it leaves (SameNbarAtBoundary); no wave leaves while another still has block barriers to take (NoOrphanBarrier); termination under weak fairness; shared-memory steps interleave freely | NW 3, 2 uses, 0-2 barriers per wave: 271,503 states, no error; NW 4: 9,397,101 states, no error |
| the first version of that protocol (leave when done = NW, reset at the next begin: the race the 2026-09-28 review found) | `formal/ImportRoundsOld.tla` | the same claims | TypeOK violated (done exceeds NW: a wave of the next use increments before the reset): the model discriminates |
| the DEVICE event with the last-arriver flush (primitives.h: per-domain sub-counter, the domain's last producer writes back and subtracts the share; consumers poll <= 0 then acquire) | `formal/EventFlush.tla` | the word never goes negative; it reaches 0 only when every producer's output is visible device-wide; a consumer past its wait reads every output (no stale read); one flush per domain; termination | 3 domains x 2 producers, 2 consumers: 1,848 states, no error |

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
