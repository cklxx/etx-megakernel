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

Correctness risk found by the inventory (not performance): imported kernels read the per-step scalars (positions,
seq_lens, slot_mapping) with scalar loads; ETX's acquires invalidate the vector caches but not the scalar cache, so a
persistent megakernel could read a previous step's value. Every run so far matched 32/32; the fix is an `s_dcache_inv`
in the acquire of tasks that read per-step data, to be added before the next session.
