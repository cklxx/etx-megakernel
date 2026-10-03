# Operation inventory: fleet's DeepSeek-V2-Lite decode tasks as the ETX port runs them

The inventory derives from source on 2026-09-27.
Sources are fleet-mi300x `src/kernels/{fleet_kernel.hip, gemv.h, attention.h, expert.h}`, `src/runtime/*`, and ETX `examples/dsv2lite/{fleet_shim.hip, graph.py, host.hip}`.

Abbreviations: FK fleet_kernel.hip, G gemv.h, A attention.h, E expert.h, RT fleet_runtime.h, SH fleet_shim.hip, GR graph.py, HO host.hip.
In tables, "RT" means dependent memory round trip.
WG = 256 threads, or 4 waves. Position 1024 has S = 1025.

Source alone cannot determine the items marked *ISA*: loop unrolling, scalarization, and LDS reuse.
Read these from the `-save-temps` ISA.

## Configuration facts that change what the bodies do

- `coherent_acts = 0` (HO): every activation load/store is plain; stage/fold take the 16 B non-atomic path.
- ETX passes no `FLAG_TOPK_PUBLISH` / `FLAG_TOPK_READ` (every gate_up WG computes softmax + top-k itself), no
  `FLAG_OPROJ_ROW_SPLIT` (o_proj K-split), **no `FLAG_KVA_SHARED` (every XCD computes all 576 kv_a rows)**, no
  `FLAG_WORKER_GROUP` (nw = 37).
- Fold: qkv of layers 2-26 and lm_head fold the 8 expert partials; qkv of layer 1 stages x[1]; expert_down uses the
  cached routing (`CHUNK_WAIT | ROUTING_CACHED`).
- Expert slots: XCD 0-5 routed slots 0-5, XCD 6/7 the two shared halves (SH:115).
- fleet's cross-task prefetch never runs under ETX (only fleet's own worker loop calls it). `FLEET_STREAM_DEPTH` is dead
  code: the depth is fixed at RPI x D = 8 x 16 B per lane per group, two groups in flight.
- `FLEET_NT_WEIGHTS=1`: every gemv weight load is non-temporal 16 B. Not non-temporal: W_UK in q_absorb (A:130), the KV
  cache (A:286-287), all activations.
- Norm gammas: 2 B bf16 scalar loads, 8 per thread, read after the reduction (FK:92).
- `__shfl_xor` lowers to `ds_bpermute`: `wave_sum` = 6 steps, `half_wave_sum` = 5.

## Building blocks

| block | per WG | file:line |
|---|---|---|
| stage_vector(src, n) | float4 loads n/1024 per thread, LDS write 4n B, 1 barrier<br> 1 RT | G:188-197 |
| stage_residual fold (bf16 partials) | per thread 8 x 16 B `expert_out[x]` (8 XCDs, 32 KiB) + 2 float4 of src (8 KiB): 10 x 16 B in flight, 40,960 B, 1 RT<br> 64 fp32 adds + 16 bf16 roundings per thread<br> 8 KiB LDS write<br> 1 barrier<br> WG (0,0) publishes 8 KiB | FK:168-199 |
| stage_residual_f32 (fp32 o_proj partials) | 2 loop iterations x (8 + 1) float4 per thread = 73,728 B per WG<br> 1-2 RT (*ISA*)<br> 8 KiB LDS<br> 1 barrier<br> WG (0,0) publishes x[1] | FK:118-142 |
| rms_norm_lds (2048) | 8 KiB LDS read, block_reduce_ordered: LDS tree of 8 levels with a barrier each (10 barriers), rsqrtf, gamma 4 KiB as 2 B scalars **after** the reduce (1 RT), 8 KiB LDS read-modify-write, 1 barrier: 11 barriers + 1 RT | FK:82-95, G:167-177 |
| gemv_rows path | K <= 256 half-wave <8><br> <= 512 <8,1><br> <= 1024 <4,2><br> <= 2048 <2,4><br> else impl <2,4> (all five instantiated at every call site: 262 registers) | G:456-466 |
| pipelined <RPI,D> | wave owns groups RPI*wave + 16m<br> per group RPI x D x 16 B per lane, 2 groups in flight<br> **one extra group issued past the last, clamped to row cnt-1, never consumed**<br> per row a 6-step wave_sum, lane 0 stores | G:265-334 |
| fma8 | 20 VALU per 16 B chunk (8 MACs) | G:85-92 |
| gemv_gate_up_rows | wave owns pairs wave + 4m<br> 8 x 16 B per lane per pair, 16 in flight, one extra pair over-issued<br> per pair 12 bpermutes, 1 expf + 1 divide (SiLU), 4 bf16 roundings, 4 B store | G:484-565 |

## Tasks

**qkv_fused** (296 WGs; FK:620-647). q rows: 384 per XCD (workers 0-13 own 11, the rest 10); kv_a: **all 576 rows on
every XCD** (workers 0-20 own 16, the rest 15); K = 2048 -> <2,4>.

| # | operation | bytes / count per WG | shared or unique | RT |
|---|---|---|---|---|
| 1 | fold (layers 2-26): 32 KiB partials + 8 KiB x[1], 10 x 16 B per lane -> LDS, barrier | 40,960 B | shared by 296 | 1 |
| 2 | rms_norm_lds: gamma 4 KiB, 11 barriers | | shared | 1 |
| 3 | q GEMV (EPI_BF16 -> act.q): useful 45,056 / 40,960 B, **issued 81,920 / 73,728 B**<br> 24,576 MACs, 72 bpermutes | | unique, NT | ~1 |
| 4 | kv_a GEMV (-> act.kv_a[xcd]): useful 65,536 / 61,440 B, **issued 98,304 B**<br> 32,768 MACs, 96 bpermutes | | **shared across the 8 XCDs (each reads all 2.36 MB)**, NT | ~1, after #3 drains |

Before the weights: 2 dependent global RTs + 12 barriers. Per layer: 14.9 MB unique, 31.5 MB requested (kv_a x 8),
**51.8 MB issued** with the over-issue.

**q_absorb** (32 WGs; A:111-150): W_UK 64 rows of the head, 2 serial un-pipelined batches of 8 rows per wave (plain
loads), 65,536 B; q_head 8 uniform loads per batch (*ISA*: scalar or vector); 32,768 MACs; per-wave partials through LDS
(2 barriers); 2 KiB store; then (shim) barrier, `s_waitcnt 0`, one agent-scope atomic add on the XCD counter the
attention polls.

**attention** (256 WGs, 16 heads x 16 KV chunks; A): chunks 0-14 hold 65 cached rows, chunk 15 holds 49 + the new row.

| # | operation | bytes per WG | RT |
|---|---|---|---|
| 1 | kv_post: kv_a_raw[0..512) 4 B loads | 2,048 B | 1 (2 if not unrolled) |
| 2 | sum of squares, 10-barrier tree, rsqrtf | | |
| 3 | **kv_a_raw reloaded** + kv_a_norm_w 1 KiB + rope part 256 B -> LDS | 3,328 B | 1 |
| 4 | RoPE: cos/sin 512 B (threads < 32), 3 barriers | 512 B | 1 |
| 5 | bf16 rounding<br> WG (0,0) writes the cache row (1,152 B)<br> barrier | | |
| 6 | PUB_WAIT: poll the XCD counter bumped by q_absorb (s_sleep 1), then `s_waitcnt 0` + `buffer_inv sc0`, barrier | | >= 1 |
| 7 | q_absorbed (4 KiB) + q_pe (256 B), barrier | 4,352 B | 1 |
| 8 | RoPE on q_pe: **cos/sin reloaded**, 2 barriers | 512 B | 1 |
| 9 | cache stream: batches of 8 positions, **not double buffered**<br> 16 loads per lane in flight<br> `vr` (k_pe) issued by all 64 lanes but only lanes 0-7 use it<br> wave 0 does 3 serial batches | useful 74,880 B, distinct 82,944 B, lane-load 147,456 B | 3 serial |
| 10 | scores: 73,728 lane-MACs (37,440 useful: 448 of 1024 wasted on k_pe)<br> 48 bpermutes per batch<br> online softmax 2 expf per position | | |
| 11 | per-wave (m, l, acc[512]) to LDS, barrier, every thread recomputes the 4 wave exps (1,024 redundant exps per WG), fp32 partial store 2,056 B | | |

20 barriers; before the first cache load **>= 5 dependent global RTs + 1 atomic poll + 18 barriers**.

**merge_uv** (256 WGs; A:358-411): m/l 128 B (32 x 4 B, same addresses for all threads) -> 16 expf + 1 reciprocal per
thread (4,096 redundant lane-exps per WG) -> acc 32,768 B of 4 B loads (2 j iterations x 2 batches: 2-4 serial RT, *ISA*)
-> LDS 2 KiB, barrier -> W_UV GEMV <8,1> with **only wave 0 working** (8 rows), 8,192 B useful, 16,384 issued. 3-5 RT before
the weights.

**o_proj** K-split (296 WGs; FK:709-716): stage o slice 1 KiB (1 RT, 1 barrier); half-wave <8> GEMV, K = 256, 55-56 rows,
useful 28,672 B, **issued 45,056 B**; 14,336 MACs; 140 bpermutes; 56 fp32 partial stores.

**norm_router** (296 WGs; FK:356-406): stage_residual_f32 73,728 B (1-2 RT) -> WG (0,0) publishes x[1] -> rms_norm_lds
(gamma 1 RT, 11 barriers) -> worker 0 writes x_norm[xcd] -> router GEMV 64 x 2048 with **only wave 0 working** (1-2 rows),
8 KiB useful, 16 KiB issued; **every XCD computes all 64 logits (router weights read 8 x)**. 2-3 RT + 12 barriers before
the weights.

**expert_gate_up** (296 WGs; FK:720-740, G:484-565): select_experts on wave 0 only: 64 logits (1 RT), max tree, expf,
sum, 6 top-k passes = **84 serial bpermutes**, barrier (also on XCD 6-7 where the result is unused) -> stage x_norm 8 KiB
(1 RT, barrier) -> 38-39 pairs, useful 311,296 / 319,488 B, issued 344,064 / 352,256 B, 155,648 MACs, 456 bpermutes,
38 (expf + divide) -> per wave `s_waitcnt 0` + an agent-scope atomic on a counter nobody reads under ETX (1,184 per layer).

**expert_down** (296 WGs; FK:742-783): routing from the same worker's LDS -> stage h 5,632 B (1 RT, barrier) -> <2,4> GEMV
with K = 1408: per lane d = 3 (and d = 2 for lanes >= 48) load chunk 0 anyway (clamped): useful 157,696 B, distinct rows
180,224 B, lane-load 262,144 B; 78,848 MACs; 336 bpermutes; 56 bf16 stores.

**dense layer 0**: gate_up as expert (x_norm stage, 37 pairs, useful 303,104 B); down via `gemv_rows_impl<2,4>` (K = 10944)
**reading h straight from global in every consume step (50.8 MB of L2 traffic per phase)**, wave 3 loads its single row
twice, residual epilogue is a dependent load then store per row.

**lm_head** (296 WGs): fold 40,960 B (1 RT) -> rms_norm_lds -> <2,4> GEMV over 345-346 rows, useful 1,417,216 B ->
fp32 logits stores -> candidate argmax: 346 scalar loads at 148 B stride, 9-barrier LDS tree. 22 barriers.

**embed / argmax**: 1 WG each; argmax reads 2,368 B, 9-barrier tree.

## Per-layer totals (MoE layer)

| phase | WGs | unique HBM | requested | issued (with over-issue) | shared reads: unique / requested |
|---|---|---|---|---|---|
| qkv (fold) | 296 | 14,942,208 | 31,457,280 | 51,838,976 | fold 40,960 / 12,124,160<br> gamma 4,096 / 1,212,416 |
| q_absorb | 32 | 2,097,152 | 2,097,152 | 2,097,152 | |
| attention | 256 | 1,179,648 (cache) | 18,874,368 | 20,938,752 | ~2.75 MB small activations |
| merge_uv | 256 | 2,097,152 | 2,097,152 | 4,194,304 | partials 526,336 / 8,421,376 |
| o_proj | 296 | 8,388,608 | 8,388,608 | 13,336,576 | |
| norm_router | 296 | 262,144 | 2,097,152 | 4,849,664 | fold 73,728 / 21,823,488 |
| gate_up | 296 | 92,274,688 | 92,274,688 | 101,974,016 | |
| down | 296 | 46,137,344 | 46,137,344 | 53,346,304 (77.6 MB lane-load) | |

Unique HBM per MoE layer: 167,378,944 B.

## Dependent round trips before each task's weight (or cache) stream

| task | chain | global RTs | barriers |
|---|---|---|---|
| qkv (fold) | fold -> LDS -> barrier -> sum of squares -> 10-barrier tree -> rsqrt -> gamma -> LDS -> barrier | 2 | 12 |
| q_absorb | none (2 serial weight batches) | 0 | 0 |
| attention | kv_a -> tree -> kv_a + norm_w -> cos/sin -> poll + inv -> q_absorbed + q_pe -> cos/sin (then 3 serial cache batches) | >= 5 + poll | 18 |
| merge_uv | m/l -> exps -> acc (2-4 RT) -> barrier -> W_UV | 3-5 | 1 |
| o_proj | o slice -> barrier | 1 | 1 |
| norm_router | f32 fold (1-2) -> tree -> gamma -> router rows | 2-3 | 12 |
| gate_up | logits -> 84 serial bpermutes -> barrier -> x_norm -> barrier | 2 | 2 |
| down | LDS ids -> h -> barrier | 1 | 1 |

## Waste and surprises (candidate fixes, each a measurable change)

1. Clamped over-issue in every pipelined GEMV (one group per wave past the last): q 80 KiB issued for 44 KiB used,
   router 16 for 8, merge 16 loads per lane for 8, o_proj 88 row loads for 56; they sit in vmcnt ahead of every later
   `s_waitcnt 0` (gate_up's chunk signal, the shim's drain, ETX's arrive).
2. gate_up: per-wave `s_waitcnt 0` + an atomic nobody reads under ETX (1,184 per layer).
3. expert_down K = 1408 in a D = 4 buffer: 45% extra load instructions (same lines).
4. merge: one wave of four works; each of a head's 16 merge WGs repeats the whole 16-chunk merge (32 KiB of 4 B loads,
   16 expf per thread).
5. router: one wave works; all 8 XCDs compute all 64 logits.
6. kv_a replicated per XCD (no KVA_SHARED): 18.9 MB requested for 2.36 MB unique.
7. softmax + top-k in all 296 gate_up WGs, including the two XCDs that do not use it; 84 serial bpermutes on one wave.
8. attention prologue: kv_a loaded twice, cos/sin loaded twice, the kv norm redone in all 256 WGs, 18 barriers before the
   cache, cache loop not double buffered, k_pe loaded by 64 lanes for 8 users.
9. block_reduce_ordered: an LDS tree with a barrier per level (10 barriers per norm) instead of wave shuffles.
10. The gamma load after the reduction adds one global RT to every norm, as 2 B scalar loads.
11. stage_residual_f32: 72 KiB per WG, 21.8 MB per phase of fabric traffic for 72 KiB of data.
12. dense_down: h re-read from global by every wave.
13. q_absorb: W_UK not non-temporal; 2 serial batches.
14. lm_head: fp32 logits; candidate read-back one cache line per 4 B.
15. qkv: two GEMV calls; the pipeline drains between q and kv_a.
16. Stale comments (SH:11-13, HO:129): attention's PUB_WAIT really polls a real counter (starts at 0, bumped by q_absorb).

*ISA* to read: unrolling of the runtime-bound loops (kv_post, stage_residual_f32, merge's j loop, embed, argmax);
whether uniform loads become scalar; LDS x reuse across a group's rows; whether dense_down's x loads issue with the weights.
