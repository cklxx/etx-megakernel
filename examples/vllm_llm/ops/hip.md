# vLLM HIP decode kernels at batch 1 on MI300X: an operation-level inventory

These are the kernels in one decode step, in the form they run in (vLLM's own compiled code). The shapes come from ETX's import of the step. The document is meant as input to the ETX megakernel performance model.

- **Machine:** gfx942, wave64, 304 CUs in 8 XCDs of 38.
- **vLLM checkout:** `~/code/vllm` at f46a9dfe (2026-08-20).
- **Assembly:** `~/code/etx/build/rocm22/qwen3-30b-a3b_base/orig_*.s` (vLLM's form, hipcc -O3, gfx942). There is no `orig_*.s` for `attn_mfma16_g6`. No build in the tree contains it, so its register figures are not verified (see §6).

## 0. Conventions

**File abbreviations.** vLLM files, relative to `~/code/vllm/csrc`:

| abbreviation | file |
|---|---|
| SG | `rocm/skinny_gemms.cu` |
| AT | `rocm/attention.cu` |
| LN | `libtorch_stable/layernorm_kernels.cu` |
| VU | `libtorch_stable/quantization/vectorization_utils.cuh` |
| PE | `libtorch_stable/pos_encoding_kernels.cu` |
| CK | `libtorch_stable/cache_kernels.cu` |
| AC | `libtorch_stable/activation_kernels.cu` |
| TK | `libtorch_stable/moe/topk_softmax_kernels.cu` |
| MS | `libtorch_stable/moe/moe_align_sum_kernels.cu` |

ETX files, relative to `~/code/etx`:

| abbreviation | file |
|---|---|
| VK | `etx/importer/vllm_kernels.py` |
| MP | `examples/vllm_llm/model.py` |
| HH | `examples/vllm_llm/host.hip` |

**Assembly citations.** `s:<file>:<line>` means `build/rocm22/qwen3-30b-a3b_base/orig_<file>.s`.

**Round trips.** **S** is a scalar round trip (`s_load` then `s_waitcnt lgkmcnt(0)`):

- **Kernel-argument loads.** Only the first one per wave should miss the scalar cache. The kernel-argument block (kernarg) of every kernel here is at most 344 B, and most are at most 64 B, a single scalar-cache line.
- **Device data loaded through the scalar cache:** `seq_lens`, `positions` and `slot_mapping` (see note N1).

**V** is a vector global-memory round trip (issue, then the `s_waitcnt vmcnt` that consumes it). On gfx9, `vmcnt` also counts outstanding stores, because there is no separate `vscnt`. So a `vmcnt(0)` after a store waits for that store's acknowledgement. Stores that are never waited on are listed but not counted.

**"Fresh"** means data written earlier in the same step by another kernel. It was usually written by blocks on other CUs or XCDs: workgroup *i* is dispatched to XCD *i* mod 8, so a single-block producer sits on XCD 0. **"Const"** means weights or tables, unchanged within the step.

**Shuffle steps** are counted as sequential dependent steps. Every `__shfl_xor` compiles to `ds_bpermute_b32` followed by `s_waitcnt lgkmcnt(0)`, which is an LDS round trip. The one exception is wvSplitK, which uses DPP (`v_*_dpp`, no LDS).

**Register counts** are from the assembly header comments (`; NumVgprs`, `; TotalNumSgprs`, `; ScratchSize`, `; Occupancy`). No kernel here uses AGPRs.

**Instruction counts.** Unless marked "dynamic", counts are static counts of the compiled function (`python -m examples.vllm_llm.isa_inventory` together with a class split of the `.s`). "Dynamic ≈" figures are my path estimates from the assembly and the trip counts given.

### 0.1 Launch sequence of one layer

MP:101-139 gives the order; HH:199-257 gives the argument values.

| # | role | kernel (VK entry) | grid x block | MP | HH |
|---|---|---|---|---|---|
| 1 | in_norm | `fused_add_rms_norm<bf16,8,true>` (layer 0: `rms_norm_kernel<bf16,8,2,true>`, §11) | 1 x 1024 | 115 | 220-222 |
| 2 | qkv | `wvSplitK_hf_sml_<bf16,64,2,16,8,2,1>` | 304 x (64,16) | 116 | 223 |
| 3 | q_norm | `rms_norm_kernel<bf16,8,3,true>` (Qwen3 only) | NH x 16 | 118 | 224 |
| 4 | k_norm | same kernel as q_norm | NKV x 16 | 119 | 225 |
| 5 | rope | `rotary_embedding_kernel<bf16,bf16,true>` | 1 x 512 | 120 | 226 |
| 6 | cache | `reshape_and_cache_kernel<bf16,bf16,kAuto>` | 1 x 16·NKV | 121 | 227 |
| 7 | attn | `paged_attention_ll4mi_QKV_mfma16_kernel<…,16,128,256,false,GQA,F16>` | (1,5,NKV) x 256 | 122 | 228-229 |
| 8 | reduce | `paged_attention_ll4mi_reduce_kernel<bf16,bf16,128,128,256,1>` | (NH,1) x 128 | 123 | 230 |
| 9 | o_proj | wvSplitK YTILE 2 (same as #2) | 304 x (64,16) | 124 | 231 |
| 10 | post_norm | `fused_add_rms_norm` | 1 x 1024 | 125 | 232 |

The last launches differ by model.

For Qwen3-30B-A3B (MoE):

| # | role | kernel | grid x block | MP | HH |
|---|---|---|---|---|---|
| 11 | router | `wvSplitK_hf_sml_<bf16,64,1,16,8,4,1>` | 304 x (64,16) | 127 | 236 |
| 12 | topk | `topkGating<8,128,4,16,64,int,bf16,SOFTMAX>` | 1 x (64,4) | 128 | 239-240 |
| – | moe_up | Triton fused_moe (not in this inventory) | 192 blocks | 129 | 244-246 |
| 13 | moe_act | `act_and_mul_kernel<…,true,true,false,false>` | 8 x 96 | 130 | 234 |
| – | moe_down | Triton fused_moe (not in this inventory) | 256 blocks | 131 | 247-249 |
| 14 | moe_sum | `moe_sum_vec_kernel<bf16,int,8,false>` | 1 x 256 | 132 | 252 |

For Qwen2.5-1.5B (dense):

| # | role | kernel | grid x block | MP | HH |
|---|---|---|---|---|---|
| 11 | gate_up | wvSplitK YTILE 2 | 304 x (64,16) | 134 | 233 |
| 12 | act | `act_and_mul_kernel` | 1 x 1024 | 135 | 235 |
| 13 | down | wvSplitK YTILE 2 | 304 x (64,16) | 136 | 253 |

**Model shapes.**

| | H | NH | NKV | HD | NQKV | other |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 2048 | 32 | 4 | 128 | (32+8)·128 = 5120 | E 128, top-8, expert intermediate MI 768, GQA 8 |
| Qwen2.5-1.5B | 1536 | 12 | 2 | 128 | (12+4)·128 = 2048 | intermediate 8960, qkv bias, GQA 6 |

- **Partition count.** It is `ceil(MAXLEN/256)`, with MAXLEN = 1072 (MP:54,103, AT:3284), which gives 5.
- **Context.** At a context of 1024 tokens with seq_len ≤ 1024, partition 4 exits immediately (AT:367-369). If the token being decoded makes seq_len 1025–1072, partition 4 does real work on 1–48 tokens. Figures below assume 4 active partitions.

**N1 (scalar-cache reads of per-step data).** The compiler reads these device arrays with `s_load` through the scalar data cache:

- `seq_lens[0]` (s:attention:46 and :921)
- `positions[0]` (s:pos_encoding_kernels:22)
- `slot_mapping[0]` (s:cache_kernels:380)

A kernel boundary invalidates that cache. Inside a persistent megakernel that re-runs these on the same CU every step, it is not invalidated unless the code does so explicitly.

---

## 1. `fused_add_rms_norm_kernel<bf16, width=8, HasWeight=true>` (in_norm and post_norm)

**Launch.**

- **Grid and block.** Grid is `num_tokens` = 1 (LN:329). Block is `min(H, 1024)` = 1024 for both models (LN:337-340). The width-8 path is used because the pointers are 16 B aligned (LN:344-359, VK:84).
- **Arguments** (HH:220):

  | argument | value |
  |---|---|
  | input | `d_norm` or `o` (normalised in place) |
  | input_stride | H |
  | residual | `resid` |
  | weight | `ln1` or `ln2` |
  | eps | eps |
  | num_tokens | 1 |
  | hidden_size | H |
  | residual_stride | H |

- **Waves:** 16.
- **Threads with data:** `vec_hidden_size = H/8`, i.e. 256 threads (waves 0-3) for 30B and 192 (waves 0-2) for 1.5B. The other 12 or 13 waves only take part in the reduction and the barriers.
- **Resources:** LDS 68 B (the cub scratch of 16 floats plus `s_variance`). 40 VGPR, 51 SGPR, no scratch, occupancy 8. Function at s:layernorm_kernels:1732.

**Per-block chain.** Each active thread runs one iteration of both loops.

| # | step | bytes/thread | source | asm |
|---|---|---|---|---|
| 1 | Kernarg `s_load` (**S**). | – | LN:119-131 | s:layernorm_kernels:1734-1737 |
| 2 | **V1**: `input[i]` 16 B (fresh: o_proj, moe_sum or down output) and `residual[i]` 16 B (written by the previous fused_add), both in flight. | 32 | LN:136-137 | :1800-1801 |
| 3 | Add, bf16 round-to-nearest-even, sum of squares. Store `residual[i]` 16 B (not waited on). | 16 W | LN:137-139 | :1886 |
| 4 | cub BlockReduce level 1: 6 `ds_bpermute` steps, each waited. All 16 waves execute it. | – | LN:142-144 | :1910-1958 |
| 5 | Lane 0 of each wave writes its partial sum to LDS. Barrier #1, inside cub. | – | – | :1969, :1974 |
| 6 | Threads < 16: `ds_read`, then 4 `ds_bpermute` steps. | – | – | :1978-2013 |
| 7 | Thread 0: `variance/H` as a full IEEE divide (`v_div_scale/fmas/fixup`), a kernarg `s_load` of eps, `v_rsq`, and an LDS write. | – | LN:146-148 | :2045, :2050 |
| 8 | Barrier #2. Every thread reads `s_variance` from LDS. | – | LN:149 | :2054, :2065 |
| 9 | **V2**: `residual[i]` again (its own store from step 3, so an L2 hit) and `weight[i]` 16 B (const). | 32 | LN:154,158 | :2091-2092 |
| 10 | Multiply, then store to `input[i]` 16 B. | 16 W | LN:159-172 | :2168 |

**Totals per launch** (one block):

| | 30B | 1.5B |
|---|---|---|
| bytes read | 16 KB (4 fresh + 4 residual + 4 residual re-read + 4 weight) | 12 KB |
| bytes written | 8 KB | 6 KB |

- Barriers: 2. Shuffle steps: 10 (6 + 4 `ds_bpermute`).
- Loads in flight per thread: 2 x 16 B.
- Instructions: static VALU 299, SALU 95, LDS 14. Dynamic ≈ 200-250 instructions per active wave; MFMA 0.

**Chain:** S → V1 → 10 LDS steps + 2 barriers + divide + rsq → V2 → store. That is **2 V** (plus 1–2 S). The V2 loads are not issued before barrier #2, although the weight load does not depend on the reduction.

---

## 2. `wvSplitK_hf_sml_<bf16, THRDS 64, YTILE, WvPrGrp 16, A_CHUNK 8, UNRL, N 1>` (all projections and the router)

**Launch and configuration.**

- **Grid** is `CuCount` = 304 (SG:1207, HH:213: last argument 304).
- **Block** is (64, 16), i.e. 16 waves (SG:1215).
- **Tile shape.** `sYT = ceil(M/(304·4))` (SG:1300). `sYT ≤ 1` gives YTILE 1, UNRL 4; `N == 1` otherwise gives YTILE 2, UNRL 2 (SG:1234-1240, VK:51-58).
- **`_sml_` variant condition:** `Kbp·N ≤ 32768` and `M % YTILE == 0` (SG:1217).
- **Active waves per group:** `_WvPrGrp = mindiv(M, 304·YTILE, 16)` (SG:1169-1179, 1216, MP:85-93).
- **Arguments** (HH:212-214): K, Kbp = K, Kap = K, M, Bx = (bias ? M : 1), By = 1, B = weight, A = activation, BIAS, C, `_WvPrGrp`, 304. The launcher passes `(K_in, Kap_in, Kbp_in)` into kernel parameters named `(K, Kbp, Kap)` (SG:1219 vs SG:352). All three equal K here.
- **Resources.**
  - LDS: static `s[32768]` bf16 = 64 KB (SG:357,383), so one block per CU and 4 waves per SIMD.
  - YTILE 2: 45 VGPR, 33 SGPR, occupancy 4 (s:skinny_gemms:8).
  - YTILE 1: 48 VGPR, 37 SGPR (s:skinny_gemms:370).
  - No scratch in either.

**Per shape.** The script re-derives `_WvPrGrp` and checks it against `vl_plan.txt`.

| shape (M x K) | YT/UNRL | wvprgrp | blocks with rows | waves with rows | m-iters per wave | K-loop trips (step) | LDS-fill threads (iterations) | weight B per active block |
|---|---|---|---|---|---|---|---|---|
| 30B qkv 5120 x 2048 | 2/2 | 9 | 285 of 304 | 2560 | 1 | 2 (1024) | 256 (1) | 72 KB |
| 30B o_proj 2048 x 4096 | 2/2 | 4 | 256 | 1024 | 1 | 4 | 512 (1) | 64 KB |
| 30B router 128 x 2048 | 1/4 | 4 | 32 | 128 | 1 | 1 (2048) | 256 (1) | 16 KB |
| 1.5B qkv 2048 x 1536 (+bias) | 2/2 | 4 | 256 | 1024 | 1 | 2 | 192 (1) | 24 KB |
| 1.5B o_proj 1536 x 1536 | 2/2 | 4 | 192 | 768 | 1 | 2 | 192 (1) | 24 KB |
| 1.5B gate_up 17920 x 1536 | 2/2 | 15 | 304 | 4560 | 2 (4400 waves) / 1 (160) | 2 per m-iteration | 192 (1) | ≤ 180 KB |
| 1.5B down 1536 x 8960 | 2/2 | 4 | 192 | 768 | 1 | 9 | 1024 (+96 threads do a 2nd) | 140 KB |

**Which blocks work.** Every block, including the ones with no rows, fills LDS with the whole activation vector before the barrier (SG:394-402). Waves with `threadIdx.y ≥ _WvPrGrp` return after the barrier (SG:404). Waves whose starting `m ≥ M` skip the while loop (SG:406,423). Active blocks therefore hold only 4, 9 or 15 streaming waves out of 16.

**Per-block chain (YTILE 2).**

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg (K, M, bias pointer) **S**, then kernarg for the activation pointer **S**. | SG:352-356 | s:skinny_gemms:10-17, :23-29 |
| 2 | LDS fill, **V (act)**: threads with `tid·8 < K` do `global_load_dwordx4` of the activation (fresh; written by one CU for norm and silu outputs, or by 32/12 blocks for attention-reduce output), then `s_waitcnt vmcnt(0)`, then `ds_write_b128`. One 16 B load in flight per thread. K = 8960: threads < 96 run a second, serial iteration. | SG:394-401 | :34-43 (load :35, write :41) |
| 3 | Kernarg `_WvPrGrp` **S**, then `__syncthreads` (the block's only barrier). | SG:402 | :47-49 |
| 4 | Row setup: `m = (blk·wv + ty)·YTILE`. Kernarg B and C pointers **S**, waited at the top of the K loop. | SG:406, 423 | :65-75, :172 |
| 5 | K loop, per trip: 4 × `global_load_dwordx4 … nt` of weights (const, **V**; UNRL·YTILE = 4 loads × 16 B in flight per lane, 4 KB per wave), plus 2 × `ds_read_b128` of the activation (UNRL·N). Then 8 × `v_mfma_f32_4x4x4_16b_bf16` (UNRL·YTILE·A_CHUNK/4), consuming with `vmcnt(3..0)`. Not software-pipelined: the next trip's loads are issued only after this trip's MFMAs. | SG:434-473 (loads 439-445, LDS 449-456, MFMA 459-472) | :169-206 loads/LDS, :150-166 MFMA |
| 6 | Cross-lane reduction by DPP only, per y: 3 × `row_shl` 1/2/3 adds, `row_shl` 4, `row_shl` 8, `row_shr` 15, `row_bcast` 15, `row_bcast` 31. That is 8 dependent DPP ops; the two y are interleaved, 16 DPP in total. | SG:520-549 | :212-234 |
| 7 | Lane 63: bias (1.5B qkv only) is 2 × `global_load_ushort` **V**, issued only after the reduction. Then 2 × `global_store_short_d16_hi` (2 B each). | SG:550-564 | :264-265, :98, :107 |
| 8 | `m += 304·wv·YTILE`; gate_up waves loop back to step 5. | SG:567 | :109-113 |

**YTILE 1 (router).** The same chain, with these differences:

- The K loop has 1 trip: 4 weight loads × 16 B (UNRL 4 × YTILE 1), 4 `ds_read_b128`, and 8 MFMA (s:skinny_gemms:527-531, 552-591, 494-512).
- DPP chain: 8 (:597-610).
- 1 store of 2 B (:457).

**Totals per launch.**

| shape | weights read | activation read into LDS (L2 traffic; unique = K·2) | written | sequential RTs per wave (V) | dynamic instructions per wave ≈ | MFMA per wave |
|---|---|---|---|---|---|---|
| 30B qkv | 20.97 MB | 304 × 4 KB = 1.25 MB | 10 KB | 1 + 2 = 3 | 245 | 16 |
| 30B o_proj | 16.78 MB | 304 × 8 KB = 2.49 MB | 4 KB | 1 + 4 = 5 | 350 | 32 |
| 30B router | 0.52 MB | 1.25 MB | 256 B | 1 + 1 = 2 | 195 | 8 |
| 1.5B qkv | 6.29 MB + 4 KB bias | 304 × 3 KB = 0.93 MB | 4 KB | 1 + 2 + 1 (bias) = 4 | 265 | 16 |
| 1.5B o_proj | 4.72 MB | 0.93 MB | 3 KB | 3 | 245 | 16 |
| 1.5B gate_up | 55.05 MB | 0.93 MB | 35 KB | 1 + 2·2 = 5 | 400 | 32 |
| 1.5B down | 27.53 MB | 304 × 17.5 KB = 5.45 MB | 3 KB | 2 + 9 = 11 | 620 | 72 |

- **Per block:** 1 barrier; DPP chain of 8 per row-set (YTILE 2 runs two chains, 16 ops); no LDS shuffles.
- **Weight bytes in flight per CU:** equal to active waves × 4 KB. That is 36 KB (30B qkv), 16 KB (o_proj, router, 1.5B qkv/o_proj/down) and 60 KB (gate_up).
- **Static counts:**
  - YTILE 2: VALU 142, SALU 77, VMEM 9, LDS 3, MFMA 8.
  - YTILE 1: VALU 131, SALU 83, VMEM 7, LDS 5, MFMA 8.
  - Loop body: ≈ 52 instructions per trip (≈ 20 VALU, 8 MFMA).
- **Wasted loads at K = 1536.** The second trip's k2 = 1 half has every lane clamped to `K−8` (SG:442). Each wave issues 2 redundant 16 B loads per row pair to one cache line, and the MFMAs run on zeroed `bigA` (SG:452).

**Chain:** S → S → V(act) → S → barrier → S → V(weights) × trips (× m-iterations) → 16 DPP → [V bias] → store. The weight stream cannot start until the activation has arrived and the barrier has passed.

---

## 3. `rms_norm_kernel<bf16, VEC 8, NUM_DIMS 3, true>` (Qwen3 q_norm / k_norm)

**Launch.**

- **Grid** is `num_tokens = numel/128`: q_norm 32 blocks, k_norm 4 (LN:245,257).
- **Block** is `min(128/8, 1024)` = 16 threads (LN:267-271, VK:81-82). That is one wave with only 16 of 64 lanes live.
- **Arguments** (HH:224-225): out = qn/kn; input = qkv or qkv+NH·HD; stride_d2 128; stride_d3 NQKV; stride_d4 0; shape_d2 = NH/NKV; shape_d3 0; weight = q_norm/k_norm weight; weight_stride 0; eps; num_tokens = NH/NKV; hidden 128.
- **Resources:** LDS 68 B; 58 VGPR, 65 SGPR, occupancy 8 (s:layernorm_kernels:773).

**Per-block chain.** Each thread runs one iteration of each loop.

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S**. Then `blockIdx / shape_d2`, an int64 division: a runtime test chooses the 32-bit SALU path. Kernarg again **S**. | LN:36-42 | :775-778, :911-936, :937-943 |
| 2 | Alignment test, then the fast path: **V1** one `dwordx4` of the q/k head (fresh: qkv wvSplitK output, one head's 128 rows written by about 7-8 blocks on different XCDs). | LN:65-66, VU:142-153 | :973, :1121-1122 |
| 3 | Sum of squares, then cub level 1: 6 `ds_bpermute` steps. LDS write. Barrier #1. | LN:68-70 | :1163-1227 |
| 4 | Level 2: `ds_read` plus 4 `ds_bpermute`. Thread 0: IEEE divide, kernarg eps **S**, `v_rsq`, LDS write. | LN:72-74 | :1231-1303 |
| 5 | Barrier #2. Read `s_variance`. Kernarg for out/weight **S**. | LN:75 | :1312-1319 |
| 6 | **V2**: input 16 B again, plus weight 16 B (const). | LN:83,86 | :1558-1559 (remainder loop; the ×2-unrolled loop at :1356 needs ≥ 2 vectors per thread) |
| 7 | Store 16 B to qn/kn. | LN:98 | :1636 |

**Totals.**

- Per block: 768 B read (256 fresh twice, 256 weight), 256 B written.
- Per launch: q_norm 24 KB read / 8 KB written; k_norm 3 KB / 1 KB.
- Barriers 2; shuffle steps 10; loads in flight 1 × 16 B, then 2 × 16 B.
- Static VALU 443, SALU 349 (the int64 division and three alignment paths); dynamic ≈ 350.

**Chain:** 3–4 S, **2 V**, 10 LDS steps, 2 barriers.

---

## 4. `rotary_embedding_kernel<bf16, bf16, IS_NEOX=true>`

**Launch.**

- **Grid** is `num_tokens` = 1. **Block** is `min(NH·rot_dim/2, 512)` = 512 for both models, i.e. 8 waves (PE:167-168, VK:88-89).
- **Arguments** (HH:226): positions, q, k, cos_sin, rot_dim 128, q_stride and k_stride (30B: NH·HD, NKV·HD on qn/kn; 1.5B: NQKV on qkv), head_stride 128, NH, NKV, 128, rope_dim_offset 0, inverse false.
- **Resources:** no LDS; 15 VGPR, 42 SGPR (s:pos_encoding_kernels:8).

**Per-block chain.**

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S**, then `positions[0]` via `s_load_dwordx2` **S** (fresh, per-step, N1). Kernarg again **S**. | PE:91-93 | :10-14, :22-25, :47-56 |
| 2 | q loop. Per iteration: 4 × `global_load_ushort` (cos, sin: const, address depends on pos; x, y: fresh q). Then `vmcnt(3..0)` (**V**, which also covers the previous iteration's stores), 4 FMA, 2 × `global_store_short`. | PE:53-61, PE:9-33 | loop :78-144 (loads :108-113, stores :140-142) |
| 3 | k loop, same body. | PE:63-73 | :184-250 |

**Trip counts:**

| | q loop (nq = NH·64) | k loop (nk = NKV·64) |
|---|---|---|
| 30B | 2048: 4 iterations for all 512 threads | 256: 1 iteration for threads < 256 |
| 1.5B | 768: threads < 256 do 2, the rest 1 | 128: threads < 128 do 1 |

**Totals.**

| | fresh q/k read | cos/sin loads (unique 256 B) | written | V RTs per block |
|---|---|---|---|---|
| 30B | 9 KB | 9 KB | 9 KB | 5 |
| 1.5B | 3.5 KB | 3.5 KB | 3.5 KB | 3 |

- All loads and stores are 2 B scalars.
- No barriers, no shuffles. Loads in flight: 4 × 2 B per thread.
- Loop body 66 instructions; dynamic ≈ 70 + 66 × 5 ≈ 400 (30B).

**Chain:** S → S(pos) → S → V × (q iterations + 1) → store.

---

## 5. `reshape_and_cache_kernel<bf16, bf16, kAuto>`

**Launch.**

- **Grid** is `num_tokens` = 1. **Block** is `min(NKV·HD/x, 512)` with x = 8: 64 threads (30B, one wave) or 32 threads (1.5B, half a wave) (CK:769-770, VK:91-95).
- **Arguments** (HH:227): key = kn or qkv+NH·HD; value = qkv+(NH+NKV)·HD; k_cache and v_cache of layer L; slot_mapping; key_stride = qstr/kstr; value_stride NQKV; NKV; 128; block 16; x 8; k_scale and v_scale.
- **Resources:** no LDS; 44 VGPR, 66 SGPR (s:cache_kernels:372).

**Per-block chain.** Thread t handles head `t/16` and 8-element chunk `t%16`.

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S**, then `slot_mapping[0]` via `s_load` **S** (N1). Early exit if slot < 0. Kernarg again **S**. Then int64 `slot/16` and `slot%16` on SALU. | CK:266-273 | :374-388, :421-628 |
| 2 | K: `vectorize_with_alignment` fast path, len = x = 8, so 1 × `dwordx4` load (**V**, fresh) → `vmcnt(0)` → 1 × `dwordx4` store into `[blk][h][c][off][8]`. | CK:304, VU:39-58 | remainder loop :917-924 of the 4×-unrolled fast loop |
| 3 | V: `for i < x` with runtime x, 8 iterations of `global_load_ushort` (fresh) → `s_waitcnt vmcnt(0)` → `global_store_short` to `value_dst[i·16]`. Each iteration waits for its load and for the previous iteration's store: **8 serial V**. | CK:306-311 | :1279-1288 |

**Totals.**

| | read (fresh) | written | stores |
|---|---|---|---|
| 30B | 2 KB | 2 KB | K: 64 × 16 B; V: 512 × 2 B |
| 1.5B | 1 KB | 1 KB | half of 30B |

- No barriers, no shuffles. Loads in flight: 1 per thread.
- Static SALU 575 (int64 arithmetic), VALU 217; dynamic ≈ 350.

**Chain:** 3 S + **9 V**. This is the longest serial global chain of any small kernel; it comes from the scalar V scatter with runtime x.

---

## 6. `paged_attention_ll4mi_QKV_mfma16_kernel<bf16, bf16, kAuto, bf16, BS 16, HS 128, 256 threads, ALIBI false, GQA, F16>`

**Launch.**

- **Grid** is `(num_seqs 1, max_num_partitions 5, NKV)` and **block** is 256 threads, i.e. 4 waves (AT:3283-3291). GQA > 4 selects mfma16 (AT:3296-3322, VK:62-71).
- **Arguments** (HH:228-229): q = qn or qkv; k/v cache; NKV; scale = 1/√128; block_tables (identity); seq_lens; query_start_loc = nullptr; MAXBLK 67; alibi nullptr; q_stride (30B: NH·HD, 1.5B: NQKV); kv_block_stride = NKV·128·16; kv_head_stride 2048; exp_sums; max_logits; tmp_out; out; 67; k_scale; v_scale.
- **Blocks with work:**
  - 30B: (1,5,4) = 20 blocks, 16 active.
  - 1.5B: (1,5,2) = 10 blocks, 8 active.
  - Partition 4 exits at AT:367-369 unless seq_len > 1024 (§0.1).
- **Template constants** (AT:341-532): NWARPS 4; TOKENS_PER_WARP 64; TLOOP 4; QKHELOOP 4; VTLOOP 4; VTLANELOOP 2; VHELOOP 2; GQA_RATIO4 2.
- **Resources:** LDS `shared_logits[4][4][16][4]` of `_B16x4` = 8192 B (AT:374). For g8 (s:attention:8): 96 VGPR, 50 SGPR, 16 B scratch, occupancy 5 (`__launch_bounds__(256,5)`, AT:321). The scratch holds spills of a 64-bit Q address and a 4-byte value (`; Folded Spill`, s:attention:128, 152, 182).
- **GQA 6 (1.5B):** no assembly exists in the tree, so register counts are not determined. The source path is identical except for the Q-load mask and the second output store (see step 3 and step 14).

**Per-block chain.** "Lane" means a lane of any wave unless stated otherwise.

| # | step | bytes/lane | source | asm |
|---|---|---|---|---|
| 1 | Kernarg `query_start_loc` **S**; branch (nullptr). Kernarg `seq_lens` ptr **S**, then `seq_lens[0]` **S** (N1). Exit if `256·p ≥ seq_len`. | – | AT:351-369 | :10-50 |
| 2 | Kernarg block_tables and stride **S**. 4 × `global_load_dword` of K block numbers (**V**, const table; address uses seq_len clamp). | 16 | AT:431-439 | :102-105 |
| 3 | Q (waves 0-1 for GQA 8; for GQA 6, wave 0 plus rows 0-1 of wave 1): scratch spill and reload, `s_waitcnt vmcnt(0)` (waits for step 2), then `global_load_dwordx4` of Q (**V**, fresh from rope), then `vmcnt(0)`, then `ds_write_b128`. | 16 | AT:441-470 | :128-160 |
| 4 | Barrier #1. | – | AT:471 | :181 |
| 5 | Qlocal: 4 × `ds_read_b128` (interleaved with step 6). | – | AT:472-489 | :206-243 |
| 6 | K: 16 × `global_load_dwordx4` (**V**; old tokens const, the newest token fresh from reshape_and_cache). About 8 in flight (`vmcnt(7)` window). | 256 | AT:497-515 | :183-250 |
| 7 | QK: 32 × `v_mfma_f32_16x16x16_bf16` (TLOOP·QKHELOOP·2). Of each 16 head rows, only GQA are valid (8 or 6), so 50% or 62.5% of each MFMA is wasted. | – | AT:598-653 | :213-310 |
| 8 | V block numbers: 4 × `global_load_dword` (**V**), issued mid-QK after most K data has arrived. | 16 | AT:537-553 | :263, :289-291 |
| 9 | Row max: 2 `ds_bpermute` steps (masks 32, 16). 16 × exp. Row sum: 2 steps. | – | AT:673-699 | :394, :406, :495, :516 |
| 10 | V: 16 × `global_load_dwordx4` (**V**), all 16 in flight, issued once the V block numbers arrive. Consumed only in step 13. | 256 | AT:555-579 | :380-381, :409-410, :499-515 |
| 11 | Barrier #2. Write per-wave max and sum to LDS (`ds_write2st64`). Barrier #3. 4 × `ds_read2`, 4 exp, rcp. Barrier #4. Write bf16 logits to LDS (2 × `write2st64_b64`). | – | AT:701-768 | :520-641 |
| 12 | Threads < GQA: store `max_logits` and `exp_sums` (2 × 4 B). Barrier #5. | – | AT:771-784 | :659-667 |
| 13 | PV: 8 × `ds_read_b128` of logits, 32 × MFMA 16x16x16 (VHELOOP·VTLOOP·VTLANELOOP·2). Barrier #6. | – | AT:792-867 | :668-733, :699 |
| 14 | Write outputs to LDS. Barrier #7. Wave 0: 2 scratch reloads (serial, `vmcnt(0)` each), 2 × `ds_read_b128`, 2 × `global_store_dwordx4` to tmp_out (GQA 6: the second store only for rows 0-1). | – | AT:870-910 | :745-788 |

**Totals per block** (seq_len ≤ 1024, full partition).

| | 30B (GQA 8) | 1.5B (GQA 6) |
|---|---|---|
| KV read | 128 KB (64 K + 64 V) | 128 KB |
| Q read | 2 KB | 1.5 KB |
| block-table traffic (unique 128 B) | 4 waves × 8 dword loads = 8 KB | same |
| tmp_out written | 2 KB | 1.5 KB |
| max/exp written | 64 B | 48 B |

A partially filled last partition still issues all 128 KB of loads: token indices are clamped to the last block (AT:435-437, 547-549), so the extra loads hit in cache.

**Totals per launch.**

| | reads | writes |
|---|---|---|
| 30B, 16 active blocks | ≈ 2.2 MB of traffic (2 MB KV) | 32 KB |
| 1.5B, 8 active blocks | ≈ 1.1 MB (1 MB KV) | 12 KB |

- Barriers: 7. Shuffle steps: 4.
- MFMA per wave: 64. Static VALU 421, SALU 178, VMEM 51 (including 2 scratch loads and 3 scratch stores), LDS 27. The kernel is nearly straight-line, so dynamic ≈ static ≈ 720 instructions per wave.
- Loads in flight per lane: K about 8 × 16 B; V 16 × 16 B.

**Chain:** S(kernarg) → S → S(seq_len) → S → V(block table) → V(Q; serialised behind the block table by `vmcnt(0)`) → barrier → V(K) → V(V block numbers) → V(V data). Softmax, 5 barriers and PV follow, then 2 scratch round trips and the store. That is **4 S + 5 V + 2 scratch**, and 7 barriers.

---

## 7. `paged_attention_ll4mi_reduce_kernel<bf16, bf16, HS 128, 128 threads, 256, NPAR_LOOPS 1>`

**Launch.**

- **Grid** is `(NH, 1)`: 32 blocks (30B) or 12 (1.5B). **Block** is `head_size` = 128, i.e. 2 waves (AT:3350-3357). `npar_loops = ceil(5/64) = 1` (VK:72-74).
- **Arguments** (HH:230): out = attn_out; exp_sums; max_logits; tmp_out; seq_lens; nullptr; `max_num_partitions` 5; nullptr.
- **Resources:** LDS 260 B; 119 VGPR, 36 SGPR, occupancy 4 (s:attention:879). All blocks work.

**Per-block chain.**

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S** (query_start_loc, nullptr). Kernarg **S**. `seq_lens[0]` **S** (N1). `num_partitions = ceil(seq_len/256)`. | AT:1437-1443 | :881-925 |
| 2 | Wave 0: `max_logits[p]` and `exp_sums[p]` (lanes ≥ num_partitions clamp to the last one), 2 × `global_load_dword` in flight (**V**, fresh from attention). | AT:1450-1490 | :956, :962 |
| 3 | Wave 0: max over the wave, 6 `ds_bpermute` steps. exp. Sum, 6 `ds_bpermute` steps. LDS writes of `shared_exp_sums[lane]` and the global sum. | AT:1477-1515 | :985-1064 |
| 4 | All threads: 16 × `global_load_ushort` of `tmp_out[j·128 + tid]` for j < 16, clamped to the last partition (**V**, fresh), all 16 in flight. Overlaps step 2-3 in wave 1. | AT:1531-1540 | :1119-1126, :1155-1162 |
| 5 | Barrier #1. With ≤ 16 partitions, the `>16` path is skipped. | AT:1541-1563 | :1167 |
| 6 | 4 × `ds_read_b128` of `shared_exp_sums[0..15]`, 16 FMA, `ds_read` of the global sum, rcp. | AT:1566-1608 | :1595-1775 |
| 7 | Store 2 B per thread (`out[head·128 + tid]`). | AT:1612-1620 | :1830 |

**Totals.**

- Per block: 4 KB of tmp_out load traffic (unique 1 KB for 4 partitions, 1.25 KB for 5) and 2 × 256 B of max/exp traffic (unique 2 × 20 B); 256 B written.
- Per launch: 30B ≈ 147 KB traffic (≈ 33 KB unique) and 8 KB written; 1.5B ≈ 55 KB (≈ 12.5 KB unique) and 3 KB.
- Barriers: 1. Shuffle steps: 12 (wave 0). Loads in flight: 16 × 2 B (plus 2 × 4 B in wave 0).
- Static VALU 405, SALU 388 (most of it the unused 17-64 partition paths); dynamic ≈ 430 (wave 0) and ≈ 300 (wave 1).

**Chain:** 3 S → **1 V** (steps 2 and 4 overlap) → 12 LDS steps in wave 0 → barrier → store.

---

## 8. `topkGating<VPT 8, 128 experts, WARPS_PER_CTA 4, BYTES_PER_LDG 16, WARP 64, int, bf16, SOFTMAX>` (Qwen3 only)

**Launch.**

- **Constants** (TK:294-313, 615-620, VK:107-111): ELTS_PER_LDG 8; THREADS_PER_ROW = 128/8 = 16; LDG_PER_THREAD 1; ROWS_PER_WARP 4.
- **Grid** is `ceil(ceil(1/4)/4)` = 1. **Block** is (64, 4) = 256 threads, 4 waves (TK:619-623).
- **Arguments** (HH:239-240): gating = router logits (bf16); finished nullptr; topk_w; num_rows 1; topk_ids; token_expert_indices; k 8; start 0; end 128; renormalize true; bias nullptr; scale 1.0; is_padding nullptr.
- **Which threads work:** `thread_row = ty·4 + tx/16`, so only wave 0, lanes 0-15, pass the `thread_row < 1` test (TK:329-336). Waves 1-3 and lanes 16-63 exit.
- **Resources:** no LDS; 37 VGPR, 35 SGPR (s:topk_softmax_kernels:8).

**Per-block chain** (lanes 0-15 of wave 0).

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S** (num_rows). Kernarg **S** (pointers). | TK:322-346 | :10-25 |
| 2 | **V1**: 1 × `global_load_dwordx4` (8 logits, fresh from the router wvSplitK, 32 active blocks). | TK:360-375 | :49 |
| 3 | Thread max over 8, then 4 `ds_bpermute` steps (masks 8/4/2/1). 8 × `expf`. Sum, 4 steps. IEEE `1/sum`, scale, NaN/Inf clamp. | TK:410-471 | :80-100, :235-253 |
| 4 | k loop, 8 iterations (runtime k). Local argmax over 8, then 4 butterfly steps × 3 `ds_bpermute` (value-for-choice, value, index), each step waited. Lane 0: 3 × `global_store_dword` (weight, id, source_row), not waited. The owner lane clears the winner. | TK:500-579 | loop :326-590, bpermutes :370-455, stores :515-525 |
| 5 | Lane 0: `scale = 1/Σ` (IEEE divide), then the renormalise loop unrolled ×2: 4 × (`global_load_dwordx2` of the weights it just stored → `vmcnt(0)` → `global_store_dwordx2`). The first wait also drains the 24 k-loop stores. **4 serial V.** | TK:582-592 | :587-620 |

**Totals per launch:** 256 B read (fresh) plus a 32 B re-read; 96 B written plus a 32 B rewrite.

- Barriers: 0. Shuffle steps: 4 + 4 + 8 × 4 = **40 sequential LDS round trips** (104 `ds_bpermute`).
- Static VALU 361, SALU 192; dynamic ≈ 290 + 8 × ~150 + 40 ≈ 1500 instructions (one wave).

**Chain:** 2 S → V1 → 40 LDS steps interleaved with about 1500 instructions → 4 V (renormalise re-read) → store. That is **5 V**.

---

## 9. `act_and_mul_kernel<bf16, bf162, silu, packed_silu, act_first=true, use_vec=true, clamp=false, 256b=false>`

**Launch.**

- **Grid** is `num_tokens`: 8 rows for the MoE (one per selected expert slot, MP:130) or 1 for 1.5B.
- **Block** is `min(d/8, 1024)` (AC:244-256, VK:99-102): MoE d = 768 gives 96 threads (2 waves, the second half-full); 1.5B d = 8960 gives 1024 threads (16 waves).
- `num_vecs = d/2/NUM_ELTS = d/8` (AC:118). MoE: 96, so 1 iteration per thread. 1.5B: 1120, so threads < 96 do 2 iterations and the rest 1.
- **Arguments** (HH:234/235): out = mc2/act; input = mc1/gu; d = MI/I; limit 0; alpha 1; beta 0.
- **Resources:** no LDS; 61 VGPR, 45 SGPR (s:activation_kernels:8).

**Per-block chain.**

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S** (d). Kernarg **S** (pointers). | AC:106-118 | :10-28 |
| 2 | Loop body: `ld128` gate and `ld128` up (**V**, fresh: Triton fused_moe_up output from 192 blocks, or gate_up wvSplitK from 304 blocks), 2 × 16 B in flight. | AC:120-128 | :65-69 |
| 3 | 8 × (`expf` + IEEE divide + multiply), about 330 VALU per iteration (8 `v_exp`, 16 `v_div_scale`, 8 `v_div_fmas/fixup`, 8 `v_rcp`). | AC:129-134, 163-170 | :70-400 |
| 4 | `st128`, 16 B. | AC:138 | :404 |

**Totals per launch.**

| | read | written |
|---|---|---|
| MoE (8 blocks × 3 KB read, 1.5 KB written) | 24 KB | 12 KB |
| 1.5B | 35 KB | 17.5 KB |

- Barriers 0, shuffles 0. Loads in flight: 2 × 16 B.
- Dynamic per wave ≈ 30 + 341 × iterations (341 or 682).

**Chain:** 2 S → 1 V (2 V for 1.5B threads < 96, serial) → about 340 VALU → store.

---

## 10. `moe_sum_vec_kernel<bf16, int32, TOPK 8, PAD_AWARE false>` (Qwen3 only)

**Launch.**

- **Block** is 256 threads (4 waves). **Grid** is `min(ceil(1·256/256), 65535)` = 1 (MS:889-893, VK:113-117).
- **Arguments** (HH:252): out = d_norm (the buffer the next fused_add reads); input = mc3 [1, 8, 2048]; num_tokens 1; d 2048; stride_token 8·2048; stride_topk 2048; nullptr; nullptr; 0; 0.
- All 256 threads do exactly 1 iteration (`n_vec` = 256).
- **Resources:** no LDS; 54 VGPR, 56 SGPR (s:moe_align_sum_kernels:494).

**Per-block chain.**

| # | step | source | asm |
|---|---|---|---|
| 1 | Kernarg **S**. Implicit-argument and dispatch-size load **S**. `i / n_vec` as an int64 division on the SALU/VALU path. | MS:383-388 | :496-527, loop head :800 |
| 2 | 8 × `global_load_dwordx4` (one per expert slot, **V**, fresh from Triton fused_moe_down, 256 blocks), all 8 in flight. | MS:399-409 | :597-620 |
| 3 | fp32 accumulate, cast, store 16 B. | MS:411-414 | :797 |

**Totals:** 32 KB read, 4 KB written. Barriers 0, shuffles 0. Loads in flight: 8 × 16 B. Static VALU 264, SALU 182; dynamic ≈ 300.

**Chain:** 2 S → **1 V** → store.

---

## 11. Launches outside the requested list

**`rms_norm_kernel<bf16,8,2,true>`, layer-0 in_norm only.**

- Grid 1; block H/8 (256 or 192). The chain is the same as §3 but over H: 58 VGPR, 59 SGPR, 2 barriers, 10 shuffle steps (s:layernorm_kernels:8). Arguments at HH:221.

**Triton `fused_moe_kernel` (up/down).**

- Not a HIP kernel from vLLM's C++ sources; its assembly is in `examples/vllm_llm/triton`.
- Inventory form (ETX import): 76 VGPR, 5 slot barriers.

**`final_norm` and `lm_head`.**

- `final_norm` is §1.
- `lm_head` is wvSplitK YTILE 2 with wvprgrp 16: 16 waves active in all 304 blocks, 15-16 m-iterations per wave, 2 K-trips each, so ≈ 32 weight round trips per wave. Weights are 622 MB (30B) or 467 MB (1.5B). Not needed per layer.

---

## 12. Cross-check: vLLM form vs. ETX imported form

From `isa_inventory` of `qwen3-30b-a3b_base`; the ETX form wraps each kernel in 1024-thread workgroups.

| kernel | vLLM VGPR / LDS / scratch / hardware barriers | ETX import VGPR / LDS / hardware + slot barriers | notable difference |
|---|---|---|---|
| wvsplitk_y2 | 45 / 64 KB / 0 / 1 | 48 / 32 896 B / 2 + 0 | LDS capped at 32 KB (VK:54 `lds_cap`); same loop (10-instruction fill, 16 B loads) |
| wvsplitk_y1 | 48 / 64 KB / 0 / 1 | 48 / 32 896 / 2 + 0 | – |
| attn_mfma16_g8 | 96 / 8 KB / 16 B / 7 | 104 / 32 896 / 0 + 7 | no scratch in ETX; `__syncthreads` becomes slot barriers (LDS spin) |
| attn_reduce | 119 / 260 / 0 / 1 | 108 / 32 896 / 0 + 1 | – |
| rms_norm_3d | 58 / 68 / 0 / 2 | 32 / 32 896 / 0 + 2 | – |
| fused_add_rms_norm | 40 / 68 / 0 / 2 | 40 / 32 896 / 3 + 0 | – |
| rope, cache, topk, silu, moe_sum | 15, 44, 37, 61, 54 | 24, 52, 40, 32, 44 | static load counts within ±1 (rope 8 vs 9, cache 21 vs 22) |

---

## 13. Summary (one layer, one block of each launch)

- **Round trips** are counted per block on the critical path. S = scalar (kernel arguments, or per-step device data through the scalar cache); V = vector global. Counts assume seq_len ≤ 1024.
- **Shuffle steps:** "DPP" means in-register cross-lane operations; every other count is `ds_bpermute` LDS round trips.

| kernel (model) | grid x block | bytes read per launch | sequential RTs per block | barriers per block | shuffle/reduction steps per block | streaming loads in flight per thread |
|---|---|---|---|---|---|---|
| fused_add_rms_norm (30B / 1.5B) | 1 x 1024 | 16 KB / 12 KB | 2 S + 2 V | 2 | 10 | 2 × 16 B |
| wvSplitK qkv 30B (5120 x 2048) | 304 x (64,16) | 20.97 MB W + 1.25 MB act | 4 S + 3 V | 1 | DPP 8 (×2 rows) | 4 × 16 B |
| wvSplitK o_proj 30B (2048 x 4096) | 304 x (64,16) | 16.78 MB + 2.49 MB | 4 S + 5 V | 1 | DPP 8 (×2) | 4 × 16 B |
| wvSplitK router 30B (128 x 2048, YTILE 1) | 304 x (64,16) (32 blocks with rows) | 0.52 MB + 1.25 MB | 4 S + 2 V | 1 | DPP 8 | 4 × 16 B |
| wvSplitK qkv 1.5B (2048 x 1536 + bias) | 304 x (64,16) | 6.29 MB + 0.93 MB | 4 S + 4 V | 1 | DPP 8 (×2) | 4 × 16 B |
| wvSplitK o_proj 1.5B (1536 x 1536) | 304 x (64,16) | 4.72 MB + 0.93 MB | 4 S + 3 V | 1 | DPP 8 (×2) | 4 × 16 B |
| wvSplitK gate_up 1.5B (17920 x 1536) | 304 x (64,16) | 55.05 MB + 0.93 MB | 4 S + 5 V | 1 | DPP 8 (×2) × 2 m-iterations | 4 × 16 B |
| wvSplitK down 1.5B (1536 x 8960) | 304 x (64,16) | 27.53 MB + 5.45 MB | 4 S + 11 V | 1 | DPP 8 (×2) | 4 × 16 B |
| rms_norm_3d q_norm / k_norm (30B) | 32 x 16 / 4 x 16 | 24 KB / 3 KB | 4 S + 2 V | 2 | 10 | 1, then 2 × 16 B |
| rotary_embedding neox (30B / 1.5B) | 1 x 512 | 18 KB / 7 KB (2 B scalar loads) | 3 S + 5 V / 3 S + 3 V | 0 | 0 | 4 × 2 B |
| reshape_and_cache (30B / 1.5B) | 1 x 64 / 1 x 32 | 2 KB / 1 KB | 3 S + 9 V | 0 | 0 | 1 × 16 B, then 1 × 2 B |
| paged_attention mfma16 GQA 8 (30B) | (1,5,4) x 256 (16 active) | ≈ 2.2 MB (128 KB KV per block) | 4 S + 5 V + 2 scratch | 7 | 4 | K ≈ 8 × 16 B; V 16 × 16 B |
| paged_attention mfma16 GQA 6 (1.5B) | (1,5,2) x 256 (8 active) | ≈ 1.1 MB | 4 S + 5 V (+ scratch, not verified) | 7 | 4 | as GQA 8 |
| paged_attention reduce (30B / 1.5B) | 32 x 128 / 12 x 128 | 147 KB / 55 KB traffic (33 / 12.5 KB unique) | 3 S + 1 V | 1 | 12 | 16 × 2 B |
| topkGating 8/128 (30B) | 1 x (64,4) (16 lanes active) | 256 B (+32 B re-read) | 2 S + 5 V | 0 | 40 | 1 × 16 B |
| act_and_mul silu, MoE 8 x 768 (30B) | 8 x 96 | 24 KB | 2 S + 1 V | 0 | 0 | 2 × 16 B |
| act_and_mul silu, 1 x 8960 (1.5B) | 1 x 1024 | 35 KB | 2 S + 2 V | 0 | 0 | 2 × 16 B |
| moe_sum_vec top-8 (30B) | 1 x 256 | 32 KB | 2 S + 1 V | 0 | 0 | 8 × 16 B |

**Per-layer totals, HIP launches only (excluding Triton).**

| | launches | S + V round trips (sum of the critical-path counts above) | barriers | LDS shuffle steps |
|---|---|---|---|---|
| Qwen3-30B-A3B | 14 | ≈ 43 S + 45 V + 2 scratch | 19 | 96 |
| Qwen2.5-1.5B | 11 | ≈ 35 S + 47 V + 2 scratch | 16 | 36 |

The barrier and shuffle figures sum the per-block values with no parallelism between kernels. The DPP chains are extra: 30B 3 × 8, 1.5B 4 × 8, plus the gate_up repeat.
