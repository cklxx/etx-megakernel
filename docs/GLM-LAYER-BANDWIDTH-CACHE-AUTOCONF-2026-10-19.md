# One GLM layer: bytes, bandwidth, cache and tile search, 2026-10-19

Offline analysis only; no GPU was used. Inputs: the GLM-5.3-Flash dimensions in
`docs/QWEN-BATCH-GLM53-RUNBOOK.md:243-248`, the measured stream rate in
`examples/vllm_llm/batch_pp.py:10`, the cache facts in
`docs/RESEARCH-sync-hardware.md:13,98`, and this repo's cache-bit conventions.
Labels: [M] measured, [E] arithmetic from stated inputs, [U] absent.

## 1. One GLM layer per step, in bytes

Dimensions (full scale): hidden 4096, MoE intermediate 2048, 288 routed experts,
top-8, one shared expert, FP8 weights with FP32 128×128 block scales
(`docs/QWEN-BATCH-GLM53-RUNBOOK.md:243-248`; `docs/QWEN-BATCH-GLM53.md:84`).

- One expert, all three matrices:
  3 × 2048 × 4096 × 1 B = **25.17 MB** (FP8). Block scales add
  3 × (2048/128) × (4096/128) × 4 B = 6.1 KB, 0.024%, ignored.
- **Batch 1, one layer:** 8 routed experts 201.3 MB + shared expert 25.2 MB +
  router (288 × 4096 × 1 B = 1.18 MB) = **227.7 MB**.
- **Batch 1, one step:** 45 layers × 227.7 MB = **10.25 GB** if every layer is
  MoE; the dense/MoE split of the 45-layer GLM-5.3 is [U] in our record.
- **B8, one layer:** 8 requests × 8 experts = 64 expert reads =
  **1,611 MB** replicated. The union of 8 requests over 288 experts is
  288 × (1 − (280/288)^8) = **58.1 experts = 1,462 MB**, so E2-style dedup
  saves 149 MB per layer, **9.2%** — less than Qwen's 19%, because 288 experts
  spread the routing wider.

## 2. What those bytes cost

At the measured 3.56 TB/s: a batch-1 layer is 64.0 µs and a step is **2.88 ms**
[E]; a B8 layer is 452 µs. At the best observed 3.93 TB/s (§3) the same bytes
take 9.4% less: 2.61 ms [E]. Qwen3-30B-A3B moves 6.08 GB/token, GLM 10.25 GB —
GLM's floor is 1.7× Qwen's before any kernel overhead.

## 3. The 3.56 TB/s against the 5.3 TB/s peak

The 3.56 TB/s figure is `hw_bench lf` **at 192 × 64 KB in flight, with 3.93 TB/s
observed as the maximum** (`examples/vllm_llm/batch_pp.py:10`) [M]. Against
MI300X's 5.3 TB/s peak that is **67.2% sustained, 74.2% at the observed best**.

- The gap to peak is 1.37–1.74 TB/s. The record does not contain a sweep of the
  in-flight bytes that produced 3.93, so whether more queues close it is [U].
- Everything in §1 uses 3.56. If the kernel reaches 3.93 the same bytes cost
  9.4% less; that is the entire claim we can make from the record.

## 4. Cache use

**What a layer re-reads.** Every weight is re-read every decode step. One
layer's B8 routed stream is 1,611 MB against **4 MB of L2 per XCD (32 MB for 8
XCDs)** (`docs/RESEARCH-sync-hardware.md:13,62`) and a **256 MB MALL**
(`docs/RESEARCH-sync-hardware.md:98`) — 50× L2 and 6.3× MALL. Nothing holds the
routed stream; it is HBM-bound by construction. The shared expert and router
(26.4 MB per layer) are the only reused part: they fit MALL at 10.3% of it and
are read by every token.

**Which loads carry which bits.** In this repo's ISA the two coherence bits are
`sc0` = GLC (L1-level coherence; `etx/tools/isa_hops.py:112`,
`etx/runtime/include/etx/primitives_amdgcn.h:27-31`) and `sc1` = SLC
(system/L2 level; `etx/tools/isa_hops.py:116`, `etx/ops/coh.py:2,27`). Recommended
policy [E, to be measured]:

| Data | Bits | Why |
|---|---|---|
| Routed-expert stream, read once per step | `slc` (SLC, bypass L2) | it cannot be reused in-step, and allocating it in L2/MALL evicts the shared expert and router |
| Shared expert, router, dense projections | plain cached, no `glc`/`slc` | reused within the step and across steps; these are the only MALL-resident candidates |
| Cross-XCD activations and payloads | keep `sc1` agent-scope loads and `sc0 sc1` system-scope atomics (`etx/ops/coh.py:27-35`) | already measured; changing them reopens a correctness surface |
| Weight pre-warming | none | L2 warming measured **−3%** (`docs/RESEARCH-sync-hardware.md:89`) |

**LDS residency at 64 KB per CU.** Staging bf16 activations of K = 4096 costs
4096 × 2 B = 8 KB per token: **6 tokens = 48 KB (75%), leaving 16 KB for
double-buffering; 8 tokens = 64 KB (100%), no headroom.** That is exactly the
gfx942 port's 6-token pass width (0.75 × 64 KB ÷ 8 KB = 6) [F]. Weights are
never LDS-resident at this size — one expert is 25.17 MB against 64 KB — so
every weight byte streams through the CU; LDS only ever holds the activation
slab. For B8 that means **two passes (6 + 2)**, or FP8 activation staging
(8 × 4096 × 1 B = 32 KB, 50%), which is numerically unproven [U].

## 5. Tile auto-optimisation

**What `examples/vllm_llm/autoconf.py` already searches.** Five build switches:
`ETX_VL_CHAIN` (2), `ETX_VL_QKNR` (3), `ETX_VL_DSL_GEMV` (2), `ETX_VL_DSL_TOPK`
(2), `ETX_VL_DSL_ATTN` (2) — 48 combinations, one invalid pair rule
(`autoconf.py:25-31,34-36`). It compiles each plan locally, costs every task
from its tile (`kmodel.kern_for_symbol`) and simulates the step on the XCD
simulator with the measured phase model (`autoconf.py:57-69`). No GPU.

**What is missing.** The search space it does not touch:

- **BM/BN** — tile constants in `etx/ops/gemv.py` and the GLM tiles, not build
  switches;
- **waves per SIMD** — occupancy of the tile, fixed by the workgroup shape;
- **VGPR budget** — the compiler report (VGPRs, spills, scratch) is not read
  back into the search;
- **LDS footprint** — the 6-token/64 KB constraint in §4 is not modelled, so the
  simulator will happily rank a tile that does not fit gfx942's LDS;
- no per-config measurement: the ranking is simulated, and the measured build
  used as the reference (`autoconf.py:75`) is a single point.

**Closing the loop on our box budget.** One session script, prepared offline:

1. enumerate the space locally (BM × BN × waves × LDS budget), free;
2. compile every candidate for gfx942 with the local matrix
   (`examples/vllm_llm/local_matrix.sh`), drop those over the VGPR / spill /
   LDS limits from the compiler report, free;
3. measure the survivors on one MI300X in one script: pinned, under `flock`,
   five interleaved rounds, B8 fixed at 8 rows, one microbench per candidate;
4. write the measured µs back into `pmodel`'s constants and re-rank.

Budget [E]: 36 survivors × 2 min = 72 min, plus 20 min setup, ≈ 1.5 h ≈ **$3 at
$1.99/h**. Run it in the same session as E4 so the box is paid once.

## 6. Three changes to build first, and the one that waits

Ranked by expected ms on MI300X (details and falsifiers in
`docs/MONO-DECODE-PORT-PLAN-2026-10-19.md:89-111`):

1. **E1 — dense column batching:** −4.84 ms at B8, −2.07 ms at B4 [E].
2. **E3 — imported `attn_reduce`:** −1.5 to −2.0 ms at B8 [E].
3. **E2 — MoE union:** −1.58 ms at B8, −0.37 ms at B4 for Qwen [E]; 9.2% of the
   expert stream for GLM's 288 experts (§1).

**Must wait for a real-weight run:** the tile search in §5 and the cache-bit
policy in §4. The GLM real-weight run has never happened
(`docs/TWO-WEEK-RESULTS-2026-10-18.md:72` [U]), so every GLM per-layer number in
this note is arithmetic on stated dimensions, not a measurement; tuning BM/BN
or `slc` against them would tune to an unrun model.
