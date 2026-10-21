# Multi-GPU support for large models: how and in what order, 2026-10-21

Owner decision, not reopened here: ETX must support multi-GPU execution and models larger than one GPU.
This document decides only how, and in what order.
GLM-5.3-Flash is the worked case: it is the large model in the repo, and the only one that does not fit one device.

Labels, as in `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:8-10`:
- [M] measured, with the file:line of the measurement;
- [E] arithmetic from cited inputs;
- [U] not in the record.

No GPU was used for this document.

## 1. What the record says

| Fact | Value | Source |
|---|---|---|
| GLM-5.3-Flash text model | 45 layers, H=4096, 288 routed experts top-8, one shared expert | `docs/QWEN-BATCH-GLM53.md:84-85` |
| Native compressed weights | 305.707 GiB (an estimate, not a measured allocation) | `docs/QWEN-BATCH-GLM53.md:185` |
| Cache at B8, context 1024 | about 6.85 / 5.74 GiB per rank | `docs/QWEN-BATCH-GLM53.md:186` |
| Device capacity in the plan | 192 GiB per MI300X | `docs/QWEN-BATCH-GLM53.md:174` |
| Current partition | PP2, whole layers: 0–23 on rank 0, 24–44 on rank 1 | `docs/QWEN-BATCH-GLM53.md:107-108` |
| Seam crossings | once per decode step, `[B,4,4096]` BF16 | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:85` |
| Cross-step overlap | none: both ranks reset before either launches | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:92-93` |
| Kernels per step | two, one per device; fusion is already at its ceiling within a rank | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:57-61` |
| Tile parallelism | one workgroup per request per layer, so only B workgroups do work | `docs/QWEN-BATCH-GLM53.md:100`, `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:41-42` |
| `threads` knob | does not split a layer into more workgroups | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:169` |
| Batch-1 bytes per sparse layer | 227.7 MB | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:34` |
| All-MoE-equivalent floor, one GPU | 10.25 GB / 2.88 ms at 3.56 TB/s [E] | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:35` |
| Streaming bandwidth | 3.56 TB/s at 192 × 64 KB in flight [M] | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:46` |
| Cross-device flag, system scope, one-way | 10.6 µs [M] (2× MI300X) | `docs/ETX_Technical_Design.md:1293` |
| Peer store bandwidth | 285 GB/s [M] | `docs/ETX_Technical_Design.md:1294` |
| Whole-GPU phase switch | 8.61 µs [M] | `docs/ETX_Technical_Design.md:1291` |
| GLM seam latency | [U]; the 6.16 µs figure is a simulator estimate, not a measurement | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:401-404` |
| Any GLM timing | none produced yet [U] | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:506-507` |
| 1.72× weight-copy factor | a constant in the runner. The repo gives no derivation and no measurement, so its origin is [U] | `examples/glm53/run_all.py:52` |
| Mono on gfx942 | does not exist as of 2026-10-20 | `docs/GLM-EXECUTION-DESIGN-2026-10-19.md:337` |
| MI325X, 8-GPU node | no measurement or design in the repo [U] | — |

## 2. The ordering argument

Every multi-GPU layout inherits each GPU's own streaming rate. Today that rate is set by B workgroups, not by the chip.

**Per-token time today, estimated [E]:**
- **Inputs:** 227.7 MB per sparse layer (§1). A contended per-workgroup rate of 3.56 TB/s / 192 = 18.5 GB/s, from the in-flight configuration at `GLM-EXECUTION-DESIGN-2026-10-19.md:46`.
- **Per layer:** one workgroup streams the layer in 12.3 ms.
- **Per token:** 42 sparse layers (`GLM-EXECUTION-DESIGN-2026-10-19.md:36`) take about 517 ms. That is about 180× the 2.88 ms floor.
- **The uncontended single-workgroup rate is [U].** If it is 5× higher, the estimate is still about 100 ms, more than 30× the floor.

Consequences:
- **No layout comparison is meaningful until the tiles use the chip.** PP2, TP2 and EP2 would be compared on a 1-workgroup stream.
- **PP2 does not add bandwidth at batch 1.** Only one rank works at a time, so its batch-1 ceiling is the one-GPU floor, 2.88 ms [E].
- **A split inside the layer runs both GPUs at once.** Its ceiling is 1.44 ms on two GPUs [E], minus the synchronisation cost (§3.3).

So the order is: first the GPU's own parallelism, then the cross-GPU cost measurement, then the layout that needs both.

## 3. Capabilities

Each subsection gives the gain, the gate, and what a failure looks like when the gate is missing.

### 3.1 Layer-parallel native tiles (split one layer over many workgroups)

- **What:** each sparse layer's routed experts, shared expert and dense projections are distributed over all CUs. Today one workgroup does a whole layer for one request.
- **Gain:** from the about 517 ms/token estimate [E] (§2) toward the 2.88 ms one-GPU floor [E].
  - Qwen evidence that a fused plan can approach the bandwidth floor on this GPU: ETX fused 4.471 ms on a 6.08 GB model [M] (`CLAUDE.md:8`, `docs/DESIGN-acquire-2026-10-05.md:233`).
  - The GLM share of the floor reachable this way is [U].
- **Gate:**
  - the slice oracle (§6 of the design doc) and the numerics bar: 32/32 tokens, per-token cosine ≥ 0.999, relative error ≤ 0.02 (`GLM-EXECUTION-DESIGN-2026-10-19.md:471-484`);
  - the same build as the timing.
- **Without the gate:** a parallel tile that changes the summation order passes token equality on tie-breaking and ships a numerical regression. The bar exists for exactly this case (`GLM-EXECUTION-DESIGN-2026-10-19.md:486-489`).

### 3.2 Measured cross-GPU cost in the production shape

- **What:** run the existing production-shaped seam probe (`examples/glm53/seam.py`, job `examples/sessions/glm53_seam.py`; `GLM-EXECUTION-DESIGN-2026-10-19.md:415-458`) for B ∈ {1,2,4,8}.
  - No new code. Minutes of GPU time [E].
- **Gain:** none by itself. It decides between §3.3 and keeping PP2.
  - The nearest measurement, the one-way cross-device flag at 10.6 µs [M], comes from a different graph (`ETX_Technical_Design.md:1293`).
- **Gate:**
  - every row and token is checked;
  - five rounds with the arm order reversed;
  - no GPU 0 minus GPU 1 timestamp arithmetic (`GLM-EXECUTION-DESIGN-2026-10-19.md:430-441`).
- **Without the gate:** a seam figure built from cross-GPU timestamp differences, or from the simulator's 6.16 µs, would size §3.3 on a number that does not exist.

### 3.3 Two-GPU split inside the layer (expert-parallel routed experts)

- **What:** each sparse layer's 288 routed experts are split across GPUs, so a token's 8 experts stream on both devices at once.
  - One combine exchange per sparse layer.
  - Attention, mHC and dense state stay with one owner or are replicated. Which is cheaper is [U].
- **Gain [E]:** the batch-1 ceiling falls from 2.88 to 1.44 ms. The exchange costs 42 × 10.6 µs = 0.45 ms, using the flag latency at `ETX_Technical_Design.md:1293`. Net ceiling ≈ 1.89 ms against PP2's 2.88 ms.
- **Gate:**
  - same as §3.1;
  - plus §3.2's measured exchange, so the cost line uses the production-shape number, not the flag benchmark.
- **Without the gate:** a split that loses on synchronisation looks like a win in a simulator. The repo already records simulator-only gains that never met their GPU threshold (PRs #23–#27, closed 2026-10-21).

### 3.4 Placement planner for G GPUs and any model size

- **What:** given the per-layer weight bytes, cache shapes, a weight-copy factor and device capacity, assign layers or experts to G ranks. Refuse any plan whose per-rank sum exceeds capacity.
  - CPU only.
  - The host already checks free memory per rank at load (`docs/QWEN-BATCH-GLM53.md:113-115`). The planner moves that check to build time.
- **Why it is a capability, not bookkeeping:**
  - At PP2, rank weights are about 305.707 / 2 = 152.9 GiB, plus about 6.85 GiB of cache, which is about 159.7 GiB of 192 [E].
  - If the 1.72× copy factor applies to weights, the total is 525.8 GiB, more than 2 × 192 GiB [E]. GLM-5.3-Flash would then need at least 3 devices, or 4 at 131.5 GiB each [E].
  - Whether the factor applies to residency at all is [U] (`run_all.py:52` reports it only as a separate bandwidth line).
- **Gain:** fit or no fit. It is the condition for "large models must be supported" on any G.
- **Gate:** a planner refusal test for a plan over capacity, and agreement with the host's measured allocations on the first box [U until run].
- **Without the gate:** a plan that fits on paper fails at load after the 1–2 minute weight read. A wrong per-rank sum is a silent out-of-memory on a paid box.

### 3.5 Cross-step pipelining for B > 1

- **What:** let rank 0 start step t+1 while rank 1 finishes step t. Today both ranks reset before either launches (`GLM-EXECUTION-DESIGN-2026-10-19.md:92-93`).
- **Gain [E]:** up to 2× throughput on two ranks at B ≥ 2, with batch-1 latency unchanged.
  - How much of the rank-1 bubble is fillable is [U]; no per-layer timing exists (`GLM-EXECUTION-DESIGN-2026-10-19.md:90`).
- **Gate:**
  - per-request 32/32 under interleaving;
  - the timing harness per batch (`GLM-EXECUTION-DESIGN-2026-10-19.md:495-534`).
- **Without the gate:** requests that share a rank's caches across steps can read another step's state. Only a per-request token gate catches that.

### 3.6 Hardware beyond two MI300X

- 8-GPU nodes and MI325X have no measurement in this repo [U]. The xGMI topology is recorded as a point-to-point mesh without a switch (`ETX_Technical_Design.md:469`).
- Any TP4 or two-group plan inherits §3.1–§3.4 unchanged. It needs §3.2's measurement repeated per topology, not assumed.

## 4. Decision list

**Build first, in this order:**

1. **Layer-parallel native tiles (§3.1).**
   - Gain: from about 517 ms/token [E] toward the 2.88 ms one-GPU floor [E].
   - Falsifying experiment: the first-box slice run, layers 22–25, batch 1, on the current tiles (`examples/glm53/run_all.py`).
   - Decision number: if it reports `fraction_of_floor` ≥ 0.25 (that is, ≤ 11.5 ms/token-equivalent), the §2 estimate is wrong, tiles are not the bottleneck, and §3.3 moves to first.
2. **Placement planner (§3.4).**
   - Gain: fit or no fit for any G. With the 1.72× factor applied to residency, GLM-5.3-Flash needs at least 3 GPUs instead of 2 [E].
   - Falsifying experiment: compare the planner's per-rank sum with the host's recorded allocations on the first box.
   - Decision number: a disagreement above 5% of device capacity (9.6 GiB) means the planner's inputs are wrong. Fix them before any G > 2 plan.
3. **Two-GPU split inside the layer (§3.3), gated by §3.2's measurement.**
   - Gain: batch-1 ceiling 1.89 vs 2.88 ms [E].
   - Falsifying experiment: the seam probe (`examples/glm53/seam.py`), B=1.
   - Decision number: build only if the measured production-shape exchange is ≤ 16 µs. Then 45 × 16 µs = 0.72 ms, half of the 1.44 ms saving [E]. Above 32 µs the saving is gone (45 × 32 µs = 1.44 ms) [E]: keep PP2 and take §3.5 instead.

**Do not build in the next month:**

1. **Byte- and cache-level tuning:** in-place a4w8 dequant, cache-bit policy, the six-token LDS slab, tile-knob sweeps.
   - Halving bytes on today's tiles takes about 517 to about 258 ms/token, still about 90× the floor [E]. §3.1's ceiling is about 180× [E].
   - §3.1 replaces the tiles' inner loop, so a dequant built into today's tiles is rework.
   - L2 pre-warming is already measured at −3% (`GLM-EXECUTION-DESIGN-2026-10-19.md:123`).
   - Number that changes this: after §3.1, a measured `fraction_of_floor` ≥ 0.5. Bytes then become the limit, and the a4w8 microbenchmark threshold in `docs/EXPERIMENTS-GLM-2026-10-20.md` decides.
2. **Deeper pipeline parallelism (PP > 2) or per-layer hand-off for batch-1 latency.**
   - Each stage adds a seam crossing and no bandwidth at batch 1 (§2). The per-layer hand-off's cost is undesigned and unmeasured (`GLM-EXECUTION-DESIGN-2026-10-19.md:407-411`).
   - Number that changes this: §3.4 showing that G > 2 is required for capacity (for example, the 1.72× factor applying to residency). PP over G ranks then becomes the cheapest layout that fits, still to be measured against §3.3.
