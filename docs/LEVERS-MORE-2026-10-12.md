# More levers, priced, with their mechanism simulated on the CPU (2026-10-12)

**Condition.** One MI300X VF, Qwen3-30B-A3B, bf16, batch 1, context 1024, 32 generated tokens. ms per decode
step on host wall unless marked device. Anchor: KEY32-on fused **4.180 wall / 4.045 device [M]**
(`results_qwen300_2026-10-11/job2_run4.log:144-145`). Need for 300 tok/s: **0.847 ms**. Labels as in
`docs/PLAN-300-CERTAIN-2026-10-12.md` (CERT): [M] measured, [E] estimate with arithmetic, [R] residual, [U]
unknown (adds nothing). Results directories below live in `~/code/etx/` (outside git). No GPU was used.

## 1. The 0.80 ms anchor drift: not a source or flag change

**Verdict: the drift is the machine, not the build.** The two builds' device code is identical, and vLLM's own
kernels, compiled from byte-identical source by the same compiler, ran **13-19% slower** in the hour session.
It is not a lever at the source level. It is a session risk with a cheap guard (section 5).

**1a. Flags.** The block quoted in the brief (`raw/hour_build.json`) is the **stamps** build (`ETX_INBODY_BUILD=1`,
`ETX_OUT=.../stamps`); `raw/hour_build.json` and `session-wklfw6bc/stamps_hour_build.json` are byte-identical.
The timed **kept** build is `session-wklfw6bc/kept_hour_build.json`, the same block without `ETX_INBODY_BUILD`.
The 10-11 build environment, recovered from the session script that ran it (`git show
7c85efb:examples/sessions/qwen300.py:355-367`) and its log (`results_qwen300_2026-10-11/job2_run4.log:27-30`):

| Variable | 10-11 build (`b_on`) | Hour kept build | Effective difference |
|---|---|---|---|
| `ETX_HIPCC_FLAGS` | `-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH` | same + `-DETX_INBODY_REPORT` | host `printf` only (1c) |
| `ETX_TOPK_KEY32`, `ETX_VL_CHECK` | 1, 1 | 1, 1 | none |
| `ETX_VL_CU` | unset (all `ETX_VL_*` stripped) | 256 | none: default 256 (`examples/vllm_llm/model.py:72`) |
| `ETX_VL_REQUESTS` | unset | 1 | none: default 1 (`model.py:193`) |
| `ETX_VL_TOUCH` | unset | 1 | none: `_on()` defaults to on (`model.py:63-67,138`) |
| `ETX_DEV_RESET` | unset | 0 | none: default 0 (`model.py:154`) |
| `ETX_PREFETCH_ISOLATE_POLLER` | unset | 0 | none: default 0 (`etx/codegen/kernel.py:151` at e068e47) |
| `VLLM`, `ETX_TORCH_HEADERS` | `~/vllm`, `~/rocm-headers/pytorch` | unset | none: import outputs hash-identical (1b) |
| Runtime `ETX_VL_PROGRESS` | 64 | 1 (`qwen300_hour.py:177-178` at e068e47) | host `printf` outside the timer |
| Runtime `ETX_SYNC` | `''` | unset | none: only `block` switches (`host.h:186-187`) |

**1b. Generated code, regenerated on this Mac.** I re-ran `etx.ops.emit`, `etx.importer.adapter` and `etx compile`
from each certificate's `imports.json` at both trees. Every hashed artefact reproduces the VM's sha256 exactly:

| File (certificate `build_identity.json`) | 10-11 hash | Reproduced at | Hour hash | Reproduced at |
|---|---|---|---|---|
| `dsl_tiles.hip` | a0bcc2dc | 7c85efb, a4ca716, d7c1234 | 29e5f250 | e068e47 |
| `megakernel_d0.hip` (VM paths substituted) | 650781f6 | 7c85efb | bc4c8cd3 | e068e47 |
| `plan_data.h` | a3fb4817 | 7c85efb | 88a08466 | e068e47 |
| `vl_plan.txt` / `plan.json` | 046be64d / b32c3299 | both | 046be64d / b32c3299 | both |
| `imp_tiles.hip`, `imports.json`, `orig_*.hip` | d4bf266a, 04274de5, same | both | identical | both |

The 10-11 commit is therefore any of 7c85efb..d7c1234 (one equivalence class for every generated file). The diff:

- `megakernel_d0.hip`: **identical** except three `#include` paths.
- `dsl_tiles.hip` (50 lines): the staged-norm helper gains `template <int FixedK = 0, int FixedR = -1>` with
  `K = FixedK ? FixedK : d.K` (identity at the defaults) and an `#ifdef ETX_DSL_NORM_HOIST` block (C5; undefined).
- `plan_data.h`: one `#if defined(ETX_PRODUCER_LEDGER)` table (undefined). `abi.h`: fields under `#if ETX_INBODY` only.
- **Device IR**: both megakernels compiled to gfx942 LLVM IR locally (`hip_ir_local.sh`, same defines). After
  removing the HIP `cuid` and inline-asm `srcloc` line numbers, **0 differing lines** in 29.2 MB (section 4, S0).
  The VM's resource remarks agree (128 VGPR, 265 SGPR spills, 29,736 B LDS in both `hipcc.log`s).
- Host (`host.hip:714,725-728,758,796,824` at e068e47): `-DETX_INBODY_REPORT` prints one line before `w0`
  and others after the timer, and passes a 5 s watchdog instead of 60 s. The device number is the in-kernel
  `token_time` span (prologue to argmax), which host code cannot lengthen.

**1c. What did change: the machine and the data [M].** Standalone launches of vLLM's **original** kernels in
the import checks (`check_p0.txt`, `check_p1040.txt`, layers 1-47, median of 47 each; S0b in section 4):

| Session | attn original p0 / p1040 us | reduce original p0 | imported attn p0 | imported reduce p0 |
|---|---:|---:|---:|---:|
| vlbest 10-09, vlbest2 10-10 | 7.30 / 8.20 | 6.60 | 9.50 | 10.30 |
| qwen300 10-11 on / off | 7.50 / 8.30, 7.40 / 8.30 | 6.60, 6.50 | 9.30, 9.50 | 10.20, 10.30 |
| hour W kept / W stamps / J kept | 8.50 / 9.30, 8.50 / 9.20, 8.70 / 9.20 | 7.50, 7.10, 7.10 | 10.70-10.90 | 11.70-12.20 |

`orig_attention.hip` has the same sha256 (6735281a) in both builds, and both use the same compiler build
(46fcb339…). So a +13-16% rise in that kernel is not ETX code. The megakernel's +19.6% device rise
(4.840 / 4.045) has the same sign and size. Second, the data were not the same either. CHECK-DSL's layer-47
inputs at the final step are deterministic: four earlier gates (10-09, 10-10, 10-11 on and off) print identical
`topk dev ids [60 117 24 14 97 79 126 122] wts [0.288143 …]`. All four hour gates print `… 126 109] wts
[0.283657 …]` (`kept_token_gate.log`, `stamps_`, `stages_`, J). The golden (bab2f1fb) and `vl.manifest`
(d95a244f) hashes agree, so the staged `vl.bin` bytes or HF snapshot on the hour VM are the [U] candidate. The
hour runner does not pack weights; it uses what was staged (`qwen300_hour.py:98`). Neither data difference
explains a slower position-0 attention kernel. A VF clock or power state does [E].

**Cheapest GPU test (part of every next session, 4 GPU minutes, $0.13):** run `rocm-smi --showclocks
--showpower --showperflevel` and the import check's original-kernel medians first. Then run the kept gate from
both trees (7c85efb and e068e47 rebuilt, same flags) as two pinned pairs, and record `sha256sum vl.bin`.
**Decision:** if the two trees are within 0.030 ms of each other, the drift is environmental and closed. Then, if
original attention is above 7.9 us (+5% over 7.5), the VM is slow: record and reprovision, or rebase every row.
Tree A slower than tree B by ≥ 0.030 ms falsifies this section.

## 2. Six new mechanisms, ranked inside the existing ledger

**Result: five candidates clear KEEP in the nominal CPU scenario; none is a measured gain.** These are
new mechanisms for existing pools, not six new pools. No source-level drift credit, precision change,
dropped token, or extra credit for KEY32/C1/C5/C6/TOUCH/X7/host controls is included.
CERT means `docs/PLAN-300-CERTAIN-2026-10-12.md`; MTD means `docs/MASTER-TABLE-2026-10-11.md`.
Evidence directory **L** = `docs/levers-more-2026-10-12/`; **I** =
`~/code/etx/results_vlbest2_2026-10-10/b_all/imp/attention.hip` (retained original source).
ISA-FIXES/MOE-BODY/DATA/design below name the corresponding files in `docs/`, including
`ISA-FIXES-2026-10-11.md`, `MOE-BODY-ANATOMY-2026-10-11.md`, `DATA.md`, `ETX_Technical_Design.md`.

Prices are **[E], conditional expected recovery for ranking**, transferring simulated device cuts to
wall at 1:1. They assume exposed loads at 0.69 us, and 50% exposure of nominal LDS service. Neither
assumption was measured inside these bodies. Zero exposure gives **zero** for every candidate;
half-load exposure gives P/R/A/T only 0.038/0.033/0.033/0.017 ms. Thus this is an experiment ranking,
not a forecast with confidence intervals. GPU minutes include a warm staged pack, correctness and short
pinned pairs; cold staging and final five-round certification have separate budgets (§5).

| Rank / ID | New mechanism | Expected ms [E] | GPU min [E] | ms/min | Lives inside; source | Kill bar |
|---|---|---:|---:|---:|---|---|
| 1 P | Prep: position + slot issued alongside q/k reads, then cos/sin alongside RMS | 0.07584 | 6 | 0.012640 | A3 0.126; `etx/ops/attnprep.py:114,124,126,141` | No exclusive ≥0.625 us/layer, or KEEP fails |
| 2 R | Reduce: max logits, exp sums and tmp_out issued together after seq_len | 0.06624 | 8 | 0.008280 | B1 reduce share 0.119; I:1467,1487,1536 | Compiler already overlaps them, or KEEP fails |
| 3 T | Topk: issue its two coherent logit loads before one wait | 0.03312 | 4 | 0.008280 | A7 0.093; `etx/ops/topk.py:54-60`; L/isa_order_rocm22.txt:10-13 | Loaded RTT <0.625 us or KEEP fails |
| 4 A | Attention: V metadata/payload overlap Q staging and K reads | 0.06624 | 10 | 0.006624 | B1 attn share 0.117; I:436,469,511,549,574 | No exclusive ≥0.625 us/layer, or KEEP fails |
| 5 S | Vector-preserving XOR LDS layout for MoE-up activation reads | 0.02337 | 6 if reopened | 0.003895 | A6 0.075; `etx/ops/gemv.py:148,226` | Nominal fails 0.030; reopen only with ≥0.625 exposed us/layer |
| 6 X | Keep each wave's activation segments across its four MoE-up rows | 0.03447 | 15 | 0.002298 | A6 0.075, alternative to S; `gemv.py:188,228,253` | Extra spills/occupancy loss erase ≥0.625 us/layer, or KEEP fails |

**KEEP for every positive arm:** fresh same-build imported-launch checks against the unchanged original,
CHECK-DSL where applicable, 32/32 HF greedy, every interleaved wall pair faster, median ≥0.030 ms and
above twice spread (CERT:168). No GPU implementation or certificate is delivered by these CPU models.
R/A alter load issue order in an imported body; keep its original math, barriers and partition order,
retain the original reference kernel, and reject any byte difference. A new self-defined reference is invalid.

**Overlap subtraction.** R+A consume `0.06624+0.06624=0.13248` of B1's 0.236, leaving 0.10352,
not another 0.13248 outside B1. X contains S's priced LDS-read benefit: `0.02337+0.03447−0.02337
=0.03447`, not 0.05784. P includes the already-priced PP5 early-load share (≤0.034,
MTD:207,413): conservatively at most `0.07584−0.034=0.04184` beyond that mechanism, still in A3.
C5 attacks QK4/RT3, different roles; no router/qkv credit is assigned here. Existing A6 skeleton service
0.015–0.023 remains an alternative allocation of its 0.075 cap, not a new line in this sum.
T realizes 0.03312 of A7's 0.093, leaving 0.05988, without any second KEY32 credit.
The combined replay is **0.27591 ms**, all inside A3/A6/A7/B1; **new ledger ceiling added = 0**.
It implies `4.180−0.27591=3.90409 ms`, **256.1 tok/s [E]**, not 300. It does not change CERT's S1/S2 bars.

## 3. Body, serial-point, host and byte audit; per-lever falsification

**3a. Start with all nine big bodies.** These are critical-path bodies, not sums over concurrent tasks.
Body `3191.8 / 4067.3 = 78.47% [M intervals, E ratio]` (MTD:341). Operational floors below are
round11 probe/ISA **[E]**, not exact bounds; they include work beyond bytes (MTD:345-355).
Each body includes all-wave drain P9. Source anatomy: MTD:178-315; floors: `docs/FLOORS-2026-10-08.md:239-247`.

| Body [M] us/step | Work per useful byte | Byte service [E] us/step; operational floor us/layer | Remaining mechanism / ledger |
|---|---|---|---|
| moe_up 770.3 | bf16 gate+up weights → fp32 dot, bf16 gate/up, SiLU×up, bf16 output | 561.8; 14.67 | LDS x stride/reloads S/X; A6. No new partial-matrix credit |
| moe_down 435.4 | bf16 weights+activation → fp32 dot, routing multiply, bf16 output | 280.9; 9.48 | Body 9.069 already below floor: no independent gain; S model says EPL=4 conflict-free |
| qkv 433.2 | bf16 weights → VALU dot; replicated R=8 residual fold/RMS and ordered wave partials | 234.1; 7.55 | Bank/reuse possible but A4 only 0.073; C5 already covers norm weight trip. No extra qkv price |
| router 265.4 | 0.53 MB/layer weights; most body is fold/RMS and one-row dot/reduction | 5.9; 1.52 | A5 0.193; C1/C5 already priced. Empty tasks are 0.28 us, not a 256-way serial cut |
| o_proj 263.5 | bf16 weight dot plus 8 KB/task repeated x staging | 187.3; 5.30 | Reuse/swizzle prototype could be a passenger; rebased allowance 0.0085 < KEEP, no new row |
| attn 259.7 | K/V bytes → unchanged QK/PV MFMA, softmax and LDS exchanges | 24.1; 2.92 | A hides metadata/payload trips in B1; seven counted barriers remain |
| reduce 254.3 | 5 partitions' logits/sums/partial vectors → original exp rescaling and ordered FMA | 0.6; 2.81 | R hides independent loads in B1, retains slot joins |
| prep 194.8 | q/k bytes → RMS, RoPE, cache update; V is copied without arithmetic | ~0; 1.52 | P removes issue dependencies in A3. Already-vectorized q/k and BS division are not new levers |
| topk 146.3 | 256 B logits → exp, butterfly sum, eight key maxima, renorm | ~0; 1.21 | T hides a coherent load response in A7, retains KEY32/softmax |

**3b. Serial inventory.** Costs below name *all source-level serial classes* in a step, including repeats.
They are not disjoint stopwatch slices. Do not add a service estimate, hop model and body containing it.
There is no trustworthy intra-body trace; the hour timestamps are invalid (CERT:177-191).
The important C5 chain is **six serial points**, not saved LDS bytes:
fold loads → square/round → block-sum joins → cached norm load → scale/round/LDS join → weight stream
(MTD:185-192,261-265). C5 only moves point 4, still pays all six kinds of work.

| Serial point(s), repeats | Cost / basis | Remove or hide; named destination |
|---|---|---|
| Launch skew; 436 path tasks' descriptor fetch, queue join, readiness observation, acquire, ready barrier | 2.6 us window [M]; path gaps 875.49 us [M]. Fetch/setup shares unsplit; bare barrier 0.0612 us [E] | Placement can hide fetch, cannot erase it; B2/B3. Existing X7/TOUCH controls, zero new credit |
| Every task: writing-wave drain/join, arrive fan-in, publication/poll | DOMAIN/device one-way 0.642–0.951 us [M historical]; acquire 0.044–0.181 us [E]; share-32 or share-1 flush cost unsplit | Keep visibility and producer counts; last-arriver already shipped. GWS is rejected, not a new zero-cost edge |
| qkv/router six-point chain above (48 each); lm_head once; input norm once | qkv/router modeled hops 0.42/1.02 us/layer [E]; machinery 3.73/4.39 [R]. Block sum two joins, scale one | C1/C5 are old. A single last-arriver RMS publisher would serialize the fold; no new saving without independent consumer work |
| Dense/MoE activation stage join; sequential row dots, per-row wave tree/deposit; clear, gather, arena reuse joins | Dense 3 helper joins; MoE staging+clear+gather+reuse+publication; ≤0.106 us/join historical service [E]. Up 64 row flushes | X reuses x while retaining multiply/add order; S changes addresses. Deleting the joins is unsafe (ISA-FIXES:109-131) |
| MoE expert-id → W base; up SiLU/round; down routing weight → round/store | 1.02 us modeled hops per role [E]; retained up/down machinery 159.5/105.5 us [R] | Neither makes W prefetch routing-independent. S/X target M.N only, not the previously killed M.B–M.S skeleton |
| Prep q/k read → RMS tree/rsqrt → position → cos/sin → RoPE → slot → K stores; V slot → V read → 8 strided stores/lane | 1.02 us/layer hop model [E], 3.04 machinery [R]; 10/16 waves and 16/64 lanes active | P explicitly executes a different dependency DAG; request count unchanged. Wider V lane scatter may reduce issue depth but has no exclusive price |
| Attention seq_len → block table → Q LDS join/K reads; V table → V reads; QK → softmax (3 joins) → PV/output (2 joins) → slot-exit rounds | 2.71 modeled hops, 2.20 machinery us/layer [E/R]; 7+≥1 counted joins | A only overlaps independent reads; cannot delete softmax/PV or slot participation |
| Reduce seq_len → max loads/tree → sum loads/exp/tree → tmp_out → shared join → original FMA/divide → exit rounds | 2.71 modeled hops, 2.58 machinery us/layer [E/R]; 1+≥1 joins | R issues three arrays together, preserves both trees and partition order |
| Topk logits → round-0 max → two exp → six-stage butterfly sum/divide → seven further maxima/readlanes → ordered selected sum/renorm/store | 0.71 hops, 2.34 machinery us/layer [E/R]; no body join | T joins independent loads once; exact KEY32, softmax and weight order remain |
| Argmax loads → 10-halving LDS tree/11 joins → next token; prologue embedding/position store; cross-step retirement/reset/upload | Argmax 9.5 us, prologue 3.9, norm 3.7 [M] outside repeated bodies | Tail allowance < KEEP; host breakdown below. No resident-loop/epoch re-sale |

Sources for the inventory: MTD:163-176,185-192,203-212,219-241,249-281,293-332.
ISA static counts (`docs/isa-fixes-2026-10-11/baseline.txt:6-19`): up 3858 instructions/239 LDS/18
barriers, down 1873/103/9, prep 1423/20/1, topk 706/30/1; wrapper publication is included.
These contain alternative branches, not executed-cycle counts. Zero MFMA in GEMV is normal batch-1 dot
work; reducing index divisions is already below 9.6 us/step (ISA-FIXES:82-102). Wave count and live-register
pressure matter only through actual resource use/issue stalls, not static instruction savings.

**3c. Mechanism arithmetic and cheapest discriminators.** `L/simulate.py:36-57` executes the load DAGs,
not just a subtraction of body totals. Nominal memory response 0.69 us comes from the old loaded-HBM
probe (`docs/MOE-BODY-ANATOMY-2026-10-11.md:35`); it is **not** a current cached-metadata latency.
Math-node 0.1/0.2/0.4 us are explicit sensitivity assumptions, not measurements. The DAG assumes unlimited
issue capacity; instruction contention or compiler's existing overlap can reduce every cut to zero.

- **P:** old critical k-wave DAG 3.26 us → 1.68: `2×0.69+0.20=1.58`, `48×1.58=75.84 us`.
  Loads of immutable per-step position/slot can begin after input readiness; neither depends on RMS
  or RoPE output. Issue cos/sin once position arrives. Retain every conversion/store and the old fallback.
  RTT 0.50–0.73 gives 57.6–79.68 us [E]. Source `attnprep.py:114-151`; CPU DAG `L/simulate.py:36-59`.
  **Kill:** emitted ISA already early, or exposed critical-wave difference <0.625 us/layer, or another
  q-wave becomes the bottleneck and pairs miss KEEP. Cheapest GPU: 6 min/$0.20, sparse wave timestamps
  plus unchanged-load-order vs reordered body, CHECK-DSL/32 tokens; keep timing stamps off for pairs.
- **R:** after seq_len, array addresses/clamped partition indices are known. Max and exp-sum loads are
  independent; tmp_out can arrive while stats trees run. `2×0.69×48=66.24 us`, within reduce's 118.8 us.
  Keep max/exp/shuffle/FMA order exactly. I:1455-1539; CPU `L/simulate.py:47-52`. Other waves already load tmp_out
  outside warp-0 stats: the new target is **warp 0 delaying the shared join**, not all eight slots' time summed.
  **Kill:** no exposed ≥0.625 us/layer or extra live tmps spill; 8 min/$0.27 for constituent byte checks
  and slot-0 wait timestamps, then unchanged-barrier pairs. No new election or C6 boundary credit.
- **A:** seq_len determines both clamped K/V table addresses. Move V table issue before Q staging/K
  fetch; issue its payload as soon as its table returns. Old DAG 4.05 → 2.67 us: `2×0.69×48=66.24 us`.
  **V is already fetched before QK/softmax** (I:574,596); “prefetch V during softmax” is not this lever.
  Source I:429-574; `L/simulate.py:53-56`. **Kill:** compiler already overlaps K/V metadata/payload or register lifetimes
  increase spills enough to miss KEEP. 10 min/$0.33: original/import byte pairs, Q/K/V issue/return stamps,
  same slot barriers. Cache contention and mfma issue cannot be settled by this DAG.
- **S:** model actual 128-bit LDS read groups, not scalar lane stride. Up addresses `8×lane+quad×4`
  collide twice per bank; switch quad to `quad XOR ((lane>>4)&1)` in both writer and reader.
  The permutation keeps each four-float vector contiguous, storage size and every logical operand unchanged.
  `L/simulate.py:82-117` enumerates every address and bank group; down EPL=4 gives 1→1, up 2→1.
  [AMD's bank-group rule](https://rocm.docs.amd.com/projects/composable_kernel/en/7.13.0-preview/conceptual/ck_tile/hardware/lds_bank_conflicts.html)
  supports this address model, not its timing. Local b128 instructions are retained in L/b128-evidence.txt:4;
  that file is **not** timed-build ISA certification. Nominal LDS service `524288/(128×2100)=1.95048 us/task`
  (MOE-BODY:48); a conservative 50% service-cut sensitivity × half exposure = 0.48762 us/layer,
  23.406 us before replay, **23.37 net** after clipping/path changes. Priced full-exposure scenario
  46.812 us is not a physical ceiling: the nominal service excludes conflict stalls. HBM weights unchanged.
  **Kill:** nominal below KEEP; no GPU spend now. Reopen only on stamped/counter-confirmed bank stalls
  ≥0.625 us/layer and native b128-preserving lowering; then 6 min/$0.20 isolated layout A/B, all original sums.
- **X:** each up wave owns four full rows, all using the same 2048/64=32 activation floats/lane
  (`docs/MOE-BODY-ANATOMY-2026-10-11.md:44-46`). Keep them in registers and read LDS once instead of four
  times: `524288→131072 B/task`, **393216 B fewer LDS reads**, zero fewer HBM weights.
  Service cut `1.95048×3/4×1/2=0.73143 us/layer`, raw 35.109 us → **34.47 net** replay.
  The oracle compares every logical operand and its order for all 64 lanes, K=2048/768 (`L/simulate.py:82-103`).
  Extra live state up to 32 VGPR/lane is a real risk at the whole-kernel 128 limit; use partial segment
  caching if needed and reprice it. No “more occupancy” credit. **Kill:** resources/loaded bandwidth regress,
  or ≥0.030 wall not obtained. CPU resources first; then 15 min/$0.50, stage stamps, exact DSL gates and pairs.
- **T:** the original topk has `load_ushort → vmcnt(0) → load_ushort → vmcnt(0)` in both local LLVM23
  and native ROCm22 listings (`L/isa_order.txt:9-12`, `L/isa_order_rocm22.txt:10-13`, retained by d33a3b8).
  Its two published logits are independent. A two-request accessor can preserve sc0/sc1 system coherence,
  distinct result registers and existing completion wait, then convert/max in the original order.
  `2×0.69→0.69 us`, `48×0.69=33.12 us`; RTT 0.50 misses KEEP (24 us), 0.73 gives 35.04 us [E].
  The old 0.71-us hop model groups the logit access as one trip; static waits expose an extra *possible*
  chain inside its residual, not a measured contradiction or a larger A7 cap. CPU outstanding-request
  oracle checks both completion orders; no result is consumed before both return (`L/simulate.py:124`).
  **Kill:** effective extra exposed wait <0.625 us/layer, plain loads replace coherent ones, resource regression,
  or KEEP fails. Cheapest GPU: 4 min/$0.13, paired two-load microprobe under real body load, then topk DSL/
  token gates and pairs. MoE's two ids/weights loads share this issue pattern but down is below its floor;
  no extra down credit or no-fence experiment is included.

Also simulated exact integer sorting: bitonic 128-key/28-stage network agrees on 512 key rows, but even
optimistic equal 8-cycle stages vs KEY32's 48 DPP stages save `20×8/2100×48=3.657 us` [E], below KEEP;
24-cycle shuffled stages give −6.583 us. **Reject sorting at $0**, preserve all probability summation order.

**3d. Host: ten named profile quantities, no new independent ≥30-us round trip.**
The extracted HOUR mean fields are in `L/host-components.txt:4-13` [M, other session]: upload 3.990,
reset submit 38.634, residency 1.046, launch 6.040, wait 4922.802, retire/readback 18.867,
instrument/other 3.798, reset device 74.980, kernel device 4869.703, total wall 4995.178 **us**.
Wait/kernel/wall are inclusive, and reset device lies in queued wait: **never sum these ten**.
`wait−kernel=53.099`, of which 35.837 is kernel work outside token timers, leaving **17.261 us** of
host wait discrepancy. H's existing formula yields 72.6 us (CERT:56-64), not 72.6 of removable host work.
Readback includes event timing and 8-B control fetch (`etx/runtime/include/etx/host.h:358-381`);
diagnostic trace fetch is conditional. Launch/residency/upload/retire individually miss 30 us.
Packing output+control into one retirement transfer could remove one copy round trip, but has no split
price above its 18.867-us containing interval. Upload+retirement together cap 22.857 us <30.
Graph launch already exists for the baseline (`host.h:413-422`); moving wait off the CPU cannot erase
35.837 us of actual kernel execution. Residency caching is ≤1.046 us, epoch belongs to A1, and the
already-priced resident loop/device reset are not new mechanisms. **GPU cost $0 now**; reopen H only
on ≥30 us of exclusive attributable round trips after diagnostics are removed. Keep abort/exit safeguards.

**3e. Bytes and same-GPU prior mechanisms.** Achieved useful rates on this path are [E], not FETCH_SIZE:
up `2415.919 MB/770.3 us=3.136 TB/s`, down `1207.960/435.4=2.774`, qkv `1006.825/433.2=2.324`,
o_proj `805.306/263.5=3.056`. The 561.8/187.3 us are `bytes/4.3`, not measured copy durations
(`docs/master-table-2026-10-11/master.txt:8,11,14`). DATA:33-34,87-88 points to measured shape/rate data:
`etx/sim/phase.py:27-46` retains 3.55 TB/s deep HBM cap and shape floors; fleet's depth sweep reaches
4.33 TB/s (`~/code/fleet-mi300x/results/microbench_summary.txt:28-34`). Those are different sessions/
shapes, not a contradiction. Repricing at 3.55 gives up 680.54 and o_proj 226.85 us: **larger floors**,
not free recoveries. Required selected bf16 weights are read once (up 50.331648 MB/layer,
o_proj 16.777216); 48 layers exceed MALL capacity. No precision/cache-byte cut is booked for them.
Up's repeated x LDS reads *can* be avoided by X; its HBM-byte floor stays 561.8. Coherent folded activation
reloads (qkv 9.4 MB/layer, MTD:195-196) already reuse small memory, and a one-writer norm can lose on latency.

Read fleet STATUS and raw results across its v0.1–v0.18 history (sessions 1–9); relevant decisions:

| Same-MI300X measurement [M], source under `~/code/fleet-mi300x/` | Applies here / ledger |
|---|---|
| Occupancy 3→1 waves/SIMD: 38.8→38.8 us expert pair, `results/microbench_f_occupancy.txt:3,6` | No occupancy-only MoE lever; A6/down. X must preserve actual stream/resource performance |
| Cross-task prefetch 3.812–3.815 vs 3.673–3.682 ms, STATUS:30; cacheable weights 4.249 vs 3.597, :81 | Reject bigger/deeper weight prefetch; B2/B3/A6. P/R/A keep the original input visibility gate |
| Split up/down 18/19 workers: 4.150–4.784 ms; gate_up 44 vs 33 us, STATUS:34,77 | Narrow streaming pools lose; B2/B3 moves cannot claim byte removal; new CPU split sweep below also loses |
| Coherent activations 3.718–3.719 vs 3.709–3.715; uncached no-fence 12.3M/16.4M stale, STATUS:32,41 | No new UC saving; retain visibility. Current ETX coherent DSL path already chosen |
| Last-arriver release 38.4 vs per-worker 46.8 us pair, `results/microbench_summary.txt:39-40` | Already-shipped combining; B2/B3, not new money. One-writer folds instead 4 us vs parallel ~1, STATUS:67-69 |
| Idle q_c publishers with useful reader work 3.593–3.596 vs 3.645–3.654, STATUS:80; router topk publisher slower, :29 | Independent reader work decides publication. Qwen has no DeepSeek W_UK absorb; cannot transfer 0.055 ms |

Section 15/research loose ends were checked: GWS **hung** (`docs/ETX_Technical_Design.md:1028`), not untested;
relay/priority/backoff and LDS pages have no above-bar exclusive Qwen price (same doc:902,908,1112;
`docs/RESEARCH-sync-hardware.md:60-98`). Direct-to-LDS is dword-only, no extra AGPR capacity.
Triton MoE importing was once research-only (`docs/RESEARCH-moe-import.md:21-51`), but its retained cache
and measured AITER alternatives already exist (DATA:27; CERT:91-95): no renewed 20-minute capture or
precision change. Resource-class splitting attacks live spills only if line tables place reloads on a
critical body, not the bandwidth-bound union that fleet already falsified. CPU cost now, GPU credit zero.

## 4. CPU simulation actually run

Reproduction from the repository root; full logs/JSON, compressed simulated plan/trace and HTML are in L.
The script hashes the retained X7 inputs; 60,362 tasks, dependencies and producer counts remain identical.
Queue and ready residuals stay attached to their original tasks; initial takes stay fixed. Baseline reproduces
all **181,086** stamps exactly. Body changes come only from the explicit DAG/LDS hypotheses above;
this extends replay with intra-body mechanism models, it does not measure their exposure.

```text
.venv/bin/python docs/levers-more-2026-10-12/simulate.py
ORACLE exact index reuse/swizzle: K=2048,768; bitonic 512/512, stages=28; clamped-address positions=33
BANK b128: EPL8 2→1; EPL4 1→1
BASELINE tasks=60362 all_stamp_error_ticks=0 span_us=4067.290
SIM P 3991.450 gain=75.840 us; R 4001.050 gain=66.240; A 4001.050 gain=66.240
SIM T 4034.170 gain=33.120 us
SIM S 4043.920 gain=23.370 us; X 4032.820 gain=34.470
SIM combined span=3791.380 gain=275.910 us body=2915.300 gap=876.080
```

All outputs above are **simulated [E]**. Combined gap rises `876.080−875.490=0.590 us`; body falls
276.500, net 275.910. It does not delete 0.425 ms of B2/B3. Conservative body clipping retains tasks
already below the old operational floor unchanged; a phase floor is not asserted to bound each task.
Prefetch-distance sweep 0/1/2/4 means **within-body issue lookahead stages**, not speculative next-token
loads. At RTT 0.69, distance 1 cuts P/R/A by 1.58/0.69/1.38 us; distance 2 by 1.58/1.38/1.38;
distance 4 adds **zero**; T cuts 0.69 at every positive distance. At RTT 0.50–0.73, P cuts 1.20–1.66,
R/A 1.00–1.46 and T 0.50–0.73. These are DAG limits;
GPU issue/return stamps, spills and paired untraced wall measurements must confirm them.

Actual layout/worker sweeps, unchanged bodies, optimistic **zero** cost for DOMAIN→DEVICE promotion:

| Workers/XCD or split | Original role placement | Heads rotated +4 XCDs | Attention partitions across 8 XCDs |
|---|---:|---:|---:|
| 6 | +13762.710 us | +13736.530 | +13763.630 |
| 14 | +5507.730 | +5469.000 | +5504.640 |
| 38, coordinate-reset queues | +401.740 | +337.550 | +399.920 |
| up18/down20 disjoint pools | +1764.400 | — | — |
| up24/down14 disjoint pools | +1403.490 | — | — |

Positive means **slower [E]**; source `L/simulate.log:30-40`. The 38 case resets slot progression,
not the exact baseline queues. No oracle worker winner is re-priced; X7 remains its old conditional row.
This model cannot price new-placement cache contention or physical CU discovery; a positive scope premium
only worsens these bounds. Therefore no GPU layout spend is recommended from this sweep.

```text
.venv/bin/python docs/levers-more-2026-10-12/timeline_check.py
# Decompresses combined-plan.json.gz and combined.bin.gz to a temporary directory, then actually runs:
.venv/bin/python -m etx.tools.timeline <tmp>/plan.json <tmp>/trace.bin --bytes docs/xcd-timeline-2026-10-11/bytes.json --json docs/levers-more-2026-10-12/timeline-full.json
.venv/bin/python -m etx.tools.timeline <tmp>/plan.json <tmp>/trace.bin --bytes docs/xcd-timeline-2026-10-11/bytes.json --grids '_24$' --html docs/levers-more-2026-10-12/timeline-layer24.html
```

The full-step CLI result is **simulated**, span **3791.380 us**, zero unknown dependencies; layer-24 HTML
visualizes that same simulated schedule. Exact CLI stdout is retained, not a hand-rendered replacement.

```text
.venv/bin/python -m examples.dsv2lite.whatif --overlap /Users/ckl/code/etx/results_2026-09-27/vlnext/overlap.log --event-us 1.4
fleet layer 5: 129.3 us = latency-bound phases 73.2 us + expert stream 56.1 us
shared f=1.00 event 1.4 bg 2/4 TB/s: 126.5/122.3 us, 3.53/3.42 ms
interleave f=1.00/1.04: 146.4/152.3 us, 4.04/4.20 ms, +12/+17% per-request latency
```

These are **arithmetic what-ifs [E]**, not an instruction simulator or Qwen savings. Qwen has no always-on
shared experts; two-request/batch-2 results violate this task's batch-1 condition. Full real interleaving
later slowed latency phases 1.46× and streams 1.4× (design:1089), unlike pure fold microbenchmarks.
The method is useful; transferring its 5.1% as a new Qwen lever is invalid.

## 5. Cost and recommended next spend

**Spent now: 0 GPU minutes, $0; CPU only.** No provisioning, GPU/Docker GPU execution or push.
Initial oracle incorrectly required five partitions at seq_len=1024 (actually four); fixed and retained
`L/initial-failure.log`. First what-if used a nonexistent overlap path and silently fell back to estimates;
retained `L/dsv2-no-overlap.log`, then reran with the real log. Scalar-bank sensitivity is retained as
withdrawn `L/scalar-sensitivity.log`; ranking uses the vector-preserving b128 model only.

**Next GPU spend remains the deciding measurement, not all six prototypes.** Preserve CERT's 15-minute
fixed-stamp tranche A and its S1 kill bar; include §1's four-minute environment guard: **19 min,
`19/60×$1.99=$0.63 [E]`**. The four-minute guard is session validation, not a source lever.
Do not repeat the 22-minute vendor search just to price these new mechanisms. If S1 passes, CPU-prepare
P first, then R/T/A with original math and native resource checks; only ready prototypes join a later job.
These mechanisms are candidates for attainable intermediate improvements even if the target route stops;
that is a separate spend decision, not an excuse to override S1/S2.

Warm-pack marginal tests P/R/T/A/X: `6+8+4+10+15=43 min`, `$1.43 [E]`; S receives **zero** until reopened.
Final combined candidate needs the existing 10-minute five-round fused/same-kernel-graph/strongest-vLLM
validation, $0.33; **19+43+10=72 min/$2.39** optional all-survivor cap, plus explicitly budgeted cold
staging if needed. No balance has been checked because no session is authorised or provisioned here.
Before any future provisioning run all AGENTS §4 local/native/dry-run gates for actual prototypes; this
documentation/simulator task claims only the CPU checks below, not that nonexistent GPU variants build.
Stop any failed arm immediately; preserve full logs, retire the VM at the registered stop and verify deletion.

Final verification: `.venv/bin/python -m pytest -q` — **1453 passed, 14 skipped**; full output in `L/pytest.log`.
