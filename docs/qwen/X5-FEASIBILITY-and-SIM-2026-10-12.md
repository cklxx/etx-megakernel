# X5 feasibility and conditional simulation — 2026-10-12

CPU only, starting tree `efd008c`; no GPU, provisioning, vendor sweep or push. Condition: one MI300X VF, Qwen3-30B-A3B, bf16
Q/K/V/output, batch 1, NH=32, NKV=4, HD=128, causal context 1024. [M] retained measurement or CPU observation; [E] conditional
arithmetic/engineering estimate; [U] unmeasured. **No new alternative is READY. B1 stays open at 0.236 ms; spend GPU minutes on G
first.** The only executable X5 build remains `kept`, a control. The existing inventory is reused, not exhaustively enumerated
again.

Source notation: `V/` = `/Users/ckl/code/vllm/`, revision `f46a9dfe2c5f57bebbd29556cbbb25eabd874226` (registered at
`docs/qwen/VERIFY-PLAN-2026-10-12.md:150`). `A` = cached `/private/tmp/qwen300-attn.py`, the AITER unified kernel blob: SHA256
`316615adc9558ce52abab9a8f066a744ba39f68cf27237ff88953414391ad3ea`; its revision/path/hash are retained in
`docs/evidence/verify-x5-2026-10-12.json:316`. This is not evidence of the pinned GPU image's dispatcher or complete AITER source.

## 1. Feasibility probe: representation and contracts, before performance

READY means today's exact adapter/operator/host path accepts the family; BUILDABLE means the following engineering work is required,
not that the GPU numerical/resource gates already pass. Effort figures are single engineer-day estimates [E], including adapter
checks, excluding paid GPU time. EXCLUDED is scoped to the named representation/revision, not every historical implementation of the
algorithm. No exclusion below is based on speed alone.

**Shared acceptance P.** Pin the immutable original and compiled import; verify all planned imported launches byte-for-byte at
positions 0/1040 and 1023; CHECK-DSL and all 32 HF greedy tokens must pass. Then measure complete warmed original/imported
whole-grid samples at pos=1023 (actual KV length 1024), three alternating batches of 100 replays per stage per layer, stamps/profile
OFF. Sum stage medians, average 48 layers; require <=7.69 us/layer and changed attention identity. Missing source/ABI coverage,
unsupported gfx942/bf16/G8, failed bytes/tokens, incomplete samples, timeout or >7.69 fails immediately. References:
`examples/vllm_llm/host.hip:684`, `examples/vllm_llm/x5.py:103`, `examples/vllm_llm/x5.py:153`; certificates enumerate the actual
imports, not a fixed 97 count (`examples/vllm_llm/import_certificate.py:99`).

- **Native block=16 control: READY.** Existing G8 specialization and reduce at `etx/importer/spec.py:53,82`; BS/PART/grid/layout at
  `etx/importer/spec.py:274`. Probe: P refreshes its isolated baseline; it never earns R2, even below the bar or under a renamed
  export (`examples/vllm_llm/x5.py:153`).
- **Native ROCM_ATTN block=32: BUILDABLE, 1 engineer-day [E].** Vendor dispatch supports block32/HD128/bf16
  (`V/csrc/rocm/attention.cu:3612,3625,3678`), while the imported specialization/operator fixes BS16
  (`etx/importer/spec.py:54,277`). Change `etx/importer/spec.py` G8 export/mangled match/Operator BS to 32;
  `examples/vllm_llm/model.py:73` must select that operator, not derive BS from G4. Change `examples/vllm_llm/host.hip:270` table
  sizing/cache strides together; pass BS32 into the existing parameterized writer (`etx/ops/attnprep.py:99`,
  `examples/vllm_llm/host.hip:400`). Identity slot_mapping=pos stays valid (`examples/vllm_llm/vl_tiles.hip:36`); repack all initial
  KV as BS32 too. PART remains 256, reduce unchanged; rebuild/seal both original and import. Probe: P on the BS32 attention+reduce
  pair; any BS16 writer/table/stride paired with the BS32 reader, stale cross-XCD KV, or partial context is immediate failure.
- **Generic paged attention v1 (no reduce): EXCLUDED from installed revision.** Registered missing source at
  `docs/evidence/verify-x5-2026-10-12.json:175`. CPU `git -C V ls-tree -r --name-only HEAD` finds no paged_attention_v1 source; `rg
  paged_attention_v1 V/vllm/_custom_ops.py V/csrc` finds no entry either. The ROCm source list instead contains native attention
  (`V/CMakeLists.txt:1420`). Probe: none, 0 GPU minutes; a missing original fails before allocation. Restoring a separately pinned
  older vendor implementation would require a new registration.
- **Generic paged attention v2 (partitioned+reduce): EXCLUDED from installed revision.** Same CPU source/API searches for v2;
  registered absence is `docs/evidence/verify-x5-2026-10-12.json:178`, ROCm list `V/CMakeLists.txt:1420`. Probe: none, 0 GPU
  minutes; no original main/reduce export exists to certify. These two verdicts correct inventory applicability, not bf16/GQA
  algorithm support elsewhere.
- **vLLM TRITON_ATTN 2D: BUILDABLE, 2 engineer-days [E].** Source has the unsegmented branch/grid
  (`V/vllm/v1/attention/ops/triton_unified_attention.py:1041,1078`). Add pinned LLVM/hsaco/meta exports in `etx/importer/spec.py`
  TRITON/OPERATORS; existing import/original machinery is `etx/importer/triton_import.py:128` and `etx/importer/vllm_kernels.py:65`.
  Change `etx/ops/attnprep.py` KV stores and `examples/vllm_llm/host.hip` allocation/ABI to flash [block,token,head,HD],
  cu_seqlens_q=[0,1], seqused_k=pos+1, matching strides and output. Change `examples/vllm_llm/model.py:346` to prep→attn→o_proj with
  no native reduce; replace its reduce/head-affinity assumptions at :584. Apply one-stage contract S below. Probe: P on the single
  actual output kernel (total layer time, not a fake zero reduce); native packed KV, unexpected scratch, bad partial masks or old
  two-role schema fails immediately.
- **vLLM TRITON_ATTN 3D: BUILDABLE, 3 engineer-days [E].** Grid and separate reduce are source-visible at
  `V/vllm/v1/attention/ops/triton_unified_attention.py:1081,1169`. Same export/writer/host files as 2D; add segment
  output/max/expsum allocations, both launch ABIs and prep→main→segment-reduce→o_proj in `model.py`. Segment count is a pinned
  specialization; do not reuse native PART256/bf16 partials. BLOCK_M=16 and BLOCK_Q=2 follow G8
  (`V/vllm/v1/attention/ops/triton_unified_attention.py:932`). Probe: P on main+reduce, recording actual dispatched grid/segments;
  wrong partial precision/stride, uninitialized tail segment or importer resource refusal fails immediately.
- **AITER unified tuples: BUILDABLE, 4 engineer-days [E] for the first tuple.** This includes 2D/3D, unshuffled/shuffled, TILE_SIZE,
  BLOCK_M/Q, segment count, warps/stages and K_WIDTH; additional tuples reuse the adapter, not a predicted price.
  Constexprs/branches are at `A:107,128,132,134,604,623,626,628,670` and retained in `docs/evidence/verify-x5-2026-10-12.json:323`.
  Pin the missing wrapper `aiter/ops/triton/attention/unified_attention.py` plus its configs/dispatcher, and the kernel path
  recorded in the evidence JSON. Add their exact exports to `etx/importer/spec.py`; use `triton_import.py` and `vllm_kernels.py` as
  above. Change `etx/ops/attnprep.py`, `examples/vllm_llm/host.hip`, `model.py` and S for the dispatched stage count, metadata, fp32
  softmax state and cache layout. Unshuffled cache is flash/NHD, not native packed KV; the installed unified backend forces its
  writer's shuffle OFF (`V/vllm/v1/attention/backends/rocm_aiter_unified_attn.py:333,355`). Shuffled requires a matching writer: its
  block-table access uses tile j directly (`A:387`), K/V reshape divides by K_WIDTH (`A:433,454`). Freeze a supported tuple with
  consistent tile/block mapping and nonzero dividing K_WIDTH; BLOCK_Q must match BLOCK_M/G=BLOCK_M/8, segments cover every KV token.
  Tile/warp/stage values are coupled compiler/dispatcher choices, not independent public knobs. Probe: P per frozen tuple and
  layout, including all reductions; dispatch fallback to retained/same identity, mismatched writer, unsupported tuple, nonzero
  unsupported scratch or byte/token failure rejects immediately.
- **AITER paged_attention_common HIP: BUILDABLE, 4 engineer-days [E].** Wrapper supplies scales/strides/tmp/max/exp/output
  (`V/vllm/_aiter_ops.py:3035`). Backend reshapes K and V differently (`V/vllm/v1/attention/backends/rocm_aiter_fa.py:1300,1318`).
  Pin missing `aiter/ops/attention.py`, `csrc/py_itfs_ck/attention_kernels.cu` and `csrc/include/attention_common.cuh` (paths in
  cached AITER tree); add actual HIP device main/reduce exports/Operators in `etx/importer/spec.py`. Adapt `etx/ops/attnprep.py`,
  host buffers/scales/ABIs and model stage/grids; reuse HIP slicing/import/original generation
  (`etx/importer/vllm_kernels.py:46,110`). Probe: P for an explicitly selected bf16 HIP path, all dispatched stages; silent
  ASM/quantized fallback, absent callable body or wrong scale/partial layout fails immediately.
- **AITER paged_attention_common ASM / pa_fwd_asm: EXCLUDED as direct imports.** Named assembly entry at `V/vllm/_aiter_ops.py:3000`
  is a host dispatch, not a callable LLVM device body. The registered importer requires an `amdgpu_kernel` IR definition
  (`etx/importer/ir_import.py:216`) and emits a device function call (`etx/importer/adapter.py:59`); loading a standalone ASM code
  object through `host.hip:690` supplies only an original/reference. An external launch is not an ETX megakernel tile. This
  exclusion concerns standalone ASM, not inline ISA already inside supported HIP/LLVM bodies. A future source-backed callable
  transplant needs a separately registered importer; the absent AITER implementation does not establish that such a transplant
  exists. Probe: none today, 0 GPU minutes; loader-only integration fails the fusion contract immediately.
- **AITER flash_attn_varlen_func, bf16 split-K: BUILDABLE, 5 engineer-days [E].** Varlen metadata/causal arguments exist
  (`V/vllm/_aiter_ops.py:2923,2952`); this wrapper has no public num_splits argument. Pin `aiter/ops/mha.py`,
  `csrc/cpp_itfs/mha_fwd_split.cu`, `csrc/include/mha_fwd.h` and the dispatched HIP/CK device bodies before selecting internal
  split-K (cached tree paths only). Add main/combine exports/Operators in `etx/importer/spec.py`; adapt host dense/flash KV,
  cumulative lengths and split partial ABI, KV writer and model stage/dependency list, applying S when the original has one stage.
  Any gather/repack needed every decode step belongs in the measured candidate cost. Probe: P on the actual dispatched
  main+combine+required layout work; invented split knob, ASM-only body, omitted gather cost or changed precision/causality fails
  immediately.

**S: one/multiple-stage contracts.** `x5.attention_signature` currently requires 48×{attn,reduce} (`examples/vllm_llm/x5.py:98`),
and `prices` requires 288 positive-duration records (:105). BUILDABLE single-stage candidates therefore need a sealed per-layer
stage manifest in `x5.py` and a matching producer in `examples/vllm_llm/host.hip:684`: identity and complete samples for every real
stage, 144 records for 48×one-stage×three samples, no dummy reduce or relaxed missing-record gate. Keep the control/identical-alias
guard (:153) and baseline refresh (:160). Multi-stage forms need the same manifest if roles differ. Host buffer snapshot/restore at
`host.hip:540` must cover new metadata/partials; retain uncached cross-XCD scratch or correct DEVICE visibility, audit KV placement
(`examples/vllm_llm/check_uc_plan.py:4`). No such adapter is implemented here; changing feasibility labels grants no build credit
and does not close the inventory.
All adapters also need explicit sealed export selection: `etx/select/variants.py:306` / `model.py:307`; defaults stay fixed.
`etx/importer/spec.py:411` must derive BS/PART/cache metadata from that selection, rather than G4; its host recipes at :441
must dispatch ABI by export, not overwrite a global role recipe. `examples/vllm_llm/host.hip:210` must validate each backend's
coverage rule instead of applying native PART256 to everything. Update `examples/vllm_llm/plan_exports.py:54` slot rules
together with model affinity and the selected grid. These are shared BUILDABLE changes, including block32's generated BS32 header.

## 2. Simulation probe: conditional value, not a kernel performance model

**Exact registered formula**, `examples/vllm_llm/x5.py:120`: `R2(t)=max(0,48*(10.645-t)/1000)` ms for a distinct certified candidate
with `0<t<=7.69`; otherwise **0**. t is the mean of per-layer stage-median sums (:113), us/layer. Historical 10.645 is a
critical-body control [E], not a fresh isolated whole-grid measurement; a future measured kept control refreshes it. Retained AITER
11.4203031 is a 31-growing-context dispatch mean, not exact 1024 import pricing (`docs/qwen/VERIFY-PLAN-2026-10-12.md:162,209,231`). All
R2 below is [E].

Positive hypothetical time sweep (not a claim these times are attainable):
`t=5.728333,7.00,7.50,7.68,7.69,7.70,8.00,10.02,10.645,11.4203` gives `R2=0.236,0.17496,0.15096,0.14232,0.14184,0,0,0,0,0` ms
respectively. For t>7.69, even a raw positive saving is rejected; no interpolation across this gate.

Each non-EXCLUDED alternative has the same registered response curve; layout, stage count and resource costs change its unknown
achieved t, not this formula. Alternatives replace the same B1 row; do not sum their R2 values. In every row below, R2 triple is at t=7.69 / 10.645 / 11.4203, in ms [E].

| Candidate | R2 triple | KEEP-qualified t, us/layer | Full B1 t, us/layer | Wall scenario |
|---|---|---:|---:|---|
| Native block16 control | 0 / 0 / 0 | unavailable: control | unavailable: control | unchanged 4.180 ms / 239.23 tok/s |
| Native block32 | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| TRITON_ATTN 2D | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| TRITON_ATTN 3D | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| AITER unified 2D, unshuffled | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| AITER unified 3D, unshuffled | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| AITER unified 2D, shuffled | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| AITER unified 3D, shuffled | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| AITER PA common HIP | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |
| AITER FA internal split-K | .14184 / 0 / 0 | 7.690 | 5.728333 | K below |

Solve raw savings for KEEP: `10.645-1000*.030/48=10.020 us/layer`. But R2(10.020)=0 under the registered gate: the largest
qualifying time is **7.690**, already pricing .14184; there is no t with registered R2 exactly .030. Full B1:
`10.645-1000*.236/48=5.728333... us/layer` (positive and below bar). The registered ceiling is rounded: the old 2.92+2.81=5.73 floor
gives `48*(10.645-5.73)/1000=.23592≈.236`; this is not a universal new-kernel floor. Both are conditional, not guarantees of wall
KEEP. The control itself has R2=0 at all times; it cannot resolve B1 by getting a faster control measurement.

**Timeline CPU check.** Ran `python -m etx.tools.timeline` on retained `~/code/etx/results_vlbest2_2026-10-10/b_all/plan.json` plus
`results_qwen300_2026-10-11/tr_on.bin`, with `--bytes docs/xcd-timeline-2026-10-11/bytes.json --json /tmp/etx-x5feas-timeline.json`.
Plan SHA256 `b32c3299d4165ba8d68c7b04461d9ed050818d397c7d99082928922836f9f315`; trace SHA256
`aa68468246acde1a8d0e19ad0fbc6d5913bc37c7a3f352f44cecec8c0bdef320`. 60362/60362 rows, zero type/coord/worker mismatches (validation
algorithm: `docs/review-pool-analysis-2026-10-11/rebase.py:40`). Critical path: 436 tasks, 3191.80 body +875.49 dep gap +0 queue gap
=4067.29 us [M CPU reanalysis]. 48 attn bodies total 259.72 us, 48 reduce bodies 254.28 us; all-layer sum/48 =10.708333, not the
registered interior-layer 10.645 (`docs/review-pool-analysis-2026-10-11/rebase.txt:7,8`). The trace supports one serial pair per
layer, not a new candidate's occupancy, layout/writer cost, partial numerics or event/body-to-wall conversion [U]. No new schedule
is synthesized from unknown kernel costs. The trace is one traced last step, not exact 1024 X5; measured trace overhead is about .022
ms (`docs/qwen/MASTER-TABLE-2026-10-11.md:107`). Idle XCD capacity is not removable time.

**K: session effect, applies individually to every row above [E].** Anchor 4.180 ms wall [M], target 3.3333, exact gap .8467 (rounded
registered .847), `docs/qwen/MASTER-TABLE-2026-10-11.md:104,110`; assume no extra cost, no overlap, and validated wall recovery b from B1
and g from G: `step=4.180-b-g`, `tok/s=1000/step`, `remaining=.8467-b-g` (or .847-b-g for rounded ledger).

| Conditional recovery b / g, ms | Step ms [E] | tok/s [E] | Remaining exact / rounded ms [E] |
|---|---:|---:|---|
| KEEP .030 / 0 | 4.150 | 240.96 | .81670 / .81700 |
| KEEP .030 / G threshold .088 | 4.062 | 246.18 | .72870 / .72900 |
| KEEP .030 / G pool .425 | 3.725 | 268.46 | .39170 / .39200 |
| R2 at bar .14184 / 0 | 4.03816 | 247.64 | .70486 / .70516 |
| R2 at bar .14184 / .088 | 3.95016 | 253.15 | .61686 / .61716 |
| R2 at bar .14184 / .425 | 3.61316 | 276.77 | .27986 / .28016 |
| Full B1 .236 / .088 | 3.856 | 259.34 | .52270 / .52300 |
| Full B1 .236 / .425 | 3.519 | 284.17 | .18570 / .18600 |

First row arithmetic: `4.180-.030=4.150`, `1000/4.150=240.96`, `.847-.030=.817`. Adding .088 or .425 subtracts it once from step and
gap; it does not turn G's stamped opportunity into a demonstrated saving. G threshold/pool:
`docs/qwen/PLAN-300TOKS-2026-10-11.md:302,176`. G at threshold is 2.93× B1's KEEP cut; its pool is 14.17× KEEP and 1.80× full B1. At
.088, G is smaller than B1's .14184 bar-price or .236 ceiling. **Run G's registered 15-minute feasibility/attribution gate first**:
G<.088 stops the route even granting full B1 (`.088+.236=.324`). With B1 only at KEEP, G=.088 gives .118<.324; at X5 bar it gives
.22984<.324; H is [U], zero. Full G+.236 still misses 300 tok/s; no target is claimed.

**Falsification.** For each surviving candidate, complete isolated samples >7.69 disprove qualification; bytes/tokens/ABI failures
disprove that tuple's eligibility. After qualification, ordinary adjacent full-model wall pairs recovering less than the assumed b,
or a combined B1+G gain different from b+g beyond measured pair spread, falsify K's transfer/additivity assumption. KEEP requires
every pair faster, median>=.030 and >2×spread, plus identical tokens/import gates; a new trace with changed bottleneck falsifies
retained-path transfer. Compare both same-kernel HIP graph and strongest legal vLLM in the same session; this simulation predicts
neither new fusion certification nor new vLLM margin (`AGENTS.md:20`, `docs/qwen/VERIFY-PLAN-2026-10-12.md:8`).

## 3. What stays unknown, and which probe settles it

- [U] Exact pinned-image AITER tuple universe, HIP/FA device exports, bf16 dispatch, split-K policy and source/image match: pin full
  source/configs and immutable original IR/hsaco/meta first; CPU work, **0 GPU minutes**. Files to settle it are the AITER
  wrapper/device paths named in §1 and staged `imp/imports.json` / `build_identity.json` (`examples/vllm_llm/import_certificate.py:99`). Current ASM
  exclusion does not establish absence of some future IR-backed export.
- [U] Each BUILDABLE tuple's importer resources/scratch, layout coverage, byte identity and 32 tokens: compile/stage
  locally/native-compiler first, then fresh certificates + P; **1.5 min correctness +.5 min pricing per frozen tuple [E]**, cold
  load may exhaust it (NOT-COMPLETE). Pricing follows the registered 30s/arm, 120s total slot (`docs/qwen/VERIFY-PLAN-2026-10-12.md:219`); at
  most control+three staged alternatives, not every unknown tuple in one slot.
- [U] Actual t, comparable control and warmed-replay representativeness (a refreshed control different from 10.645 rebases this curve): `examples/vllm_llm/host.hip:706` X5 samples and fresh control, same .5-minute
  pricing probe; no GPU estimate of t is made. Missing one-stage manifest needs CPU changes before any GPU probe; it is not a reason
  to omit a missing reduce record.
- [U] B1 wall transfer, new writer/gather/wrapper/visibility overhead, changed critical path, trace perturbation and B1+G overlap:
  **separate simulation validation**, three adjacent ordinary control/candidate 32-token pairs, **2 min per staged candidate [E]**
  after correctness, plus a same-build trace within that allowance if time remains. `examples/sessions/qwen300_hour.py:766,733`
  implements the pair rule; full per-step logs, candidate `plan.json` and `ETX_TRACE_RAW` settle the path. Pair budget is a proposed
  reservation, not an expansion of the registered B tranche.
- [U] Recoverable G atoms vs structural gap time, stamp validity and interaction: existing **15-minute A/S1 probe [E budget]**,
  output `A_edges.json` from `examples/sessions/qwen300_hour.py:664`, with valid inbody timestamps and perturbation checks. The
  old 875.49 us gap is not G=.425. Combine validated implementations and re-pair before adding their gains.
- [U] New same-kernel graph/fused and strongest-vLLM gains, and repeated attainment of 300 tok/s: five full interleaved
  graph/fused/reference rounds with all correctness gates, **5 min [E]** reserved in D, not P. Fresh raw logs plus
  `pairs.py`/`vlbest_report.py` settle them. If load/check/trace time exceeds any proposed allowance, report NOT-COMPLETE, never a
  shorter population.

CPU verification: `.venv/bin/python -m pytest -q` — **1490 passed, 14 skipped in 181.92s**.
`.venv/bin/python -m examples.sessions.qwen300_hour --dry-run` — **PASS**; X5 CPU inventory — **PASS**, `complete=false`, `survivor_builds=["kept"]`.
Only this document and candidate feasibility labels/reasons in `x5.py` change. Control/alias guards, bar, runnable builds and
completeness guards remain intact.
