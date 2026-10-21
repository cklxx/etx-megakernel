# MoE body anatomy — 2026-10-11

**Finding:** the redundant partial matrix/gather is removable in source for both current shapes, but no qualified data prices its exclusive elapsed saving. The current evidence cannot certify a 0.3–0.5 ms pool. The report reconciles the measured bodies explicitly instead of assigning an ablation residual to unmeasured phases.

Scope: Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X (gfx942), ROCm 7.14.0. CPU-only source and evidence audit; no GPU run or provisioning. Work is on `wf/moeanat`, based on **`a4ca716`**. Unless a commit is named explicitly, tracked source/file:line references use that worktree snapshot. This matters because `main` advanced during the audit; the version update is recorded in §3.2.

Evidence convention: `[M]` is a measurement with a source location; `[E]` is an estimate with arithmetic; `[U]` is unknown and contributes nothing to a numeric total. Instruction counts are not phase latency measurements. In source references below, `G` is `etx/ops/gemv.py`, `MODEL` is `docs/qwen/MODEL-vllm-kernels.md`, and `PLAN-300TOKS` is `docs/qwen/PLAN-300TOKS-2026-10-11.md`. Overlapping costs must not be added as if serial.

## 1. Evidence, identity, and reconciliation

### 1.1 Which measurement is being anatomised

All external `results_*` paths below are relative to `/Users/ckl/code/etx`, not this worktree. The supplied numbers belong to **`results_qwen300_2026-10-11/attempt1/tr_off.log`**, not the current top-level file. The archive reports span **4295.4 us [M]** (`:26`), **32/32 [M]** (`:25`), device median **4.297 ms**, wall **4.474 ms**, **233 tok/s [M]** (`:464-466`). The four flags are recorded at `results_qwen300_2026-10-11/certificate_off/build_identity.json:23`. Its DSL-source hash at `:6` matches the available `build/qwen300-rocm-off/dsl_tiles.hip` (checked locally); the local full-model source/plan hashes do **not** match the certificate, so that local megakernel assembly is not treated as the timed ISA. Hardware is MI300X VF, gfx942, ROCm 7.14 (`results_qwen300_2026-10-11/record.log:7,15,18`). The current top-level log reports **4.299/4.471 ms [M]** (`:464-466`); do not silently substitute it. Both logs bracket the same build-identity hash (`:3,467`), but are different observations.

Summing the printed mean-body column of the archived trace gives **761.50 us up**, **445.04 us down**, and **3086.96 us for the nine layer roles [E]** (recomputed, with every input line and hashes, in [trace-sums.json](../moe-body-anatomy-2026-10-11/trace-sums.json)). Therefore the exact rounded-input targets here are `761.50/48 = 15.864583 us up`, `445.04/48 = 9.271667 us down`, total `1206.54/48 = 25.13625 us/layer [E]`. The brief's **15.86/9.27** are rounded descriptions. **1067 us is not this trace's MoE body sum**: `761.50+445.04 = 1206.54 us [E]`. Its provenance is [U].

The body is an **all-task mean for a layer**, not a measured particular task or the last task's critical body. Tasks run concurrently: do not multiply by the task count. Body timing includes `etx_call_body`, **all-wave store drain and its final barrier**, between `tr[1]` and `tr[2]` (`etx/codegen/kernel.py:314,323-333`). Event arrival, next-task fetch and dependency wait lie outside it. Source-tile end is thus not trace-body end.

`3086.96/4295.4 = 71.867% [E]` is a sum of mean layer bodies divided by the span, **not a disjoint critical-path partition**. It excludes prologue, input norm, lm_head and argmax (`attempt1/tr_off.log:28-30,462-463`). The complement is not simply gap/wait. A critical-path partition needs the raw task trace and matching plan.

### 1.2 The two older analyses cannot be transplanted

The XCD report explicitly uses `results_tiles/tr_all.bin`, a **4412.1 us [M]** span, with **1785.1 us machinery and 1551.3 us body residual [E]** (`docs/qwen/XCD-TIMELINE-2026-10-11.md:12,19,88-102`). Its “machinery” is measured body **minus a byte/hop model**, not intra-body measurement. Neither those totals nor their split is a phase price for this trace.

The ladder is older again: **18.37/11.11 us [M]** controls versus **11.55/7.64 us [M]** NOVMEM (`etx/ops/gemv.py:17-29`; raw finding `results_2026-10-01/round37/FINDING.md:3-17`). NOVMEM **still loads one weight vector per wave**, reusing it; it is not literally zero weight reads (`gemv.py:237-245`). It retains dot, activation LDS reads and loop arithmetic, but changes dependencies and compiler scheduling and gives wrong output. Its marginal differences cannot be summed as exclusive phases.

For example the old up residual is `18.37 - 1.50 - (18.37-11.55) - (18.37-15.45) - (18.37-17.81) = 6.57 us [E]`; down gives `11.11 - 1.43 - 3.47 - 2.47 - 0.43 = 3.31 us [E]`. These are **subtraction residuals of separate experiments**, not measured loop slices. In particular **6.5 us each for up and down is unsupported**. `[U]` exclusive current-loop time contributes no saving.

### 1.3 Parameters and ceiling before a new experiment

One up task owns **32 gate plus 32 up rows**, K=2048; one down task owns **64 rows**, K=768 (`examples/vllm_llm/model.py:358-360`). bf16 weight bytes are `64×2048×2 = 262144 B up` and `64×768×2 = 98304 B down [E]`. With **192/256 tasks [M]** (`attempt1/tr_off.log:37-38`), the phase bytes are `50.331648/25.165824 MB [E]`. At the repository's **4.3 TB/s achievable-rate convention** (`AGENTS.md`, §2), the bandwidth service floors are **11.705034/5.852517 us/layer [E]**. They are phase-wide floors, not each CU's bandwidth, and overlap instruction work.

The corresponding residuals are `15.864583-11.705034=4.159549 us up`, `9.271667-5.852517=3.419150 us down`, `48×7.578699/1000 = 0.363778 ms/step [E]`. **This is diagnostic headroom above an ideal rate, not a removable pool.** It includes essential dots, staging, wave sum, serial latency, rounding, epilogues and publication. Removing **0.324 ms** requires `324/48=6.75 us/layer [E]`, about `324/363.778=89.1% [E]` of that ideal-rate headroom. The more exact plan gap is **0.323667 ms [E]**, `6.743 us/layer` (`docs/qwen/PLAN-300TOKS-2026-10-11.md:136,146`).

Measured older hardware parameters: scalar hit **0.113 us**, loaded vector L2/MALL/HBM **0.73/0.71/0.69 us**, imported-round barrier **0.106 us** (`docs/qwen/MODEL-vllm-kernels.md:320-329`). These are other-session microbenchmarks, not measured current phase times. `bench/hw/hw_bench.hip:16-18,23-28` describes the primitive/clock/latency tests. No qualified current shader-clock, SALU issue rate, LDS latency, or exp/reciprocal throughput measurement was found [U]. Nominal **2.1 GHz** is a reported device property (`results_2026-10-01/round21/streamw.log:1`), not a loaded-clock measurement. Cycle-to-us prices using it must be `[E]`.

The shipped build's full-model allocation is **128 VGPR**, **44 B/lane scratch**, **10 VGPR and 265 SGPR spills**, **29736 B LDS [M compiler metadata]** (`results_qwen300_2026-10-11/off_hipcc.log:4-12`). Spill counts are static allocation metadata, not dynamic MoE reload counts; their MoE latency share is [U]. An isolated no-spill compile cannot explain away full-model spills.


## 2. Execution-order anatomy and phase ledger

### 2.1 Mapping and units

The path is one **combined GU1 pass** for up, not two streams in succession. Up: `SEG=64×8=512`, `spr=2048/512=4`, `nrt=64`, `S=256`, `per=256/16=16`, `sb=16×wave`, `se=sb+16 [E]` (`gemv.py:174-184,773-777`). Waves before the gate/up split read the gate base; waves after it add the gap `(768-32)×2048` bf16 elements (`:188-196,777`). Each wave consumes **four complete rows [E]**. Its default pipeline has **U=2**, **eight trips**, each issuing two **16 B/lane** loads, `16 waves×64 lanes×2×16 = 32768 B` in flight/task [E] (`:240-248,354-359`). This is half the old full-depth pipeline; do not use the comment's old 64 KB as today's depth.

Down: `SEG=64×4=256`, `spr=768/256=3`, `nrt=64`, `S=192`, `per=192/16=12`, `sb=12×wave`, `se=sb+12 [E]`. Every wave again owns **four whole rows**, with **U=4**, **three trips**, **8 B/lane** loads and `16×64×4×8 = 32768 B` in flight/task [E] (`:174-184,345-370`). **Whole-row ownership is true for down too.** The shipped isolated specialization only proves/enables up (`bf16_300/source.py:62-91`); its lack of a down implementation is not a mathematical ownership failure. For these full task shapes the pipeline tail executes **zero vectors [E]** because `16%2=0`, `12%4=0`. Runtime guards still exist.

In the following table, counts are **per task**, unless explicitly “per wave” or “per lane.” Instruction families come from the emitted gfx942 lowering and the CPU assembly audit described below. `BW` is a phase-wide byte-service estimate, **not an observed phase contribution**. `T_LDS(B)=B/(128×2100) us [E]` is a nominal no-conflict service model using the repository's **128 B/clock/CU assumption** (`gemv.py:49-50`) and nominal **2.1 GHz** (`round21/streamw.log:1`). It is not a measured LDS latency or a bank/issue-contention model. A symbolic instruction price `N/(I×f) us [E]` leaves issue rate `I` and loaded clock `f` [U]; it cannot contribute numeric saving. **Every individual phase's exclusive share of the observed body is [U]**: there are no qualified intra-body timestamps for this run. The table deliberately does not turn service costs into fabricated elapsed slices.

Classes are mutually exclusive: **I** = work tied to the required bytes and original bf16 numerical computation; **II** = implementation machinery a better mapping can shorten/delete; **III** = machinery already specialized by the named shipped flag. Class III describes the already-applied transformation; it grants no fresh saving. An essential calculation with no material transfer (SiLU, for example) is class I because the original numerical output requires it; the customer's three classes otherwise have no category for essential ALU work.

### 2.2 Execution-order phase ledger

`G` below means `etx/ops/gemv.py`. “up/down share” is the **exclusive observed us/layer**, not the operation service price.

| Order / phase | Source | Instruction/memory cost, up / down | Class | up/down share |
|---|---|---|---|---|
| A. Argument/grid reads and outer valid-block guard | G:853-860,465-479 | Scalar `s_load_dword[xN]`, waits and compare/branch; argument block is 160 B (G:80). Independent fields overlap. K$ hit price 0.113 us/hop [E using older M at MODEL:322]; actual dependent hit count/time [U]. | III — SCALAR | [U] / [U] |
| B. Slot, row and request range | G:730-734 / 793-797 | Runtime `blk/tpe`, multiply/subtract, `min(M,r0+rpt)`, empty-range guard. NB=1 removes request splitting. Fixed `tpe=24/32` and `rpt=32/64` [E from model.py:358-360 and trace task counts]; runtime division still requires scalar/vector reciprocal and correction instructions. Price N/(I×f) [E], I/f [U]. | II | [U] / [U] |
| C. LDS layout and expert routing | G:738-748 / 801-810 | Form pointers to xs/g/part and ids; coherent int id load (same address in every lane), wait, expert-stride 64-bit multiply/add. Useful ID bytes 4 B/task; request amplification is not 4 B of traffic. Loaded coherent dependency may cost 0.69–0.73 us [E from MODEL:321], current price [U]. Separate top-k already ran: mode=0, so G:745/807 recomputation is not executed (host.hip:458-465 under ETX_EMIT_FOLD_TOPK=0; certificate plan retains standalone top-k). | I | [U] / [U] |
| D. Stage activation bf16 → fp32 LDS, then stage barrier | G:389-407,763 / 821; coh.py:41,47-51 | Up/down read 2048×2=4096 / 768×2=1536 B [E]. Two `global_load_dwordx2 sc0 sc1` per active lane, 256/96 active lanes; unpack shifts/ands, two `ds_write_b128` per active lane. LDS write 8192/3072 B → 0.030476/0.011429 us nominal [E]. One coherent group latency, not two serial pointer-chase hops. Stage rendezvous adds bare 0.0612 us [E using impbar_hw.log:4] plus producer skew/drain [U]. | I | [U] / [U] |
| E. Select combined gate/up stream (up only); narrow path (down) | G:773-778 / 366-370 | Up validates K divisibility, whole-row ownership, no straddling and arena capacity; succeeds: `gp1=8`, `gp2=16`, `128%16=0 [E]`. GU1 removes the second partial clear/gather and its three helper barriers relative to two calls; do not credit it again. Down selects EPL=4; rejected SPLIT_TAIL stays off (G:289-292). Guard instruction time [U]. | III — GU1 (up); II (down path selection) | [U] / [U] |
| F. Segment/range setup | G:177-184 | Readfirstlane on K/r0/r1/wave; compute spr/S/per/sb/se. `cur=sb/spr` and pointer arithmetic follow clear/barrier, at G:188-201. SCALAR removed per-segment signed division and uses an incremental (row,seg) walk (G:215-224); initial division and runtime range work remain. Price N/(I×f) [E], [U] rate. | III — SCALAR | [U] / [U] |
| G. Zero the full per-wave partial matrix | G:185 | `16×64×4=4096 B` of `ds_write_b32` for each task [E]; one zero/thread. Nominal 0.015238 us each [E]. Only 64 of 1024 cells ever get nonzero contributions: `960/1024=93.75%` are empty [E]. | II | [U] / [U] |
| H. Clear rendezvous | G:186 | `s_waitcnt lgkmcnt(0); s_barrier`; bare 0.0612 us [E, impbar_hw.log:4], arrival imbalance and prior stores [U]. Stops a zero from racing a deposit. | II | [U] / [U] |
| I. Base/range start and accumulator init | G:188-204,218 | GU1 adjusts weight base for the up half; scalarized base+lane, initial division `cur`, incremental `seg`, acc=+0. No memory payload. Static source arithmetic; latency N/(I×f) [E], rates [U]. | III — SCALAR (remaining GU1 base handled in E) | [U] / [U] |
| J. Issue the unrolled weight group; advance/test pipeline loop | G:235-248; wload G:127-132 | 256/192 wave-vector loads (`global_load_dwordx4` / `global_load_dwordx2`, NT), 16384/12288 lane-vector loads [E]. Weights 262144/98304 B/task. Phase-wide service **11.705034/5.852517 us [E]** as §1.3, overlapping K–M. Issue/drain latency [U]. | I (weight requests); loop control separated in K | [U] / [U] |
| K. Consume control: next-row test, segment offsets and walk | G:219-224,248 | 256/192 consumed segments/task [E], 16/12 per wave. At each consume: compare row to cur; address LDS at seg×SEG+lane×EPL; increment seg, reset at spr and increment row. Four rows/wave; three transition flushes plus one final flush. Runtime loops cost 8/3 trips/wave [E]. SCALAR already replaces division per segment; any new constant-shape deletion has [U] time. | III — SCALAR | [U] / [U] |
| L. Row transition: flush the preceding row before next dot, then reset accumulator | G:220,205-214 | 3 transition flushes/wave + final at G:260 = 4/wave, 64/task [E]. Branch/masks and acc reset are machinery, apart from M's reduction/deposit. Price N/(I×f) [E], rates [U]. | II | [U] / [U] |
| M1. Wave sum of acc | G:208-211; gfx942.yaml:124-131 | Per flush: 1 `ds_bpermute_b32`, 1 `ds_swizzle_b32`, 4 DPP moves, 6 fp32 adds; 64 flushes/task [E]. The order is the original xor tree (32,16,8,4,2,1); only lane 0 result valid. Current latency [U]; counting each shuffle as a coherent memory hop is wrong. | III — WSUM | [U] / [U] |
| M2. Lane-0 partial read/add/deposit | G:209 | Lane 0 reads +0, adds wave sum, writes one float: `64×(4+4)=512 B LDS` [E], nominal 0.001905 us/task [E]. `+=` carries an LDS read → wait → add → write dependency, not just a store. Replace by original-order virtual-zero epilogue under whole-row ownership. | II | [U] / [U] |
| N. Unpack weights, read staged x, calculate dot and add to acc (inside each consume) | G:136-154,223 | Up/down 64×2048=131072 / 64×768=49152 bf16 multiplications with the original fp32 sums [E]. LDS read `64×K×4 = 524288/196608 B` → 1.950476/0.731429 us nominal service [E]. CPU ISA uses 2/1 `ds_read_b128`, 4/2 packed fp32 multiplies, shifts/ands and fp32 additions per 8/4 element vector (audit below). Actual LDS bank service/VALU dependency/overlap [U]. No precision change. | I | [U] / [U] |
| O. Pipeline tail and final nonempty-wave guard | G:250-260 | Tail guard and final sb<se guard execute; zero tail loads/consumes in these shapes [E], final flush still runs. Runtime per-unrolled-entry tail masks are unnecessary on a constant trip mapping; elapsed control cost [U]. | II | [U] / [U] |
| P. All-wave reduction rendezvous | G:267 | Wait for deposits then `s_barrier`; bare 0.0612 us [E], slowest-wave skew and LDS drain [U]. Cannot read incomplete partials. | II | [U] / [U] |
| Q. Wave-order epilogue gather | G:268-275 | Threads 0..63 (wave 0 only) sum **16** partials in w2=0..15 order. `64×16×4=4096 B LDS`, 1024 fp32 additions [E]; nominal LDS 0.015238 us [E], but dependent gather latency and compiler unroll scheduling [U]. 15 zeros per row, one useful partial [E]. | II | [U] / [U] |
| R-up. Round each gate/up dot to bf16, keep as fp32 in g | G:159,777 | 64 round-to-nearest-even bf16 conversions + 64 LDS float writes = 256 B [E]; nominal LDS 0.000952 us. Emits integer rounding/NaN handling, not truncation (ISA audit). Correctness requires the two intermediate bf16 roundings. | I | [U] / — |
| R-down. Load routing weight; multiply dot, round and store | G:831-832; coh.py:33-38 | ws load is written **before the helper call in source** and may overlap its pipeline; it is not a second routing/top-k phase. Useful 4 B/task ws; 64 fp32 multiplies, 64 bf16 conversions, **128 B coherent output** [E]. ISA `global_store_short sc0 sc1`. Required output service 256×128/(4.3e6)=0.00762 us phase-wide [E]; drain latency [U]. | I | — / [U] |
| S. Helper second barrier / arena reuse | G:277 | Bare 0.0612 us [E], plus imbalance [U]. Up makes g visible to the subsequent SiLU readers; down protects shared arena lifetime before next task. Retain one join in the conservative ownership rewrite. | II | [U] / [U] |
| T-up. Final row guard, load gate/up values, SiLU×up, round and store | G:786,790,160; coh.py:33-34 | Threads 0..31 handle rows; 64 LDS float reads=256 B → 0.000952 us nominal [E]; 32 `exp`, division, multiply and bf16 roundings; **64 B coherent output** [E]. CPU lowering includes exp, reciprocal, division correction/FMA sequence: more than a lone exp (audit). Output service 192×64/(4.3e6)=0.002858 us phase-wide [E]; latency [U]. | I | [U] / — |
| U. Tile return → task all-wave publication drain and barrier | kernel.py:323-333; gfx942.yaml:28 (DOMAIN release) | `s_waitcnt` store drain plus `s_barrier`; all **16 waves** must drain own stores [E from 1024 threads]. Bare 0.0612 us [E]; exposed coherent-output completion, dispatch call and actual wave skew [U]. Arrive happens after trace body and is excluded. | I | [U] / [U] |

The recurring order is **J → K → (L/M if changing row) → N**, repeated for each vector; then **O/M(final) → P → Q/R → S**. Down's ws request is issued before J in source, though consumed at R-down. Source phases are not compiler scheduling fences. Up's SiLU is after S; neither routing weight nor a recomputed top-k is part of up's final store in this mode. These distinctions prevent double-charging the route and moving the partial clear to the wrong end of the loop.

### 2.3 What is and is not priced by ISA

The audit locally compiles the actual source text with all four shipped flags and coherent/NT loads, restores source line tables, links local ROCm device libraries and emits gfx942 assembly. It retains dynamic arguments and fallback paths. The isolated wrapper uses ROCm **7.2.4 headers/libraries with Homebrew LLVM 23**, not the session's full-model ROCm compiler allocation. Static counts across all alternatives are not dynamic counts for this task. The persistent megakernel's scratch and dispatch costs are absent. See [CPU audit](../moe-body-anatomy-2026-10-11/isa-audit.txt) and [reproduction script](../moe-body-anatomy-2026-10-11/isa-audit.py).

A key emitted-code finding is that this compile's **up range/row path still contains vector compares, cndmasks and division corrections**, whereas down's incremental walk is mostly SALU. The existence of SCALAR does not prove that every downstream branch is uniform; the GU1-derived range can lose that property in compiler analysis. Fixed-shape constants and explicit wave-uniform range variables are therefore a narrower candidate than changing the numerical algorithm. Their current exclusive time and full-model instruction mapping remain [U].

`RBATCH` acts in **dsl_stage_norm_at** (`G:499-515`): batch the R row loads and residual while preserving row-index addition order. Neither of these MoE task bodies calls it: `dsl_stage` at G:763/821 is a simple vector stage. Do not assign a MoE phase to RBATCH or credit its shipped qkv/router benefit here. Likewise WSUM, SCALAR and GU1 savings are already in the measured body; this report does not count them again.

### 2.4 Reconciliation, with no hidden residual allocation

There is **no measured exclusive per-phase split**. An exact elapsed reconciliation is therefore:

| Elapsed account | up us/layer | down us/layer |
|---|---:|---:|
| Qualified exclusive measured phase credits | 0 | 0 |
| Unattributed measured body, including all A–U and their overlap | 761.50/48 = **15.864583 [E from M]** | 445.04/48 = **9.271667 [E from M]** |
| Total | **15.864583** | **9.271667** |

The zeros mean **no separately measured credit**, not absent work. `[U]` entries add nothing. This account is against the brief's rounded **15.86/9.27**, and rounds to those values; task means themselves were printed to 0.01 us. ISA and old ablations cannot identify how much of a wait retires VMEM, LDS, ALU dependencies, spills or a late wave. Assigning the entire residual to “loop machinery” would be the defect, not its cure.

For comparison only, an additive **byte-service bookkeeping model** closes as `11.705034 + 4.159549 = 15.864583 up`, `5.852517 + 3.419150 = 9.271667 down [E]`. The second terms are *unallocated above-ideal-byte-service balances*, not phase measurements or deletable time. The nominal LDS services, bare barriers and scalar-hit estimates above are deliberately **not** added to the byte estimate: they overlap it and each other. A full phase-price anatomy cannot be obtained by CPU arithmetic alone from these data.


## 3. Removable pool and smallest change

### 3.1 What can actually be summed

**Qualified exclusive removable pool: 0 us/layer = 0 ms/step credited.** That is an evidence result: no current measurement qualifies any separate phase as removable. It does not assert that the physical opportunity is zero. Nothing supports a measured **0.3–0.5 ms** pool here [U].

The one source-proven redundant structure is the **cross-wave partial matrix when every row has one owner**. Its removable **gross service work** is computable:

| Component removed per task | up | down | Nominal service price [E] |
|---|---:|---:|---|
| Partial clear (G:185) | 4096 B | 4096 B | 4096/(128×2100) = 0.015238 us each |
| Partial read/add/deposit (G:209) | 512 B | 512 B | 512/(128×2100) = 0.001905 us each |
| Cross-wave partial gather (G:274) | 4096 B | 4096 B | 4096/(128×2100) = 0.015238 us each |
| Clear and partial-join barriers (G:186,267); retain G:277 | 2 | 2 | 2×0.0612 = 0.1224 us each, older bare-barrier measurement at `results_2026-09-28b/impbar_hw.log:4` |
| **Gross service total** | | | **0.154781 us up + 0.154781 us down = 0.309562 us/layer**, `48×0.309562/1000 = 0.014859 ms/step [E]` |

Using the **0.106 us** older imported-round barrier value (`docs/qwen/MODEL-vllm-kernels.md:329`) instead gives `2×8704/(128×2100)+4×0.106 = 0.488762 us/layer`, `×48/1000 = 0.023461 ms/step [E]`. These two benchmark choices are a **sensitivity calculation, not a confidence interval or an upper bound**. LDS latency, scoreboard stalls, one-wave gather serialization and slowest-wave skew could make the observed interval larger; overlap with the stream could make net saving smaller. Newly distributed output stores or register pressure could make it negative. The virtual ordered zero additions must remain until bit-equivalence permits their removal; this arithmetic assigns no ALU saving to them.

Thus the calculable instruction/memory service pool is **about 0.015–0.023 ms gross [E]**, with **net exposed saving [U]**. It plainly cannot carry **0.324 ms** on its priced terms: `0.014859/0.324=4.59%`, `0.023461/0.324=7.24% [E]`. An exclusive measured saving of `324/48=6.75 us/layer [E]` would need roughly **14–22 times** this nominal gross service: `6.75/0.488762=13.81`, `6.75/0.309562=21.80 [E]`. A large serial gather/loop/skew cost remains a hypothesis [U], not added to this total.

### 3.2 The 0.158 ms is not an algorithmic ceiling

The plan's **0.158 = 0.139 up + 0.019 down ms [E]** is a **body-minus-serial-floor ceiling on the older `results_tiles` trace**, using round11 hop assumptions (`docs/core/PLAN-generic-decoder-2026-10-08.md:224,230`; `PLAN-300TOKS:99,132`). The brief calls it conservative; the plan calls the allowances **generous ceilings** and the ledger **deliberately optimistic** (`PLAN-300TOKS:3,33`). It neither prices every current phase nor proves an algorithm-wide ceiling for a new mapping. Removing or overlapping a hop invalidates that serial-floor premise. This audit **does not establish that 0.158 is the true maximum**. It also does not supply evidence to replace it with 0.3–0.5 ms.

Do not subtract old `streamw` times from today's body to claim a gain. The older pure-stream task medians were **14.08/8.20 us [M]** (`results_2026-10-01/round21/streamw.log:7,9`), in another session and mapping. They show why phase-shape achieved bandwidth matters. They are not a latency charge to current phases and do not certify the current candidate's floor.

There is also a whole-target ledger issue. **0.324 ms is the gap after already crediting the MoE row's 0.158 ms** (`PLAN-300TOKS:132,135-136,146`). If a new anatomy replaces that row, total MoE saving must be approximately `0.158 + 0.323667 = 0.481667 ms [E]` to fill the old gap while other rows remain at their optimistic endpoints. A total **0.300 ms** row supplies only `0.300-0.158=0.142 ms additional [E]`; a **0.500 ms** row supplies `0.342 ms additional [E]`. Comparing a total 0.324 ms MoE pool directly with the remaining target gap double-counts the old row.

On the **mean-body / ideal-4.3-TB/s bookkeeping basis alone**, only **0.363778 ms [E]** remains above weight service (§1.3), already smaller than **0.481667 ms [E]** before any essential work. This is a useful countercheck, **not a critical-path or wall-time impossibility proof**. Task mean is not critical-task body; mapping and achieved stream rate can change; their qualified effect is [U]. The evidence supports **“cannot presently carry the gap”**, not “a proof that every bf16 tile is capped at 0.158.”

**Upstream update discovered at final verification.** At **`main=a312354`**, the protected plan has already replaced the KEY32 estimate with its **0.231 ms measured paired saving**, and its optimistic ledger is now **0.900 ms**, leaving **0.237667 ms** on the historical anchor (`docs/qwen/PLAN-300TOKS-2026-10-11.md` **at a312354:128,135-138**). It explicitly retains the original **0.324 ms / 6.743 us/layer** pursuit rule (**:138,148**). This report evaluates that requested rule; it does not describe 0.324 as the latest ledger's shortfall. Filling the *updated* accounting gap by replacing the MoE row would require `0.158+0.237667=0.395667 ms total [E]`, still no qualified pool here. KEY32 does not price any new MoE-body slice.

The same main update annotates the XCD report's phase-specific bandwidth denominators, but retains its older raw trace and the quoted machinery taxonomy (`docs/qwen/XCD-TIMELINE-2026-10-11.md` **at a312354:13-21,275-290**). No newer intra-body stamps result from that correction. The ideal 4.3-TB/s convention here is labelled an optimistic weight-service bookkeeping model, not an achieved per-phase rate.

The target **1.137667 ms** itself belongs to the plan's **4.471 ms** historical anchor: `4.471-1000/300=1.137667 ms [E]`. The specified trace's own wall is **4.474 ms**, so its standalone arithmetic gap is `4.474-1000/300=1.140667 ms [E]`; neither is a measured improvement against the other session. This small basis difference does not change the conclusion.

### 3.3 Largest item and the smallest removal

**Largest timed removable item: [U].** Naming the **6.5 us residual** as that item would invent an attribution. The largest **provably redundant reduction data structure** is the partial matrix/gather above: no row in either audited shape has contributions from two waves. It is the smallest concrete experiment supported by this source anatomy.

In **`dsl_gemv_rows_nb` (G:168-278)**, add a fixed whole-row-owned path:

1. Guard exact shape/participation and retain the generic fallback: up `K=2048, nrt=64 combined GU1, rpt=32, 16 waves/1024 threads`; down `K=768, nrt=rpt=64, 16 waves/1024 threads [E]` as §2.1. Also require valid original row bounds and no wave crossing the gate/up discontinuity. Up needs the existing gap/base handling unchanged (G:188-196,773-777).
2. Delete **G:185-186** (partial zero/clear barrier). In **G:209**, keep `dsl_wave_sum0(acc)`, and let its lane 0 call the epilogue for `r0+cur` directly. Recreate `0.f+v` and the **w2=0..15** virtual-zero additions in the original order, with the nonzero at its owner index. Reset acc and retain the order of every `dsl_dot`/acc update (G:220-224).
3. Delete the partial rendezvous/gather **G:267-276**; retain **G:277** as one all-wave join. Up still materializes rounded gate/up values in g before the original SiLU at **G:790**. Down calls the unchanged multiply/round/store epilogue at **G:832** from each row's owner wave. Keep the task's **all-wave publication drain/barrier** at `kernel.py:325,332`.

The existing isolated up clone already implements this core edit (`bf16_300/source.py:73-81`) and falls back on the original path (`:85-91`). It is **prepared, not GPU validated** (`bf16_300/README.md:59-65`). Extending its proof to down requires a separate guarded path, not simply enabling the up template with EPL=4. This task changes no production tile.

Correctness risks, specifically:

- **Wave sum:** WSUM lane 0 is the only defined result; retain its xor-tree order and lane mask (`gfx942.yaml:117-131`). Using another lane's partial or a sequential reduction changes dots.
- **bf16 rounding:** up must round gate and up **separately before SiLU×up**, then round output; down must multiply fp32 dot by ws **before** the final bf16 rounding (`G:13-15,159-160,777,790,832`). No conversion to reduced precision is proposed.
- **Partial order:** every original zero addition, including the deposit's initial +0, is part of the numerical reference. Dropping them without proof can change signed zero/exceptional behavior; test byte equivalence on captured finite model inputs and inspect compiler math flags. Preserve actual w2 order, never completion order.
- **LDS lifetime:** up gate and up rows are owned by different waves. Keep the join before SiLU; no successor may reuse xs/g until all consumers finish. The down epilogue changes thread ownership but not row ownership.
- **Payload visibility:** the original down output is emitted by wave 0's lanes; the rewrite emits from multiple waves. Every producing wave must drain before the existing publication barrier. A thread-0-only drain is unsafe (`kernel.py:324-332`). Coherent store flags must remain.

After that structural deletion, the narrow next candidate is **constant/wave-uniform range and pipeline specialization** at **G:183-184,201,218-224,240-260**, especially the still-vector up path in the CPU ISA. It removes no required bf16 arithmetic. Its exclusive latency and ranking against gather/publication are [U]; it cannot be named the largest item by time yet.


## 4. Probe coverage and decisive GPU measurement

### 4.1 Existing probe semantics

`examples/vllm_llm/bf16_300/source.py:13-25,94-156` emits **twelve separate pair builds [M source inventory]**. Each selected task/wave retains two `s_memrealtime` stamps, then writes a **32 B record** (`:28-46`); **10 ns/tick** → `(end-begin)×0.01 us` (`attribution.py:22-23`). A compiler memory clobber constrains placement; it **does not retire global/LDS memory** (`source.py:35-39`). Its landmarks mean:

| Pair | Actual interval / limitation |
|---|---|
| arguments | Before block/grid/argument reads → after dsl_args (`source.py:144-152`); wrapper import initialization precedes it. |
| routing | Args complete → expert base We complete; includes slot/row/LDS-layout work, not just the id load (`:135-137`). |
| stage | Expert base → original stage barrier completes (`:138-140`). |
| zero | Before clear → after the original clear barrier (`:117-118`). Range arithmetic before the clear is excluded. |
| pipeline | Before s0/w array → after final flush (`:119-120`). Includes weight issue/consume, dots, all transitions/flushes and tails. It is not a pure loop-control price. |
| flush | Entry/exit of the **first selected named cur row** (`:125-127`); includes wave sum and partial deposit. It measures neither all flushes nor all rows. |
| join | Pipeline end → after original all-wave partial join (`:121-122`); includes waiting for other waves. |
| gather | After join → before reuse barrier (`:123-124`); selected wave 0 does gather and epilogue, other waves skip them. Up includes gate/up rounding/g writes; down includes multiply/round/output issue. |
| reuse | Before → after helper final barrier (`:123-124`), including wave skew. |
| publication | After body source → after original DOMAIN release and barrier (`:153`); excludes initial dispatch, but includes output drain. |
| body | Before arguments → after publication, matching the body+drain convention better than tile return alone (`:144,153`). |
| empty | Adjacent stamps after publication (`:154`); calibrates issue overhead, not resource-induced perturbation. |

The important **uncovered intervals** are: stage end → zero begin (path selection/range setup); zero end → pipeline begin (weight base, cur division, accumulator/walk setup); reuse end → publication begin (**up's final SiLU/output**, or remaining body control). The current set cannot close the ledger even by adding non-overlapping issue pairs. Pipeline and flush overlap; body overlaps everything. A flush or join sample on an early-finishing wave is not the current path's exclusive slice.

`dispatcher.py:18,34-72` emits **four separate dispatcher pair builds [M source inventory]** on a **two-task graph** (`:21-31`): descriptor/FIFO broadcast; wait/setup/acquire; exit/arrivals; empty. Note that the exit pair begins **after the body drain/barrier** at thread-0 return (`:67-71`): it does **not** include publication drainage already inside the body. This graph has neither the full model's task mix nor its contention; it is not a current MoE loop attribution.

The qualification gate requires same session/compiler/math identity, equal **VGPR, AGPR, VGPR/SGPR spills, scratch, LDS and static block capacity**, **32/32** and all imports byte equal; **actual last task and last wave**, with body perturbation **≤0.5 us [acceptance threshold]** (`attribution.py:12-21`). It still returns `exclusive_removable_us_layer=None` (`:25`). The old local compile already shows differing allocations for many pairs (`README.md:99-115`). A small empty-pair duration cannot repair changed compiler allocation. No qualified GPU timestamps or exclusive-removal certificate are present in this package.

### 4.2 Decisive experiment, registered before execution

**Question to decide:** does eliminating the whole-row partial matrix expose a real correct saving, or merely delete hidden LDS work? This needs a legal original-vs-specialized A/B, accompanied by landmarks; a further wrong-output ablation cannot settle it.

Use one MI300X VF, ROCm 7.14, bf16 Qwen batch 1/context 1024, cached existing weights/golden. Keep a **single host process with inputs resident** to avoid reloading the weight file for each sparse pair. First capture independent original outputs at selected decode positions and immutable up/down task inputs/IDs/weights, then run each original and candidate against the same inputs. Compare output bytes before timing the candidate; `host --check-dsl` is tolerance-based and is **not** a substitute for this new byte-equivalence gate (`G:30-31`). Full-model candidate must also have all imported-instance checks and **32/32**.

Extend the pair set by the three uncovered intervals above. To separate setup, loop and reduction, select disjoint boundaries at **G:177/185**, **G:187/235**, **G:267/268**, **G:268/277**, and up **G:786 / task RELEASE_DOMAIN**. For the loop, sample one unrolled trip at a time, using the original order at **G:240,245,248,250,260** and separate transition flushes at **G:220/209**. Each build still keeps only one begin/end pair, saves records after the measured body, and includes an empty/control build with matched allocation. Do not sum the trip/flush observations into a new body until all executed trips and boundary gaps are covered. Forced `vmcnt/lgkmcnt` fences belong in a **separate labelled retirement diagnostic**; they change overlap and cannot price the original critical path.

For path qualification, record actual **per-wave pre-join and pre-publication completion** along with the full task trace, using separate sparse builds with records indexed by `(task,wave)` after the body. The existing one-wave record pointer cannot be shared by simultaneously selected waves (`source.py:45,155`); allocate distinct records. Sweep all participating waves to detect a migrating slowest wave; gather/SiLU work is in wave 0 in the control. Validate last-task/last-wave evidence in the actual model run, not in a one-block replay. Repeat on representative early/middle/late layers and positions **0/1040**; use the current-plan body distribution to select tasks, not default task/wave zero. A representative isolated task is useful for cause, but cannot by itself certify the aggregate mean or wall saving.

Run **five interleaved original/candidate pairs [E experiment design]**, with production timestamps disabled for the decisive saving; sparse instrumented runs qualify attribution separately. Measure untraced **host wall and device** time and per-layer/critical body changes. Also replay the candidate's **same kernels in a HIP graph** to preserve the fusion baseline. If quoting a new system lead, include the strongest tested legal vLLM S3 in that same session; its run is not necessary to answer the anatomy alone.

**Go/no-go:** first require exact output and qualifying instrumentation. For a small skeleton continuation, use the existing **2.083 us/layer exclusive opportunity** pursuit bar (`PLAN-300TOKS:107`), then correct A/B net gain. To carry the brief's local threshold require `48×Δ(up+down)/1000 ≥0.324 ms [E]`, on actual critical/step time, every pair faster and spread reported. To repair the **whole-target ledger**, require **≥0.481667 ms total MoE saving [E]** (§3.2), and replace the old row rather than add a second one. If the measured result is small or hidden, close this deletion as a target-carrying route; retain any intermediate validated improvement under the existing KEEP rule. If it is large, recompute the full ledger and validate the new same-kernel graph/fused wall comparison (`pairs.py:138-163`).

### 4.3 Commands: available interfaces versus missing driver

The following **CPU commands exist today** and were used/read here; they launch no GPU. They emit/prove source and compiler resource differences, not phase latencies:

```bash
.venv/bin/python docs/moe-body-anatomy-2026-10-11/isa-audit.py
.venv/bin/python -m examples.vllm_llm.bf16_300.build --task 0 --wave 0 --row 0
# Compile the isolated prepared family locally, in an isolated output tree:
ETX_DEVLIBS=/Users/ckl/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode   .venv/bin/python -m examples.vllm_llm.bf16_300.build --compile --task 0 --wave 0 --row 0
```

Do **not** run the latter two against this worktree's shared `build` symlink: use an isolated ordinary output tree, as the original preparation did (`README.md:125-150`). This audit compiles under `/tmp` and leaves the shared build untouched. Preserve each selector's emitted identity before selecting another: the current build CLI has no output-directory argument (`build.py:105-108,175-181`). On a VM compile the generated independent probe TUs with **the VM's compiler**, rather than treating these local resource numbers as its result. The existing `rocm_build_check.sh` targets ROCm **7.2.4**, not the supplied session's **7.14** (`rocm_build_check.sh:2-5`).

For a **future qualified full-model original and candidate build**, these GPU/check commands already exist. Here `ANAT_MODEL` is an already cached packed model directory and `ANAT_BUILD` a separately built `ETX_VL_CHECK=1` directory with its own run/plan/identity; no old certificate is transferred:

```bash
ANAT_MODEL=/root/llm/qwen3-30b-a3b
ANAT_BUILD=/root/sess/moe_anatomy_original
ANAT_LOG=/root/sess/moe_anatomy_evidence
mkdir -p "$ANAT_LOG"
timeout 240s stdbuf -oL -eL .venv/bin/python   -m examples.vllm_llm.import_certificate --run-build "$ANAT_BUILD"   --model "$ANAT_MODEL" -- --check 48 --check-pos 0 --repeat 1   > "$ANAT_LOG/check_p0.log" 2>&1
timeout 240s stdbuf -oL -eL .venv/bin/python   -m examples.vllm_llm.import_certificate --run-build "$ANAT_BUILD"   --model "$ANAT_MODEL" -- --check 48 --check-pos 1040 --repeat 1   > "$ANAT_LOG/check_p1040.log" 2>&1
ETX_TRACE=1 ETX_TRACE_RAW="$ANAT_LOG/task_trace.bin"   timeout 180s stdbuf -oL -eL .venv/bin/python   -m examples.vllm_llm.import_certificate --run-build "$ANAT_BUILD"   --model "$ANAT_MODEL" -- --repeat 1 > "$ANAT_LOG/task_trace.log" 2>&1
```

Repeat the checks with `ANAT_BUILD=/root/sess/moe_anatomy_candidate`. For each built arm, capture untraced fused and graph runs with the same certified interface; graph adds **`--unfused`**. Feed five interleaved fused/graph logs to:

```bash
.venv/bin/python -m examples.vllm_llm.pairs --basis wall --pairs 5   /root/sess/moe_anatomy_evidence/f1.log /root/sess/moe_anatomy_evidence/g1.log   /root/sess/moe_anatomy_evidence/f2.log /root/sess/moe_anatomy_evidence/g2.log   /root/sess/moe_anatomy_evidence/f3.log /root/sess/moe_anatomy_evidence/g3.log   /root/sess/moe_anatomy_evidence/f4.log /root/sess/moe_anatomy_evidence/g4.log   /root/sess/moe_anatomy_evidence/f5.log /root/sess/moe_anatomy_evidence/g5.log
```

**The missing executable matters:** `bf16_300` has **no GPU launcher, full-model pack or host record reader** (`README.md:131-150`). Those commands alone cannot run its timestamp probes or the new down specialization. There is **no existing exact “run the probes” command that closes this split**. Before any paid run, implement and CPU-check the resident-input host driver, record allocation/reading, additional landmarks, byte-equivalence capture, full-model sampling integration and a single `gpu_jobs.py` session job. Its required invocation contract is: select original/candidate and pair, original argument blocks and recorded `(layer, task, wave, row)`, iterations, deadline, output path, and independent original-output reference; return full raw records, resource identity, body perturbation, checks and task/wave completion evidence. An invented command against a nonexistent launcher would conceal the same readiness problem this audit found.

### 4.4 Time and price, not spent

**Estimated diagnostic session: 30 minutes [E planning budget]**, assuming cached image/model and a resident-input driver: **8 min** VM compiler/resource gates + **2 min** load/initialize + **8 min** import/token/byte-output checks + **5 min** sparse issue/retirement diagnostics + **5 min** interleaved untraced pairs + **2 min** evidence/cleanup. These are budget allocations, not measured durations; actual driver/build runtime is [U]. At the customer's **$1.99/hour**, `30/60×1.99 = $0.995 ≈ $1.00 [E]`. A hard **40 min** cap costs `40/60×1.99 = $1.327 [E]`; the extra `10/60×1.99 = $0.332` is a budget margin, not approval to spend it. A full new S3 system comparison/image setup is outside this cached diagnostic budget [U].

Nothing was provisioned, downloaded, or run on a GPU here: **GPU minutes/spend = 0**. The future job must preserve full logs, print progress, enforce substep deadlines and probe silence, then delete its VM immediately and record the closing balance, per AGENTS §4. **Do not provision against the currently incomplete launcher/allocation gate.**


## 5. Counter-argument and unresolved evidence

### 5.1 Strongest case against a large removable pool

The control is already specialized in the ways the old ladder did not have: one gate/up pass, scalarized arguments/incremental segment walk, lane-0-only wave reduction, and reduced default load depth (`G:156-224,345-359,767-783`; flags recorded at `results_qwen300_2026-10-11/certificate_off/build_identity.json:23`). The weight bytes and required fp32 accumulation order remain; WSUM's dependency tree, bf16 conversions, SiLU and routing output cannot be discarded as machinery. Zero and gather consume modest useful LDS service; they may run while the weight pipeline is already outstanding. A barrier issue count is not its exposed delay, and its observed long interval may merely wait for an essential dot/stream on another wave.

The source's obvious redundancy therefore does **not** establish a large performance redundancy. Its smallest correct rewrite distributes down stores from one wave across all owner waves, potentially increasing coherent request pressure and breaking store coalescing [U], even with unchanged useful bytes. It can also change live registers/scratch in the full persistent kernel. Both effects can erase nominal LDS savings. Current full-model spill/occupancy metadata and the probe allocation mismatches make that concern concrete (§1.3, §4.1).

The older no-stream ablation is particularly vulnerable to this counter-argument: it executes real dots and activation reads, with all waves' row/flush dependencies intact, on an intentionally wrong program whose generated schedule changes. **63% / 69% retained** does not mean those percentages are legally removable. Its own marginal “reduction” cut contains required wave arithmetic and coherent outputs, not just empty-partial machinery. The subtraction residual has no unique causal owner. The most defensible present position is: **the residual has not been partitioned; the clear/gather redundancy is real; a large exposed saving is unproved**.

### 5.2 Evidence that would decide the argument

- **Confirm a large anatomy:** qualified stamps cover setup, every exercised pipeline group/flush, join, gather, final SiLU and publication, and close the actual body on the actual critical task/wave within measured perturbation. A legal guarded rewrite preserves output bytes, full-model **32/32**, and all imports; five interleaved pairs show the corresponding **untraced** step/host saving (§4.2). A long gather issue interval alone is insufficient.
- **Refute the candidate's causal price:** clear/gather intervals or total correct A/B saving are small, disappear under concurrent streaming, or shift into store drain/scratch; the above-ideal balance mostly belongs to required dot/latency/rounding/publication. Record the observed cut without reallocating the leftover residual to another unmeasured phase.
- **Invalidate the instrument:** changed resource allocation, missing last-task/last-wave proof, body perturbation beyond the existing **0.5 us** limit, wrong outputs, or incomplete run (`attribution.py:12-21`). Such data produce no price and no removable credit.

Consulted existing work before naming a lever: `docs/core/DATA.md:32-35` inventories latency/stream/flag data; the fleet status rejects split worker groups, uncached allocations alone and idle-worker prefetch (`~/code/fleet-mi300x/docs/STATUS.md:300-304`), and records the occupancy experiment (`:130-161`, raw `results/microbench_f_occupancy.txt`). ETX design §15 records the failed weight-touch experiment and measured store/visibility/overlap behavior (`docs/core/ETX_Technical_Design.md:971,1030,1042`). None supplies the missing current MoE phase timestamps. This report proposes no repeat of those measured levers.

### 5.3 Corrections and limits of this deliverable

1. **1067 us** has no identified source in the delivered evidence [U]; the specified trace's rounded mean MoE sum is **1206.54 us [E]** (§1.1).
2. The requested log values are in **attempt1**, while the top-level file has a different observation (§1.1). The raw top-level binary also cannot be assumed to be that archived observation; it was not used to invent a critical split.
3. **71.9%** is a ratio of summed layer means to a span, not a critical-path partition; its complement includes unlisted task bodies, not just gap/wait (§1.1).
4. **1785.1/1551.3 us** are another trace's estimated taxonomy; **6.5 us** is an older subtraction residual, and is not supported for both tasks (§1.2). NOVMEM keeps one weight-vector request per wave.
5. The **0.158 ms** allowance is an older-trace ceiling under a serial-floor model, described as optimistic in the plan. It is not proved to be the true algorithmic ceiling (§3.2).
6. Current down **does not straddle rows**: `per=12`, `spr=3`, hence whole-row ownership. The plan's generic warning that down *can* straddle (`PLAN-300TOKS:97`) is valid for other shapes, but cannot exclude this guarded shape (§2.1). No protected plan was edited.
7. **0.324 ms** is the requested historical threshold, additional to an already-credited MoE row; meeting that threshold by replacing the row needs about **0.482 ms total**. The latest main ledger shortfall is **0.238 ms**, and its corresponding replacement requirement is **0.396 ms [E]**, as §3.2 records.
8. **Largest removable phase by elapsed time, exact per-phase shares, actual shader/LDS/ALU/exp costs under this workload, full-model dynamic MoE spill cost, and net candidate saving are [U].** This is a complete source/operation anatomy with an explicit measured-body residual, **not a measured phase-price certificate**. The current inputs cannot supply that certificate under a CPU-only constraint. The existing probes also lack the launcher/integration and complete landmarks required to obtain it (§4).

### 5.4 Local verification

- Recomputed all **432 layer rows per log [E: 9 roles×48 layers]** for the archived and top-level traces; checked every stored source line, body, task count, aggregate and SHA256 against the raw files. Archived sum **3086.96 us**, top-level sum **3092.97 us [E from M]** are intentionally separate (`trace-sums.json:3-16,453-466`).
- Enumerated every segment of the exact up/down shape; each of the **64 rows** has one owner, all row-local segments remain in order, and each wave has **8/3 full pipeline trips [E]** from §2.1. No tail executes.
- Compiled both mapped original tile wrappers to gfx942 ISA with local device libraries. They have **56/54 VGPR**, **zero VGPR/SGPR spills and scratch**, **21120 B LDS [M local compiler metadata]** (`isa-audit.txt:73-79,133-139`). This verifies compilation/operation inventory, **not** runtime math, full-model allocation, or GPU speed.
- Rechecked the weight-service reconciliation, gross service totals and dollar arithmetic (§§1.3,3.1,4.4); `git diff --check` passed. No full-model test or GPU correctness certificate was generated for the proposed rewrite.

The requested document and its reproducible CPU evidence are committed on `wf/moeanat`. No production kernels or protected documents were changed; no GPU time, network spend or push occurred.

