# MoE body anatomy — 2026-10-11

Scope: Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X (gfx942), ROCm 7.14.0. CPU-only source and evidence audit; no GPU run or provisioning. Work is on `wf/moeanat`.

Evidence convention: `[M]` is a measurement with a source location; `[E]` is an estimate with arithmetic; `[U]` is unknown and contributes nothing to a numeric total. Instruction counts are not phase latency measurements. Overlapping costs must not be added as if serial.

## 1. Evidence, identity, and reconciliation

### 1.1 Which measurement is being anatomised

All external `results_*` paths below are relative to `/Users/ckl/code/etx`, not this worktree. The supplied numbers belong to **`results_qwen300_2026-10-11/attempt1/tr_off.log`**, not the current top-level file. The archive reports span **4295.4 us [M]** (`:26`), **32/32 [M]** (`:25`), device median **4.297 ms**, wall **4.474 ms**, **233 tok/s [M]** (`:464-466`). Hardware is MI300X VF, gfx942, ROCm 7.14 (`results_qwen300_2026-10-11/record.log:7,15,18`). The current top-level log reports **4.299/4.471 ms [M]** (`:464-466`); do not silently substitute it. Both logs bracket the same build-identity hash (`:3,467`), but are different observations.

Summing the printed mean-body column of the archived trace gives **761.50 us up**, **445.04 us down**, and **3086.96 us for the nine layer roles [E]** (recomputed, with every input line and hashes, in [trace-sums.json](moe-body-anatomy-2026-10-11/trace-sums.json)). Therefore the exact rounded-input targets here are `761.50/48 = 15.864583 us up`, `445.04/48 = 9.271667 us down`, total `1206.54/48 = 25.13625 us/layer [E]`. The brief's **15.86/9.27** are rounded descriptions. **1067 us is not this trace's MoE body sum**: `761.50+445.04 = 1206.54 us [E]`. Its provenance is [U].

The body is an **all-task mean for a layer**, not a measured particular task or the last task's critical body. Tasks run concurrently: do not multiply by the task count. Body timing includes `etx_call_body`, **all-wave store drain and its final barrier**, between `tr[1]` and `tr[2]` (`etx/codegen/kernel.py:314,323-333`). Event arrival, next-task fetch and dependency wait lie outside it. Source-tile end is thus not trace-body end.

`3086.96/4295.4 = 71.867% [E]` is a sum of mean layer bodies divided by the span, **not a disjoint critical-path partition**. It excludes prologue, input norm, lm_head and argmax (`attempt1/tr_off.log:28-30,462-463`). The complement is not simply gap/wait. A critical-path partition needs the raw task trace and matching plan.

### 1.2 The two older analyses cannot be transplanted

The XCD report explicitly uses `results_tiles/tr_all.bin`, a **4412.1 us [M]** span, with **1785.1 us machinery and 1551.3 us body residual [E]** (`docs/XCD-TIMELINE-2026-10-11.md:12,19,88-102`). Its “machinery” is measured body **minus a byte/hop model**, not intra-body measurement. Neither those totals nor their split is a phase price for this trace.

The ladder is older again: **18.37/11.11 us [M]** controls versus **11.55/7.64 us [M]** NOVMEM (`etx/ops/gemv.py:17-29`; raw finding `results_2026-10-01/round37/FINDING.md:3-17`). NOVMEM **still loads one weight vector per wave**, reusing it; it is not literally zero weight reads (`gemv.py:237-245`). It retains dot, activation LDS reads and loop arithmetic, but changes dependencies and compiler scheduling and gives wrong output. Its marginal differences cannot be summed as exclusive phases.

For example the old up residual is `18.37 - 1.50 - (18.37-11.55) - (18.37-15.45) - (18.37-17.81) = 6.57 us [E]`; down gives `11.11 - 1.43 - 3.47 - 2.47 - 0.43 = 3.31 us [E]`. These are **subtraction residuals of separate experiments**, not measured loop slices. In particular **6.5 us each for up and down is unsupported**. `[U]` exclusive current-loop time contributes no saving.

### 1.3 Parameters and ceiling before a new experiment

One up task owns **32 gate plus 32 up rows**, K=2048; one down task owns **64 rows**, K=768 (`examples/vllm_llm/model.py:358-360`). bf16 weight bytes are `64×2048×2 = 262144 B up` and `64×768×2 = 98304 B down [E]`. With **192/256 tasks [M]** (`attempt1/tr_off.log:37-38`), the phase bytes are `50.331648/25.165824 MB [E]`. At the repository's **4.3 TB/s achievable-rate convention** (`AGENTS.md`, §2), the bandwidth service floors are **11.705034/5.852517 us/layer [E]**. They are phase-wide floors, not each CU's bandwidth, and overlap instruction work.

The corresponding residuals are `15.864583-11.705034=4.159549 us up`, `9.271667-5.852517=3.419150 us down`, `48×7.578699/1000 = 0.363778 ms/step [E]`. **This is diagnostic headroom above an ideal rate, not a removable pool.** It includes essential dots, staging, wave sum, serial latency, rounding, epilogues and publication. Removing **0.324 ms** requires `324/48=6.75 us/layer [E]`, about `324/363.778=89.1% [E]` of that ideal-rate headroom. The more exact plan gap is **0.323667 ms [E]**, `6.743 us/layer` (`docs/PLAN-300TOKS-2026-10-11.md:136,146`).

Measured older hardware parameters: scalar hit **0.113 us**, loaded vector L2/MALL/HBM **0.73/0.71/0.69 us**, bare round barrier **0.106 us** (`docs/MODEL-vllm-kernels.md:320-329`). These are other-session microbenchmarks, not measured current phase times. `bench/hw/hw_bench.hip:16-18,23-28` describes the primitive/clock/latency tests. No qualified current shader-clock, SALU issue rate, LDS latency, or exp/reciprocal throughput measurement was found [U]. Nominal **2.1 GHz** is a reported device property (`results_2026-10-01/round21/streamw.log:1`), not a loaded-clock measurement. Cycle-to-us prices using it must be `[E]`.

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
| C. LDS layout and expert routing | G:738-748 / 801-810 | Form pointers to xs/g/part and ids; coherent int id load (same address in every lane), wait, expert-stride 64-bit multiply/add. Useful ID bytes 4 B/task; request amplification is not 4 B of traffic. Loaded coherent dependency may cost 0.69–0.73 us [E from MODEL:321], current price [U]. Separate top-k already ran: mode=0, so G:745/807 recomputation is not executed. | I | [U] / [U] |
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
| U. Tile return → task all-wave publication drain and barrier | kernel.py:323-333; gfx942.yaml release lowering | `s_waitcnt` store drain plus `s_barrier`; all **16 waves** must drain own stores [E from 1024 threads]. Bare 0.0612 us [E]; exposed coherent-output completion, dispatch call and actual wave skew [U]. Arrive happens after trace body and is excluded. | I | [U] / [U] |

The recurring order is **J → K → (L/M if changing row) → N**, repeated for each vector; then **O/M(final) → P → Q/R → S**. Down's ws request is issued before J in source, though consumed at R-down. Source phases are not compiler scheduling fences. Up's SiLU is after S; neither routing weight nor a recomputed top-k is part of up's final store in this mode. These distinctions prevent double-charging the route and moving the partial clear to the wrong end of the loop.

### 2.3 What is and is not priced by ISA

The audit locally compiles the actual source text with all four shipped flags and coherent/NT loads, restores source line tables, links local ROCm device libraries and emits gfx942 assembly. It retains dynamic arguments and fallback paths. The isolated wrapper uses ROCm **7.2.4 headers/libraries with Homebrew LLVM 23**, not the session's full-model ROCm compiler allocation. Static counts across all alternatives are not dynamic counts for this task. The persistent megakernel's scratch and dispatch costs are absent. See [CPU audit](moe-body-anatomy-2026-10-11/isa-audit.txt) and [reproduction script](moe-body-anatomy-2026-10-11/isa-audit.py).

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

Pending: measured versus conditional removable pool, comparison with the requested 0.324 ms, largest item and correctness invariants.

## 4. Probe coverage and decisive GPU measurement

Pending: existing isolated timestamps, blind spots, exact commands, time and dollar budget. Commands are a future protocol, not executed here.

## 5. Counter-argument and unresolved evidence

Pending: strongest alternative explanation and falsification criteria.
