# Verification gaps — 2026-10-11

Starting commit: `9f91f3a`. CPU only; no GPU, provisioning, downloads,
network spend, timing experiment or performance claim. All compiler counts
are static observations of the named compiler, not residency measurements.

## Report generator checkpoint

`python -m etx.tools.verify_gaps ASSEMBLY --ir IR --output JSON` preserves
all instructions of every AMDHSA kernel, with zero-based instruction index,
assembly line, operands and unfiltered text. Input hashes identify the build.
Missing kernel input fails closed. No device is launched.

Non-GPU real-input dry run:

```text
VERIFY {"instructions": {"etx_megakernel_d0": 101898, "etx_unfused_d0": 27715}, "kernels": 2, "output": "/tmp/etx-verifygaps/generator.json"}
```

Local gfx942 compile with `ETX_DEVLIBS` (LLVM 23, ROCm 7.2.4 libraries):
`docs/isa-fixes-2026-10-11/census.py baseline --out /tmp/etx-verifygaps/local`
exited 0 and printed:

```text
CENSUS baseline: /private/tmp/etx-verifygaps/local/baseline.txt
```

No new shell script: shell syntax gate is not applicable at this checkpoint.
Focused generator tests: `2 passed in 1.27s`. Full suite result is recorded
in the checkpoint log. Further checkpoints below will add gap evidence.

Full suite (collected before the two generator tests were added):
`1322 passed, 14 skipped in 189.98s (0:03:09)`; the two new tests also passed
separately. The final full suite will include all new tests.

## Gap 1 — LDS and memory ordering

The retained baseline/C1/C5 sources, plans, IR, assembly and import bundle were
SHA-256 checked against `whole-resources.json`, then rebuilt locally with the
installed ROCm 7.2.4 LLVM 22 compiler in `etx-rocm22` (`--pull=never`,
`--network=none`, no device). Debug-line rebuilds retain the same instructions
and resources. Full before/after `amdgpu-lower-module-lds` dumps provide the
field layout; object attribution uses matched instructions, with a uniquely
sized remaining object/field bijection when matching cannot disambiguate.
Every byte reconciles with the final compiler LDS total.

Artifact: `verify-gaps-2026-10-11/{baseline,c1,c5}-memory.json.gz`, ordinary gzip
JSON, all kernel instructions plus the complete memory/wait/barrier sequence,
source locations, wait counter scopes and independent-load sites.
`memory-summary.json` retains all object names, half-open byte ranges and
source declarations without the instruction listing.

The megakernel's **11 objects / 29,736 bytes** are identical in all three arms:

- [0,21120): `etx_lds_arena`, imported tile arena, 21,120 bytes.
- [21120,29568): `etx_run_task::etx_lds`, DSL arena, 8,448 bytes.
- [29568,29632): `etx_imp_nbar[16]`, 64 bytes.
- [29632,29664), [29664,29696): `etx_imp_bar_count[8]` and `etx_imp_bar_gen[8]`, 32 bytes each.
- [29696,29716): `etx_megakernel_d0::s_task`, 20-byte task descriptor.
- [29716,29720), [29720,29724), [29724,29728): `s_tid`, `s_domain`, `s_worker`, 4 bytes each.
- [29728,29732), [29732,29736): `etx_imp_done`, `etx_imp_exit`, 4 bytes each.

Unfused: **7 objects / 29,704 bytes**; the same first five ranges, then done
[29696,29700) and exit [29700,29704). Its DSL arena belongs to `etx_unfused_d0`.
No additional compiler-promoted LDS object appears in these compiler outputs.
Arena views are aliases of the named arena, not additional LDS allocations.
**C5 added 0 objects / 0 bytes to LDS.** Its scratch change is not an LDS change.

Count definition, identical in every arm: a load issued after a wait within
the same basic block whose address/data/control registers do not transitively
depend on a load that wait completed. Branches, stores and barriers terminate
the candidate window. These are register-independent serialized **static
sites**, not executed load counts. Read-read aliasing is not a write hazard;
external memory-order requirements, legal speculation, and exposed latency
are not established by this count. The tool deliberately prints
`proven_overlapable_loads: null`. No wait removal is certified.

Baseline/C1/C5 megakernel site counts: **106 / 106 / 112**, deltas **0 / +6**
against baseline. Unfused: **92 / 92 / 95**, deltas **0 / +3**. C5's whole-kernel
+6 consists of +4 vector-global and +2 scalar-load sites, with LDS site count
unchanged. This is the compiler-reorder delta; it is not an attribution of
all those sites to the norm-weight prefetch or a time effect. The requested
stronger count of loads that *should legally overlap* remains **open**.
The LDS inventory and unfiltered memory ordering are closed.

CPU build driver dry run printed `VERIFY CPU {baseline,c1,c5} stage={0,1,2,3}
DRY_RUN` (12 lines); the actual build printed the same 12 lines with `BUILD`.
Both exited 0. Each stage has a 180-second subprocess deadline and full logs.
No shell script added. Full suite at this checkpoint:
`1327 passed, 14 skipped in 207.60s (0:03:27)` (five generator/memory tests
included). LLVM 23 gfx942/ETX_DEVLIBS baseline compile also exited 0 above.

## Gap 2 — spill pairs and the reduction dependence slice

Artifact: `etx/tools/spill_chain.py` and `{baseline,c1,c5}-spills.json.gz`.
Every static spill store/reload has its instruction index, assembly line,
physical-register value version, concrete lane or scratch-word slots, and
all CFG-reaching spill stores. Store value/control definitions are named by
instruction index rather than guessed C++ variable names. `allocated.mir`'s
WWM-reserved VGPRs distinguish compiler SGPR spills from semantic broadcasts.
Scratch slots are tracked per 4-byte word, including partial x2 reloads.
**Unmatched reloads: 0 in every kernel/arm.**

Definition, unchanged across arms: the conservative **MAY-dependence slice**
from non-scratch global/scalar loads to the final lane-pair addition of every
DSL `dsl_wave_sum0` reduction (`quad_perm:[1,0,3,2]`, including an addition
moved into a successor's lane-zero branch). The slice includes explicit
registers, implicit exec/vcc/scc, CFG-reaching definitions through backedges,
SGPR lane spills, scratch words and branch control dependence up to the
immediate postdominator. LDS aliases are conservatively joined. It is the
intersection of load-descendant and final-reduction-ancestor sets.
This is **not a longest executed timing path**. Alternative dispatch arms,
loop trips and conservative LDS aliasing can give false positives.

Megakernel static reload instructions (scratch / SGPR lane):
**5 / 6327** baseline, **5 / 6349** C1, **10 / 6327** C5.
On the defined MAY slice: **5 / 4482 = 4487**, **5 / 4497 = 4502**,
**10 / 4482 = 4492**, respectively. Delta from baseline: C1 **+15**,
C5 **+5**. Final reduction sites: **89 / 90 / 89**. No CFG block lacks an
exit path. Both reload classes and slice counts are **0** in unfused in every
arm. The native metadata's 6/8 VGPR and 255/261 SGPR spill counts describe
allocated spills, not the number of static reload instructions. One reload
can carry two dwords, and many branch sites can reload the same spill slot.
Do not compare those different count bases as if they were equivalent.

The pair inventory and repeatable static slice counts are closed. The brief's
stronger assertion that a reload lies on **the actual critical chain** is
still **open**: the tool prints `executed_critical_reload_count: null` and
per-reload `executed_critical_chain: null`. No GPU or dynamic path was supplied.
It does not replace that unknown with the MAY count. Source register versions
also do not recover optimized-away C++ variable names.

Full suite at this checkpoint: `1329 passed, 14 skipped in 281.13s (0:04:41)`.
The native three-arm builds and local gfx942/ETX_DEVLIBS compile above exited
0; this checkpoint adds Python analysis, no device logic or shell script.
Raw CLI stdout is preserved in the three `*-report.log` files; it includes
both per-kernel counts. New tests cover CFG spill-store merges, lane-spill
classification, missing slot provenance, transitive address dependence,
DPP/swizzle modifier parsing and a final reduction add across a branch.

## Gap 3 — polling-wave prefetch isolation

**Static/code verification closed; GPU acceptance pending and flag OFF.**
Flag: `ETX_PREFETCH_ISOLATE_POLLER=1` **at emission**, e.g. on the environment
of `build.sh`; it is not a hipcc `-D`. The default and explicit `=0` emit the
starting commit's source byte-for-byte. With `=1`, the optional `pre_wait`
argument-cache hook moves from `threadIdx.x < 64` (wave 0) to
`64 <= threadIdx.x < 128` (wave 1). The generic weight-prefetch hook excludes
wave 0 with `threadIdx.x >= 64`; the existing weight touch and next-task ARGPF
hooks already exclude the poller. A one-wave launch omits the optional hook
because no spare wave exists. No barrier, boundary, payload math, worker,
precision or parameter is changed.

Yes, wave 0 prefetches today: `etx_pre_*` calls `etx_imp_touch`, which issues
**scalar argument-cache loads**, not VMEM weight loads. The exact wave-0
primitive specialization in `polling-original.s` has four prefetch instructions:

```text
4: s_load_dword s6, s[4:5], 0x9c
5: s_load_dword s7, s[4:5], 0x0
6: s_load_dword s8, s[4:5], 0x40
7: s_load_dword s9, s[4:5], 0x80
8: s_waitcnt lgkmcnt(0)
12: global_load_dword v1, v0, s[2:3] sc1
14: s_waitcnt vmcnt(0)
```

Indices 0/1 are required kernel-argument loads; 17 is the abort-pointer load;
29/39 are abort/event polls. Their scopes are printed without filtering in
`polling-build.log` and `polling-{original,isolated}.json`. The specialization
uses the **real runtime primitives**, removes outer dispatch/wave predicates
and is **not an executed full-dispatch trace**. Full original/ON ISA is also
retained as `isolation-{original,on}.s.gz` for inspecting those predicates.
The isolated wave-0 specialization has **0 prefetch instructions**; event and
abort polls and their vmcnt(0) waits remain. These scalar prefetched values
are drained by **lgkmcnt(0) before the event poll**. A vmcnt(0) waiting on a
VMEM prefetch is **not demonstrated on this argument-hook path**. The brief's
co-running-stream interference evidence does not establish that mechanism.

Full Qwen plan: **60,362 tasks / 304 workers**, OFF source identical to
`9f91f3a`. Independently compiled `.text` files are committed for direct byte
comparison. OFF `.text` byte identity **true**:
`0c65425003800f4db327ce1508433a5c1a368c3b7b286e421e03cf026da3b0a2`
for both original and OFF; ON hash
`aaadb2712504d5508f672ad287d36edaf8ae42c92c6e4d4b7d633b01bb58ce7a`.
All three arms: megakernel **128 VGPR / 28 scratch bytes per lane /
29,736 LDS bytes / compiler occupancy 4**; unfused **103 / 0 / 29,704 / 4**.
These are compiler resources, not executed correctness or residency evidence.

Future keep rule **[E]** (registered decision threshold, not a measured effect):
isolated round trip **strictly below 2.0 us** AND total gain **at least
0.030 ms/step**. Basis: the brief's prescribed threshold and the separate
co-running-stream observation in `XCD-PREFETCH-2026-10-11.md`; no time effect
has been measured here. The flag remains OFF until a GPU session passes it.

CPU gates: full suite **1342 passed, 14 skipped in 227.17s (0:03:47)**;
local LLVM 23 gfx942 compile with explicit ETX_DEVLIBS for both polling arms
exited 0; all 12 full-plan native build/assemble/extract stages exited 0.
Both new driver dry runs exited 0 (stdout saved). No shell script added;
`bash -n etx/importer/hip_ir_local.sh` on the reused compiler helper exited 0.
No CHECK-DSL, imported-launch byte check, HF tokens or GPU timing was run.
