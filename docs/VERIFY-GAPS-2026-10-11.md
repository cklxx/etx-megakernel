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
