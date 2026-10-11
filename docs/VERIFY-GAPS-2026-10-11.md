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
