# Final local acceptance

Status: LOCAL PASS. GPU execution is NOT RUN. GPU spend is $0.
The candidate remains off by default.
The checked source has 324 files. The receipt has 17 commands, all with exit 0.
`final-v7/commands.json` records each exact command, elapsed time and raw log.
`local-ready.json` records the checked source hashes.
No source file changed during the final acceptance.

## Executed checks

- Full repository suite: **2,052 passed, 17 skipped, zero failures**, exit 0, 590.04 seconds.
- Focused suite: **165 passed, one skipped**, exit 0, 16.96 seconds.
- Native shared-header simulation: 32 synthetic cases and 96 completion/reuse epochs, exit 0.
  Ordered IDs match in 30/32 cases. Expert sets match in 32/32. Mixing bits match in 0/32.
  These are not real-model pass rates. The retained sparse witness still changes expert 7 to 8.
- Session graph resolution and dry run: exit 0. The graph has one planned GPU lane.
- Shell syntax: build, matrix, ROCm check and session wrapper all exit 0.
- Full local matrix: **16/16**, each rc=0. It uses gfx942 and the real ROCm device libraries.
- Both baseline and producer: real ROCm 7.2.4 clang 22 device assembly and host-device pass, exit 0.
- Both arms: native LLVM 23 device IR and gfx942 assembly, exit 0.
- Both arms: native host object compilation, exit 0. The objects are Mach-O arm64.
- Actual 48-layer emitted plan audits pass. Baseline has 60,362 tasks and 436 events.
  Candidate has 47,690 tasks and 340 events. Its o_proj body names the producer export.
  Its plan has no router or top-k grid. It reuses the existing selector in the producer body.
- Default-off source and kept phase instances match c432c77 byte for byte.

The include maps change only container paths to the same host-mounted source files.
Original sources, native copies, imported bitcode, assembly and host objects are retained as gzip files.
`final-v7/build-artifacts.json` records uncompressed hashes, formats and native resource fields.
The two large plan JSON files are losslessly compressed.
`final-v7/matrix-logs/` retains all 16 variants' unfiltered compiler/plan logs and configurations as gzip.

## Compiler resource limits

ROCm 7.2.4 clang 22 reports 128 VGPRs and 29,736 LDS bytes for both megakernels.
Baseline has 8 spills and 36 scratch bytes per lane.
Candidate has 6 spills and 28 scratch bytes per lane.
Both unfused kernels have 103 VGPRs, zero spills and 29,704 LDS bytes.
These are compiler reports. They are not GPU occupancy or timing measurements.

Native LLVM 23 reports 13,113 and 7,358 in its spill-count field for baseline and candidate.
Those values differ greatly from ROCm clang 22.
They are retained as compiler evidence and are not used as VM resource estimates.
The project already requires the VM compiler check for this reason.
No native LLVM 23 executable was run on a GPU.

## Failures and corrections

The first native CPU arithmetic build failed with Apple's missing C++ standard headers.
The corrected build uses the installed LLVM toolchain.
The first ROCm producer build failed because the combined emitter removed the header include.
The emitter now embeds the actual shared header text.

Two early full-suite attempts were interrupted for source corrections.
One complete run found eight host-dispatch fixture setup errors: 2,041 passed, 17 skipped, 8 errors.
The fixture now supplies the capture buffers. A new test executes the real C++ writer with CPU copies.
It checks bytes, synthetic provenance and overwrite refusal.
A later focused run caught a timing type error. The rule now uses the original decimal log text.
A successful full-suite/matrix run then found container-only include paths in native LLVM compilation.
Only those include paths are now mapped, with source preservation and a mapping receipt.
One later full run caught a home-path dependency in the new test. The test now uses a temporary fixture.

All failed and interrupted raw logs remain in this directory.
The intermediate 2,050-test pass is not presented as the final source result.
The final 2,052-test pass follows every code and test correction.
These failures cost local CPU work only. No paid GPU time was used.
Exact phase exit codes and elapsed times remain in each attempt's `commands.json`.

## Skips

The following are skips, not passes. The full raw output has the same records.

```text
SKIPPED [1] tests/test_acquire_contract.py:459: real build not present: /Users/ckl/code/.wt/etx-producer-scoring/build/rocm22/qwen3-30b-a3b_xcd_qknr (set ETX_CONTRACT_BUILDS)
SKIPPED [1] tests/test_acquire_contract.py:459: real build not present: /Users/ckl/code/.wt/etx-producer-scoring/build/rocm22/mixtral-8x7b_imported (set ETX_CONTRACT_BUILDS)
SKIPPED [1] tests/test_bandwidth_phases.py:64: Round11 acceptance needs ETX_DATA_ROOT with the archived pair
SKIPPED [1] tests/test_codegen.py:50: hipcc not installed
SKIPPED [1] tests/test_coh_model.py:19: measured plan not on this machine
SKIPPED [1] tests/test_coh_model.py:28: measured plan not on this machine
SKIPPED [1] tests/test_coh_model.py:33: measured plan not on this machine
SKIPPED [1] tests/test_floors.py:115: ISA recount needs ETX_ISA_S and ETX_ISA_MAP
SKIPPED [1] tests/test_floors.py:226: set VLLM_SRC to a vLLM v0.27.1 checkout
SKIPPED [1] tests/test_floors.py:240: Round11 floors check needs ETX_DATA_ROOT with the archived pair
SKIPPED [1] tests/test_gpu_jobs.py:210: descendant sweep reads /proc (Linux, the box)
SKIPPED [1] tests/test_importer.py:51: local toolchain unavailable; set ETX_LLVM, ETX_VLLM and ETX_ROCM_HEADERS
SKIPPED [1] tests/test_layer_budget.py:50: Round11 acceptance gate did not run: data set is missing. Set ETX_DATA_ROOT to its parent directory.
SKIPPED [1] tests/test_pairs.py:162: round48 log not present at /Users/ckl/code/.wt/etx-producer-scoring/results_2026-10-01/round48/interleave.log (set ETX_DATA_ROOT)
SKIPPED [1] tests/test_phase_model.py:51: plans not present
SKIPPED [1] tests/test_phase_model.py:57: plans not present
SKIPPED [1] tests/test_vlbest_session.py:348: set ETX_VLLM_SOURCE_ROOT to inject a vLLM source checkout
```

## Pending hardware gates

There is no complete retained real Qwen router input/weight set on this machine.
Actual-model capture and replay are prepared and tested with labelled CPU fixtures.
They are not executed on real Qwen activations yet.
Fresh 97-instance import checks, actual routing/expert data checks, 32/32 tokens,
GPU visibility, progress, elapsed tails, net gain, fusion and S3 comparison are pending.
No production-quality correctness or performance verdict is claimed.

Linux HIP executable linking and loading were not attempted on macOS.
The prepared VM build must perform the real link before any capture.
No fake compiler, replacement kernel or fabricated model capture supplies those missing gates.
The runner requires the committed source receipt and pre-staged model/runtime inputs.
Its persistent lock and budget prevent concurrent runners and silent budget extension.
