# Imported operator specification

`etx/importer/spec.py` defines the 21 imported operators for bf16 decode.
Each record contains the template, constants, supported shapes, grid rule, block rule, argument expressions, layouts, data types, effects, and aliases.
The four MoE records use the reviewed capture data in `etx/importer/specs/moe-decode-v1.json`.
The capture checks bind the constants to the TTIR, LLVM IR, metadata, and binary fingerprints.
`MOE_BN` is an alias for the capture's `BLOCK_SIZE_N`.

The graph builder obtains imported grids from these records.
The build step uses the template list and checks the argument ABI from the compiler.
The build report stores each operator record in `imports.json`.
The graph builder and adapter reject a report that conflicts with the current record.
Resource-only records remain available for simulation tests. They cannot produce a host adapter.

The adapter emits the host descriptors and two argument include files.
The loader uses these files to construct arguments.
It checks model shapes against compiled shapes before it constructs each imported argument block.
`Args::grid` checks the completed block before the host uploads it for execution.
The reference checker uses these same checked blocks.
The Python argument packer applies the same scalar checks.

For Mixtral, the up kernel requires `BLOCK_SIZE_N=16` and 3584 blocks.
The loader rejects the old 896-block grid with `BLOCK_SIZE_N/MOE_BN` in the error message.
For GEMV, `CuCount` must equal the one-dimensional grid size.
The loader rejects `CuCount=304` with a 256-block grid.
These checks do not depend on a difference between the imported kernel and the reference kernel.

The CPU regression test compiles the actual `Args` implementation with the generated header.
It also compiles the old loader logic without the new semantic check.
The old logic accepts both historical faults. The new logic rejects both faults.
The tests accept the Qwen and Mixtral MoE argument blocks.
Launch-record digests preserve the grids from before this change for four models, with XCD replication on and off.

The effects describe logical operator inputs and outputs.
The host still assigns model buffers, layers, and replicas to those inputs and outputs.
This change does not authorize new event removal or buffer reuse.
It does not change kernel arithmetic or replace the independent model reference.
GPU numerics still require the import check and the complete generation check on MI300X.

## Local verification, 2026-10-06

The required interpreter completed the full suite: 275 tests passed and 6 tests skipped.
The first run lacked the optional reference dependencies. Their installation corrected that environment failure.
The final run also passed the local Qwen2.5-0.5B reference test.

All 16 entries in `examples/vllm_llm/local_matrix.sh` passed with local LLVM 23 and the ROCm 7.2.4 device libraries.
Qwen3-8B and Qwen2.5-1.5B passed `base`, `chain`, `xcd`, `xcd_s1`, `uc`, and `uc_s1`.
Qwen3-30B-A3B and Mixtral-8x7B passed `base` and `chain`.
These checks include device compilation, host syntax, and the applicable UC plan checks.
No entry from the requested 12-build set remains unchecked.

Early build attempts used headers from before the last source edit and failed host syntax checks.
A clean final matrix passed. Full logs remain under `/tmp/etx-spec-verified` on the local machine.
No GPU session ran. GPU cost was zero.
