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

## Block ranges and plan scope, 2026-10-06

The specification now separates launcher defaults from kernel limits.
Each block axis has inclusive lower and upper bounds.
Equal bounds require an exact value.
Python checks these bounds. The host emitter uses the same bounds to produce C++ checks.
A regression test changes one rule and checks both paths.

The following operators accept `block.x` from 1 through 1024, with `block.y=block.z=1`:

- `rms_norm_2d` and `rms_norm_3d`
- `fused_add_rms_norm`
- `silu_and_mul`
- `rope_neox`
- `moe_sum_k8` and `moe_sum_k2`

The kernel loops advance by `blockDim.x`.
The norm kernels also pass `blockDim.x` to a reduction with capacity for 1024 threads.
Thus, the launcher default does not define the only legal block size.
For example, `rms_norm_2d` accepts 1024 threads at `H=16384`.
The old rule required 2048 threads.
The vLLM launcher applies the cap in `csrc/libtorch_stable/layernorm_kernels.cu`, lines 255–256 and 269–271.
The kernel loops and reduction appear at lines 65–81 and 133–151.
The activation, rotary, and MoE sum loops appear at these locations:

- `csrc/libtorch_stable/activation_kernels.cu:120`
- `csrc/libtorch_stable/pos_encoding_kernels.cu:54` and line 65
- `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:385`

`reshape_and_cache` requires at least `NKV*16` threads for the supported head size of 128.
It accepts extra threads up to 1024. Each extra thread returns without a write.
See `csrc/libtorch_stable/cache_kernels.cu`, lines 274–282.
A smaller block omits head segments, so the lower bound remains necessary.
The other block rules retain the block shapes for their compiled templates, fixed grid mapping, or capture metadata.
Legal block ranges do not establish numerical equality with a different launch block.
The import check and full model check remain necessary before a performance claim.

The graph builder checks only the exports that the plan uses.
A missing export stops the build with its name in the error message.
The builder checks unused entries separately and records each result in `<plan>.imports-check.json`.
An unused entry cannot stop the graph build because its block or model specification is stale.
This permits a merged `imports.json` from a partial `--only` harvest.
The adapter still checks descriptor integrity before it emits all requested adapters.
Old recorded block guidance does not change the compiled ABI.
The import check uses the current bounds for the actual descriptor and still checks the ABI, templates, and capture constants.

The `BLOCK_SIZE_N/MOE_BN` and `CuCount` guards remain unchanged.
The tests still reject the two Mixtral faults before execution and name the incorrect parameter.

## Deliberate partition limit

The `MAXLEN=17000` rejection remains. The code before the operator specification accepted this value.
This is a deliberate behavior change for the imported `attn_reduce` specialization.
It has `NPAR_LOOPS=1`, a wave size of 64, and `PARTITION_SIZE=256`.
It supports at most 64 partitions, or 16384 tokens.
`MAXLEN=17000` requires 67 partitions.

The checked vLLM revision is `f46a9dfe2c5f57bebbd29556cbbb25eabd874226`.
In [`csrc/rocm/attention.cu:2958`](https://github.com/vllm-project/vllm/blob/f46a9dfe2c5f57bebbd29556cbbb25eabd874226/csrc/rocm/attention.cu#L2958),
the source states the limit as `warp_size * NPAR_LOOPS`.
Line 2959 allocates the shared array with that size.
At [lines 3352–3361](https://github.com/vllm-project/vllm/blob/f46a9dfe2c5f57bebbd29556cbbb25eabd874226/csrc/rocm/attention.cu#L3352),
the vLLM launcher selects `NPAR_LOOPS=2` when the partition count exceeds 64.
Thus, this limit applies to the imported specialization. It is not a general vLLM context limit.
Python and generated host tests accept 16384 tokens and reject 16385 and 17000 tokens.

## Verification of the range change

Ten new regression cases failed before their fixes. All now pass.
The focused suite passed 52 tests.
The full suite passed 346 tests and skipped 6 tests with `/Users/ckl/code/etx/.venv/bin/python`.
All 16 entries in `examples/vllm_llm/local_matrix.sh` passed with LLVM 23.1.1 and `ETX_DEVLIBS` set to the ROCm 7.2.4 device libraries.
These entries include all four models and all variants in the script.
The matrix checked device compilation, host syntax, and the applicable UC plans.

Full matrix logs remain in `/tmp/etx-spec-ranges-matrix`.
The suite log is `/tmp/etx-spec-ranges-pytest.log`.
The regression logs before the fixes are `/tmp/etx-spec-red.log` and `/tmp/etx-spec-old-guidance-red.log`.
No GPU session ran. GPU cost was zero.
