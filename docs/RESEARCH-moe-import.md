# Research: importing vLLM's MoE path into ETX

Status: research only (no code yet) · Date: 2026-09-26 · Model: Qwen3-30B-A3B (128 experts, top-8, hidden 2048, expert intermediate 768), MI300X, batch 1, vLLM 0.27.1 ROCm build (AITER off, its fastest configuration: 4.84 ms/token)

## 1. What vLLM runs for one MoE layer at batch 1

The ROCm unquantized MoE backend without AITER is `TritonExperts` in `fused_moe/oracle/unquantized.py`.
At batch 1, `num_tokens * top_k * 4 = 32 <= 128 experts`.
Thus, `_prepare_expert_assignment` selects **naive block assignment**, without a `moe_align_block_size` kernel.
The block-to-expert map uses the top-k expert ids directly.

| Step | Kernel | Kind | Launch at batch 1 | ETX status |
|---|---|---|---|---|
| router logits | `wvSplitK_hf_sml_` (YTILE 1, M = 128, K = 2048) | HIP | 304 blocks × 1024 threads | imported (`wvsplitk_y1`) |
| top-k + softmax + renormalise | `topkGating<VPT, 128, 4, ...>` (`libtorch_stable/moe/topk_softmax_kernels.cu`) | HIP | 1 block × 256 threads (64 × 4) | importable with the existing pipeline (one manifest entry) |
| experts gate_up | `fused_moe_kernel` (`fused_moe.py`), `MUL_ROUTED_WEIGHT=False`, `top_k=8` | **Triton** | 8 × 1536 / 64 = **192 programs** × 256 threads | needs the Triton path |
| activation | `act_and_mul_kernel` (silu) | HIP | 8 rows | imported (`silu_and_mul`) |
| experts down | `fused_moe_kernel`, `MUL_ROUTED_WEIGHT=True`, `top_k=1` | **Triton** | 8 × 2048 / 64 = **256 programs** × 256 threads | needs the Triton path |
| sum over the 8 experts | `moe_sum` (`libtorch_stable/moe/moe_align_sum_kernels.cu`) | HIP | 1 block × min(2048, 1024) threads | importable (one manifest entry) |

Triton has no tuned bf16 file for `E=128,N=768` on MI300X, only MI308X and fp8 variants.
Thus, `get_default_config` selects `BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2`.
Each program uses MFMA for a 16 × 64 tile, with the token in one row.
The kernel uses only `tl.program_id(0)`, without `num_programs`, and returns early per program.
These properties suit the importer.

ETX already handles four of the six launches per MoE layer as HIP kernels.
The two expert GEMMs use two specialisations of one Triton kernel.

## 2. Getting the Triton kernel's IR

- The rocm/vllm image builds Triton from ROCm's fork, `release/internal/3.7.x` at `0263a6a`. That Triton bundles **LLVM 23** (llvm-project `1f126a6`, June 2026).
- The Triton cache contains the exact IR that vLLM executed after a vLLM run.
  Its path is `<TRITON_CACHE_DIR>/<hash>/fused_moe_kernel.{llir,amdgcn,hsaco,json}`.
  The `.json` contains `num_warps`, shared-memory size, and the constexpr signature.
  Capture requires one vLLM run with `TRITON_CACHE_DIR` set to a retained directory, or a few minutes of GPU time.
  `vm_run.sh` includes this step.
- Local compilation is possible in principle. Triton can compile for `gfx942` without a GPU through the LLVM IR stage.
  But a local Triton build would differ, so its IR would not be the kernel that vLLM executes.
  Use local compilation only for importer development.

## 3. What the importer needs for Triton kernels

| Gap | Why | Change |
|---|---|---|
| Dynamic LDS | Triton kernels use one `@global_smem = external addrspace(3) global [0 x i8]`<br> the size is in the cache `.json` | accept one dynamic LDS global with an explicit size and map it into the arena like static LDS |
| Extra parameters | recent Triton appends scratch pointers (global and profile scratch) to every kernel | pass null / a small buffer<br> the argument block layout already comes from the IR signature |
| Intrinsic ids | `tl.program_id` lowers to `llvm.amdgcn.workgroup.id.x`, threads to `llvm.amdgcn.workitem.id.x` | already handled (the intrinsic form) |
| Buffer intrinsics, `readfirstlane`, `sched_barrier` | Triton's AMD backend uses raw buffer loads built from kernel arguments | nothing to do: plain IR, no dispatch state |
| **Toolchain version** | Triton's IR is LLVM 23<br> ROCm 7.2.4's hipcc is LLVM 22 and cannot be relied on to read LLVM 23 IR | build the megakernel's device code with LLVM 23 (upstream clang with the ROCm 7.2.4 headers and device libraries, the toolchain `etx/importer/hip_ir_local.sh` already uses) into a code object, and load it with `hipModuleLoad` instead of linking it into the host binary |
| **Register budget** | Triton compiles `fused_moe_kernel` for 256 threads, where it may use more than 128 VGPRs<br> inside a 1024-thread megakernel every tile gets 128 | measure its VGPR count from the cached `.hsaco`<br> if it spills, use a 512-thread megakernel for MoE (256 VGPRs), which needs `wvSplitK` instantiated with 8 waves per group instead of vLLM's 16 |

The toolchain change is the largest item. The HIP imports are unaffected by it: the same LLVM 23 compiles them locally today, with 0 spills for the Qwen3-8B set.

## 4. Why MoE is worth it, and the risk

- vLLM reads about 6.1 GB per token in 4.84 ms, about 26% of HBM bandwidth against about 60% for dense Qwen3-8B.
  The difference comes from small launches around experts and 16-row tiles that process one row.
  There are six launches per MoE layer across 48 layers.
- This is where a megakernel should gain the most: router, top-k, the two expert GEMMs and the sum depend on each other through a handful of values, and none of them needs the whole GPU.
- The risk matches the dense results: if cross-chiplet synchronization costs more than a kernel boundary, more small launches increase the loss.
  Measure dense `ETX_VL_XCD` first to determine whether the synchronization problem is solved.

## 5. Plan and cost

| Step | Where | Effort |
|---|---|---|
| Import `topk_softmax` and `moe_sum` (manifest entries), local compile check | local | small |
| Importer: dynamic LDS, scratch parameters | local | small |
| LLVM 23 device build + `hipModuleLoad` path in the ETX host runtime | local, verified on GPU | medium |
| Capture vLLM's Triton cache and kernel trace for Qwen3-30B-A3B | GPU, about 20 min (download 61 GB, one vLLM run) | small |
| Import the two `fused_moe_kernel` specialisations, `--check` them against vLLM, then the full model | GPU, about 1-2 h | medium |

Measure the dense synchronization variants first with `examples/vllm_llm/vm_next.sh`.
Add Triton capture to that session, so MoE work can continue locally afterward.
