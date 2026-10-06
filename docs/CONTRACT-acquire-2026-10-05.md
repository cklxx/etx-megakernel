# Memory contract: imported pair `attn_reduce` -> `wvsplitk_y2`

Status: static contract with a local checker. No GPU run occurred for this document.
Labels: [M] measured or read from a build file, [E] estimate, [U] unknown.
This document is the local preparation for candidate 2 of `docs/NEXT-2026-10-03.md`.
`docs/DESIGN-acquire-2026-10-05.md` section 5, steps 3 and 5, requires it.

## 1. Scope

The contract covers one imported producer/consumer pair on gfx942:

- Producer: vLLM's `paged_attention` reduce kernel, export `attn_reduce`, task `reduce_L`.
- Consumer: vLLM's `wvSplitK` GEMV, export `wvsplitk_y2`, task `o_proj_L`.
- The pair occurs once in each layer L. The consumer task waits on one DEVICE event. Only the producer tasks of the same layer arrive on that event.

The contract applies to two real builds in `~/code/etx/build/rocm22`:

| build | pairs | producer tasks | consumer tasks | example event |
|---|---|---|---|---|
| `qwen3-30b-a3b_xcd_qknr` | 48 | 4 on domains 0, 1, 3, 4 | 304 on all 8 domains | `E_4`, word 12 |
| `mixtral-8x7b_imported` | 32 | 4 on domains 0, 1, 2, 4 | 304 on all 8 domains | `E_5`, word 6 |

A domain is one XCD. In every pair of both builds, the consumer tasks run on XCDs without a producer task [M].
Thus the tensor `attn_out` crosses XCDs in every pair.

The contract does not cover other imported pairs. It does not cover dynamic tasks, the relay mode, or a DSL tile.
The newer `r34_final` build runs a DSL o_proj tile. That tile does not use this acquire, so the contract does not apply to it.

## 2. Tensors of the pair

The table lists every pointer argument of the two kernels. The checker reads the access kinds from `imports.bc` [M].

| tensor | writer | reader | ordered by | crosses XCDs | access in the body |
|---|---|---|---|---|---|
| `attn_out` | `reduce_L` | `o_proj_L` | DEVICE event of the pair | yes | producer: plain store. Consumer: `llvm.memcpy` from global memory into LDS, 16 B |
| `exp_sums`, `max_logits`, `tmp_out` | attention tasks `attn_L` | `reduce_L` | the attention event, outside this contract | not checked | plain loads |
| `seq_lens` | host, before the step | `reduce_L` | kernel launch | no device writer | plain load |
| `wo` weights | host, before the run | `o_proj_L` | kernel launch | no device writer | nontemporal loads |
| `o` | `o_proj_L` | the next phase | the next event, outside this contract | not checked | plain store |
| bias, `ptr5`, `ptr7` | none | none | not applicable | not applicable | the host passes null. The bodies compare each pointer with null before its loads |

`attn_out` is one buffer of `NH * HD` bf16 values. Both models have 32 heads of dimension 128, so the buffer has 8192 B.
All layers use the same buffer. The attention kernel also receives `attn_out` as argument 16, but its body does not access that argument.

## 3. Event order

The emitted megakernel runs each task in three phases. The order below comes from `megakernel_d0.hip`, `etx_lowering.h` and `primitives.h` [M].

1. Every wave of the producer workgroup writes its part of `attn_out` with plain stores.
2. Every wave executes `s_waitcnt vmcnt(0)`. This is `ETX_RELEASE_DOMAIN`, the all-wave drain.
3. `__syncthreads()` makes all waves reach the same point.
4. Thread 0 executes `buffer_wbl2 sc1` and `s_waitcnt vmcnt(0)`. The writeback moves dirty L2 lines of this XCD to memory.
5. Thread 0 decrements the event word with a relaxed atomic at agent scope.
6. Thread 0 of the consumer workgroup polls the event word with a relaxed atomic load at agent scope. It stops when the count is 0.
7. Thread 0 executes `buffer_inv sc1`. This is `ETX_ACQUIRE_DEVICE`. It invalidates the L1 cache of the CU and the L2 cache of this XCD.
8. `__syncthreads()` releases all waves of the consumer workgroup.
9. The consumer body loads `attn_out` from global memory.

The real builds use an arrive share of 1. Thus each producer workgroup does steps 4 and 5 itself.
If the share is above 1, the runtime counts arrivals per domain first. Only the last producer of the domain does steps 4 and 5.
The checker accepts this path only if the plan places the same number of producers on each domain.

The macros in step 2, step 4 and step 7 agree with the agent-scope code sequences for gfx942 in the LLVM AMDGPU memory model.
This document did not check that correspondence against hardware.

## 4. The hazard that the acquire prevents

All layers reuse the same `attn_out` addresses. The consumer of layer L-1 loaded these addresses into the L1 and L2 caches of its XCD.
Without step 7, the consumer of layer L can read the value of layer L-1 from a cache line.
This read gives a wrong result and no error.
Step 7 removes those lines. The loads in step 9 then miss in the caches and read the value that step 4 wrote back.

A transformed consumer can remove step 7 only if no load of `attn_out` can hit a stale cache line.
The candidate in `docs/DESIGN-acquire-2026-10-05.md` uses coherent `sc0 sc1` loads for this purpose.
A transformed consumer must also keep these conditions:

- The consumer body reads no other device-written tensor. In this pair, `wo` is read-only and the bias is null.
- No other task writes `attn_out` between the arrive and the consumer loads.
- The pre-wait hook reads only the argument block.

## 5. Two accepted risks

**Aborted wait.** The wait loop returns early if the host or the spin limit sets the abort flag.
The consumer then runs its body on data that can be stale.
The host treats the step as failed in `etx/runtime/include/etx/host.h`. Thus no output of that step becomes a result.

**Pre-wait touch.** Before the wait, `etx_pre_wvsplitk_y2` reads 128 B of the argument block through the scalar cache.
The host writes the argument block before the launch. Thus the touch does not depend on the event.
It does not read `attn_out`.

## 6. Forbidden compile flags

Some macros remove a fence of section 3 at compile time. They leave no trace in the build directory.
Thus the checker requires the device compile flags of the build as input.

| flag | effect |
|---|---|
| `-DETX_COHERENT_DEVICE_DATA` | removes the acquire. It assumes that every cross-XCD tensor is uncached. `attn_out` is cached. |
| `-DETX_VL_UC` | selects the uncached plan of the same assumption |
| `-DETX_NO_ALLWAVE_DRAIN` | removes step 2 |
| any other `-D` macro outside the allowlist | unknown effect |
| `-U`, `-include`, `-imacros` | can change a lowering macro |

The allowlist is `ETX_DSL_NT`, `ETX_DSL_VECNORM`, `ETX_VL_CHECK`, `ETX_DSL_PREP_EARLY`, `ETX_DSL_UNIFORM`, `ETX_NO_PRE_WAIT`, `ETX_NO_TOUCH` and `ETX_BACKOFF_STMT`.
Other flags such as `-O3` pass.

## 7. What the checker verifies

`etx/tools/acquire_contract.py` checks a build against this contract. Any unknown item is a violation.

1. **IR.** The tool reads `imports.bc` with `llvm-dis`. It checks the offsets and types of the wrapper arguments.
   It checks the `readonly`, `writeonly`, `readnone` and `noalias` attributes of every specialization body.
2. **IR allowlists.** Every opcode and every callee must be on a list. Volatile accesses, atomics and inline assembly fail.
   Only address spaces 1 and 3 are permitted. The only fences permitted have workgroup scope.
3. **Pointer origins.** Every global access must come from one parameter through `getelementptr`, `phi` or `select`.
   A pointer that the body loads from memory fails.
4. **Access kinds.** Each parameter must show exactly the access kinds of section 2. For example, a plain load of `wo` fails.
5. **Bindings.** In `etx/importer/spec.py`, only attention argument 16 and reduce argument 0 name `attn_out`.
6. **Emitted source.** For each layer, the source must match the templates of section 3 line by line.
   The producer and the consumer must use the same event and word. No other task may arrive on that word.
7. **Plan.** The event must have DEVICE scope, and its count must equal the number of producer tasks.
   Each consumer task must wait on that word alone. Both task types must be static.
8. **Lowering and runtime.** The DEVICE macros must have the exact text for gfx942. No relaxed DEVICE acquire may exist.
   The texts of the arrive function, the wait loop, the touch function and the pre-wait hook must match the contract.
9. **Flags.** The flags of section 6.

The test file `tests/test_acquire_contract.py` mutates each item and requires a failure.
A sweep removed each of the 36 guards of the checker one at a time. At least one test failed for each removal [M].

## 8. Limits

- The checker is static. It compares emitted files with this contract. It does not prove the GPU memory model.
- Steps 4 and 7 are single-thread cache operations. They cover the other waves through the hardware caches and the barrier.
  The LLVM memory model specifies fences for each thread, so it does not describe this pattern [U].
- The checker does not prove that step 7 completes before the barrier releases the other waves [U].
- The checker does not check the order of the consumer reads before the next write of `attn_out` in layer L+1. The dependency chain of the plan provides that order. No tool checks it.
- The checker examines the IR, not the gfx942 ISA. An ISA check is step 4 of `docs/DESIGN-acquire-2026-10-05.md` section 5.
- The contract does not include a measurement of stale reads. The microbenchmark of the design document must supply that measurement.

## 9. How to run

The flags must be the device flags of the build. `examples/vllm_llm/build.sh` uses `-O3 -DETX_DSL_NT -DETX_DSL_VECNORM` by default.
Use the `--cflags=` form, because `argparse` reads a separate value that starts with `-D` as an option.

```
python -m etx.tools.acquire_contract --build ~/code/etx/build/rocm22/qwen3-30b-a3b_xcd_qknr --cflags="-O3 -DETX_DSL_NT -DETX_DSL_VECNORM"
```

Exit codes: 0 means PASS. 1 means FAIL, with one `VIOLATION` line for each violation. 2 means REFUSED because an input is absent.
A FAIL means: keep the acquire, and do not build the transformed pair.

Results on 2026-10-05 [M]:

| build | result | pairs | pairs across XCDs |
|---|---|---|---|
| `qwen3-30b-a3b_xcd_qknr` | PASS | 48, layers 0-47 | 48 |
| `mixtral-8x7b_imported` | PASS | 32, layers 0-31 | 32 |
| `qwen3-30b-a3b_xcd_qknr` with `--cflags="-DETX_COHERENT_DEVICE_DATA"` | FAIL, 1 violation | not applicable | not applicable |

`tests/test_acquire_contract.py` runs the two real builds from `ETX_CONTRACT_BUILDS`. The default is `~/code/etx/build/rocm22`.
If a build is absent, the test skips.
