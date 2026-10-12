# Fleet F/W port to ETX — 2026-10-12

**Implementation complete; GPU verdict unmeasured.** F is an off-by-default per-XCD expert-fold path. W is an off-by-default, read-only ETX queue audit; the retained trace supplies no critical GQA worker blockage, so there is no W permutation or W timing arm. No GPU was provisioned or run for this work; GPU cost was $0. Local compiler results are resource measurements, not decode performance. Implementation and reproduced command outputs follow.

## Implementation and contract

`-DETX_DSL_LOCAL_FOLD` (or `=1`) enables both tile emission and graph wiring. Absent, `=0`, or `-UETX_DSL_LOCAL_FOLD` leaves it off. The parser reads both `ETX_CFLAGS` and Docker's `CFLAGS`, rejects conflicting settings, and the emitted device source rejects a missing compiler define. `ETX_DSL_LOCAL_FOLD_VERIFY=1` is an additional compiler diagnostic, off by default, meaningful only with F enabled. `kept_build.py` and its defaults were not changed. (`etx/ops/local_fold.py:14`, `etx/ops/local_fold.py:74`, `etx/ops/local_fold.py:126`; emission hook `etx/ops/gemv.py:991`; model hooks `examples/vllm_llm/model.py:651`.)

F targets **qkv layers 1–47 and final lm_head**, adding **384 tasks**, **384 count-one local event words**, and **65,536 bytes** of scratch: eight private 2,048-float vectors. Every assembler is pinned to worker slot 0 of its XCD, waits for the original 256-producer global expert event, then publishes its own local event. Consumers keep their 256-task geometry and use their logical XCD's vector. Existing P5 pins do not advance the round-robin cursor; the original consumers' worker assignments remain unchanged. These are structural counts, not runtime observations. (`etx/ops/local_fold.py:151`, `etx/ops/local_fold.py:169`; `etx/passes/p5_queues.py:67`; build output: `tasks=60746 events=484`, versus `tasks=60362 events=436` in `build/foldlocal_on_final.log:55` and `build/foldlocal_off_final.log:55`. Here `events` counts event tensors, not words.)

The generator takes the **actual original fold source**, rather than implementing a different sum: fp32 accumulation from zero in slot order 0…7, bf16 rounding of the expert sum, then bf16 rounding after adding the residual. It also reuses the original norm suffix and retains the per-lane square accumulation order. The fp32 scratch carries already bf16-rounded values; weights, activations and all original bf16 boundaries remain bf16. Only assembler XCD 0 writes the canonical residual; final lm_head's residual output remains null. (`etx/ops/local_fold.py:38`, `etx/ops/local_fold.py:81`; original arithmetic `etx/ops/gemv.py:547`, `etx/ops/gemv.py:584`, `etx/ops/gemv.py:628`, `etx/ops/gemv.py:660`; residual ownership `examples/vllm_llm/host.hip:427`.)

The scratch lifetime follows the unchanged global graph: all qkv readers finish before subsequent attention/router/MoE work permits the next assembly; all final-head readers finish before argmax and the next step. qkv reads resid2 and writes resid, router reads resid and writes resid2, and the head writes neither. F does not target router's **R=1** fold or o_proj. (`examples/vllm_llm/host.hip:427`, `examples/vllm_llm/host.hip:437`, `examples/vllm_llm/host.hip:440`; graph transformation `etx/ops/local_fold.py:142`.)

**Visibility deliberately follows ETX's measured protocol.** Assemblers use the original coherent operand loads, cached private-vector stores, and the runtime's all-wave drain/barrier before local publication. Consumers perform `ETX_ACQUIRE_DOMAIN()` once plus a workgroup barrier before cached staging. ETX's default is **sc1**, not fleet's sc0: ETX measured stale reused 8-KiB payloads with sc0. The feature rejects `ETX_ACQUIRE_DOMAIN_RELAXED` and `ETX_NO_ALLWAVE_DRAIN`. Coherent metadata is retained for the ordinary coherent inputs/outputs; the cached scratch exception has its explicit acquire inside the body. This retains a real local event, but **does not promise fleet's L1-only cache benefit**; the sc1 invalidation and extra join are costs. Logical coordinates also address scratch in the same-kernel unfused launches, whose physical XCD placement is unconstrained. (`etx/ops/local_fold.py:95`, `etx/ops/local_fold.py:116`, `etx/ops/local_fold.py:132`; `etx/machine/arch/gfx942.yaml:27`; `etx/codegen/kernel.py:333`; `etx/codegen/kernel.py:561`; fleet acquire: `../fleet-mi300x/src/runtime/fleet_runtime.h:233`.)

The optional verification path stages the actual published vector **before** recomputing the coherent original fold, compares every float bit, and traps on mismatch without repairing the staged vector. It is excluded from timing. It has only been compiled locally, not run on a GPU. (`etx/ops/local_fold.py:67`.)

**Host-check integration:** the host's sequential import checker launches instance `n` at launch `n+1`. Each new assembler therefore gets an explicit, commented argument-layout alias in `vl_plan.txt`, sharing the following consumer's role/export/geometry/offset. Host setup writes the same argument block twice; actual assembler geometry is eight tasks in `plan.json`/`plan_data.h`. This preserves the checker sequence and shifts imported instance IDs consistently. The DSL check's name-based lookup resolves the actual consumer. No host or runtime file was changed. (`etx/ops/local_fold.py:189`; `examples/vllm_llm/model.py:658`; `examples/vllm_llm/host.hip:608`, `examples/vllm_llm/host.hip:647`, `examples/vllm_llm/host.hip:927`; `examples/vllm_llm/import_certificate.py:101`.)

## Ceiling before a performance claim

Historical ETX account: Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X VF, KEY32-on trace. All entries below are **microseconds on the historical critical path**, not a new measurement or a current-baseline prediction. (`docs/MASTER-TABLE-2026-10-11.md:3`, `docs/MASTER-TABLE-2026-10-11.md:120`.)

| Role | Body [M] | Weight bytes / 4.3 TB/s [E] | Hops [E] | Machinery [R] | Gap [M] |
|---|---:|---:|---:|---:|---:|
| qkv | 433.2 | 234.1 | 20.0 | 179.0 | 183.2 |
| lm_head | 151.9 | 144.7 | 0.0 | 7.2 | 1.7 |
| router, excluded | 265.4 | 5.9 | 49.0 | 210.5 | 129.2 |
| o_proj, excluded | 263.5 | 187.3 | 20.0 | 56.2 | 96.7 |

Source: `docs/MASTER-TABLE-2026-10-11.md:128`, `docs/MASTER-TABLE-2026-10-11.md:132`, `docs/MASTER-TABLE-2026-10-11.md:133`, `docs/MASTER-TABLE-2026-10-11.md:137`.

The broad containing interval is **186.2 us machinery + 184.9 us gaps = 371.1 us**. The retained weight-byte floor is **378.8 us**, plus 20.0 us of modeled qkv hops; F does not change the weight stream. The narrower historical qkv body-opportunity estimate was only **73 us**. These are ceilings containing several mechanisms, not amounts F can recover. The qkv row includes layer 0, which F does not target. (`docs/MASTER-TABLE-2026-10-11.md:80`, `docs/MASTER-TABLE-2026-10-11.md:128`, `docs/MASTER-TABLE-2026-10-11.md:137`.)

**Removable by this scheme alone:** duplicate expert/residual fetches and ordered fold additions in QK2/LM1. Their exclusive duration is **[U]**. Norm reduction, norm weights/scale/rounding, canonical residual publication, weight streaming, and the original global completion remain; local assembly, publication, acquisition and staging are added. No part of the old **184.9-us gaps is automatically removed**, and zero saving is booked. Norm hoist and router narrowing are already kept, so neither is a new lever or second credit. The old trace predates that refresh; its raw refreshed timing records were lost. (`docs/MASTER-TABLE-2026-10-11.md:185`, `docs/MASTER-TABLE-2026-10-11.md:317`; `examples/vllm_llm/kept_build.py:5`, `examples/vllm_llm/kept_build.py:16`; `docs/BASELINE-STATE-2026-10-12.md:15`, `docs/BASELINE-STATE-2026-10-12.md:19`.)

Logical traffic arithmetic **[E, not HBM traffic]**, per targeted fold: original `256 × 36,864 = 9,437,184 B`; F `8 × 36,864 + 8 × 8,192 + 256 × 8,192 = 2,457,600 B` for assembler input, private stores and consumer staging. Difference **6,979,584 B**; unchanged norm-weight and residual traffic is omitted from both sides. Dividing that logical difference by HBM bandwidth would not establish a saving. (`docs/MASTER-TABLE-2026-10-11.md:186`, `docs/MASTER-TABLE-2026-10-11.md:196`; `etx/ops/local_fold.py:54`, `etx/ops/local_fold.py:95`.)

Fleet's old global last-writer fold serialized about **4 us at each fold point**. F differs because eight concurrent per-XCD assemblers consume the existing global completion and add only local publications; no globally last writer assembles a single vector for everybody. Nevertheless, each local assembly/publication/stage tail can lose, especially with ETX's sc1 acquire and the spills below. Fleet's **259.7-us pool is not used**. (`../fleet-mi300x/docs/STATUS.md:67`; `../fleet-mi300x/results/first-two-2026-10-12.md:7`; `../fleet-mi300x/results/lever-space-2026-10-12.md:70`.)

## W: dependency-to-queue audit, then stop

The real body-preserving decision is P5's unpinned round-robin **worker slot within an existing domain**. The head-local prep/attention/reduce branch fixes domains and dependencies but has no worker pin. The audit validates every descriptor/trace identity, complete static queues, timestamps and GQA topology, then reconstructs ETX's critical path and separates preceding-worker body blockage from unstamped arrive/fetch delay. It does not mutate bodies, queues or edges. (`etx/passes/p5_queues.py:67`; `examples/vllm_llm/model.py:589`; `etx/ops/worker_audit.py:31`, `etx/ops/worker_audit.py:78`, `etx/ops/worker_audit.py:114`.)

Command actually run:

```sh
ETX_DSL_WORKER_AUDIT=1 .venv/bin/python -m etx.ops.worker_audit \
  results_vlbest2_2026-10-10/b_all/plan.json \
  results_qwen300_2026-10-11/tr_on.bin
```

Output: **`STOP_NO_CRITICAL_QUEUE_BLOCKING`**, 60,362 identities checked, zero mismatches; 304 workers; 436 critical tasks; **4067.29 = 3191.80 body + 875.49 gap us**; zero critical queue predecessors. (`build/rocm22/foldlocal_worker_audit.json:3`.)

| GQA role | Critical tasks | Critical body us | Critical gap us | Preceding-worker body delay us | Late-taken us |
|---|---:|---:|---:|---:|---:|
| prep | 48 | 194.80 | 81.09 | 0 | 0 |
| attn | 48 | 259.72 | 52.52 | 0 | 0 |
| reduce | 48 | 254.28 | 56.56 | 0 | 0 |

Source: `build/rocm22/foldlocal_worker_audit.json:12`. Qualified saving **0 us**; **no W permutation or GPU W/FW arm**. This is a historical screen, not proof about the refreshed baseline or F graph. Flag absent/`0` returns `OFF` without reading inputs. (`etx/ops/worker_audit.py:147`, `etx/ops/worker_audit.py:167`.)

## Local gates

The baseline includes the brief's entire flag set **and** production's default NT/VECNORM flags, plus already-kept router narrowing. `rocm_build_check.sh` directly invokes clang with `$CFLAGS`, whereas production `build.sh` adds NT/VECNORM separately; omitting them would test the wrong norm/stream branch. (`examples/vllm_llm/build.sh:64`, `examples/vllm_llm/build.sh:68`; actual clang argv `build/foldlocal_on_final.log:58`, `build/foldlocal_off_final.log:58`.)

Commands actually run, with absolute output directories under `$HOME/code`:

```sh
BASE='-DETX_DSL_NT -DETX_DSL_VECNORM -DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH -DETX_INBODY_REPORT -DETX_DSL_NORM_HOIST -DETX_TOPK_KEY32'
ETX_VL_ROUTER_NARROW=1 ETX_CFLAGS="$BASE -DETX_DSL_LOCAL_FOLD" \
  bash examples/vllm_llm/rocm_build_check.sh qwen3-30b-a3b base \
  /Users/ckl/code/etx/build/rocm22/foldlocal
ETX_VL_ROUTER_NARROW=1 ETX_CFLAGS="$BASE" \
  bash examples/vllm_llm/rocm_build_check.sh qwen3-30b-a3b base \
  /Users/ckl/code/etx/build/rocm22/foldlocal_off
```

The final runs enabled shell xtrace **inside** the Docker shell; full, unfiltered logs retain the actual device compiler invocation, including all ten F-on defines. Both device compilation and `host.hip`'s device pass succeeded. Resources below are ROCm **7.2.4 clang 22**, gfx942, local x86 container, not hardware residency. (`build/foldlocal_on_final.log:58`, `build/foldlocal_on_final.log:62`, `build/foldlocal_on_final.log:67`; corresponding off log.)

| Build / function | VGPR | AGPR | Scratch B/lane | VGPR spills | LDS B/block | Waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|
| F off, megakernel | 128 | 0 | 36 | 8 | 29,736 | 4 |
| F on, megakernel | 128 | 0 | 44 | 10 | 29,736 | 4 |
| F off/on, unfused | 103 | 0 | 0 | 0 | 29,704 | 4 |

Source: `build/rocm22/foldlocal_off/resource.log:5`, `build/rocm22/foldlocal/resource.log:5`. **F adds two VGPR spills and 8 B/lane scratch**; do not infer unchanged occupancy cost from the same nominal four waves. The additional VERIFY build is diagnostic only; its local build result is recorded below when complete.

**Off-path identity:** before modifying source, the kept build was emitted at `foldlocal_off` and copied to `foldlocal_off_snapshot`. Rebuilding in the same original path avoids path-name differences in emitted assembly. A byte comparison found `dsl_tiles.hip`, `imp_tiles.hip`, `megakernel_d0.hip`, `plan.json`, `plan_data.h`, `vl_plan.txt`, `imports.bc`, `resource.log`, and `mk.s` identical. The device assembly was also assembled with ROCm clang into ELF objects; `cmp` passed. This is local device-object identity, not a claim about a production linked executable. (`build/foldlocal_object.log:1`.)

```text
off dsl_tiles.hip SHA256 704d808ed572fcbb04e5186e15b7806cc11ee17561a3310c684ca00855c8a7ee
off resource.log  SHA256 3b32167c6efd65620cb7fc516bd5b3d020423fc0bc67cb8f31d26d90afd4a505
off mk.s          SHA256 9f675e40792a2518582e99d0792bc89cd7d68e531b94d43c6fe9c1aadbac3a2d
off mk.o          SHA256 7f2808151f878b6fefd3b2f68867b50c13b199ad0552bdb5eeb42357c9b38729
```

Test results: pending final local test completion; no GPU correctness certificate has been claimed.

## Registered next GPU experiment — proposal, not executed

Contract: one MI300X VF, bf16 Qwen3-30B-A3B, batch 1, context 1024, the full 32-token greedy sequence, same GPU and session for every comparison. **B** is freshly rebuilt kept code (router narrowing, norm hoist and KEY32 on); **F** is B plus `ETX_DSL_LOCAL_FOLD`; **V** is F plus `ETX_DSL_LOCAL_FOLD_VERIFY=1`, correctness only. **G(B)/G(F)** replay each arm's exact kernels through its own HIP graph; G(F) includes the assembler launches. **S3** is vLLM `ROCM_AITER_UNIFIED_ATTN` with the pinned legal configuration. No closed tile route or extra workgroup is included. (`examples/vllm_llm/kept_build.py:5`; `examples/sessions/qwen300.py:187`; `etx/codegen/plan_header.py:83`.)

1. **Correctness first, B then V then F.** Fresh same-build import checks at decode positions 0 and 1040; 97/97 imported instances, not a transferred certificate. Full free-running 32/32 HF tokens, `CHECK-DSL OK`, then unfused 32/32 for each identity. V additionally compares every consumer's staged vector to the ordered reference under real decode traffic. Repeat V three times, then retain no V timings. Existing certified command form is `python3 -m examples.vllm_llm.import_certificate --run-build BUILD --model MODEL -- --check 48 --check-pos POS`; full gates use `--repeat 1 --check-dsl` and `--repeat 1 --unfused`. (`examples/vllm_llm/import_certificate.py:72`; `examples/sessions/qwen300_hour.py:588`.)
2. **Short deciding screen before the full timing campaign.** One fresh traced B/F pair, with the exact newly emitted plan for each trace. Compare original expert-producer completion through the last corresponding fold consumer's completion, charging the entire new assembly/event/acquire/stage path and its retained GEMV. This existing task-trace metric is broader than norm-ready; it must not be mislabeled as a measured isolated fold or norm-stage saving. If it does not shorten, stop F. Current task stamps do not isolate norm-ready; that finer decomposition remains [U]. (`etx/codegen/kernel.py:323`, `etx/codegen/kernel.py:343`; `etx/tools/timeline.py`.)
3. **F must beat B:** five complete adjacent pairs, order **BF, FB, BF, FB, BF**, untraced, `--repeat 1` full 32-token runs. Define each saving as B host-wall minus F host-wall. Require every saving positive, median **at least 0.030 ms**, and median **strictly greater than twice `(max saving − min saving)`**. Missing rounds, any correctness/residency failure, failed trace screen, or a missed threshold ends the attempt; leave F off. The 0.030-ms rule is the existing ETX KEEP rule, extended here from three to five pairs. (`examples/sessions/qwen300_hour.py:305`.)
4. **Both required baselines:** five B/G(B) pairs and five F/G(F) pairs, alternating fused/graph order within each series. F's fusion must return `ESTABLISHED` from `pairs.py`: every relative host-wall gain positive, median **>2%** and **>2×spread**. Also five adjacent F/S3 pairs, alternating order; report all deltas and claim LEAD only if every F−S3 delta is negative and median ≤−0.050 ms. Failure is reported even if F beats B. (`examples/vllm_llm/pairs.py:138`; `examples/sessions/qwen300_hour.py:311`.)

Budget **[E]**, using the repository's historical **$1.99/GPU-hour**, not a current rental quote: 10 minutes checks, 5 minutes trace screen, 10 minutes B/F pairs, 25 minutes fusion/system pairs, 10 minutes margin = **60 warm minutes, $1.99**. Allow at most 15 minutes cold setup (**$0.50**): absolute cap **75 minutes / $2.49**. If checks or measurements cannot finish inside their slice, stop with NOT-COMPLETE; do not shorten the sequence or drop arms. No W tranche is reserved. Before any provisioning, prepare/commit the one-session job and complete the remaining local session gates; fetch all evidence before immediate VM deletion and verify no VM remains. The session job itself is outside this brief's allowed files and was not created here. (Rate: `docs/MASTER-TABLE-2026-10-11.md:29`; session/correctness budget policy: user-provided project instructions, sections 3–4.)

## [U] and limits

- [U] Loaded GPU visibility of all private vectors, physical cache ownership, full 32-token residual/scratch lifetime, and actual residency. Local source/graph/CPU tests are not hardware cache tests.
- [U] Fresh 97/97 checks at both positions, 32/32 HF equality, DSL numerical gate and unfused correctness for each new executable identity.
- [U] Net benefit after local publication, sc1 invalidation, extra barrier and the measured spill increase; the exclusive currently removable fold interval is not isolated.
- [U] Queue attribution on a trace of the refreshed kept baseline and on F. W's historical zero-blockage finding does not substitute for either.
- [U] Production linked executable identity and VM compiler/runtime behavior. The local gate is device compilation, host device-pass compilation and off-path device-object comparison.
- [U] Current GPU price, provisioning duration and whether all registered arms fit the estimated cap. No performance saving or 300-token/s outcome is claimed.

Each unknown corresponds to the unexecuted GPU arms or the explicit local-gate limits above. F remains off; W adds no scheduling policy.
