# Qwen expert row merge probe

Registration: `QWEN-EXPERT-ROW-MERGE-2026-10-15`.
Status: local preparation with the two final-code review findings repaired; final independent re-review pending. No GPU result. Full-model default: OFF.
Owner branch: `experiment/qwen-expert-early-merge`.
Base: `c432c77b7f778799ecbb89db9af89902d5923c00`.

## Decision and prior evidence

Test whether 32 last-producer row owners can replace repeated consumer folds and RMS statistics at a net benefit. A screen pass permits a real-input probe, then a full-model implementation trial. It does not permit a production change or a KEEP claim.

The retained research is `/Users/ckl/.hermes/cache/scratch/codex-expert-early-merge-result.md`. Its source and data hashes are in `expert-early-merge-research/manifest.json` beside that file. Do not regenerate that research. It matched 60,362 historical task identities. Row-group readiness slack had a median of 0.55 us. The final group had no slack. These are historical observations, not a forecast for this candidate. The older global single-writer fold losses do not isolate this design.

The down weight stream remains 25,165,824 bytes per layer. Its bandwidth floor at the repository's assumed 4.3 TB/s is 5.853 us. The existing DEVICE completion remains. Thus this probe removes no complete dependency boundary. Current exposed fold service, protocol tail cost and net gain are unknown. The 10.4 us compile-time schedule model uses assumed 3 us tile durations. It tests the graph and liveness only. It is not an elapsed-time estimate.

The full qkv subgraph reads at least 96,468,992 weight bytes: 50,331,648 for up, 25,165,824 for down and 20,971,520 for qkv. At 4.3 TB/s its weight-only floor is 22.435 us. The head subgraph has 697,827,328 weight bytes and a 162.285 us floor. These derived floors omit payload, reset and spill traffic. No current measured full-window baseline is retained. The historical trace's last-down-body to first-consumer-ready interval had a 1.755 us median; last-down-body to last-consumer-done had a 12.885 us median, including the head outlier. These intervals predate the kept norm hoist. They neither isolate the old fold nor give a removable gap budget. This probe keeps that completion event, so it must win through less total work after charging its added tail. Its fresh control measurement is a required first device condition.

## Frozen arms

`examples/vllm_llm/expert_merge/source.py` extracts the kept GEMV source with exact, unique anchors. Both arms call the existing up and down tile bodies. The control consumer uses the original vector fold and RMS arithmetic. The candidate appends row-owner work inside the down body. Its consumer reads the merged vector and shared inverse RMS. It then runs the same normalized bf16 staging and consumer GEMV body.

`prepare generate` freezes the full source map, flags, consumer role, diagnostic mode and generated files in `identity.json`. The identifier is the SHA-256 of the canonical manifest before the identifier and generated hashes are added. Emission and native compilation use a clean experiment environment. Native `build.json` adds the executable hash, compiler version and command. A changed source, flag, input or generated file requires a new build. Final local identifiers and resource facts are in the preparation result and its evidence index.

The flags are WSUM, SCALAR, GU1, RBATCH, NORM_HOIST, VECNORM and KEY32. The plan publishes system-coherent DSL payload access, batch one and one request. It has 192 up tasks, 256 down tasks and 256 consumer tasks. Physical workers remain 8 domains with 38 workers each. Up and down task `(slot,j)` stays on the slot's domain. Each domain's 24 up tasks feed its down tasks through a DOMAIN event. All 256 down tasks feed the retained DEVICE event. No imported kernel is removed: this isolated DSL subgraph has zero imports. The full model and its imported attention/RMS kernels are unchanged.

There are four builds per input: control and candidate, each with and without diagnostics. Both qkv (5,120 output rows) and final head (151,936 output rows) have build entry points. The full model has 47 next-layer qkv folds and one final head fold. Layer-zero qkv is outside this mechanism. The probe starts after upstream routing. Slot-ordered selected weights and routing weights are fixed inputs. It does not test routing selection.

The isolated pack stores the eight selected experts in slot order and remaps their IDs to 0 through 7. Each matrix and its row/column arithmetic stay the same in both arms. Absolute addresses, cross-expert address spacing and full-model cache context differ from the original allocation. A real-input screen therefore applies to this packed replay layout. A later model trial must preserve the original weight allocation and recheck the gain. The probe cannot certify that transfer.

## Ownership and publication

There are 32 groups of 64 output rows and eight expert-slot producers per group. Each producer finishes its original bf16 down output, drains every producing wave and joins the block. Thread zero decrements the group's counter from eight. The returned last-producer decision is broadcast in LDS. All waves take the same branch. Non-owners return. No task waits for another task inside the appended body.

A row owner sums slots in order 0 through 7. It preserves `bf16(bf16(fp32_slot_sum) + residual)`. It writes its 64 merged values and, for qkv only, the same canonical residual slice. It writes eight original per-thread square accumulators at indices `8*group` through `8*group+7`. It does not compress these to one group scalar.

After another all-wave drain and block join, the owner decrements the statistics counter from 32. The final owner loads the 256 partials. Other threads contribute positive zero. It uses the original 64-lane butterfly, ordered sum of 16 wave results, division, epsilon and `rsqrtf`. It writes the inverse RMS scalar. Runtime drainage and the original full-down completion occur after the body returns. The final row/statistics tail is therefore charged. No duplicate generated arrival exists for either custom counter.

All new cross-XCD payloads use the kept system-coherent accessors. The RMW uses the original DEVICE event atomic scope and order. This is a source-level protocol. CPU checks cannot certify AMD visibility. Cached replacements, relaxed acquisition, disabled all-wave drainage and batch/extended variants are rejected or unsupported.

The argument table owns a 4 KiB merged vector, 1 KiB partial array, 4-byte scalar, and 33 unpadded 32-bit counters. Counters are host-owned, not generated events. Every replay resets 32 counters to eight and one to 32. The reset is in both fused and same-kernel graph measurement windows. Payloads need no zero fill because all entries are overwritten. The original DOMAIN and DEVICE counters also reset. A fresh host process owns each paired arm. Diagnostic checks reuse the same allocated payloads for 12 resets with alternating activation signs and permuted row task orders. Head output has no canonical residual store.

The import LDS arena is 21,120 bytes, separate from runtime LDS. All appended election and reduction storage fits that arena. Native residency is a mandatory runtime gate. Compiler resource reports, spills and actual loaded occupancy must be inspected before a device performance result. CPU resource estimates do not prove loaded occupancy.

## Cost account

The candidate retains the entire up/down weight stream, local up-to-down edges, full DEVICE completion, consumer norm-weight reads, normalized bf16 rounding, consumer GEMV, runtime drainage and runtime retirement. It adds 256 row-counter RMWs and 32 statistics-counter RMWs, block broadcasts and joins, 32 row merges, partial writes, one statistics reduction, scalar publication, merged/scalar reads and 132 bytes of reset payload. It can add cache-line contention, backpressure, code size, register live ranges and spill traffic. Those costs are unknown until measured.

The research derived 9,437,184 logical fold-input request bytes per control fold and 1,092,612 for shared inverse RMS, excluding unchanged norm weights, contribution stores, canonical residual stores and counters. These are logical requests. They are not measured HBM traffic or a time saving. The new implementation retains an additional runtime drain. The final group's complete merge and scalar tail has no historical slack. Do not add readiness slack to removed consumer service as independent savings.

## Correctness and performance gates

The runner checks both fused and same-kernel graph paths for all four builds before any timing. It requires exact bytes for up activations, down contributions, canonical residuals, normalized bf16 inputs and complete consumer outputs across 12 resets. Diagnostics also require exact merged bf16 values, all 256 fp32 partials, inverse RMS and each of the 256 consumers' normalized vectors. It checks finite output, valid row-owner coverage and zero custom counters. A changed payload must change the up activation. A failed recheck removes the latest authorization pointer. Old logs remain.

Performance builds contain no diagnostic stores or stamps. Five pairs alternate control/candidate order on the same GPU lane. Each arm records 200 samples after 20 warmups. Raw samples and complete logs remain. Both arms use device events around a HIP graph that contains reset, complete up/down/consumer work and runtime retirement. The same-kernel graph arm uses the same tile bodies in separate launches, including the new protocol and reset. The reported polling-based `wall_us` is diagnostic only. It is not the model's host-wall metric.

The registered screen requires every paired device gain to be positive, median gain at least 0.625 us, and median greater than twice `(max_gain - min_gain)`. Both arms must finish all samples. A fail stops this route. A synthetic pass permits only a real-input probe. No real captures were available at preparation. Before extrapolation, use real inputs from layers 0, 23 and 46 for qkv, plus layer 47 for head, at positions 0 and 1040. Report every input separately. One passing input is not complete real-input coverage.

A later model trial remains gated on fresh same-build import certificates at positions 0 and 1040, DSL checks and 32/32 HF greedy tokens. Its KEEP rule is five complete interleaved host-wall pairs: every gain positive, median at least 0.030 ms and greater than twice the spread. It must also report each build's same kernels in a HIP graph and the strongest tested legal vLLM configuration in that session. This probe makes no ETX fusion or system claim.

## Local reproduction

Use the repository `.venv` with pytest, NumPy and PyYAML. CPU arithmetic needs clang++ with C++ headers. Local gfx942 compilation needs LLVM 23 and the retained ROCm 7.2.4 headers/device libraries. Offline checks need the existing `etx-rocm22` x86 image and extracted ROCm 7.2.4 LLVM 22 toolchain. They use no network and no GPU. No image pull is permitted.

```sh
P=.venv/bin/python
E=/Users/ckl/.hermes/cache/scratch/expert-merge-local-new
$P -m examples.vllm_llm.expert_merge.validate unit "$E/unit"
$P -m examples.vllm_llm.expert_merge.validate probe "$E/probe"
$P -m examples.vllm_llm.expert_merge.validate offline "$E/offline"
$P -m examples.vllm_llm.expert_merge.validate matrix "$E/matrix"
$P -m examples.vllm_llm.expert_merge.validate entrypoints "$E/entrypoints"
ETX_REFERENCE_PYTHON=/Users/ckl/code/etx/.venv/bin/python PYTEST_ADDOPTS=-ra \
  $P -m examples.vllm_llm.expert_merge.validate full "$E/full"
```

Each stage needs a new output directory. Heavy stages use the shared advisory lock `/Users/ckl/.hermes/cache/scratch/qwen-prep-heavy.lock`. They stop below a 4 GiB free-space reserve and have fixed command deadlines. The 16-build matrix runs one build at a time. Full pytest removes passing test intermediates. Keep logs, receipts and failed artifacts. Do not remove source, other tasks' output or original research.

## Authorized-GPU commands for a future operator

These commands are prepared only. No rental or remote execution is authorized by this record. Proposed isolated session cap: 1,800 seconds on one MI300X. Estimated cost is $1 at an assumed $2 per GPU-hour, before provider billing increments and setup. This is not a verified price. Check the balance and actual rate before any later authorization. A retry must use the original saved deadline. It cannot silently extend this budget. A separate input/session needs a separate authorized budget. Do not run the entire real-input coverage campaign under a one-input estimate.

On an already authorized single-GPU machine, from the repo root:

```sh
python3 -m etx.tools.gpu_jobs examples/sessions/qwen_expert_merge.py \
  --gpus 0 --cpu-slots 1 --logdir /local/new-expert-merge-session
```

Use the existing `tools/session.sh push|start|status|fetch` transport for that job file only after authorization. The session builds four executables, checks all paths, then measures five interleaved fused and same-kernel graph pairs. A failed dependency blocks timing. The runner has a nonblocking session lock, unique attempt paths, full stdout, exit receipts, silence warnings, a fixed saved deadline and precise blocked/resume messages. A hung device check ends that process. Do not reuse its device state. No synthetic run can issue a model KEEP verdict.

For a real capture, create a private checking host. It leaves shared router and attention modules unchanged:

```sh
python3 -m examples.vllm_llm.expert_merge.capture emit /local/capture-host-new
ETX_VL_CHECK=1 ETX_TOPK_KEY32=1 ETX_DEV_RESET=0 ETX_VL_REQUESTS=1 \
ETX_VL_CU=256 ETX_VL_TOUCH=1 ETX_PREFETCH_ISOLATE_POLLER=0 \
ETX_VL_ROUTER_NARROW=1 ETX_VL_FOLD_TOPK=0 \
ETX_HIPCC_FLAGS='-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH -DETX_INBODY_REPORT -DETX_DSL_NORM_HOIST' \
ETX_OUT=/local/capture-build-new bash /local/capture-host-new/build.sh /local/packed-qwen
python3 -m examples.vllm_llm.expert_merge.capture run /local/capture-L23-P1040 \
  --host-build /local/capture-build-new --packed /local/packed-qwen --layer 23 --position 1040
EEM_INPUT=/local/capture-L23-P1040 python3 -m etx.tools.gpu_jobs \
  examples/sessions/qwen_expert_merge.py --gpus 0 --cpu-slots 1 --logdir /local/real-expert-session-new
```

The generated build attests that its compiler command used the private capture host. A checkout records its Git revision. An archive without `.git` requires the complete `tools/session.sh stage` `SOURCE_IDENTITY` receipt (commit, branch, archive SHA-256 and UTC staging timestamp); missing or malformed receipts are rejected. The attestation retains the receipt hash and provenance alongside the source, binary and plan hashes and the compiler command. Capture sealing requires its binary/plan identity, successful command receipt, complete same-position imported-launch coverage, all capture markers, input sizes and hashes. An incomplete capture stays blocked. Capture and prefix replay need their own explicit GPU time budget; they are not hidden inside the 30-minute isolated screen. Do not recapture retained valid inputs.

## Integration

All implementation files are new under `expert_merge/`, plus one session file and one test file. No shared index, model, host, lease or production default is changed. Preserve main's router work at `2fdbed13a5574f1d86ee10602a8fe71a30c720b6`. After serialized integration, regenerate source identities and private capture hosts and rerun acceptance. Source anchors must fail closed if the merged host or GEMV changes. Shared experiment-index registration is deferred to that integration step.

## Preparation evidence

The original evidence below belongs to preparation commit `172a22f3809f0ef9a31ebd369fe48fb175dcddd0`. The independent final-code review at `/Users/ckl/.hermes/cache/scratch/astra-expert-final-code-review.md` found two pre-GPU blockers in that commit. The review is preserved. The scoped repair adds an all-thread block join between the control's vector-fold writes and its cross-wave diagnostic LDS copy, inside `EEM_DIAGNOSTIC` and before the block-uniform `store` branch. Both qkv and head use this path. Performance builds contain no added join. The second repair accepts and validates the staged archive identity described above. Neither repair changes the byte gates, repeated payload checks, timing authorization, performance thresholds or full-model default.

Repair evidence is under `/Users/ckl/.hermes/cache/scratch/expert-final-fixes-evidence/`; the scoped result is `/Users/ckl/.hermes/cache/scratch/codex-expert-final-fixes-result.md`. All eight build identities were regenerated because the manifest includes the changed source files. Unaffected compiler evidence is retained only with explicit source/preprocessed-code equivalence; the two changed control diagnostic configurations were freshly compiled. This local repair does not constitute a passed independent re-review or a device result.

Actual repair acceptance: the new regressions fail three times on the reviewed source (both consumer roles and archive attestation), then pass all three after the repair. The full experiment and session-transport test files pass **46 tests in 27.15 seconds**. They include successful no-Git receipt attestation, seven missing/malformed identity refusals, a normal Git checkout, and the original strict correctness/timing refusals. All eight identities were regenerated; `build-reuse.json` records old/new identities and preprocessing hashes. The six unaffected device translation units match after build-path/identity normalization. Both changed control diagnostic builds freshly pass LLVM 23 gfx942 compilation with device libraries and host syntax, ROCm 7.2.4 LLVM 22 device assembly/host compilation, and HSA gfx942 device linking with `--no-undefined`. The private capture host was regenerated and passed syntax checking; its generated source hash matches the original. No native GPU executable was linked or run. Original arithmetic, protocol, 16-build matrix and full-suite evidence is retained as historical evidence, not relabelled as a new run. Raw logs, expected failure exits and final archive receipts are indexed in the repair result.

Durable local evidence is under `/Users/ckl/.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/`. The final result is `/Users/ckl/.hermes/cache/scratch/codex-expert-early-merge-prep-result.md`. Its `validation.json` and `SHA256SUMS` identify the committed sources, commands, true exits, all eight probe identities and compiler reports. Failed attempts remain in that directory.

All eight qkv/head, control/candidate and diagnostic/performance configurations passed local LLVM 23 gfx942 compilation and host syntax. They also passed ROCm 7.2.4 LLVM 22 device assembly, the host device pass and x86 host-object compilation. All eight assembly outputs linked to HSA gfx942 shared code objects with `--no-undefined`. The first link attempt lacked `libxml2.so.2`. Task-local Ubuntu libraries resolved it; the existing image and host toolchain were not changed. The retained link script and package hashes make that step repeatable. Native HIP host executable linking is still unavailable locally because `libamdhip64` is absent. No code object was loaded or executed.

The ROCm 22 performance megakernel reports 87 VGPRs for control and 110 for candidate, 21,152 LDS bytes and zero scratch bytes per lane for both. It reports 19 and one SGPR spills, respectively, and zero VGPR spills. These are compiler facts, not GPU timings or residency proof. LLVM 23 reports different allocation and scratch, so do not substitute its report for the VM compiler's report. The serial 16-build baseline matrix passed. CPU arithmetic passed 64 synthetic qkv/head cases in each of the off/fast contraction builds. The protocol model passed 128 tokens, 4,096 row owners, all eight owner slots and 128 quiescent resets. The final targeted run passed 20 tests. The full suite passed 2,052 tests and skipped 17 in 488.45 seconds. The skips require historical data/builds, source/toolchain environment settings or Linux `/proc`. Exact reasons remain in the raw log.

The first full run had one environment failure: this worktree's small venv lacks `transformers`, but the local Qwen2.5 checkpoint made its reference test run. An existing CPU environment passed that test, then the entire suite passed with `ETX_REFERENCE_PYTHON` set to it. The retained wrapper enables offline mode, caps CPU threads and uses task-local caches. No dependencies were installed into either venv. This existing four-token CPU reference check provides no Qwen3 GPU acceptance.

中文结论：隔离的真实 tile 实验已实现，默认关闭完整模型。CPU、编译和脚本检查不能证明 GPU 可见性或性能；必须先通过真实输入子图门槛，再考虑完整模型。GPU 花费为 0。
