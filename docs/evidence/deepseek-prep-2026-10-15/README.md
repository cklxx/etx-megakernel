# DeepSeek preparation evidence, 2026-10-15

This is local preparation evidence. No GPU result is present. GPU spending is $0.

Source commit: `a0c3e72` (`prep/deepseek-two-gaps-2026-10-15`, base `24eabf5`). The later evidence commit changes
only documents and evidence metadata. `archive-acceptance.json` binds the tested code, tests, queues, archive hash,
and actual check exit codes. The source map must remain identical after that commit.

Raw evidence is retained at `/Users/ckl/.hermes/cache/scratch/deepseek-open-prep-evidence/`.
It includes the local scripts used to make these records. `raw-evidence-sha256.json` binds their bytes.

- `archive-acceptance.json`: focused tests and both dependency dry runs from the committed-source archive.
- `build-identity.json`: exact current generated plan/header/source comparisons, retained artifact checksums, patched
  Fleet source hashes, build flags, new x1km artifacts, and fresh host-object hashes for all five required arms.
- `source-closure.json`: all 27 device include/patch source files match the retained compiler evidence revision.
- `native-retained-identity.json`: kept/x8 native source identity; these device artifacts were reused, not rebuilt.
- `rocm-x1/`: new ROCm 7.2.4 gfx942 device assembly, ISA/object, complete HIP host-object and register reports.
- `generated/x1km/`: new native LLVM 23 device IR/codegen/object and host IR. Other `generated/` directories retain
  source and fresh ROCm host objects; no full device compiler matrix ran.
- `host-wall-*.raw.log`: fresh complete x86 HIP host objects after the matched host-clock correction.
- `cpu-runner-probe.json`: CPU fixture success and producer-failure branches. Failure rc=9 skips dependent diagnostics
  and still runs the final job. These are runner contract probes, not model evidence.
- `real-r5-binding.*`, `real-r7-binding.*`: real local-cache refusals, exit 1. The old payloads remain unavailable.
- `b4/`: counter/prefix/clock/eligibility tests, actual retained-plan graph refusals, raw logs and actual exit codes.

The first x1km build returned 2 because of quoting in the added embedded shell. It had already produced device and
host objects. `rocm-x1-first*` preserves that failed attempt. The quoting fix, outer and embedded shell syntax tests,
and corrected complete run are retained. No failed attempt is described as a successful run.

The target compiler is the installed offline ROCm 7.2.4 clang 22, in local image
`sha256:084a446de60ebf2192b2000d1da2fc784aea0f48a8b3e7cf672f0b60a26314af`.
This is not evidence of compatibility with a future DigitalOcean image. New x1km target resources: 256 VGPR,
75 AGPR, 108 SGPR, 0 scratch bytes/lane, 26668 LDS bytes, occupancy 1 wave/SIMD. Kept/x8 ROCm device artifacts
and native artifacts are reused only after the identity checks above. B4 x2/x4 retained generated sources match;
their prior target build evidence remains distinct from the fresh host-only checks.

`rocm-x1/x1km/link.exit` is 77: local `libamdhip64.so` is absent. No HIP executable/native load is claimed.
Real references, native linking/loading, target compiler compatibility, profiler support, and all correctness and
performance results remain native/device gates.

The same-kernel graph baseline is locally UNAVAILABLE for all current B1/B2/B4 plans. The retained design §15.7
already reports the LDS/in-body-wait failure. No invalid replay is queued. B4 attribution and corrected vLLM system
data may complete; the final two-baseline performance claim remains INCOMPLETE. Native ETX expert picks are not
measured by the separate per-B HF routing estimate.

The initial preparation left the Qwen heavy lock intact and did not run the full suite or integrate.

## Final local acceptance (SOL_CLOSE)

The tested final revision is `fcb5f649463573c9dc728f8d4de46d5bb5a27c7a`, DeepSeek implementation `a0c3e72` unchanged, plus process-cleanup fix `fcb5f6`. The previously excluded graph tests
passed: **14 passed, 83 deselected in 6.90 seconds**. The final complete suite returned exit 0: **2168 passed, 14 skipped in 508.28s (0:08:28)**.
No tests were deselected in the full run. Skips are reported in its raw log; they are not passes.

[Closure evidence](closure/) retains full logs, exact commands/exits, drivers, source-fixture provenance and
[the review status](closure/review-status.json). The two existing Qwen archive transport tests use a real
committed source fixture without historical DeepSeek GPU data, as in the earlier Qwen integration; their source bytes
and commands are unchanged. All other tests use the pinned worktree. This is local transport/test evidence.
The combined-source Qwen readiness receipt was refreshed for the changed process supervisor, optional batch
profiler wrapper and new tests. Its other 354 source bindings and all compiler inputs are unchanged. The prior
compiler records are reused with an explicit source delta, not recertified as fresh builds. The refreshed receipt
matches all 359 keys in the real committed source fixture; both real archive transport cases were rechecked after
that refresh. This preserves the existing Qwen preparation without adding an experiment.
The shared advisory lock was acquired correctly; its stale filename did not block work. A 4 GiB disk reserve was
monitored throughout. To restore the reserve, our source tar was compressed losslessly to
`deepseek-open-prep-evidence/source-a0c3e72.tar.gz`; its decompressed SHA256 remains
`465824e7d42c90fafd88a20b4050b24751f12d89042ee05bf417668dc3f70e5c`. The duplicate extraction was removed.
Raw logs and build artifacts remain retained. Historical paths in the initial receipt describe that initial run.

Independent review status at evidence sealing: **independent review pending**. A missing report prevents main integration and
push; it is not a pass. The first full run returned exit 1: 1 failed, 2165 passed, 14 skipped in 512.73 seconds. The failure exposed
asynchronous process-group cleanup on macOS without `/proc`. The bounded fix `fcb5f6` changes only the supervisor
and two regression tests; both tests fail before the fix. Relevant tests then passed (61 passed, 1 Linux-only skip),
and the complete final suite above passed on the frozen revised source. The original failed run remains in the raw
logs. The independent review must cover both the original two diagnostics and this bounded cleanup diff.
DeepSeek implementation/build sources are unchanged. Valid compiler evidence above is reused. No GPU, cloud,
provider, purchasing or account operation was performed. The same-kernel replay limitation remains unresolved.
