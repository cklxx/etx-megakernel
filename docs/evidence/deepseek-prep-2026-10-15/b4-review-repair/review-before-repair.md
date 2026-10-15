# DeepSeek final code review — CC_REVIEW

Date: 2026-10-15

## Verdict

**BLOCKED: two defects in the B4 diagnostic path need correction before the two-session package is ready for a GPU-spend decision.** No blocking defect was found in the x8 correctness path or in the added process-group cleanup change within this review scope.

**中文结论：B4 路径有两项阻断缺陷，修复并做定向复核后再决定 GPU 开支。x8 路径和进程清理补丁未发现阻断问题。未运行测试、构建或 GPU；不支持融合性能结论。**

### Covered revision

- Worktree: `/Users/ckl/code/.wt/etx-deepseek-two-gaps`.
- Branch: `prep/deepseek-two-gaps-2026-10-15`.
- Initial review pin: `1a23336e1bdf1c78c63ec35f3dab136f88acc2b0`.
- Current revision at closure: **`3a449a7db98f44eaf42f259011e3bcf996327b9b`**. The branch ref was checked again before this report; it had not changed from the preceding check.
- The review spans the original diagnostic implementation and the later bounded cleanup change in `etx/session/proc.py`. The operator states that the DeepSeek implementation is unchanged. This review did not run Git or independently compare every working-tree byte with either commit. It must not be described as a review that stayed at the initial pin.

File references below are relative to the worktree above.

## Blocking findings

### R1 — Timed rounds reject valid per-batch vLLM selections

**Locations:** `examples/dsv2lite/batch_attribution.py:177-193`; `examples/dsv2lite/batch_session.py:61-72`; `etx/session/batch.py:101-111`.

Selection is separate for B1, B2, and B4. However, each selected configuration is then run at **all three** batch sizes. The round fails if that configuration is illegal at any of those sizes, including a size for which another configuration was selected.

**Failing condition:** default is legal and selected at B1; AITER is legal and selected at B2/B4; default fails correctness at B4. These selections meet the per-B selection rule. Nevertheless, a timed default invocation also runs B4, and line 193 stops the rounds and all dependent captures. The same problem occurs with the configurations reversed. This can consume setup, reference, build, and selection work without testing the valid selected set to completion.

This is not a request to waive a full-call or short-call correctness gate. The queue should not add known-invalid, unselected batch/configuration combinations to the timed set.

**Minimum acceptance:** run each selected configuration only at the batch sizes assigned to it. Keep all full and short gates for every invoked batch size. Add a CPU contract case with different legal subsets for default and AITER. It must complete five pairs for each selected B. A failure in any selected combination must still stop the remaining rounds and captures.

### R2 — vLLM traffic can be marked complete with missing counter dispatches

**Locations:** `examples/dsv2lite/batch_attribution.py:54-81`, `:118-119`, and `:260-266`.

The capture requests a kernel trace, but `ranged_traffic()` never reads it. The function checks the marker set, boundary crossings, and the counter rows that are present. It has no check that each kernel dispatch inside a call range has a corresponding FETCH_SIZE record. Unlike the ETX path at line 253, it supplies no expected dispatch count.

**Failing condition:** a full-call range contains two kernel dispatches, but its counter CSV contains only one of them. If that remaining value is positive and the full-minus-short value stays positive, the current code accepts the range. The capture can then write `status=COMPLETE` with understated traffic. Removing one counter row from a multi-dispatch range is sufficient; the whole range does not need to be missing. Such data cannot support the registered traffic-growth attribution.

The device may provide complete data, but the current parser cannot enforce the declared refusal of missing records. This is a static completeness defect, not a claim that a target capture has already lost records.

**Minimum acceptance:** compare counter records with the retained kernel-dispatch trace for each marked call. Match dispatch identities with their process and agent scope. Refuse missing, extra, duplicate, or boundary-crossing dispatch records. Treat an explicitly measured zero separately from a missing record. Add a CPU fixture with two traced dispatches and only one counter record; it must leave attribution INCOMPLETE. Keep the full-minus-short result labelled as an estimate.

## Checks completed

### Eight-request correctness path

- Inspected the arm registry, registered queue, reference preparation, diagnostic loop, parser, classifier, and finalizer.
- `x1km` matches x8's numerical recipe: the five levers, L1 loads, shared kv_a, four-wave merge, and D4 off. The one-request and eight-request schedules are not claimed to be equal (`examples/dsv2lite/arms.py:100-127`).
- Reference reuse requires a matching tuple, producer recipe, successful producer receipt, and retained producer-log hash. Missing provenance causes refusal or fresh production; the queue does not certify old bytes in place (`examples/dsv2lite/reference_prepare.py:48-105`). The weight-pack hash gate remains.
- The six declared diagnostics use request-specific inputs. Completed numerical failures do not prevent the later declared diagnostics. Tokens alone cannot pass: all expected finite layer metrics and the token/warm-up gates are required (`examples/dsv2lite/x8_correctness.py:20-49`, `:75-101`).
- The classifier distinguishes a matched single-request failure, repeated multi-request failure, non-reproduction, and unresolved evidence. It does not establish a race or a historical repair.
- The final lane can record normal dependency failures. A hard session stop can prevent a pending finalizer; this limit is documented.

### Four-request attribution path

- Inspected the registered B1/B2/B4 queue, gate identities, selection, rounds, captures, routing, and result assembly.
- Reference/build dependencies and per-request hidden/token gates precede timing. A selected timed failure stops later work. R1 identifies the excess combinations checked in that path.
- The host now uses the same steady-clock endpoints around fused and unfused execution. Parameter copies precede the interval. Device timestamps remain separate (`examples/dsv2lite/host.hip:204-217`). This does not turn old HIP-event measurements into matched host-wall results.
- The pinned vLLM image and default/AITER candidate set are explicit. Both full and one-token calls, including warm-up, have token gates. The claim is limited to these two candidates, not all vLLM configurations.
- ETX traffic requires 64 megakernel dispatches. Present counter values must be finite, nonnegative, and have valid spans. Profiled timings do not supply performance verdicts. R2 identifies the missing vLLM dispatch-coverage check.
- Body/wait analysis calls the existing complete-trace validator. It does not identify tile-internal waits as ETX event waits.
- HF routing runs separately at B1, B2, and B4. Each run checks its bound 33-token trajectory. The projection uses that B's executed prefix, not an averaged subset. Native ETX pick equality remains unmeasured.

### Unavailable same-kernel baseline

`graph_eligibility()` returns UNAVAILABLE for the current plans and also refuses unknown plans (`examples/dsv2lite/batch_attribution.py:84-93`). The registered eligibility jobs write `passed=False`. The graph execution path rechecks eligibility before launch. The final result requires complete graph results at all three batch sizes before it can mark the full performance claim complete (`:324-330`).

No ordinary registered path was found that converts the unavailable baseline into a certified fusion or full two-baseline performance claim. The documented LDS lifetime limit is **not a new finding**. A replay-safe kernel is not an acceptance requirement for this review. Correctly scoped attribution and corrected-vLLM system data may still be collected after R1 and R2 are fixed and the device gates pass.

### Archive, runner, and cleanup

- Inspected fresh-archive staging, module entrypoints, build entrypoints, producer success dependencies, optional finalizer dependencies, and aggregate deadline handling.
- Queue defaults are 3600 seconds for x8 and 7200 seconds for B4. The shared runner permits an explicit `ETX_SESS_MAX_SECONDS` override; retain the registered limits for any later authorized run. A session deadline is not a provider billing or VM-deletion limit.
- Inspected `etx/session/proc.py:53-81` and the added tests at `tests/test_session_proc.py:44-64`. Cleanup now waits for asynchronous process-group removal without `/proc`, under the existing 30-second bound, and reports a persistent group. Linux descendant scanning remains. No blocking defect was found in this bounded change. This does not establish descendant coverage outside a process group on non-Linux systems.

## Limits and next action

This was a read-only code review. No repository file was changed. No tests, builds, network calls, GPU work, or extra agents were used. The registration and committed preparation evidence were read; reported compiler/test passes were not rerun or independently certified here. Final full-suite acceptance remains with SOL_CLOSE.

Native linking/loading, producer execution, real model correctness, target ROCm compatibility, profiler availability and CSV format, counter semantics, and all measurements remain device gates. First-step hidden checks do not certify every layer on every decode step. Token-window collisions remain a stated limit.

Correct R1 and R2, retain focused regression evidence, and request a bounded recheck of those changes. Do not broaden this into a kernel redesign. This report grants no GPU-spend, merge, push, or deployment authorization.
