# DeepSeek B4 repair — scoped recheck

Date: 2026-10-15

## Verdict

**R1 CLOSED. R2 CLOSED. No new blocking regression was found in the reviewed repair scope.**

The two B4 blockers in `cc-gpt-deepseek-final-review.md` are resolved. This closes those findings only. It is not a new full-system review, a device acceptance result, or authorization to spend on a GPU.

## Revision and exact scope

- Previous reviewed revision: `3a449a7db98f44eaf42f259011e3bcf996327b9b`.
- Repair code revision: **`40ddf3677e8a351c62ba2ece0ef493044f633cd6`**.
- Evidence revision: **`62b529df7000a013c8cb3e78e861c07c1ae37c53`**.
- Worktree: `/Users/ckl/code/.wt/etx-deepseek-two-gaps`.
- The branch ref read during this recheck was the evidence revision above.

The source review covered only:

1. `BatchSession.vllm()` subset validation, command construction, prompt-file selection, and output-set checks in `examples/dsv2lite/batch_session.py`.
2. Selected-config assignment and failure handling in `AttributionSession.rounds()`, its capture precondition, and the single-batch profiling call in `examples/dsv2lite/batch_attribution.py`.
3. Dispatch identity/span validation, range coverage, measured-zero handling, and failure persistence for vLLM traffic in that same file.
4. The corresponding regression tests in `tests/test_dsv2_batch_attribution.py` and retained acceptance records.

The prior x8, process-cleanup, kernel, and reference-producer reviews were not repeated. Qwen receipt changes were not audited. Source files were inspected with read-only tools. No Git diff or independent source-hash calculation was run; revision and test-source bindings use the retained receipts and the observed branch ref.

## R1 — CLOSED: selected vLLM batch subsets

### Code basis

- `examples/dsv2lite/batch_attribution.py:220-238` derives the batch set assigned to each selected configuration. Each timed vLLM call receives only that set. The round requires exactly that output set and a legal result for each invoked B.
- `examples/dsv2lite/batch_session.py:56-77` validates the explicit subset, constructs `--batches` from it, supplies the required prompt prefix through `max(invoked)`, and refuses missing or extra batch outputs. It still uses the existing full/short correctness parser; it does not filter failed gates from an invoked batch.
- `examples/dsv2lite/batch_attribution.py:241-269` retains partial results, marks a failed round INCOMPLETE, raises, and prevents captures unless all five selected pairs complete.
- `examples/dsv2lite/batch_attribution.py:154-162` and `:299-306` keep profiling to one explicitly selected batch. The profiling wrapper checks that batch instead of silently changing the command after prompt selection.

### Regression coverage inspected

`tests/test_dsv2_batch_attribution.py:67-132` covers both directions of the mixed default/AITER selection, completion of five pairs, selected-call failures in full/short warm-up and sample phases, prevention of captures after failure, and missing/extra batch outputs. The command fixture at `:440-452` checks the single-batch profiler command and its prompt count.

**Conclusion:** the original valid mixed-selection case can complete without invoking an unselected, known-invalid combination. A failure in an invoked selected combination still stops the experiment. No gate waiver was introduced.

## R2 — CLOSED: counter coverage against traced dispatches

### Code basis

- `examples/dsv2lite/batch_attribution.py:54-81` requires process, agent, and dispatch IDs, valid timestamp spans, and a retained kernel-dispatch trace.
- `:84-116` rejects duplicate records, overlapping call ranges, boundary crossings, missing or extra dispatch identities, and timestamp differences. Each call must have a nonempty traced set and a matching measured set.
- The comparison binds worker dispatches by their own process/agent/dispatch identity. It does not incorrectly require a worker PID to equal the marker-emitting PID.
- `:33-47`, `:114-123` preserve explicitly measured zeros while rejecting invalid values. All five full-minus-short estimates must remain positive. The result remains labelled as an estimate; prefill/cache cancellation is not claimed as proved.
- `:307-316` preserves an INCOMPLETE attribution result when the coverage parser refuses a capture.

### Regression coverage inspected

`tests/test_dsv2_batch_attribution.py:135-183` covers the exact original case: two traced dispatches with only one counter record. It checks both parser refusal and durable INCOMPLETE capture output. Tests at `:195-243` cover extra and duplicate records, incorrect or absent scope fields, timestamp mismatch, crossing dispatches, missing trace data, overlapping ranges, scoped dispatch-ID reuse, and complete measured-zero records.

**Conclusion:** a partial counter set can no longer pass merely because the remaining values and subtraction are positive. Completeness is checked against the retained trace. Native trace/counter format and collection reliability remain device gates, not results established by these fixtures.

## Retained evidence used

Evidence directory: `/Users/ckl/.hermes/cache/scratch/deepseek-b4-review-repair-evidence/`.

| Record | Retained result |
| --- | --- |
| `regression-before.log` / `.json` | Four primary regression cases failed before repair; exit 1. |
| `focused-final.log` / `.json` | 308 passed in 25.98 seconds; exit 0. |
| `full-suite.log` / `.json` | 2190 passed, 14 skipped in 490.55 seconds; exit 0, no deselection. |
| `archive-after-receipt.log` / `.json` | 2 passed in 13.86 seconds; exit 0. |
| `entrypoint-probe.json` | Final no-Git B4 help and dependency dry-run probes record exit 0. |

The focused record names the pre-commit HEAD `3a449a7...`, but its three source hashes match the repair hashes in `acceptance.json` and the full-suite record at `40ddf36...`. It is evidence for the repaired working-tree bytes, not a test of an unchanged old commit. These hash values were compared as retained records, not recomputed in this review.

The first full-suite attempt was stopped for the disk reserve and records exit -15; it is not a pass. The initial bounded archive fixture also lacked required historical input metadata and failed. The handoff retains that failure and records the corrected fixture probe. Packaging and the two Qwen transport tests used the bounded committed-source fixture; the other tests used the worktree. No fresh compiler result is inferred from this repair evidence.

## Remaining limits

- No tests, builds, network calls, GPU work, source edits, or extra agents were performed for this recheck. Only this requested scratch report was written.
- Real model/reference execution, native linking/loading, target profiler CSV format and metric semantics, and all device measurements remain execution gates.
- The same-kernel DeepSeek graph baseline remains **UNAVAILABLE**. The full two-baseline performance claim remains **INCOMPLETE**. These repairs do not establish a fusion or performance result and do not require a new replay-safe kernel.
- No open finding remains from R1 or R2. Broader closure and integration remain outside this recheck. Any GPU allocation still needs a separate decision and authorization.
