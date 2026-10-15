import json,re
from pathlib import Path
b=Path(__file__).parent;s=json.loads((b/'local-closure.json').read_text());a=json.loads((b/'acceptance.json').read_text())
full=a['results']['full-suite']['summary'];focused=a['results']['focused-final']['summary'];archive=a['results']['archive-after-receipt']['summary']
result=f'''# DeepSeek B4 R1/R2 repair: local acceptance complete, scoped recheck pending

Both CC findings are repaired and committed on the existing branch. Local acceptance is complete. Independent
scoped repair recheck remains PENDING; no main merge/push, GPU/cloud operation, spend ($0) or new experiment occurred.

- Worktree: `{s['worktree']}`.
- Branch: `{s['branch']}`; clean.
- Reviewed base: `{a['reviewed_base']}`.
- Repair implementation: `{s['code_revision']}`.
- Evidence/docs/readiness commit: `{s['metadata_commit']}`. No runtime/test source changed in this child commit.
- Main remains `{s['main_revision']}`; main status is {'clean' if not s['main_status'] else repr(s['main_status'])}. Main was not modified. No remote operation was needed.

R1: each timed configuration invokes only batch sizes assigned by selection; its command, golden prompt count and
returned batch keys match that subset. Captures invoke one selected B. Every invoked full/short warm-up and sample
correctness gate remains mandatory. A selected failure stops remaining rounds and captures. Both directions of
mixed default/AITER selection now complete all five pairs in CPU fixtures.

R2: every marked call requires the retained kernel trace and exact FETCH_SIZE coverage by Process_Id, Agent_Id,
Dispatch_Id and timestamps. Missing, extra, duplicate, misscoped or boundary-crossing records and overlapping ranges
refuse attribution. Explicit measured zero remains a present measurement; full-minus-short estimates remain positive
and labelled as estimates. The two-dispatch/one-counter fixture leaves capture attribution INCOMPLETE.

Four primary regressions fail against the reviewed code (exit 1), exposing both mixed-selection directions and both
missing-dispatch parser/capture failures. Final focused acceptance: **{focused}**, exit 0, including the 14 graph cases.
Final complete suite on frozen repair commit: **{full}**, exit 0, no deselection. Skips and their reasons remain in the
unfiltered full log; they are not passes. Post-receipt archive transport acceptance: **{archive}**, exit 0. Fresh no-Git
B4 help and registered dependency dry run both return 0; no GPU job is executed. The first bounded-fixture help
probe returned 1 because the Qwen fixture excludes DeepSeek result metadata. That failure is retained; the final
probe includes the exact tracked inputs.json recipe manifest from the repair commit and records its hash. Final whitespace and source-hash
checks pass.

The first full-suite attempt stopped at the reserve guard (exit -15 after 350.63 s), not a test pass. Packaging copied
2.6 GiB of historical results. Only our reproducible temporary project copy was removed. Packaging and the two Qwen
archive transport tests now use the established committed source fixture without historical DeepSeek results;
all 122 packaging metadata/test/package inputs match. All other tests use the pinned worktree. The actual stopped
attempt and bounded fixture adjustment are retained. The advisory lock was acquired correctly; file existence was
not treated as ownership. A 4 GiB reserve guard monitored heavy runs.

No unchanged compiler matrix was rerun. All 27 retained device include hashes and all 163 original raw evidence files
verify. Device/host sources, generated plan/emitter inputs, compiler flags, build scripts and queues are unchanged by
this repair. Qwen's broad readiness receipt changes one attribution-test source binding; 358 others remain unchanged.
All 359 bindings match the committed fixture after refresh and the active worktree after the evidence commit.

Evidence:

- Committed current index: `docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/README.md`.
- Committed exact source/check receipt: `docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/acceptance.json`.
- Raw logs, actual exits, drivers, source freeze, compiler reuse, fixture and local commit verification: `{b}/`.
- Before/after regressions: `regression-before.*`, `regression-after.*`, `focused-final.*`.
- Full acceptance: `full-suite.*`; interrupted attempt: `full-suite-reserve-stop.*`; recovery: `reserve-recovery.json`.
- Receipt/archive: `compiler-reuse.json`, `qwen-receipt-probe.*`, `archive-after-receipt.*`, `entrypoint-probe.json`.
- Final exact local state: `local-closure.json`. Committed evidence files are bound by `sha256.json`.
- Earlier preparation/cleanup closure records remain at the existing parent evidence index and `closure/`.

Original independent review: `/Users/ckl/.hermes/cache/scratch/cc-gpt-deepseek-final-review.md`, covering `3a449a7`.
It found no blocking x8 or process-cleanup issue. Its B4 R1/R2 findings are locally repaired; this does not constitute
independent approval. Bounded handoff: `{b}/scoped-recheck.md`. Suggested independent report:
`/Users/ckl/.hermes/cache/scratch/cc-gpt-deepseek-b4-repair-recheck.md`. Review only the three-file
`3a449a7..40ddf36` repair diff and retained acceptance. No review agents were launched. No idle review polling.

Keep this clean worktree for the scoped recheck. Main integration/push remain gated on a nonblocking independent
repair report; GPU spending requires its separate authorization. No broader audit or new replay-safe kernel is
required by this repair. Native linking/loading, real producer/model correctness, target ROCm and profiler CSV/metric
semantics, and all measurements remain device gates. Original lost r5/r7 payloads are still absent. Same-kernel HIP
graph replay remains UNAVAILABLE; full two-baseline performance claim remains INCOMPLETE.
'''
(b.parent/'codex-deepseek-closure-result.md').write_text(result)
(b.parent/'deepseek-open-prep-status.md').write_text(f'''# FIX_B4_TWO_REVIEW_FINDINGS

Status: local repair acceptance complete and committed; independent scoped R1/R2 recheck PENDING.
Branch `{s['branch']}`; clean worktree `{s['worktree']}`.
Repair code `{s['code_revision']}`; metadata/evidence `{s['metadata_commit']}`.
Main remains `{s['main_revision']}`. No main merge/push, GPU/cloud action, spending, new experiment or compiler matrix rerun.

R1 invokes exactly assigned selected batch subsets; all invoked full/short/warm-up gates stay strict. R2 requires exact
process/agent/dispatch and timestamp coverage against the retained kernel trace; incomplete records fail closed,
measured zeros remain measurements. Four before-fix regressions fail. Focused: {focused}; full: {full}, exit 0,
no deselection; refreshed archive: {archive}, exit 0. Fresh no-Git B4 help/dependency dry run and final hash/receipt
checks pass. The stopped reserve-guard attempt and temporary-copy recovery are retained. All 27 device include
hashes, 163 original raw files and 359 active Qwen source bindings verify. No unchanged compiler matrix reran.

CC review received: no blocking x8/cleanup issue, B4 R1/R2 repaired locally. Remaining action: independent scoped
repair recheck of `3a449a7..40ddf36`, no useful local work pending. No polling or review agents launched.
Handoff: `{b}/scoped-recheck.md`.
Evidence: `{b}/acceptance.json`; committed index `docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/README.md`.
Result: `/Users/ckl/.hermes/cache/scratch/codex-deepseek-closure-result.md`.
Same-kernel replay remains UNAVAILABLE; full two-baseline performance claim INCOMPLETE. Native/device gates remain.
''')
print('Updated existing status and closure result for',s['metadata_commit'])
