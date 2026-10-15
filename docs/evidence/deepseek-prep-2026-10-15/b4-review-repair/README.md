# B4 R1/R2 repair acceptance

Code pin: `40ddf3677e8a351c62ba2ece0ef493044f633cd6`; reviewed base: `3a449a7db98f44eaf42f259011e3bcf996327b9b`.
Only two B4 findings were repaired. Independent scoped recheck remains **PENDING**; no main merge/push or GPU spend.

- `review-before-repair.md`: original CC blockers; no blocking x8/cleanup finding.
- `source-freeze.json`: exact repaired Python/test hashes and commit.
- `regression-before.*`: four failures exposing the reviewed defects. `regression-after.*` and `focused-final.*`: passing repair checks.
- `full-suite.*`: **2190 passed, 14 skipped in 490.55s (0:08:10)**, exit 0, no deselection.
- `full-suite-reserve-stop.*`: interrupted first attempt, exit -15; never a pass. `reserve-recovery.json` records only our temporary-copy cleanup and fixture validation.
- `fixture.json`, `bounded_archive_plugin.py`: exact committed bounded source fixture used by two archive transport tests and packaging. Other tests use the worktree.
- `archive-after-receipt.*`: **2 passed in 13.86s**, exit 0, after Qwen readiness refresh.
- `compiler-reuse.json`: zero compiler reruns, 27 device source hashes and 163 original raw evidence files verified; one Qwen test binding changed, 358 unchanged.
- `qwen-receipt-probe.*`: all 359 bindings match the committed fixture.
- `entrypoint-probe.json`, `archive-b4-*.log`: fresh no-Git help and registered dependency dry run, exit 0. No jobs executed. Initial fixture help exit 1 is retained: the bounded Qwen fixture excluded required DeepSeek inputs.json with historical result trees. Final probe adds exactly that tracked recipe manifest from the repair commit; its hash is recorded. No source fix was needed.
- `acceptance.json`: check results and limits. `scoped-recheck.md`: bounded independent repair handoff.

Drivers and raw logs retain real commands, tested hashes and exits. Focused checks ran before the code commit and are
bound to the same three SHA256 values; full and post-receipt archive acceptance ran on committed `40ddf36`.
The final evidence/docs/readiness child commit changes no runtime or test source.
Raw evidence: `/Users/ckl/.hermes/cache/scratch/deepseek-b4-review-repair-evidence/`.

Real profiler CSV format/semantics, native linking/loading, producer/model correctness and all measurements remain
device gates. Full-minus-short traffic is an estimate. Same-kernel replay remains UNAVAILABLE; full two-baseline
performance claims remain INCOMPLETE. No x8, cleanup, compiler, kernel or queue source change is part of this repair.
