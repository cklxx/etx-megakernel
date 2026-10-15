# Three-path integration acceptance, 2026-10-15

LOCAL PREPARATION PASS. Both independent reviews close pre-GPU blockers.
GPU execution, native HIP executable link/load and device acceptance were NOT run.
GPU spend: $0. All normal production defaults remain OFF.

The integration branch was created in the new isolated worktree
`/Users/ckl/code/.wt/etx-qwen-three-integration` from producer main
`bcc3c98b421164ae70e42d566bf836d0da35425f`. Its two merge parents are attention
`db1694bcdbd2345401f42a58e5f41b4b4ee43d02` and expert repair
`11743ed0c18ce4a2acc4af5493c678af1a62691a` (includes `172a22f`).
No shared runtime, model, host, lease or launcher source changed against main.
See `integration-boundary.json` and the two scoped review-source comparisons.

## Acceptance and exact scope

- Full suite: **2129 passed, 14 skipped, 2 deselected**, 490.40 seconds, exit 0.
  The two archive tests were deliberately run separately: **2 passed**, 15.67
  seconds, exit 0. This is not a claimed single 2131-test run.
- C6: fresh local LLVM 23 and offline ROCm 7.2.4 LLVM 22 full control/envelope,
  retained performance probe and diagnostic builds. Current capture/hybrid hosts
  compile against the combined ordinary kept plan. Fresh source/artifact hashes
  and native resource gates pass. Local/native imported IR matches the reviewed
  preparation after the narrowly checked provenance normalization; raw hashes
  remain distinct. `attention-identities.json` records all six manifests.
- Expert: all eight qkv/head x control/candidate x diagnostic/performance builds
  were freshly generated, locally compiled and offline ROCm-compiled. All eight
  `tiles.hip` and `plan_data.h` files byte-match the repaired reviewed preparation.
  Regenerated capture host passes syntax against the combined kept plan.
  `expert-identities.json` gives all eight new identities; no old binary was
  relabelled. Native executable runtime linking/load/execute remain open.
- Producer: all **328** prior source hashes still match, with **29** additional
  isolated experiment/test files. The prior 16-variant matrix and producer
  compiler results are reused explicitly, not rerun. `producer-compile-reuse.json`,
  `compiler-environment-reuse.json` and the 201-entry historical artifact audit
  justify reuse. The refreshed receipt matches all **357** combined source keys.
  Previous receipt and checksum list are preserved byte for byte here.
- Both CPU simulations, shell syntax and all three separate session job graphs
  pass. CPU simulation is not a GPU experiment.
- Bounded archive tests use an actual committed combined-source Git fixture.
  Only `examples/dsv2lite/results/` historical data is omitted; every producer
  source key is independently checked for exact coverage and hash equality.
  The two unchanged tests receive this fixture through a saved pytest ROOT
  monkeypatch. Real archive/push/stage/start paths, real independent interpreter
  and dependencies, actual preflight/report run; transport and compute substitutes
  are explicitly labelled. Source receipt is accepted, interpreter passes,
  synthetic box inputs fail, compute is skipped and the real report is
  NOT-COMPLETE. This is local transport acceptance, not a model result.
- From the no-Git archive, all three job graphs and four CLI entrypoints pass.
  Expert capture generation/attestation succeeds with a valid staged identity,
  and missing/malformed identities each fail with exit 1 as required. Binary
  and plan in this attestation check are non-executable transport fixtures.

`identity-summary.json`, `sources.json`, `generated-artifacts.json`, all command
lists and individual status receipts bind these statements. Generated assembly,
IR, bitcode and host objects remain at the indexed scratch paths; full raw build
logs, resource reports and source maps are durable here. Full plans are
losslessly stored as `plan.json.gz`; `compressed-plans.json` binds raw and packed
hashes. Drivers are
saved as `.txt` so copying evidence cannot expand the source-map glob and
invalidate its own manifests. No test/source gate is weakened.

## Independent review

[Attention final review](reviews/astra-attention-final-code-review.md): PASS at
`db1694b`, no blocking implementation defect. This integration freshly performs
its required host/plan/import/identity checks; it does not transfer a binary ID.

[Expert original review](reviews/astra-expert-final-code-review.md) and
[repair re-review](reviews/astra-expert-fixes-rereview.md): both prior defects
CLOSED at `11743ed`. Every scoped implementation/test/job byte is unchanged,
and generated math/plan comparisons plus fresh dependency compiles confirm
continued coverage. `reviews.json` retains report hashes and original paths.

## Failures and local resource controls

The first merge attempt had an uninitialized no-checkout index and failed with
exit 2. Only this new worktree's index was initialized; the subsequent merge was
clean. Both logs and true exits remain here.

The first full-suite attempt failed collection (pytest exit 2, supervisor exit 1):
two unrelated DeepSeek tests directly import a historical inputs file excluded
by sparse checkout. The retry temporarily linked the existing main historical
data directory, ran the identical tests and passed. It removed that local data
view after all tests exited. No data was copied, modified or regenerated, and no
source was changed to bypass the tests. Both attempts are retained.

Heavy work was serialized under `qwen-prep-heavy.lock`, with fixed deadlines,
progress output, actual child exits and a 4 GiB reserve. More than 6 GiB remained
at sealing. No other worker's tree/artifacts were altered or deleted.

## Final committed-source archive

The exact integration merge commit `bef4cbf708e5dae66d90a04f81c33435a6f13988` also passed a
small execution-source archive check (3338240 bytes). The refreshed receipt
was accepted, the real interpreter passed, synthetic box inputs were refused
before compute and the real report was NOT-COMPLETE. All three archived session
graphs and the strict expert staged identity passed. See `committed-archive/`.
Only evidence metadata follows this source commit; source maps remain identical.

## Remaining GPU-only work

See the durable [three-path queue](../../../../docs/QWEN-THREE-GPU-QUEUE-2026-10-15.md)
and [combined registration](../../../../docs/QWEN-THREE-EXPERIMENTS-2026-10-15.md).
No fourth experiment, aggregate budget, default switch or extra research is added.
Device linking/loading, physical residency, visibility, repeated real byte gates,
captures, imports/DSL/32 tokens and each path's paired performance gates remain
mandatory. A C6 local screen or expert synthetic screen is not a model-wall claim.
Separate GPU authorization, real inputs and a budget are still required.
