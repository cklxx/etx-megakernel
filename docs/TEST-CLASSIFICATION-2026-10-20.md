# Merge-failure classification — 2026-10-20

The merge `02e6753` ("Merge the Qwen batch baseline and GLM 5.3 execution
design") is the tip of `main` on both remotes. A full-suite run of it on the
x86 Linux test host reported **18 failed, 2306 passed, 88 skipped, 9 errors**
(exactly reproduced here). This note classifies every one of those 27
failure/error entries against the pre-merge commit `6bc0428`, and records that
none is a regression from the merge.

## Method

Both trees were tested with the same remote harness: rsync the worktree to the
x86 host with `.git`, `.venv`, `build/`, `__pycache__`, `*.pyc` and `*.o`
excluded, then run

```
PYTHONPATH=. /data/etx-venv/bin/python -m pytest -q -p no:cacheprovider \
    --tb=no -rf --junitxml=<report>.xml
```

in the synced copy. The remote destination was wiped before each run, so no
file from one tree survived into the other. JUnit reports were compared by
`(classname, name)`; the 20 entries common to both trees were also compared by
failure message, and all 20 messages are identical apart from pytest's
per-run temporary-directory name.

- Baseline: `6bc0428` → **11 failed, 2139 passed, 59 skipped, 9 errors**
  (400.93 s)
- Merge: `02e6753` → **18 failed, 2306 passed, 88 skipped, 9 errors**
  (407.73 s)
- Baseline set is a subset of the merge set; **7 entries appear only in the
  merge**, and all 7 are new tests introduced by the merge.

## Classification

`Env` = environmental (test host lacks a dependency the test requires, or the
harness deliberately does not sync it). `Real` = a regression caused by the
merge. No entry is `Real`.

### Shared with the baseline — 20 (pre-existing, not caused by the merge)

| # | Test | Kind | Class | Cause |
|---|------|------|-------|-------|
| 1 | `test_bench_sim.py::test_recompute_is_exact_and_rounds_are_paired` | error | Env | No `.git` in synced tree |
| 2 | `test_bench_sim.py::test_offline_report_rejects_incomplete_or_invalid_pairs[missing]` | error | Env | No `.git` |
| 3 | `test_bench_sim.py::test_offline_report_rejects_incomplete_or_invalid_pairs[duplicate]` | error | Env | No `.git` |
| 4 | `test_bench_sim.py::test_offline_report_rejects_incomplete_or_invalid_pairs[seed]` | error | Env | No `.git` |
| 5 | `test_bench_sim.py::test_offline_report_rejects_incomplete_or_invalid_pairs[trace]` | error | Env | No `.git` |
| 6 | `test_bench_sim.py::test_offline_report_rejects_incomplete_or_invalid_pairs[failure]` | error | Env | No `.git` |
| 7 | `test_bench_sim.py::test_corrupt_artifact_invalidates_previous_success` | error | Env | No `.git` |
| 8 | `test_bench_sim.py::test_refuse_overwrite` | error | Env | No `.git` |
| 9 | `test_bench_sim.py::test_inflated_makespan_fails_trace_and_report` | error | Env | No `.git` |
| 10 | `test_bench_sim.py::test_exception_is_recorded_and_fails_run` | failure | Env | No `.git` |
| 11 | `test_bench_sim.py::test_nonfinite_sample_is_retained_as_strict_json` | failure | Env | No `.git` |
| 12 | `test_cpu_check.py::test_formal_missing_dependency_fails` | failure | Env | No java (`formal/check.sh`: "java unavailable") |
| 13 | `test_cpu_check.py::test_dependency_failure_preserves_failed_manifest` | failure | Env | No `.git` |
| 14 | `test_dsv2_stage_table.py::test_every_stage_has_a_lever_and_the_rows_partition_the_token` | failure | Env | No pinned fleet export and no fleet repository on the host |
| 15 | `test_packaging.py::test_wheel_contains_importer_shim_headers` | failure | Env | `build` module absent from the test venv |
| 16 | `test_producer.py::test_off_source_byte_identity_and_emitter_closure` | failure | Env | `git show c432c77:...` — no `.git` |
| 17 | `test_producer_readiness.py::test_fresh_archive_interpreter_preflight_and_failure_report` | failure | Env | `git archive HEAD` — no `.git` |
| 18 | `test_producer_readiness.py::test_documented_push_stage_attach_start_with_local_transport` | failure | Env | `tools/session.sh` runs `git status`/`git archive` — no `.git` |
| 19 | `test_qwen_staging.py::test_stage_pack_on_a_packed_model_writes_the_pinned_golden_and_the_config` | failure | Env | Session command calls `.venv/bin/python`; the harness excludes `.venv` |
| 20 | `test_verify_gaps.py::test_prefetch_isolation_default_source_identity_and_on_wave_scope` | failure | Env | `git show 9f91f3a:...` — no `.git` |

### Only in the merge — 7 (all new tests, all environmental)

| # | Test | Kind | Class | Cause |
|---|------|------|-------|-------|
| 21 | `test_glm53_model.py::test_unique_score_kpool_selection_beyond_budget_matches_independent_hf` | failure | Env | No `torch` in the test venv |
| 22 | `test_glm53_model.py::test_clipped_swiglu_matches_hf_at_saturation_boundaries` | failure | Env | No `torch` |
| 23 | `test_glm53_model.py::test_snapshot_reader_accepts_hf_symlink_shards_and_rejects_path_traversal` | failure | Env | No `torch` |
| 24 | `test_glm53_model.py::test_hf_chunk_prefill_then_32_exact_native_greedy_tokens[False-False]` | failure | Env | No `torch` |
| 25 | `test_glm53_model.py::test_hf_chunk_prefill_then_32_exact_native_greedy_tokens[True-True]` | failure | Env | No `torch` |
| 26 | `test_glm53_model.py::test_bf16_conv_output_rounds_before_silu` | failure | Env | No `torch` |
| 27 | `test_qwen_batch_correctness.py::test_native_script_supervises_both_compiler_stages` | failure | Env | `batch_native.sh:15` calls `.venv/bin/python`; the harness excludes `.venv` |

## Counts

| Class | Shared | New | Total |
|-------|-------:|----:|------:|
| Environmental — no `.git` in the synced tree | 16 | 0 | 16 |
| Environmental — missing module (`build`, `torch`) | 1 | 6 | 7 |
| Environmental — missing interpreter (`.venv/bin/python`) | 1 | 1 | 2 |
| Environmental — missing binary (java) | 1 | 0 | 1 |
| Environmental — missing model/export data | 1 | 0 | 1 |
| **Real regression from the merge** | **0** | **0** | **0** |
| **Total** | **20** | **7** | **27** |

## Conclusion and disposition

No failure is caused by the merge, so no code is changed on this branch. In
particular, no test gains a skip: the 7 new failures would remain hidden if
the merge's own tests were gated on the environment, and the baseline shows
the 20 shared ones are properties of the test host, not of either commit.

One check worth recording: `test_native_script_supervises_both_compiler_stages`
passes (1 passed, 0.66 s) when a `python` is placed at `.venv/bin/python` in
the synced tree, which confirms its only blocker is the excluded interpreter
and that the new script's logic runs under the synthetic compiler fixture. It
was not retained as a change because the missing interpreter is the
environmental condition under test, not a merge regression.

Reproducing the merge run needs a host with `.git` (or a checkout, rather than
an rsync copy), the `build` package, `torch`, java, and the fleet export; none
of those is part of this repository's tree.
