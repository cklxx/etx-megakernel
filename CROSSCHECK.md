# Independent merged-package crosscheck

**Worst discrepancy: WP-41b (`6929a08`) introduces an unreported import-order regression.** A coverage case passes 3/3 times in isolation, fails 3/3 times after preloading the canonical model module, and passes 3/3 times under the same preload when only the fresh model builder is replaced with the first-parent source. The wrapper resolves another module's cached output paths instead of the freshly loaded module's settings. The full suite has **32 failed / 3756 passed / 97 skipped / 18 errors**; 18 errors and two failures show the resulting default chain-output-path exception. This does not disprove the board's narrowly reported Qwen B1 before/after identity, but it is an undisclosed behavior change outside that check.

All 26 specified pytest selections have completed; compile/device evidence limits are stated below. All tests run on integration head `22240af754e6992954fa3f94c594776a24830fc1`, on CPU only. No product code is edited.

## Inputs and scope

The originally named `/tmp/o55-lead/` files were absent. The owner supplied `/data/briefs/INTEGRATION.md` and `/data/briefs/WORK-BREAKDOWN.md` during the audit. Their SHA256 values are respectively `3bc905713b75b63b3a173a19139bc4d060e06dac69e6361f0ecf6ac2adf462da` and `735ec38d6df7ba6a95fca50fedf76680ee22550c7ac1bcf1bd2828a9716a4603`.

Only this document is an output. GPU execution, Docker and network installs are excluded. Compile/device checks and historical independent checks are not certified by CPU pytest results. The reset WP-S4a row and unmerged WP-04 are excluded. Rows with multiple merges are audited as one package at the final integrated head; superseded amendments are explicitly labelled.

## Baseline attribution discrepancy

`BRIEF.md` calls **30 failed / 3497 passed / 96 skipped** the baseline on `84d439b`. The board instead assigns **30 failed / 3345 passed / 94 skipped** to `84d439b`, and **30 failed / 3497 passed / 96 skipped** to snapshot `3ccd8a4`. These are recorded claims, not independently rerun historical suites. Compare the current full run with both; the attribution in the brief is inconsistent with the board.

## Three merge-diff reviews

Reviewed using `git show --first-parent <merge>` and the corresponding first-parent diffs (rather than the default combined merge diff).

- **WP-41b, `6929a08`:** the merge replaces the two example builders with calls to `build_decoder(DecoderConfig.from_example(...))` (`examples/llm/model.py:219-223`, `examples/vllm_llm/model.py:426-430`). `DecoderConfig.from_example` imports the canonical module at `etx/compiler/frontend/decoder.py:70`, instead of using the calling module's resolved settings. `etx/tools/shared_state.py:469-525` deliberately loads a fresh `coverage_model` with temporary `ETX_VL_CHAINS` and other settings. If the canonical module was already imported, its cached `CHAIN_FILE` points at the default absent build directory, and the fresh wrapper ignores the temporary chain path. The controlled comparison below establishes this merge's behavior change. Also, `tests/test_frontend_parity.py:54-56,96-103` now generates its expected result by invoking that same generic builder through the example wrapper. A passing test no longer independently establishes pre-migration parity. The board's Qwen B1 worker report is not disproved, but it does not cover this import-order case, and the merge adds no frozen old-builder oracle.
- **WP-S1b, `b7205bd`:** instrumentation is inserted into the scalar CPU `glm_w` reader, including repeated scale lookups (`examples/glm53/verify_graph.py:182` onward). The CLI observes module8 at position zero for k=0,1,3,5 and always returns failure, including when production remains unmeasured (`examples/spec/ledger.py:299-339`). The FAIL verdict matches the board; the aggregate excess and missing-baseline counts do not (details below). It does not measure physical HBM traffic, GPU tile execution, production, or other starting positions. No additional board/code contradiction was established by this diff review.
- **WP-S4 amendment 2, `22240af`:** the merge introduces explicit ETX operations and CPU-compiled C++ bodies, plus planning for gfx942 and sm_90. `build_graph` refuses capacity beyond `index_topk`; the all-visible path has no sparse index ranking. The retained Torch path dequantizes storage to float32; neither path simulates activation/cache quantization. CPU execution follows graph order and does not validate GPU synchronization or device compilation. This supports the board's narrow “module8 MTP is an ETX graph with CPU tile execution” claim, not production MTP or GPU qualification. No additional board/code contradiction was established by this diff review.

## Complete-suite result

Command: `/data/etx-venv/bin/python -m pytest -q tests`.

Completed result: **32 failed, 3756 passed, 97 skipped, 18 errors in 2659.72s (0:44:19)**, exit **1**, 3903 total outcomes. The errors are included separately, not folded into failures.

| Recorded reference | Failed delta | Passed delta | Skipped delta | Errors now |
|---|---:|---:|---:|---:|
| Brief: 30 failed / 3497 passed / 96 skipped (misattributed to `84d439b`) | +2 | +259 | +1 | 18 |
| Board: `84d439b`, 30 failed / 3345 passed / 94 skipped | +2 | +411 | +3 | 18 |
| Board: `3ccd8a4`, 30 failed / 3497 passed / 96 skipped | +2 | +259 | +1 | 18 |

The board does not explicitly list baseline error counts; it lists no errors. Added tests and environment dependencies prevent interpreting these aggregate deltas as a regression count.

The complete run used `TMPDIR=/data/etx-run/wp-xv-audit/tmp`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, Linux 7.2.9-1-cachyos x86_64, Python 3.14.7 and pytest 9.1.1. The same environment is used for package checks. No GPU or Docker build is requested, and no packages are installed.

An earlier invocation of the same full-suite command was **NOT COMPLETE**: `/tmp` (a 16 GiB tmpfs) filled and pytest ended in an internal error near 75%, with `OSError: [Errno 28] No space left on device` and no valid summary. It cost approximately 46 minutes. Its partial counts are not reported as a result. Logs were preserved, then the full command was rerun with scratch on `/data` (over 200 GiB available). No other agents' temporary files were removed.

Evidence logs on this machine: `/data/etx-run/wp-xv-audit/logs/full.log` (completed run), `logs/full-aborted.log`, and `logs/packages-aborted.log`. These scratch logs are not part of the commit; the exact completed summary and failed/error names below are retained in this document.

The common coverage exception is `FileNotFoundError` at `etx/compiler/frontend/decoder.py:462`, on `open(CHAIN_FILE, "w")`. The absent object is the default output parent directory, not a required input fixture. This affects the Mixtral and Phi cases in `tests/test_coverage_gate.py`; the Qwen cases do not produce this failure. The board already acknowledges `tests/test_glm53_slice.py::test_wait_splits_stream_completion_from_device_copies` as an open regression; it also fails here. Other full-suite failures include missing historical fleet/cache artefacts, sealed-document assertions, and tool/environment assumptions. Read-only baseline source inspection (`git show 84d439b:examples/vllm_llm/model.py`) finds the same default `build/chain_tiles.hip` at line 164 and unguarded `open(CHAIN_FILE, "w")` at line 664. The unguarded write itself predates WP-41b, but the controlled parent-builder comparison below shows that selecting the wrong cached module settings is new. No historical full suite was rerun. No product fixes or output-directory workaround were applied.

### Full-suite failed/error names

```text
FAILED tests/test_coverage_gate.py::test_all_model_layers[mixtral-8x7b] - Fil...
FAILED tests/test_coverage_gate.py::test_all_model_layers[phi-3.5-moe] - File...
FAILED tests/test_cpu_check.py::test_formal_missing_dependency_fails - Assert...
FAILED tests/test_documentation_ownership.py::test_move_destinations_and_anchors_exist
FAILED tests/test_documentation_ownership.py::test_sealed_evidence_bytes_are_unchanged
FAILED tests/test_documentation_ownership.py::test_historical_aliases_resolve_to_recorded_targets
FAILED tests/test_documentation_ownership.py::test_moved_documents_and_snapshots_keep_relative_links
FAILED tests/test_dsv2_d5.py::test_default_plan_and_kernel_are_byte_identical
FAILED tests/test_dsv2_kva.py::test_default_and_explicit_replicated_are_byte_identical
FAILED tests/test_dsv2_stage_table.py::test_every_stage_has_a_lever_and_the_rows_partition_the_token
FAILED tests/test_experiment_columns.py::test_source_bound_capture_binding_sites
FAILED tests/test_experiment_columns.py::test_nonmonotonic_ready_arrivals_have_independent_free_producer_queues
FAILED tests/test_glm53_module_gate.py::test_injected_bugs_fail_the_module_gate[shared expert dropped]
FAILED tests/test_glm53_module_gate.py::test_injected_bugs_fail_the_module_gate[KDA beta unsquashed]
FAILED tests/test_glm53_module_gate.py::test_injected_bugs_fail_the_module_gate[routed scale dropped]
FAILED tests/test_glm53_module_gate.py::test_injected_bugs_fail_the_module_gate[kpool tail off by one]
FAILED tests/test_glm53_module_gate.py::test_production_qualification_and_verify_end_to_end_on_a_cpu_launcher
FAILED tests/test_glm53_run_all.py::test_the_plan_lists_every_step_in_order
FAILED tests/test_glm53_run_all.py::test_a_window_that_does_not_straddle_the_pp2_split_is_refused
FAILED tests/test_glm53_run_all.py::test_environment_variables_fill_every_parameter
FAILED tests/test_glm53_run_all.py::test_an_unusable_max_memory_names_the_default
FAILED tests/test_glm53_slice.py::test_wait_splits_stream_completion_from_device_copies
FAILED tests/test_kernel_library.py::test_moe_up_has_one_library_owned_copy
FAILED tests/test_manifest_class.py::test_round30_a_derived_decision_cannot_be_contradicted_by_a_define
FAILED tests/test_packaging.py::test_wheel_contains_importer_shim_headers - s...
FAILED tests/test_producer.py::test_off_source_byte_identity_and_emitter_closure
FAILED tests/test_producer_readiness.py::test_fresh_archive_interpreter_preflight_and_failure_report
FAILED tests/test_producer_readiness.py::test_documented_push_stage_attach_start_with_local_transport
FAILED tests/test_qwen_batch_correctness.py::test_native_script_supervises_both_compiler_stages
FAILED tests/test_qwen_staging.py::test_stage_pack_on_a_packed_model_writes_the_pinned_golden_and_the_config
FAILED tests/test_session_references.py::test_session_python_references_exist[examples/sessions/experiment_program.py]
FAILED tests/test_test_paths.py::test_test_modules_have_no_home_dependencies
ERROR tests/test_coverage_gate.py::test_shared_state_acceptance[mixtral-8x7b]
ERROR tests/test_coverage_gate.py::test_shared_state_acceptance[phi-3.5-moe]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[mixtral-8x7b-write]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[mixtral-8x7b-edge]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[mixtral-8x7b-compiled_edge]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[mixtral-8x7b-count]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[mixtral-8x7b-tail]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[mixtral-8x7b-duplicate]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[phi-3.5-moe-write]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[phi-3.5-moe-edge]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[phi-3.5-moe-compiled_edge]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[phi-3.5-moe-count]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[phi-3.5-moe-tail]
ERROR tests/test_coverage_gate.py::test_missing_write_or_edge_names_the_site[phi-3.5-moe-duplicate]
ERROR tests/test_coverage_gate.py::test_missing_queue_entry[mixtral-8x7b] - F...
ERROR tests/test_coverage_gate.py::test_missing_queue_entry[phi-3.5-moe] - Fi...
ERROR tests/test_coverage_gate.py::test_wrong_argument_stride[mixtral-8x7b]
ERROR tests/test_coverage_gate.py::test_wrong_argument_stride[phi-3.5-moe] - ...
```

## Package Accept crosscheck

Each command below was run separately, sequentially, at the final integrated head. Amendments add their named pytest files; `-rfE` is omitted to follow the brief's requested `pytest -q <test files>` form (it only changes reporting). All result columns explicitly include failed, passed, skipped and error counts; absent categories in pytest's summary mean zero. **DIFF** is a count finding required by the brief, not automatically a correctness regression. Board rows are historical and sometimes report broader selections than the work-breakdown Accept.

| Package | Board claim | Failed / passed / skipped / errors | Exit | Count comparison | Command |
|---|---|---:|---:|---|---|
| WP-10 | 45 passed | 0 / 45 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_p1_occupancy_nv.py tests/test_machine.py tests/test_passes.py` |
| WP-11 | 61 passed, 2 skipped | 0 / 61 / 2 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_primitives_neutral.py tests/test_acquire_contract.py` |
| WP-S2 | 12 passed | 0 / 16 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_glm_verify_graph.py` |
| WP-20 | 2 passed, 1 skipped | 0 / 2 / 1 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_host_backend.py` |
| WP-31 | 43 passed | 0 / 43 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_serve_sampling.py` |
| WP-02 | 442 passed, 3 skipped | 0 / 442 / 3 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_no_vllm_imports.py tests/test_experiment_validation.py tests/test_experiment_review_repair.py tests/test_dsv2_batch.py tests/test_qwen300_hour.py tests/test_vlbest_report.py` |
| WP-21 | 24 passed, 1 skipped | 0 / 24 / 1 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_loader.py` |
| WP-30 | 10 passed | 0 / 10 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_serve_kv.py` |
| WP-14 | 16 passed | 0 / 16 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_ir_import_nvptx.py` |
| WP-S6 | 52 passed | 0 / 52 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_glm_verify_moe.py` |
| WP-S3 | 12 passed | 0 / 12 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_glm_rollback.py` |
| WP-S1 | 40 passed | 0 / 53 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_verify_ledger.py` |
| WP-22 | 45 passed, 3 skipped | 0 / 19 / 1 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_step_api.py` |
| WP-S6b | 108 passed | 0 / 121 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_glm_verify_graph.py tests/test_glm_verify_moe.py tests/test_verify_ledger.py` |
| WP-17 | 24 passed | 0 / 16 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_prims.py` |
| WP-40 | 47 passed | 0 / 47 / 0 / 0 | 0 | Match | `/data/etx-venv/bin/python -m pytest -q tests/test_model_contract.py` |
| WP-S5 | 67 passed, 1 skipped | 0 / 5 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_spec_loop.py` |
| WP-S1-amendment1 | 67 passed (ledger + verify graph) | 0 / 53 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_verify_ledger.py` |
| WP-18 | 36 passed | 0 / 20 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_ops_wave32.py` |
| WP-41 | 56 passed | 0 / 9 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_frontend_parity.py` |
| WP-P1a | 76 passed (batch union + verify MoE + rollback) | 0 / 12 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_glm_batch_union.py` |
| WP-S8 | 27 passed | 0 / 22 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_proposer.py` |
| WP-S4a-amendment2 | 58 passed, 5 failed, 0 errors (includes batch union) | 5 / 46 / 0 / 0 | 1 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_mtp_reference.py tests/test_glm_rollback.py tests/test_glm_verify_graph.py tests/test_glm53_module_gate.py` |
| WP-41b | 71 passed | 0 / 9 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_frontend_parity.py` |
| WP-S1b | ledger + verify graph pass; no count | 0 / 69 / 0 / 0 | 0 | No count claimed | `/data/etx-venv/bin/python -m pytest -q tests/test_verify_ledger.py tests/test_glm_verify_graph.py` |
| WP-S4-amendment2 | 27 passed, 5 failed | 0 / 17 / 0 / 0 | 0 | **DIFF** | `/data/etx-venv/bin/python -m pytest -q tests/test_glm_mtp.py tests/test_mtp_reference.py` |
| WP-19 | 6/6 binaries compiled, never run | — / — / — / — | — | Not independently verified | No pytest Accept exists. Specified Accept is `$CUDA bash bench/calib/cuda/build.sh` plus six `--help` invocations; Docker is forbidden by this brief. |
| WP-47a | 5 failed, 216 passed in unspecified GLM suites; gfx942 ISA identity; sm_90 build | — / — / — / — | — | Not independently verified | No named pytest files in Accept. It specifies CUDA build, ROCm ISA comparison and weights-free TOKENS-EQUAL; no substitute suite was invented. |

Counts are not deduplicated across rows: shared test files were rerun for each package. WP-19/WP-47a are audited as evidence limits, not passing packages. Their compile claims cannot be established by a CPU-only pytest rerun, and the board does not name the GLM selection behind 216 passed / 5 failed. WP-04 and the reset WP-S4a row have no current merge and are excluded.

### Interpreting count findings

There are 14 count mismatches. Several are explained by broader selections, or by tests added later:

- WP-S2 gained four tests at the integrated head: 16 versus 12. WP-S1 now has 53 ledger tests versus 40, and WP-S6b has 16 + 52 + 53 = 121 versus 108. Later changes are an explanation, not a falsified historical result.
- WP-22's claimed 45 passed / 3 skipped numerically equals host backend (2/1), loader (24/1) and step API (19/1) combined; its named Accept only selects step API. This is a scope inference from independently run component counts, not a fresh combined run.
- WP-S5's 67 passed / 1 skipped numerically equals spec loop (5), sampling (43), and step API (19/1). Its named Accept only selects five spec-loop tests.
- WP-41's 56 numerically equals frontend parity (9) plus model contract (47). WP-P1a explicitly names the broader board scope: batch union (12) + verify MoE (52) + rollback (12) = 76, versus 12 in its specified Accept. WP-S8's 27 numerically equals proposer (22) + spec loop (5).
- WP-S4a's board scope adds batch union: 46 passing tests + 12 = 58, retaining the same five failures. The specified amended selection gives 46 passed / 5 failed. WP-S4's board scope is consistent with adding module gate (10 passed / 5 failed) to its named 17 passing tests: 27 passed / 5 failed.
- WP-S1 amendment 1's board count names ledger + verify graph, while the work breakdown says the CLI plus existing ledger tests. This audit runs that named ledger selection (53 passed). The later WP-S1b combined selection independently gives 69 passed, versus the older combined count of 67. Its old “missing observed” CLI outcome is intentionally superseded by WP-S1b, not a current-code contradiction.
- WP-17 (24 versus 16), WP-18 (36 versus 20), and WP-41b (71 versus 9) remain count findings for the exact commands shown. This audit did not recover the lead's historical expanded commands, so it cannot certify how those larger counts were assembled.

The five WP-S4a failures are exactly the five names in `/tmp/WP-S4a-baseline-84d439b.log` (that historical log records 5 failed / 10 deselected). The historical log was read, not regenerated. No package selection produced a setup error. Logs and command/exit records are in `/data/etx-run/wp-xv-audit/packages/`; all table counts above come from actual executions, not collection-only estimates.

## Controlled import-order reproduction (three runs per condition)

The unexpectedly passing isolated reruns made this a suspected order-dependent/flaky case. It was run three times in each condition; it is repeatable within each condition.

| Condition | Each run | Times, seconds |
|---|---|---|
| Current tree, ordinary isolated pytest | 1 passed, exit 0, three times | 7.98 / 7.91 / 7.96 |
| Current tree, canonical module preloaded | 1 failed, exit 1, three times | 0.14 / 0.13 / 0.14 |
| First-parent builder source, same canonical preload | 1 passed, exit 0, three times | 7.93 / 7.93 / 7.94 |

Isolated command (same scratch/thread environment as above):

```sh
/data/etx-venv/bin/python -m pytest -q 'tests/test_coverage_gate.py::test_all_model_layers[mixtral-8x7b]'
```

Preloaded command, repeated three times:

```sh
/data/etx-venv/bin/python -c "import examples.vllm_llm.model; import pytest; raise SystemExit(pytest.main(['-q', 'tests/test_coverage_gate.py::test_all_model_layers[mixtral-8x7b]']))"
```

Each preloaded failure ends at `etx/compiler/frontend/decoder.py:462`: `FileNotFoundError: .../examples/vllm_llm/build/chain_tiles.hip`, opening an output in write mode. No generated product directory was created to mask the result. The case is not an intrinsically random numerical failure: canonical import state determines the outcome.

For the parent comparison, the following scratch harness substitutes only the fresh model source from `6929a08^1`; all other code, the test, and the canonical preload remain current. This is a source-substitution control, not a complete checkout/test of the parent commit. The initial scratch harness failed to import `examples`; adding the worktree to `sys.path` corrected the harness before the three reported test executions.

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import examples.vllm_llm.model  # Prime the canonical module with default output paths.
import importlib.abc
import importlib.util
import subprocess
import pytest
source = subprocess.check_output(['git', 'show', '6929a08^1:examples/vllm_llm/model.py'], text=True)
original = importlib.util.spec_from_file_location
class BeforeLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__file__ = str(__import__('pathlib').Path('examples/vllm_llm/model.py').resolve())
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
def before_spec(name, path, *args, **kwargs):
    if name == 'coverage_model':
        return importlib.util.spec_from_loader(name, BeforeLoader(), origin=str(path))
    return original(name, path, *args, **kwargs)
importlib.util.spec_from_file_location = before_spec
raise SystemExit(pytest.main(['-q', 'tests/test_coverage_gate.py::test_all_model_layers[mixtral-8x7b]']))
```

It was run three times with `/data/etx-venv/bin/python /data/etx-run/wp-xv-audit/check-parent.py`. Logs: `logs/coverage-repeat-{1,2,3}.log`, `logs/coverage-preloaded-{1,2,3}.log`, `logs/coverage-parent-{1,2,3}.log` under `/data/etx-run/wp-xv-audit/`. Other package tests were not identified as flaky by this audit; a single successful package run is not a general flake guarantee.

## FC-2a CLI: matching verdict, mismatched aggregate claims

Command: `/data/etx-venv/bin/python -m examples.spec.ledger --fc2a`, same environment as above. Exit **1**, observes module8 at k=0,1,3,5 and prints **FC-2a FAIL** with measured per-tensor reasons. It never prints “missing observed”; it does report “missing positive observed baseline” for specific tensors. Production is not executed. This is a correctness/reader-count result, not GPU performance or physical HBM traffic.

The board reports fixed-weight excess **116.5 / 350.8 / 585.1 MB**, routed k=0 excess **about 50.6 MB**, and **16 tensors** without observed baseline. Those aggregate claims are not reproduced by this single CLI execution:

| Aggregate | Board claim | This run |
|---|---:|---:|
| Fixed excess, k=1 | 116.5 MB | 58,268,288 B = 58.268288 MB |
| Fixed excess, k=3 | 350.8 MB | 175,404,544 B = 175.404544 MB |
| Fixed excess, k=5 | 585.1 MB | 292,540,800 B = 292.540800 MB |
| Routed excess, k=0 | about 50.6 MB | 25,328,128 B = 25.328128 MB |
| Missing positive baseline | 16 tensors | 2 distinct tensor identifiers, 8 reason occurrences across four k cases |

MB means decimal 1,000,000 bytes. These are derived sums of the **signed** `excess=` values in each semicolon-delimited CLI reason for the corresponding k and kind, counting each printed reason once. Fixed counts are 101 reason rows per k; routed counts are 12 rows at k=0. The board's totals are approximately twice these sums. The board does not provide its aggregation command here, so double counting is a hypothesis, not a confirmed cause.

The two distinct missing-baseline identifiers are `l1/self_attn.indexer.index_kpool_compress_ape` and `l2/self_attn.indexer.index_kpool_compress_ape`, each repeated at k=0,1,3,5. The board's individual example **2,641,920 B observed versus 524,416 B expected** is reproduced for gate/up expert weights; it is the aggregate/count claim that differs. Raw output: `/data/etx-run/wp-xv-audit/logs/fc2a.log`.

No fixes were applied. The only committed deliverable is this report; `BRIEF.md` remains untracked and is not included.
