# Example commands

Total commands: **174**.

Run `python -m examples.cli <name> [arguments]` from the repository root.
Run `python -m examples.cli <name> --help` for source-derived help without importing the example.
Execution runs the original module as `__main__`; arguments, errors and exit codes belong to that module.

## Inventory and limits

This checkout has 174 commands with main guards; 147 also construct `ArgumentParser`.
The brief's 149-parser count is a historical snapshot. The inventory includes scripts with manual
`sys.argv` handling and no argument parser, rather than excluding them.
Names are paths relative to `examples/`, with slashes replaced by dots and `.py` removed.
All entries expose a lazy callable in `examples.commands.REGISTRY[name].callable`;
`module`, `entrypoint`, and `arguments` record the original module, main block and argument declarations.

Help lists source declarations (including helper functions, subparsers and dynamic expressions),
not evaluated defaults or a claim that every declaration applies to every mode.
For scripts without argparse declarations, it includes the original main block and module docstring.
No example is imported just to build the registry or show help. Actual execution requires the
same dependencies, input files, compilers and hardware as the original.

**Status scope:** “Broken here” below means a direct, unconditional top-level import references
a module absent from the local Python search path (checked with the acceptance interpreter,
without importing the example). These commands stay registered. “Runtime unverified” means
execution has not been qualified; it does not certify GPU support, transitive dependencies or inputs.
Tests import the lazy adapters and resolve every original module; they do not execute GPU scripts.

## Families

| Family | Count |
| --- | ---: |
| dsv2lite | 35 |
| experiment_program | 32 |
| glm53 | 19 |
| llm | 5 |
| sessions | 8 |
| spec | 1 |
| vllm_llm | 74 |

## dsv2lite (35)

| Command | Module | Status |
| --- | --- | --- |
| `dsv2lite.arms` | `examples.dsv2lite.arms` | Runtime unverified |
| `dsv2lite.batch_attribution` | `examples.dsv2lite.batch_attribution` | Runtime unverified |
| `dsv2lite.batch_floor` | `examples.dsv2lite.batch_floor` | Runtime unverified |
| `dsv2lite.batch_goldens` | `examples.dsv2lite.batch_goldens` | Runtime unverified |
| `dsv2lite.batch_mechanism` | `examples.dsv2lite.batch_mechanism` | Runtime unverified |
| `dsv2lite.batch_session` | `examples.dsv2lite.batch_session` | Runtime unverified |
| `dsv2lite.build_resources` | `examples.dsv2lite.build_resources` | Runtime unverified |
| `dsv2lite.bytes` | `examples.dsv2lite.bytes` | Runtime unverified |
| `dsv2lite.coh_check` | `examples.dsv2lite.coh_check` | Runtime unverified |
| `dsv2lite.compare_fleet` | `examples.dsv2lite.compare_fleet` | Runtime unverified |
| `dsv2lite.followup_2026_10_13` | `examples.dsv2lite.followup_2026_10_13` | Runtime unverified |
| `dsv2lite.inplace_session` | `examples.dsv2lite.inplace_session` | Runtime unverified |
| `dsv2lite.isolate_report` | `examples.dsv2lite.isolate_report` | Runtime unverified |
| `dsv2lite.ledger` | `examples.dsv2lite.ledger` | Runtime unverified |
| `dsv2lite.multibatch_trace` | `examples.dsv2lite.multibatch_trace` | Runtime unverified |
| `dsv2lite.multibatch_trace_summary` | `examples.dsv2lite.multibatch_trace_summary` | Runtime unverified |
| `dsv2lite.patch_fleet` | `examples.dsv2lite.patch_fleet` | Runtime unverified |
| `dsv2lite.patch_stamps` | `examples.dsv2lite.patch_stamps` | Runtime unverified |
| `dsv2lite.phase_table300` | `examples.dsv2lite.phase_table300` | Runtime unverified |
| `dsv2lite.pinned` | `examples.dsv2lite.pinned` | Runtime unverified |
| `dsv2lite.pregpu_report` | `examples.dsv2lite.pregpu_report` | Runtime unverified |
| `dsv2lite.reference_prepare` | `examples.dsv2lite.reference_prepare` | Runtime unverified |
| `dsv2lite.reference_rotated` | `examples.dsv2lite.reference_rotated` | Runtime unverified |
| `dsv2lite.replay` | `examples.dsv2lite.replay` | Runtime unverified |
| `dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.kv_convert` | `examples.dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.kv_convert` | Runtime unverified |
| `dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.model_analysis` | `examples.dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.model_analysis` | Runtime unverified |
| `dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.pack_weights` | `examples.dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.pack_weights` | Runtime unverified |
| `dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.reference_run` | `examples.dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.reference_run` | Runtime unverified |
| `dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.taskgraph` | `examples.dsv2lite.results.2026-10-14-k1.b_alld4r0l1kvs_stamps.fleet_src.host.taskgraph` | Runtime unverified |
| `dsv2lite.routing_overlap` | `examples.dsv2lite.routing_overlap` | Runtime unverified |
| `dsv2lite.routing_stats` | `examples.dsv2lite.routing_stats` | Runtime unverified |
| `dsv2lite.stage_table` | `examples.dsv2lite.stage_table` | Runtime unverified |
| `dsv2lite.verify_lever` | `examples.dsv2lite.verify_lever` | Runtime unverified |
| `dsv2lite.whatif` | `examples.dsv2lite.whatif` | Runtime unverified |
| `dsv2lite.x8_correctness` | `examples.dsv2lite.x8_correctness` | Runtime unverified |

## experiment_program (32)

| Command | Module | Status |
| --- | --- | --- |
| `experiment_program.columns.__main__` | `examples.experiment_program.columns.__main__` | Runtime unverified |
| `experiment_program.columns.build` | `examples.experiment_program.columns.build` | Runtime unverified |
| `experiment_program.columns.bundle` | `examples.experiment_program.columns.bundle` | Runtime unverified |
| `experiment_program.columns.capture_pack` | `examples.experiment_program.columns.capture_pack` | Runtime unverified |
| `experiment_program.columns.micro_build` | `examples.experiment_program.columns.micro_build` | Runtime unverified |
| `experiment_program.columns.patch_capture` | `examples.experiment_program.columns.patch_capture` | Runtime unverified |
| `experiment_program.columns.patch_historical_capture` | `examples.experiment_program.columns.patch_historical_capture` | Runtime unverified |
| `experiment_program.columns.seal_build` | `examples.experiment_program.columns.seal_build` | Runtime unverified |
| `experiment_program.columns.seal_capture` | `examples.experiment_program.columns.seal_capture` | Runtime unverified |
| `experiment_program.issue_order.__main__` | `examples.experiment_program.issue_order.__main__` | Runtime unverified |
| `experiment_program.issue_order.diagnostic` | `examples.experiment_program.issue_order.diagnostic` | Runtime unverified |
| `experiment_program.issue_order.diagnostic_match` | `examples.experiment_program.issue_order.diagnostic_match` | Runtime unverified |
| `experiment_program.issue_order.diagnostic_run` | `examples.experiment_program.issue_order.diagnostic_run` | Runtime unverified |
| `experiment_program.issue_order.integration` | `examples.experiment_program.issue_order.integration` | Runtime unverified |
| `experiment_program.issue_order.local_matrix_prepare` | `examples.experiment_program.issue_order.local_matrix_prepare` | Runtime unverified |
| `experiment_program.issue_order.matrix_archive` | `examples.experiment_program.issue_order.matrix_archive` | Runtime unverified |
| `experiment_program.issue_order.matrix_continue` | `examples.experiment_program.issue_order.matrix_continue` | Runtime unverified |
| `experiment_program.issue_order.matrix_finish` | `examples.experiment_program.issue_order.matrix_finish` | Runtime unverified |
| `experiment_program.issue_order.matrix_last` | `examples.experiment_program.issue_order.matrix_last` | Runtime unverified |
| `experiment_program.issue_order.receipts` | `examples.experiment_program.issue_order.receipts` | Runtime unverified |
| `experiment_program.local_acceptance` | `examples.experiment_program.local_acceptance` | Runtime unverified |
| `experiment_program.model_build` | `examples.experiment_program.model_build` | Runtime unverified |
| `experiment_program.publication_build` | `examples.experiment_program.publication_build` | Runtime unverified |
| `experiment_program.publication_run` | `examples.experiment_program.publication_run` | Runtime unverified |
| `experiment_program.router.__main__` | `examples.experiment_program.router.__main__` | Runtime unverified |
| `experiment_program.routing.__main__` | `examples.experiment_program.routing.__main__` | Runtime unverified |
| `experiment_program.row_ops.__main__` | `examples.experiment_program.row_ops.__main__` | Runtime unverified |
| `experiment_program.row_ops.padding` | `examples.experiment_program.row_ops.padding` | Runtime unverified |
| `experiment_program.runtime_link` | `examples.experiment_program.runtime_link` | Runtime unverified |
| `experiment_program.session` | `examples.experiment_program.session` | Runtime unverified |
| `experiment_program.stage` | `examples.experiment_program.stage` | Runtime unverified |
| `experiment_program.validation` | `examples.experiment_program.validation` | Runtime unverified |

## glm53 (19)

| Command | Module | Status |
| --- | --- | --- |
| `glm53.analyze_logs` | `examples.glm53.analyze_logs` | Runtime unverified |
| `glm53.bench` | `examples.glm53.bench` | Runtime unverified |
| `glm53.cost_model` | `examples.glm53.cost_model` | Runtime unverified |
| `glm53.cpu_check` | `examples.glm53.cpu_check` | Runtime unverified |
| `glm53.graph_cpu` | `examples.glm53.graph_cpu` | Runtime unverified |
| `glm53.model_build` | `examples.glm53.model_build` | Runtime unverified |
| `glm53.module_gate` | `examples.glm53.module_gate` | Runtime unverified |
| `glm53.mtp_graph` | `examples.glm53.mtp_graph` | Runtime unverified |
| `glm53.oracle` | `examples.glm53.oracle` | Runtime unverified |
| `glm53.pairs` | `examples.glm53.pairs` | Runtime unverified |
| `glm53.parallel_variants` | `examples.glm53.parallel_variants` | Runtime unverified |
| `glm53.prepare` | `examples.glm53.prepare` | Runtime unverified |
| `glm53.reference` | `examples.glm53.reference` | Runtime unverified |
| `glm53.residency` | `examples.glm53.residency` | Runtime unverified |
| `glm53.run` | `examples.glm53.run` | Runtime unverified |
| `glm53.run_all` | `examples.glm53.run_all` | Runtime unverified |
| `glm53.seam` | `examples.glm53.seam` | Runtime unverified |
| `glm53.stage` | `examples.glm53.stage` | Runtime unverified |
| `glm53.sweep` | `examples.glm53.sweep` | Runtime unverified |

## llm (5)

| Command | Module | Status |
| --- | --- | --- |
| `llm.model` | `examples.llm.model` | Runtime unverified |
| `llm.model_sliced` | `examples.llm.model_sliced` | Runtime unverified |
| `llm.prep` | `examples.llm.prep` | Runtime unverified |
| `llm.ref_numpy` | `examples.llm.ref_numpy` | Runtime unverified |
| `llm.vllm_bench` | `examples.llm.vllm_bench` | Runtime unverified |

## sessions (8)

| Command | Module | Status |
| --- | --- | --- |
| `sessions.inbody` | `examples.sessions.inbody` | Runtime unverified |
| `sessions.moe_rpt` | `examples.sessions.moe_rpt` | Runtime unverified |
| `sessions.nvchk` | `examples.sessions.nvchk` | Runtime unverified |
| `sessions.nvcov` | `examples.sessions.nvcov` | Runtime unverified |
| `sessions.qwen300` | `examples.sessions.qwen300` | Runtime unverified |
| `sessions.qwen300_hour` | `examples.sessions.qwen300_hour` | Runtime unverified |
| `sessions.vlbest` | `examples.sessions.vlbest` | Runtime unverified |
| `sessions.vlbest3` | `examples.sessions.vlbest3` | Runtime unverified |

## spec (1)

| Command | Module | Status |
| --- | --- | --- |
| `spec.ledger` | `examples.spec.ledger` | Runtime unverified |

## vllm_llm (74)

| Command | Module | Status |
| --- | --- | --- |
| `vllm_llm.arms` | `examples.vllm_llm.arms` | Runtime unverified |
| `vllm_llm.attention_early_merge.build` | `examples.vllm_llm.attention_early_merge.build` | Runtime unverified |
| `vllm_llm.attention_early_merge.build_model` | `examples.vllm_llm.attention_early_merge.build_model` | Runtime unverified |
| `vllm_llm.attention_early_merge.offline` | `examples.vllm_llm.attention_early_merge.offline` | Runtime unverified |
| `vllm_llm.attention_early_merge.session` | `examples.vllm_llm.attention_early_merge.session` | Runtime unverified |
| `vllm_llm.attention_early_merge.simulate` | `examples.vllm_llm.attention_early_merge.simulate` | Runtime unverified |
| `vllm_llm.autoconf` | `examples.vllm_llm.autoconf` | Runtime unverified |
| `vllm_llm.batch_correctness` | `examples.vllm_llm.batch_correctness` | Runtime unverified |
| `vllm_llm.batch_pp` | `examples.vllm_llm.batch_pp` | Runtime unverified |
| `vllm_llm.bench` | `examples.vllm_llm.bench` | Runtime unverified |
| `vllm_llm.bench_params` | `examples.vllm_llm.bench_params` | Runtime unverified |
| `vllm_llm.bf16_300.build` | `examples.vllm_llm.bf16_300.build` | Runtime unverified |
| `vllm_llm.bf16_300.inbody_cpu` | `examples.vllm_llm.bf16_300.inbody_cpu` | Runtime unverified |
| `vllm_llm.bf16_300.launch` | `examples.vllm_llm.bf16_300.launch` | Runtime unverified |
| `vllm_llm.bf16_300.launch_build` | `examples.vllm_llm.bf16_300.launch_build` | Runtime unverified |
| `vllm_llm.bf16_300.mutations` | `examples.vllm_llm.bf16_300.mutations` | Runtime unverified |
| `vllm_llm.bf16_300.pack` | `examples.vllm_llm.bf16_300.pack` | Runtime unverified |
| `vllm_llm.bf16_300.protocol` | `examples.vllm_llm.bf16_300.protocol` | Runtime unverified |
| `vllm_llm.bw_spec` | `examples.vllm_llm.bw_spec` | Runtime unverified |
| `vllm_llm.byte_ledger` | `examples.vllm_llm.byte_ledger` | Runtime unverified |
| `vllm_llm.check_uc_plan` | `examples.vllm_llm.check_uc_plan` | Runtime unverified |
| `vllm_llm.compare` | `examples.vllm_llm.compare` | Runtime unverified |
| `vllm_llm.decode` | `examples.vllm_llm.decode` | Runtime unverified |
| `vllm_llm.dense_bytes` | `examples.vllm_llm.dense_bytes` | Runtime unverified |
| `vllm_llm.dsl_ceiling` | `examples.vllm_llm.dsl_ceiling` | Runtime unverified |
| `vllm_llm.environment_guard` | `examples.vllm_llm.environment_guard` | Runtime unverified |
| `vllm_llm.expert_merge.capture` | `examples.vllm_llm.expert_merge.capture` | Runtime unverified |
| `vllm_llm.expert_merge.cpu` | `examples.vllm_llm.expert_merge.cpu` | Runtime unverified |
| `vllm_llm.expert_merge.inputs` | `examples.vllm_llm.expert_merge.inputs` | Runtime unverified |
| `vllm_llm.expert_merge.prepare` | `examples.vllm_llm.expert_merge.prepare` | Runtime unverified |
| `vllm_llm.expert_merge.protocol` | `examples.vllm_llm.expert_merge.protocol` | Runtime unverified |
| `vllm_llm.expert_merge.runner` | `examples.vllm_llm.expert_merge.runner` | Runtime unverified |
| `vllm_llm.expert_merge.validate` | `examples.vllm_llm.expert_merge.validate` | Runtime unverified |
| `vllm_llm.folds` | `examples.vllm_llm.folds` | Runtime unverified |
| `vllm_llm.g6_session` | `examples.vllm_llm.g6_session` | Runtime unverified |
| `vllm_llm.head_local_protocol` | `examples.vllm_llm.head_local_protocol` | Runtime unverified |
| `vllm_llm.import_certificate` | `examples.vllm_llm.import_certificate` | Runtime unverified |
| `vllm_llm.isa_inventory` | `examples.vllm_llm.isa_inventory` | Runtime unverified |
| `vllm_llm.kcmp_report` | `examples.vllm_llm.kcmp_report` | Runtime unverified |
| `vllm_llm.kmodel` | `examples.vllm_llm.kmodel` | Runtime unverified |
| `vllm_llm.layer_trace` | `examples.vllm_llm.layer_trace` | Runtime unverified |
| `vllm_llm.measured` | `examples.vllm_llm.measured` | Runtime unverified |
| `vllm_llm.moe_predict` | `examples.vllm_llm.moe_predict` | Runtime unverified |
| `vllm_llm.nvref_report` | `examples.vllm_llm.nvref_report` | Runtime unverified |
| `vllm_llm.pairs` | `examples.vllm_llm.pairs` | Runtime unverified |
| `vllm_llm.plan_exports` | `examples.vllm_llm.plan_exports` | Runtime unverified |
| `vllm_llm.plan_imports` | `examples.vllm_llm.plan_imports` | Runtime unverified |
| `vllm_llm.pmodel` | `examples.vllm_llm.pmodel` | Runtime unverified |
| `vllm_llm.polling_isa_cpu` | `examples.vllm_llm.polling_isa_cpu` | Runtime unverified |
| `vllm_llm.prefetch_isolation_cpu` | `examples.vllm_llm.prefetch_isolation_cpu` | Runtime unverified |
| `vllm_llm.prep` | `examples.vllm_llm.prep` | Runtime unverified |
| `vllm_llm.producer_local` | `examples.vllm_llm.producer_local` | Runtime unverified |
| `vllm_llm.producer_replay` | `examples.vllm_llm.producer_replay` | Runtime unverified |
| `vllm_llm.producer_session` | `examples.vllm_llm.producer_session` | Runtime unverified |
| `vllm_llm.producer_stage` | `examples.vllm_llm.producer_stage` | Runtime unverified |
| `vllm_llm.router_study` | `examples.vllm_llm.router_study` | Runtime unverified |
| `vllm_llm.router_study_guards` | `examples.vllm_llm.router_study_guards` | Runtime unverified |
| `vllm_llm.sensitivity` | `examples.vllm_llm.sensitivity` | Runtime unverified |
| `vllm_llm.tile_ab` | `examples.vllm_llm.tile_ab` | Runtime unverified |
| `vllm_llm.touch_arm_cpu` | `examples.vllm_llm.touch_arm_cpu` | Runtime unverified |
| `vllm_llm.validate` | `examples.vllm_llm.validate` | Runtime unverified |
| `vllm_llm.vendor.benchmark_moe` | `examples.vllm_llm.vendor.benchmark_moe` | Broken here: missing direct import `ray`, `vllm` |
| `vllm_llm.vendor_gemv` | `examples.vllm_llm.vendor_gemv` | Runtime unverified |
| `vllm_llm.verify_gaps_cpu` | `examples.vllm_llm.verify_gaps_cpu` | Runtime unverified |
| `vllm_llm.vlbest3_g0` | `examples.vllm_llm.vlbest3_g0` | Runtime unverified |
| `vllm_llm.vlbest3_tuner` | `examples.vllm_llm.vlbest3_tuner` | Runtime unverified |
| `vllm_llm.vlbest_report` | `examples.vllm_llm.vlbest_report` | Runtime unverified |
| `vllm_llm.vllm_profile` | `examples.vllm_llm.vllm_profile` | Runtime unverified |
| `vllm_llm.vm_preflight` | `examples.vllm_llm.vm_preflight` | Runtime unverified |
| `vllm_llm.watchdog_cpu` | `examples.vllm_llm.watchdog_cpu` | Runtime unverified |
| `vllm_llm.whatif_sim` | `examples.vllm_llm.whatif_sim` | Runtime unverified |
| `vllm_llm.x5` | `examples.vllm_llm.x5` | Runtime unverified |
| `vllm_llm.xcd_account` | `examples.vllm_llm.xcd_account` | Runtime unverified |
| `vllm_llm.xcd_run` | `examples.vllm_llm.xcd_run` | Runtime unverified |
