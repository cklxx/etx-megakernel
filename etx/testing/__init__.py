"""Supported test-policy probes; implementation details stay behind this surface."""
from .boundary import check_external_sources, check_api_layers, check_migrated_entry_points
from .fixtures import fixture
from .conformance import conformance, ConformanceResult, forbid_compilation
from .inventory import check_test_inventory
from .executors import scripted_executor, recorded_executor, protocol_fault_executor
from .jobs import (observe_job_processes, observe_unrelated_child, job_runner_probe,
                   job_transport_interrupt_probe)
from .jobs_faults import job_fault_probe
from .historical import sealed_original_host
from .packaging import installed_package_probe
from .families import census as scaffold_census
from .models import model_isolation_probe, synthetic_llm_checkpoint
from .commands import run_command
from .arrays import write_npz
from .qualification import qualification_record, qwen_staged_model
from .certificates import certified_import_session, certificate_envelope, timing_log
from .glm import glm_cpu_stage, glm_module_checkpoint, glm_reference_record
from .ledger import packed_manifest_text
from .traces import simulated_trace_files
from .dsv2lite import dsv2lite_conformance
from .gemm_rs import provider_conformance as gemm_provider_conformance
from .protocol import protocol_conformance, provenance_conformance, operator_provider_conformance
from .acceptance import (acceptance_results, verify_acceptance_archive,
                         acceptance_dependency_failure, acceptance_probe)
from .sealed import (synthetic_sealed_stages, interrupted_sealed_export,
                     sealed_child_probe, sealed_cleanup_interrupt_probe,
                     sealed_resume_probe, sealed_dependency_probe,
                     sealed_cli_list_probe, sealed_historical_replay_probe,
                     sealed_provenance_probe, sealed_contract_probe)

__all__ = ['check_external_sources', 'check_api_layers', 'check_migrated_entry_points', 'external_test_policy', 'fixture', 'conformance',
           'ConformanceResult', 'forbid_compilation', 'check_test_inventory', 'scripted_executor', 'recorded_executor',
           'protocol_conformance', 'provenance_conformance', 'operator_provider_conformance', 'protocol_fault_executor',
           'acceptance_results', 'verify_acceptance_archive', 'acceptance_dependency_failure',
           'acceptance_probe', 'observe_job_processes', 'observe_unrelated_child', 'job_runner_probe', 'job_fault_probe', 'job_transport_interrupt_probe',
           'synthetic_sealed_stages', 'interrupted_sealed_export', 'sealed_child_probe',
           'sealed_cleanup_interrupt_probe',
           'sealed_resume_probe', 'sealed_dependency_probe', 'sealed_cli_list_probe',
           'sealed_historical_replay_probe', 'sealed_provenance_probe', 'sealed_original_host', 'sealed_contract_probe',
           'installed_package_probe', 'scaffold_census', 'model_isolation_probe', 'synthetic_llm_checkpoint', 'run_command', 'write_npz', 'qualification_record', 'qwen_staged_model', 'certified_import_session', 'certificate_envelope', 'timing_log', 'glm_cpu_stage', 'glm_module_checkpoint', 'glm_reference_record', 'packed_manifest_text', 'simulated_trace_files', 'gemm_provider_conformance', 'dsv2lite_conformance',
           'plan_export_instances']


def plan_export_instances(*args, **kwargs):
    from .plan_exports import plan_export_instances as probe
    return probe(*args, **kwargs)


def external_test_policy(root):
    """Create the pytest fixture-boundary plugin (requires the dev extra)."""
    from .pytest_policy import external_test_policy as create
    return create(root)
