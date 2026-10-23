"""Public library contracts. Declaration does not imply provider support or execution."""
from collections.abc import Mapping
from .artifacts import SourceArtifact
from .compiler import CompilerRequest
from .build import DeviceCheck, BuildArtifact
from .runs import ArtifactRunRequest, ArtifactRun
from .sealed import SealedPreparationRequest, SealedSessionRequest
from .execution import ProcessRequest, ProcessResult
from .transport import SessionTransportRequest
from .models import BuildRefused, ModelBuildRequest
from .sessions import ExperimentSessionRequest
from .probes import NativeProbeRequest
from .assets import AssetPreparationRequest
from .harness import HarnessCommandRequest
from .jobs import JobSpec, JobSession, JobSessionRequest, JobOutcome, JobSessionResult
from .types import (Configuration, Inspection, KernelSpec, PlanRequest, PlannedWorkload,
                    SCHEMA_VERSION, SimulationResult, WorkloadSpec)


__all__ = ['Configuration', 'Inspection', 'KernelSpec', 'PlanRequest', 'PlannedWorkload',
           'SCHEMA_VERSION', 'SimulationResult', 'WorkloadSpec', 'model', 'kernel', 'inspect', 'plan',
           'simulate', 'ProcessRequest', 'ProcessResult', 'execute', 'read_process_result',
           'SourceArtifact', 'emit', 'save_sources', 'load_sources', 'DeviceCheck', 'check_device_sources',
           'check_codegen', 'BuildArtifact', 'build', 'load_build',
           'ArtifactRunRequest', 'ArtifactRun', 'run_artifact', 'read_artifact_run',
           'source_snapshot', 'verify_source_snapshot', 'CompilerRequest', 'compiler_command', 'environment_options',
           'run_protocol_experiment', 'report_protocol_experiment', 'run_protocol_study',
           'run_cpu_acceptance', 'cpu_acceptance_fixtures', 'JobSpec', 'JobSession', 'JobSessionRequest',
           'JobOutcome', 'JobSessionResult', 'load_job_session', 'run_jobs', 'verify_job_record',
           'SealedPreparationRequest', 'SealedSessionRequest', 'prepare_sealed_session', 'run_sealed_session',
           'SessionTransportRequest', 'run_session_transport', 'BuildRefused', 'ModelBuildRequest', 'build_model',
           'ExperimentSessionRequest', 'run_experiment_session', 'experiment_session_arms',
           'NativeProbeRequest', 'run_native_probe', 'AssetPreparationRequest', 'prepare_model_assets',
           'HarnessCommandRequest', 'run_harness_command']


def run_harness_command(request: HarnessCommandRequest) -> int:
    """One historical command of a registered experiment harness; its own gates decide what may run."""
    from etx.contracts.harness import run_harness_command as run
    return run(request)


def prepare_model_assets(request: AssetPreparationRequest) -> int:
    """Pack a model's weights and write its references; may download and use a GPU as the provider documents."""
    from etx.contracts.models import prepare_model_assets as prepare
    return prepare(request)


def run_native_probe(request: NativeProbeRequest) -> int:
    """Plan, build, run or check a native microbenchmark probe; runs touch the GPU of the machine it is on."""
    from etx.contracts.probes import run_native_probe as run
    return run(request)


def experiment_session_arms(provider: str, root) -> Configuration:
    """The arms each arm-taking operation of a model's experiment session accepts, declared by repository root."""
    from etx.contracts.models import experiment_session_arms as arms
    return Configuration(arms(provider, root))


def run_experiment_session(request: ExperimentSessionRequest) -> int:
    """One operation of a model's prepared experiment session; device operations run only on its box."""
    from etx.contracts.models import run_experiment_session as run
    return run(request)


def build_model(request: ModelBuildRequest) -> Configuration:
    """Plan a model workload and link it with its host; EXECUTABLE is not device qualification."""
    from etx.contracts.models import build_model as build
    return build(request)


def run_session_transport(request: SessionTransportRequest) -> int:
    """Drive a GPU box (stage/push/preflight/start/status/fetch/postmortem); provisioning stays separate."""
    from etx.contracts.transport import run_session_transport as run
    return run(request)


def prepare_sealed_session(request: SealedPreparationRequest) -> Configuration:
    from etx.contracts.sealed import prepare_sealed_session as prepare
    return prepare(request)


def run_sealed_session(request: SealedSessionRequest, *, render=False) -> Configuration:
    from etx.contracts.sealed import run_sealed_session as run
    return run(request, render=render)


def load_job_session(path) -> JobSession:
    from etx.contracts.jobs import load_job_session as load
    return load(path)


def run_jobs(request: JobSessionRequest) -> JobSessionResult:
    from etx.contracts.jobs import run_jobs as run
    return run(request)


def verify_job_record(directory) -> Configuration:
    from etx.contracts.jobs import verify_job_record as verify
    return verify(directory)


def run_cpu_acceptance(output, *, formal=False, require_external=False, rocm_image=None,
                       rocm_root=None, local_matrix=False) -> int:
    from etx.contracts.acceptance import run_cpu_acceptance as run
    return run(output, formal=formal, require_external=require_external, rocm_image=rocm_image,
               rocm_root=rocm_root, local_matrix=local_matrix)


def cpu_acceptance_fixtures(output) -> int:
    from etx.contracts.acceptance import cpu_acceptance_fixtures as fixtures
    return fixtures(output)


def run_protocol_study(name: str, *, values=None, size=None) -> Configuration:
    from etx.contracts.experiments import run_protocol_study as run
    return run(name, values=values, size=size)


def run_protocol_experiment(config, *, output, executor=None) -> int:
    from etx.contracts.experiments import run_protocol_experiment as run
    return run(config, output=output, executor=executor)


def report_protocol_experiment(output) -> int:
    from etx.contracts.experiments import report_protocol_experiment as report
    return report(output)


def compiler_command(request: CompilerRequest) -> int:
    from etx.contracts.compiler import compiler_command as command
    return command(request)


def environment_options(environ) -> Configuration:
    """Planner options selected by a build environment's ETX_COHERENCE, ETX_PREFETCH_ISOLATE_POLLER, ETX_INBODY_EMIT.

    The compiler never reads the process environment; commands pass their environment through this.
    """
    from etx.contracts.planning import environment_options as options
    return Configuration(options(environ))


def source_snapshot(root, *, paths) -> Configuration:
    from etx.contracts.emission import source_snapshot as snapshot
    return snapshot(root, paths=paths)


def verify_source_snapshot(root, record) -> Configuration:
    from etx.contracts.emission import verify_source_snapshot as verify
    return verify(root, record)


def build(artifact: SourceArtifact, *, output, docker, image, deadline_s=90) -> BuildArtifact:
    from etx.contracts.emission import build as compile_and_link
    return compile_and_link(artifact, output=output, docker=docker, image=image, deadline_s=deadline_s)


def run_artifact(request: ArtifactRunRequest, *, output, executor=None, execute_native=False) -> ArtifactRun:
    from etx.contracts.runs import run_artifact as run
    return run(request, output=output, executor=executor, execute_native=execute_native)


def read_artifact_run(directory, *, request: ArtifactRunRequest) -> ArtifactRun:
    from etx.contracts.runs import read_artifact_run as read
    return read(directory, request=request)


def load_build(directory) -> BuildArtifact:
    from etx.contracts.emission import load_build as load
    return load(directory)


def check_device_sources(artifact: SourceArtifact, *, output, docker, image, deadline_s=90) -> DeviceCheck:
    from etx.contracts.emission import check_device_sources as check
    return check(artifact, output=output, docker=docker, image=image, deadline_s=deadline_s)


def check_codegen(codegen, *, toolchain, docker='docker', logs=None, deadline_s=600) -> Configuration:
    """Offline compile/link of an `etx compile` directory; LINKED is not device qualification."""
    from etx.contracts.emission import check_codegen as check
    return check(codegen, toolchain=toolchain, docker=docker, logs=logs, deadline_s=deadline_s)


def emit(planned: PlannedWorkload) -> SourceArtifact:
    from etx.contracts.emission import emit as emit_sources
    return emit_sources(planned)


def save_sources(artifact: SourceArtifact, directory) -> SourceArtifact:
    from etx.contracts.emission import save_sources as save
    return save(artifact, directory)


def load_sources(directory) -> SourceArtifact:
    from etx.contracts.emission import load_sources as load
    return load(directory)


def execute(request: ProcessRequest, *, output, executor=None) -> ProcessResult:
    from etx.contracts.execution import execute as execute_process
    return execute_process(request, output=output, executor=executor)


def read_process_result(directory, *, request: ProcessRequest) -> ProcessResult:
    from etx.contracts.execution import read_process_result as read_result
    return read_result(directory, request=request)


def simulate(planned: PlannedWorkload, *, seed: int = 0) -> SimulationResult:
    from etx.contracts.simulation import simulate as simulate_protocol
    return simulate_protocol(planned, seed=seed)


def plan(request: PlanRequest) -> PlannedWorkload:
    from etx.contracts.planning import plan as plan_graph
    return plan_graph(request)


def model(name: str, *, config: Mapping) -> WorkloadSpec:
    return WorkloadSpec(name=name, kind='model', config=Configuration(config))


def kernel(spec: KernelSpec) -> WorkloadSpec:
    if not isinstance(spec, KernelSpec):
        raise TypeError('kernel requires KernelSpec')
    return WorkloadSpec(name=spec.name, kind=spec.implementation, config=spec.config)


def inspect(subject: WorkloadSpec) -> Inspection:
    if not isinstance(subject, WorkloadSpec):
        raise TypeError('inspect currently accepts WorkloadSpec only')
    return Inspection(SCHEMA_VERSION, subject.identity, subject)
