# API consolidation contract

## Layers and target tree

- `etx.api` owns public versioned library records and lifecycle implementation.
- `etx.external` is the supported caller facade; CLI adapters and E2E tests import it, never compiler objects.
- `etx.testing` is the supported test/conformance facade; internal probes execute behind it.
- `etx.external` delegates to `etx.api`; it is not a second implementation or a second runner.
- Internal planner/codegen/runtime/staging adapters remain implementation details.

```text
etx/
  external/           caller facade; CLI delegates here
  api/                immutable public declarations and lifecycle records
  testing/            fixtures, test executors and internal conformance probes
  contracts/          internal provider/planner/emitter/executor protocols
  artifacts/          one asset/build identity and validation contract
  build/              source emission, toolchain and linker adapters
  execution/          one validated runner; native hosts are adapters
  experiments/        one task/experiment engine
  evidence/           qualification and replayable verdicts
  models/             model-provider implementation, not separate CLIs
  operators/          graph/imported/native operator providers
  passes/, codegen/, runtime/, importer/, session/  existing internals reused
  tools/              CLI/analysis adapters, not independent lifecycles
examples/<model>/     workload/asset declarations (compatibility wrappers during migration)
tests/e2e/           caller-only integration tests and helpers
```

- The tree is a target, not a claim that these modules all exist.
- Existing commands remain operational until their wrappers are migrated; no command is removed by the declaration slice.
- Compatibility wrappers are transitional; model directories must not retain independent runner/build implementations after their merge gate.

## Implemented slice

```python
from etx.external import KernelSpec, WorkloadSpec, inspect, kernel, model

workload = model('glm53', config={'batch': 2, 'options': {'devices': 8}})
operator = kernel(KernelSpec('handoff', 'native', {'entry': 'handoff'}))
record = workload.to_dict()
assert WorkloadSpec.from_dict(record) == workload
assert inspect(workload).status == 'DECLARED'
```

- Schema v1 supports declarations for model, graph, imported and native workloads.
- Configuration accepts only finite JSON values and string keys; nested containers are copied on input and output.
- Identity covers schema, name, kind and complete configuration with canonical JSON ordering.
- A declaration does not resolve a provider, validate model shapes, prepare assets, compile, launch or establish correctness.
- `plan(PlanRequest(...))` now supports declarative graph workloads through the existing compiler; it returns `PlannedWorkload` with detached machine/options/plan snapshots and a deterministic identity.
- Graph config contains `tensors`, `events`, `grids` and optional `codegen`, or selects one registered `provider` (`splitk_sum`, `moe_layer` or two-rank `gemm_rs`). These forms cannot be mixed. Provider bindings are explicit (`n`, `B`/`seed`, or GEMM `M`/`K`/`N`); derived runtime tensors and `n_gg` cannot be overridden. GEMM retains its supported two-rank native contract and requires positive M/K/N multiples of 256/32/128. MoE's optional boolean `small` is request-local, never read from `ETX_MOE_SMALL` by declarative planning. Model/imported/native providers remain rejected until their adapters land.
- Packaged target names only are supported in this slice; arbitrary target files are not implicitly loaded. Public planning records explicit options and ignores ambient `ETX_COHERENCE`; legacy compiler behavior is unchanged. The planner and code generator (`etx/ir`, `etx/passes`, `etx/codegen`) never read the process environment: `coherence`, `prefetch_isolate_poller` and `inbody_emit` are `PassOptions` fields, and the commands that historically inherited `ETX_COHERENCE`, `ETX_PREFETCH_ISOLATE_POLLER` and `ETX_INBODY_EMIT` (`python -m etx compile`, `etx.tools.model_build`) translate them with `environment_options(os.environ)`. Records list the two emission switches only when they are on.
- `simulate(planned, seed=...)` revalidates the plan against the current compiler/machine before invoking the existing CPU protocol simulator. Its backend is `simulation` and its timing scope is `simulated_protocol`, never device timing or numerical qualification.
- `etx.testing.fixture('handoff', devices=..., batch=...)` declares the tiny protocol fixture; `conformance('planning', ...)` checks request isolation, compiler verification, ownership and SYSTEM scope behind the testing facade. This is not a GLM provider or numerical fixture.
- `emit(planned)` revalidates the plan and uses existing emitters to return a `SourceArtifact`: every rank kernel/header, rank-zero compatibility header, runtime headers, plan JSON and decisions. `save_sources` exclusively creates a directory with the manifest written last; `load_sources` checks identity, payload hashes, rank coverage and relative non-symlink payloads. Relocation is supported. `SOURCES` is not compiled or executable.
- `check_device_sources(artifact, output=..., docker='/usr/bin/docker', image=...)` handles self-contained gfx942 artifacts only, resolves an already-installed image to its immutable ID, uses `--pull=never --network=none` without GPU devices, retains full per-rank logs and explicitly removes each named container even on timeout. HIP RDC emission is followed by AMDGPU backend compilation; the output must be ELF64 little-endian with `EM_AMDGPU`, not a host object containing an offload/bitcode bundle. Every rank and every captured native tile unit must succeed for `DEVICE_CHECKED`; `DeviceCheck.tiles` records tile invocations separately from ranks and the object inventory covers all successful units, including partial failures. Native recipes are validated against explicit provider dimensions before output creation; no ambient macro selection is used. Objects still need a later link gate. This is neither an executable nor numerical, residency or performance qualification. Imported/external tile source closure remains pending; registered operator linking is the separate build gate below.
- Source identities cover emitted payloads/runtime snapshots and planned requests; full generator/import/toolchain provenance and signed/trusted qualification are not yet an artifact contract. The offline receipt records its compiler/image identity and invocation logs, separately from any production VM compiler.
- `build(source_artifact, output=..., docker=..., image=...)` reuses the same offline compiler/container/process path, then links captured rank/tile/host sources only after all device checks succeed. Registered Split-K, default/small MoE and two-rank GEMM are supported on gfx942. `BuildArtifact` records the device receipt, actual linker outcome and executable hash; `EXECUTABLE` requires a linked x86-64 ELF with a nonempty HIP device-image section, not numerical qualification. Failed checks block linking; failed links retain evidence but cannot supply an executable payload. No binary is launched.
- `load_build(directory)` validates source identity, unit inventories/recipes, objects, compiler/image/version/cleanup and linker process records, binary hash and HIP section. Relocation works without consulting checkout files. Missing/corrupt evidence and symlink payloads refuse. Legacy MoE host supports seed 12345 only; other declared seeds refuse before building. Multi-rank inline tile linking refuses duplicate-definition translation units; separate link mode remains supported.
- `run_artifact(ArtifactRunRequest(...), output=..., executor=..., execute_native=False)` validates the complete retained build against the caller's expected identity before invoking the existing process supervisor. Requests bind the artifact directory, build identity, absolute cwd, arguments, explicit environment and deadlines. No automatic native execution: without an explicit substitute, `execute_native=True` is required, and the linked host requires Linux x86-64. This opt-in is not a device-readiness or numerical gate. Desktop tests use real retained linked artifacts with explicitly scripted/recorded execution; no generated executable is launched.
- `ArtifactRun` binds the request and `ProcessResult`; `read_artifact_run(directory, request=...)` checks identity and full log integrity after evidence relocation. It validates historical process evidence without requiring the original build to remain present; it is not a new launch authorization or build verification. Neither exit zero nor scripted/recorded completion is qualification, and no host-process duration is advertised as GPU timing. Inputs/assets beyond the build, device compatibility/residency and numerical/performance qualification remain pending. The caller must keep executable files stable during launch; integrity checks are not a sandbox against concurrent mutation.
- `prepare`, `qualify` and general experiment/qualification contracts remain pending; protocol and CPU-acceptance compatibility entry points exist below, without success-shaped stubs.
- `execute(ProcessRequest(...), output=...)` is a lower-level bounded process capability, not an artifact runner: absolute argv/cwd, explicit environment (not inherited), deadline/silence intervals, exclusive output directory, full binary stdout/stderr, supervisor diagnostics and request-bound checksummed results. The request record includes the supplied environment; do not pass secrets that should not be persisted.
- Host execution transports the request to an isolated worker using the existing `etx.session.proc.run` supervisor; it does not sweep children of the caller. The supervisor's cleanup allowance is up to 30 seconds beyond the command deadline. `host-process` means a host command ran, not proof that its child used only a CPU or that GPU numerics are correct.
- `scripted_executor(request, output=bytes, returncode=...)` supplies one exact expected response without executing argv. `recorded_executor(directory)` rechecks saved request/output evidence and always labels replay as `recorded`. Neither creates a build certificate or device qualification. `read_process_result(directory, request=...)` rejects incomplete, mismatched or output-corrupt evidence; these hashes are integrity checks, not authenticated provenance.
- Future records add explicit assets, bindings, target/toolchain, gate identity and result evidence rather than exposing a mutable compiler `Plan`.

## Planning and simulation through the same facade

```python
from etx.external import PlanRequest, plan, simulate
from etx.testing import fixture, conformance

workload = fixture('handoff', devices=2, batch=4)
planned = plan(PlanRequest(workload, options={'force_mode': 'static'}))
result = simulate(planned, seed=17)
assert result.backend == 'simulation'
assert not result.deadlock
probe = conformance('planning', devices=2, batch=4)
```

## Registered operators use the same lifecycle

```python
from etx.external import KernelSpec, PlanRequest, kernel, plan, simulate, emit

workload = kernel(KernelSpec('moe', 'graph', {'provider': 'moe_layer', 'small': True}))
planned = plan(PlanRequest(workload, bindings={'B': 4, 'seed': 12345}))
assert not simulate(planned, seed=17).deadlock
source = emit(planned)
assert source.status == 'SOURCES'
assert not source.external_sources
assert 'examples/tiles/moe.hip' in source.files  # Logical path, captured package bytes.
```

The split-K, MoE and GEMM implementations live once in `etx/operators`; examples delegate. Native bodies now live in `etx/operators/tiles`, with compatibility symlinks at their original paths. Their legacy graph names and native tile-source strings are preserved. Desktop public emission matches six legacy output files for each of 24 captured plan cases (default/small capture × two operators × size 2/8 × three scheduling modes; split-K repeats across the two captures). Source artifacts additionally capture the exact tile bytes under those logical paths and a checksummed `etx_native_inputs.json` payload listing HIP units and explicit D/H/S/FF macros, including small-mode dimensions. The additional payloads intentionally change artifact identity, not legacy generated bytes. Unknown/custom tile sources remain external. The device checker compiles these recorded native units separately and refuses missing or inconsistent recipes before creating outputs; inline rank compilation receives the same explicit dimensions. Source closure alone does not prove the units compile, and object compilation does not prove they link. This source stage alone is not a model provider, linked build or device qualification; `build` separately performs the linking described above.

The native-input packaging regression passed **419 tests, 3 skips** on desktop, then reproduced all 24 × six legacy payload hashes. Offline ROCm 6.4.1 compilation passed **7 tests**: the existing eight-rank success/failure checks, Split-K, default/small MoE, small-inline MoE, and a deliberately broken native tile. The broken tile leaves its rank successful but forces the whole check to `FAILED`; complete source and compiler logs are retained in `API-NATIVE-DEVICE-EVIDENCE-20261023-01/`. These are object-compilation checks only, without linking, GPU devices, numerics or performance.

## Migrated compatibility entry points

The existing CLI now also exposes `etx sources request.json --out sources`, `etx build sources --out build --docker /usr/bin/docker --image <installed-image>`, and `etx inspect-build build`. `sources` consumes the JSON returned by `PlanRequest.to_dict()`; these adapters call the same external lifecycle. Build/inspection return nonzero for failed builds and never launch the executable. Desktop API link suite: **12 passed in 59.50s**, including two-rank GEMM, failure gates, relocation and source/object/executable/compiler-log corruption. Actual CLI build and inspect output matched the API receipt (`API-BUILD-CLI-20261023.log`). Historical compile commands remain source-only.

The same CLI provides `etx run request.json --out evidence --execute` and `etx inspect-run evidence --request request.json`, consuming `ArtifactRunRequest.to_dict()`. Omitting `--execute` refuses before native execution; the inspection command validates saved evidence without executing the artifact. The desktop runner checks use the prior real Split-K build through `ETX_API_RUNNER_BUILD`, so no recompilation is needed to test scripted success/failure/deadline, replay, relocation and prelaunch corruption rejection.

- `compiler_command(CompilerRequest(...))` now owns `compile/explain/archs`; the CLI only parses arguments. It shares payload emission with source artifacts while preserving the historical file inventory and source-only meaning of `compile`. Explicit graph Python files are executable input. Legacy custom target files and ambient/default options remain supported; this compatibility request is not the isolated declarative `PlanRequest` path.
- `run_protocol_experiment(config, output=...)` and `report_protocol_experiment(output)` own the existing `bench_sim run/report` workflow. Immutable JSON configuration crosses the API; internal plans do not. Pair order, seed identity, full sample traces, checksum checks, failure invalidation and pure offline report recomputation remain intact.
- `source_snapshot(root, paths=...)` records an explicit filesystem scope, with `commit=None` and `dirty=None`; `verify_source_snapshot` detects additions, removals and mutations in that scope. Copied-tree protocol runs use this snapshot rather than inventing Git metadata and revalidate it before their final manifest. Git checkouts retain their historical provenance. A scoped snapshot is not a whole-repository or authenticated identity.
- `run_cpu_acceptance(output, ...)` and `cpu_acceptance_fixtures(output)` own the CPU acceptance lifecycle. `etx.tools.cpu_check` is a thin compatibility CLI. Acceptance retains its inherited environment policy and internal process supervisor, which does **not** serialize inherited environment secrets as public process requests. Whole-suite inventory, external-skip classification, prerequisites, logs and optional explicit compiler/formal/matrix gates are preserved. Focused tests are not a full acceptance pass.
- `run_protocol_study('variance' | 'routing-skew', values=..., size=...)` returns immutable, explicitly simulation-only rows and uses the same simulation/trace-validation boundary as protocol experiments. Historical seed/routing/default compiler options remain unchanged; those options intentionally differ from the frozen paired benchmark. `bench/sim_experiments.py` only renders these rows. Optional `--size` permits explicit reduced checks without changing default n=256/B=32. Reduced legacy parity does not establish full-size completion; the old full-size study exceeded the bounded desktop capture window.
- `run_jobs(JobSessionRequest(...))` owns scheduler entry-point execution through the extracted `etx.experiments.jobs` engine. `JobSpec` and `JobSession` are immutable declarations; `load_job_session(path)` explicitly executes a legacy Python JOBS file and returns immutable declarations. A request may also name that file directly for compatibility. CLI defaults, lane ordering/pin folding, dependencies, environment/interpreter precedence, progress, deadlines, machine/only selection, leases and successful sealed records retain historical behavior; failure-path repairs are listed below. Declarative job names permit legacy dots but reject path traversal; requests reject empty GPU-lane availability when a GPU job would otherwise wait forever.
- Scheduler outcomes expose immutable timestamps/status/log paths, not live `Job`, `Runner` or process handles. `verify_job_record` checks existing record integrity, not device correctness or authenticated provenance. `host-process` means shell children ran; `not-executed` labels dry runs. CPU commands on logical GPU lanes do not establish hardware execution. Public requests run the same scheduler in an isolated worker, preventing its subreaper sweep from reaching unrelated caller children or concurrent scheduler calls. Inherited environment stays internal, not serialized into transport/result records. Linux parent death requests worker cleanup; interruption waits for supervisor cleanup with periodic diagnostics rather than killing the supervisor out from under its children. Forced termination of the worker itself and kernel-stuck processes are not proven recoverable.
- Ordinary scheduler exceptions now clean owned job trees before sealing/releasing the lease. Aborted seals explicitly record `failure` and remain incomplete even if all job commands had already returned success; cleanup failure leaves the record unsealed. Effective lane dependencies are checked after GPU-pin folding. These intentionally repair failure/deadlock paths; successful job scheduling and seal format are preserved. The legacy loader normalizes path-like cwd and stringifies declared environment values as the historical launcher did. Common artifact/qualification/resume integration remains pending.
- `etx.tools.gpu_jobs` now delegates execution to the external API but temporarily reexports legacy `Job`, `Runner`, loader and budget symbols for unmigrated callers. Transport/provisioning shells and record/experiment CLIs remain migration debt; S7 is not complete. No automatic provisioning is introduced.
- `prepare_sealed_session(SealedPreparationRequest(...))` and `run_sealed_session(SealedSessionRequest(...))` own sealed preparation, list/dry-run/preflight/run/resume/export. Requests and returned `Configuration` snapshots are detached; run still requires explicit activation. Registered module/interpreter allowlists, raw admission recomputation, input seals, dependencies, budgets and incomplete-state refusal are preserved. `render=True` preserves legacy CLI rendering before immutable JSON normalization. No command is removed.
- S6 preparation/session/evidence/source-inventory implementations live in the library; old module paths delegate or reexport for unmigrated consumers. Source snapshots now include library `etx/build` and source files named `build.*`, rather than mistaking them for generated build directories. Copied trees record unknown Git identity and a separate inventory digest; declared revisions remain explicitly labelled, not authenticated. Flat relocation-aware seals remain compatible.
- Stage commands reuse `etx.session.proc.run` in an isolated worker through the same worker transport as jobs. Ambient environment is inherited internally, never serialized. Full binary logs and ordinary exit codes (including 124) are preserved; timeout and interruption now use the common descendant sweep instead of the old process-group-only loop. Signals request interruption without raising inside cleanup. Raw logs remain after failure; no shortened run becomes successful. Kernel-stuck/force-killed-worker recovery is unqualified.
- Synthetic stage dispatch is bounded behind `etx.testing`, writes explicitly non-admitted CPU-only decisions, and restores nested/exceptional contexts. Historical source fixtures carry fixed revision/path/checksum receipts; they do not claim fresh hardware evidence. Native providers/builders and their resources remain in examples, so S6 is partial, not consolidated. The single catalog declaration now lives at `etx/experiments/specs.json`, ships in the wheel, and the old examples path is a compatibility symlink.
- G04 routing configuration, raw decoding, log gates and session receipt verification now live in `etx.experiments.sealed_routing`. Legacy routing imports reexport the same functions; the column consumer uses the same primitives but retains its extra exact-file-set predicate and `ProtocolError`. The session retains its separate config-digest binding. Neither path promotes capture completion to optimization admission. The native capture CLI/process loop is not yet consolidated.
- Protocol and study graph/runtime providers now use package-owned split-K/MoE implementations, with exact pre-extraction desktop signatures for default and small modes. Fresh legacy graph-file loads still select independent import-time MoE dimensions; protocol frozen benchmarks retain their default-size guard. Compatibility studies still read their historical small-mode environment selection. Non-Git snapshots always cover the library and also include checkout wrappers/project metadata when present; persisted-scope mutation checks remain mandatory. Installed run/report and reduced studies work without `examples` imports. Native tile/host source packaging and offline linking are covered for all three registered operators; full-model/imported builds and CPU acceptance's optional legacy build/formal/matrix adapters remain migration dependencies; this is not full-family consolidation.

## Installed-package scope

`etx.testing.installed_package_probe(output)` builds a fresh wheel from only `etx/` and `pyproject.toml`, installs with `--no-deps --no-index`, then runs an isolated Python outside the checkout. The emitted workflow source is checked by the same external-test AST policy before execution; only the fixed bootstrap performs installation-path and module-origin inspection. Build dependencies must already be installed; no dependency fetching is requested. Each subprocess has a 90-second deadline and a retained full log. The probe clears `ETX_*`, `PYTHON*` and `PIP_*` variables, restores offline pip settings, and checks that every loaded ETX module comes from the installed target.

Desktop D14 passed the installed external workflow: 12 handoff plan/simulation/emission/source-reload combinations (1/2/8 ranks × B1/2/4/8), the retained historical host fixture, the 13-entry disabled sealed catalog, and host-process execution followed by record relocation/revalidation. Original importer-header package assertions remain. The installed probe additionally covers 12 registered-operator combinations (split-K/default MoE/small MoE × sizes 1/2/4/8), a separate two-rank GEMM source-artifact save/load case with packaged host/tile closure, paired split-K/MoE protocol run/report with exact replay, and both five-row reduced studies at size 2. D17 passed the protocol/study expansion; `API-OPERATORS-20261023.log` records **87 passed** for the later registered-operator/installed/policy scope. All remain CPU simulation/source records. This is CPU/source/resource coverage only: no full model provider, native sealed preparation/run, linked executable, device numerics or performance is qualified. The prior wheel import failed because the catalog resource was under unshipped `examples/`; that failure remains recorded.

## Test policy and remaining migration

- New caller E2E tests live under `tests/e2e` and import only `etx.external`, `etx.testing`, pytest, standard libraries and checked `tests.e2e` helpers.
- `check_external_sources()` scans every Python file under that tree, including unused helpers, and rejects facade submodule imports, non-exported facade names, relative imports, common dynamic-loading/direct-process calls and dynamic fixture requests.
- `external_test_policy(root)` registers through the root conftest; it rejects nonpublic inherited fixtures (including dependency chains and autouse fixtures) at collection and checks dynamically created fixtures at setup.
- Built-in pytest fixtures and fixtures implemented under `etx.testing` or the checked E2E tree are allowed; unrelated third-party fixtures require an explicit supported adapter.
- The scanner follows common import/assignment aliases and rejects private attributes, reflection entry points, monkeypatch/mock patch calls, import-state access, unchecked helper imports and direct process entry points. Environment setup remains supported. It is an architectural check, not a security sandbox or proof against arbitrary Python obfuscation.
- `check_migrated_entry_points(root)` prevents `bench_sim`, `cpu_check`, `bench/sim_experiments.py` and sealed `stage.py` from importing private lifecycles, public-library internals or test-only executors. This is an import guard, not a sandbox. The generic compiler's private loader shim and scheduler's legacy symbol reexports are still migration debt, not silently whitelisted as complete.
- Root `tests/conftest.py` keeps `pinned_src` for internal tests; migrated E2E tests cannot request it.
- `tests/api_inventory.json` records every root Python/shell test with a reviewed file-level classification and migration owners. Its historical baseline contains 78 legacy workflow files (9 Python E2E, 68 mixed, 1 shell E2E); 69 remain workflow exceptions after protocol, CPU-acceptance, scheduler, sealed-session, installed-package and isolated DeepSeek compiler lifecycle cases moved to external-only E2E and their conformance stayed behind `etx.testing`, and all 25 plan-export plus 13 GLM pair CLI cases moved intact out of their legacy root files. Those two root files and active inventory rows are retired; historical baseline entries remain. Moving CLI tests does not migrate the model CLI implementation. New unclassified tests and workflow exceptions outside this reviewed baseline fail. Changes to the baseline itself require review; a mutable JSON file cannot authenticate its own history.
- API-02 remains partial until the legacy tests migrate; `API-TEST-INVENTORY-REVIEW.md` preserves the static review rationale, not execution evidence.
- Internal unit tests are not renamed E2E; their negative plan/ABI/ISA tests migrate behind the conformance surface without deleting coverage.

| Legacy migration group | Owner TODO |
|---|---|
| GLM module/model/phased whole-model tests | API-15, API-24 |
| Qwen benchmark/import/selection integration | API-11, API-17, API-24 |
| DeepSeek recorded session E2E | API-18, API-24 |
| Sealed experiment session integrations | API-23, API-24 |
| Remaining legacy E2E classification and inherited fixtures | API-02, API-24 |

## Executable acceptance for the first slice

```sh
/data/etx-venv/bin/python -m pytest -q tests/e2e tests/test_api_boundary.py tests/test_api_fixture_policy.py tests/test_api_inventory.py tests/test_api_conformance.py tests/test_session_proc.py
```

- Execute from `/data/etx-run` on `ssh desktop`; results belong in the scoreboard, not in this design contract.
- Use the real `/data/etx-venv` environment: the copied tree's `.venv/bin/python` symlink resolves with `/usr` as its prefix and does not expose the venv's pytest installation.
- The default CPU suite skips the opt-in Docker compiler test. To run the real eight-rank object check and rank-7 failure case, use a new evidence directory (existing evidence is never overwritten):

```sh
ETX_TEST_ROCM_DOCKER=docker.m.daocloud.io/rocm/dev-ubuntu-22.04:6.4.1 \
ETX_API_BUILD_EVIDENCE=/data/etx-api-evidence/<new-run> \
/data/etx-venv/bin/python -m pytest -q -s tests/e2e/test_api_device_check.py
```

- Those checks compile a tiny no-op protocol fixture, not a full model; no GPU is accessed. Historical compiler runs are retained rather than regenerated when only documentation or unrelated policy changes.
