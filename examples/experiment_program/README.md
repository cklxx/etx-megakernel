# Thirteen-record local preparation

This is the implementation of the accepted `specs.json` program. The specification is retained verbatim as research authority; its original `implementation_status` and `runnable_now` fields describe the research snapshot, not current preparation. The checked-in [preparation manifest](preparation_manifest.json) maps all13 records to actual code, controls, build identities and unresolved gates. The external final `readiness.json` records the exact combined Git revision, acceptance and staging results or explicit blockers. No GPU outcome is inferred from compilation.

The discoverable entry point is:

```sh
PYTHONPATH="$PWD" /Users/ckl/code/etx/.venv/bin/python -m examples.experiment_program list
```

The entry maps every G01–G13 record to executable code and its dependencies. All device paths default off. `session.py` supports per-record selection, structured preflight, dry run, bounded execution, sealed resume and verified local archive export. `examples/sessions/experiment_program.py` wraps it as one pinned GPU lane for the existing `etx.tools.gpu_jobs` runner. Loading or dry-running the job file does not execute a device, install software or contact a service.

## Implementations and local commands

- **G01:** `router/` keeps all256 router tasks, elects after128 useful owners, preserves the existing router/KEY32 arithmetic, and checks all eight physical readers. `python -m examples.experiment_program.router build --build NEW_BUILD` compiles the uninstrumented and separate diagnostic modules plus a linked replay host. `preflight --build BUILD --capture POS0 --capture POS1040` seals96 original retained cases. `--diagnostic` only exports native per-task/wave landmarks; it cannot produce an optimization admission. `etx/ops/last_router.py` provides the default-off whole-model intervention; `router.model_contract` validates all48 reset tasks, router coverage, global publication dependencies and actual static-queue liveness.
- **G02/G03/G05/G13:** `columns/` contains shared-column arithmetic, historical source-bound replay, native producer capture hooks, real static ETX expert graphs, grouping and winner-only model integration. `python -m examples.experiment_program.columns --record G02 --dry-run` reports exact missing native capture requirements. Use the package README for capture/build commands; select the other three IDs through the same entry. Missing post-prologue native operands are not replaced by model goldens or CPU-generated tensors.
- **G04:** `routing/` and guarded hooks in `examples/dsv2lite/{fleet_shim,host}.hip` capture the actual routed selector consumers. `python -m examples.experiment_program.routing preflight --config CAPTURE_CONFIG --out NEW_OUTPUT` checks all source/plan/compiler/flags/resources/model/input/prefix/golden identities. `analyze` decodes the complete B2/B4 ×32 ×26 ×8 population, preserving actual slot/XCD IDs and mix bits. This is capture only; it never returns an optimization gain.
- **G06/G07/G08:** `issue_order/` changes only registered dependency issue order, preserves exact imported ABI/math, maps native instructions back to original sources, and retains all boundary inputs. `python -m examples.experiment_program.issue_order --record G06 preflight --build BUILD --capture CAPTURE --layers 0,24,47` checks the sealed build and all four adjacent contexts. Separate sparse probes require actual matched allocation and perturbation qualification before attribution.
- **G09/G10/G11:** `row_ops/` contains register-retained activation segments, exact original-row scratch removal, native producer checking/capture, complete output poisoning and safe fold probes. `python -m examples.experiment_program.row_ops records` explains the runnable paths. G10 down remains conditional on the exact admitted up receipt. G11 is exposure only and requires qualified controls plus G12's actual matching graph trace. The integrated G11 path is `publication_run`: the full current256-down/256-QKV subgraph, original operand capture, original final-arrival issue/forced-retirement bracket and typed consumer read/fold/norm/output reader. GPU qualification remains unmeasured.
- **G12:** `validation.py` and its preparation helper require a raw identity-bound optimization winner, prepare the specific full-model source/build, and enforce fresh correctness, host-wall model pairs, same-kernel graph and strongest tested legal system gates. DeepSeek same-kernel graph validation is explicitly `UNAVAILABLE_LDS_LIFETIME`; no substitute certificate is issued. Follow [G12.md](G12.md) for the concrete prepare/build/activate and execution interfaces.

Every optimization's five seeded paired populations must be complete, correct and uninstrumented. Admission requires every saving positive, median at least2% of the matched control and greater than twice the range. Qwen layer admission additionally requires625ns; full-model KEEP additionally requires30,000ns. A micro pass only permits G12. Native traffic counters and sparse timing never become claims when incomplete or unqualified.

## Staging and session configuration

`python -m examples.experiment_program.stage --inventory INVENTORY.json --out SESSION.json --session-output FUTURE_OUTPUT` emits concrete argument arrays for all13 from explicit staged paths. It copies no model data; absent captures/configuration are listed as missing prerequisites. The inventory has a `staging_root` and optional per-record `build`, `captures`, `capture`, `pack`, `config`, `dependencies`, `variant` and `inputs` fields. Column records may instead supply a registered `program` configuration to execute the prepared batch/role sequence. G12 receives the actual micro-session root for registered winner dispatch, without rewriting original sealed configurations.

A session JSON has the exact published `base`, a positive `budget_seconds` no greater than14,400, optional explicit decision flags, and `records` keyed by selected G IDs. Each record contains `preflight` and `command` as argument arrays for its prepared module, nonempty `input_files` mapping absolute staged paths to SHA256, `timeout_seconds` no greater than900 (G12 aggregate at most13,500; each native G12 call still at most900), and a relative `decision_file`. An optimization that reports a win also needs `admission` with its exact build/input/output arguments for raw evidence revalidation. Its `build/identity.json` must be in the session input seal. Commands must name the current staged project Python. The preflight action cannot contain device authorization; module-specific safe actions are checked.

Run locally safe paths first:

```sh
python -m examples.experiment_program dry-run --config SESSION.json --out NEW_SESSION
python -m examples.experiment_program preflight --config SESSION.json --out NEW_PREFLIGHT
```

For a future separately authorized machine, the job file accepts `ETX_EXPERIMENT_CONFIG` and `ETX_EXPERIMENT_PYTHON`. It consumes pre-staged sealed files, preserves full logs, pins all comparisons to one GPU lane and enforces an aggregate deadline. No provisioning or remote transfer is implemented. `run --execute` is an explicit later operator action; preparing this repository never invokes it.

A missing prerequisite skips only its record. A correctness/liveness failure, nonzero native exit or exhausted deadline stops the attempt, preserves its logs and refuses ambiguous resume. Completed dependencies are revalidated from raw evidence. The runner schedules G12 after optimization records and G11 after G12, including explicitly reordered selections. Export writes an owned local tar archive, verifies every byte and replaces it atomically; shared symlinks and unowned archives are refused.

## Local compiler and limits

Builds use the preexisting offline `etx-rocm22` image with `--pull=never --network=none`, deployment clang22/ROCm7.2.4, gfx942, actual device libraries and a read-only local linker-dependency tree. They retain complete commands, exit receipts, native ISA/resource metadata, input/source hashes and real linked replay hosts. Micro hosts use `dlopen` to bind the actual HIP runtime only on a separately authorized GPU machine; `--help` performs no device discovery.

`model_build.py` prepares complete current model closures without copying a baseline build tree. Ordinary full-model linking requires an existing real HIP runtime library and original reference kernels; a missing library returns a documented blocker, never a fake link or certificate. Final readiness and any exceptions require Amber's independent review of the exact combined commit before GPU admission.

## Explicit path relocation

For a different staging filesystem, set `ETX_PROGRAM_RELOCATIONS` to an explicitly sealed JSON file with schema `etx.path-map.v1` and a `paths` object mapping absolute origin prefixes to absolute staged prefixes. Resolution uses the longest prefix; chained/cyclic mappings are refused. The original build/capture metadata remains byte-identical and every expected SHA256 is verified against the staged bytes. The mapping file itself belongs in each session input seal. No mapping is inferred from a hostname, stale IP, credential or remote checkout.

Review repair of `94a29c0` is frozen by the commit containing the current
`preparation_manifest.json`. The experiment authority/specifications are unchanged.
The exact combined revision and raw final receipts live in
`/Users/ckl/.hermes/cache/scratch/etx-experiment-preparation/review-repair-receipt.md`.

The G01 entry stages its decisive, matched sparse/empty and quiescent rollover
program. G09/G10 use finite row recipes with both positions and raw up evidence
before down. G11 now uses `etx.g11.program.v1` recipes and the integrated
`publication_run` entry point. Its actual control/probe/empty gfx942 modules have
matched native descriptors; the reserve operation preserves every instruction
byte. Compiler SGPR spill counts remain separate evidence. The genuine runtime
and original-checker capture host are linked. Earlier rejected builds and all
prior12 closures remain intact. No GPU qualification is transferred.

G11 preparation and future operator path:

1. `publication_build --native ORIGINAL_CHECKER --out NEW_RAW --runtime GENUINE_LIB`
   builds only this subgraph plus its three layer hosts and original capture hook.
   Apply `match_diagnostics.match(NEW_RAW, NEW_MATCHED, [['control','probe','empty']])`
   to reserve identical descriptors without changing instructions. Final host and
   source sealing uses `publication_build --matched NEW_MATCHED --out NEW_FINAL
   --runtime GENUINE_LIB`. All are Python modules under `examples.experiment_program`.
2. `publication_run capture --identity IDENTITY --model MODEL --layer L
   --position P --output NEW_CAPTURE` prints the original-checker job. Only a
   separately authorized future `--execute` runs it. Run layers1/24/47 at0/1040;
   it exports all8 original down blocks and the QKV block, seals the complete97
   checking transcript and original module/host/input provenance, then assembles
   live aliases and exact slot/residual/QKV references. Norm LDS is not presented
   as an independently exported intermediate. No CPU/model-generated reference
   substitutes for this capture.
3. Seal all six resulting packs and the publication identity in an
   `etx.g11.program.v1` recipe (`publication_preparation`, `packs` keyed `L1-p0`
   etc., `inputs` path-to-SHA256, optional `deadline_seconds`). Stage it as G11's
   `program`, `build` and `publication_preparation`. This selects
   `publication_run program`, never isolated row replay.
4. A complete active same-session G12 trace at both positions and a live fold
   decision are prerequisites for device execution. The host restores pristine
   inputs, poisons write-only references, checks all exact original global
   outputs/intermediates, and traverses4096 consumer-task/wave selectors in three
   rotated modes. Separate warmup/coverage launches observe all three controls.
   Timed control has diagnostic pointers disabled. Publication bounds and consumer
   phases share one invocation's hardware clock; G12 clocks are never subtracted.
   All512 tasks/all16 waves, actual CU/XCD placement, final consumer, phase order,
   exact references and measured absolute perturbation<=0.5us are reread from raw
   exports before qualification. Failed native gates close the diagnostic.

Complete raw export storage is reserved for all six cases before any device
child, with a protected2GiB floor. G11 reports exposure, never a removable-work
price or speedup. Preparation is awaiting independent acceptance; native
correctness/placement/perturbation are unmeasured. Original missing-runtime and
failed receipts remain available.

G13 observes three reachable completion/order states in two paired groups. The
combined/request-order state is unreachable and is never claimed as a fourth
control. Queue edges are reconstructed from complete native task traces and
compared with the actual finite runtime queue selection. Body-duration changes
remain a contention qualification. Historical attribution remains UNKNOWN.

All12 original software findings are independently CLOSED at e8d10ec. The remaining
canonical G11 preparation is tracked separately; exposure is never a removable-work
price or speedup, and its execution remains opportunistic under an active G12
matching trace and fold decision.
