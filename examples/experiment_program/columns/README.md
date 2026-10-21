# Native columns and request readiness (G02/G03/G05/G13)

All switches are off by default. The package implements local preparation and a future explicitly enabled resident-input GPU replay. Current local results establish source generation, gfx942 compilation/linking and finite queue/resource checks. GPU correctness, actual placement, readiness fidelity, traffic and performance remain **UNMEASURED** until the separately authorized session executes them.

## Implemented mechanisms

G02 shares each native weight stream between two/four independently ordered accumulators. It preserves Fleet K1 Q3072/KVA576 partitioning, original wave reduction, bf16 conversions, gate/up interleaving and scaled down arithmetic. Shared-expert roles retain their actual XCD6/XCD7 placement. G05 groups actual consumed expert IDs in first-encounter order, gathers original request-slot inputs, shares the stream, and scatters to original slots. Singleton groups invoke the exact original helper. Group discovery, metadata clearing/packing and incremental metadata upload occur only in the G05 candidate, inside its host-wall bracket.

G02/G05 execute actual ETX persistent fixed queues, with the accepted **75 workers/domain for both B2 and B4**, and native request service slots `38*(request%2)+worker`. Every exogenous ready producer occupies its own domain's unused worker37, permitting nonmonotonic captured arrivals. Source-bound producer placement, all task counts, raw traces and writing-worker completion are checked in separate diagnostic probes. Decisive pairs disable instrumentation. A private generated dispatcher aliases dead bootstrap/descriptor scratch with staging only across proved lifetime barriers; arithmetic staging remains B2=16384/B4=32768 bytes. ARGPF/BAR2/relay modes are refused by this alias transform.

G03 executes the historical b5eaa67 NB1/NB2 dense body in a common ETX executable with matched LDS and the retained qkv11 row/domain/worker assignment. Its ready producers occupy independent unused workers. Simultaneous versus captured-ready runs support a paired difference-in-differences diagnosis. G13 executes the same ff9ef92 unshared up/down bodies in both arms: 384 up tasks, 512 down tasks and two independent ready producers. Runtime combined versus per-request completion controls and two legal static queue orders change visibility/order; the finite proof rejects the producer-blocking combined/request-first order. Neither diagnostic reconstructs the missing identity or causal timing of old GPU runs.

The resident driver allocates inputs once, poisons every writable endpoint outside each bracket, checks original/candidate/sealed-golden bytes before timing and after every pair, and checks G13 up intermediates after each arm. It retains five seeded interleaved AB/BA pairs and every raw sample, per-request ready/done endpoints, stdout/stderr and exit. Admission requires both positions and both readiness regimes. Early-request latency losses cannot disappear behind a final-span gain.

## Offline build and verification

Use the shared Python and an explicit task scratch directory. No command below executes a GPU or downloads anything. Builds use the already installed `etx-rocm22` image, ROCm7.2.4 LLVM22, sealed device libraries and read-only link dependencies. Keep at least 2GiB free before every build.

```sh
export PYTHONPATH=$PWD
export ETX_PYTHON=/Users/ckl/code/etx/.venv/bin/python
export ETX_COLUMN_FLEET_SRC=/Users/ckl/code/etx-experiment-columns-build/cooked-fleet
export ETX_COLUMNS_BUILD=/Users/ckl/code/etx-experiment-columns-build
$ETX_PYTHON -m pytest -q tests/test_experiment_columns.py --basetemp=/Users/ckl/.hermes/cache/scratch/etx-experiment-preparation/columns/review-test-scratch
$ETX_PYTHON -m examples.experiment_program.columns.build --out "$ETX_COLUMNS_BUILD"
$ETX_PYTHON -m examples.experiment_program.columns.micro_build --out "$ETX_COLUMNS_BUILD/native-ready-final"
$ETX_PYTHON -m etx compile examples/experiment_program/columns/g03_graph.py --arch gfx942 --out "$ETX_COLUMNS_BUILD/g03-ready-final" --inline-tiles --relay off --no-event-elim
bash examples/experiment_program/columns/build_g03.sh "$PWD" "$ETX_COLUMNS_BUILD"
bash examples/experiment_program/columns/build_extended.sh "$PWD" /Users/ckl/code/etx-cache/fleet-3fb67a6 "$ETX_COLUMNS_BUILD" native-capture-04
bash examples/experiment_program/columns/build_capture.sh "$PWD" "$ETX_COLUMNS_BUILD"
$ETX_PYTHON -m examples.experiment_program.columns.seal_build --build "$ETX_COLUMNS_BUILD"
```

Destinations for generated plans, capture patches and identity seals must be fresh. Retained prior build attempts are evidence and are not overwritten to make a pass. `build_capture.sh` consumes the two fresh `historical-capture-ready-{b5eaa67,ff9ef92}` trees emitted by `patch_historical_capture --source .../etx/ops/gemv.py --host .../examples/vllm_llm/host.hip --revision REV --nb 1 --out DEST`. That patcher emits all six native entrypoint probes itself. `seal_build` includes the final sources, actual Linux host ELFs, device modules, descriptors, metadata, queue proofs and resource decisions. A linked AMD descriptor and compiler occupancy determine residency; AGPR is already included in unified `.vgpr_count` and is counted once.

`columns-run`, eight `native-ready-final/*/columns-run`, `g03-ready-run` and `g13-ready-run` are real Linux executables using the dynamic HIP module API. They load `libamdhip64.so` only at future explicit execution. Original whole-model hosts are also compiled to real host objects. Their ordinary full-model host link currently returns the genuine prerequisite code77 because the retained local toolchain lacks the HIP runtime library; no executable/model certificate is fabricated from those objects.

## Native input capture and sealing

`patch_capture` binds source-owned Fleet QKV post-fold/RMSNorm and expert post-stage sites before the actual weight loads, then records each original task's outputs after all writing waves drain. It writes only private kernel/shim/host/model copies. Actual expert IDs, scaled arithmetic weights and unscaled G04 routing weights, native worker/XCD/HWID, epoch, input bytes, output bits, source pointer offsets and timestamps are retained. The capture host requires B2/B4, context1024, 32 greedy tokens, repeat2, per-request warmup/hidden/greedy gates and the full native task population. Fleet cached decode positions are **1024 and1040**, selected by token0 and16; position0 is not executed by this host.

`patch_historical_capture` binds all six pinned historical DSL bodies at their true staged-input boundary and captures their exact original row outputs, including folded top-k LDS weights and native down-weight expressions. The generated historical host sets argument/position-bound capture configurations and disables capture before checker replays. It requires full B2/32-token greedy execution. Historical positions are0 and1040. These are explicitly **NEW_CAPTURE_NOT_HISTORICAL_RUN_RECONSTRUCTION**. Target probes compile the actual retained arithmetic bodies and the original patched host object; they do not certify a rebuilt complete historical model or recover discarded old inputs/stamps.

The future owning model build must compile the generated private `model.py` and matching capture-bound host/shim with the registered frozen build recipe. Ordinary runtime linking must first satisfy the real HIP-library prerequisite. `seal_capture --recipe RECIPE --out producer.json` checks complete original raw logs, all native row exports and source/build/input/checkpoint seals. `capture_pack --receipt producer.json --out NEW_DIRECTORY` copies exact native operand/golden/checkpoint bytes; it performs no synthetic operand generation or arithmetic reconstruction. G05 packing requires G04's complete capture receipt and exact consumed IDs/weights with matched input/prefix/golden identities.

Current retained goldens lack these post-prologue operands and native task outputs. Therefore current dry runs report `BLOCKED_NATIVE_CAPTURE`. This is a missing future GPU-produced input, not an absent capture hook or absent native queue implementation.

## Registered session path

One program configuration names `schema="etx.columns.program.v1"`, `record`, `build`, `capture_root` and a `dependencies` list. Optional controls are `warmup`, `seed`, `timeout` and G02 `model_batch` (default4). No arbitrary shell commands or case list can be supplied. The capture tree is `<capture_root>/<case>/p<position>/capture.json`.

G02 runs `qkv-B2`, then conditional `qkv-B4`, then `shared-B{2,4}-x{6,7}-{gate_up,down}` gated on its own batch's QKV result. G05 runs `routed-B4-gate_up` and `routed-B4-down` and emits a rigorously validated matched role bundle. G03 uses `G03-B2`; G13 uses the same `G13-B2` captures for request-order and both-up-first controls. All run serially; native failures stop the program and retain evidence. Ordinary misses only gate their own dependents.

```sh
python -m examples.experiment_program.columns --record G02 --program /staging/G02-program.json --dry-run
# Only inside a separately authorized GPU session:
python -m examples.experiment_program.columns --record G02 --program /staging/G02-program.json --out /session/G02 --execute
```

G03/G13 require the actual raw-qualified G02 QKV receipt; G05 requires G04 `complete=true` capture evidence, not an optimization winner. Missing/refused captures block program preflight. Aggregate admission recursively rereads every raw child, captured source, task probe, original golden, full sample population and actual exit. G05 needs both matched roles, real native mix and singleton coverage. G02 shared-expert winners do not authorize changing other roles. `integration.prepare`/`prepare_grouped` create actual private frozen Fleet source/graph interventions for G12; `local_validation=True` permits compiler-only preparation and explicitly sets `timing_eligible=false`. Full-model resource and whole-model correctness/timing gates still apply. Actual whole-model native capacity failures close that proposed implementation without a resource waiver.

Staging uses explicit `ETX_PROGRAM_RELOCATIONS=/sealed/mapping.json` (`schema="etx.path-map.v1"`, `paths` maps original prefixes to staged prefixes). The mapping's own SHA enters the input seal; every original file SHA must remain identical. No implicit path guessing, rewriting old seals or relabeling old binaries is permitted.
