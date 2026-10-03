# CPU protocol benchmark foundation

This is the first executable piece of the evidence-driven roadmap: reproducible
**protocol simulation**, not a Qwen/Mixtral inference benchmark. It runs the actual
ETX compiler and existing cost-table simulator on the split-K and MoE example
graphs. No GPU, weights, tensor computation, production operators or inference
engine are executed. A protocol pass does not establish numerical correctness,
output-buffer coverage, cache visibility correctness or measured GPU performance.

## Run and recompute

From the repository root, with Python 3.11 or newer:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m etx.tools.bench_sim run \
  --config bench/configs/protocol-gfx942-v1.json \
  --out build/bench/protocol-v1
.venv/bin/python -m etx.tools.bench_sim report build/bench/protocol-v1
.venv/bin/python -m pytest -q tests/test_bench_sim.py tests/test_sim.py
```

Use a new output directory for each run; existing directories are refused.
Both commands return nonzero for invalid experiments. Compilation/simulation
exceptions are retained as failed samples. An interrupted run without its final
manifest is incomplete and cannot produce a valid report. `report` needs only the
saved records, not the example modules, weights or a GPU.

## Frozen protocol v1

`bench/configs/protocol-gfx942-v1.json` defines four small cases, three rounds and
an explicit starting seed. `size` means split-K row groups (`n`) or MoE forward
batch (`B`), **not service concurrency**. The MoE fixture uses its default shape
and routing seed 12345; `ETX_MOE_SMALL=1` is rejected. `duration_cv` controls modeled
task-duration variation. It does not inject hardware measurement noise or vary
routing between rounds. To study other conditions, save a new config.

Every case compares static, dynamic and hybrid scheduling with identical task
bodies, sizes and per-round duration seeds. Event elimination and prefetch are
disabled in all arms to isolate scheduling. Arm order rotates each round. This is
not a fused/unfused or ETX/vLLM comparison: those require future device adapters.

Timing is the simulator's makespan in microseconds, from first dispatch until
all workers complete. Python wall time and compile time are not latency samples.
Warmup is zero because each simulation starts fresh without device cache state.
Reports show medians, min/max and median **paired** percentage reduction relative
to static. They do not report confidence intervals, infer hardware noise, or claim
GPU improvements from seed variation. Positive percentages mean lower modeled
latency; negative percentages mean a regression.

## Artifacts and acceptance

| File | Contents |
|---|---|
| `manifest.json` | Schema/kind, exact config, timing boundary, explicit simulation labels, Git commit/dirty flag, source hashes, Python/platform and raw-artifact hashes |
| `plans.json` | Per-arm task contracts, dependencies, tensor shapes, runtime routing, bindings, options and compiler decision logs |
| `samples.jsonl` | Every arm/round/seed, complete simulator timeline and budgets, blocked states, validation failures or exceptions |
| `summary.json` | Recomputed validity, failures and machine-readable comparisons |
| `report.md` | Human-readable report generated solely from saved records |

The source hashes include tracked and untracked Python/YAML/JSON/TOML files so a
dirty run is distinguishable from its parent commit. Preserve the corresponding
source checkout for replay; hashes identify content but do not archive it. The
artifact hashes detect accidental changes, not adversarial modification.

Acceptance checks require no deadlock, finite positive makespan, finite budgets,
every planned task exactly once, valid segment times/worker IDs, no worker overlap,
producer completion before consumer start, and budget/trace consistency. This is a
protocol-level check, not a model-output gate. Saved checks are repeated during
report generation, including pair completeness, uniqueness and seed consistency.
Any failed or missing sample withholds **all** comparisons, rather than silently
dropping failed runs. A failed report rebuild invalidates any previous summary.

Tests inject missing/duplicate tasks, nonfinite timings, dependency violations,
deadlock flags, budget conflicts, malformed pairs and simulator exceptions. A
real dynamic-simulator defect found by these checks is covered by a regression:
visiting all producers in start-time order does not mean they have finished;
a consumer must wait for the latest producer's event readiness.

## Next adapters

Add real tensor references and full output-write/finite-value checks independently
of this timing simulator. Then add recorded native-Graph, ETX same-operator
fused/unfused and strong-engine adapters, with frozen token inputs, complete output
records, matched decode intervals, device/host timing labels and paired samples.
Qwen and Mixtral generation, actual GPU noise/precision gates and service latency
remain unimplemented. Do not mark roadmap stage 1 complete based on this harness.

`sim_experiments.py` remains the earlier exploratory variance/routing-skew study;
its printed estimates are historical model outputs and are not imported as new
measurements by this runner. The dynamic-readiness fix can change its estimates.
