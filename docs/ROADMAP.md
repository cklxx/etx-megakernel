# ETX: project scope and evolution roadmap

**Success.** ETX succeeds when, in one session on one MI300X, Qwen3-30B-A3B bf16 decode at batch 1 (the committed 1024-token golden prompt, 32 greedy tokens) runs **more than 2% faster in host wall time per decode step** fused than with the same kernels replayed one by one through a HIP graph, as judged by `examples/vllm_llm/pairs.py`; vLLM's best configuration, measured in the same session on the same basis, is reported beside it.
*Validity:* five interleaved pairs on one GPU, with the machine recorded; each pair is the mean of steps 1024-1054 (`docs/SPEC-nv-reference-2026-10-08.md:21-26`); `host --check` OK on every imported launch; 32/32 tokens; no debug environment variables (`examples/vllm_llm/pairs.py:20-23`). *Tolerance:* every pair's gain is positive and the median exceeds twice the gain spread (`pairs.py:34`) - more than 0.093 ms at 4.548 ms (`SPEC:292`). *Measurable today:* yes, with the `nvref` session and its fusion arm; the expectation is "not met", at 1.4-2.1% [E]. *Needs an NVIDIA machine:* the same sentence on H100 or B200, which first needs a tested CUDA emission (`README:203`), an importer that reads more than AMDGPU IR (`PLAN:273`) and the owner's choice of GPU (`PLAN:465`); BabelStream on both GPUs plus the same-model vLLM or SGLang reference (`PLAN:458-459`) would set the cross-GPU utilisation target, which has no number today.

SPEC means `docs/SPEC-nv-reference-2026-10-08.md`; PLAN means `docs/PLAN-generic-decoder-2026-10-08.md`; README means `README.md`; pairs.py means `examples/vllm_llm/pairs.py`.

Proposed direction, 2026-10-03. Repository evidence reviewed at
[`5ef3882`](https://github.com/cklxx/etx-megakernel/commit/5ef3882c7030e6095f3fa308969790fabafd33fa)
(committed 2026-10-02). This is a documentation update, not an implementation or
GPU revalidation. Historical measurements below retain their original scope;
future deliverables and acceptance gates are proposals.

This roadmap supersedes the [2026-09-30 roadmap](ROADMAP-2026-09-30.md), preserved
unchanged as a historical record. Its predictions and queued experiments are not
current commitments. Existing experiment reports remain the source of measured evidence.

## 1. Scope and the claim to prove

ETX should be a GPU compilation and runtime system that reuses high-performance
operators and automatically selects low-latency execution plans for a model,
request workload and target hardware. Start with a low-latency MI300X decode
backend, demonstrate performance transfer across models and workloads, then
integrate with an existing inference service. Extend hardware, model structures
and precision separately after those foundations hold.

The wider organization can address inference efficiency end to end. This
repository should first own compilation, fusion, scheduling and runtime.
Standalone operator optimization, quantization and speculative decoding can
proceed alongside it, with their gains measured separately.

The core question, consistent with [CLAUDE.md](../CLAUDE.md), is whether reusing
production operators with better task partitioning, data visibility and scheduling
reduces complete inference time and repeated per-model manual optimization.
There are three distinct tests:

1. **Execution:** savings with the same operators exceed scheduling,
   synchronization and resource constraints introduced by ETX.
2. **Transfer:** the compiler chooses useful plans for unseen shapes, context
   lengths and request counts without a new manual tuning campaign each time.
3. **Service:** users receive results sooner after model loading, prefill, KV
   management, sampling and streaming output are accounted for.

The eventual execution space should include ordinary launches, Graph replay,
local megakernels and full megakernels. Legality and measurement should determine
whether and where to fuse.

The research space is competitive: [Event Tensor](https://arxiv.org/abs/2604.13327)
addresses dynamic task dependencies and static/dynamic scheduling;
[Ada-MK](https://arxiv.org/abs/2605.11581) studies resource-constrained search and
decode-megakernel integration with an existing engine. These are related-work
references, not ETX measurements. ETX must establish the additional value of
chiplet visibility handling, operator reuse and execution-policy selection;
generating one large kernel alone is insufficient.

## 2. Current evidence and its limits

ETX is a compiler/runtime prototype with real-model validation. Unified
acceptance, automatic performance transfer and full serving integration remain
unproven. See the [architecture](ARCHITECTURE.md), [README](../README.md),
[data index](DATA.md) and [model-transfer report](GENERALITY-2026-10-02.md).

| Area | Existing foundation | Still to establish |
|---|---|---|
| Compilation/runtime | Task graphs, events, machine models, seven passes, simulator and device runtime | Automatic selection of a near-best complete execution plan |
| Operator reuse | Compiled-kernel import and handwritten DSL paths | Unified validation of arguments, grids, numerical semantics and compiled resources |
| Model transfer | Full generation checks recorded for Qwen and Mixtral | Repeatable performance transfer beyond correctness |
| Hardware/service | MI300X model runs and microbenchmarks; two-GPU operator experiments | NVIDIA validation, cross-architecture performance and full distributed inference serving |

The README's 2026-10-02 Qwen3-30B-A3B bf16, batch-1, single-MI300X report records
4.823 ms/token for ETX versus 4.863 for vLLM default, approximately a 0.8% advantage.
Treat this as a reported small margin pending a common timing protocol and noise
validation. Earlier results used different configurations and timing bases; the
[vLLM ladder](VLLM-LADDER-2026-10-01.md) documents how mixing device and host
measurements changed a conclusion. Do not combine rows from different runs to
infer a fusion margin.

The Mixtral report records 32/32 generation matches and, in its later DSL-path
update, 10.365 ms/token versus vLLM's 8.686. This is evidence of correctness
transfer with a remaining performance gap, subject to the same common-protocol
requirement. Imported and DSL paths must stay separately identified.

The next task is to turn manually acquired single-model knowledge into rules the
compiler can check, select and reuse. Machine-model entries for other GPUs and
two-GPU operator results do not establish validated hardware backends or serving.

## 3. Architecture: retain the layers, complete their contracts

| Layer | Responsibility to develop | Existing location |
|---|---|---|
| Model/operator description | Shapes, layouts, precision, compile-time constants and launch conditions; derive arguments once | `examples/llm/`, `examples/vllm_llm/`, `etx/importer/` |
| Dependencies/state | Task dependencies, reads/writes, aliases and lifetimes; later cross-token state | `etx/ir/` |
| Hardware model | Separate legal capabilities from calibrated performance costs | `etx/machine/`, `bench/calib/` |
| Plan selection | Joint operator, task-granularity, placement, fusion-boundary and schedule selection | `etx/passes/`, `etx/sim/` |
| Code generation/runtime | Execute validated plans, manage state and explicitly fall back for unsupported configurations | `etx/codegen/`, `etx/runtime/` |
| Acceptance/experiments | Reference outputs, raw samples, conditions, failures and generated reports | `tests/`, `bench/`, experiment scripts, `docs/DATA.md` |

Keep the existing distinction between capabilities and costs, and the scheduler's
record of selection reasons. Calibrated costs must never replace correctness rules.
Three contracts matter first:

- **Complete operator specifications.** An entry point is insufficient: record
  supported shapes, template constants, launch grids, layouts and read/write
  effects. Mixtral's grid/compiled-constant mismatch left large output regions
  unwritten even when original-versus-imported checks passed with the same wrong
  parameters; see rounds 40–42 of the model-transfer report.
- **Actual compiled resources.** [P1](../etx/passes/p1_tiling.py) takes tile shapes
  from the frontend, checks resources and derives workgroup counts. It is not
  automatic tiling search. Select candidates using actual registers, shared
  memory, spills and complete execution time.
- **Explicit data relationships.** [P3](../etx/passes/p3_event_elim.py) rejects
  recomputation when writes are absent or in-place behavior cannot be ruled out.
  Opaque argument blocks hide model dataflow. Add reads, writes and aliases before
  expanding transformations; do not relax the safety checks.

## 4. Six stages, each with an independent acceptance gate

These are proposed stages, not existing version numbers. Set numerical acceptance
thresholds before experiments, including measurement precision, selection regret,
regression limits and service latency targets. This roadmap does not claim those
thresholds have already been agreed or met.

### Stage 1 — Establish a trustworthy performance baseline

Freeze a single-request Qwen3-30B-A3B bf16 benchmark on MI300X: input token IDs,
generation interval, statistics, completion boundary and warmup. Save outputs
and individual timings, not just medians. Bind records to code, model, toolchain,
engine configuration and machine conditions; retain failures alongside valid runs.

Use three correctness layers: import fidelity; complete model semantics; and
full output coverage with finite numerical values. Add injected NaNs, missing
writes, conflicting arguments and synchronization faults. Keep the existing
import-fidelity and 32/32 HF-greedy gates; coverage and fault tests supplement them.

Retain three distinct controls:

| Control | What it measures |
|---|---|
| Native operators in a Graph | Import/wrapping overhead |
| ETX fused/unfused with the same operators | Fusion benefit |
| Strong real inference engine configuration | System competitiveness |

**Deliver:** frozen benchmark, machine-readable experiment records, failure tests
and automatically generated reports.

**Accept when:** known wrong results fail and cannot enter performance summaries;
valid conclusions can be recomputed; paired repeated measurements meet a
predeclared precision target. A lead, tie, loss or insufficient evidence is an
acceptable scientific outcome; none permits weakening correctness gates.

### Stage 2 — Make production operators safe to reuse

Unify shapes, layouts, precision, compile-time constants, launch conditions and
argument-block definitions in one operator specification. Generate the information
needed by graph construction, builds, loading and checking from that specification.
Reject unsupported configurations at compile/load time with the missing capability
or conflicting parameter named explicitly.

Keep an independent full-layer reference derived from model weights and semantics;
it must not reuse all parameters produced by the path under test.

**Deliver:** unified operator specification, generated arguments/grids and full-layer
differential checks.

**Accept when:** Qwen and Mixtral use the same workflow; a shape change no longer
requires maintaining duplicate handwritten constants; deliberate parameter conflicts
are caught before execution.

### Stage 3 — Automatically choose execution plans

Start with a bounded search over a few task granularities, workgroup configurations,
fusion boundaries and schedule modes. Use costs to prune clearly unsuitable
candidates, then select by measurement. [P4](../etx/passes/p4_schedule_mode.py)
already weighs imbalance against queue/cross-domain costs and records its reasons;
calibrate those costs and check whether they rank candidates correctly.

Allow ordinary Graphs, partial fusion and full fusion. A resource-heavy operator
should not unconditionally constrain every other phase's threads and registers.
Judge every candidate by complete decode time, not a faster isolated operator.

**Deliver:** candidate generator, selector, plan cache and decision records.

**Accept when:** on held-out inputs, automatic plans stay within a predeclared
margin of the best plan in the searched set; unsupported or materially regressing
configurations have a reliable fallback. This is bounded-search quality, not a
claim of global optimality.

### Stage 4 — Demonstrate transfer across models and workloads

Prioritize Qwen and Mixtral, gradually covering 1K/4K/8K contexts and 1/2/4
requests. Record request count, batch per forward and service concurrency separately.
Mixtral's report identifies MoE task width as a major gap: up requires multiple
execution waves, while down underutilizes workers. Use it to test reusable task
partitioning rules.

Search existing operator candidates and legal task configurations first. If new
operator implementations are necessary, isolate their gains in a separate ablation.

**Deliver:** model/workload support matrix, performance curves and fallback ranges.

**Accept when:** optimization rules do not depend on model names, new shapes have
independent tests, and reports include improved, neutral and regressing regions.

### Stage 5 — Integrate with an existing inference service

Integrate ETX decode into one existing framework while retaining its prefill,
request management and KV management. Ada-MK provides related precedent, but ETX
must independently validate interface and state-management costs.

Define KV layout, cache ownership, stream synchronization and output-delivery
boundaries first; then support request admission, completion, cancellation and
resource reclamation. Select plans according to load. Compare against a properly
configured engine: vLLM's [CUDA Graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
includes runtime dispatch among Graph modes, so a fixed weak baseline is inadequate.

**Deliver:** working backend integration, state-management interfaces and service
regression tests.

**Accept when:** complete request latency improves, tail latency and stability meet
agreed limits, and goodput under those latency limits has no unexplained regression.
Include requests of different lengths, not copies of one prompt. Distinguish cold
startup/model-loading costs from steady-state request latency.

### Stage 6 — Extend hardware and computational structures

Change hardware and model structure separately; do not simultaneously change GPU,
architecture, precision and scheduler. On hardware, reproduce on another MI300X,
then another AMD generation, then a selected NVIDIA platform. Another MI300X only
establishes instance reproducibility, not cross-architecture transfer.

Keep hardware differences data-driven where possible, while allowing new backend
code for instructions, visibility rules and operators. Adding YAML is not a
completed port; CUDA hardware validation remains open in the current repository.

For models, add general state capabilities before new structures. Hybrid attention,
recurrence and shared experts need new computation/state descriptions. The
[Qwen hybrid-architecture analysis](GENERALITY-2026-10-02.md) identifies persistent
cross-token state beyond the current model path's complete representation.

**Deliver:** a second hardware backend or a new model-structure capability, with a
complete migration record.

**Accept when:** most changes stay in the layers that should vary, core IR and
scheduling interfaces remain stable, and correctness and performance pass separately.

## 5. Research priorities and stopping rules

The main research question is joint selection of task granularity, placement,
synchronization and fusion boundaries for fixed semantics and a set of operator
candidates. Separate visibility correctness from cost prediction: the former can
produce wrong answers; the latter usually produces slower ones.

Advance one falsifiable hypothesis at a time. Measure the deciding unknown before
building the full design. Record the cost expected to fall, the outcome that would
refute the hypothesis and what work stops if it fails. Retain positive and negative
results in the data index.

Parallel research branches need separate owners, benchmarks and quality gates:

| Branch | Independent question and accounting |
|---|---|
| Low precision | Net gain after conversion, scale reads and quality changes; compare against the same precision, separately from bf16 fusion |
| Speculative decoding | Net gain from fewer target forwards; require multi-token verification, state commit/rollback and count only accepted, delivered tokens |
| Long context | Changes in KV, attention and weight-access bottlenecks; start after context-scaling experiments identify a concrete limit |

These branches may deliver larger total gains. They must not obscure a negative
result for the execution/compiler claim.

### Do not rebuild the refuted NT-off proposal

The [split-phase design addendum](DESIGN-split-phase.md#addendum-same-day-the-deciding-experiment-refuted-the-design)
records paired, interleaved batch-2 measurements on MI300X, with both requests
matching 32/32 tokens:

| Weight-load policy | Reported wall ms/step |
|---|---|
| NT on | 6.875 / 6.879 / 6.883 |
| NT off | 6.916 / 6.921 / 6.924 |

Disabling NT was slower. The addendum refutes the original cache-reuse proposal
and requires a mechanism that first demonstrates a shorter reuse interval. Keep
NT enabled for this measured setup. Revisit only after a substantive change in
hardware, workload or access order, with a new decisive experiment; do not repeat
construction of the rejected design.

Also defer simultaneous expansion to many new models, a full serving-framework
rewrite and dynamic-scheduler rewrites without evidence of benefit.

## 6. Responsibilities and review criteria

People may cover multiple roles, but keep the responsibilities distinct:

| Responsibility | Accountable for |
|---|---|
| Compiler/runtime | Operator specifications, dependencies, code generation and device protocols; fewer bespoke changes per model |
| Performance research | Bottleneck decomposition, candidates, costs and hardware experiments; falsifiable hypotheses and complete benefit accounting |
| Independent acceptance | Correctness, baselines, raw records and reports; optimization authors cannot relax gates to pass a candidate |

Coding agents may generate candidates, tests and log analysis. Acceptance rules
and final reference outputs need independent management.

Maintain four durable outputs: runnable code; correctness and fault-injection
tests; measurement data including negative results; reproducible research reports.
Review complete latency on target workloads, regressions on held-out configurations
and manual work needed to add a configuration. At the service stage, add goodput
under latency limits and output per GPU-time unit.

## 7. First three work packages

| Package | Scope | Completion condition |
|---|---|---|
| Trustworthy acceptance | Full output coverage, finite values, fault injection and failure exit status | Known errors are detected; invalid runs cannot enter performance results |
| Unified benchmark | Common timing interval, raw samples, version binding, paired trials and report generation | Current Qwen conclusion can be recomputed as lead, tie, loss or insufficient evidence |
| Unified operator specification | One source for shapes, constants, grids and arguments, initially Qwen and Mixtral | Duplicate derivation removed; incorrect configurations rejected before execution |

Start automatic task partitioning and fusion search after these packages pass.
The order is trustworthy results → safe operator reuse → automatic plan selection
→ model/workload transfer → service benefit → hardware/new-structure expansion.
Each stage should add reusable capability, not only a faster benchmark number.

Execution status for this update: documentation only; no runtime changes, GPU runs
or new measurements. No open issues or pull requests were found during the initial
repository check on 2026-10-03. Earlier P0 issue creation is therefore unconfirmed;
this roadmap does not imply that implementation work or issue tracking has begun.
