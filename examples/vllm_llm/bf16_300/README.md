# BF16 300 tok/s: CPU probe preparation

This implements preparation for step 2 of section 7 of
`docs/qwen/PLAN-300TOKS-2026-10-11.md`. It is **not a model implementation,
performance result or correctness certificate**. No GPU, provisioning or network
spend was used. The vendor-MoE-swap route remains CLOSED at the supplied
30.72 us/layer; reduced precision is excluded.

The compiled numerical resource gates pass. Actual co-residency is UNVERIFIED,
so the complete fusion gate is INCOMPLETE: **stop before model integration and
hardware**. The plan's thresholds remain 2.083 us/layer exclusive removable
skeleton work and 1.042 us/layer net fusion saving. Neither has been measured or
credited. The current LEAD build, packs, results and certificate are untouched.

## Prepared deliverables

1. `source.py:13,94` emits twelve independent sparse timestamp pairs in the
   current MoE up/down tile: arguments, routing, staging, partial zeroing,
   pipeline, one named row's flush, join, gather, reuse, publication, whole body
   and an empty pair. The body begins before argument/grid reads and retains
   the production scalar argument path. One selected task/wave holds two
   timestamps, then saves a 32-byte record after the body. `s_memrealtime` ticks
   are 10 ns. These are **issue intervals**, with no added retirement fences.
   The existing release/barrier remains at publication. A sampled flush is not
   all flushes, and overlapping pairs cannot be added together.
   `dispatcher.py:21,34` adds descriptor/FIFO, wait/setup/acquire, exit/
   publication and empty probes to an inlined two-task graph of the original
   up/down tiles. This prices ETX machinery separately from tile-loop work.
   `source.py:57` returns exactly the unchanged production emitter output;
   all instrumentation is confined to independent translation units.
   `attribution.py:11` rejects missing last-task/last-wave evidence, differing
   allocation (including SGPR spills), differing session/compiler/math,
   missing import/token checks or body perturbation greater than 0.5 us. It
   returns no exclusive removable-work credit. Most compiled probe allocations
   differ from the control/calibration and are currently unqualified.

2. `protocol.py:21` models single-owner prep, all-wave store drainage, early
   publication, epoch acquire, unique partition execution/arrival, all-partition
   acquire, original-index reduction, output drainage, retirement and quiescent
   reset including uint32 rollover. `protocol.py:102,106,123` checks inactive-slot
   participation, FIFO/dependency liveness and queue reset. The dry run enumerates
   all 120 completion orders with sixteen producing waves; an order-sensitive
   sum must remain 12.0 regardless of arrivals.
   `fusion.hip:30` compiles the real prep body and canonical imported
   `attn_mfma16_g8`/`attn_reduce` adapters, with bounded context live ranges and
   sequential reuse of their LDS arena. External head state is 40 bytes;
   the four-word broadcast adds 16 bytes. Actual XCD identity selects a queue.
   Each nonempty queue contains five tickets for one head, with ticket zero
   owning prep; surplus blocks retire. Epoch/head/partition coordinates are
   checked. Prep publishes before its attention call, each producing wave
   drains before completion, and the last partition invokes the unchanged
   original reduce over indexed global partials. The ready/output token is
   `(uint64_t)epoch + 1`, leaving zero unready even at uint32 rollover. Imported
   adapters retain uniform rounds and participation of inactive waves.
   This is a standalone kernel; a host must prove five resident slots per
   nonempty XCD and reset queues/state only after all blocks retire. It has no
   integration with the model's subsequent ETX output event.

3. `source.py:62` prepares an isolated **up-only** specialization for K=2048,
   32 gate plus 32 up rows, rpt=32, sixteen waves and 1024 threads. Each wave
   owns four complete rows. It removes empty partial zeroing and cross-wave
   gathering; preserves the original wave sum and virtual zero additions in
   original wave order, bf16 rounding and SiLU; and retains the gate/up join
   before SiLU. The original path is the fallback. Down remains unchanged and
   receives no up ownership proof. GPU byte equivalence is still required.

4. `build.py:30,52,81` emits isolated identities and parses gfx942 compiler
   resource metadata. The numerical gate rejects more than 128 VGPR, more than
   29,752 bytes LDS, or zero estimated resident blocks. It records AGPR, both
   spill counts and scratch as well. Static register/LDS/thread capacity is an
   estimate, not verified co-residency. All builds are timing-ineligible.

5. `fusion.hip:18,67,90,107` adds the checking-only stage snapshots immediately
   after prep, each attention partial, and reduction, before successors or
   publication. A snapshot list can cover all original output ranges, including
   in-place qkv/KV. `check.py:13,20,28` captures immutable bytes, original binary
   identity and constituent/partition identity, then checks coverage and exact
   equality. References must come from independent original constituent
   launches, never from the fused adapter. No original launches were executed
   here and no certificate is issued. A future driver must supply complete
   output lists; this preparation cannot itself establish their provenance.

## Compiled gates

Local toolchain: Homebrew LLVM 23, gfx942, local ROCm 7.2.4 headers/device
libraries. These are not ROCm 7.14.0 VM compiler results. The table comes from
`results_bf16_300_cpu_20261011/compile.json` and assembly metadata, not declarations.
`up/down` entries list separate kernels. All rows have AGPR=0.
`V/S` denotes VGPR/SGPR spill counts; scratch is bytes/lane. Blocks/CU is a
static upper-capacity estimate for 1024-thread blocks. **Actual co-residency is
UNVERIFIED for every row.**

| Variant suffix (`bf16_300_`) | VGPR | V/S spills | Scratch | LDS bytes | Static blocks/CU |
| --- | ---: | ---: | ---: | ---: | ---: |
| moe_control (up/down) | 56/54 | 0/0 | 0 | 21,256 | 2 |
| moe_fixed_up (up/down) | 54/54 | 0/0 | 0 | 21,256 | 2 |
| fusion | 110 | 0/0 | 0 | 8,344 | 1 |
| fusion_check | 112 | 0/0 | 0 | 8,344 | 1 |
| timestamp_arguments (up/down) | 60/58 | 0/0 | 0 | 21,256 | 2 |
| timestamp_routing (up/down) | 60/56 | 0/0 | 0 | 21,256 | 2 |
| timestamp_stage (up/down) | 54/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_zero (up/down) | 62/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_pipeline (up/down) | 60/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_flush (up/down) | 62/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_join (up/down) | 60/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_gather (up/down) | 60/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_reuse (up/down) | 60/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_publication (up/down) | 56/54 | 0/0 | 0 | 21,256 | 2 |
| timestamp_body (up/down) | 58/56 | 0/0 | 0 | 21,256 | 2 |
| timestamp_empty (up/down) | 56/54 | 0/0 | 0 | 21,256 | 2 |
| dispatcher_control (persistent) | 128 | 0/0 | 0 | 21,148 | 1 |
| dispatcher_descriptor (persistent) | 126 | 0/0 | 0 | 21,148 | 1 |
| dispatcher_wait (persistent) | 127 | 0/1 | 0 | 21,148 | 1 |
| dispatcher_exit (persistent) | 127 | 4/1 | 20 | 21,148 | 1 |
| dispatcher_empty (persistent) | 127 | 0/1 | 0 | 21,148 | 1 |

Every dispatcher build also emits the same `unfused_d0` reference:
54 VGPR, 0 AGPR, 0/0 spills, 0 scratch, 21,120 bytes LDS, static capacity 2.
The standalone fusion's allocation is not a projection of the full-model
megakernel. Adding it to that kernel can change registers/spills/residency and
requires a new gate.

## Isolation and CPU use

Variants are named `bf16_300_<lever>`, compiled into
`build/bf16_300/<lever>`. Dry-run sources go into a separate `dry_run/` child,
so a dry run cannot replace a compiled identity. Each compiled identity hashes
the emitted source, IR, assembly, compiler, device libraries and imported
constituents. Full unfiltered import/compiler logs stay beside each build.

`packed/bf16_300/cpu_manifest.json` is deliberately metadata only: weights,
golden, KV and certificate are absent, timing eligibility is false. It does not
claim the plan's full executable/input/pack identity. Results and all mutation
logs are under `results_bf16_300_cpu_20261011/`. No old build/pack/result/certificate
is an output destination. The pre-existing worktree `build` symlink into the
main checkout was replaced with a local ignored directory before compiling.

Use the main checkout's interpreter:

```bash
/Users/ckl/code/etx/.venv/bin/python -m examples.vllm_llm.bf16_300.build
ETX_DEVLIBS=/Users/ckl/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode \
  /Users/ckl/code/etx/.venv/bin/python -m examples.vllm_llm.bf16_300.build --compile
```

The optional `--task`, `--wave`, `--row` compile-time selectors choose one sparse
tile sample; use a separate preserved build directory for each selector if
collecting future samples. Default selectors are zero. Dispatcher probes sample
task 1 and use the existing four-u64/task trace allocation. The two-task graph
expects original up/down DSL arguments at offsets 0/160. No GPU launcher or
full-model pack is included. Timestamp variants must never be timed as a
performance configuration or carry a historical certificate.

## Acceptance and mutation evidence

Final command results are recorded in the local results directory:

* Main interpreter, entire suite:
  `/Users/ckl/code/etx/.venv/bin/python -m pytest -q`: exit 0;
  **1265 passed, 16 skipped in 176.93s (0:02:56)** (`pytest.log`).
* Local gfx942 command above: exit 0; all 21 variants compiled and passed the
  numerical gate (`compile.log`, `compile.json`).
* `ETX_PYTHON=/Users/ckl/code/etx/.venv/bin/python bash examples/vllm_llm/local_matrix.sh build/bf16_300/matrix`:
  exit 0; **16/16** builds reported `rc=0` (`local_matrix.log`). Each complete
  build log is at `build/bf16_300/matrix/<model_variant>/matrix.log`.
* CPU dry run: exit 0; 21 variants prepared, 120 completion orders, sixteen
  waves, epoch rollover `4294967295 -> 0`, `gpu_certificate=false`
  (`dry_run.log`, `dry_run.json`).
* `bash -n examples/vllm_llm/local_matrix.sh etx/importer/hip_ir_local.sh`:
  exit 0.
* `/Users/ckl/code/etx/.venv/bin/python -m examples.vllm_llm.bf16_300.mutations`:
  exit 0; **71/71 RED**, all files restored (`mutations.log`,
  `mutations/report.json`). Run mutations alone, never beside builds/tests.

Each mutation reads the edited file back and persists/prints token counts
**before** invoking or reading the verdict: original `1 -> 0`, replacement `1`.
All targeted pytest invocations exit 1 with a failed relevant check, not a
collection failure. The 33 actual Python guard removals exercise invalid owners,
premature publications, missing wave drains, stale epochs, duplicate execution/
arrivals, incomplete fan-in/acquires, reduction order, duplicate reduce,
premature retirement/reset, missing slot participation, queue cycles/endpoints,
snapshot coverage/identity/bytes, source anchors, precision/batch identity,
compiler metadata/resource gates, selectors and attribution. The remaining
27 fusion, 4 fixed-up and 7 timestamp token mutations exercise CPU source
correspondence checks. They do **not** execute mutant GPU code or establish
hardware memory visibility. `tests/test_bf16_300.py` has 80 passing CPU tests.

## Stopping point and plan discrepancy

There is no full-model fusion integration, output-event bridge, GPU host driver,
fresh original-constituent byte check, 32/32 HF token result, runtime
co-residency measurement, phase attribution or performance number. Probes with
changed allocation/spills cannot pass the attribution gate. A VM-compiler
container check was not performed; it remains required before any provisioning.
These are explicit blockers rather than inferred passes. This task stops at the
prepared, compiled standalone probes/protocol and CPU checker.

The plan's section 2 resource statement calls attention-reduce **2,176 B per
slot** and estimates `8 * 2176 = 17,408 B`. The locally imported declaration is
**272 B per slot, eight slots = 2,176 B total**
(`build/bf16_300/fusion/imp/imports.json`),
while attention is 8,192 B. Sequential reuse therefore takes the 8,192-byte
arena, with the compiled complete fusion allocation at 8,344 B. The plan was
not edited. Its 29,752-byte limit remains the gate. There is no other identified
brief/plan conflict; the actual-co-residency requirement cannot be fulfilled
under the task's CPU-only constraint and is reported as incomplete.

## In-body follow-up, 2026-10-11

The no-launcher/no-pack statements above describe the original preparation
checkpoint. `launch_build.py`, `launcher.hip` and `launch.py` now build/replay
the twelve sparse MoE probe pairs and read every raw sample and completion row.
`pack.py` reuses the full-model `prep.py` weight format and exports portable
original argument/input snapshots plus independent original-output references
from the existing checking host (`ETX_BF16_CAPTURE_BUILD=1`). Nothing runs on a
GPU during `--local` builds or `pack.py --dry-run`.

The launcher accepts original/candidate selector, pair, captured argument block,
layer/task/wave/row, iterations, deadline, output prefix and independent reference.
It alternates unchanged original/stamped pairs; the selector chooses the primary
reported arm, and all returned records identify the diagnostic companion.
The unstamped original has no fabricated timestamps. Replay covers one expert's
MoE up/down task group, including mode-zero `_fold` aliases; it does not implement
the standalone fusion prototype, dispatcher probes or full-model token checking.

See [the record/CPU evidence/session contract](../../../docs/qwen/INBODY-INSTRUMENT-2026-10-11.md)
and `examples/sessions/inbody.py` for the single 900-second gpu_jobs attempt.
GPU perturbation, full-model correctness and exclusive removable-work credit
remain unmeasured. The enabled allocations differ; no historical certificate
or performance eligibility is transferred to these probes.
