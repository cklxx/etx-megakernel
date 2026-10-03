# ETX

ETX is a DSL-agnostic, chiplet-aware compiler for dynamic GPU megakernels.
It fuses many operators into one persistent kernel.
Shape dynamism and data-dependent control flow are first-class inputs.
Examples include MoE routing and speculative decoding.

One IR targets MI300X's 8 XCDs, MI250X's two GCDs, MI355X, H100, and B200.
The compiler reads a machine model instead of a branch on an architecture name.

Repository: https://github.com/cklxx/etx-megakernel (private). Design page (the spec): https://etc-dynamic-megakernel-arch.q1293822641.workers.dev
(source in `docs/site/index.html`). Code map: `docs/ARCHITECTURE.md`.

Proposed scope, evidence limits, and staged acceptance gates: [project roadmap](docs/ROADMAP.md)
(updated 2026-10-03).

## Fusing the state-of-the-art kernels (2026-09-25)

ETX now imports compiled kernels instead of implementations that ETX authors rewrite.
`etx/importer/` takes a kernel's AMDGPU IR and converts `amdgpu_kernel` to an inlinable device function.
The current source is vLLM's HIP code. Triton IR is next.
Arguments come from an argument block. Block/thread ids come from the ETX task.

LDS uses a shared arena. Each workgroup can contain several blocks with slot barriers.
The importer generates ETX tiles around this function.
`examples/vllm_llm` runs Qwen decode with the twelve kernels that vLLM uses on MI300X at batch 1.
These include `wvSplitK`, `paged_attention_rocm`, `rms_norm`, `fused_add_rms_norm`, `rotary_embedding`, `reshape_and_cache`, and `silu_and_mul`.
Execution follows vLLM's stream order, with one ETX grid per vLLM launch.

The comparison uses the same kernels in three forms: ETX fusion, separate ETX launches through a HIP graph, and vLLM with `custom_ops=all`.
A vLLM kernel trace bounds the possible fusion benefit.
See design section 15.9.

    python -m etx.importer.vllm_kernels --vllm ~/code/vllm --out build/imp --block rms_norm_2d=512 --block reshape_and_cache=128
    VLLM=~/vllm bash examples/vllm_llm/build.sh ~/llm/qwen3-8b            # on the GPU host
    nohup bash examples/vllm_llm/vm_run.sh > ~/vl.log 2>&1 &              # the whole comparison on a fresh VM

## Status: ETX beats vLLM at batch 1 (2026-10-02)

Conditions: Qwen3-30B-A3B bf16 decode, batch 1, one MI300X, wall clock.
The run matches 32/32 tokens against HF greedy.
Every DSL operator of the last layer is within 2 bf16 ulps of a CPU reference:

| | ms/token | vs vLLM |
|---|---|---|
| vLLM 0.27.1 default | 4.863 | — |
| vLLM + AITER | 4.975 | +2.3% |
| **ETX megakernel** | **4.823** | **−0.8% — ahead** |
| ETX unfused — same tiles, HIP-graph replay | 4.998 | +2.8% |

**The recorded round23 comparison shows a 1.2% fusion benefit for the same DSL tiles: 4.937 versus 4.998 ms/token.**
This percentage does not describe the newer 4.823 entry above.
The corresponding newer paired raw logs are absent from the inspected archive. Thus, a new fusion percentage remains unverified.
The previous claim of a 4.2% loss refers to the day before.

The system improved 17.5% against itself, with confirmation on a second machine.
That machine used bare metal and ROCm 7.2.4.
Its fusion margin was +0.224 against −0.235 on 2026-09-30.
These DSL results do not establish the all-imported claim.
The all-imported comparison still loses in [docs/WHY-DSL-TILES-2026-10-01.md](docs/WHY-DSL-TILES-2026-10-01.md).

The acquire fix saved −0.317 ms. Each task previously used `buffer_inv sc1`, which invalidated the whole L2 and made each phase start cold.
Removal of an earlier fold saved −0.231. That fold made 448 expert tasks per layer each repeat a 128-logit softmax.

Asynchronous host work per step saved −0.152. Non-temporal weight loads saved −0.106.
The coherent paths previously issued one memory request per lane per 2-byte element.
Vector accesses saved −0.105 in the folded norm and −0.074 in the attention prologue.

ETX is ahead of vLLM by 0.8% in interleaved runs with non-overlapping ranges.
The budget explains the remaining time.
Handoff is at the event protocol's floor. Workgroup start delay is below the hardware dispatcher's own jitter.
Host work is 0.135 ms. The instruction cache hit rate is 99.90%.

**The orchestration budget is exhausted. Our own hand-written tiles account for 1.29 ms of the step.**
Tile quality was never ETX's claim. Generality on 2026-10-02: Mixtral-8x7B runs 32/32 through the same compiler with zero compiler changes.
Its shape is top-2 of 8, G=4, K=14336.
Every failure came from host-side shape glue. Shape keys now make unknown shapes fail with an explicit import error.
The DSL path reaches 10.365 ms/token against vLLM's 8.686. The floor is ~7.3-8.5.

Qwen-tuned task-width constants explain the remaining gap. See [docs/GENERALITY-2026-10-02.md](docs/GENERALITY-2026-10-02.md).
The config rejects out-of-class architectures by name, such as Qwen3.6's hybrid linear attention.

Full report: [docs/ETX-report-2026-10-01.pdf](docs/ETX-report-2026-10-01.pdf), per-phase budget in
[docs/BUDGET-2026-10-01.md](docs/BUDGET-2026-10-01.md), measured data index in [docs/DATA.md](docs/DATA.md).

## Earlier: the real model runs (2026-09-23)

`examples/dsv2lite/` puts fleet-mi300x's DeepSeek-Coder-V2-Lite decode graph in ETX. A shim links fleet's own tile bodies.
On one MI300X, it decodes 32 tokens identical to HuggingFace greedy. All 27 layers pass fleet's per-layer gate.
It runs at **3.76-3.78 ms/token** against fleet's hand-written 3.557 ms on the same GPU.
The earlier 3.70 measurement used the relay's hierarchical acquire, which proved unsafe and is now off.

Compiler and runtime rules reduced the gap from 54% to 6%, without changes to fleet's tile code.
The rules include last-arriver flush, agent-scope polling, in-body q_c wait, and one acquire per workgroup.
They also include one call site per tile body, a lean all-static loop, and per-grid immediates.
The sequence included a per-XCD relay with hierarchical acquire and a whole-program device build.
`examples/dsv2lite/ab.sh` re-runs the A/B on one VM. Build:

```bash
bash examples/dsv2lite/build.sh /path/to/fleet-mi300x      # needs hipcc + fleet's build/ (weights, cache, golden tokens)
cd /path/to/fleet-mi300x && /path/to/etx/build/dsv2lite/run --tokens 32 --context 1024 --repeat 2
ETX_TRACE=1 .../run --tokens 4          # per-phase timeline (wait / body / first-ready / last-done per task type)
```

## More models, same compiler (2026-09-24)

`examples/llm` builds the graph from a Hugging Face `config.json`.
It runs one set of generic batch-1 tiles, with no model-specific runtime code.
The results use one MI300X and Hugging Face greedy decode as the reference, with bf16 and 32 tokens.
They are deterministic over three runs:

| Model | Teacher-forced argmax | Free-running | ms/token |
|---|---|---|---|
| Qwen2.5-1.5B (dense, QKV bias, tied) | 31/32 (the miss is an exact HF tie) | 24/32, diverging at that tie | 5.11 |
| Qwen3-8B (dense, q/k norm) | 32/32 | 32/32 | 13.4 |
| Qwen3-30B-A3B (MoE, 128 experts, top-8) | 32/32 | 32/32 | 14.4 |

The tiles are general, not tuned. On a fresh VM, `bash examples/llm/vm_run.sh` prepares, builds
and runs all three.

Head-to-head with the **generic, untuned** tiles, 1024-token context, batch 1, one MI300X
(2026-09-25; `examples/llm/vm_bench.sh`). Superseded for Qwen3-30B-A3B by the 2026-10-01 table above, which
uses the DSL tiles and the coherence design:

| Model | vLLM, best of default / AITER | ETX tiles, unfused (one kernel per grid, HIP graph) | ETX megakernel | Hand-written fleet |
|---|---|---|---|---|
| Qwen2.5-1.5B | 1.86 ms | 4.05 ms | 4.47 ms | - |
| Qwen3-8B | 4.91 ms | 9.45 ms | 10.07 ms | - |
| Qwen3-30B-A3B | 4.84 ms | 9.07 ms | 10.48 ms | - |
| DeepSeek-Coder-V2-Lite | 4.12 ms (AITER<br> default 6.36) | - | 3.78 ms (fleet's tiles) | 3.57 ms |

With fleet's tuned tiles, ETX beats vLLM by 8%. With the generic tiles, it is about 2x slower.
The per-phase trace attributes that gap to the tiles, not to synchronization.

The per-XCD sliced graph in `examples/llm/model_sliced.py` uses two device-wide events per layer.
It puts the megakernel within 2-5% of the same tiles as one kernel per grid through a HIP graph.
It does not put the megakernel ahead. At batch 1, the graph is a barrier chain with no overlap opportunity.
See design document, section 15.7.

## Starting point (reworked 2026-09-23)

The architecture is broad. The proof is narrow. The target is one real model on
one machine with a known hand-written result: DeepSeek-Coder-V2-Lite-Base,
batch 1, one MI300X, where fleet-mi300x runs at 3.60 ms/token. Order of work:
M0 baselines (done: the per-step fixed cost is decomposed, see below), M1
fleet's task graph in ETX (done: `examples/dsv2lite/graph.py`, verified and
simulated), M2 fleet's tile bodies through a shim (next), M3 match 3.60 ms,
M4 generalise.

Baselines on MI300X: cooperative launch of an empty 608-workgroup kernel 16 µs,
ordinary launch 1.6 µs, empty worker loop 28 µs, unfused two-kernel split-K
step 5.7 µs. The v0.3 per-task completion atomic and idle atomic polling cost
15 µs per task slot; removing them (v0.4 runtime) took the split-K step from
101 µs to 23 µs (cooperative) / 12 µs (ordinary, `ETX_LAUNCH=ordinary`).

## What runs today (no GPU needed)

```bash
make setup                      # uv venv + deps
make test                       # 30-ish tests: IR, edge maps, machine tables, passes, simulator, codegen
.venv/bin/python -m etx archs
.venv/bin/python -m etx explain examples/moe_layer.py --arch gfx942 --sim
.venv/bin/python -m etx compile examples/moe_layer.py --arch gfx942 --out build/moe_gfx942 --sim
```

`compile` verifies the graph and runs the seven L4 passes.
It writes the persistent kernel source, `megakernel_d0.hip`, the per-arch lowering header, and tile prototypes.
It also writes `plan.json` with queues, descriptors, event layout, and push lists.
`decisions.log` explains each static / dynamic / hybrid choice and the events that recomputation eliminates.

`--sim` runs the host-side protocol simulator.
It reports the deadlock check, makespan, and busy / wait / scheduling budget.

## What has run on hardware (MI300X, ROCm 7.2.4, 2026-09-22)

* `bench/calib/*.hip` ran and their numbers are in `etx/machine/arch/gfx942.yaml`
  (`costs:` block, marked MEASURED): counter one-way 642 ns same-XCD / 742 ns
  cross-XCD with the inv-L1 poll, DEVICE fences 181 / 120 ns, flag one-way ~930 ns
  with zero stale payloads, ticket-ring queue 76 M pops/s at 512 poppers versus a
  CAS ring that collapses past 8.
* All three examples executed on the GPU with real tiles and passed their CPU
  references: split-K (static / hybrid / dynamic, two sizes), a full MoE layer
  (device-side top-2 routing, runtime counter initialisation, gather and range
  maps; 3.2 ms per step with per-token GEMV reference tiles), and GEMM +
  reduce-scatter across two MI300X (shared system-scope events in fine-grained
  memory, peer reads over xGMI, 0.365 ms per step).
* Measured on the short-task MoE configuration (`ETX_MOE_SMALL=1`, B=64): static
  0.445 ms vs hybrid 0.654 ms vs dynamic 0.653 ms; Pass 3 event elimination 0.546
  vs 0.653 ms in the hybrid regime. The l2-warm prefetch showed no effect.
* Cross-device: system-scope flag one-way 10.6 µs, peer store 285 GB/s; whole-GPU
  counter barrier 8.6 µs at 304 workgroups.
* Five runtime defects were found only on hardware and are now rules with tests
  (see `docs/ETX_Technical_Design.md`, Appendix C).

```bash
python -m etx compile examples/splitk_sum.py --arch gfx942 --out build/splitk_sum
hipcc -O2 --offload-arch=gfx942 -fgpu-rdc -I etx/runtime/include -I build/splitk_sum \
      build/splitk_sum/megakernel_d0.hip examples/tiles/splitk.hip examples/hosts/splitk.hip -o build/splitk_sum/run
build/splitk_sum/run 50
```

## What does not run yet

* CUDA emission is untested (no NVIDIA machine was available).
* The Triton host-DSL emitter produces a skeleton, not a runnable kernel.
* The tiles are correctness references (per-token GEMVs), so there is no
  fusion-gain figure against a tuned unfused baseline yet.

## Layout

```
etx/ir         L2  Event Tensor IR, edge maps, instantiation, six verification checks
etx/machine    L3  MachineModel + arch/*.yaml (gfx90a, gfx942, gfx950, sm_90, sm_100)
etx/passes     L4  P1 tiling/resource classes, P2 affinity, P3 event elimination,
                   P4 schedule mode, P5 queues, P6 memory, P7 cross-barrier prefetch
etx/codegen    L5  lowering table -> etx_lowering.h, persistent kernel emitter, plan.json
etx/runtime    L5  ABI header, device primitives (arrive/wait/pop/push) built on the macros
etx/frontends  L1  tile contract (hip_link / cuda_link / triton_fn / builtin), adapters
etx/sim            protocol simulator (deadlock, makespan, critical-path budget)
examples/          splitk_sum (paper fig. 3), moe_layer (both dynamisms), gemm_rs (2 devices)
bench/calib/       phase-0 microbenchmarks that fill the cost tables
tests/             includes test_no_arch_branches.py enforcing "hardware is data"
docs/              ARCHITECTURE.md (code map), DESIGN-v0.1.md, site/ (published page)
```
