# ETX

A DSL-agnostic, chiplet-aware compiler for dynamic GPU megakernels. It fuses
many operators into one persistent kernel while keeping shape dynamism and
data-dependent control flow (MoE routing, speculative decoding) as first-class
inputs, and it targets different machines (MI300X's 8 XCDs, MI250X's two GCDs,
MI355X, H100, B200) from one IR by reading a machine model instead of
branching on an architecture name.

Repository: https://github.com/cklxx/etx-megakernel (private). Design page (the spec): https://etc-dynamic-megakernel-arch.q1293822641.workers.dev
(source in `docs/site/index.html`). Code map: `docs/ARCHITECTURE.md`.

## Status: the real model runs (2026-09-23)

`examples/dsv2lite/` is fleet-mi300x's DeepSeek-Coder-V2-Lite decode graph in
ETX with fleet's own tile bodies linked through a shim. On one MI300X it
decodes 32 tokens identical to HuggingFace greedy with all 27 layers inside
fleet's per-layer gate, at **3.70 ms/token** against fleet's hand-written
3.556 ms on the same VM. The gap closed from 54% to 4% through compiler and
runtime rules (last-arriver flush, agent-scope polling, in-body q_c wait, one
acquire per workgroup, one call site per tile body, a lean all-static loop,
per-grid immediates, a per-XCD relay with hierarchical acquire, and a
whole-program device build), with no change to fleet's tile code.
`examples/dsv2lite/ab.sh` re-runs the A/B on one VM. Build:

```bash
bash examples/dsv2lite/build.sh /path/to/fleet-mi300x      # needs hipcc + fleet's build/ (weights, cache, golden tokens)
cd /path/to/fleet-mi300x && /path/to/etx/build/dsv2lite/run --tokens 32 --context 1024 --repeat 2
ETX_TRACE=1 .../run --tokens 4          # per-phase timeline (wait / body / first-ready / last-done per task type)
```

## Starting point (reworked 2026-09-23)

The architecture is broad; the proof is narrow. The target is one real model on
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

`compile` verifies the graph, runs the seven L4 passes, writes the persistent
kernel source (`megakernel_d0.hip`), the per-arch lowering header, tile
prototypes, `plan.json` (queues, descriptors, event layout, push lists) and
`decisions.log` (why each subgraph is static / dynamic / hybrid, which events
were eliminated by recomputation). `--sim` runs the host-side protocol
simulator: deadlock check, makespan, busy / wait / scheduling budget.

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
  vs 0.653 ms in the hybrid regime; the l2-warm prefetch showed no effect.
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
