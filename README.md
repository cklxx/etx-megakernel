# ETX

A DSL-agnostic, chiplet-aware compiler for dynamic GPU megakernels. It fuses
many operators into one persistent kernel while keeping shape dynamism and
data-dependent control flow (MoE routing, speculative decoding) as first-class
inputs, and it targets different machines (MI300X's 8 XCDs, MI250X's two GCDs,
MI355X, H100, B200) from one IR by reading a machine model instead of
branching on an architecture name.

Repository: https://github.com/cklxx/etx-megakernel (private). Design page (the spec): https://etc-dynamic-megakernel-arch.q1293822641.workers.dev
(source in `docs/site/index.html`). Code map: `docs/ARCHITECTURE.md`.

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
* The generated split-K megakernel (`examples/splitk_sum.py`, static schedule)
  executed on the GPU and matched the CPU reference bit for bit in the fp32
  tolerance: 320 tasks, 64 events, 85 µs best per step.

```bash
python -m etx compile examples/splitk_sum.py --arch gfx942 --out build/splitk_sum
hipcc -O2 --offload-arch=gfx942 -fgpu-rdc -I etx/runtime/include -I build/splitk_sum \
      build/splitk_sum/megakernel_d0.hip examples/tiles/splitk.hip examples/hosts/splitk.hip -o build/splitk_sum/run
build/splitk_sum/run 50
```

## What does not run yet

* Only the split-K example has real tile bodies; `examples/tiles/moe.hip` and
  `gemm_rs.hip` are ABI stubs.
* CUDA emission is untested; `t_dev_ns` (cross-GPU) is uncalibrated (needs a 2-GPU VM).
* The Triton host-DSL emitter produces a skeleton, not a runnable kernel.
* `etx_deps_ready` always takes the static head first; the ready bitmap is planned.

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
