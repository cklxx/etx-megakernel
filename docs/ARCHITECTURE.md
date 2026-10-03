# ETX code architecture

The published design page is the specification.
This file maps its sections to code and states the rules for that code.
Section numbers refer to `docs/site/index.html`.

## Three hard decisions, enforced

| decision | where it lives | how it is enforced |
|---|---|---|
| hardware differences only as data | `etx/machine/arch/*.yaml` | `tests/test_no_arch_branches.py` fails if an arch name appears in `ir/ passes/ sim/ codegen/ frontends/` |
| capability vs cost | `capabilities:` / `visibility:` (correctness) vs `costs:` (performance) blocks in each YAML | `MachineModel.vis()` raises on a missing scope<br> `t_sync_us()` falls back and logs when a cost is uncalibrated |
| scheduler chosen by cost model with reasons | `passes/p4_schedule_mode.py` | every grid gets `plan.reasons[grid]`<br> `tests/test_passes.py::test_reasons_are_printed` |

## Layer map

| page § | layer | module | input -> output |
|---|---|---|---|
| 06 | L1 tile contract | `frontends/tileop.py`, `hip_link.py`, `triton_host.py` | TileBody handles<br> link-mode prototypes header<br> Triton skeleton |
| 07 | L2 IR | `ir/types.py`, `ir/edgemap.py`, `ir/dims.py` | Graph of TaskGrid + ETensor with symbolic dims and four edge-map forms |
| 07 6.4 | verification | `ir/verify.py` (checks 1-4), `passes/plan.py::verify_plan` (5-6 + static-queue order) | list of Issues / errors |
| 08 | L3 machine model | `machine/model.py`, `machine/arch/` | MachineModel: exec-domain tree, visibility, capabilities, costs, lowering snippets |
| 05 | performance model | `sim/protocol.py` | makespan, busy / wait / sched budget per plan |
| 09 P1 | tiling, resource classes | `passes/p1_tiling.py` | wg/CU, workers per domain, kernel instances |
| 09 P2 | affinity | `passes/p2_affinity.py` | task -> domain, event scopes, cross-edge counts |
| 09 P3 | event elimination | `passes/p3_event_elim.py` | rewritten graph (producer inlined into consumers' prologue) |
| 09 P4 | schedule mode | `passes/p4_schedule_mode.py` | static / dynamic / hybrid per grid + reason |
| 09 P5 | queues | `passes/p5_queues.py` | per-worker static queues, local/global capacities, push lists |
| 09 P6 | memory | `passes/p6_memory.py` | event memory type by scope, tensor placement hints |
| 09 P7 | prefetch | `passes/p7_prefetch.py` | cross-barrier weight prefetch entries |
| 10 | L5 lowering | `codegen/lowering.py` | `etx_lowering.h` (RELEASE/ACQUIRE/POLL/ARRIVE per scope, BACKOFF, DOMAIN_ID) |
| 10 | L5 kernel | `codegen/kernel.py`, `runtime/include/etx/*.h` | persistent kernel per device<br> `plan.json` for the host |
| 11 | dynamism | `ir/instantiate.py` + examples | symbolic shapes instantiated per step<br> runtime maps from routing tensors |

## Data flow of one compile

```
Graph (symbolic)  --verify-->  Instance (one step's tasks/events)
   --P1--> worker counts  --P2--> domains + scopes  --P3--> smaller graph (re-P2)
   --P4--> modes  --P5--> queues  --P6--> memory  --P7--> prefetch  --verify_plan-->  Plan
Plan --codegen--> megakernel_d<i>.hip + etx_lowering.h + etx_tiles.h + plan.json
Plan --sim-->     deadlock check, makespan, budget
```

Each graph and machine pair has one kernel.
Each step has a `plan.json` with shape scalars, descriptors, static queues, event counts, and push lists.
Shape dynamism requires a new plan, never a new kernel.

## Runtime ABI (fixed)

`runtime/include/etx/abi.h`: `etx_ctx` carries coordinate, shape scalars,
argument table, event base, position (domain, worker), LDS. Tile bodies are
`extern "C" __device__ void <symbol>(const etx_ctx*)`. Six primitives:
`etx_wait_<SCOPE>`, `etx_arrive_<SCOPE>`, `etx_pop`, `etx_push`,
`ETX_BACKOFF`, `ETX_DOMAIN_ID`. Everything scope-specific is a macro from the
generated lowering header.

## Adding things

* New hardware: add `machine/arch/<name>.yaml` with the tree, visibility per scope, capabilities, and costs.
  Run `bench/calib`. No Python changes are necessary.
* New frontend: produce TileBody handles.
  Answer the three questions in `Graph.call_device`.
  Link mode needs nothing else. Host-DSL mode needs a skeleton emitter like `frontends/triton_host.py`.
* New workload: a `build()` graph in `examples/`, plus `bindings()` and
  `runtime()` for its dynamism.
* New optimisation: add a pass that reads the Plan and the MachineModel.
  The pass writes data into the Plan and a line into `plan.log`.

## Known gaps (v0.3)

* Dynamic consumer lists are materialised host-side per step. The in-kernel
  inverse-map alternative (MPK-style ranges) is not implemented.
* Sentinel-value signalling is a capability bit and a helper, not a lowering
  option (measured: counters are the right primitive for barriers).
* CUDA emission untested on hardware. Triton emitter is a skeleton.
* Tiles are correctness references. No tuned unfused baseline yet.

## Verified on hardware (MI300X, 2026-09-22)

Split-K, the full MoE layer, and two-device GEMM + reduce-scatter all pass their CPU references.
See `README.md` and the design document's Appendix C for the numbers and five runtime rules from these hardware tests.
