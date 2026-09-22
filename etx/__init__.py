"""ETX: a DSL-agnostic, chiplet-aware compiler for dynamic GPU megakernels.

Layers (see docs/ARCHITECTURE.md):
  etx.frontends  L1  tile contract (TileOp) and frontend adapters
  etx.ir         L2  Event Tensor IR, edge maps, verification
  etx.machine    L3  machine model (exec-domain tree, visibility, capability, cost) from YAML
  etx.passes     L4  placement and scheduling passes -> Plan
  etx.codegen    L5  lowering table + persistent-kernel emitters
  etx.sim            host-side protocol simulator (deadlock, makespan)
"""

__version__ = "0.1.0"
