"Dependent global-load chains in AMDGPU assembly (gfx9 syntax).\n\nRule, agreed by the review on 2026-10-08:\n* A hop is one wait on the data chain. Follow each load result register, VGPR or SGPR, to the address or the\n  control of the next load, or to the event publish. Each wait on that chain is one hop, for any count value.\n* Loads in flight together are one hop: a load costs one hop above the deepest value that it reads.\n* Infeasible paths are pruned. A lattice tracks each 64-bit mask as Z, no lane, or E, the entry exec mask.\n  A pinned register is a wave-uniform register with a known value, for example the slot index of an\n  imported kernel. A v_cmp of a pinned register with an immediate gives Z or exec. A branch on a known\n  mask takes one edge only. Copies of a body for different slots then do not chain.\n* Control is control dependence. A vector memory or LDS operation reads exec. A scalar branch adds its\n  condition until the immediate post-dominator of the branch block. `s_or_b64 exec, exec, sN` and\n  `s_mov_b64 exec, sN` restore a saved mask: exec then has the value of sN only. Code after the\n  reconvergence point is not control dependent on the branch. This is the agreed rule for data dependency.\n* Spills continue the chain. A scratch store keeps the value of its data register in a slot keyed by the\n  base register and the offset. A scratch reload reads that slot and costs one hop of class 'scratch'.\n  SGPR spills to VGPR lanes, v_writelane and v_readlane, and AGPR copies keep the value without a hop.\n* LDS is not a global hop. A ds_read reads the LDS value, which is the deepest value that any ds_write stored.\n  It counts as an 'lds' step with no cost.\n* A global store completes one hop after its sources. A store enters the chain only through the drain at the\n  end of the region, which is the release before the publish. Stores with sc1 are coherent. Other stores\n  drain at the cached price. A drain before s_barrier is a vmcnt(0) wait off the data chain. It does not\n  count by default. barrier_drain=True counts it, for a global-memory exchange between waves.\n\nClasses: 'coherent' vector load with sc1, 'plain' vector load without sc1, 'scalar' s_load, 'scratch' reload,\n'store' drain, 'lds' step. Prices, in us, come from a preset:\n* 'warm': coherent = RTT, plain = cached, scalar = cached, scratch = cached, store = RTT with sc1, else cached.\n* 'acquired': as 'warm', but plain = RTT. A task after `buffer_inv sc1` misses on each new line.\n\nLoops: layout order is structured. A back edge goes to a block at or before the branch. The tool runs the\nregion with each loop at one trip, then with one and two extra trips of that loop. The difference gives the\nhops per extra trip. A trip count is 'constant N at line L' when the loop exit compares with an immediate."
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from .syntax import (
    CLASSES,
    _REG,
    _ALIASES,
    _BRANCH,
    _NODEF,
    _CARRY_OUT,
    _SALU_SCC,
    regs,
    Insn,
    Block,
    _IMPDEF,
    _impdef_regs,
    parse,
    function_lines,
    label_line,
    blocks,
)
from .state import (
    BOT,
    ZERO,
    _key,
    vmax,
    Prices,
    State,
    _f_or,
    _disjoint,
)
from .execute import (
    hop,
    _slot_key,
    _CMP,
    _mkey,
    _mval,
    _nz,
    _m_and,
    _m_or,
    _m_xor,
    _m_andn2,
    _exec,
    _scc,
    LANE_MODELS,
    _forget,
    _forms,
    _masks,
    feasible,
    step,
    _step,
)
from .analysis import (
    Region,
    region,
    _cfg,
    _ipdom,
    _loop_order,
    run,
    _loops,
    chain_end,
    counts,
    trip_source,
    analyse,
    main,
    owner_of,
    owned_region,
)

if __name__ == "__main__":
    raise SystemExit(main())
