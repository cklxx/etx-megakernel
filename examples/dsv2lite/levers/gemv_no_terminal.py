"""Lever Q, gemv_no_terminal: qkv's q and kv_a GEMVs stop issuing the never-reduced terminal group.

fleet's pipelined GEMV (gemv.h @3fb67a6) issues the next group before reducing the current one and only then checks
the end, so every active wave issues exactly one group past its last row ("unconditional; clamped past the end").
This lever peels the tail in a private copy of the helper: full pairs keep issue-before-reduce in one basic block, the
last one or two groups reduce without issuing another. Scope: the three gemv_rows calls in TASK_QKV_FUSED only; every
other call site keeps the original helper byte for byte. Consumed padded rows stay.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _function, _once

PIPELINE_OLD = """    FLEET_ISSUE_GROUP(bufA, k)
    while (true) {
        const int k2 = k + stride;
        FLEET_ISSUE_GROUP(bufB, k2)          // unconditional; clamped past the end
        FLEET_REDUCE_GROUP(bufA, k)
        if (k2 >= cnt) break;
        const int k3 = k2 + stride;
        FLEET_ISSUE_GROUP(bufA, k3)
        FLEET_REDUCE_GROUP(bufB, k2)
        if (k3 >= cnt) break;
        k = k3;
    }
"""

PIPELINE_NO_TERMINAL = """    FLEET_ISSUE_GROUP(bufA, k)
    // Complete pairs retain unconditional issue-before-reduce in each basic
    // block. Peel the one/two-group tail, never predicate individual loads.
    while (k + 2 * stride < cnt) {
        const int k2 = k + stride;
        FLEET_ISSUE_GROUP(bufB, k2)
        FLEET_REDUCE_GROUP(bufA, k)
        const int k3 = k2 + stride;
        FLEET_ISSUE_GROUP(bufA, k3)
        FLEET_REDUCE_GROUP(bufB, k2)
        k = k3;
    }
    if (k + stride < cnt) {
        const int k2 = k + stride;
        FLEET_ISSUE_GROUP(bufB, k2)
        FLEET_REDUCE_GROUP(bufA, k)
        FLEET_REDUCE_GROUP(bufB, k2)
    } else {
        FLEET_REDUCE_GROUP(bufA, k)
    }
"""


def apply(src: Path) -> None:
    name = "gemv_no_terminal"
    gemv = src / "kernels/gemv.h"
    kernel = src / "kernels/fleet_kernel.hip"
    g, k = gemv.read_text(), kernel.read_text()
    # Keep the original helper/dispatch byte-for-byte: router, expert_down,
    # merge, o_proj, dense_down and lm_head still bind to those originals.
    piped = _function(g, "template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_pipelined(")
    new_piped = _once(piped, "void gemv_rows_pipelined(", "void gemv_rows_pipelined_no_terminal(", name)
    new_piped = _once(new_piped, PIPELINE_OLD, PIPELINE_NO_TERMINAL, name)
    g = _once(g, piped, piped + "\n\n" + new_piped, name)
    dispatch = _function(g, "__device__ inline void gemv_rows(")
    new_dispatch = _once(dispatch, "void gemv_rows(", "void gemv_rows_no_terminal(", name)
    if new_dispatch.count("gemv_rows_pipelined<") != 3:
        raise SystemExit(f"patch {name}: expected three full-wave dispatch paths")
    new_dispatch = new_dispatch.replace("gemv_rows_pipelined<", "gemv_rows_pipelined_no_terminal<")
    g = _once(g, dispatch, dispatch + "\n\n" + new_dispatch, name)
    qkv = _function(k, "        case TASK_QKV_FUSED: {\n            // layer input:")
    if qkv.count("gemv_rows(") != 3:
        raise SystemExit(f"patch {name}: expected q plus both kv_a branches")
    k = _once(k, qkv, qkv.replace("gemv_rows(", "gemv_rows_no_terminal("), name)
    gemv.write_text(g)
    kernel.write_text(k)
