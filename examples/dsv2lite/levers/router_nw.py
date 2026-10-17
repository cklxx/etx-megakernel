"""Lever router_nw: a copy of fleet's task_norm_router whose router rows split over `nw` workers, not kWorkersPerXCD.

ETX_DSV2_ROUTER_SPREAD=1 (graph.py build_router_spread, fleet_xb.hip fleet_norm_router_spread) runs each request's
norm + router on 18 workers of its slot, so the two requests of a slot run their routers at the same time. fleet's
task_norm_router hard-codes the row split `worker, kWorkersPerXCD` in its router gemv_rows call. This lever appends
task_norm_router_nw: the text of task_norm_router AS THE EARLIER LEVERS LEFT IT (apply it last), with one extra
parameter `nw` and that one argument changed. Every original helper stays byte for byte, so a build without the
switch is unchanged.

Bitwise: gemv_rows picks its GEMV path by K only (K = hidden = 2048: gemv_rows_pipelined<2, 4>); a router row is one
worker's fma8 over its 4 chunks per lane and one wave_sum, the same for any nw. Only the worker that owns a row changes.
The staging, the fold of the 8 o_proj partials, the RMSNorm and worker 0's x_norm publication do not depend on nw.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _function, _once

NAME = "router_nw"
SIGNATURE = "__device__ void task_norm_router("
OLD_TAIL = "const TaskDescriptor& t, const RuntimeState& rt) {"
NEW_TAIL = "const TaskDescriptor& t, const RuntimeState& rt, int nw) {"
OLD_SPLIT = "1.f, d.n_routed, d.hidden, 0, 1, worker, kWorkersPerXCD);"
NEW_SPLIT = "1.f, d.n_routed, d.hidden, 0, 1, worker, nw);"


def apply(src: Path) -> None:
    kernel = src / "kernels/fleet_kernel.hip"
    k = kernel.read_text()
    if "void task_norm_router_nw(" in k:
        raise SystemExit(f"patch {NAME}: already applied")
    original = _function(k, SIGNATURE)
    new = _once(original, "void task_norm_router(", "void task_norm_router_nw(", NAME)
    new = _once(new, OLD_TAIL, NEW_TAIL, NAME)
    new = _once(new, OLD_SPLIT, NEW_SPLIT, NAME)
    # fleet's top-k publication (FLAG_TOPK_PUBLISH) counts kWorkersPerXCD arrivals and declares a __shared__ flag. The
    # ETX router tiles never set the flag (the expert tasks select their own top-k), so the copy ends before it: no new LDS.
    cut = "    // The softmax + 6-pass top-k is 36 dependent shuffles"
    if new.count(cut) != 1 or "FLAG_TOPK_PUBLISH" not in new[new.index(cut):]:
        raise SystemExit(f"patch {NAME}: the top-k publication block is not the pinned text")
    new = new[:new.index(cut)].rstrip() + "\n    (void)tc; (void)t; (void)rt;   // used by the dropped publication only\n}"
    k = _once(k, original, original + "\n\n// etx lever router_nw: task_norm_router with the router rows split over nw workers\n" + new, NAME)
    kernel.write_text(k)
