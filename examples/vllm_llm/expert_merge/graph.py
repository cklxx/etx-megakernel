"""The complete up -> down -> next consumer window; no model imports needed."""
import os
from etx.frontends import hip_link
from etx.ir import Graph, Resource


def build():
    arm = os.environ["EEM_ARM"]
    if arm not in ("control", "candidate"):
        raise ValueError("unknown expert merge arm")
    g = Graph("qwen_expert_merge_" + arm)
    g.codegen.update(coh=True, nb=1, requests=1)
    # Host-owned reset image. These counters have no generated event arrival.
    for name, size in (("impargs", 480), ("merged", 4096), ("partials", 1024),
                       ("inverse", 4), ("counts", 132), ("staged", 256*2048*2),
                       ("order", 4), ("owners", 128)):
        g.tensor(name, (size,), bytes_per_elem=1, role="scratch")
    args = list(g.tensors)
    g.etensor("up_done", (8,), wait_count=24)
    g.etensor("down_done", (1,), wait_count=256)
    res = Resource(threads=1024, vgpr=128, lds_bytes=8*1024+256)
    src = os.environ["EEM_TILES"]
    def link(symbol, **kw):
        return hip_link(src, symbol, coherence="coherent", arg_bytes=160, **kw)
    # Cost values are placeholders for liveness simulation, not a forecast.
    g.call_device("moe_up", (8, 24), link("etx_eem_up", pre_wait="etx_pre_x_dsl_moe_up_fold"),
                  args=args, resource=res, consts=(0, 0, 24),
                  out_edges={"up_done": "xw->x"}, domain_map="xw->x", duration_us=3.0)
    g.call_device("moe_down", (8, 32), link("etx_eem_down", pre_wait="etx_pre_x_dsl_moe_down_fold"),
                  args=args, resource=res, consts=(160, 0, 32),
                  in_edges={"up_done": "xw->x"}, out_edges={"down_done": "xw->(0)"},
                  domain_map="xw->x", duration_us=3.0)
    g.call_device("consumer", (256,), link("etx_eem_consumer", pre_wait="etx_pre_dsl_gemv_fold",
                                         touch="etx_touch_dsl_gemv_fold"),
                  args=args, resource=res, consts=(320,), in_edges={"down_done": "i->(0)"}, duration_us=3.0)
    return g


def bindings(**kw):
    return kw


def runtime(bindings):
    return {}
