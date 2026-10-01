"""The feature manifest: what emit_kernel emitted is published to the host by the same artefact the same compile
produced, so the two cannot disagree.

Twice this class of bug produced wrong answers with no diagnostic -- a flag the final hipcc got and the emitter did
not (ETX_VL_NB, 2026-10-01 round 14), and a flag build.sh dropped from its array whenever ETX_VL_COH was also set
(dev_reset, round 19: the host skipped its own per-step reset believing in a tail that was never compiled, every
step after the first became a no-op, 0/32 tokens). A conditionally emitted feature the host must know about belongs
in plan.graph.codegen and in the manifest, never in a -D.
"""
import re

from etx.codegen import emit_kernel
from etx.codegen.plan_header import emit_plan_header
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from etx.passes import compile_graph

RES = Resource(threads=256, lds_bytes=0, vgpr=64)
FEATURES = ("argpf", "bar2", "dev_reset")

def body_without_manifest(src: str) -> str:
    """Emission minus the feature-manifest block. The manifest names every macro it governs, including the ones
    that are off, so a test asserting a feature's code is absent must look past it (etx/codegen/manifest.py)."""
    marker = "// ---- feature manifest"
    if marker not in src:
        return src
    head, rest = src.split(marker, 1)
    return head + rest.split("\n\n", 1)[-1]



def _plan(**codegen):
    g = Graph("fm")
    g.tensor("t", (64,), role="activation", bytes_per_elem=2)
    g.etensor("E1", (1,), wait_count=4)
    g.call_device("a", (4,), hip_link("a.hip", "etx_tile_a", arg_bytes=136), resource=RES, args=["t"],
                  consts=(96, 160), out_edges={"E1": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
    g.call_device("b", (4,), hip_link("b.hip", "etx_tile_b"), resource=RES, args=["t"],
                  in_edges={"E1": "i->(0)"}, domain_map="i->(0)", duration_us=1.0)
    g.codegen.update({k: v for k, v in codegen.items() if v})
    return compile_graph(g, "gfx942", {}, {})


def _manifest(header: str) -> dict[str, int]:
    return {f: int(re.search(rf"#define ETX_EMIT_{f.upper()} (\d)", header).group(1)) for f in FEATURES}


def test_manifest_reports_every_feature_both_ways():
    for feat in FEATURES:
        off, on = _plan(), _plan(**{feat: True})
        assert _manifest(emit_plan_header(off))[feat] == 0, feat
        assert _manifest(emit_plan_header(on))[feat] == 1, feat


def test_manifest_matches_what_the_kernel_actually_emitted():
    """The invariant that makes the mismatch impossible: the manifest bit is set exactly when the emitted device
    code carries the feature. Checked through a marker unique to each feature's emission."""
    marker = {"argpf": "etx_argpf_bytes", "bar2": "s_task2", "dev_reset": "p.n_reset_ops"}
    for feat in FEATURES:
        for want in (False, True):
            plan = _plan(**{feat: want})
            assert _manifest(emit_plan_header(plan))[feat] == int(want), (feat, want)
            assert (marker[feat] in emit_kernel(plan)) == want, (feat, want)


def test_dev_reset_tail_is_not_behind_a_compile_define():
    """A -D is exactly what build.sh dropped; the tail must depend on the plan alone."""
    src = emit_kernel(_plan(dev_reset=True))
    assert "p.n_reset_ops" in src
    body = body_without_manifest(src)
    assert "#ifdef ETX_DEV_RESET" not in body and "#ifndef ETX_DEV_RESET" not in body


def test_dev_reset_tail_fences_the_exit_counter():
    """The exit counter and the arrives are relaxed atomics on different addresses, so the counter reaching
    gridDim.x does not by itself order the arrives before the reset: a late arrive would decrement an event word
    the reset had already refreshed. Release before the increment, acquire after the full count."""
    src = emit_kernel(_plan(dev_reset=True))
    tail = re.search(r"  if \(p\.n_reset_ops\) \{.*?\n  \}", src, re.S).group(0)
    inc = tail.index("atomicAdd(p.ctrl_abort + 2, 1)")
    assert "__threadfence();" in tail[:inc], "no release fence before the exit increment"
    assert "__threadfence();" in tail[inc:], "no acquire fence before the reset"
