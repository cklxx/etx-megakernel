"""Scratch bound for ETX_DSV2_ATTN_XB (design only, no tile): B = 4, 8 KV chunks, request r's qabs / attention / merge on
16 workers (r // 2) * 16 .. of its slot, so the two requests of a slot run in one wave. Durations only."""
import os, sys, json
for k in list(os.environ):
    if k.startswith("ETX_DSV2_"): del os.environ[k]
from etx.passes import PassOptions, compile_graph
from etx.sim.protocol import simulate
from examples.dsv2lite import graph as dsv
from examples.dsv2lite.graph import XCDS, HEADS, QABS, DUR, _tile, _res

tag, mode, attn_us, merge_us, qabs_us = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
CH = 8
orig = dsv.build_attention_block


def attn_block(g, q, L, e_in, e_in_map, fold, extra_in=None, xb=False, xbo=False):
    assert xb and xbo and q.paired
    n = q.n
    dsv._layer_tensor(g, L)
    kvs = os.environ.get("ETX_DSV2_KVA", "replicated") == "shared"
    sfx = "_kvs" if kvs else ""
    off = q.off + (q.r // 2) * 2 * CH
    g.etensor(n(f"E_qabs_{L}"), (HEADS,), wait_count=QABS)
    g.etensor(n(f"E_attn_{L}"), (HEADS,), wait_count=CH)
    g.etensor(n(f"E_merge_{L}"), (XCDS,), wait_count=(HEADS // XCDS) * CH)
    ein = "hs->(0)" if kvs else "hs->(h/2)"
    g.call_device(n(f"qabs_{L}"), (HEADS, QABS), _tile("fleet_q_absorb"), resource=_res(paired=True), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hs->(h/2)", worker_map=f"hs->((h%2)*2+s+{off})",
                  in_edges={n(f"E_qkv_{L}"): ein}, out_edges={n(f"E_qabs_{L}"): "hs->h"}, duration_us=qabs_us)
    g.call_device(n(f"attn_{L}"), (HEADS, CH), _tile("fleet_attention" + sfx), resource=_res(paired=True), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hc->(h/2)", worker_map=f"hc->((h%2)*{CH}+c+{off})",
                  in_edges={n(f"E_qkv_{L}"): "hc->(0)" if kvs else "hc->(h/2)"},
                  out_edges={n(f"E_attn_{L}"): "hc->h"}, duration_us=attn_us, duration_cv=0.1)
    g.call_device(n(f"merge_{L}"), (HEADS, CH), _tile("fleet_merge_uv"), resource=_res(paired=True), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hc->(h/2)", worker_map=f"hc->((h%2)*{CH}+c+{off})",
                  in_edges={n(f"E_attn_{L}"): "hc->h"}, out_edges={n(f"E_merge_{L}"): "hc->(h/2)"}, duration_us=merge_us)
    return n(f"E_oproj_{L}")


if mode == "new":
    dsv.build_attention_block = attn_block
os.environ.update(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_KVA="shared", ETX_DSV2_REQUESTS="4",
                  ETX_DSV2_QKV_XB="1", ETX_DSV2_OPROJ_XB="1", ETX_DSV2_LMHEAD_XB="1")
plan = compile_graph(dsv.build(), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))
ms = []
for seed in range(3):
    r = simulate(plan, seed=seed); assert not r.deadlock, r.blocked[:5]; ms.append(r.makespan_us)
r0 = simulate(plan, seed=0, cv_scale=0.0)
print(json.dumps(dict(tag=tag, mode=mode, attn_us=attn_us, merge_us=merge_us, qabs_us=qabs_us, tasks=len(plan.tasks),
                      cv0=round(r0.makespan_us, 1), seeds=[round(m, 1) for m in ms], mean=round(sum(ms) / 3, 1))), flush=True)
