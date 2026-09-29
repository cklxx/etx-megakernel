"""Qwen3-30B-A3B with vLLM's kernels (examples/vllm_llm/model.py): the MoE layer is vLLM's launch sequence minus the fill
and the copy, and with ETX_VL_XCD the expert path (gate_up, silu, down) runs one selected expert per XCD with
XCD-local hand-offs only; top-k and moe_sum are the only cross-XCD waits of the MoE half. Uses a synthetic
imports.json (slot counts and argument sizes as the real import reports them)."""
import importlib
import json
from pathlib import Path

from etx.ir import Scope
from etx.passes import compile_graph, verify_plan

ROOT = Path(__file__).resolve().parents[1]
# export -> (slots, arg bytes): etx.importer.vllm_kernels output for Qwen3-30B-A3B (MoE slots: fused_moe / silu = 1)
IMPORTS = {"wvsplitk_y2": (1, 80), "wvsplitk_y1": (1, 80), "attn_mfma16_g8": (4, 160), "attn_reduce": (8, 80),
           "rms_norm_2d": (4, 100), "rms_norm_3d": (16, 100), "fused_add_rms_norm": (1, 72), "rope_neox": (2, 108),
           "reshape_and_cache": (16, 96), "silu_and_mul": (1, 48), "topk_softmax_e128": (4, 104), "moe_sum_k8": (4, 96),
           "fused_moe_up": (1, 136), "fused_moe_down": (1, 144), "qknorm_rope": (4, 80), "dsl_attnprep": (1, 112)}


def _graph(tmp_path, monkeypatch, xcd: bool, layers: int = 3, qknr: str = "0", attn_head: bool = False):
    imp = tmp_path / "imports.json"
    imports = {e: {"slots": s, "arg_bytes": a} for e, (s, a) in IMPORTS.items()}
    if attn_head:
        imports["attn_mfma16_g8"]["slots"] = 1                        # ETX_VL_SLOTS=attn_mfma16_g8=1
    imp.write_text(json.dumps(imports))
    monkeypatch.setenv("ETX_VL_DSL_ATTN", "head" if attn_head else "0")
    monkeypatch.setenv("ETX_VL_IMPORTS", str(imp))
    monkeypatch.setenv("ETX_LLM_CONFIG", str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))
    monkeypatch.setenv("ETX_LLM_LAYERS", str(layers))
    monkeypatch.setenv("ETX_VL_XCD", "1" if xcd else "0")
    monkeypatch.setenv("ETX_VL_CHAIN", "1" if xcd else "0")
    monkeypatch.setenv("ETX_VL_PLAN", str(tmp_path / "vl_plan.txt"))
    monkeypatch.setenv("ETX_VL_CHAINS", str(tmp_path / "chain_tiles.hip"))
    monkeypatch.setenv("ETX_VL_QKNR", qknr)
    import examples.vllm_llm.model as vm
    vm = importlib.reload(vm)
    return vm, vm.build()


def test_moe_layer_is_vllms_sequence(tmp_path, monkeypatch):
    vm, g = _graph(tmp_path, monkeypatch, xcd=False)
    rows = [ln.split() for ln in (tmp_path / "vl_plan.txt").read_text().splitlines()[2:]]
    layer1 = [(r[1], r[3], int(r[4])) for r in rows if r[2] == "1"]
    assert [x[1] for x in layer1] == ["in_norm", "qkv", "q_norm", "k_norm", "rope", "cache", "attn", "reduce", "o_proj",
                                      "post_norm", "router", "topk", "moe_up", "moe_act", "moe_down", "moe_sum"]
    grid = {role: (exp, gx) for exp, role, gx in layer1}
    # vLLM's launch shapes (rocprofv3 trace, results_2026-09-27/vlnext/moe): router 304 WGs, top-k 1, gate_up 192,
    # silu 8, down 256, moe_sum 1
    assert grid["router"] == ("wvsplitk_y1", 304) and grid["topk"] == ("topk_softmax_e128", 1)
    assert grid["moe_up"] == ("fused_moe_up", 192) and grid["moe_act"] == ("silu_and_mul", 8)
    assert grid["moe_down"] == ("fused_moe_down", 256) and grid["moe_sum"] == ("moe_sum_k8", 1)
    plan = compile_graph(g, "gfx942", {}, {})
    assert verify_plan(plan) == []


def test_moe_xcd_expert_path_is_xcd_local(tmp_path, monkeypatch):
    vm, g = _graph(tmp_path, monkeypatch, xcd=True)
    plan = compile_graph(g, "gfx942", {}, {})
    assert verify_plan(plan) == []
    shapes = {n: tuple(plan.inst.grid_shapes[n]) for n in ("topk_1_x", "moe_up_1", "moe_act_1", "moe_down_1", "moe_sum_1_x")}
    assert shapes == {"topk_1_x": (8,), "moe_up_1": (8, 24), "moe_act_1": (8, 1), "moe_down_1": (8, 32), "moe_sum_1_x": (8,)}
    # XCD x runs expert slot x of every launch on the expert path
    for t in plan.tasks:
        if t.grid in ("moe_up_1", "moe_act_1", "moe_down_1", "topk_1_x", "moe_sum_1_x"):
            assert t.domain == t.coord[0], (t.grid, t.coord, t.domain)
    # what each launch of the MoE half waits on: its predecessor's event and that event's scope
    waits = {}
    for n in ("router_1", "topk_1_x", "moe_up_1", "moe_act_1", "moe_down_1", "moe_sum_1_x"):
        (ev,) = g.grid(n).in_edges
        waits[n] = plan.events[ev].scope
    assert waits["moe_up_1"] == waits["moe_act_1"] == waits["moe_down_1"] == waits["router_1"] == Scope.DOMAIN
    assert waits["topk_1_x"] == waits["moe_sum_1_x"] == Scope.DEVICE


def test_xcd_simulation_runs_the_plan_and_obeys_littles_law(tmp_path, monkeypatch):
    """etx.sim.xcd on the compiled MoE xcd plan: every task runs, and streams are limited by bandwidth when the GPU
    is full and by latency when few CUs stream."""
    from etx.codegen.kernel import emit_plan_json
    from etx.sim.xcd import Cost, Hw, _rates, simulate
    vm, g = _graph(tmp_path, monkeypatch, xcd=True, layers=2)
    plan = json.loads(emit_plan_json(compile_graph(g, "gfx942", {}, {})))

    def cost(name, d):
        return Cost(front_us=0.5, chain_us=1.0, stream_bytes=64e3 if name.startswith(("qkv", "moe_", "o_proj", "router")) else 0,
                    inflight=16e3, exposed_us=0.3)
    rows, st = simulate(plan, cost, Hw(bw_max=5e6, lat_us=0.2))
    assert st["unfinished"] == 0 and len(rows) == len(plan["descs"]) and st["makespan_us"] > 0
    full = {i: [1e5, 0, 0, 16e3, 0.0] for i in range(304)}
    assert abs(sum(_rates(full, Hw(bw_max=5e6, lat_us=0.2)).values()) - 5e6) < 1.0      # bandwidth-bound
    few = {i: [1e5, 0, 0, 16e3, 0.0] for i in range(4)}
    assert abs(_rates(few, Hw(bw_max=5e6, lat_us=0.2))[0] - 16e3 / 0.2) < 1e-6            # latency-bound


def test_two_requests_interleave_and_stay_independent(tmp_path, monkeypatch):
    """ETX_VL_REQUESTS=2: every launch twice, the requests alternate in the graph (so in every worker's queue), no event
    crosses requests, and the XCD simulation runs both to the end."""
    from etx.codegen.kernel import emit_plan_json
    from etx.sim.xcd import Cost, Hw, simulate
    monkeypatch.setenv("ETX_VL_REQUESTS", "2")
    vm, g = _graph(tmp_path, monkeypatch, xcd=True, layers=2)
    assert vm.REQUESTS == 2
    names = [gr.name for gr in g.grids]
    assert names[:2] == ["r0_prologue", "r1_prologue"] and names[-2:] == ["r0_argmax", "r1_argmax"]
    body = names[2:-2]
    assert all(a.startswith("r0_") and b.startswith("r1_") and a[3:] == b[3:] for a, b in zip(body[0::2], body[1::2]))
    for gr in g.grids:
        r = gr.name[:3]
        assert all(ev.startswith(r) for ev in list(gr.in_edges) + list(gr.out_edges)), gr.name
    plan = compile_graph(g, "gfx942", {}, {})
    assert verify_plan(plan) == []
    pj = json.loads(emit_plan_json(plan))
    rows, st = simulate(pj, lambda n, d: Cost(front_us=0.5, chain_us=1.0), Hw())
    assert st["unfinished"] == 0
    rows_ = [ln.split() for ln in (tmp_path / "vl_plan.txt").read_text().splitlines()[2:]]
    assert {r[-1] for r in rows_} == {"0", "1"} and rows_[0][-1] == "0" and rows_[1][-1] == "1"


def test_adapter_rounds_and_pre_wait():
    """Kernels with slot barriers run the hardware-barrier round protocol around their block; every tile form has a
    pre-wait function that touches its argument block; kernels without slot barriers keep the early exit."""
    from etx.importer.adapter import emit_adapters
    imp = {"reduce": {"kernel": "k", "block": [128, 1, 1], "slot_threads": 128, "slots": 8, "grid_offset": 64, "lds_bytes": 0,
                      "slot_barrier": True, "barriers": 1},
           "gemv": {"kernel": "g", "block": [64, 16, 1], "slot_threads": 1024, "slots": 1, "grid_offset": 64, "lds_bytes": 0,
                    "slot_barrier": False, "barriers": 1}}
    src = emit_adapters(imp)
    red = src[src.index("void etx_tile_reduce("):src.index("void etx_pre_reduce(")]
    assert "etx_imp_rounds_begin();" in red and "etx_imp_rounds_end();" in red and "return;" not in red
    gem = src[src.index("void etx_tile_gemv("):src.index("void etx_pre_gemv(")]
    assert "rounds" not in gem and "if (slot >= 1) return;" in gem
    for f in ("etx_pre_reduce(", "etx_pre_x_reduce(", "etx_pre_reduce_at(", "etx_pre_gemv("):
        assert f in src
    assert "etx_imp_argblock(c->args, c->cst[0])" in src


def test_qknr_replaces_qk_norm_and_rope(tmp_path, monkeypatch):
    """ETX_VL_QKNR: vLLM's fused_qk_norm_rope (grid ceil((32 + 4) / 8) = 5, block 256) in place of rms_norm_3d x2 +
    rotary; =1 chains it with the cache write (one task), =grid keeps it a grid of its own (its blocks on 2 CUs)."""
    from etx.codegen.kernel import emit_plan_json
    from examples.vllm_llm import kmodel as km
    from examples.vllm_llm.xcd_run import cost_fn
    vm, g = _graph(tmp_path, monkeypatch, xcd=False, qknr="1")
    rows = [ln.split() for ln in (tmp_path / "vl_plan.txt").read_text().splitlines()[2:]]
    layer1 = [(r[1], r[3], int(r[4]), int(r[8])) for r in rows if r[2] == "1"]
    assert [x[1] for x in layer1][:5] == ["in_norm", "qkv", "qknr", "cache", "attn"]
    assert [x for x in layer1 if x[1] == "qknr"] == [("qknorm_rope", "qknr", 5, 2)]
    vm, g = _graph(tmp_path, monkeypatch, xcd=True, layers=2, qknr="1")
    names = json.loads(emit_plan_json(compile_graph(g, "gfx942", {}, {})))["types"]
    assert "chain_qknr_cache_1" in names and not any(n.startswith("qknr_") for n in names)
    p = km.Params()
    cost = cost_fn("qwen3-30b-a3b", p)("chain_qknr_cache_1", {})
    kern = {k.role: k for k in km.MOE}
    want = 2 * km.terms(kern["qknr"], p, "etx_task")["total"] + km.terms(kern["cache"], p, "etx_task")["total"]
    assert abs(cost.front_us + cost.chain_us - want) < 1e-9          # the fused part in 2 rounds of 4 slots, then the cache
    vm, g = _graph(tmp_path, monkeypatch, xcd=True, layers=2, qknr="grid")
    names = json.loads(emit_plan_json(compile_graph(g, "gfx942", {}, {})))["types"]
    assert "qknr_1" in names and "cache_1" in names and not any(n.startswith("chain_") for n in names)


def test_attention_block_by_kv_head(tmp_path, monkeypatch):
    """ETX_VL_DSL_ATTN=head: prep h -> attention (p, h) -> reduce h stay on XCD h (DOMAIN-scope events with per-head
    shape); the qkv -> prep and reduce -> o_proj hand-offs stay device-wide."""
    from etx.codegen.kernel import emit_plan_json
    vm, g = _graph(tmp_path, monkeypatch, xcd=True, layers=2, attn_head=True)
    plan = compile_graph(g, "gfx942", {}, {})
    assert verify_plan(plan) == []
    pj = json.loads(emit_plan_json(plan))
    names = {v["id"]: k for k, v in pj["types"].items()}
    by = {}
    for d in pj["descs"]:
        by.setdefault(names[d["type"]], []).append(d)
    assert len(by["prep_1"]) == 4 and len(by["attn_1"]) == 20 and len(by["reduce_1"]) == 4
    assert [d["domain"] for d in by["prep_1"]] == [0, 1, 2, 3]
    assert all(d["domain"] == d["coord"][0] // 5 for d in by["attn_1"])          # partition p of head h on XCD h
    assert [d["domain"] for d in by["reduce_1"]] == [0, 1, 2, 3]
