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
           "fused_moe_up": (1, 136), "fused_moe_down": (1, 144)}


def _graph(tmp_path, monkeypatch, xcd: bool, layers: int = 3):
    imp = tmp_path / "imports.json"
    imp.write_text(json.dumps({e: {"slots": s, "arg_bytes": a} for e, (s, a) in IMPORTS.items()}))
    monkeypatch.setenv("ETX_VL_IMPORTS", str(imp))
    monkeypatch.setenv("ETX_LLM_CONFIG", str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))
    monkeypatch.setenv("ETX_LLM_LAYERS", str(layers))
    monkeypatch.setenv("ETX_VL_XCD", "1" if xcd else "0")
    monkeypatch.setenv("ETX_VL_CHAIN", "1" if xcd else "0")
    monkeypatch.setenv("ETX_VL_PLAN", str(tmp_path / "vl_plan.txt"))
    monkeypatch.setenv("ETX_VL_CHAINS", str(tmp_path / "chain_tiles.hip"))
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
