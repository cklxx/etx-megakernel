"""The expert slot -> XCD mapping (round 35). The XCD-affine expert path is correct exactly when whole top-k
slots land on one XCD (slot s = task / tpe is contiguous, so TOPK % NX == 0); any other TOPK runs the MoE block
unpinned with device-scope events instead of refusing. Qwen3-30B-A3B (TOPK == NX) must plan byte-identically --
that gate is run against the real import tables in CI-by-hand; here the structural facts are pinned."""
import importlib
import json
from pathlib import Path

import pytest

from etx.passes import compile_graph, verify_plan

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = {
    "attn_mfma16_g8": (1, 160), "attn_mfma4_g4": (1, 160), "attn_reduce": (0, 120),  # reduce slots set per shape
    "rms_norm_2d": (4, 64), "fused_add_rms_norm": (4, 64), "rope_neox": (4, 96), "reshape_and_cache": (4, 96),
    "silu_and_mul": (4, 48), "rms_norm_3d": (4, 64), "topk_softmax_e128": (4, 104),
    "fused_moe_up": (1, 160), "fused_moe_down": (1, 160), "moe_sum_k8": (4, 64),
    "wvsplitk_y1": (2, 96), "wvsplitk_y2": (2, 96),
}

MIXTRAL = {"model_type": "mixtral", "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
           "intermediate_size": 14336, "num_hidden_layers": 2, "vocab_size": 32000, "rms_norm_eps": 1e-5,
           "rope_theta": 1e6, "num_local_experts": 8, "num_experts_per_tok": 2}


def _build(tmp_path, monkeypatch, cfg_path: str, attn: str):
    from etx.ops import attnprep as _prep, gemv as _gemv, topk as _topk
    cfg = json.loads(Path(cfg_path).read_text())
    imp = {e: {"slots": s or max(1, cfg["num_attention_heads"] // cfg["num_key_value_heads"]), "arg_bytes": a}
           for e, (s, a) in IMPORTS.items()}
    for mod in (_gemv, _topk, _prep):
        imp.update(json.loads(json.dumps(mod.import_entries())))
    (tmp_path / "imports.json").write_text(json.dumps(imp))
    for k, v in {"ETX_VL_IMPORTS": str(tmp_path / "imports.json"), "ETX_LLM_CONFIG": cfg_path,
                 "ETX_LLM_LAYERS": "2", "ETX_VL_XCD": "1", "ETX_VL_CHAIN": "1",
                 "ETX_VL_PLAN": str(tmp_path / "vl_plan.txt"), "ETX_VL_CHAINS": str(tmp_path / "chains.hip"),
                 "ETX_VL_DSL_GEMV": "1", "ETX_VL_DSL_TOPK": "1", "ETX_VL_DSL_ATTN": attn, "ETX_VL_DSL_FOLD": "1"}.items():
        monkeypatch.setenv(k, v)
    for k in ("ETX_VL_IMPORTED", "ETX_VL_REQUESTS", "ETX_VL_BATCH", "ETX_VL_NB"):
        monkeypatch.delenv(k, raising=False)
    import examples.vllm_llm.model as vm
    vm = importlib.reload(vm)
    return vm, vm.build()


def _rows(tmp_path):
    rows = [ln.split() for ln in (tmp_path / "vl_plan.txt").read_text().splitlines()[2:]]
    return [{"export": r[1], "role": r[3], "tasks": int(r[8]), "rep": int(r[12])} for r in rows]


def test_topk_eq_nx_is_the_identity(tmp_path, monkeypatch):
    """TOPK == NX: the mapping degenerates to slot s on XCD s -- every MoE instance is the (NX, per) affine split."""
    vm, g = _build(tmp_path, monkeypatch, str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"), "head")
    assert verify_plan(compile_graph(g, "gfx942", {}, {})) == []
    moe = [r for r in _rows(tmp_path) if r["role"] in ("moe_up", "moe_down")]
    assert moe and all(r["rep"] == vm.NX for r in moe)
    for r in moe:                                   # whole slots per XCD: per-XCD task count is a multiple of tpe
        tpe = r["tasks"] // vm.load_dims().TOPK
        assert (r["tasks"] // vm.NX) % tpe == 0


def test_topk_multiple_of_nx_keeps_whole_slots_per_xcd(tmp_path, monkeypatch):
    """TOPK = 2 * NX: still affine -- XCD x holds slots [2x, 2x + 2) whole, with no tile change (s = blk / tpe)."""
    cfg = json.loads((ROOT / "examples/llm/configs/qwen3-30b-a3b.json").read_text())
    cfg["num_experts_per_tok"] = 16
    (tmp_path / "top16.json").write_text(json.dumps(cfg))
    vm, g = _build(tmp_path, monkeypatch, str(tmp_path / "top16.json"), "head")
    assert verify_plan(compile_graph(g, "gfx942", {}, {})) == []
    moe = [r for r in _rows(tmp_path) if r["role"] in ("moe_up", "moe_down")]
    assert moe and all(r["rep"] == vm.NX for r in moe)
    for r in moe:
        tpe = r["tasks"] // 16
        assert (r["tasks"] // vm.NX) % tpe == 0     # whole slots per XCD


def test_topk_not_dividing_nx_runs_unpinned(tmp_path, monkeypatch):
    """Mixtral's top-2: the MoE block (and its top-k) plans UNPINNED inside an otherwise XCD-affine build,
    instead of refusing. The host contract holds trivially: one top-k copy, read as copy 0."""
    (tmp_path / "mixtral.json").write_text(json.dumps(MIXTRAL))
    vm, g = _build(tmp_path, monkeypatch, str(tmp_path / "mixtral.json"), "0")
    assert verify_plan(compile_graph(g, "gfx942", {}, {})) == []
    rows = _rows(tmp_path)
    moe = [r for r in rows if r["role"] in ("topk", "moe_up", "moe_down", "moe_act", "moe_sum")]
    assert moe and all(r["rep"] == 1 for r in moe), moe
    dense = [r for r in rows if r["role"] == "qkv" and r["tasks"] == 256]
    assert dense                                     # the rest of the build kept the XCD path


def test_affine_rep_implies_divisibility(tmp_path, monkeypatch):
    """The host addresses per-XCD top-k copies as rr * TOPK and the tile derives slot = blk / tpe, so a replicated
    (rep > 1) MoE instance is only ever planned when TOPK % NX == 0 -- the invariant the NOREP incident (round 27)
    showed must be enforced, not assumed."""
    for cfg_name, attn, topk in (("qwen3-30b-a3b", "head", 8), ("mixtral", "0", 2)):
        if cfg_name == "mixtral":
            (tmp_path / "mixtral.json").write_text(json.dumps(MIXTRAL))
            path = str(tmp_path / "mixtral.json")
        else:
            path = str(ROOT / "examples/llm/configs" / f"{cfg_name}.json")
        vm, _ = _build(tmp_path, monkeypatch, path, attn)
        for r in _rows(tmp_path):
            if r["role"] in ("topk", "moe_up", "moe_down") and r["rep"] > 1:
                assert topk % vm.NX == 0, (cfg_name, r)
