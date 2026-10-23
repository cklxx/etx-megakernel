"""The decode-step byte ledger: derivation against packed manifests, the floor, and every refusal."""
from pathlib import Path

import pytest

from etx.models.vllm_llm import byte_ledger as bl
from examples.llm.model import load_dims

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "examples/llm/configs"

# `bytes` of the vl.manifest prep.py wrote on the VMs ("wrote .../vl.bin (...) and vl.manifest: {... "bytes": N}"):
# results_2026-09-27/vlnext/prep_qwen2.5-1.5b.log, prep_qwen3-8b.log, results_2026-09-28b/prep_qwen3-30b-a3b.log
PACKED_BYTES = {"qwen2.5-1.5b": 3087703040, "qwen3-8b": 16381745152, "qwen3-30b-a3b": 61064519680}


def manifest(model):
    return bl.read_manifest(bl.manifest_from_config(CONFIGS / f"{model}.json"))


def edit(model, drop=(), add=()):
    lines = [ln for ln in bl.manifest_from_config(CONFIGS / f"{model}.json").splitlines()
             if not any(ln.startswith(d) for d in drop)]
    return bl.read_manifest("\n".join(lines + list(add)))


@pytest.mark.parametrize("model", sorted(PACKED_BYTES))
def test_derived_layout_matches_packed_manifests(model):
    cfg, _ = manifest(model)
    assert cfg["bytes"] == PACKED_BYTES[model]


def test_qwen_moe_weights_match_the_dsl_prep_formula():
    """examples/llm/prep.py's independent active_bytes formula (qkv, o, router, top-k experts, lm_head) plus the
    small tensors it leaves out (norms, embedding row, rotary row, final norm) is the ledger's weight total."""
    d = load_dims(str(CONFIGS / "qwen3-30b-a3b.json"))
    r = bl.ledger(*manifest("qwen3-30b-a3b"), batch=1, context=1024)
    attn = (d.NQKV * d.H + d.H * d.NH * d.HD) * 2
    mlp = (d.E * d.H + d.TOPK * 3 * d.MI * d.H) * 2
    active = d.L * (attn + mlp) + d.V * d.H * 2
    small = sum(row["norms"] for row in r["layers"]) + sum(v for k, v in r["globals"].items() if k != "lm_head")
    assert r["weights"] - small == active
    assert round(active / 1e9, 2) == 6.08             # docs/qwen/MODEL-vllm-kernels.md:481


def test_phimoe_session_floor():
    r = bl.ledger(*manifest("phi-3.5-moe"), batch=1, context=1024)
    assert r["active"] == 2
    assert r["layers"][0]["kv_read"] == 1024 * 2 * 8 * 128 * 2
    assert round(r["total"] / 1e9, 4) == 13.1537
    assert round(r["floor_ms"], 3) == 3.059






def refused(cfg, tensors, match, **kw):
    kw.setdefault("batch", 1)
    kw.setdefault("context", 1024)
    with pytest.raises(bl.Refusal, match=match):
        bl.ledger(cfg, tensors, **kw)


def test_refuses_missing_cfg_key():
    refused(*edit("phi-3.5-moe", drop=["cfg TOPK "]), "missing input: cfg TOPK")


def test_refuses_missing_tensor():
    refused(*edit("phi-3.5-moe", drop=["t L5.w13 "]), r"missing input: tensor L5\.w13")


def test_refuses_tied_model_without_lm_head_only_when_untied():
    refused(*edit("qwen3-30b-a3b", drop=["t lm_head "]), "missing input: tensor lm_head")
    bl.ledger(*manifest("qwen2.5-1.5b"), batch=1, context=1024)     # tied: lm_head reads embed


def test_refuses_tensor_without_rule():
    refused(*edit("phi-3.5-moe", add=["t L0.wfoo 0 16"]), r"tensor L0\.wfoo has no byte rule")
    refused(*edit("phi-3.5-moe", add=["t L32.wo 0 16"]), r"tensor L32\.wo has no byte rule")
    refused(*edit("phi-3.5-moe", add=["t router_bias 0 16"]), "tensor router_bias has no byte rule")


def test_refuses_unknown_expert_union():
    refused(*manifest("phi-3.5-moe"), "missing input: active experts per layer at batch 2", batch=2)
    r = bl.ledger(*manifest("phi-3.5-moe"), batch=2, context=1024, active_experts=3)
    assert r["active"] == 3
    refused(*manifest("phi-3.5-moe"), "active experts 5 outside", batch=2, active_experts=5)


def test_refuses_context_beyond_cache():
    refused(*manifest("phi-3.5-moe"), "outside 1..maxlen 1072", context=1073)
    refused(*manifest("phi-3.5-moe"), "outside 1..maxlen 1072", context=0)


def test_refuses_batch_below_one():
    refused(*manifest("phi-3.5-moe"), "batch 0 must be at least 1", batch=0)


def test_refuses_wrong_embed_and_rotary_sizes():
    refused(*edit("phi-3.5-moe", drop=["t embed "], add=["t embed 0 16"]), "tensor embed has 16 elements")
    refused(*edit("phi-3.5-moe", drop=["t cos_sin "], add=["t cos_sin 0 1000"]), "not a multiple of maxlen 1072")


def test_refuses_truncated_layer_count(monkeypatch):
    monkeypatch.setenv("ETX_LLM_LAYERS", "2")
    with pytest.raises(bl.Refusal, match="ETX_LLM_LAYERS is set"):
        bl.manifest_from_config(CONFIGS / "phi-3.5-moe.json")


def test_refuses_expert_tensor_not_divisible():
    refused(*edit("phi-3.5-moe", drop=["t L3.w2 "], add=["t L3.w2 0 17"]), "not a multiple of E")


def test_refuses_malformed_manifest_line():
    with pytest.raises(bl.Refusal, match="not a cfg, cfgf or t line"):
        bl.read_manifest("cfg L 32\nweights.bin\n")


