"""CPU-only model.json loading. Paths are relative to the contract, never the cwd.

The returned dictionary is a resolved contract, including config_sha256 and arch.
Config-derived arch entries may be omitted; redundant entries must agree exactly.
Attention is the one explicitly overridable inference in the model contract.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re



class ContractError(ValueError):
    """An invalid model contract or config."""


def _validate(value, spec, path):
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    typ = spec.get("type")
    if typ and (not isinstance(value, types[typ]) or
                (typ in ("integer", "number") and isinstance(value, bool))):
        raise ContractError(f"{path}: expected {typ}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path}: expected finite number")
    if "enum" in spec and value not in spec["enum"]:
        raise ContractError(f"{path}: expected one of {spec['enum']}")
    if isinstance(value, dict):
        for key in spec.get("required", []):
            if key not in value:
                raise ContractError(f"{path}.{key}: required")
        props = spec.get("properties", {})
        for key, item in value.items():
            if key in props:
                _validate(item, props[key], f"{path}.{key}")
            elif spec.get("additionalProperties") is False:
                raise ContractError(f"{path}.{key}: unknown field")
            elif isinstance(spec.get("additionalProperties"), dict):
                _validate(item, spec["additionalProperties"], f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            raise ContractError(f"{path}: too few entries")
        if spec.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise ContractError(f"{path}: duplicate entries")
        for i, item in enumerate(value):
            _validate(item, spec.get("items", {}), f"{path}[{i}]")
    if isinstance(value, str):
        if len(value) < spec.get("minLength", 0):
            raise ContractError(f"{path}: empty string")
        if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
            raise ContractError(f"{path}: invalid value {value!r}")
    if type(value) in (int, float):
        if value < spec.get("minimum", -math.inf) or value > spec.get("maximum", math.inf):
            raise ContractError(f"{path}: out of range")


def validate_contract(data: dict, *, base_dir: str | Path = ".") -> dict:
    """Validate and resolve a JSON object without modifying the caller's data."""
    schema = json.loads(Path(__file__).with_name("schema.json").read_text())
    _validate(data, schema, "model")
    result = deepcopy(data)
    base = Path(base_dir).resolve()
    config = (base / data["config"]).resolve()
    try:
        raw = config.read_bytes()
        c = json.loads(raw)
        if not isinstance(c, dict):
            raise ValueError("config must be an object")
        text = c.get("text_config", c)
        if not isinstance(text, dict):
            raise ValueError("text_config must be an object")
        extended = (text.get("model_type") in ("glm5_next", "glm5_next_text")
                    or text.get("first_k_dense_replace", 0) or text.get("mlp_only_layers")
                    or text.get("decoder_sparse_step", 1) != 1
                    or text.get("n_shared_experts", 0)
                    or text.get("shared_expert_intermediate_size", 0))
        # Contracts describe model metadata, independently of example graph
        # dimensions and ETX_LLM_* experiment overrides.
        if not extended:
            if c.get("model_type") == "olmo2":
                raise ValueError("olmo2 full-width q/k norms are not supported")
            for key in ("hidden_size", "num_attention_heads", "vocab_size"):
                if type(c[key]) is not int or c[key] <= 0:
                    raise ValueError(f"{key}: expected positive integer")
            expert_keys = ("num_experts", "num_local_experts", "n_routed_experts")
            experts = next((int(c[k]) for k in expert_keys if c.get(k)), 0)
            if not experts and any("expert" in k for k in c):
                raise ValueError("config mentions experts but no supported expert count is set")
            if experts and not 1 <= int(c.get("num_experts_per_tok", 0)) <= min(experts, 16):
                raise ValueError("num_experts_per_tok outside supported Top-K range")
            float(c.get("rms_norm_eps", 1e-6))
            float((c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta", 10000.0)))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        raise ContractError(f"config {config}: {exc}") from exc
    c = text
    layers = c.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        raise ContractError("config.num_hidden_layers: expected positive integer")
    experts = next((c[k] for k in ("num_experts", "num_local_experts", "n_routed_experts")
                    if c.get(k)), 0)
    moe = bool(experts)
    pattern = ["moe" if moe else "dense"] * layers
    if "mlp_layer_types" in c:
        pattern = [{"sparse": "moe", "dense": "dense"}.get(k, k)
                   for k in c["mlp_layer_types"]]
    elif moe:
        first = c.get("first_k_dense_replace", 0)
        step = c.get("decoder_sparse_step", 1)
        if type(first) is not int or not 0 <= first <= layers or type(step) is not int or step < 1:
            raise ContractError("config: invalid dense/sparse layer pattern")
        dense = c.get("mlp_only_layers", [])
        pattern = ["dense" if n < first or n in dense or (n + 1) % step else "moe"
                   for n in range(layers)]
    if len(pattern) != layers or any(k not in ("dense", "moe") for k in pattern):
        raise ContractError("config: invalid layer pattern")
    rope = "neox"
    for key in ("rope_scaling", "rope_parameters"):
        params = c.get(key) or {}
        kind = params.get("rope_type") or params.get("type") or "default"
        if kind not in ("default", "longrope"):
            raise ContractError(f"config: unsupported rope type {kind}")
        if params:
            rope = "longrope" if kind == "longrope" else "neox"
    if c.get("qk_rope_head_dim") == 0:
        rope = "none"
    attention = "mla" if c.get("kv_lora_rank") is not None else "gqa"
    if c.get("layer_types") and all(k == "linear_attention" for k in c["layer_types"]):
        attention = "kda"
    inferred = dict(attention=attention, rope=rope,
                    norm="layernorm" if c.get("model_type") == "phimoe" else "rmsnorm",
                    ffn="moe" if moe else "dense", tie_embeddings=bool(c.get("tie_word_embeddings", False)),
                    layer_pattern=pattern)
    if moe:
        topk = c.get("num_experts_per_tok", 0)
        shared = c.get("n_shared_experts", 0)
        if c.get("shared_expert_intermediate_size") and not shared:
            shared = 1
        if (type(experts) is not int or type(topk) is not int or not 1 <= topk <= experts
                or type(shared) is not int or shared < 0):
            raise ContractError("config: invalid expert counts")
        inferred.update(experts=experts, topk=topk, shared=shared,
                        norm_topk=bool(c.get("norm_topk_prob", c.get("model_type") == "mixtral")))
    supplied = result.get("arch", {})
    for key, value in supplied.items():
        if key == "attention":
            continue
        if key not in inferred or type(value) is not type(inferred[key]) or value != inferred[key]:
            raise ContractError(f"arch.{key}: conflicts with config (inferred {inferred.get(key)!r})")
    inferred.update(supplied)
    result["arch"] = inferred
    result["config"] = str(config)
    result["config_sha256"] = hashlib.sha256(raw).hexdigest()
    result["bindings"].setdefault("batch_sizes", [1, 2, 4, 8])
    result["correctness"].setdefault("tolerance", {"cos": 0.999, "rel": 0.02})
    result["correctness"]["golden"] = str((base / result["correctness"]["golden"]).resolve())
    names = [entry["etx_name"] for entry in result["weights"]["map"]]
    if len(set(names)) != len(names):
        raise ContractError("weights.map: duplicate etx_name")
    return result


def load_contract(path: str | Path) -> dict:
    """Load model.json and record the SHA256 of the exact HF config bytes."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ContractError(f"{path}: {exc}") from exc
    return validate_contract(data, base_dir=path.resolve().parent)
