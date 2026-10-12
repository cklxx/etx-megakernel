"""Variant declarations (docs/PLAN-generic-decoder-2026-10-08.md section 6.1, section 7 item 1).

Each tile variant declares, as data, which phase it can run, on which machines, for which shapes, and what it costs
in VGPRs, LDS and workgroup width. The `shape` predicate holds the rule that examples/vllm_llm/host.hip states today
as a `die()`; the line is cited beside each rule. The interface contract is docs/SEC7-interface-fix.md.

This module does not read the environment. It imports no tile generator (etx.ops.gemv reads ETX_* at import time),
so the resource numbers below are the generators' constants for the default build, cross-checked by
tests/test_select_planner.py.
"""
from __future__ import annotations

import json
import os

# moe_up rows per task. Must equal host.hip ETX_VL_MOE_RPT (default 32).
MOE_UP_RPT = int(os.environ.get("ETX_VL_MOE_RPT", "32"))
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from etx.importer.spec import OPERATORS


@dataclass(frozen=True)
class Budget:
    """One budget level of section 6.2."""
    threads: int      # workgroup size: 1024 or 512
    vgpr: int         # VGPR limit per lane: 128 at 1024 threads, 256 at 512 threads
    lds: int          # LDS bytes per workgroup: 65536 on gfx942


@dataclass(frozen=True)
class Resources:
    vgpr: int
    lds: int
    threads: int = 0  # the workgroup size that the variant needs; 0 = any size


Shape = Mapping[str, int]


@dataclass(frozen=True)
class Variant:
    export: str                                   # tile symbol, for example "dsl_gemv"
    phase: str                                    # phase key, for example "qkv" or "qkv@0"
    kind: str                                     # "dsl" | "imported" | "generic"
    backend: tuple[str, ...]                      # machine names, for example ("gfx942",)
    dtype: str                                    # "bf16"
    shape: Callable[[Shape], bool]                # applicability predicate on the phase shape
    resources: Resources
    cost: Callable[[Shape, Budget], float]        # body time estimate, us per phase instance
    params: Mapping[str, tuple] = field(default_factory=dict)   # free parameters and their legal values
    rule: str = ""                                # the host.hip rule that `shape` replaces, as text

    def __post_init__(self):
        if self.kind not in ("dsl", "imported", "generic"):
            raise ValueError(f"{self.export}: kind must be 'dsl', 'imported' or 'generic', not {self.kind!r}")

    def legal(self, machine: str, shape: Shape, budget: Budget) -> bool:
        """Rule 1 of docs/SEC7-interface-fix.md section 5."""
        r = self.resources
        return (machine in self.backend and bool(self.shape(shape)) and r.vgpr <= budget.vgpr
                and r.lds <= budget.lds and r.threads in (0, budget.threads))


# ---- registry -----------------------------------------------------------------------------------------------------

REGISTRY: list[Variant] = []


def register(v: Variant) -> Variant:
    if any(x.export == v.export and x.phase == v.phase for x in REGISTRY):
        raise ValueError(f"variant {v.export} for phase {v.phase} is already registered")
    REGISTRY.append(v)
    return v


def variants_for(phase: str, variants: list[Variant] | None = None) -> list[Variant]:
    """The variants of one phase, in declaration order."""
    return [v for v in (REGISTRY if variants is None else variants) if v.phase == phase]


# ---- model config -> phases ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Structure:
    """The fusion choices that change the SET of phases. The planner does not choose them (yet): a fold deletes a
    producer phase, and section 6.2 selects one variant per phase of a fixed set. `default_structure` derives them
    the way examples/vllm_llm/model.py does in its default build (ETX_VL_DSL_FOLD=1, derived top-k fold and
    attention-prep choice)."""
    fold_norms: bool   # in_norm / post_norm / final_norm folded into qkv / router / lm_head (dsl_gemv_fold)
    fold_topk: bool    # the top-k recomputed inside the expert tiles (no topk phase)
    attn_head: bool    # dsl_attnprep replaces q_norm, k_norm, rope and cache


THREADS = 1024     # model.py THREADS: the megakernel workgroup
CU = 256           # model.py ETX_VL_CU default: the dense phase width (wvSplitK y-tile rule, fold model)
NX = 8             # XCDs on MI300X


def load_cfg(model_cfg):
    """`examples.llm.model.Dims`, a path to a config.json, or the config dict itself -> Dims."""
    from examples.llm.model import Dims, load_dims
    if isinstance(model_cfg, Dims):
        return model_cfg
    if isinstance(model_cfg, (str, Path)):
        return load_dims(str(model_cfg))
    if isinstance(model_cfg, Mapping):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "config.json")
            Path(p).write_text(json.dumps(dict(model_cfg)))
            return load_dims(p)
    raise TypeError(f"model_cfg: expected Dims, a path or a dict, not {type(model_cfg).__name__}")


def default_structure(d) -> Structure:
    # host.hip:394: dsl_gemv_fold is written for the MoE model (E > 0). model.py folds for a dense model too, and
    # host.hip then stops; here a dense model keeps its norms. PhiMoE (ln_norm) has no imported LayerNorm: fold.
    fold_norms = bool(d.moe) or bool(getattr(d, "ln_norm", False))
    fold_topk = False
    if d.moe and fold_norms and not getattr(d, "sparsemixer", False):
        from examples.vllm_llm import folds          # pure shape model, reads no environment
        fold_topk = bool(folds.decide(d, CU, NX).get("topk", False))
    attn_head = bool(d.qk_norm) and d.G + 2 <= THREADS // 64      # model.py _attn_head
    return Structure(fold_norms=fold_norms, fold_topk=fold_topk, attn_head=attn_head)


def phase_shapes(model_cfg, structure: Structure | None = None) -> dict[str, dict[str, int]]:
    """Phase key -> shape, in launch order, as examples/vllm_llm/model.py instances() lays out one decode step.

    A key `<role>@0` exists only when layer 0 of the role needs a different variant than the other layers.
    A shape has the Dims fields as integers, `M` / `K` for a GEMV phase, `fold` (the phase's tile takes the folded
    prologue) and `first_only` (the phase runs at layer 0 only)."""
    d = load_cfg(model_cfg)
    st = structure or default_structure(d)
    ln = bool(getattr(d, "ln_norm", False))
    base = dict(H=d.H, NH=d.NH, NKV=d.NKV, HD=d.HD, G=d.G, E=d.E, TOPK=d.TOPK, MI=d.MI, INTER=d.INTER, V=d.V,
                qk_norm=int(d.qk_norm), moe=int(d.moe), bias=int(d.bias), sparsemixer=int(getattr(d, "sparsemixer", False)),
                ln_norm=int(ln), fold=0, fold_topk=int(st.fold_topk), first_only=0)
    out: dict[str, dict[str, int]] = {}

    def add(key, **kw):
        out[key] = {**base, **{k: int(v) for k, v in kw.items()}}

    fn = st.fold_norms
    if not fn:
        add("in_norm@0", first_only=1)
        add("in_norm")
        add("qkv", M=d.NQKV, K=d.H)
    elif not ln:
        add("in_norm", first_only=1)
        add("qkv@0", M=d.NQKV, K=d.H)
        add("qkv", M=d.NQKV, K=d.H, fold=1)
    else:
        add("qkv", M=d.NQKV, K=d.H, fold=1)
    if st.attn_head:
        add("prep")
    else:
        if d.qk_norm:
            add("q_norm")
            add("k_norm")
        add("rope")
        add("cache")
    add("attn")
    add("reduce")
    add("o_proj", M=d.H, K=d.NH * d.HD)
    if not fn:
        add("post_norm")
    if d.moe:
        add("router", M=d.E, K=d.H, fold=fn)
        if not (fn and st.fold_topk):
            add("topk")
        add("moe_up", M=d.MI, K=d.H, fold=fn)
        add("moe_down", M=d.H, K=d.MI, fold=fn)
        if not fn:
            add("moe_sum")
    else:
        add("gate_up", M=2 * d.INTER, K=d.H)
        add("act")
        add("down", M=d.H, K=d.INTER)
    if not fn:
        add("final_norm")
    add("lm_head", M=d.V, K=d.H, fold=fn)
    return out


# ---- cost (placeholder) -------------------------------------------------------------------------------------------
# PLAN 6.1: "an estimate of body time for a given occupancy, a function of bytes and waves". Until a GPU measures the
# per-variant rates, every variant of a phase gets the SAME estimate: the phase floor plus its weight bytes at the
# copy ceiling. Equal costs make the selection fall to the tie rule (dsl first, then declaration order).
PHASE_FLOOR_US = 2.64          # examples/vllm_llm/folds.py
COPY_B_PER_US = 4.3e6          # 4.3 TB/s, the MI300X ceiling the PLAN's utilisation figures use


def floor_cost(s: Shape, b: Budget) -> float:
    return PHASE_FLOOR_US


def gemv_cost(s: Shape, b: Budget) -> float:
    return PHASE_FLOOR_US + s["M"] * s["K"] * 2 / COPY_B_PER_US


def expert_up_cost(s: Shape, b: Budget) -> float:      # gate and up rows: 2 x MI x H per selected expert
    return PHASE_FLOOR_US + s["TOPK"] * 2 * s["M"] * s["K"] * 2 / COPY_B_PER_US


def expert_down_cost(s: Shape, b: Budget) -> float:
    return PHASE_FLOOR_US + s["TOPK"] * s["M"] * s["K"] * 2 / COPY_B_PER_US


# ---- declarations -------------------------------------------------------------------------------------------------
GFX942 = ("gfx942",)
DSL_VGPR = 128                                   # __launch_bounds__(1024): 512 registers / 4 waves per SIMD
GEMV_LDS = 4 * (4096 + 2 * 64 + 16 * 64 + 32)    # etx/ops/gemv.py LDS_BYTES at NB = 1: 21120
DSL_RES = Resources(vgpr=DSL_VGPR, lds=0, threads=THREADS)
GEMV_RES = Resources(vgpr=DSL_VGPR, lds=GEMV_LDS, threads=THREADS)
GEMV_PHASES = ("qkv@0", "qkv", "o_proj", "router", "gate_up", "down", "lm_head")
FOLD_PHASES = ("qkv", "router", "lm_head")
BIGK_MAX = 65536                                 # host.hip:463-464 with ETX_EMIT_DSL_BIGK (model.py derives it)


def _dsl(export, phase, shape, rule, res=DSL_RES, cost=floor_cost, params=None):
    register(Variant(export, phase, "dsl", GFX942, "bf16", shape, res, cost, params or {}, rule))


def _gen(export, phase, shape, rule, cost=floor_cost):
    # Reserve the full LDS budget conservatively. The LlmParams ABI is not yet
    # supported by the vLLM host; this declaration is for selection only.
    register(Variant(export, phase, "generic", GFX942, "bf16", shape,
                     Resources(vgpr=DSL_VGPR, lds=65536, threads=THREADS), cost, {}, rule))


# Imported LDS that the CPU registry knows: build.sh records that wvsplitk_y1/y2 declare 32768 B each (round 16).
IMPORTED_LDS = {"wvsplitk_y1": 32768, "wvsplitk_y2": 32768, "wvsplitk_l40_y2": 40960}


def _imp(export, phase, shape, rule, cost=floor_cost):
    op = OPERATORS[export]
    lo = hi = 1
    for a, b in op.block_rule:
        lo *= int(a) if a.isdigit() else 1
        hi *= int(b) if b.isdigit() else 1
    # The megakernel hosts an imported block inside its workgroup: a fixed 1024-thread block needs a 1024 budget.
    threads = THREADS if lo == hi == THREADS else 0
    # VGPR: the megakernel's __launch_bounds__ caps every linked kernel at 128; the true count needs the ISA.
    # LDS: imports.json carries lds_bytes at build time; the CPU registry has IMPORTED_LDS only, else 0.
    register(Variant(export, phase, "imported", GFX942, "bf16", shape,
                     Resources(vgpr=DSL_VGPR, lds=IMPORTED_LDS.get(export, 0), threads=threads),
                     cost, {}, rule))


def _gemv_ok(s):          # host.hip:358
    return s["K"] % 256 == 0 and s["K"] <= 4096


for _ph in GEMV_PHASES:
    _qkv = _ph.startswith("qkv")
    _dsl("dsl_gemv", _ph,
         (lambda s, q=_qkv: s["fold"] == 0 and _gemv_ok(s) and not (q and s["bias"])),
         "host.hip:358 K % 256 == 0 and K <= 4096; host.hip:361 no unfolded qkv bias",
         GEMV_RES, gemv_cost, {"tasks": (CU,), "rows_per_task": ("ceil(M / tasks)",)})
    if _ph in FOLD_PHASES:
        _dsl("dsl_gemv_fold", _ph, lambda s: s["fold"] == 1 and _gemv_ok(s) and s["E"] > 0,
             "host.hip:393 K % 256 == 0 and K <= 4096; host.hip:394 E > 0",
             GEMV_RES, gemv_cost, {"tasks": (CU,), "rows_per_task": ("ceil(M / tasks)",)})
    # model.py wvsplitk_ytile: y = 1 when ceil(M / (CU * 4)) <= 1; the spec allows K <= 16384
    _imp("wvsplitk_y1", _ph, lambda s: s["fold"] == 0 and s["K"] <= 16384 and s["M"] <= CU * 4,
         "spec: K <= 16384; model.py wvsplitk_ytile == 1", gemv_cost)
    _imp("wvsplitk_y2", _ph, lambda s: s["fold"] == 0 and s["K"] <= 16384 and s["M"] > CU * 4 and s["M"] % 2 == 0,
         "spec: K <= 16384; model.py wvsplitk_ytile == 2", gemv_cost)
    _imp("wvsplitk_l40_y2", _ph,
         lambda s: s["fold"] == 0 and 16384 < s["K"] <= 20480 and s["K"] % 8 == 0 and s["M"] > CU * 4 and s["M"] % 2 == 0,
         "SG:357,383,394-395,449-455 K-vector in LDS; SG:1199 K % 8; SG:1217 M % YTILE; lds_cap 40 KB", gemv_cost)

_dsl("dsl_topk", "topk", lambda s: s["E"] <= 128, "host.hip:382 E <= 128 (one wave, 2 logits per lane)")
for _e in (128, 8):
    _imp(f"topk_softmax_e{_e}", "topk", lambda s, e=_e: s["E"] == e and not s["sparsemixer"],
         f"shape-keyed: E == {_e}; softmax only (no sparsemixer)")

_dsl("dsl_moe_up_fold", "moe_up",
     lambda s: s["fold"] == 1 and s["E"] <= 128 and s["MI"] % 256 == 0 and s["H"] % 256 == 0 and max(s["H"], s["MI"]) <= BIGK_MAX,
     "host.hip:431 MI % 32, H % 256, MI % 256, H and MI <= 65536 (DSL_BIGK), E <= 128",
     GEMV_RES, expert_up_cost, {"rows_per_task": (MOE_UP_RPT,), "tasks_per_expert": (f"MI / {MOE_UP_RPT}",)})
_dsl("dsl_moe_down_fold", "moe_down",
     lambda s: s["fold"] == 1 and s["E"] <= 128 and s["MI"] % 256 == 0 and s["H"] % 256 == 0 and max(s["H"], s["MI"]) <= BIGK_MAX,
     "host.hip:431 MI % 32, H % 256, MI % 256, H and MI <= 65536 (DSL_BIGK), E <= 128",
     GEMV_RES, expert_down_cost, {"rows_per_task": (64,), "tasks_per_expert": ("H / 64",)})
_dsl("dsl_moe_up", "moe_up", lambda s: s["fold"] == 0 and s["MI"] % MOE_UP_RPT == 0 and s["H"] % 256 == 0 and s["H"] <= BIGK_MAX,
     f"host.hip:463 MI % {MOE_UP_RPT} == 0, H % 256 == 0, H <= 65536 (DSL_BIGK)",
     GEMV_RES, expert_up_cost, {"rows_per_task": (MOE_UP_RPT,), "tasks_per_expert": (f"MI / {MOE_UP_RPT}",)})
_dsl("dsl_moe_down", "moe_down", lambda s: s["fold"] == 0 and s["H"] % 64 == 0 and s["MI"] % 256 == 0 and s["MI"] <= BIGK_MAX,
     "host.hip:464 H % 64 == 0, MI % 256 == 0, MI <= 65536 (DSL_BIGK)",
     GEMV_RES, expert_down_cost, {"rows_per_task": (64,), "tasks_per_expert": ("H / 64",)})
for _exp, _ph in (("fused_moe_up", "moe_up"), ("fused_moe_down", "moe_down"),
                  ("fused_moe_up_e8", "moe_up"), ("fused_moe_down_e8", "moe_down")):
    _key = dict(H=2048, MI=768, E=128, TOPK=8) if not _exp.endswith("_e8") else dict(H=4096, MI=14336, E=8, TOPK=2)
    _imp(_exp, _ph, lambda s, k=_key: s["fold"] == 0 and all(s[n] == v for n, v in k.items()),
         f"shape-keyed Triton launch: {_key}; needs the moe_act phase (silu_and_mul)",
         expert_up_cost if _ph == "moe_up" else expert_down_cost)
for _k in (8, 2):
    _imp(f"moe_sum_k{_k}", "moe_sum", lambda s, k=_k: s["TOPK"] == k, f"shape-keyed: TOPK == {_k}")

_dsl("dsl_attnprep", "prep",
     lambda s: s["HD"] == 128 and s["NH"] % s["NKV"] == 0 and s["G"] + 2 <= THREADS // 64 and s["qk_norm"],
     "host.hip:378 HD 128, NH % NKV == 0, G + 2 <= waves per workgroup, q/k norm")
for _g in range(1, 17):
    _exp = f"attn_mfma4_g{_g}" if _g <= 4 else f"attn_mfma16_g{_g}"
    _imp(_exp, "attn", lambda s, g=_g: s["HD"] == 128 and s["G"] == g,
         "AT:3295-3346 G 1..16; AT:3620-3629 HEAD_SIZE 64/128 (128 here)")
_gen("llm_attn", "attn", lambda s: s["HD"] % 8 == 0 and s["HD"] <= 192 and s["G"] <= 16,
     "examples/llm/tiles.hip:138 HD <= 192; :199 dot8 requires HD % 8 == 0; :203 G <= 16")
_imp("attn_reduce", "reduce", lambda s: True, "")
_imp("rms_norm_2d", "in_norm@0", lambda s: not s["ln_norm"], "RMSNorm only (no imported LayerNorm, model.py:277)")
_imp("rms_norm_2d", "in_norm", lambda s: s["first_only"] == 1 and not s["ln_norm"], "layer 0 only: no residual add")
_imp("fused_add_rms_norm", "in_norm", lambda s: s["first_only"] == 0 and not s["ln_norm"], "RMSNorm only")
for _ph in ("post_norm", "final_norm"):
    _imp("fused_add_rms_norm", _ph, lambda s: not s["ln_norm"], "RMSNorm only")
_imp("rms_norm_3d", "q_norm", lambda s: True, "")
_imp("rms_norm_3d", "k_norm", lambda s: True, "")
_imp("rope_neox", "rope", lambda s: True, "")
_imp("reshape_and_cache", "cache", lambda s: True, "")
_imp("silu_and_mul", "act", lambda s: True, "")
