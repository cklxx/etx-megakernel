"""Configuration-driven batch-one decoder graph frontend.

``build(config)`` owns graph construction; example-specific dimension loading,
partition selection and kernel selection remain in the adapters. The adapters
never call the examples' build functions. Example caller migration is deliberately
separate because those paths belong to other work packages.

Use ``DecoderConfig.from_example("llm")`` or ``from_example("vllm")``
with the same environment as the corresponding example. Explicit configurations
can also be made with the dataclass constructor (no example import is needed).
Imported launch metadata describes CPU graph construction, not GPU correctness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import math
import json
import os

from etx.frontends import hip_link
from etx.ir import Graph, Resource
from etx.codegen import manifest


@dataclass(frozen=True)
class DecoderConfig:
    """Resolved dimensions, tile bindings and schedule choices for one decoder.

    ``settings`` uses the existing vLLM option names to keep the migration
    mechanical. ``per_request`` contains already allocated argument offsets;
    every build copies it because batching annotates launch records in place.
    Native configurations need only dims, resource, partition, context and chunk.
    """

    backend: str
    dims: Any
    resource: Resource | None = None
    partition: Mapping[str, int] = field(default_factory=dict)
    context: int = 1024
    chunk: int = 32
    tiles: str = "examples/llm/tiles.hip"
    imports: Mapping[str, dict] = field(default_factory=dict)
    launches: list[dict] = field(default_factory=list)
    per_request: list[list[dict]] = field(default_factory=list)
    argument_bytes: int = 0
    settings: Mapping[str, Any] = field(default_factory=dict)
    environ: Mapping[str, str] = field(default_factory=dict)
    example_dir: Path = Path("examples/vllm_llm")
    fold_topk: bool = False
    moe_block_m: int = 0
    local_fold: bool = False
    producer_score: bool = False

    @classmethod
    def from_example(cls, backend: str = "llm") -> "DecoderConfig":
        """Resolve an example's current settings without constructing its graph.

        As with the old entry points, set import-time ETX options before importing
        the example module. Callers that reload examples for an A/B should resolve
        a fresh config after that reload.
        """
        if backend == "llm":
            from examples.llm import model as m
            d = m.load_dims()
            ctx, chunk = m._ctx_chunk()
            return cls(backend, d, m.resource(d), m.partition(d), ctx, chunk, m.T)
        if backend != "vllm":
            raise ValueError(f"unsupported decoder backend: {backend!r}")
        from examples.vllm_llm import model as m
        d = m.load_dims()
        imp = m.imports()
        base = m.instances(d)
        skip = set(filter(None, os.environ.get("ETX_VL_SKIP", "").split(",")))
        if skip:
            if m.XCD or m.REQUESTS > 1:
                raise SystemExit("ETX_VL_SKIP cannot be combined with ETX_VL_XCD or ETX_VL_REQUESTS (a replica or its GEMV could be dropped alone)")
            base = [i for i in base if i["role"] not in skip]
        from etx.importer.spec import validate_model_imports
        validate_model_imports(imp, d, {i["export"] for i in base})
        per_request, total = [], 0
        for r in range(m.REQUESTS):
            ins = [dict(i, req=r) for i in base]
            total = m.plan_offsets(ins, imp, total)
            per_request.append(ins)
        names = (
            "ARGPF",
            "ATTN_WIDE",
            "BAR2",
            "BATCH",
            "BATCH_ROLES",
            "CHAIN",
            "CHAIN_FILE",
            "CHAIN_ROLES",
            "COH",
            "CU",
            "DEV_RESET",
            "DSL",
            "DSL_ATTNHEAD",
            "DSL_FILE",
            "DSL_FOLD",
            "DSL_GEMV",
            "DSL_TOPK",
            "LDS_BUDGET",
            "MAXLEN",
            "NX",
            "PART",
            "PIN_ROLES",
            "REP_ROLES",
            "REQUESTS",
            "THREADS",
            "TOUCH",
            "WPX",
            "XAFF_ROLES",
            "XCD",
        )
        from etx.importer.spec import launch_for
        from etx.ops import local_fold, producer
        return cls(backend, d, imports=deepcopy(imp), launches=deepcopy(base),
                   per_request=deepcopy(per_request), argument_bytes=total,
                   settings={name: deepcopy(getattr(m, name)) for name in names},
                   environ=dict(os.environ), example_dir=m.HERE,
                   fold_topk=m.fold_topk(d), local_fold=local_fold.enabled(),
                   producer_score=producer.enabled(),
                   moe_block_m=launch_for(d, "up").block_m if d.moe and not m.DSL_GEMV else 0)


def build(config: DecoderConfig) -> Graph:
    """Build a native or imported decoder without invoking an example builder."""
    if config.backend == "llm":
        if config.resource is None:
            raise ValueError("native decoder requires a resource")
        if config.chunk <= 0 or config.context < config.chunk or config.context % config.chunk:
            raise ValueError("native decoder context must be a positive multiple of chunk")
        return _build_native(config)
    if config.backend == "vllm":
        return _build_imported(config)
    raise ValueError(f"unsupported decoder backend: {config.backend!r}")


def _duration(nbytes: float, tasks: int) -> float:
    return max(1.0, nbytes / tasks / (4.0e12 / 304) * 1e6)


def _build_native(config: DecoderConfig) -> Graph:
    d = config.dims
    p = config.partition
    ctx, chunk = config.context, config.chunk
    NC = ctx // chunk
    res = config.resource
    g = Graph(f"llm_{d.name}")
    g.tensor("llm", (1,), role="weight", bytes_per_elem=8)          # placeholder: the tiles read __constant__ params

    def grid(name, shape, sym, L, ins, outs, dur):
        return g.call_device(name, shape, hip_link(config.tiles, sym), resource=res, args=["llm"], consts=(L,),
                             in_edges=ins, out_edges=outs, duration_us=dur, duration_cv=0.05)

    def ntask(rows, rpt):
        return math.ceil(rows / rpt)

    g.etensor("E_embed", (1,), wait_count=1)
    grid("embed", (1,), "llm_embed", -1, {}, {"E_embed": "i->i"}, 1.0)
    e_in = "E_embed"
    for L in range(d.L):
        n_qkv, n_o = ntask(d.NQKV, p["qkv"]), ntask(d.H, p["o"])
        g.etensor(f"E_qkv_{L}", (1,), wait_count=n_qkv)
        g.etensor(f"E_post_{L}", (d.NKV,), wait_count=d.G + 2)
        g.etensor(f"E_attn_{L}", (d.NKV,), wait_count=NC)
        g.etensor(f"E_merge_{L}", (1,), wait_count=d.NH)
        g.etensor(f"E_o_{L}", (1,), wait_count=n_o)
        grid(f"qkv_{L}", (n_qkv,), "llm_qkv", L, {e_in: "i->(0)"}, {f"E_qkv_{L}": "i->(0)"}, _duration(d.NQKV * d.H * 2, n_qkv))
        grid(f"post_q_{L}", (d.NH,), "llm_post_q", L, {f"E_qkv_{L}": "i->(0)"}, {f"E_post_{L}": f"i->(i/{d.G})"}, 1.0)
        grid(f"post_kv_{L}", (d.NKV, 2), "llm_post_kv", L, {f"E_qkv_{L}": "ij->(0)"}, {f"E_post_{L}": "ij->i"}, 1.0)
        grid(f"attn_{L}", (d.NKV, NC), "llm_attn", L, {f"E_post_{L}": "hc->h"}, {f"E_attn_{L}": "hc->h"}, 3.0)
        grid(f"merge_{L}", (d.NH,), "llm_merge", L, {f"E_attn_{L}": f"i->(i/{d.G})"}, {f"E_merge_{L}": "i->(0)"}, 1.0)
        grid(f"oproj_{L}", (n_o,), "llm_oproj", L, {f"E_merge_{L}": "i->(0)"}, {f"E_o_{L}": "i->(0)"}, _duration(d.H * d.NH * d.HD * 2, n_o))
        g.etensor(f"E_down_{L}", (1,), wait_count=ntask(d.H, p["edn" if d.moe else "dn"]))
        if d.moe:
            n_r, n_egu, n_edn = ntask(d.E, p["r"]), ntask(d.MI, p["egu"]), ntask(d.H, p["edn"])
            g.etensor(f"E_r_{L}", (1,), wait_count=n_r)
            g.etensor(f"E_egu_{L}", (1,), wait_count=d.TOPK * n_egu)
            grid(f"router_{L}", (n_r,), "llm_router", L, {f"E_o_{L}": "i->(0)"}, {f"E_r_{L}": "i->(0)"}, _duration(d.E * d.H * 2, n_r))
            grid(f"egu_{L}", (d.TOPK, n_egu), "llm_egu", L, {f"E_r_{L}": "sb->(0)"}, {f"E_egu_{L}": "sb->(0)"},
                 _duration(d.TOPK * 2 * d.MI * d.H * 2, d.TOPK * n_egu))
            grid(f"edn_{L}", (n_edn,), "llm_edn", L, {f"E_egu_{L}": "i->(0)"}, {f"E_down_{L}": "i->(0)"},
                 _duration(d.TOPK * d.H * d.MI * 2, n_edn))
        else:
            n_gu, n_dn = ntask(d.INTER, p["gu"]), ntask(d.H, p["dn"])
            g.etensor(f"E_gu_{L}", (1,), wait_count=n_gu)
            grid(f"gateup_{L}", (n_gu,), "llm_gateup", L, {f"E_o_{L}": "i->(0)"}, {f"E_gu_{L}": "i->(0)"}, _duration(2 * d.INTER * d.H * 2, n_gu))
            grid(f"down_{L}", (n_dn,), "llm_down", L, {f"E_gu_{L}": "i->(0)"}, {f"E_down_{L}": "i->(0)"}, _duration(d.H * d.INTER * 2, n_dn))
        e_in = f"E_down_{L}"
    n_lm = ntask(d.V, p["lm"])
    g.etensor("E_lm", (1,), wait_count=n_lm)
    g.etensor("E_argmax", (1,), wait_count=1)
    grid("lmhead", (n_lm,), "llm_lmhead", -1, {e_in: "i->(0)"}, {"E_lm": "i->(0)"}, _duration(d.V * d.H * 2, n_lm))
    grid("argmax", (1,), "llm_argmax", -1, {"E_lm": "i->i"}, {"E_argmax": "i->i"}, 2.0)
    return g


def _build_imported(config: DecoderConfig) -> Graph:
    d = config.dims
    ARGPF = config.settings['ARGPF']
    ATTN_WIDE = config.settings['ATTN_WIDE']
    BAR2 = config.settings['BAR2']
    BATCH = config.settings['BATCH']
    BATCH_ROLES = config.settings['BATCH_ROLES']
    CHAIN = config.settings['CHAIN']
    CHAIN_FILE = config.settings['CHAIN_FILE']
    CHAIN_ROLES = config.settings['CHAIN_ROLES']
    COH = config.settings['COH']
    CU = config.settings['CU']
    DEV_RESET = config.settings['DEV_RESET']
    DSL = config.settings['DSL']
    DSL_ATTNHEAD = config.settings['DSL_ATTNHEAD']
    DSL_FILE = config.settings['DSL_FILE']
    DSL_FOLD = config.settings['DSL_FOLD']
    DSL_GEMV = config.settings['DSL_GEMV']
    DSL_TOPK = config.settings['DSL_TOPK']
    LDS_BUDGET = config.settings['LDS_BUDGET']
    MAXLEN = config.settings['MAXLEN']
    NX = config.settings['NX']
    PART = config.settings['PART']
    PIN_ROLES = config.settings['PIN_ROLES']
    REP_ROLES = config.settings['REP_ROLES']
    REQUESTS = config.settings['REQUESTS']
    THREADS = config.settings['THREADS']
    TOUCH = config.settings['TOUCH']
    WPX = config.settings['WPX']
    XAFF_ROLES = config.settings['XAFF_ROLES']
    XCD = config.settings['XCD']
    HERE = config.example_dir
    environ = config.environ
    imp = config.imports
    from etx.importer.spec import validate_model_imports
    base = deepcopy(config.launches)
    import_validation = validate_model_imports(imp, d, {i['export'] for i in base})
    per_req = deepcopy(config.per_request)
    total = config.argument_bytes
    # the imported tiles use the import arena of etx/import.h (sized by the adapter); lds_bytes here is only
    # the ETX tiles' own LDS (argmax: one value and one index per thread)
    res = Resource(threads=THREADS, vgpr=128, agpr=0, lds_bytes=8 * THREADS + 256)
    g = Graph(f"vl_{d.name}" + (f"_r{REQUESTS}" if REQUESTS > 1 else ""))
    if d.moe and not DSL_GEMV:
        g.codegen["moe_block_m"] = config.moe_block_m
    if ARGPF:
        g.codegen["argpf"] = True
    if BAR2:
        g.codegen["bar2"] = True
    if DEV_RESET:
        g.codegen["dev_reset"] = True
    # The fold decision is DERIVED (fold_topk), so the host must learn it from the plan, never from an env var it
    # reads independently: build.sh gated the tile's define on ETX_VL_FOLD_TOPK while the plan derived the opposite,
    # and the expert tiles recomputed a top-k the plan had already given its own phase -- 0.63 ms, measured round 30.
    g.codegen["fold_topk"] = config.fold_topk
    LAST_ROUTER=environ.get('ETX_VL_LAST_ROUTER','0')=='1'
    g.codegen['last_router']=LAST_ROUTER
    if LAST_ROUTER:
        if not (d.H==2048 and d.E==128 and d.TOPK==8 and REQUESTS==1 and CU==256 and NX==8 and
                DSL_GEMV and DSL_FOLD and not config.fold_topk and not BATCH and
                environ.get('ETX_VL_ROUTER_NARROW','0')=='1' and not environ.get('ETX_VL_PRODUCER','0')=='1'):
            raise SystemExit('G01 model integration requires exact kept QwenB1/CU256/narrow/fold/no producer recipe')
        g.tensor('g01_state',(48,32),bytes_per_elem=4,role='runtime')
    # MoE K beyond the tiles' 4096-float staged window (Mixtral moe_down: 14336): the chunked path is compiled in
    # only when a shape needs it -- it measured +0.12 ms on Qwen as a resident runtime branch (round 43).
    g.codegen["dsl_bigk"] = bool(DSL_GEMV and d.moe and max(d.H, d.MI) > 4096)
    # PhiMoE-class: LayerNorm fold + output biases. The tile layout (two extra arg pointers) was sized at emit time
    # by ETX_VL_DSL_EXT (build.sh, from the manifest prep wrote); the plan derives the SAME fact from the config here
    # and the two must agree -- a mismatch is the round-14/19/30 silent class, so it dies loudly instead.
    g.codegen["dsl_ext"] = bool(getattr(d, "ln_norm", False) or getattr(d, "bias_o", False) or getattr(d, "bias_lm", False))
    if DSL_GEMV:
        _ab = imp.get("dsl_gemv", {}).get("arg_bytes", 160)
        if bool(g.codegen["dsl_ext"]) != (_ab == 176):
            raise SystemExit(f"dsl_ext layout mismatch: the plan derives dsl_ext={g.codegen['dsl_ext']} from the "
                             f"config but the emitted tile layout has arg_bytes={_ab} (ETX_VL_DSL_EXT at emit time); "
                             "build.sh must export ETX_VL_DSL_EXT=1 exactly for the models whose manifest says so")
    # Compile-time decisions the tiles and the host must share. They are published through the manifest
    # (etx/codegen/manifest.py) and MUST NOT also be passed as -D: build.sh re-deriving the same decision is how
    # ETX_VL_NB (round 14), ETX_DEV_RESET (rounds 19/20) and the top-k fold (round 30) each produced a build that
    # looked healthy and was wrong. COH in particular was derived twice -- here and in build.sh's own default
    # expression -- which agreed only by hand-matching.
    g.codegen["coh"] = COH
    g.codegen["nb"] = REQUESTS if BATCH else 1
    g.codegen["requests"] = REQUESTS
    g.codegen["uc"] = environ.get("ETX_VL_UC", "0") == "1"
    # The variant of each phase, published through the manifest (plan_data.h; docs/core/SEC7-interface-fix.md). The host
    # compares every plan instance with it and derives nothing itself. Until the planner of section 6.2 lands, the
    # selection is the one instances() made above, so the published value equals today's choice by construction.
    g.codegen[manifest.SELECTION] = manifest.selection_of(((i["role"], i["layer"], i["export"]) for i in base),
                                                          {"threads": THREADS, "vgpr": res.vgpr, "lds": LDS_BUDGET})
    g.tensor("impargs", (total,), role="weight", bytes_per_elem=1)
    g.tensor("vl", (1,), role="weight", bytes_per_elem=8)         # placeholder: the native tiles read __constant__ params
    adapters = environ.get("ETX_VL_ADAPTERS", str(HERE / "build" / "imp_tiles.hip"))

    def src_of(i):                                             # DSL ops link their generator's tiles, imports the adapter's
        return DSL_FILE if i["export"].startswith("dsl_") else adapters

    def touch_of(i, x):                                        # the touch hook of the dense DSL GEMVs (plain or x form)
        return f"etx_touch_{'x_' if x else ''}{i['export']}" if TOUCH and i["export"] in ("dsl_gemv", "dsl_gemv_fold", "dsl_gemv_router_fold", "dsl_gemv_producer") else None

    def coh_of(i):                                             # the DSL tiles are the coherent ones (their source is ours)
        return "coherent" if COH and i["export"].startswith("dsl_") else "imported"

    def caps_of(i):        # the DSL family implements the agent-scope variant (etx/ops/coh.py ETX_DSL_COH_L1) for every
        # model, with write-through (system-scope) stores: ETX_COHERENCE=l1 switches it, no flush is added
        return ("coherent_l1", "coherent_stores") if i["export"].startswith("dsl_") else ()
    native = str(HERE / "vl_tiles.hip")
    pre = (lambda r: f"r{r}_") if REQUESTS > 1 else (lambda r: "")

    prev = {}
    prev_x = {}                                                # the previous event is per XCD ((NX,) shape)
    for r in range(REQUESTS):
        g.etensor(f"{pre(r)}E_pro", (1,), wait_count=1)
        g.call_device(f"{pre(r)}prologue", (1,), hip_link(native, "vl_prologue"), resource=res, args=["vl"], consts=(r,),
                      out_edges={f"{pre(r)}E_pro": "i->i"}, duration_us=2.0)
        prev[r] = f"{pre(r)}E_pro"; prev_x[r] = False
    # ETX_VL_CHAIN: runs of CHAIN_ROLES within one layer become one grid of one task (at most 4, the consts)
    groups_r = []
    for ins in per_req:
        groups, cur = [], []
        for i in ins:
            chainable = CHAIN and i["role"] in CHAIN_ROLES
            if cur and (not chainable or cur[-1]["layer"] != i["layer"] or len(cur) == 4):
                groups.append(cur); cur = []
            if chainable:
                cur.append(i)
            else:
                groups.append([i])
        if cur:
            groups.append(cur)
        groups_r.append(groups)
    chains = {}
    order = []                                                 # graph (and plan-file) order: requests alternate
    for n in range(len(groups_r[0])):
        for r in range(REQUESTS):
            order.append((r, n))
    FT = config.fold_topk

    def is_batched(grp):
        if not BATCH or len(grp) != 1:
            return False
        i = grp[0]
        if i.get("noaff"):                             # the batched MoE rides the slot-per-XCD split (S = s * nb + rq)
            return False
        if i["role"] in BATCH_ROLES and i["export"].startswith("dsl_gemv"):
            return True
        if FT or not (DSL or DSL_TOPK):                              # the batched MoE path needs the standalone top-k's ids/wts
            return False
        if i["role"] == "topk" and i["export"] == "dsl_topk":
            return True
        return i["role"] in ("moe_up", "moe_down") and i["export"].startswith("dsl_moe")

    for r, n in order:
        groups = groups_r[r]
        grp = groups[n]
        bat = is_batched(grp)
        if bat and r > 0:                                      # one node serves every request
            grp[0]["skip"] = True
            continue
        P = "" if bat else pre(r)
        ev = f"{P}E_{n}"
        if bat:
            grp[0]["nb"] = REQUESTS

        def ins_of(m):                                         # in-edges: every request's previous event when batched
            return {prev[rr]: m for rr in range(REQUESTS)} if bat else {prev[r]: m}

        def advance(e, x):
            if bat:
                for rr in range(REQUESTS):
                    prev[rr] = e; prev_x[rr] = x
            else:
                prev[r] = e; prev_x[r] = x
        if LAST_ROUTER and len(grp)==1 and grp[0]['role']=='router':
            i=grp[0];init=f'g01_reset_{i["layer"]}';done=init+'_done'
            g.etensor(done,(1,),wait_count=1)
            g.call_device(init,(1,),hip_link(src_of(i),'etx_tile_g01_reset'),resource=res,args=['g01_state'],
                          consts=(i['layer'],),in_edges=ins_of('i->(0)'),out_edges={done:'i->(0)'},domain_map='i->(0)',duration_us=1.0)
            prev[r]=done;prev_x[r]=False
        if len(grp) == 1 and grp[0]["rep"] > 1 and grp[0]["role"] in REP_ROLES:
            i = grp[0]
            g.etensor(ev, (NX,), wait_count=1)
            name = P + (f"{i['role']}_{i['layer']}_x" if i["layer"] >= 0 else f"{i['role']}_x")
            g.call_device(name, (NX,), hip_link(src_of(i), f"etx_tile_rep_{i['export']}", pre_wait=f"etx_pre_x_{i['export']}", coherence=coh_of(i), coherence_caps=caps_of(i), arg_bytes=i.get("arg_bytes", 0)),
                          resource=res, args=["impargs"],
                          consts=(i["offset"], i["stride"], i["copyoff"]), in_edges=ins_of("x->(0)"), out_edges={ev: "x->x"},
                          domain_map="x->x", duration_us=3.0)
            prev_x[r] = True
        elif len(grp) == 1 and grp[0]["rep"] > 1:
            # a grid split over the XCDs, (XCDs, tasks per XCD) pinned xw->x, each part waiting on its XCD's part of the
            # previous launch: the GEMV after a replica (38 per XCD) or an XCD-affine MoE launch (one expert slot per XCD)
            i = grp[0]
            if bat and i["role"] in ("moe_up", "moe_down"):    # batched MoE: requests x slots tasks, one node
                i["tasks"] *= REQUESTS                         # (a batched dense GEMV that is also a replica -- layer-0
                i["grid"] = (i["grid"][0] * REQUESTS, i["grid"][1], i["grid"][2])   # qkv -- keeps its task count: NB columns share tasks)
            per = i["tasks"] // NX
            if per * NX != i["tasks"] or (i["role"] not in XAFF_ROLES and per != WPX):
                raise SystemExit(f"{i['export']}: {i['tasks']} tasks do not split over {NX} XCDs as {i['role']} needs")
            nxt = groups[n + 1][0] if n + 1 < len(groups) else None
            local_next = nxt is not None and len(groups[n + 1]) == 1 and nxt["role"] in XAFF_ROLES and nxt["rep"] > 1
            if LAST_ROUTER and i['role']=='router':local_next=False # all256 completions include winner fanout before any expert
            if local_next:        # the next launch's part on this XCD waits for this XCD's part only
                g.etensor(ev, (NX,), wait_count=per)
                out = {ev: "xw->x"}; this_x = True
            else:
                g.etensor(ev, (1,), wait_count=i["tasks"])
                out = {ev: "xw->(0)"}; this_x = False
            name = P + (f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"])
            dur = max(2.0, i["M"] * i["K"] * 2 / 4.0e12 * 1e6) if i["export"].startswith("wvsplitk") else 3.0
            g.call_device(name, (NX, per), hip_link(src_of(i), f"etx_tile_x_{i['export']}", pre_wait=f"etx_pre_x_{i['export']}", touch=touch_of(i, True), coherence=coh_of(i), coherence_caps=caps_of(i), arg_bytes=i.get("arg_bytes", 0)),
                          resource=res, args=["impargs","g01_state"] if LAST_ROUTER and i['role']=='router' else ["impargs"],
                          consts=(i["offset"], i["stride"], per, i['layer']) if LAST_ROUTER and i['role']=='router' else (i["offset"],i["stride"],per), in_edges=ins_of("xw->x" if prev_x[r] else "xw->(0)"), out_edges=out,
                          domain_map="xw->x", duration_us=dur)
            prev_x[r] = this_x
        elif len(grp) == 1 and DSL_ATTNHEAD and grp[0]["role"] in ("prep", "attn", "reduce"):
            # the attention block by kv head: prep h -> attention (partition p, head h) -> reduce h, all on XCD h
            i = grp[0]
            name = P + f"{i['role']}_{i['layer']}"
            link = hip_link(src_of(i), f"etx_tile_{i['export']}", pre_wait=f"etx_pre_{i['export']}", coherence=coh_of(i), coherence_caps=caps_of(i), arg_bytes=i.get("arg_bytes", 0))
            if i["role"] == "prep":
                nkv = i["tasks"]
                g.etensor(ev, (nkv,), wait_count=1)
                g.call_device(name, (nkv,), link, resource=res, args=["impargs"], consts=(i["offset"],),
                              in_edges={prev[r]: "i->(0)"}, out_edges={ev: "i->i"}, domain_map="i->i", duration_us=3.0)
            elif i["role"] == "attn":
                parts = i["parts"]; nkv = i["tasks"] // parts
                if nkv * parts != i["tasks"]:
                    raise SystemExit(f"ETX_VL_DSL_ATTN=head needs one attention block per task (ETX_VL_SLOTS={i['export']}=1)")
                g.etensor(ev, (nkv,), wait_count=parts)
                dm = f"i->(i%{NX})" if ATTN_WIDE else f"i->(i/{parts})"   # placement only; the edges stay per kv head
                g.call_device(name, (i["tasks"],), link, resource=res, args=["impargs"], consts=(i["offset"],),
                              in_edges={prev[r]: f"i->(i/{parts})"}, out_edges={ev: f"i->(i/{parts})"}, domain_map=dm, duration_us=8.0)
            else:
                nkv = i["tasks"]
                if i["grid"][0] != nkv * i["G"]:
                    raise SystemExit(f"ETX_VL_DSL_ATTN=head needs the reduce's slots to equal G = {i['G']} (one kv head per task)")
                g.etensor(ev, (1,), wait_count=nkv)
                g.call_device(name, (nkv,), link, resource=res, args=["impargs"], consts=(i["offset"],),
                              in_edges={prev[r]: "i->i"}, out_edges={ev: "i->(0)"}, domain_map="i->i", duration_us=4.0)
        elif len(grp) == 1:
            i = grp[0]
            g.etensor(ev, (1,), wait_count=i["tasks"])
            name = P + (f"{i['role']}_{i['layer']}" if i["layer"] >= 0 else i["role"])
            pin = {"domain_map": "i->(0)"} if i["role"] in PIN_ROLES else {}
            g.call_device(name, (i["tasks"],), hip_link(src_of(i), f"etx_tile_{i['export']}", pre_wait=f"etx_pre_{i['export']}", touch=touch_of(i, False), coherence=coh_of(i), coherence_caps=caps_of(i), arg_bytes=i.get("arg_bytes", 0)),
                          resource=res, args=["impargs","g01_state"] if LAST_ROUTER and i['role']=='router' else ["impargs"],
                          consts=(i["offset"],0,0,i['layer']) if LAST_ROUTER and i['role']=='router' else (i["offset"],), in_edges=ins_of("i->(0)"), out_edges={ev: "i->(0)"}, **pin,
                          duration_us=2.0 if not i["export"].startswith("wvsplitk") else max(2.0, i["M"] * i["K"] * 2 / 4.0e12 * 1e6))
        else:
            sig = "__".join(x["export"] for x in grp)
            chains[sig] = [x["export"] for x in grp]
            g.etensor(ev, (1,), wait_count=1)
            name = P + "chain_" + "_".join(x["role"] for x in grp) + f"_{grp[0]['layer']}"
            pin = {"domain_map": "i->(0)"} if any(x["role"] in PIN_ROLES for x in grp) else {}
            g.call_device(name, (1,), hip_link(CHAIN_FILE, f"etx_chain_{sig}", pre_wait=f"etx_pre_chain_{sig}"), resource=res, args=["impargs"],
                          consts=tuple(x["offset"] for x in grp), in_edges={prev[r]: "i->(0)"}, out_edges={ev: "i->(0)"},
                          duration_us=2.0 * len(grp), **pin)
        advance(ev, prev_x[r] if (len(grp) == 1 and grp[0]["rep"] > 1) else False)
    if chains:
        with open(CHAIN_FILE, "w") as f:
            f.write("// generated by examples/vllm_llm/model.py (ETX_VL_CHAIN=1): chained launches, one task each\n#pragma once\n")
            f.write(f'#include "{adapters}"\n')
            for sig, exps in chains.items():
                f.write(f'extern "C" __device__ __attribute__((always_inline)) void etx_chain_{sig}(const etx_ctx* c) {{\n')
                f.write("  const unsigned char* base = etx_imp_argblock(c->args, 0);\n")
                f.write("".join(f"  etx_all_blocks_{e}(base + c->cst[{k}]);\n" for k, e in enumerate(exps)))
                f.write("}\n")
                # before the wait: every part's argument block into the scalar cache
                f.write(f'extern "C" __device__ __attribute__((always_inline)) void etx_pre_chain_{sig}(const etx_ctx* c) {{\n')
                f.write("".join(f"  etx_pre_{e}_at(c, c->cst[{k}]);\n" for k, e in enumerate(exps)))
                f.write("}\n")
    for r in range(REQUESTS):
        g.etensor(f"{pre(r)}E_argmax", (1,), wait_count=1)
        g.call_device(f"{pre(r)}argmax", (1,), hip_link(native, "vl_argmax"), resource=res, args=["vl"], consts=(r,),
                      in_edges={prev[r]: "i->(0)"}, out_edges={f"{pre(r)}E_argmax": "i->i"}, duration_us=3.0)

    # Off by default; tile emission, graph and host launch rows share one flag.
    from etx.ops import local_fold
    if config.local_fold:
        local_fold.apply(g, d)
    from etx.ops import producer
    if config.producer_score:
        producer.apply(g, d)
    plan = environ.get("ETX_VL_PLAN")
    if plan:
        Path(str(plan) + '.imports-check.json').write_text(json.dumps(import_validation, indent=2) + '\n')
        insts = [i for r, n in order for i in groups_r[r][n]]
        if config.local_fold:
            insts = local_fold.plan_instances(insts)
        with open(plan, "w") as f:
            f.write(f"# vl plan {d.name}: instance export layer role gx gy gz offset tasks M K wvprgrp rep stride copyoff req nb; total {total} bytes\n")
            f.write(f"total {total} parts {math.ceil(MAXLEN / PART)} maxlen {MAXLEN} requests {REQUESTS}\n")
            for n, i in enumerate(insts):
                if i.get("local_fold_argument_alias"):
                    f.write("# local-fold assembler: argument-layout alias of the next row; actual launch has 8 tasks\n")
                if i.get("skip"):                    # ETX_VL_BATCH: request 0's node runs this group for everyone
                    continue
                gx, gy, gz = i["grid"]
                f.write(f"{n} {i['export']} {i['layer']} {i['role']} {gx} {gy} {gz} {i['offset']} {i['tasks']} "
                        f"{i.get('M', 0)} {i.get('K', 0)} {i.get('wvprgrp', 0)} {i['rep']} {i['stride']} {i['copyoff']} {i['req']} {i.get('nb', 1)}\n")
    return g
