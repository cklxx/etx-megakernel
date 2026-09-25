"""Generic decoder-only LLM (batch-1 decode) as an ETX graph, built from a Hugging Face config.json.

Covers the Llama / Qwen2 / Qwen3 dense family (GQA, optional QKV bias, optional per-head q/k
RMSNorm, tied or untied embeddings) and the Qwen3-MoE family (softmax router, top-k experts,
optional top-k renormalisation). One step decodes one token; the host launches one step per token.

Select the model with ETX_LLM_CONFIG=<path to config.json> (defaults to the bundled Qwen3-8B config).
The same file is imported by the host tooling for the partition constants (rows per task), so the
graph and the tiles agree on every task's rows.

Per layer (dense):                                   events (all counters; scopes chosen by P2/P4)
  qkv    (n_qkv,)   norm(x) . Wqkv rows (+bias)      E_qkv   (1,)    <- all qkv tasks
  post_q (NH,)      q-norm, RoPE -> q                E_post  (NKV,)  <- G q heads + k + v of that group
  post_kv(NKV, 2)   k-norm, RoPE -> K cache; V cache
  attn   (NKV, NC)  one KV head group x one chunk    E_attn  (NKV,)  <- NC chunks
  merge  (NH,)      combine the chunk partials       E_merge (1,)    <- NH heads
  oproj  (n_o,)     x += Wo . o                      E_o     (1,)
  gateup (n_gu,)    h = silu(g) * u of norm(x)       E_gu    (1,)
  down   (n_dn,)    x += Wd . h                      E_down  (1,)    -> next layer's qkv
MoE layers replace gateup/down by
  router (n_r,)     logits = Wr . norm(x)            E_r     (1,)
  egu    (TOPK, n_egu) expert slot s of the token    E_egu   (1,)    (each task recomputes the top-k
  edn    (n_edn,)   x += sum_s w_s Wd[e_s] . h_s     E_down  (1,)     from the logits: no top-k phase)
Head: lmhead (n_lm,) partial argmax over vocab rows -> E_lm -> argmax (1,).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from etx.frontends import hip_link
from etx.ir import Graph, Resource

T = "examples/llm/tiles.hip"
HERE = Path(__file__).resolve().parent
WORKERS = 304              # 8 XCDs x 38 CUs (default when the LDS allows one workgroup per CU)
CTX = int(os.environ.get("ETX_LLM_CTX", "1024"))
CHUNK = int(os.environ.get("ETX_LLM_CHUNK", "32"))   # attention chunk (positions per task); small so short contexts still spread
HBM_BPS = 4.0e12           # for the duration estimates only


@dataclass
class Dims:
    name: str
    H: int
    NH: int
    NKV: int
    HD: int
    INTER: int
    L: int
    V: int
    eps: float
    theta: float
    tie: bool
    bias: bool
    qk_norm: bool
    moe: bool
    E: int = 0
    TOPK: int = 0
    MI: int = 0
    norm_topk: bool = False

    @property
    def G(self) -> int:
        return self.NH // self.NKV

    @property
    def NQKV(self) -> int:
        return (self.NH + 2 * self.NKV) * self.HD


def load_dims(path: str | None = None) -> Dims:
    path = path or os.environ.get("ETX_LLM_CONFIG") or str(HERE / "configs" / "qwen3-8b.json")
    c = json.loads(Path(path).read_text())
    mt = c.get("model_type", "")
    H, NH = c["hidden_size"], c["num_attention_heads"]
    moe = "num_experts" in c and c.get("num_experts", 0) > 0
    if moe and (c.get("mlp_only_layers") or c.get("decoder_sparse_step", 1) != 1):
        raise ValueError("mixed dense / MoE layer layouts are not supported by this example")
    nl = int(os.environ.get("ETX_LLM_LAYERS", "0")) or c["num_hidden_layers"]      # tests: fewer layers
    return Dims(name=Path(path).parent.name if Path(path).name == "config.json" else Path(path).stem,
                H=H, NH=NH, NKV=c.get("num_key_value_heads", NH), HD=c.get("head_dim", H // NH),
                INTER=c.get("intermediate_size", 0), L=min(nl, c["num_hidden_layers"]), V=c["vocab_size"],
                eps=float(c.get("rms_norm_eps", 1e-6)), theta=float(c.get("rope_theta", 10000.0)),
                tie=bool(c.get("tie_word_embeddings", False)), bias=mt == "qwen2" or bool(c.get("attention_bias", False)),
                qk_norm=mt.startswith("qwen3"), moe=moe, E=c.get("num_experts", 0), TOPK=c.get("num_experts_per_tok", 0),
                MI=c.get("moe_intermediate_size", 0), norm_topk=bool(c.get("norm_topk_prob", False)))


def workers(d: "Dims") -> int:
    """Workers the plan will get on MI300X: 8 XCDs x 38 CUs x wg/CU (no relay by default). Two workgroups fit
    a CU when the LDS allows it (vgpr is declared 128, so __launch_bounds__ caps the registers accordingly)."""
    wg = 2 if 2 * (lds_bytes(d) + 256) <= 65536 else 1
    if os.environ.get("ETX_LLM_WG"):
        wg = int(os.environ["ETX_LLM_WG"])               # experiment: force the residency
    return 8 * 38 * wg


def rows_per_task(rows: int, target: int = WORKERS, mult: int = 4) -> int:
    r = max(1, math.ceil(rows / target))
    return ((r + mult - 1) // mult) * mult


def partition(d: Dims) -> dict[str, int]:
    """Rows per task for every GEMV-shaped grid; the host passes these to the tiles."""
    W = workers(d)
    p = {"qkv": rows_per_task(d.NQKV, W), "o": rows_per_task(d.H, W), "lm": rows_per_task(d.V, W)}
    if d.moe:
        p["r"] = rows_per_task(d.E, target=32)
        p["egu"] = rows_per_task(d.MI, target=max(1, W // d.TOPK))
        p["edn"] = rows_per_task(d.H, W)
    else:
        p["gu"] = rows_per_task(d.INTER, W)
        p["dn"] = rows_per_task(d.H, W)
    return p


def _ctx_chunk() -> tuple[int, int]:
    """CTX / CHUNK the host will use: from llm.manifest next to the config when it exists (prep.py wrote it)."""
    cfg = os.environ.get("ETX_LLM_CONFIG")
    if cfg:
        man = Path(cfg).parent / "llm.manifest"
        if man.exists():
            kv = dict(l.split()[1:3] for l in man.read_text().splitlines() if l.startswith("cfg "))
            return int(float(kv["CTX"])), int(float(kv["CHUNK"]))
    return CTX, CHUNK


def lds_bytes(d: Dims) -> int:
    """Input vector + GEMV partials (4 waves x rows per task) + attention scratch, in floats, plus 4 KB."""
    vec = max(d.H, d.NH * d.HD, d.INTER if not d.moe else 0, d.TOPK * d.MI)
    part = 4 * 520                        # largest task: lm_head rows (<= 516) x 4 waves; lm_head also keeps its logits
    lm = 2 * d.H + 520 + part             # sliced lm_head: folded residual + normed input + logits + partials
    return (max(vec + 4 * 64, lm)) * 4 + 4096


def _us(nbytes: float, tasks: int) -> float:
    return max(1.0, nbytes / tasks / (HBM_BPS / 304) * 1e6)


def build() -> Graph:
    d = load_dims()
    p = partition(d)
    ctx, chunk = _ctx_chunk()
    NC = ctx // chunk
    res = Resource(threads=256, vgpr=256 if workers(d) == 304 else 128, agpr=0, lds_bytes=lds_bytes(d))
    g = Graph(f"llm_{d.name}")
    g.tensor("llm", (1,), role="weight", bytes_per_elem=8)          # placeholder: the tiles read __constant__ params

    def grid(name, shape, sym, L, ins, outs, dur):
        return g.call_device(name, shape, hip_link(T, sym), resource=res, args=["llm"], consts=(L,),
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
        grid(f"qkv_{L}", (n_qkv,), "llm_qkv", L, {e_in: "i->(0)"}, {f"E_qkv_{L}": "i->(0)"}, _us(d.NQKV * d.H * 2, n_qkv))
        grid(f"post_q_{L}", (d.NH,), "llm_post_q", L, {f"E_qkv_{L}": "i->(0)"}, {f"E_post_{L}": f"i->(i/{d.G})"}, 1.0)
        grid(f"post_kv_{L}", (d.NKV, 2), "llm_post_kv", L, {f"E_qkv_{L}": "ij->(0)"}, {f"E_post_{L}": "ij->i"}, 1.0)
        grid(f"attn_{L}", (d.NKV, NC), "llm_attn", L, {f"E_post_{L}": "hc->h"}, {f"E_attn_{L}": "hc->h"}, 3.0)
        grid(f"merge_{L}", (d.NH,), "llm_merge", L, {f"E_attn_{L}": f"i->(i/{d.G})"}, {f"E_merge_{L}": "i->(0)"}, 1.0)
        grid(f"oproj_{L}", (n_o,), "llm_oproj", L, {f"E_merge_{L}": "i->(0)"}, {f"E_o_{L}": "i->(0)"}, _us(d.H * d.NH * d.HD * 2, n_o))
        g.etensor(f"E_down_{L}", (1,), wait_count=ntask(d.H, p["edn" if d.moe else "dn"]))
        if d.moe:
            n_r, n_egu, n_edn = ntask(d.E, p["r"]), ntask(d.MI, p["egu"]), ntask(d.H, p["edn"])
            g.etensor(f"E_r_{L}", (1,), wait_count=n_r)
            g.etensor(f"E_egu_{L}", (1,), wait_count=d.TOPK * n_egu)
            grid(f"router_{L}", (n_r,), "llm_router", L, {f"E_o_{L}": "i->(0)"}, {f"E_r_{L}": "i->(0)"}, _us(d.E * d.H * 2, n_r))
            grid(f"egu_{L}", (d.TOPK, n_egu), "llm_egu", L, {f"E_r_{L}": "sb->(0)"}, {f"E_egu_{L}": "sb->(0)"},
                 _us(d.TOPK * 2 * d.MI * d.H * 2, d.TOPK * n_egu))
            grid(f"edn_{L}", (n_edn,), "llm_edn", L, {f"E_egu_{L}": "i->(0)"}, {f"E_down_{L}": "i->(0)"},
                 _us(d.TOPK * d.H * d.MI * 2, n_edn))
        else:
            n_gu, n_dn = ntask(d.INTER, p["gu"]), ntask(d.H, p["dn"])
            g.etensor(f"E_gu_{L}", (1,), wait_count=n_gu)
            grid(f"gateup_{L}", (n_gu,), "llm_gateup", L, {f"E_o_{L}": "i->(0)"}, {f"E_gu_{L}": "i->(0)"}, _us(2 * d.INTER * d.H * 2, n_gu))
            grid(f"down_{L}", (n_dn,), "llm_down", L, {f"E_gu_{L}": "i->(0)"}, {f"E_down_{L}": "i->(0)"}, _us(d.H * d.INTER * 2, n_dn))
        e_in = f"E_down_{L}"
    n_lm = ntask(d.V, p["lm"])
    g.etensor("E_lm", (1,), wait_count=n_lm)
    g.etensor("E_argmax", (1,), wait_count=1)
    grid("lmhead", (n_lm,), "llm_lmhead", -1, {e_in: "i->(0)"}, {"E_lm": "i->(0)"}, _us(d.V * d.H * 2, n_lm))
    grid("argmax", (1,), "llm_argmax", -1, {"E_lm": "i->i"}, {"E_argmax": "i->i"}, 2.0)
    return g


def bindings(**overrides) -> dict:
    return {}


def runtime(bindings, rng=None):
    return {}


if __name__ == "__main__":      # print the partition the host needs: `python examples/llm/model.py`
    d = load_dims()
    print(json.dumps({"dims": d.__dict__, "rows_per_task": partition(d), "ctx": CTX, "chunk": CHUNK, "lds": lds_bytes(d)}))
