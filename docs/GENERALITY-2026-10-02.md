# Generality, measured: a second MoE through the ETX path (2026-10-02)

Goal: Mixtral-8x7B (top-2 of 8 experts, GQA G=4) end to end through the imported-kernel path, exercising the
TOPK < NX unpinned MoE mapping (commit 6012b5b) that had only ever been planned. DeepSeek-V2-Lite was rejected
first: MLA's projections do not fit the path's GQA parameterisation, and its `first_k_dense_replace` mixed layout
is now refused loudly instead of planned wrongly (examples/llm/model.py).

## How far the ladder got

| step | result |
|---|---|
| config + local plan | passes: 58307 tasks / 452 events, MoE and top-k unpinned (TOPK % NX != 0), attnprep correctly derived OFF (no q/k norm) |
| container device build | passes: megakernel VGPR 128, spill 0, LDS 41384; attn_mfma4_g4 (G=4) imports cleanly |
| weights + prep + HF golden | passes: 93.41 GB packed, manifest TOPK=2 MI=14336 norm_topk=1 |
| `--check` vs vLLM originals | **every imported launch byte-identical for layers 0-1** |
| vLLM itself vs the HF golden | 32/32 (8.662 ms/token on this box) |
| ETX tokens vs the HF golden | **0/32 — the remaining failure, root-caused below, not yet fixed** |

## The finding: the imported path is shape-keyed at three levels

The kernels we import are not one artefact per operator; they are one artefact per operator PER SHAPE, at three
different levels, and only the first level fails loudly:

1. **Export selection** (loud). `wvsplitk_y1` missing for a non-Qwen MoE because the check script re-derived MoE
   detection from `num_experts` alone — the round-28 bug class in a second copy. Fixed: the check script now asks
   `load_dims`, the one derivation. The silu block size `MI/8` exceeded the 1024-thread workgroup at Mixtral's
   expert width (1792) — capped; vLLM's act kernels stride, so the block is shape, not correctness.
2. **C++ template instantiations** (silent). `topk_softmax_e128` is `topkGating<8, 128, ...>` and `moe_sum_k8` is
   `moe_sum_vec_kernel<..., 8, ...>`: expert count and top-k are template arguments. Run at top-2-of-8 they compute
   deterministic garbage, and `--check` CANNOT catch it — the original receives the same wrong template. Fixed:
   shape-keyed manifest entries (`topk_softmax_e8`, `moe_sum_k2`; the template parameters derive as
   BYTES = min(16, 2E), VPT = max(1, E/(ELTS·64))·ELTS) and the export names in build.sh / rocm_build_check.sh /
   model.py are now `topk_softmax_e$(E)` / `moe_sum_k$(TOPK)`, so an uncaptured shape fails at import with a name
   instead of emitting EOS.
3. **Triton cache entries** (silent, OPEN). `fused_moe_up`/`fused_moe_down` are binaries harvested from the Triton
   cache vLLM wrote while running Qwen3-30B-A3B, with `top_k = 8` (up) and `top_k = 1, MUL_ROUTED_WEIGHT` (down)
   baked as constexprs, plus Qwen-shaped block configs and divisibility specialisations. At top-2 the token→row
   mapping is wrong and the first generated token becomes EOS (id 2). **The fix is mechanical and unstarted**: run
   vLLM once on Mixtral with a mounted TRITON_CACHE_DIR, harvest the top-2 entries into
   `examples/vllm_llm/triton/<export>/` keyed by shape, and name them per shape like level 2.

Progression of the first wrong token as the levels were fixed: 8192 (levels 2+3 wrong) → 2 = EOS (level 3 only).

## What generality cost, in edits

Zero compiler changes for the plan, the runtime, the events, the placement — the TOPK<NX branch, G=4 attention
import and the whole megakernel machinery worked unmodified. The edits were all in the model/data layer, each a
shape fact: the Mixtral config; `norm_topk` derived for `model_type=mixtral` (it always renormalises, has no
config key); Mixtral weight naming in prep (`block_sparse_moe`, `w1/w3/w2`); the two script fixes and the
manifest entries above. The one genuinely open compiler-adjacent item is level 3's per-shape Triton capture flow.

DSL-tile note: the DSL GEMV path is closed to Mixtral by `K <= 4096` (host.hip guard; moe_down K = MI = 14336) —
the LDS staging buffer is K floats. A K-chunked staging variant would open it; recorded, not attempted.

Timing context (NOT a comparison — ETX's output is still wrong): vLLM 8.662 ms/token; the wrong-output ETX runs
were ~12.7 (e128 top-k alone was ~5 ms of that: 12.9 → 7.5 with the DSL top-k, which takes E at runtime).
