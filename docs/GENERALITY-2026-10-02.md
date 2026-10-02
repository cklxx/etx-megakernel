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

## Round 39: the Triton harvest finished — kernels faithful, one dispatch-glue bug remains

Harvested the Mixtral-shape `fused_moe_up`/`fused_moe_down` from vLLM's Triton cache (ran vLLM once with a mounted
`TRITON_CACHE_DIR`, eager so decode compiles at the true M=1; classified the entries by `arith.mulf`/`topk_weights`
in the `.ttir` — mulf=0 is up, mulf=1 is down with `MUL_ROUTED_WEIGHT` — and by block constants against vLLM's
`E=8,N={14336,4096}` tuned configs). Committed as shape-keyed `fused_moe_up_e8` / `fused_moe_down_e8`; the exports
are now `fused_moe_{up,down}` for E=128 (Qwen, byte-stable) else `_e{E}`, same pattern as `topk_softmax_e{E}` /
`moe_sum_k{TOPK}`.

Result: **every imported launch is byte-identical to vLLM's original — `--check 32` reports 0 of 384 launches
differ, all 32 layers.** The first wrong token moved EOS(2) → 6495 (HF 28711) at position 0: the kernels now run the
correct shape, so the garbage is gone, but the chained output is still wrong.

Because `--check` validates each kernel on reference I/O and finds all faithful, the remaining bug is NOT a kernel —
it is ETX's hand-built MoE **dispatch metadata** for top-k ≠ 8: `num_tokens_post_padded`/`sorted_token_ids`/
`expert_ids` wired at `host.hip:216,405-409` with `ntpp_val = TOPK * MOE_BM` and `ti_r` as the sorted ids. This was
written for Qwen's top-8 single-token layout; at top-2 the per-block expert assignment the Triton `fused_moe_kernel`
reads is not what ETX hands it. **Operator: moe_up/moe_down dispatch glue (not the kernels). Layer: 0 onward.
Magnitude: argmax wrong at the first decode step (6495 vs 28711).** This is a host/compiler change, not a cache
harvest, so it is the next round's item — the mechanical harvest this round owned is complete and correct.

Zero-compiler-change verdict, updated: the plan, events, placement and every imported kernel generalise to Mixtral
with no compiler change; the one remaining compiler-side item is the MoE dispatch metadata for top-k ≠ 8.

## Round 40: the top-k≠8 blocker localised to MoE dispatch *values*, not reached to 32/32

The directive was to fix ETX's MoE dispatch metadata for top-k≠8. Diagnosis narrowed it hard but did not close it;
recording precisely, no code changed this round (so Qwen is untouched; the box plan stays 60362/436).

Ruled out, each with the test:
- **Kernel fidelity**: `--check 32` launches BOTH vLLM's original and the imported kernel with ETX's *production*
  argument block (host.hip:493 `av[k] = ab.data()+x.off+d->off[k]`) — 0 of 384 differ. So `--check` proves fidelity
  but is BLIND to a wrong argument: both kernels process the same wrong value identically. This reframes round 39.
- **Replica / unpinned wiring** (the round-35 NOREP hint): `ETX_VL_XCD=0`, single domain, rep=1 everywhere — same
  divergence (6495), so not the per-XCD `rr` addressing.
- **Config**: my mixtral-8x7b.json matches the HF config field-for-field (sliding_window None, rope_theta 1e6,
  head_dim 128); nothing ETX derives is wrong.
- **Argument order**: the harvested up/down `.ttir` and 22-param `.llir` signatures are identical Qwen↔Mixtral, so
  the Qwen-validated wiring (host.hip:405-409) maps positionally with only TOPK/MI/H substituted.

First divergence: generated token 0, argmax wrong (ours 6495, HF 28711), present from the simplest single-domain
plan. Since config/order/replicas/kernels are all eliminated, the fault is a dispatch VALUE that is correct at
TOPK=8 and wrong at TOPK=2 — the candidates are `expert_ids` ordering (ETX passes `topk_ids` in router order;
vLLM's `moe_align_block_size` sorts tokens by expert id, and the down kernel's `MUL_ROUTED_WEIGHT` pairs
`topk_weights` with that sorted order) and `num_valid_tokens`/`EM`. `--check` cannot distinguish these because it
feeds both sides the same values. The deciding experiment, beyond this round's budget: capture vLLM's actual
`sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` for one Mixtral `fused_moe` call (instrument the container
run) and diff against ETX's uploaded values at host.hip:216,405-409. That diff names the exact wrong scalar.

Verdict unchanged and now sharper: plan, events, placement and every imported kernel generalise with zero compiler
change; the one open item is the MoE dispatch *values* (not the layout, not the kernels) for top-k ≠ 8.

## Round 41: the MoE-dispatch hypothesis is refuted from vLLM's source; the bug is upstream

Route 1 did not name a wrong scalar — it proved there is none. vLLM 0.27.1 `fused_moe.py:1556` gates the naive
block path on `expert_map is None and num_tokens*top_k*4 <= global_num_experts`. At decode (num_tokens=1): Qwen
`1*8*4=32 ≤ 128` and Mixtral `1*2*4=8 ≤ 8` — BOTH naive. In the naive branch (`:1566`) vLLM sets
`sorted_token_ids=None`, `expert_ids = topk_ids.view(-1)`, `num_tokens_post_padded = topk_ids.numel()*BLOCK_SIZE_M`,
and the kernel (`:408` `naive_block_assignment` true) reads `off_experts = expert_ids[pid_m]`,
`offs_token = where(offs==0, pid_m, num_valid)`, `a` row `offs_token // top_k`. That is **exactly**
host.hip:216/405-409: `ntpp = TOPK*MOE_BM`, `expert_ids = ti_r = topk_ids` in router order, `num_valid = TOPK`,
single activation row. It degenerates correctly at both shapes with no change. Round 40's "wrong dispatch value"
was wrong: the dispatch is a verbatim, correct implementation of the naive path.

Confirmed by experiment: `ETX_VL_DSL_TOPK=1` (a runtime top-k that takes E/TOPK as data, not a baked kernel)
leaves the divergence bit-identical (6495 at token 0). So routing/top-k is not it either.

**What is left, after eliminating config, argument order, replicas, kernel fidelity, MoE dispatch and routing:**
the pieces that differ Qwen↔Mixtral and that `--check` cannot validate because it feeds both sides ETX's args —
the **G=4 attention path** (`attn_mfma4_g4`/`attn_reduce`, which no end-to-end-correct ETX model has exercised;
every shipped model is G=8) and ETX's own non-imported tiles (embedding gather, argmax). The divergence is at
generated token 0 with argmax 6495 vs 28711.

**Next experiment (the directive's final step):** dump ETX's layer-0 output activation and compare against a torch
forward computed directly from the HF weights on disk — that isolates attention vs MLP vs embedding without a vLLM
reference. Stopping here per the directive, with the suspect set reduced to the attention path and the ETX tiles.
No code changed this round (nothing in the MoE dispatch was wrong); Qwen is untouched.

## Round 42: two scalar bugs named and fixed — Mixtral-8x7B reaches 32/32

The layer-0 activation bisect ran as directed and the prime suspect is CLEARED: at position 0 (BOS, 1-layer build,
`ETX_DUMP` stage dump in host.hip) hn/qkv/attn_out match a torch forward of the on-disk HF weights to bf16 noise
(corr 1.00000), and at positions 1 and 300 the full stage chain — rope (theta 1e6, neox), G=4 attention over 2 and
301 keys, router, top-k — matches too. **attn_mfma4_g4/attn_reduce are correct.** (The dumped `o`/`dn` buffers are
stale in the imported path — rlog computed downstream of o_proj matched while `o` didn't; dump artifact, not a bug.)

What the bisect actually named, in order:

**Bug 1 — fused_moe grid vs the harvested binary's BLOCK_SIZE_N.** mc1 was written only in slot 0's gate half;
slot 1 and every up column were zeros (mc2 = silu(g)·0 = 0 legitimately, so the whole MoE output was ~0 and the
model collapsed to attention-only). Cause: `MOE_BN = 64` is the Qwen harvest's BLOCK_SIZE_N; the Mixtral e8 ttir
bakes BM=16 BK=256 **BN=16**, so the grid `TOPK*ceil(2*MI/64)` launched 896 of the 3584 blocks the binary needs.
`--check` is blind to a wrong grid for the same reason it is blind to wrong argument values: it launches BOTH
kernels with ETX's grid. Fix: model.py keys MOE_BN by shape exactly as the kernel name is keyed (`MOE_BN_E8 = 16`).

**Bug 2 — wvSplitK's CuCount argument vs the launched grid.** After bug 1 the model was right at steps 0–1 but
wrong at 174 of 1024 teacher-forced prompt positions — precisely the positions whose true argmax token id falls in
rows the lm_head never writes. host.hip passed `CuCount=304` (the device's CU count) while the gemv grid is
`CU=256` blocks (the DSL-tuning default in model.py). The kernel's row loop strides by `CuCount*_WvPrGrp*YTILE`
(skinny_gemms.cu:623,791), so rows `[256*wv*YT, 304*wv*YT)` of every pass — for V=32000, [8192,9728) mod 9728 —
were never written and held stale values. Only lm_head loops (every other gemv fits one pass: qkv 6144 = 256*12*2
exactly, o_proj 4096 = 256*8*2 exactly), which is why 28 rounds of Qwen gates never tripped it: a hole only shows
when the true argmax lands in it (Qwen's goldens never did; Mixtral's "):\n" tokens 28711/28732 ≡ 9255/9276 mod
9728 sit inside). Deterministic, reproduced identically by --unfused, found by diffing per-step argmax against a
single HF prefill (`ETX_TOKTRACE`), then solving which input each 4096-row logits block had consumed. Fix:
host.hip passes the grid size (`x.gx*x.gy*x.gz`) as CuCount.

Gates, all green:
- Mixtral-8x7B: **32/32 generated tokens match HF greedy** (b_r42_fix2, imported mode), two runs.
- Mixtral `--check 32`: 0 launches differ (the imported kernels remain byte-faithful under the fixed grids).
- Qwen3-30B-A3B: plan byte-identical to the round-37 reference; GPU run **32/32** and **CHECK-DSL OK** (0 values
  beyond 2 bf16 ulps, layer 47), median 4.679 ms/token — unchanged.

Diagnostics added (both env-gated, zero cost when unset): `ETX_DUMP`/`ETX_DUMP_STEP` stage-buffer dump after one
step; `ETX_TOKTRACE` per-step argmax trace over the whole teacher-forced prompt.

GENERALITY VERDICT: the second MoE model — different expert count (8 vs 128), different top-k (2 vs 8), different
GQA ratio (G=4 vs G=8), untied embeddings, vocab 32000 — runs end-to-end through the unchanged ETX compiler
(plan, events, placement, importer) to exact HF-greedy agreement. Everything model-specific that broke was
host-side glue, and each break was a scalar keyed to shape: the Triton binaries' baked constexprs (rounds 38-39),
their launch grids (bug 1), and a grid-vs-stride constant (bug 2, a latent bug for every model, exposed by vocab
position). The megakernel machinery itself needed nothing.

Timed runs (3 interleaved rounds, context 1024, batch 1, all runs 32/32 vs HF greedy; logs results_2026-10-01/round42/):

| config                      | per-token median (3 rounds)        |
|-----------------------------|------------------------------------|
| ETX megakernel (imported)   | 16.719 / 16.756 / 16.826 ms        |
| vLLM 0.27.1 default         |  8.681 /  8.697 /  8.711 ms        |
| vLLM 0.27.1 AITER           | 12.488 / 12.499 / 12.525 ms        |

ETX Mixtral is ~1.9x slower than vLLM default — expected: this is the all-imported mode (the Qwen equivalent was
6.435 vs 4.937 in round 24), with Qwen-tuned CU/grid settings and none of the DSL-tile or fusion tuning applied to
this shape. The generality claim this document is for is CORRECTNESS of the compiler path on a second model, which
is now closed; making Mixtral fast is its own (unstarted) tuning ladder. One decode trace captured
(round42/trace_r42_mixtral.{bin,log}, median 16.751 ms on-device).

## Round 44: Qwen3.6-35B-A3B — refused at the config, correctly

The third candidate (`Qwen/Qwen3.6-35B-A3B`, model_type `qwen3_5_moe`, 71.9 GB bf16) was examined and **not
attempted**: it is not a shape variant of the family this path expresses, it is a different architecture class.
Decided from `config.json` and the checkpoint index alone — no download, no GPU time.

What the checkpoint actually contains (verified against `model.safetensors.index.json`, 1045 tensors):

| fact | value | why it stops the ladder |
|---|---|---|
| hybrid attention | **30 of 40 layers are `linear_attention`** (gated delta-rule: `A_log`, `dt_bias`, `conv1d` k=4, 32 value heads × 128), full attention only every 4th layer | ETX has no linear-attention tile family, and the recurrent state is **cross-step persistent state** — a tensor lifetime the per-step task graph does not have (the KV cache is today's only cross-step state, owned by imported kernels) |
| `attn_output_gate: true` | output gating on the 10 full-attention layers | not expressible in the current qkv/o_proj path |
| shared expert | `shared_expert_intermediate_size: 512`, 164 tensors | the MoE block has no shared-expert branch (routed-only) |
| experts | 256 routed, top-8, moe_int 512 | would need new keyed harvests — moot given the rows above |
| wrapper | multimodal (`model.language_model.*` paths, vision tower), plus an `mtp.*` head (19 tensors) | mechanical prep work — also moot |

The G of the full-attention layers is 16/2 = 8, which the existing `attn_mfma16_g8` import serves — but only 10 of
40 layers are full attention, so there is no partial path.

What running this model would require, in order of size: (1) a linear-attention tile family (short conv + gated
delta-rule recurrence) **and a persistent cross-step state class in the runtime**; (2) a shared-expert branch in
the MoE block; (3) output gating on full-attention layers; (4) multimodal path remapping and the MTP strip in prep
(mechanical); (5) E=256 keyed harvests (mechanical). Items 1–3 are new compute structures, not shape facts — the
config-key derivations, shape-keyed exports and grid fixes from rounds 38–42 cover none of them and all of them
are outside what rounds 38–42 established generality FOR.

Verdict for the record: the generality claim holds **within the architecture class** (GQA attention + routed-MoE
decoder); Qwen3.6's generation moved the class (hybrid linear attention, shared experts). The refusal happened at
step 0 with a named reason — which is the behaviour rounds 38 and 28 built (DeepSeek-V2-Lite's MLA and the mixed
dense/MoE layout are refused the same way), and the opposite of the silent-wrong-answer failures this document
exists to prevent.
