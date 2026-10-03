# Generality, measured: a second MoE through the ETX path (2026-10-02)

The goal is end-to-end Mixtral-8x7B execution through the imported-kernel path, with top-2 of 8 experts and GQA G=4.
This tests the TOPK < NX unpinned MoE mapping from commit 6012b5b, which previously had only a plan.
DeepSeek-V2-Lite was the first rejected candidate. Its MLA projections do not fit this path's GQA parameters.
The config now explicitly rejects its `first_k_dense_replace` mixed layout instead of an incorrect plan. See examples/llm/model.py.

## How far the ladder got

| step | result |
|---|---|
| config + local plan | passes: 58307 tasks / 452 events, MoE and top-k unpinned (TOPK % NX != 0), attnprep correctly derived OFF (no q/k norm) |
| container device build | passes: megakernel VGPR 128, spill 0, LDS 41384<br> attn_mfma4_g4 (G=4) imports cleanly |
| weights + prep + HF golden | passes: 93.41 GB packed, manifest TOPK=2 MI=14336 norm_topk=1 |
| `--check` vs vLLM originals | **every imported launch byte-identical for layers 0-1** |
| vLLM itself vs the HF golden | 32/32 (8.662 ms/token on this box) |
| ETX tokens vs the HF golden | **0/32 — the remaining failure, root-caused below, not yet fixed** |

## The finding: the imported path is shape-keyed at three levels

An imported artefact belongs to one operator and one shape.
Shape dependencies occur at three levels. Only the first level fails with an explicit error:

1. **Export selection** (loud). `wvsplitk_y1` missing for a non-Qwen MoE because the check script re-derived MoE
   detection from `num_experts` alone — the round-28 bug class in a second copy. Fixed: the check script now asks
   `load_dims`, the one derivation. The silu block size `MI/8` exceeded the 1024-thread workgroup at Mixtral's
   expert width (1792) — capped. vLLM's act kernels stride, so the block is shape, not correctness.
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

The plan, runtime, events, and placement required zero compiler changes.
The TOPK<NX branch, G=4 attention import, and megakernel execution worked without changes.
All edits concerned shape facts in the model/data layer.

These edits include the Mixtral config and `norm_topk` for `model_type=mixtral`, which always renormalises and has no config key.
They also include weight names in prep, `block_sparse_moe` and `w1/w3/w2`, plus the two script fixes and manifest entries above.
The open compiler-adjacent item is the per-shape Triton capture flow at level 3.

The `K <= 4096` guard in host.hip prevents Mixtral from using the DSL GEMV path.
Mixtral moe_down has K = MI = 14336, but the LDS staging buffer holds K floats.
A variant that stages K chunks would permit this path. This option is recorded but untested.

Timing context only: **ETX's output is still wrong, so these figures are not a performance comparison.**
vLLM takes 8.662 ms/token. ETX runs with wrong outputs took ~12.7.
The e128 top-k alone cost ~5 ms. Runtime-E DSL top-k changed 12.9 → 7.5.

## Round 39: the Triton harvest finished — kernels faithful, one dispatch-glue bug remains

The Mixtral-shape `fused_moe_up`/`fused_moe_down` now come from vLLM's Triton cache.
One eager vLLM run with a mounted `TRITON_CACHE_DIR` compiled decode at the actual M=1.
Classification used `arith.mulf`/`topk_weights` in `.ttir`: mulf=0 means up, and mulf=1 means down with `MUL_ROUTED_WEIGHT`.
Block constants also matched vLLM's tuned `E=8,N={14336,4096}` configs.

The committed shape-keyed entries are `fused_moe_up_e8` / `fused_moe_down_e8`.
Exports remain `fused_moe_{up,down}` for Qwen at E=128, byte-stable, and use `_e{E}` otherwise.
This matches the `topk_softmax_e{E}` / `moe_sum_k{TOPK}` pattern.

**Every imported launch is byte-identical to vLLM's original.**
`--check 32` reports 0 differences among 384 launches across all 32 layers.
At position 0, the first wrong token changes from EOS, id 2, to 6495 against HF 28711.
The kernels now use the correct shape. The chained output remains wrong.

Because `--check` validates each kernel on reference I/O and finds all faithful, the remaining bug is NOT a kernel —
it is ETX's hand-built MoE **dispatch metadata** for top-k ≠ 8: `num_tokens_post_padded`/`sorted_token_ids`/
`expert_ids` wired at `host.hip:216,405-409` with `ntpp_val = TOPK * MOE_BM` and `ti_r` as the sorted ids. This was
written for Qwen's top-8 single-token layout; at top-2 the per-block expert assignment the Triton `fused_moe_kernel`
reads is not what ETX hands it. **Operator: moe_up/moe_down dispatch glue (not the kernels). Layer: 0 onward.
Magnitude: argmax wrong at the first decode step (6495 vs 28711).** This is a host/compiler change, not a cache
harvest, so it is the next round's item — the mechanical harvest this round owned is complete and correct.

Zero-compiler-change verdict, updated: the plan, events, placement and every imported kernel generalise to Mixtral
with no compiler change. The one remaining compiler-side item is the MoE dispatch metadata for top-k ≠ 8.

## Round 40: the top-k≠8 blocker localised to MoE dispatch *values*, not reached to 32/32

The directive was to fix ETX's MoE dispatch metadata for top-k≠8.
This round narrowed the diagnosis but did not resolve it.
No code changed. Qwen is unchanged, and the box plan remains 60362/436.

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
change. The one open item is the MoE dispatch *values* (not the layout, not the kernels) for top-k ≠ 8.

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

`ETX_VL_DSL_TOPK=1` confirms this result experimentally.
This top-k takes E/TOPK at runtime instead of constants in a compiled kernel.
The divergence stays bit-identical: 6495 at token 0.
Thus, routing/top-k is not the cause.

**What is left, after eliminating config, argument order, replicas, kernel fidelity, MoE dispatch and routing:**
the pieces that differ Qwen↔Mixtral and that `--check` cannot validate because it feeds both sides ETX's args —
the **G=4 attention path** (`attn_mfma4_g4`/`attn_reduce`, which no end-to-end-correct ETX model has exercised;
every shipped model is G=8) and ETX's own non-imported tiles (embedding gather, argmax). The divergence is at
generated token 0 with argmax 6495 vs 28711.

**Next experiment, the directive's final step:** dump ETX's layer-0 output activation.
Compare it with a torch forward from the HF weights on disk.
This separates attention, MLP, and embedding without a vLLM reference.
The round stops here as directed, with attention and ETX tiles as the suspects.
No code changed because the MoE dispatch was correct. Qwen is unchanged.

## Round 42: two scalar bugs named and fixed — Mixtral-8x7B reaches 32/32

The layer-0 activation comparison clears the main suspect.
At position 0, the BOS of a 1-layer build, the `ETX_DUMP` stage dump in host.hip matches a torch forward from HF weights.
hn/qkv/attn_out agree within bf16 noise, with corr 1.00000.
Positions 1 and 300 also match through the full stage chain.
That chain includes neox rope with theta 1e6, G=4 attention over 2 and 301 keys, router, and top-k.

**attn_mfma4_g4/attn_reduce are correct.**
The imported path leaves stale `o`/`dn` dump buffers.
rlog after o_proj matched although `o` did not. This is a dump artifact, not a compute bug.

What the bisect actually named, in order:

**Bug 1 — fused_moe grid versus the harvested binary's BLOCK_SIZE_N.**
mc1 contained writes only in slot 0's gate half. Slot 1 and every up column contained zeros.
Thus, mc2 = silu(g)·0 = 0 was legitimate, and the whole MoE output was ~0.
The model reduced to attention-only execution.

`MOE_BN = 64` comes from the Qwen harvest's BLOCK_SIZE_N.
Mixtral's e8 ttir uses BM=16 BK=256 and **BN=16**.
The grid `TOPK*ceil(2*MI/64)` launched only 896 of the required 3584 blocks.
`--check` cannot detect a wrong grid because both kernels receive ETX's grid, just as both receive its argument values.
The fix keys MOE_BN by shape in model.py, as the kernel name already does: `MOE_BN_E8 = 16`.

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

Two optional diagnostics now exist: `ETX_DUMP`/`ETX_DUMP_STEP` for a stage-buffer dump after one step, and `ETX_TOKTRACE` for per-step argmax.
The latter covers the whole teacher-forced prompt.
Both use environment flags and cost nothing when unset.

GENERALITY VERDICT: the second MoE model reaches exact HF-greedy agreement through the unchanged ETX compiler.
The plan, events, placement, and importer need no changes.
The model differs in expert count, 8 vs 128, and top-k, 2 vs 8.
It also differs in GQA ratio, G=4 vs G=8, has untied embeddings, and uses vocab 32000.

Every model-specific failure came from a shape-dependent host scalar.
These were baked Triton constexprs in rounds 38-39, launch grids in bug 1, and a grid-versus-stride constant in bug 2.
The last was latent in every model, but vocabulary position exposed it.
The megakernel mechanism needed no changes.

Timed runs (3 interleaved rounds, context 1024, batch 1, all runs 32/32 vs HF greedy; logs results_2026-10-01/round42/):

| config                      | per-token median (3 rounds)        |
|-----------------------------|------------------------------------|
| ETX megakernel (imported)   | 16.719 / 16.756 / 16.826 ms        |
| vLLM 0.27.1 default         |  8.681 /  8.697 /  8.711 ms        |
| vLLM 0.27.1 AITER           | 12.488 / 12.499 / 12.525 ms        |

ETX Mixtral is ~1.9x slower than vLLM default, as expected for this all-imported mode.
The Qwen equivalent was 6.435 vs 4.937 in round 24.
This run retains Qwen-tuned CU/grid settings, without DSL-tile or fusion optimizations for the new shape.

This document's generality claim concerns compiler-path correctness on a second model. That claim is now established.
Mixtral performance is a separate, unstarted optimization sequence.
One decode trace exists at round42/trace_r42_mixtral.{bin,log}, with a median of 16.751 ms on-device.

## Round 44: Qwen3.6-35B-A3B — refused at the config, correctly

The third candidate is `Qwen/Qwen3.6-35B-A3B`, model_type `qwen3_5_moe`, with 71.9 GB bf16 weights.
It was examined but **not attempted**.
It belongs to a different architecture class, rather than a shape variant of this path's family.
The decision used only `config.json` and the checkpoint index, with no download or GPU time.

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


## Round 43: Mixtral on the DSL path

The DSL tiles now support K > 4096 through chunked staging of x.
The path is numerically exact and passes CHECK-DSL at layer 31.
Mixtral reaches **10.365 ms/token against 17.058 all-imported, a 1.65x speedup**.
vLLM takes **8.686, which leaves ETX 1.19x behind**.
The stream floor is ~7.3-8.5 ms, where vLLM operates.

The trace attributes the gap to Qwen-tuned MoE task widths, not the chunked path.
moe_up has 896 tasks on 304 workers, which requires three serial waves.
moe_down has 128 tasks and leaves 58% of workers idle.
MoE accounts for 73% of the layer. Per-shape task widths are the next control to evaluate.

An interleaved check caught a Qwen regression of +0.12 ms from the resident big-K branch.
The feature manifest now selects `dsl_bigk` from max(H, MI) > 4096.
With the feature off, Qwen's emitted source is token-identical.


## Round 46: Phi-3.5-MoE — refused at the config; the class boundary is sharper than we knew

Shape facts fit (E=16, TOPK=2, G=4, no shared expert; MI=6400 would exercise the big-K DSL path), but the
refusal is decisive and named: **PhiMoE routes with `sparsemixer`, not softmax-topk** — per selection round,
mask logits where (max−s)/max(|s|,max) > 2·0.01, softmax the masked row, gather at the argmax; the ids equal
plain top-2 but the weights are never renormalized across the pair. Even vLLM runs it as Python
(`phimoe.py:175,283`), so there is nothing to import, and both of our top-k ops compute softmax-topk: the
build would pass every structural gate and generate wrong weights — the silent class this document exists to
prevent. Secondary, tractable items if a sparsemixer op is ever written (an afternoon in `etx/ops/topk.py`;
deterministic at inference): the LongRoPE cos/sin cache (additive branch in `prep.py:28`, kernel reusable) and
o_proj/lm_head bias in the gemv epilogue (qkv bias exists on the import path; the rest is unverified).

**Corrected claim:** the class this week validated is "GQA attention + *softmax-topk* routed MoE" — one notch
narrower than "routed MoE". No download, no GPU time was spent. The refusal cost one config read.

## Round 47: PhiMoE support — three of four hard features land; moe_up OOB localized

Phi-3.5-MoE from microsoft/Phi-3.5-MoE-instruct tests four structures that no prior model used.
Its shape is H=4096, 32L, G=4, E=16, top-2, MI=6400.
It uses LayerNorm, qkv/o_proj/lm_head biases, LongRoPE, and sparsemixer routing.
The new model-local code uses the `dsl_ext` feature manifest.
Qwen's plan is byte-identical, and its GPU run remains 32/32 + CHECK-DSL with wall time 4.851:

1. **sparsemixer routing** — `dsl_topk` mode 2 (etx/ops/topk.py): per-round masked softmax, ids = plain top-2,
   weights the per-round masked softmax at the winner. Cross-checked against vLLM's python on 2000 random rows
   (E=8 and 16): identical ids, weights to 1e-6. Reference added to --check-dsl.
2. **LayerNorm fold (mode 4)** — mean-centred + biased, in dsl_stage_norm_at, both VECNORM and scalar.
3. **Output bias** — o_proj/lm_head/folded-qkv, in the gemv epilogue and the fold; two trailing arg pointers
   (bias, LayerNorm bias) appended only when ETX_VL_DSL_EXT sized the layout, asserted to agree with the plan's
   own config derivation (the round-14/19/30 drift class).
4. **LongRoPE** — longrope cos/sin cache in prep.py with the config's explicit short/long mscale.

The import arena previously took its size from the first tile file to include etx/import.h.
For Qwen, imp_tiles.hip came first because layer 0 uses imported rms_norm.
For a model that folds layer 0, dsl_tiles.hip came first.
That folded-layer-0 model received the default 16-byte arena.
emit.py now defines ETX_IMPORT_LDS before the include, with an #ifndef guard and the imports.json maximum.
The size no longer depends on include order.

**Verified correct via --check-dsl** (relaunch each op, 0 beyond 2 ulps): qkv (folded, LayerNorm mode 4, R=0),
o_proj (bias), router (folded, LayerNorm). So the LayerNorm fold, the bias epilogue, and sparsemixer-fed routing
are correct end to end.

**Open, single remaining item:** `dsl_moe_up` faults with a memory aperture violation — localized precisely
(qkv/o_proj/router pass, moe_up is the first expert-indexed op to run), XCD-independent (crashes with XCD off
too, so not the unpinned slot mapping), expert weight shapes verified (w1/w3 [6400,4096], w2 [4096,6400], estride
2·MI·H correct). Mixtral's DSL moe_up (K=4096, same path) ran; Phi differs in E (16 vs 8) and MI (6400 vs 14336),
neither of which the tile's mode-0 path uses for addressing. The fault is in the MoE expert GEMV for this shape
and needs a focused pass, not a shared-box rebuild loop. Everything that passed is committed and gated.

## Round 48: the PhiMoE "moe_up OOB" was a missing barrier in the new LayerNorm fold — all three gates green

The round-47 aperture violation required a live step.
Synthetic zero-weight 2-layer shapes with sparse vl.bin and no downloads cleared E=16, MI=6400, and the exact Phi shape.
A real-weight 2-layer model used the full vl.bin through an L=2 manifest and cleared values at rest.

The new ETX_UNFUSED_SYNC mode adds one synchronization per unfused launch and identified moe_up_0.
An emit-gated expert-id guard showed every MoE task in every step read expert id 0x7fffffff.
That initial top-k argmax value survives only when all router logits are NaN.
ETX_DUMP after step 0 of an L=1 build located the first NaN in the layer-0 qkv fold output.
The input embedding row was clean. The same operator was bit-correct when relaunched alone.

The cause was a race in the new LayerNorm fold, mode 4.
Every thread writes part of staged xs[], but the mean pass reads the whole array without a barrier.
The RMS fold needed no such barrier because each thread reduced only its private partial as it staged data.
That precedent made the new pattern look safe.

At full occupancy, the mean pass read unwritten LDS, whose junk bits could produce NaN.
An isolated kernel won the race, so --check-dsl relaunches passed while the fused step failed.
NaN logits also made vl_argmax emit 0x7fffffff as a token id.
The next prologue then read embed[2^31-1], which made the crash location vary.

One __syncthreads() before the mean pass fixes both VECNORM and scalar paths.
A clamp on the sparsemixer winner index also prevents all-NaN logits from producing a wild weight pointer.

**Phi-3.5-MoE end to end:** MI300X, context 1024, bs 1, and 32/32 tokens against HF greedy.
CHECK-DSL reports 0 of 31248 listed output values beyond 2 bf16 ulps at layer 31.
The operators include LayerNorm-folded qkv/router, biased o_proj, sparsemixer top-k, and both MoE stages.
CHECK-COVERAGE reports 1055 positions and 0 wrong beyond ties.

Three interleaved timed runs give ETX wall times of 6.459 / 6.481 / 6.466 ms/token.
The median on-device time is 6.34.
vLLM v0.27.1 gives 11.035 / 11.078 / 11.057 with default settings and 11.398 / 11.335 / 11.360 with AITER.
ETX is 1.71x faster.
vLLM's greedy output also matches the HF golden 32/32, so both systems process identical tokens.

The --check-dsl fold branch previously recomputed softmax-topk weights for a sparsemixer model.
Its moe_down reference differed by up to 14x while the device result was correct.
The reference now uses the model's routing function.
