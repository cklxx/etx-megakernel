# What the mono-decode PRs would cost us, and what to build first, 2026-10-19

Inputs: [vllm-project/vllm#60397][pr] (DeepSeek-V4.1-Flash mono decode),
[#60216][glm] (GLM-5 mono decode), [#59705][mm] (MiniMax-M3 mono decode), and
this repo's measurements. PR claims cite the PR. gfx942 port facts come from a
fork commit *referencing* #60397 (`maeehart/vllm` `a3569e0`), not from the PR
text, and are labelled **[F]**. Everything else is our estimate [E] or
unmeasured [U].

Size classes for code we would have to write: **XS** < 100 lines, **S** 100–300,
**M** 300–600, **L** > 600. All sizes are [E] unless a PR states its own count.

## 0. The candidate on MI300X: Qwen, not GLM

**Memory.** One MI300X has 192 GiB of HBM. The gfx942 port computes on its own
copies of the dense weights, because gfx942 has no scaled FP4 or FP8 MFMA and
uses FNUZ FP8 [F]. The arithmetic:

- **Qwen3-30B-A3B, bf16:** 30.5e9 parameters × 2 B = 61 GB = 56.8 GiB ≈ 57 GiB.
  57 GiB of weights plus its copies fits inside 192 GiB with room to spare.
- **GLM-5.3:** 305 GiB of checkpoint payload
  (`docs/TWO-WEEK-RESULTS-2026-10-18.md:74`) against 192 GiB of HBM — 305 − 192
  = **113 GiB over one card before any copy is made**. GLM-5.3 is out on one
  card, plainly.

So the MI300X mono-decode candidate is **Qwen3-30B-A3B**, and every direction
ranked below is a Qwen direction. GLM stays a two-card PP2 question
(`examples/glm53/`), which none of the three PRs covers.

**Rows and passes.** The gfx942 port takes decode steps of up to **42 rows** at
TP4 (7 requests × (1 + 5 DSpark drafts) = 42) [F], and one GEMV pass covers
**6 tokens** instead of the gfx950 kernels' 16, because a gfx942 workgroup has
64 KB of LDS [F]. Our target step is B8 = **8 rows**, one token a request:
8 ≤ 42, so **B8 is inside the cap**, and a batched dense GEMV of 8 columns is
ceil(8/6) = **2 passes** of the gfx942 pass width.

**Per-layer budgets do not transfer.** The gfx950 per-layer figures we have
(89 µs KDA, 110 µs sparse) are gfx950 numbers. With 64 KB of LDS the gfx942
GEMV pass is 6 tokens wide, so a per-layer budget taken from them must be
scaled for the narrower pass or marked [U] on gfx942. They are not used in the
ranking below.

## 1. Model-specific work, per PR

| PR | What we would have to write in our repo | What we reuse | Size |
|---|---|---|---|
| **#60397** DeepSeek-V4.1-Flash (40 layers, 30 eligible) | A model path we do not have. K1 + K2 composition for the 30 standard layers, the single FFN launch for the other 10, the mHC seams, MXFP8 attention linears and AITER MXFP4 expert reads, plus 1..48-row eligibility | The plan compiler (`etx/passes/plan.py`) and the launcher pattern; the stages themselves are ATOM's, MIT, commit `97359d6` [P] (PR, *Layout*) | **L** [E]; the PR states no line count [U] |
| **#60216** GLM-5 (78 layers) | A per-layer composition in `examples/glm53/model.py`: dense MLP packed once as six expert-shaped MXFP4 slices, W_UK/W_UV cut from `kv_b_proj`, the indexer reading and writing the paged FP8 index cache and writing top-2048 slots, the residual stream carried through every layer, one epoch advance per step [P] (GLM PR, *Details*) | Our GLM plan (`examples/glm53/model_build.py`) and the PP2 boundary hand-off; the kernel is an AITER dependency, `aiter.ops.flydsl.glm5_mono` [P] (GLM PR, *Overview*) | **M** [E]; the GLM PR's own vLLM integration is 545 lines outside tests, almost all in one file, plus an 11-line hook and one env var [P] |
| **#59705** MiniMax-M3 (60 layers, 57 sparse) | Nothing we need: no MiniMax path exists here. If we ever target it: one fused launch per sparse layer for 1..16 tokens, TP4, no PP/DP [P] (MiniMax PR, *Summary*) | — | **L** [E]; the PR states no line count [U] |

The GLM row is a two-card port by §0's arithmetic; on one MI300X only the Qwen
path is live, and the Qwen path is not described by any of the three PRs [U].

## 2. Shared runtime pieces, same size classes

These are model-independent: the same five pieces serve any PR we port.

| Piece | What the PRs say | Where it goes in our repo | Size |
|---|---|---|---|
| Per-step routing decision | A plain Python decision per step, taken when a FULL graph was captured and on every eager step; PIECEWISE captures always declined, because they replay mixed batches [P] (PR, *Design*) | Host-side plan selection: `examples/vllm_llm/host.hip`, `examples/glm53/model_host.hip` | **XS–S** [E] |
| Epoch-tagged hand-off | Epoch tags on the hand-offs replace per-step mailbox clears, so no host step hook is needed and the launches are graph-safe; a given step width's words only ever hold tags [P] (PR, *Design*, *Layout*) | `etx/runtime/include/etx/inbody.h` and the plan lowering (`etx/passes/plan.py`) | **S** [E] |
| Per-width scratch | One scratch per step width, 3–16 MB, at most 8 widths under speculation [P] (PR, *Design*) | Host allocator in `examples/glm53/model_host.hip`, `examples/vllm_llm/host.hip` | **XS** [E] |
| Bounded waits with error words | Not described by any of the three PRs [U]. Our host already bounds a launch at 120 s (`examples/glm53/model_host.hip:75`); a device-side wait that times out and records an error word is new | `etx/runtime/include/etx/inbody.h` | **S** [E] |
| Fail-stop | An explicit opt-in that cannot run raises, listing every reason [P] (PR, *Failure modes*) | Startup checks in `model_host.hip` / `host.hip` | **XS** [E] |

Shared total: roughly **400–1,000 lines** [E], paid once regardless of model.

## 3. The port order, then the directions ranked by measured ms

**Construction order:** weight-copy layout, then GEMV pass width, then the
seams.

1. **Weight-copy layout** — the gfx942 kernels read their own copies of the
   dense weights, made when the weights are processed at load [F]. Build this
   first: every later stage reads the layout it fixes, and the 192 GiB budget in
   §0 is decided here. Size **S–M** [E].
2. **GEMV pass width** — one pass covers 6 tokens [F]; a B8 dense GEMV is two
   passes. This is the tile's column width, `etx/ops/gemv.py` (`DSL_NB`) fed by
   `examples/vllm_llm/model.py:202-203`. Size **XS–S** [E].
3. **The seams** — attention reduce, the all-reduce, the FFN seam and the MoE
   seam move inside the per-layer launch [P] (PR, *Purpose*). In our repo this
   is `etx/importer/spec.py` (attn_reduce) and the emission in
   `examples/vllm_llm/model.py`. Size **M** [E].

The order is construction order, not value order. The value ranking below is by
the milliseconds our own MI300X measurements put on the table; all of it is
inside the port's caps (B8 = 8 rows < 42; 8 columns = 2 passes of 6).

1. **E1 — column-batch the dense GEMVs.** **−4.84 ms at B8, −2.07 ms at B4**
   [E] from measured byte deltas (19.6755 → 2.4594 GB at B8, 9.8377 → 2.4594 GB
   at B4) at the measured 3.56 TB/s. Code: `examples/vllm_llm/model.py:202-203`
   (`BATCH`, `BATCH_ROLES`) and `model.py:541`; the tile is `etx/ops/gemv.py`.
   Falsifier: a rocprof `FETCH_SIZE` capture shows dense bytes/step did not
   fall ~Nx, or the paired step improves < 2%.
2. **E3 — fix the imported `attn_reduce` scaling.** **−1.5 to −2.0 ms at B8**
   [E], from the measured +2,473 µs of the +2,484 µs imported excess at position
   0 (per-launch median 6.2 → 12.7 µs, B8). Code:
   `etx/importer/spec.py:82,244,282` and the call site
   `examples/vllm_llm/model.py:362`. Falsifier: the B8 per-launch median does
   not fall back toward 6.2 µs, or the step improves < 2%.
3. **E2 — column-batch the MoE to the routed union.** **−1.58 ms at B8,
   −0.37 ms at B4** [E] (64 → 51.6 distinct experts at B8). Code:
   the MoE instance emission in `examples/vllm_llm/model.py` (`XAFF_ROLES`,
   `model.py:193`) and the fold guard `examples/vllm_llm/host.hip:495`.
   Falsifier: expert bytes/step do not approach the union bound, or the step
   improves < 2%.
4. **E7 — hierarchical sentinel polling for the relay (independent).** Ceiling
   **≈ 0.464 ms at B4, capped under 1 ms** [E], from the measured 963 µs
   dependency gap and the measured 8.6 µs whole-GPU barrier at 304 workgroups.
   Code: the relay barrier in the runtime plus a trace counter before any code.
   Falsifier: the counted barrier ceiling is below 0.9 ms — then write no code.

The 89 µs KDA and 110 µs sparse gfx950 budgets are not part of any rank: their
gfx942 values need the 6-token pass width applied and are [U] until measured.

## 4. Three things we should not build

1. **E6, the alternative slot map.** Its 0.366 ms queue gap is 1/9 of the
   3.37 ms target and was measured and judged irrelevant; the adversarial
   review deleted it. Our measurement kills it.
2. **E5, B1 weight prefetch.** The premise is measured-negative: the router/top-k
   output must be known before the expert weights are read, so that serial hop
   cannot be prefetched. Our measurement kills it.
3. **A GLM-5 mono-decode layer for gfx950.** #60216 already ships it (545 lines
   outside tests [P]), so writing our own duplicates a PR. The gfx942 / MI300X
   variant is **not** covered by any of the three PRs — it is [U], and §0 says
   it cannot fit one card anyway; if it is ever built, it is a new two-card
   kernel, not this work.

[pr]: https://github.com/vllm-project/vllm/pull/60397
[glm]: https://github.com/vllm-project/vllm/pull/60216
[mm]: https://github.com/vllm-project/vllm/pull/59705
