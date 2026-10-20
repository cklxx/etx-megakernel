# vLLM mono decode (#60397) against our E1–E3 and the GLM plan, 2026-10-19

Upstream source for every claim marked (PR): [vllm-project/vllm#60397][pr],
"[ROCm][Perf][DSv4.1] Mono decode layer for DeepSeek-V4.1-Flash on MI355X"
(Fangzhou-Ai, opened 2026-10-07). Anything the PR does not state is [U] and
carries no number. Our own code is cited by file and line.

Labels: [P] stated in the PR. [M]/[E] this repo's measured/estimated numbers.
[U] not in the PR (or not measured here).

## 1. What the PR changes

- **One switch, one launch per layer.** Opt-in `VLLM_ROCM_MONO_DECODE=1`.
  Every backbone layer of a decode step runs on persistent FlyDSL launches [P]
  (PR, *Purpose*): K1 covers the mHC attention seam, wqkv, q/kv norms, the KV
  insert and wq_b; K2 covers the attention back (split attention, combine,
  wo_a, wo_b), the attention TP all-reduce, the FFN seam, the MoE (router,
  top-6, shared expert, MXFP4 experts) and its TP all-reduce. K1 + K2 replace
  the ~40 kernels the current path launches per standard layer [P].
- **Two layer classes.** K1 + K2 on 30 layers (3–7, 9–13, 15–19, 21–23,
  25–27, 29–31, 33–35, 37–39); a single FFN launch on the other 10 (layer 0,
  the Engram layers 1/14, and the compressor/indexer layers 2/8/20/24/28/32/36)
  [P] (PR, *Scope*). The FFN launch starts from wo_b's unreduced output [P].
- **All-reduces move inside the kernels.** Both TP all-reduces run in-kernel
  over symmetric peer memory, exact fp32 sum in rank order then bf16 [P]
  (PR, *Purpose*, *Design*).
- **No second weight copy.** Attention linears stay vLLM's row-major MXFP8,
  routed experts stay in AITER's shuffled MXFP4 layout [P] (PR, *Design*).
- **Conditions.** Decode only, ≤48 rows (8 requests × 1 + 5 DSpark drafts),
  TP2 or TP4, eager or a FULL CUDA graph [P] (PR, *Scope*). The path declines
  PIECEWISE captures, prefill, mixed and larger steps [P].
- **Refusals.** CDNA version outside the kernels' (`get_cdna_version()` in
  `CDNA_VERSIONS`, CDNA4 / gfx950 today), TP other than 2 or 4, FlyDSL missing,
  expert/data parallelism or EPLB, `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4`, a
  routed-expert backend other than `AITER_MXFP4_BF16`, or fewer than 256
  compute units [P] (PR, *Failure modes*).
- **Graph safety.** Epoch-tagged hand-offs instead of per-step mailbox clears,
  and one scratch per step width (3–16 MB, at most 8 widths) [P] (PR, *Design*).
- **Provenance.** `stages/` and `common/` are adapted from ROCm/ATOM's
  DeepSeek-V4.1 mono decode (MIT, commit `97359d6`) [P] (PR, *Layout*).

## 2. The PR's measured numbers [P]

All from the PR's *Test Result* and *Provenance* sections; conditions are the
PR's own (MI355X, gfx950, real checkpoint weights, cold weights, launches
chained in one HIP graph unless stated otherwise).

| Measurement | Result |
|---|---|
| Kernel level, µs per layer, 96 points (TP2/TP4 × concurrency 1–8 × 1–6 tokens per request × two layer types) | every point faster, **1.10–2.18×**; residual cosine ≥ 0.99999; MoE output 1,234/1,260 tokens at cosine ≥ 0.999, the other 26 are top-6 near-ties (6th/7th score gap ≤ 3.2e-3) |
| FFN launch, µs per layer, 6 tokens per request | TP2 **1.01–2.14×**, TP4 **1.12–2.32×** (e.g. TP2 c1 136→81 µs, TP4 c1 111→57 µs) |
| Numerics test (GPU) | cosine ≥ 0.9993 vs vLLM's chain, relative error 0.3–0.5% (bounds 1–1.5%), 300 graph replays bit-exact; race stress 100K replays with no mismatch |
| GSM8K, 5-shot, 1,319 questions, concurrency 8 | current path 0.9742 ± 0.0044 strict; K1+K2 0.9750 at TP2, 0.9719 at TP4; with the FFN launch 0.9735 / 0.9719 |
| Accuracy vs `VLLM_ROCM_MONO_DECODE=0` (thinking on, 8 requests in flight) | GPQA-Diamond 85.3 → 85.1 (Δ −0.2 ± 0.6), AIME25 99.4 → 98.3 (Δ −1.0 ± 0.7) |
| RULER NIAH 32K/128K/256K, 40 samples a cell | 40/40 in every cell for both mono and the reference |
| AgentX, TP2 vs same-node mono-off | p90 interactivity **1.14–1.35×**, total throughput **1.04–1.16×** |
| AgentX, TP4 vs InferenceX's run | p90 interactivity **1.35–1.52×**, total throughput **1.04–1.15×** |
| AgentX, decode on the same requests (FFN launch vs K1+K2 alone) | **7–9% faster** at every point |

The PR states its kernel, GSM8K and AgentX numbers were taken while the kernels
were hosted in AITER (ROCm/aiter#6196), and that the per-width scratch change
leaves kernel time unchanged (24 points, mean ratio 1.000, range 0.996–1.009)
[P] (PR, *Provenance*). It does not state a merge status [U].

## 3. Mapping onto our E1–E3, and the file each would change

| Our experiment | PR component that maps | File in our repo that would have to change | Caveat |
|---|---|---|---|
| **E1** — column-batch the dense GEMVs (`ETX_VL_BATCH=1` on qkv, o_proj, router, lm_head) | The PR's GEMV stage (`vllm/models/deepseek_v41/amd/mono/stages/gemv.py` [P], PR *Layout*) inside K1/K2, reading each weight once in place for all ≤48 rows [P] | `examples/vllm_llm/model.py` — the `BATCH` / `BATCH_ROLES` emission (`model.py:202-203`, consumer at `model.py:541`); the tile that consumes `nb` is `etx/ops/gemv.py` | The PR batches rows inside one persistent launch; our E1 changes plan emission, not the tile body. Neither the PR's row count nor its per-layer gain is ours to claim |
| **E2** — column-batch the MoE to the union of the batch's routed experts | The PR's MoE stage (`stages/moe.py`, `moe_plan.py`, `router.py` [P], PR *Layout*), reading routed experts for up to 48 rows inside K2 [P] | `examples/vllm_llm/model.py` (MoE instance emission, the `XAFF_ROLES`/slot path at `model.py:193`); the fold refusal at `examples/vllm_llm/host.hip:495` | The PR text does **not** state that expert reads are deduplicated to the routed union — it only states the ≤48-row step and that experts stay in AITER's shuffled MXFP4 layout [P]. The union mechanism is [U] in the PR; E2 is still unproven upstream |
| **E3** — fix the imported `attn_reduce` scaling | K2's "attention back (split attention, combine, wo_a, wo_b)" and the in-kernel attention TP all-reduce [P] (PR, *Purpose*); the PR's `stages/seam.py` also holds "attention reduce" [P] (PR, *Layout*) | `etx/importer/spec.py` — the `attn_reduce` launch spec (`spec.py:82,244,282`); its call site is `examples/vllm_llm/model.py:362` | This is the closest one-to-one match: the PR folds the reduce into the layer launch, while our imported `attn_reduce` is a separate kernel per request. The PR gives no per-stage split for attention reduce alone [U] |

E1 and E2 are plan-shape changes; E3 is an import-shape change. The PR shows
the end state of all three at once (one launch per layer), so it is evidence
that the direction is live upstream [P], not a measurement we can borrow: its
GPU is MI355X (gfx950), its model is DeepSeek-V4.1-Flash and its rows are
≤48 — none of which matches our Qwen3-30B-A3B B4/B8 condition.

## 4. What the sibling mono-decode PRs imply for our GLM-5.3 PP2 plan

The PR's *Duplicate check* names them: "The sibling mono-decode PRs are for
other models: (MiniMax-M3) and (GLM-5)" [P]. On the rendered page the links
resolve to:

- **vllm-project/vllm#60216, "[ROCm][Perf][GLM-5] Fused sparse-layer decode via
  AITER MonoKernel"** — opt-in `VLLM_ROCM_GLM_MONO_DECODE=1`, one AITER
  MonoKernel launch per decoder layer per GPU for pure decode steps of 1, 2, 4
  or 8 tokens, TP4 or TP8, on **gfx950**; the launch includes the sparse
  indexer and both TP all-reduces; the kernel carries the full residual stream
  through all 78 layers, with the final norm on it [P] (#60216, *Overview*,
  *Details*). Its stated limitation is "Only gfx950 and GLM-5
  (`glm_moe_dsa`)", and concurrency above 8 uses the regular path [P]. The page
  shows it closed on 2026-10-12 after merge conflicts [P].
- **vllm-project/vllm#59705, "[ROCm][Perf][MiniMax-M3] Fused sparse-layer
  decode (mono decode kernel)"** — one fused launch per sparse layer per rank
  for decode steps of 1..16 tokens; build-time checks require "Tensor parallel
  size is 4", "No PP/DP", fp8 KV and index caches, and block size 128 [P]
  (#59705, *Summary*).

Three consequences for our 45-layer PP2 plan:

1. **Upstream has already chosen a different decomposition.** Both siblings are
   single-node tensor-parallel paths, and #59705 excludes pipeline parallelism
   explicitly ("No PP/DP" [P]). Nothing in either PR describes a PP path [U].
   Our plan (`examples/glm53/`, `docs/TWO-WEEK-RESULTS-2026-10-18.md:74`) is
   PP2 across two rank-0/1 halves — rank 0 layers 0–23, rank 1 layers 24–44,
   with a cross-GPU residual hand-off
   (`docs/evidence/qwen-batch-glm53/completion/report.md:34`), 305.71 GiB
   payload and 163.96 GiB per rank [E]. The GLM-5 sibling instead keeps the
   whole residual stream and every layer on each TP rank in one kernel [P],
   which a 45-layer PP2 split by construction cannot do.
2. **The GLM-5 sibling is not a drop-in baseline for our plan.** It runs TP4 or
   TP8 on gfx950 with steps of 1/2/4/8 tokens [P]; our plan is 2 GPUs, PP2,
   and our measured correctness floor is a 4-layer fixture (480/480)
   (`docs/TWO-WEEK-RESULTS-2026-10-18.md:70`). Matching its step shape would
   mean giving up PP, not tuning it [E].
3. **On MI300X there is no upstream mono-decode baseline at all.** #60397
   gates on `get_cdna_version()` in `CDNA_VERSIONS`, which is "CDNA4 / gfx950
   today" [P], and #60216 repeats "Only gfx950" [P]. MI300X is gfx942 — the
   arch every MI300X build in this repo compiles for, e.g.
   `examples/vllm_llm/build.sh:83` (`--arch gfx942`). So neither PR can run on
   the GPU we measure on, and our code cannot be compared against them there.
   Whether upstream gained a gfx942 port outside these PRs is [U] here.

**Answer to the baseline question:** our GLM PP2 code can serve as a baseline
on MI300X only against vLLM's regular (non-mono) path, which is what v0.27.1
offers on gfx942; it cannot be compared to the mono-decode PRs, because those
kernels refuse gfx942 [P] and are TP-only [P].

## 5. What the PR does not say [U]

- No MI300X / gfx942 number, and no statement that gfx942 can run the path; the
  CDNA gate names CDNA4 / gfx950 only [P].
- No Qwen3-30B-A3B or any non-DeepSeek-V4.1 measurement [U].
- No union dedup of routed-expert reads [U].
- No pipeline-parallel path or measurement [U].
- No merge status and no comparison against ATOM's own numbers at matched
  conditions beyond the provenance note [U].
- No per-stage split for attention reduce alone [U].

[pr]: https://github.com/vllm-project/vllm/pull/60397
