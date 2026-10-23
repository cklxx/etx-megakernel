# vLLM baseline for GLM-5.3-Flash on MI325X: feasibility before Phase 2 — 2026-10-23

**Preparation only.** Nothing here was run. Phase 2 measures the vLLM baseline first (B1/2/4/8, same
box, same model); until that baseline exists no throughput multiple is claimed, and nothing in Phase 2 is
measured before Phase 1 has a verdict. This file records what public sources say about whether the
baseline can run at all, so a GPU session does not discover it on paid time.

## What the sources say (read 2026-10-23, not reproduced)

- GLM-5.3-Flash is `Glm5NextForConditionalGeneration` (34 KDA + 11 sparse-MLA layers, native FP8).
- **Upstream vLLM on gfx942 is not shown to run this model.**
  - The GLM-5.3-Flash support branch refuses ROCm at import ("currently supports NVIDIA SM90 and above
    only"); with the guard removed, the ROCm sparse indexer `forward_hip()` drops `gate_score` and the
    kpool arguments. Open, no upstream fix linked
    ([vllm#53943](https://github.com/vllm-project/vllm/issues/53943)).
  - A fork branch (`ZJY0516/vllm@glm-release`, PRs ZJY0516#5/#7) claims gfx942/gfx950 support, with
    GSM8K 5-shot 0.9704 on 8× MI300X; the reporter saw non-deterministic output at temperature 0 for
    long prompts on gfx950. AITER changes (ROCm/aiter#4919, #5049) were not upstream at the time.
  - vLLM v0.30.0 on gfx942 cannot boot GLM-5.3-Flash: `ROCMAiterMLASparseImpl` lacks
    `record_logical_topk_ready`. The fix (#57252, merged 2026-09-17) is in v0.30.1rc0/main, not in
    v0.30.0. Workaround on 8× MI308X TP=4: a no-op method plus `--kernel-config '{"enable_jit_warmup": false}'`;
    long context needed further open patches (#56381, #56584)
    ([vllm#59027](https://github.com/vllm-project/vllm/issues/59027)).
  - The same v0.30.0 crash on 8× MI325X TP=8 is reported for GLM-5.3 (`GlmMoeDsa`, not Flash), with the
    same no-op workaround; no decode latency or throughput numbers at B1/2/4/8
    ([vllm#58575](https://github.com/vllm-project/vllm/issues/58575)).
- The ROCm ATOM image does not register the architecture
  ([ROCm/ATOM#2316](https://github.com/ROCm/ATOM/issues/2316)); support is in review, tested only on
  MI355X ([ROCm/ATOM#2481](https://github.com/ROCm/ATOM/pull/2481)).
- SGLang is reported to have merged GLM-5.3-Flash recipes for MI300X/MI325X/MI355X (sgl-project/sglang#36608,
  cited in vllm#53943); not checked here.

## Consequences for the Phase 2 session (to decide before provisioning)

1. **The baseline may not exist as stock vLLM on gfx942.** The candidates are vLLM main or ≥ v0.30.1rc0
   (fix #57252 included), or the fork branch; each needs its exact version recorded and its own
   HF-greedy correctness check before its throughput counts as a baseline.
2. **"Strongest legal configuration" needs a search**, as in the Qwen sessions: attention backend,
   AITER MoE on/off, KV-cache dtype, TP8 versus DP×TP. An untested configuration is not the strongest.
3. **If no vLLM build runs the model correctly on the box**, the 10× target has no denominator; report
   that as NOT-COMPLETE rather than substitute another system. Whether SGLang may stand in is the
   owner's decision.
4. **Offline check before booking:** pull the chosen image or wheel and confirm it imports and
   registers `Glm5NextForConditionalGeneration` with ROCm (no GPU needed for the import guard); this
   decides item 1 at no GPU cost.
