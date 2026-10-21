# ETX optimisations: what is a rule, what is derived, what is an experiment

Read this before a port to a new model.
All results below come from one MI300X on 2026-10-01.
They use wall ms/token, 32/32 tokens against HF greedy, and a clean `--check-dsl` result.
Sources are `../qwen/RESULTS-2026-10-01.md`, `../qwen/BUDGET-2026-10-01.md`, `../qwen/PHASES-2026-10-01.md`, `../qwen/PARITY-2026-10-01.md`, and `../qwen/WHY-DSL-TILES-2026-10-01.md`.

**A universal rule should not be a switch. A model-dependent choice should not be a constant.**
Previously, both were `ETX_VL_*` environment variables tuned for Qwen3-30B-A3B.
Each new model required a person to select them again.

## Classification

| Switch | Class | Measured | Why |
|---|---|---|---|
| `ETX_VL_COH` | **universal** | −0.317 | A tile whose activation accesses are coherent needs no `buffer_inv sc1` acquire. The invalidate costs **4.1×** on an L2-resident phase. The dependent-chain cost per hop is **10.6× on bare metal and 11.4× on the VF**. Both machines show this hardware effect. Hardware behaviour, not a model property. Imported kernels cannot have it — their loads come from vLLM's source. |
| `ETX_VL_NT` | **universal** | −0.106 | Weights are read once per token and have no reuse, yet they evict what does (the MALL-resident KV cache). True of every decode model. |
| `ETX_VL_VECNORM` | **universal** | −0.105 | A coherent access bypasses L1/L2, so a 2-byte accessor issues one memory request *per lane per element*. The rule is "coherent accessors as wide as the data allows" — never a choice. |
| `ETX_VL_DSL_GEMV`, `_TOPK`, `_ATTN=head` | **universal** | +1.13 ms if off | All-imported measured 6.435 against 4.937, and with imported kernels fusion still *loses* (−0.171). The tiles earn their place by being **editable**, not faster — on kernel speed the two sets tie (+0.043 ms, inside noise). |
| `ETX_VL_XCD`, `ETX_VL_CHAIN` | **universal** | in every good build | XCD-affine replicas and chaining one-block launches<br> structural, not model-specific. |
| import prune | **universal** | −0.015 | `ETX_IMPORT_LDS` is the max over *every* imported kernel, used or not, and the megakernel carries that arena in every phase. Purely mechanical<br> never should have been a choice. |
| host work async | **universal** | −0.152 | 20 synchronous HIP calls per step, a dead memset, a 50 µs poll sleep. Model-independent. |
| `ETX_VL_DSL_FOLD`, `ETX_VL_FOLD_TOPK` | **derived** | −0.231 | **The clearest case.** See below. |
| `ETX_VL_DSL_ATTN=head` applicability | **derived** | — | `dsl_attnprep` fuses a q/k norm, so it applies only to a model that *has* one, with G+2 heads fitting the waves. Defaulting it on unconditionally made Qwen2.5-1.5B refuse to plan. |
| `ETX_VL_CU` | **derived, not yet implemented** | −0.067 | 304 is oversubscribed: the per-workgroup rate collapses from ~31 KB/µs (≤74 workgroups) to 8.9 at 296, aggregate optimum near 192. 256 measured best for Qwen3-30B-A3B's dense shapes. Deriving it needs the rate curve as a function of bytes-per-task, which `hw_bench streamw` can measure<br> today it is a documented constant. |
| `ETX_VL_ARGPF`, `_BAR2`, `_ATTN_WIDE`, `_PREP_EARLY`, `_BATCH`, `_NOREP`, `_UC`, `ETX_DEV_RESET`, `_QKNR` | **experiment, off** | all ≥ 0 | Each measured neutral or negative with its number recorded beside the code. Do not resurrect without a new mechanism. |
| `ETX_VL_TOUCH` | **universal (unvalidated)** | — | On in every good build, but its value was never isolated *after* `ETX_VL_COH` changed what survives a phase boundary. A 20-second A/B would settle it. |

`ETX_VL_IMPORTED=1` turns every DSL tile off at once and restores the all-imported build, so round 24's comparison
stays reproducible. It is the only way to get the old defaults.

## The derived decision: which phases to fold

A fold deletes a one-block producer phase.
In exchange, every consumer task repeats the producer's work.
Whether this wins depends on the shapes, not just the phase.
In the same model, it benefited input norm but lost 0.676 ms/token for MoE top-k.
`examples/vllm_llm/folds.py` compares these costs:

```
keep_us = phase floor × (producer tasks / workers) + the producer's own body + one hand-off
fold_us = rounds × (rematerialised bytes / rate)     bandwidth: each consumer task re-reads the inputs
        + hotspot(consumer tasks)                    contention: many tasks reading one small buffer uncached
        + rounds × serial steps × hop                latency: dependent work on wave 0 while the others wait
```

Inputs a new model must supply: the producer's task count and body, the consumer's task count, the bytes each
consumer task re-reads, how many of those come from one small shared buffer, and the depth of any dependent
selection. Constants (`PHASE_FLOOR_US` 2.64, `HANDOFF_US` 1.00, `RATE_B_US` 12354, `HOP_US` 0.690) are machine
properties from `../qwen/BUDGET-2026-10-01.md`, measured at the tiles' own 1024-thread width.

It reproduces both measured outcomes and generalises:

| model | in_norm | post_norm | top-k |
|---|---|---|---|
| Qwen3-30B-A3B (E=128, top-8) | FOLD 2.98 vs 5.98 | FOLD 0.66 vs 4.98 | **KEEP 14.17 vs 9.04** |
| Mixtral-8x7B shape (E=8, top-2) | FOLD 1.99 vs 5.98 | FOLD 1.33 vs 4.98 | **KEEP 12.59 vs 9.04** |
| Qwen2.5-1.5B, Qwen3-8B (dense) | FOLD 0.50 / 1.33 | FOLD | no candidate |

Both MoE models retain top-k, for different shape reasons.
Qwen3 has 448 expert tasks that each repeat a TOPK-deep selection.
Mixtral has 1024 tasks that contend on 16 bytes of logits despite a trivial scan.

**The hot-spot term has one calibration point.**
The bandwidth term alone predicts norm within 20%, but it predicts ~0.02 µs for top-k, an error of three orders of magnitude.
There, 448 tasks read the same 256 B of logits through uncached accesses.
These accesses cannot share a cache line, so the requests serialize.

That term determines the top-k decision and fits only one measurement.
Treat a prediction that depends on it as a hypothesis to test.
`etx/ops/gemv.py` provides `DSL_ABLATE_*` for this purpose. A timing-only ablation costs 20 seconds.

## What is still hardcoded to Qwen3-30B-A3B

Found by building plans for other shapes with no code changes. Fixed here unless marked:

1. **MoE detection read only Qwen's config key.** `num_experts` only; Mixtral spells it `num_local_experts` and
   DeepSeek `n_routed_experts`, so both loaded as **dense models with no error** — a silently wrong plan. Now every
   spelling is read and a config that mentions experts but matches none is refused. *(fixed)*
2. **Expert width assumed Qwen's schema.** `moe_intermediate_size` only; Mixtral's expert width is
   `intermediate_size`, so the plan had zero-sized expert phases. *(fixed)*
3. **The attention-prep tile was defaulted on regardless of q/k norm.** Qwen2.5-1.5B has none and refused to plan.
   Now derived from `d.qk_norm` and the wave budget. *(fixed)*
4. **The reduce's slots had to equal G and nothing set them.** G=8 happened to equal `attn_reduce`'s own default for
   Qwen3-30B-A3B, so it was never noticed; Qwen3-8B (G=4) refused to plan. `build.sh` now derives `attn_reduce=$G`.
   *(fixed)*
5. **The XCD-affine expert path requires `TOPK == NX == 8`**, because it maps expert slot *s* to XCD *s*. Qwen3-30B-A3B's
   top-8 equals the 8 XCDs by coincidence. Mixtral (top-2) refuses — loudly, which is correct, but it is the one
   blocker left for a second MoE shape. **Not fixed:** a general mapping is slot → XCD *s mod NX*, which needs the
   host's per-XCD expert buffers to follow, and `host.hip` addresses them by slot today. *(open)*
6. **q/k norm is detected from the architecture name**, not a config key, because HF configs do not expose it. An
   unrecognised architecture gets `False`, which costs a slower plan but never a wrong one. *(accepted)*

Plans built with no code changes after the fixes: Qwen2.5-1.5B 29526 tasks / 228 events, Qwen3-8B 39434 / 292,
Mixtral-8x7B shape 59203 / 292 (with `ETX_VL_XCD=0`), Qwen3-30B-A3B unchanged at 60362 / 436.

## Two divergences that caused silent wrong answers

Three times this day, a feature reached one compile stage but not another.
Each build appeared healthy and computed the wrong result.
The prevention mechanism stores a conditional feature in `plan.graph.codegen`.
The emitter publishes `ETX_EMIT_<FEATURE>` in `plan_data.h`, which both kernel and host include.
Thus, they cannot disagree.
**Do not add a `-D` flag for a feature that the host must also know.**

`examples/vllm_llm/rocm_build_check.sh` duplicates the export and slot logic of `build.sh` through string matches.
The two scripts diverged again here. They now agree, but the duplicate logic remains and merits removal.
The container's `--block` arguments still differ from those in `build.sh`, so its LDS figure is not the shipped figure.
Use the box build for resource numbers.


## The three correctness gates

1. **32/32 tokens against HF greedy** — end to end, every run.
2. **`--check-dsl`** — every DSL operator of the last layer within 2 bf16 ulps of a CPU reference.
3. **`--check-coverage`** — teacher-force the full golden sequence and diff every position's argmax against the
   HF prefill reference (`prefill_*.txt`, written by `examples/llm/prep.py` at golden time or `--prefill-only`
   for existing goldens). Mismatches at HF top-1 margins <= 0.5 are bf16 argmax ties and are tolerated; anything
   larger fails. This is the only gate that sees an output row no kernel writes: round 42's wvSplitK hole
   (lm_head rows [8192, 9728) never written) passed the other two for weeks. `ETX_DEBUG_HOLE=<M'>` re-creates
   that bug shape on demand as the gate's negative test. Setup-time asserts cover the host-controlled siblings
   (MoE task counts vs slots, attention partitions vs maxlen). Limitation: the gate sees only rows the
   reference's argmax visits. A hole above every argmax of the sequence is invisible to all three gates.
