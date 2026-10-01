# Why the DSL tiles, and not simply vLLM's kernels (2026-10-01)

Measured answer to a question previously settled only by argument. Raw logs: `results_2026-10-01/round24/`.

Measured on today's codebase (HEAD 7cbd872), VF box, ROCm 7.14. Wall ms/token, three interleaved rounds.
Gate: 32/32 tokens against HF greedy on every build. `--check-dsl` checks DSL operators only, so it does not
apply to an all-imported build; `--check` would need `ETX_VL_CHECK=1`, a different binary with vLLM's original
kernels linked in, which perturbs timing — not built. The end-to-end token gate is the one that applies to all
three equally.

## Switch sets

| build | switches | tasks | events | megakernel LDS | spill |
|---|---|---|---|---|---|
| `imp` — all vLLM kernels | `ETX_VL_XCD=1 ETX_VL_CHAIN=1` | 67994 | 628 | 41448 | 8 |
| `qknr` — all vLLM + fused q/k-norm-rope | `+ ETX_VL_QKNR=1` | 67994 | 628 | 41448 | 8 |
| `dsl` — control, today's best | `+ DSL_GEMV, DSL_TOPK, DSL_ATTN=head, DSL_FOLD, TOUCH, COH, FOLD_TOPK=0, CU=256` | 60362 | 436 | 29736 | 16 |

`ETX_VL_COH`, `ETX_VL_NT`, `ETX_VL_VECNORM` and `ETX_VL_CU` act only on DSL tiles and are no-ops for the
imported builds — that is the point of the experiment, not an oversight. The host-side improvements
(async `reset_step`, the dead memset, the poll yield) are in `host.h` and apply to all three.
`qknorm_rope` is instantiated 48 times in `qknr` and 0 times in `imp`, so the variant took effect; the task
count coincides because the chain merges the launches it replaces.

## Result

| build | fused | unfused | fusion margin |
|---|---|---|---|
| `imp` | 6.435 | 6.264 | **−0.171 (fusion loses)** |
| `qknr` | 6.071 | 5.787 | **−0.284 (fusion loses)** |
| `dsl` | **4.938** | 5.002 | **+0.064 (fusion wins)** |

Medians of three; spread ≤ 0.035 fused, ≤ 0.041 unfused. All runs 32/32.
vLLM itself on this box: 4.862.

Two numbers, both decisive:

1. **All-imported is 1.13 ms (23%) slower than the DSL tiles** end to end — 6.071 at best against 4.938.
2. **With vLLM's kernels, fusion still loses**, by 0.171 and 0.284 ms, exactly as it did on 2026-09-30
   (5.875 fused / 5.623 unfused, device basis, bare metal). The fusion margin turned positive this year
   *only* for builds whose tiles we generate.

That second point is the mechanism. `ETX_VL_COH` removes the per-task `buffer_inv sc1` by making the tile's
activation accesses coherent — which requires editing the tile. An imported kernel's loads come from vLLM's
source, so its task keeps the acquire, and the acquire invalidates the whole XCD L2 for every phase that
follows. Nothing in the imported path can opt out.

## Per-phase, layer 11 (decode traces, µs)

| phase | `imp` tasks / body | `dsl` tasks / body |
|---|---|---|
| in_norm | 8 / 2.50 | — folded into qkv |
| qkv | 304 / 6.75 | 256 / 10.04 (incl. the fold) |
| q/k norm + rope + cache | 1 / **16.24** (chained, one workgroup) | prep 4 / 5.11 |
| attn | 5 / 13.18 | 20 / 6.06 |
| reduce | 4 / 5.76 | 4 / 5.79 |
| o_proj | 304 / 6.45 | 256 / 5.26 |
| router | 304 / 2.51 | 256 / 3.16 |
| moe_up | 192 / 20.57 | 192 / 18.70 |
| moe_act (silu) | 8 / 1.88 | — folded away |
| moe_down | 256 / 10.57 | 256 / 11.73 |
| **layer period** | **≈122** | **≈95** |

Total span 6347.1 against 4804.1 µs.

**vLLM's kernels are not worse in isolation**, but the qkv row above is not like-for-like and the first reading of
it was wrong (corrected 2026-10-01, round 25). Our `dsl_gemv_fold` folds the input `fused_add_rms_norm` into the
qkv prologue; `imp` runs it as the separate `in_norm` phase. Comparing the same work:

| | `imp` | `dsl` |
|---|---|---|
| bodies, in_norm + qkv | 2.50 + 6.75 = **9.25** | **10.04** |
| layer-delta, in_norm + qkv | 3.4 + 11.3 = **14.7** | **15.2** |

0.5-0.8 µs/layer, not 3.29. And the GEMV alone is ours by a margin: layer 0 is the one layer the DSL build does
*not* fold (`model.py`: `fold = ... (role == "qkv" and layer > 0)`), and `qkv_0`'s body is **6.29 µs against
`wvsplitk`'s 6.75** on the identical 5120x2048 shape, with 256 workgroups against 304 — more rows per task in less
time. The fold costs 3.75 µs (10.04 - 6.29) and is still the right choice for us, because our own separate
`in_norm` phase costs more: 5.9 µs of layer-delta at layer 0 against the fold's 3.75, our 8-task x-form replica
being dearer than vLLM's. We beat `wvsplitk` on o_proj (5.26 against 6.45) and moe_up (18.70 against 20.57) too.
**There is no GEMV gap to close.**

**What the imported build loses on is structure, which the kernels dictate:**

- the q/k norm + rope + cache chain runs as **one task on one workgroup, 16.24 µs serialized**, where the DSL
  path spreads the same work over 4 per-kv-head tasks at 5.11;
- attention runs 5 tasks against 20, so it cannot be placed by kv head and its partitions cannot overlap;
- an extra `moe_act` phase and `moe_sum` survive, which the folds remove — 628 events against 436;
- every imported task's acquire invalidates L2, which shows up as the waits around them: o_proj 42.78 against
  24.49, reduce 35.98 against 15.78, moe_act 32.54.

## Conclusion

On today's codebase, importing vLLM's kernels everywhere costs **+1.13 ms/token (+23%)** against our DSL
tiles, and **fusion does not pay at all** (−0.171, −0.284) because the acquire cannot be removed from a kernel
we did not generate.

The answer to "why not just use vLLM's kernels" is therefore not that they are slow — per task they are
competitive. It is that using them forces vLLM's launch structure (its phase
boundaries, its one-workgroup chains, its separate silu and moe_sum launches) and forbids the one change that
made fusion pay. The DSL tiles earn their place by being editable, not by being faster.
