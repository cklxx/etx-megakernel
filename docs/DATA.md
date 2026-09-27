# Measured data: what exists, where, and what uses it

Rule: a measurement is taken once and reused. Before planning a GPU run, check this file; a run is only for a new
condition (new code, new configuration) or to validate a prediction, never to regenerate something listed here.

The full data sets live outside git (they are large and binary) in `~/code/etx/results_<date><session>/`, each with a
`SHA256SUMS` of every file (verify with `cd <dir> && shasum -a 256 -c SHA256SUMS`). The hash of each `SHA256SUMS` is recorded
below, so the local copy can be checked against this file. Small, human-readable subsets are committed under
`examples/*/results/<date>/`. **The full sets have no off-machine copy yet.**

| set | session (code) | files / size | SHA256SUMS hash (first 16) | contents | used by |
|---|---|---|---|---|---|
| `results_2026-09-27` | `vm_next.sh` (7d8f874), 1x2 MI300X | 1,123 / 203 MB | 0728bffcc0088752 | sync_bench and overlap_bench logs; vLLM rocprofv3 kernel traces (CSV) for Qwen2.5-1.5B, Qwen3-8B (custom_ops=all) and Qwen3-30B-A3B; **vLLM's Triton cache for the MoE model (88 fused_moe_kernel IR/hsaco files)**; ETX Qwen variants (runs, `--check`); ETX DeepSeek trace + plan (sc0), fleet trace; vLLM DeepSeek and Qwen3-30B-A3B batch 1/2/4 JSON | design 15.11; `vllm_profile.py analyze`; small-kernel share (15.13); the MoE import (Triton IR) |
| `results_2026-09-27b` | `examples/sessions/ds_gap.py` (3d0083f) | 39 / 40 MB | 120d600b5059f706 | DeepSeek A/B runs (fleet, sc0, sc1, 2 WG/CU bounds, params, relay); ETX trace + **prologue stamps** + plan; fleet trace + task graph; routing overlap (4 prompts, 32 steps, per layer) | `compare_fleet.py`, `ledger.py`, `whatif.py --routing`, design 15.12 |
| `results_2026-09-27c` | `examples/sessions/ds_interleave.py` (8f04931) | 38 / 57 MB | 16201211b2822240 | two-request builds (router / experts / none) runs; traces + plans of the router and uncoupled builds | `phase_compare.py`, `ledger.py`, design 15.13 |
| `results_2026-09-27d` | `examples/sessions/ds_model.py` (b610466) and `ds_evstride.py` (e6c0249) | 84 / 72 MB | 5e4ce2e5e8a2fa39 | hw_bench (stream, shared, handoff incl. padded counters, wg, clock, pair); in-situ phases (one request alone, two-request pairs); hardware counters of whole tokens (x1, x2); the ETX_EV_STRIDE A/B and a stride-32 trace | `isolate_report.py`, `ledger.py`, `docs/MODEL-dsv2.md` section 5, design 15.14 |

Earlier sets (2026-09-22 .. 26) are summarized in the design document (section 15) and `examples/*/results/`; their raw
traces were not kept.

## Not kept (regenerated on each VM today; to keep from the next session on)

| artefact | size | cost to regenerate | action |
|---|---|---|---|
| fleet reference outputs: `build/golden.npz`, `golden_tokens.txt`, `golden_hidden.bin`, `fleet_cache.bin` (request 0) and `build_r1/` (request 1) | ~70 MB | ~2 min of GPU (HF reference run) per request | fetch once, push to the next VM, skip `ds_prep` / `ref_r1` when present |
| `build/weights.bin` (packed DeepSeek weights) | 31 GB | ~1 min download + ~1 min pack | regenerate (too large to move) |
| Qwen packed weights and goldens (`~/llm/<model>/vl.bin`, `golden_c1024.txt`) | 3-16 GB; goldens small | minutes | goldens are in `examples/llm/results`; packs regenerate |
| vLLM docker image | ~30 GB | ~5-10 min pull | pull only in sessions that run vLLM |

## Measurements that answer a question already (do not repeat)

- vLLM at batch 1 / 2 / 4 on DeepSeek-V2-Lite and Qwen3-30B-A3B, same prompts (2026-09-27); vLLM batch 1 on Qwen 1.5B / 8B
  (default and custom_ops=all, 2026-09-26).
- fleet on DeepSeek-V2-Lite: 3.543-3.553 ms/token on this VM type (four sessions).
- Hardware: kernel boundary 1.6 us; device barrier 7.6-7.8 us; hand-offs packed / padded; stream rates at the real bytes
  per task; staging costs; workgroup primitives; clock under FMA load; co-residency pair factors; counters (i-cache, TLB).
- Routing overlap of concurrent requests (uniform: 11.38 of 11.44 experts for two).
