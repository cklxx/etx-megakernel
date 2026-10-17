# Measured data: what exists, where, and what uses it

Rule: take each measurement once and reuse it.
Before a GPU run, consult this file.
Run only a new condition, such as new code or configuration, or validate a prediction.
Do not regenerate data that this file already lists.

The full data sets are large and binary, so they live outside git in `~/code/etx/results_<date><session>/`.
Each set has a `SHA256SUMS` file for all its files.
Verify a set with `cd <dir> && shasum -a 256 -c SHA256SUMS`.
The table below records each `SHA256SUMS` hash for comparison with the local copy.
Small, human-readable subsets reside under `examples/*/results/<date>/` in git.

Release `data-2026-09-27` has off-machine copies in the private repositories `cklxx/etx-data` and `metalopscloud/amdproject`.
It contains one tarball per set, `TARBALLS.sha256`, and this file. A download check verified the copies.
Restore with `gh release download data-2026-09-27 --repo cklxx/etx-data && shasum -a 256 -c TARBALLS.sha256 && tar xzf <set>.tgz`.

| tarball | sha256 (first 16) |
|---|---|
| results_2026-09-27.tgz (28 MB) | 21bd605370f16f19 |
| results_2026-09-27b.tgz (2 MB) | 3c598aa6f19b9872 |
| results_2026-09-27c.tgz (6 MB) | 49a8b3f89008fdd4 |
| results_2026-09-27d.tgz (7 MB) | 45c0f6da519031f9 |

| set | session (code) | files / size | SHA256SUMS hash (first 16) | contents | used by |
|---|---|---|---|---|---|
| `results_2026-09-27` | `vm_next.sh` (7d8f874), 1x2 MI300X | 1,123 / 203 MB | 0728bffcc0088752 | sync_bench and overlap_bench logs<br> vLLM rocprofv3 kernel traces (CSV) for Qwen2.5-1.5B, Qwen3-8B (custom_ops=all) and Qwen3-30B-A3B<br> **vLLM's Triton cache for the MoE model (88 fused_moe_kernel IR/hsaco files)**<br> ETX Qwen variants (runs, `--check`)<br> ETX DeepSeek trace + plan (sc0), fleet trace<br> vLLM DeepSeek and Qwen3-30B-A3B batch 1/2/4 JSON | design 15.11<br> `vllm_profile.py analyze`<br> small-kernel share (15.13)<br> the MoE import (Triton IR) |
| `results_2026-09-27b` | `examples/sessions/ds_gap.py` (3d0083f) | 39 / 40 MB | 120d600b5059f706 | DeepSeek A/B runs (fleet, sc0, sc1, 2 WG/CU bounds, params, relay)<br> ETX trace + **prologue stamps** + plan<br> fleet trace + task graph<br> routing overlap (4 prompts, 32 steps, per layer) | `compare_fleet.py`, `ledger.py`, `whatif.py --routing`, design 15.12 |
| `results_2026-09-27c` | `examples/sessions/ds_interleave.py` (8f04931) | 38 / 57 MB | 16201211b2822240 | two-request builds (router / experts / none) runs<br> traces + plans of the router and uncoupled builds | `phase_compare.py`, `ledger.py`, design 15.13 |
| `results_2026-09-27d` | `examples/sessions/ds_model.py` (b610466) and `ds_evstride.py` (e6c0249) | 84 / 72 MB | 5e4ce2e5e8a2fa39 | hw_bench (stream, shared, handoff incl. padded counters, wg, clock, pair)<br> in-situ phases (one request alone, two-request pairs)<br> hardware counters of whole tokens (x1, x2)<br> the ETX_EV_STRIDE A/B and a stride-32 trace | `isolate_report.py`, `ledger.py`, `docs/MODEL-dsv2.md` section 5, design 15.14 |
| `results_2026-09-28` | `examples/sessions/moe_gate.py` (eb7a20d + be6dfb7), 1x MI300X | 112 / 107 MB | f95453fa252c5859 | dense stride 32 / 1 and unfused runs (1.5B, 8B), MoE (Qwen3-30B-A3B) --check and fused / unfused runs, ETX_TRACE of 1.5B xcd_s1 and MoE xcd, rocprofv3 kernel traces of the 1.5B and MoE unfused builds, plans<br> `try1/` the first start (build failure) | `examples/vllm_llm/compare.py`, docs/MODEL-moe.md section 7 |
| `results_2026-09-28b` | `examples/sessions/moe_fix.py` (083f3f8..30ef875, stopped after the vLLM batch-1/2 runs), 1x MI300X | 123 / 199 MB | 3c0f34f869745869 | hw_bench lat / lf / launch / evx (+ evx2: timestamp after the arrive), imp_bench (sw, rounds)<br> Qwen2.5-1.5B and Qwen3-30B-A3B runs of every fix variant (2 rounds), --check of the fixes and of the fused q/k norm + rope, the fused q/k norm + rope builds (chained, grid, unfused), two interleaved requests<br> vLLM default / enable_qk_norm_rope_fusion batch 1 and 2, **rocprofv3 kernel trace of vLLM's fastest config** (vprof_qknr) | docs/MODEL-vllm-kernels.md 10-11, `bench_params.py` |
| `results_2026-09-30` | `examples/sessions/dsl_gemv.py` (ffb465d), 2x MI300X bare metal, ROCm 7.2.4 | 34 / 52 MB | d4031ea630a8df13 | the DSL builds measured and found worth nothing (build 3 + fold + touch 5.895 fused / 5.660 unfused against build 2's 5.934)<br> the decode trace that produced the first per-phase table<br> `hw_bench stream` at the real phase shapes in both row mappings, `lfx` (per-XCD bandwidth, 1/2/4/8), `sdat` (the sentinel handoff, measured slower than our events) | docs/RESULTS-2026-09-30.md<br> the phase-floor and row-mapping findings<br> `etx/sim/phase.py`'s achievable-rate table |
| `results_2026-10-01` | 17 loop iterations + the skinny-GEMM round, driven ad hoc from HEAD d0f41ab..64ff37f<br> **1x MI300X VF, ROCm 7.14** (`gpubox`) and **1x MI300X bare metal, ROCm 7.2.4** (`baremetal/`) | 221 / 484 MB | 067af62b1fce29b5 | the day that took 5.895 -> 4.907 and flipped the fusion margin negative -> +0.200: per-round logs (`round2..round16`), the 20-run interleaved confirmation (`final/`, incl. `NOCOH.md`), the vLLM fusion ladder (`vllm_ladder/`), the batch sweep (`batch/`), the width sweep (`width/`), the decode trace of the shipped build (`final_trace/`), `prof_unfused/` (rocprofv3 counters of the unfused kernel), and `baremetal/` — the bare-metal validation plus the **full `hw_bench` suite** (`hw_all.log`: lat, launch, evx, lfx, cohload, moegather, sdat, lf) | docs/RESULTS-2026-10-01.md, docs/BUDGET-2026-10-01.md, docs/VALIDATION-baremetal-2026-10-01.md |
| `results_tiles` (`results_tiles.tgz`, committed archive) | `examples/sessions/tiles.py`; builds `ctl`, `wsum`, `scalar`, `gu1`, `rbatch`, `all`; Hot Aisle MI300X | 195 files (109 top-level entries) / 31 MB; archive 6,347,816 bytes | archive SHA256: `096509a9ce7b76e0207811220780c8aa467c07d12529d3c3a463b8b9970b1fa2` (no SHA256SUMS manifest in this set) | Four-flag tile A/B gains, bit-identical outputs and 32/32 + CHECK-DSL gates; build resources, per-build plans, full logs and raw traces including `tr_all.bin`; answers current latency and remaining body/start-delay bounds | `examples/vllm_llm/tile_ab.py`, `docs/PLAN-generic-decoder-2026-10-08.md` section 3, `docs/SPEC-nv-reference-2026-10-08.md`, README, `docs/XCD-TIMELINE-2026-10-11.md` (per-XCD account of `tr_all.bin`) |
| `results_nvref_2026-10-09` | `examples/sessions/nvref.py` and serial vLLM stability follow-up; MI300X VF (`results_nvref_2026-10-09/report.log:20`) | 84 regular files / 72 top-level entries / 1,845,701 bytes (local inventory; `docs/NVREF-2026-10-09.md` §6) | tar.gz stream SHA256 (gzip timestamp disabled): `790da540de82ac38d63b0e5cbb52e5cf83945760320b61b70a307649d6b16790` (not a manifest hash; command in session record §6) | Same-GPU wall comparison: INSUFFICIENT; wall fusion rule met; import identity, host excess, provenance and default-vLLM outliers (`results_nvref_2026-10-09/report.log:15-23`, `results_nvref_2026-10-09/vs_summary.log:19-26`) | `docs/NVREF-2026-10-09.md`, SPEC §11 observation, `docs/GAINS-LEDGER.md`; next outlier-trigger or paired-statistics experiment |

| `results_nvref2_2026-10-09` | `examples/sessions/nvref2.py`; AMD Developer Cloud MI300X VF, ROCm 7.14.0; successful runner `session-7de5pood` | 89 regular files / 40 top-level entries / 1,800,281 bytes (local inventory; `docs/NVREF2-2026-10-09.md`, Data inventory) | tar.gz stream SHA256 (gzip timestamp disabled): `26d508936f9c0708b7892d8b0d95c2fe0a2ba6e28a1e6d72916c8e169cd5e8c9` (not a manifest hash; command in session record) | Re-rendered LEAD, one VF session; wall fusion rule met again; §12 spread prediction MISSED; top-level status/runner/record belong to aborted attempt; correct evidence in `session-7de5pood` | `docs/NVREF2-2026-10-09.md`, `docs/GAINS-TABLE.md`, SPEC §12 outcome, `docs/GAINS-LEDGER.md` |

| `results_vlbest_2026-10-09` (`~/code/etx/results_vlbest_2026-10-09/`, outside git) | `examples/sessions/vlbest.py`; AMD Developer Cloud MI300X VF; finished runner `session-iv3fdc3m` | 201 regular files / 20134708 file-content bytes (local filesystem inventory; session document, Data inventory) | No SHA256SUMS manifest present; not checksum-verified | Strongest tested legal vLLM search and paired host-wall comparison: LEAD, fusion ESTABLISHED, timed-build CERT OK (`session-iv3fdc3m/report.log:6-16`); raw arm logs/JSON, certificate, build artefacts and earlier attempts retained | `docs/VLLM-STRONGEST-2026-10-09.md`, `AGENTS.md` §1; future search of dropped or untested levers |


| `results_vlbest2_2026-10-10` (`~/code/etx/results_vlbest2_2026-10-10/`, outside git) | `examples/sessions/vlbest.py`; MI300X VF, ROCm 7.14.0 (`record.log:7,18`) | Raw arm logs, saved selections, status and runner logs; build identity and transferred checks | No SHA256SUMS manifest present; not checksum-verified | Answers whether the expanded search changes the strongest tested reference: S3 remains strongest; paired LEAD and fusion ESTABLISHED on both stated bases (`report.log:6-11`). S1 donor loses; f-arms enable the pass but replace no patterns; C2 lacks compatible winners. Import checks transfer between identical build identities (`build_identity.log:3`), not a fresh import check | `docs/VLBEST2-2026-10-10.md`, `AGENTS.md` §1; corrected offline `vlbest_report.py`; reuse for future applied-fusion and MoE search decisions; `b_all/plan.json` is the plan of `docs/XCD-TIMELINE-2026-10-11.md` |


| `results_qwen300_hour_20261012b` (`~/code/etx/`, local only, 757 MB) | Qwen300 F0; MI300X VF M3000100, ROCm 7.14.0, batch 1, context 1024, 32 greedy tokens; `88f46c2` + `654a8c0` | 410.74 s fail-closed; raw logs, certificates and diagnostics | [Six small-file SHA256 checksums](evidence/qwen300-hour-2026-10-12b-SHA256SUMS); bulk not checksum-certified here | Answers local guard/anchor and A headroom; A.stop NOT-COMPLETE because load stamps perturb bodies; B/C/D NOT-RUN | [Session record and limitations](SESSION-2026-10-12b-F0.md); repair instrumentation before certifying the route |

Earlier sets (2026-09-22 .. 26) are summarized in the design document (section 15) and `examples/*/results/`; their raw
traces were not kept.

## CPU protocol simulation artifacts (not measurements)

The [benchmark harness](../bench/README.md) writes version-bound plans, raw paired
samples, artifact hashes and recomputable reports under a caller-selected directory
(e.g. `build/bench/`). These are modeled scheduler estimates, not GPU data or model
correctness evidence. Keep failed records; do not mix them into the measured sets
above. The frozen initial config is `bench/configs/protocol-gfx942-v1.json`.

## Not kept (regenerated on each VM today; to keep from the next session on)

| artefact | size | cost to regenerate | action |
|---|---|---|---|
| fleet reference outputs: `build/golden.npz`, `golden_tokens.txt`, `golden_hidden.bin`, `fleet_cache.bin` (request 0) and `build_r1/` (request 1) | ~70 MB | ~2 min of GPU (HF reference run) per request | fetch once, push to the next VM, skip `ds_prep` / `ref_r1` when present |
| `build/weights.bin` (packed DeepSeek weights) | 31 GB | ~1 min download + ~1 min pack | regenerate (too large to move) |
| Qwen packed weights and goldens (`~/llm/<model>/vl.bin`, `golden_c1024.txt`) | 3-16 GB<br> goldens small | minutes | goldens are in `examples/llm/results`<br> packs regenerate |
| vLLM docker image | ~30 GB | ~5-10 min pull | pull only in sessions that run vLLM |

## Machines (absolute numbers are not interchangeable between them)

| box | ROCm | megakernel VGPR / spill | notes |
|---|---|---|---|
| Hot Aisle, bare metal MI300X | 7.2.4 | 128 / 9 | every number before 2026-10-01<br> `rocprofv3` aborts at startup here |
| long-term box, MI300X **VF** (virtualised) | 7.14 | 128 / 15 | the 2026-10-01 loop<br> `rocprofv3` writes its data then segfaults at teardown |

The plan is byte-identical across the machines, with 60362 tasks / 436 events.
Only register allocation differs. **Ratios and the fusion margin transfer, but absolute ms/token does not.**
The machine parameters agree. See `docs/VALIDATION-baremetal-2026-10-01.md`.
The dependent hop is 103-104 ns, or 1107-1175 ns after `buffer_inv sc1`.
The kernel boundary is 1.61 us. A coherent activation read costs 0.50-0.55 us against 2.89-2.91 for inv + plain.

**`rocm/vllm:latest` has been removed from Docker Hub.** `examples/sessions/moe_fix.py` and every session that copied
its pattern will fail on a fresh box. The working tag on 2026-10-01 was `vllm/vllm-openai-rocm:v0.27.1`.

## Measurements that answer a question already (do not repeat)

- vLLM at batch 1 / 2 / 4 on DeepSeek-V2-Lite and Qwen3-30B-A3B, same prompts (2026-09-27); vLLM batch 1 on Qwen 1.5B / 8B
  (default and custom_ops=all, 2026-09-26).
- fleet on DeepSeek-V2-Lite: 3.543-3.553 ms/token on this VM type (four sessions).
- Hardware: kernel boundary 1.6 us; device barrier 7.6-7.8 us; handoffs packed / padded; stream rates at the real bytes
  per task; staging costs; workgroup primitives; clock under FMA load; co-residency pair factors; counters (i-cache, TLB).
- Routing overlap of concurrent requests (uniform: 11.38 of 11.44 experts for two).
- ETX_EV_STRIDE 32 vs 1 on the vLLM-kernel Qwen graphs (0.5-3%), the imported MoE kernels' per-launch cost against vLLM (2026-09-28).

- **backup data-2026-10-02** (releases on cklxx/etx-data and metalopscloud/amdproject): results_2026-09-30 + results_2026-10-01 as one tarball, sha256 5f1b9d3a0bd439e8..., download-verified. Covers every round of the 2026-10-01/02 session.

## Qwen producer-scoring CPU preparation, 2026-10-15

`examples/vllm_llm/results/2026-10-15-producer-prep/` holds the executed local checks.
It includes numerical and schedule simulations, command receipts, full test logs,
plan proofs, and offline compiler reports.
Its `SHA256SUMS` file binds the retained artifacts.
The simulation executes the shared production scoring header.
Its weights and activations are synthetic. They are not model captures.
The sparse near-tie witness and historical trace questions reuse the prior router study.
No complete retained Qwen router activation and weight set was found.
Actual-model capture and replay are prepared but pending GPU execution.
There is no new GPU measurement in this set.
See `docs/QWEN-PRODUCER-PREP-2026-10-15.md` and its runbook.

## Measured GPU queue and multibatch traces, 2026-10-15 UTC

The [results index](../examples/gpu-live/README.md) links the completed queue's exact source bindings,
five-pair baselines, rejected correctness gates and profiler failure. Reuse these measurements; the expert
screen lost all five pairs, attention had mixed signs, and the producer and x8 paths did not pass correctness.

`examples/gpu-live/results/2026-10-16-multibatch/` contains the nine gated final-token trace summaries,
the original raw-record seal/manifest, and a deterministic summary derivation. B2 has 109,892 tasks and
B4 has 219,784; their median instrumented spans are 4.339200 and 8.670160 ms. These are attribution data,
not new unprofiled candidate performance or accepted HBM counters. The full 32-file local raw export was
verified complete; its manifest SHA256 is `91f63452b962ef7cf760e9b2b447d2b40f575d59c9e3c33d559e4390ce391c15`.

Published receipts omit connection details as described in the publication metadata. Original seals bind
the original bytes, and public derivative checksums are separate. No model weight pack or executable binary
is included. The shared-weight prototype was cancelled before edits, build or measurement.

## GLM-5.3-Flash module-gate offline artefacts, 2026-10-17

`~/code/etx-cache/glm53-module/` (outside git) holds the output of
`python -m examples.glm53.module_gate prepare`. That is a synthetic native-FP8
checkpoint (deterministic: `weights_sha256` is pinned in
`docs/evidence/glm53-module-gate/glm-model-api.json`), its production stage, the HF
reference and the CPU prediction of the native body. No GPU measurement is in this
set. Push `stage/` and `reference.{json,logits}` to the box; do not regenerate them
there. The box has no torch, and the reference is bound to these exact shard bytes.

| file | sha256 (first 16) |
|---|---|
| `checkpoint/model.safetensors` | 4a0ddc1ba6fbf6f0 |
| `stage/stage.json` | 7234a63c2b9171a6 |
| `reference.json` | aac9bc165a215023 |
| `reference.logits` | bad0ed0d913e95e7 |
| `cpu-prediction.json` | 4f66242113f00334 |

The CPU prediction: 32/32 tokens for both requests, median per-step logit error
0.453, maximum 4.98, HF minimum top-2 margin 3.56. See
[the runbook's module gate](QWEN-BATCH-GLM53-RUNBOOK.md#module-gate).
