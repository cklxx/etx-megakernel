# DeepSeek D1-D5: go/no-go thresholds, written before the code (2026-10-13)

Work order: `/tmp/claude-dsv2-d1d5.md` (kailun, 2026-10-13); scope: `docs/SCOPE-300TOKS-2026-10-11.md` section 2.
Branch `dsv2-d1d5` off `dsv2-session`. CPU only until a box is opened. Every switch defaults to today's behaviour and
the default path stays byte-identical (checked per item). Pair rules: 5 interleaved pairs in one session, device basis
(embed -> argmax), every invocation 32/32 HF-greedy tokens and 27/27 layers. "KEEP bar" = 30 us/token
(`docs/BASELINE-STATE-2026-10-12.md`, the project's lever bar); "2% bar" = 70 us at ~3.5 ms.

Context measured 2026-10-13 (`examples/dsv2lite/results/2026-10-13-ds_inplace`): the ds_inplace arms ran the SAFE sc1
acquire (no flag); the 3.474 ms record of 2026-09-27 ran `-DETX_ACQUIRE_DOMAIN_RELAXED`. sc1 cost: +242 us in
2026-09-27b (3.987 vs 3.745), consistent with today's sc1 default 3.709 ms.

| item | done on CPU when | GPU threshold (fixed now) |
|---|---|---|
| D1 safe option in the sessions | no DeepSeek session passes the relaxed flag implicitly; relaxed only as a named, explicit arm; default build byte-identical | not a speed lever: GO = the safe build passes 32/32 + 27/27 in every invocation. The paired cost safe - relaxed is RECORDED, expected +235 +- 30 us [E] |
| D2 coherent fleet accessors (Qwen's ETX_VL_COH, ported) | the emitted megakernel has 0 per-task domain `buffer_inv sc1` on the covered events and keeps the producer writeback (counted in the generated code and the ISA); default byte-identical | GO if (a) coherent vs sc1: every pair faster and median >= 70 us; (b) coherent vs relaxed: median difference <= max(2 x relaxed-pair spread, 20 us); (c) 32/32 + 27/27 everywhere |
| D3 four K-chunks (352 rows) | the 4-chunk plan builds and verifies (events, chunk waits), default byte-identical | GO only if every pair faster than one chunk and median >= 30 us and > 2 x spread; otherwise DROP (prior: 512-row chunks 4.020 ms, fleet STATUS.md:33) |
| D4 shared experts during routing + placement check | (a) shared-expert gate_up on the two shared XCDs carries no routing; (b) every routed slice's gate_up and down are on one XCD; (c) the build REFUSES an invalid placement, and a test in the suite proves the refusal fires; default byte-identical | GO if the trace shows the shared XCDs' gate_up ready before the layer's routing completes, and every pair faster with median >= 30 us and > 2 x spread (ceiling ~0.18 ms, scope section 4) |
| D5 compiler prefetch (p7 + kernel hook) | the hook emits a prefetch body for fleet's tiles behind a switch; default byte-identical | DROP if the median is inside the spread or any pair is slower; KEEP only if every pair faster, median >= 30 us and > 2 x spread (prior: --prefetch-next 4% slower, idle prefetch no gain) |

## Scope change (2026-10-13, owner via kailun, `/tmp/claude-dsv2-priority.md`)

D1, D2 and D4 only, today. **D3 and D5 are parked** (not started): their prior evidence is negative or neutral and the
owner did not ask for them. Their thresholds above stay as written for whenever they are picked up.

## D2 on CPU (done; the GPU pairs wait for a box)

Switch: `ETX_DSV2_COH=1` (plan) + lever `coherent_loads` (tiles); session arm `coh`. Default build unchanged: plan
`09011c20`, ROCm 7.2.4 default ISA `a105cb16` (re-verified with the shim edit, which is `#ifdef FLEET_COH_LOADS` only).

Memory model. Consumer: every payload another task wrote is read with relaxed system-scope atomics through
address_space(1) (`global_load ... sc0 sc1`, past L1 and L2; the lowering Qwen's ETX_VL_COH builds use), so the tile
needs no acquire invalidate. Producer: stores stay plain and the release writes the XCD's L2 back (fleet measured that
write-back necessary, `fleet_runtime.h:247-255`): the new compiler mode `coherence="coherent_loads"` is a coherent
consumer but a PLAIN producer, so DOMAIN releases feeding it become `ETX_RELEASE_DEVICE` (buffer_wbl2) and DEVICE events
keep the last-arriver flush (no `_nf`). One hand-off has no ETX edge (attention waits for q_absorbed inside its body):
the shim's q_absorb wrapper writes back before bumping that counter. Not covered, and why that is safe: KV-cache rows
come from earlier tokens = earlier launches (host.hip launches once per token); dense_down reads h from global inside
gemv_rows, so it stays plain and keeps its acquire; q_absorb without PUB_WAIT is unreachable from the shim.

| ROCm 7.2.4, gfx942 | default | coh |
|---|---|---|
| `buffer_inv sc1` in etx_megakernel_d0 | 276 | 3 (dense_dn_0's relay-branch pair + the relay worker's, runtime-off) |
| `buffer_wbl2` in etx_megakernel_d0 | 114 | 249 |
| `global_load ... sc0 sc1` | 0 | 485 |
| VGPR / AGPR / SGPR / scratch / LDS / waves | 256 / 48 / 108 / 0 / 26,668 / 1 | 256 / 36 / 106 / 0 / 26,668 / 1 |
| plan: grids with no acquire | 0 of 218 waiting grids | 217 of 218 (all but dense_dn_0) |

(`fleet_decode_step`, fleet's own kernel compiled into the TU but never launched, keeps its 3 in both.)
Proof: `.venv/bin/python -m pytest -q tests/test_dsv2_lever_coherent_loads.py tests/test_coherence.py` (15 passed) and
`.venv/bin/python -m examples.dsv2lite.verify_lever coherent_loads` (record `results/levers/coherent_loads.json`).
`coh_check.py` refuses (exit 2, before the compiler) a coherent plan with unpatched tiles and patched tiles with a plain
plan; a test proves both refusals fire.

Unknown until a box: correctness under traffic (32/32 + 27/27), and the price. The per-task invalidate is gone, but
every task now ends in a whole-L2 write-back (249 sites, ~one per producer task instead of one per domain per DEVICE
event), and payload reads are MALL round trips. If (a) passes and (b) fails, the next step is coherent STORES too (the
full Qwen form, `_nf` arrives, no write-back), which the work order's "keep the writeback" rules out today.
