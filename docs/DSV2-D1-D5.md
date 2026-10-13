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

## D4 on CPU (done; the GPU pairs and trace wait for a box)

Switch: `ETX_DSV2_D4=1` (plan) + `-DETX_DSV2_D4=1` (tiles); `ETX_DSV2_D4_ROWS` (default 32); session arm `d4`, pairs vs
default plus `d4trace` (criterion (a)). Default unchanged: plan `09011c20`, ROCm 7.2.4 default ISA `a105cb16`.
Implemented in ETX's port (graph.py `build_moe_layer_d4`, fleet_shim.hip `fleet_d4_*`); the pinned fleet export is
not edited, so fleet's own launcher (`fleet_launch.hip:543`, `taskgraph.py:420`) is untouched.

Placement per MoE layer (`examples/dsv2lite/d4_placement.py`):
- routed XCD k (routing slot k): router, then gate_up / down of rows [32, 1408) of slot k; gate_up's worker 0 publishes
  slot k's (id, weight) and bumps the owning shared XCD's counter (written back first);
- shared XCD 6 + x: NO router. Its gate_up folds the o_proj partials and norms by itself and waits on o_proj only
  (starts during routing), runs shared half x, then -- in-body wait on the counter, as fleet's q_c publication -- rows
  [0, 32) of slots 3x..3x+2 from the same staged x; one down for the half and the three slices (fp32 accumulator,
  one bf16 store; each slice's routing weight folded into its h).
Build-time refusal (graph.build(), so `etx compile` exits non-zero before writing a plan): a split slice (gate_up and
down on different XCDs), rows uncovered or covered twice, a row boundary not a multiple of 32 (an h cache line written
by two XCDs), a shared XCD whose first stage needs routing, and a realised grid whose domain_map puts a stage on another
XCD than the table (evaluated per task coordinate). D4 + D2 together is refused (the d4 bodies are not audited for
coherent loads). Tests: `tests/test_dsv2_d4.py` (17), including the refusals through graph.build() and `etx compile`.

| ROCm 7.2.4 | default | d4 |
|---|---|---|
| VGPR / AGPR / SGPR / scratch / LDS / waves | 256 / 48 / 108 / 0 / 26,668 / 1 | 256 / 54 / 108 / 0 / 26,672 / 1 |

Expected [E], from fleet's per-XCD stamps (gate_up 27.5 us + 5.5 prologue for 1408 rows, down 28.4 + 1.6; router 9 =
7.4 fold/norm + 1.6 logits): the shared XCDs drop the router logits and the gate_up routing, about 7.1 us of slack per
layer; each routed row moved costs them three. With 32 rows the routed XCDs end 1.3 us earlier and the shared ones still
before them: about 1.3 us/layer, 33 us/token, at the 30 us KEEP bar. 64 rows overshoots (the shared XCDs become the
stragglers, a loss). The estimate is crude, so the GPU decides. Numerics change (a routed expert's output is split over
two bf16 partials; the shared XCD's slices are summed in fp32 before one rounding), so 32/32 + 27/27 must be re-proven
on the GPU.
