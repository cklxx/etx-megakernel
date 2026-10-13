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

## GPU results 2026-10-13 (box 165.245.139.183, ROCm 7.14; runs `$ETX_CACHE/results_2026-10-13-d1d5/run{1,2}`)

Every invocation 32/32 + 27/27. Device medians: default 3.668-3.672, fleet 3.542, all 3.564 (-108, GO), relaxed 3.439
(-232, D1 cost as expected), coh 3.677 (+9: **D2 NO-GO**, (a) and (b) fail; 237 us behind relaxed), d4 3.936 (+271:
**D4 NO-GO**; criterion (a) holds, shared gate_up ready 11.5 us before routing completes in 26/26 layers).

## Next two arms, thresholds written before the code (2026-10-13, same box)

| arm | what | rule (fixed now) |
|---|---|---|
| allrelaxed | all five levers + `-DETX_ACQUIRE_DOMAIN_RELAXED` (UNSAFE acquire: a measurement of the prize, never a candidate) | no verdict; its gain over `all` is the most a safe acquire as cheap as sc0 could add to the best arm. If `all - allrelaxed` < 150 us, the acquire route alone cannot close the 231 us to 3.333 ms |
| d4r0 | D4's placement with 0 slice rows (shared XCDs: own fold + norm, no router, no slices) | splits D4's +271 us: d4r0 slower than default by >= 200 us -> the placement itself is the loss; by <= 50 us -> the slices are |

Results (run 3, 05:08-05:11): **allrelaxed -326 us vs default (spread 8)**: the unsafe acquire is worth ~218 us on top
of `all`, above the 150 us bar, so the acquire route can close most of the gap to 3.333 ms if a SAFE acquire costs
what sc0 costs. **d4r0 -62 us vs default (spread 8, every pair faster)**: D4's placement without slices is a gain;
the slices are the +271 us (the shared down, 35.7 us per task against 17.5 for a routed down, is the straggler).

| arm | what | rule (fixed now, before the code) |
|---|---|---|
| alld4r0 | all five levers + D4 placement, 0 slice rows; three-way default / all / alld4r0 | KEEP if every pair faster than `all`, median >= 30 us and > 2 x spread; 32/32 + 27/27 everywhere |

## The deciding microbenchmark for a safe cheap acquire (rule written before the run)

`bench/sync/sync_bench.hip loadscope`: the ping-pong with NO acquire invalidate and the payload read with plain (xn,
control), sc0 (xl0) or sc1 (xl1) loads on one XCD, and sc1 (cl1) or sc0 sc1 (cl01) loads across XCDs behind a
`buffer_wbl2 sc1` release. A mode is SAFE only with 0 stale words at 64 B, 8 KB and 64 KB over 2000 rounds, and only
if `xn` goes stale in the same run (the check detects staleness). If xl1 is safe: next lever "D2b" = sc1 payload loads,
no per-task invalidate, no per-task write-back (DEVICE events keep their flush). If only cl01 is safe: that is D2's
form, measured at no gain, and the safe-acquire route is closed.

Results, runs 4-5 (05:12-05:16, `$ETX_CACHE/results_2026-10-13-d1d5/run4`, `run5`):
- **alld4r0 3.506 ms vs all 3.567 ms: -61 us, spread 8, every pair faster: KEEP.** Best safe arm (285 tok/s).
  Against default (3.666): -160 us.
- **loadscope**: control xn stale at 64 B / 8 KB (detection proven; 64 KB clean by L1 eviction); xl0 (sc0 loads)
  stale like xcd; **xl1 (sc1 loads, no invalidate, same XCD) 0 stale at all sizes**, one-way 1.047 / 1.351 /
  4.721 us (xcd1, the safe buffer_inv sc1: 1.055 / 1.409 / 4.165); cl1 and cl01 (cross XCD behind buffer_wbl2) 0 stale.
  By the rule: **xl1 is safe -> D2b** (sc1 payload loads, no per-task invalidate, no per-task write-back on
  same-XCD events).

## D2b — sc1 payload loads, no per-task invalidate, no per-task write-back (thresholds written before the code)

Design from the loadscope result: fleet's tiles read every payload another task wrote with AGENT-scope loads
(`global_load ... sc1`: skips the CU's stale L1, xl1; reads behind a write-back across XCDs, cl1). New compiler mode
`coherence="coherent_l1"`: no acquire invalidate; NOT a coherent consumer for the write-back rule (a same-XCD payload is
read from the shared L2, so DOMAIN releases stay plain); a plain producer (DEVICE events keep the last-arriver flush).
Switch `ETX_DSV2_COH=l1` + lever `l1_loads`; with the five levers it also covers attn_prologue_live's own payload reads.
dense_down stays plain. With D4 only at 0 slice rows (the slices' h reads are plain).

| arm | rule (fixed now) |
|---|---|
| l1 (vs default and relaxed, three-way) | D2's rule: GO if (a) every pair faster than default and median >= 70 us, and (b) median(l1 - relaxed) <= max(2 x relaxed-pair spread, 20 us) |
| alld4r0l1 (all + D4 placement + l1; three-way default / alld4r0 / alld4r0l1) | KEEP if every pair faster than alld4r0, median >= 30 us and > 2 x spread. The 300 tok/s line is 3.333 ms device |
| both | 32/32 + 27/27 in every invocation; any stale-data failure is a NO-GO regardless of time |

## Decision register (2026-10-13): every measured arm, its rule, when the rule was fixed, the outcome

Re-derived by `examples.dsv2lite.inplace_session.verdict` from the fetched pairs JSONs (`$ETX_CACHE/results_2026-10-13-d1d5/run1..4`).
From the commit that adds this register on, every pairs JSON carries one `verdict` field and the runner prints a `VERDICT` line; before it,
an arm's decision lived in an arm-specific field (`d2`, `d4`, `keep_vs_all`) while its per-reference `go` fields were
blanked to None -- a reporting defect that made a decided arm look undecided.

| arm | rule (as fixed) | rule fixed in | measured in | outcome |
|---|---|---|---|---|
| all | GO: every pair faster than default, median >= 70 us, > 2 x spread | inplace_session GO_US (10-12) | run 1 | **GO** (-108 us, spread 18) |
| coh (D2) | (a) every pair faster than default, median >= 70 us; (b) within max(2 x relaxed spread, 20 us) of relaxed | 84eaa1b | run 2 | **NO-GO** (+9 us; 237 us behind relaxed) |
| d4 (D4) | KEEP: every pair faster than default, median >= 30 us, > 2 x spread | 84eaa1b | run 2 | **NO-GO** (+271 us) |
| alld4r0 | KEEP: every pair faster than `all`, median >= 30 us, > 2 x spread | **7e08d27, 05:12:12 UTC**; the measuring session started 05:12:30 UTC from that commit (`run4/runner.log`, SOURCE_IDENTITY) | run 4 | **KEEP** (-61 us vs all, spread 8, 5/5 pairs faster; `keep_vs_all.keep = true`). 3.506 ms device median |
| l1 (D2b) | D2's rule | 8d69e52 | **NOT COMPLETE** (box destroyed in round 3 of 5) | none |
| alld4r0l1 | KEEP vs alld4r0 | 8d69e52 | **never ran** | none |
| relaxed, allrelaxed, d4r0, baseline, singles | no decision: attribution / measurement | — | runs 1-3 | NO RULE (by design) |

The 3.506 ms figure is a measurement; the decision to keep alld4r0 rests on the rule above, fixed before its run.

## Incomplete: D2b (l1), run of 05:25 UTC, box destroyed mid-session

`examples/dsv2lite/results/2026-10-13-runs/sess/l1_r{0,1,2}_*.raw.log`: three of five rounds, every invocation 32/32 +
27/27. Device medians, ms -- default 3.661 / 3.671 / 3.665, relaxed 3.445 / 3.441 / 3.432, **l1 3.333 / 3.341 / 3.329**.
NOT a result: the rule needs five interleaved rounds, and round 3 was cut. It is the first measurement to repeat; the
alld4r0l1 arm (the best arm + D2b) never ran.

## Known losses and provenance (2026-10-13)

- **No box-side hashes.** No run directory carried a SHA256SUMS written on the box. `SHA256SUMS` in each
  `$ETX_CACHE/results_2026-10-13-d1d5/run*` and in `examples/dsv2lite/results/2026-10-13-runs/` were computed AFTER the
  fetch (rsync over SSH): they pin the copies we hold, not what the box wrote.
- **Stamps.** The run-1 STAMPS2 files (trace_default / trace_all, 117,440,544 B each) are NOT lost: they are in
  `$ETX_CACHE/results_2026-10-13-d1d5/run1` and re-derive the box's stage tables exactly (spans 3939.2 / 3842.4 us).
  They are outside git (size) and not yet in a data release.
- **Per-stage attribution of the new arms is missing**, independent of the box: d4, alld4r0, l1 and alld4r0l1 never had
  a stamped build or trace (only an unstamped d4 trace). Recovering it needs a GPU run with their `--stamped` builds.
- **D2b**: see "Incomplete" above.
