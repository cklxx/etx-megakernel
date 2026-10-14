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

## D5 resumed: binding threshold before implementation (brief 06, 2026-10-13)

The owner has resumed D5 for CPU-only implementation. `alld5` = all five tile levers
(`gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live`) + compiler early weight loading.
Compare exactly two arms, ordered (`all`, `alld5`), in five interleaved rounds (10 invocations), with the verdict
against `all`. Every invocation must pass 32/32 HF-greedy tokens and 27/27 layers.

**DROP if the median gain is inside the spread (|median| <= spread) or any pair is slower; KEEP only if every pair
is faster, median >= 30 us and median > 2 x spread.** Any other completed outcome is DROP; an incomplete or
incorrect run cannot KEEP. Gains are `all - alld5`, on the device embed-to-argmax basis. No GPU run is authorized
by this brief. Default plan `09011c20` and ROCm 7.2.4 default ISA `a105cb16` must remain unchanged; the enabled
arm must compile offline with zero scratch before it is eligible for any later GPU session.

Prior evidence is adverse: fleet's retained-register prefetch was 4% slower and idle-worker cache warming gave no
gain. D5 uses only immutable, routing-independent weights; benefit, runtime correctness and cache retention remain
unknown until the complete paired experiment. Drop-if-neutral is binding.

### D5 implementation and CPU evidence

`ETX_DSV2_D5=1` selects `fleet_d5.hip`, attaches `hip_link(prefetch=...)` siblings and annotates their opaque
FleetParams weight argument for P7. No switch keeps the old source, graph and generated kernel. `PassOptions.prefetch`
already defaults to true; gfx942's `async_copy_to_lds: dword_only` chooses `l2_warm`. P7's queue entries are advisory
(they omit queue heads); the existing pre-wait kernel hook controls execution. The FleetParams annotation alone is
not a weight-address proof because the struct also contains activation pointers.

The new source calls the pinned fleet weight-address planner with only dimensions, weights and static coordinates.
Only QKV, merge W_UV, o_proj, router, lm_head and shared expert halves are selected. Routed experts return without
loading; the dense layer's norm has zero rows. Waves 1-3 each issue one 16-byte load per lane, consumed by an asm sink
inside the hook; wave 0 remains available to poll. Nothing is held for the body across the event wait, and the hook
has no LDS, activation reads, stores or block barriers. Fleet's pinned source and original tile bodies are unchanged.

The first offline attempt emitted the hook in every layer's switch case and failed the scratch gate (96 B/lane;
256 VGPR, 96 AGPR). Opt-in shared prefetch dispatch now provides one call site per sibling, with per-type constants
initialized before dispatch. The final five-lever + D5 build on gfx942, ROCm 7.2.4/clang 22, has **256 VGPR, 101 AGPR,
108 SGPR, 0 B/lane scratch, 26,668 B LDS, 1 wave/SIMD**. The default has 256/48/108 registers, zero scratch, the same
LDS and occupancy. These are offline compiler resource reports, not runtime measurements.

Source counts (generated dispatch plus its included hook, **not ISA instruction counts**), for one request,
27 layers, 256 threads: each eligible task issues 192 lane-vector loads of 16 B (3,072 requested bytes).

- QKV: 7,992 tasks, 1,534,464 lane-vector loads.
- Merge W_UV: 6,912 tasks, 1,327,104 loads.
- o_proj: 7,992 tasks, 1,534,464 loads.
- Router: 7,696 tasks, 1,477,632 loads; dense-layer norm: 296 tasks with zero loads.
- Shared gate/up: 1,924 tasks, 369,408 loads; routed gate/up: 5,772 tasks with zero loads.
- Shared down: 1,924 tasks, 369,408 loads; routed down: 5,772 tasks with zero loads.
- lm_head: 296 tasks, 56,832 loads. Other task kinds have no hook.

Total: 34,736 eligible tasks, 6,669,312 lane-vector loads, 106,708,992 requested bytes/token. [E] If all of those bytes
required memory traffic at 4.3 TB/s, their ideal service time would be `106708992 / 4.3e12 * 1e6 = 24.816 us`.
This is neither a gain prediction nor measured traffic: repeated cache hits, eviction, competition with the producer,
the existing acquire invalidation and dispatch/register costs are unknown. The prior negative results remain relevant.

Reproduction: `.venv/bin/python -m examples.dsv2lite.verify_lever alld5` runs the five lever gates, stamps gate and
`tests/test_dsv2_d5.py`, then offline default/arm device + host builds and the default ISA check; the record is
`examples/dsv2lite/results/levers/alld5.json`. `pytest -q -s tests/test_dsv2_d5.py` prints the address/count audit.
Final verification exited 0 with 41 CPU gates passing, both offline device/host builds passing and the default ISA
unchanged. The required `.venv/bin/python -m pytest -q tests/test_dsv2_*.py tests/test_coherence.py` exited 0:
**133 passed**. Additional compiler/pass/manifest checks: 42 passed, 1 skipped (optional external-tool gate).
The CPU harness executes the exact pinned C++ planner and exact D5 address code over all coordinates and lanes,
replacing only the device load/sink with an address observer. It has no activation/KV/LDS fields and uses inaccessible
weight allocations; every observed 16-byte span is checked against independently specified weight regions and row
ownership. It does not establish GPU correctness.

Default identities: plan `09011c207b1e92febf29a2c2efc2d944ef9ad7ac8a0803633579c8444d821f8a`, generated kernel
`751293fad5f5c98ec6dbf9eeca3c73319b1d4ded4b6ca73996a65757c0ffae5d`, ROCm ISA
`a105cb16a414ee0f9f697c9de13205419d09674f5c69f6ee804cf36942377724`. Tests pin the first two with the switch both
unset and zero; the offline verifier checks the third. GPU cost: $0, no box. Runtime 32/32 + 27/27, speed and the
KEEP/DROP verdict remain unknown. `.venv/bin/python -m examples.dsv2lite.inplace_session pairs alld5` is the future
two-way experiment against `all`;
this brief's two-way choice supersedes the older D5 budget slot below that named `alld4r0` as the reference.

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

## The next paid hour (plan, 2026-10-13): minute budget from today's runner records

Measured costs (`$ETX_CACHE/results_2026-10-13-d1d5/run*/runner.log`, `results/2026-10-13-runs/sess/runner.log`):
one pairs invocation = 5.9 s (3-way pairs: 15 invocations in 88-90 s, 4 jobs; 2-way: 10 in 58-60 s, 11 jobs); one
traced token = 7-8 s (3 jobs); builds 21-34 s each, stamped 28-33 s, all in parallel during prep (20 vCPUs); fresh-box
prep to `preflight ok` = 3 min 59 s (run 1: apt 11, prep_env 87, prep_dl 30, prep_ref 41 on GPU, prep_pack 30,
prep_sums 20, preflight 20); box-setup + session start <= 51 s (04:07:58 staged -> 04:08:49 runner start); staging
volume 85.6 MB tree + 10 MB fleet export (rsync dry run). Unmeasured: rsync throughput over the SOCKS proxy.

| minute (T = box handed over) | step | cost basis |
|---|---|---|
| 0:00-0:03 | connect THROUGH the proxy wrapper from the first attempt (direct SSH failed for ~25 min today; proxied login 3.5 s), stage, box-setup (HIP headers), start | 95.6 MB staged; <= 51 s setup [M]; proxy throughput [U] |
| 0:03-0:07 | prep chain to preflight; all builds (incl. stamped) in parallel | 3 min 59 s [M] |
| 0:07:00 | pairs_l1 (D2b rerun, 3-way default / relaxed / l1, 5 rounds) | 15 inv, 88 s [M] |
| 0:08:30 | pairs_alld4r0l1 (3-way default / alld4r0 / alld4r0l1) | 15 inv, 88 s [M] |
| 0:10:00 | pairs D3 on the best arm (2-way alld4r0 / +D3) -- ONLY if built and proven locally before the box | 10 inv, 59 s [M] |
| 0:11:00 | pairs D5 on the best arm (2-way alld4r0 / +D5) -- same condition | 10 inv, 59 s [M] |
| 0:12:00 | stamped traces: default, alld4r0, alld4r0l1 (recovers the lost per-stage record) | 3 inv, 24 s [M] |
| 0:12:30 | box-side SHA256SUMS of the session directory (closes today's provenance gap) | seconds |
| 0:13-0:40 | CONTINGENCY, not planned work: SSH trouble (today: ~25 min), one rerun of a failed arm (<= 88 s), staging retries. Only if everything above is complete before 0:30: Astra's alld4s (3-way, 88 s) | — |
| 0:40 | hard stop for GPU work (session cap ETX_SESS_MAX_SECONDS=1800 from 0:03 ends the runner by 0:33 anyway) | — |
| 0:40-0:55 | FETCH + VERIFY, reserved, never squeezed: ~385 MB (3 x 121 MB stamp+trace, ~20 MB logs); verify the box-side SHA256SUMS, re-derive every verdict and the stage tables, commit | throughput [U] -> 10 min reserved; verify ~2 min |
| 0:55-1:00 | destroy, confirm 0 droplets | — |

GPU work totals 318 s (5.3 min) with D3 and D5, 200 s without; the hour is bounded by connection, prep and fetch,
not by GPU time. D3 and D5 run only if, before the box opens: code committed, default plan 09011c20 / ISA a105cb16
unchanged, ROCm 7.2.4 build at 0 scratch, CPU gates passing, rule written here first. Their slots stay empty otherwise.
Prerequisites for the stamped rerun (CPU, before the box): stage_table must map the D4 grids (shgu / sdown) to stages
and the d4 bodies must write STAMPS2 slots; without that the alld4r0 / alld4r0l1 traces give spans but no stages.

## D3 — four K-chunks of 352 rows (thresholds written before the code, 2026-10-13)

Design (ETX port): gate_up signals each 352-row chunk of h per wave as its rows land (fleet's CHUNK_SIGNAL, 4 chunks,
148 = 37 workers x 4 waves signals per chunk); down no longer waits for the whole gate_up through `E_gu` but for the
routing (`E_norm`, as gate_up) and then, inside its body, for chunk 0, 1, 2, 3 before staging that chunk of h (fleet's
CHUNK_WAIT). Every chunk wait is followed by `buffer_inv sc1` (today's safe acquire: fleet's own `buffer_inv sc0`
does not invalidate L1, run 5). A worker's gate_up precedes its down in its queue (ROUTING_CACHED kept), so the
in-body waits cannot deadlock. Switch: `ETX_DSV2_KCHUNKS=4` (plan) + `-DETX_DSV2_KCHUNKS=1` (tiles) + lever
`kchunk352` (fleet's kExpertKChunk 2048 -> 352); a build with one but not the others fails to compile. Not combinable
with D4 (its expert bodies do not chunk): refused at build time.

| arm | rule (fixed now) |
|---|---|
| alld3 = all five levers + D3, 2-way against `all` (one chunk), 5 rounds, 10 invocations | **GO only if every pair is faster than `all`, median >= 30 us and median > 2 x spread; otherwise DROP.** 32/32 + 27/27 in every invocation. Context, not the bar: fleet's 512-row tiling ran 4.020 ms, the tiling alone cost 0.42 ms because down runs on the same workers (`fleet:docs/STATUS.md:33,68`) |

## D4 slices without shared-XCD stragglers: fixed GPU gate (2026-10-13, before code)

New arm `alld4s`: all five levers + D4 with **32 slice rows** per routed expert. Run five interleaved three-way
rounds **default / alld4r0 / alld4s**, alternating direction between rounds. KEEP only if **every pair is faster
than alld4r0**, the median device gain is **>= 30 us/token and > 2 x spread**, and **32/32 tokens + 27/27 layers**
pass in every invocation. Otherwise DROP (an incomplete run is NOT-COMPLETE). Compare with default as well, but
`alld4r0`, the measured best safe arm, is the deciding baseline.

The deciding per-layer target is <= **8 us** extra work on each shared XCD with slices on: shared down task body
<= **25 us**, shared gate_up <= **36.7 us** (the measured 33.7 us + 3 us), and shared down last completion no later
than routed down. These are GPU targets, not CPU timing claims. The measured 32-row build shortened routed completion
by **7.3 us/layer** (72.9 -> 65.6 us after o_proj), but shared down completed at 77.1 us with a 35.7 us mean body.
The recoverable routed improvement is a ceiling of about **190 us/token [E]** (26 MoE layers x 7.3 us), not a prediction.
For the 3 slices, gate/up plus down reads 1,179,648 weight bytes per shared XCD/layer (bf16, hidden 2048, rows 32);
the two shared XCDs' combined 2,359,296 bytes have a **0.55 us [E]** whole-device bandwidth floor at 4.3 TB/s.
Placement, latency and dependency constraints make that optimistic floor unattainable in general.

CPU-only work: reduce the slices' serial staging and routing overhead, retain the placement checks and rows=0 path,
compile with ROCm 7.2.4, record resources and source/ISA round-trip counts. No GPU is provisioned by this change.
If the gate passes, keep slices; if it fails, retain `alld4r0`. The task-body targets and shared/routed completion
ordering need a GPU trace in addition to the paired verdict.

### D4s implementation and CPU evidence

The optimized path is restricted to `rows=32, spx=3`; other slice sizes retain their old implementation. The graph,
placement table and routing publication are unchanged. Shared gate_up still finishes its shared half before it
waits for routing. Wave 0 broadcasts the wait result, then lanes 0..2 each load one `(id, weight)` pair coherently
into the existing `Scratch.ids/ws`; one workgroup barrier publishes all six words. The 96 slice rows are flattened
across the shared XCD's 37 workers and four waves, using Fleet's `wave_dot<2,4>` and unchanged BF16/SiLU epilogue.
The shared down reuses that routing cache: the compiled queues place shgu immediately before sdown on the same
worker, with no intervening task. D4 already refuses multiple requests and uncached routing. A lowered-plan test
now checks this LDS lifetime, in addition to the existing placement refusals.

Shared down stages all 96 weighted h values in the otherwise-unused `Scratch.row` before its shared-half GEMV;
`stage_vector`'s existing barrier publishes them. It does not overwrite `vec` for each slice. After the shared-half
GEMV, one barrier makes `part` visible to the new row owners. Four lanes now compute each K=32 row, with separate
slice reductions and exactly `((shared + slice0) + slice1) + slice2` before BF16 rounding. This uses all 64 lanes for
16 independent output rows instead of four useful lanes in each 32-lane half-wave. Scaling stays on h, before the
dot. Adding +0 before the shortened shuffle tree preserves the original inactive lanes' signed-zero normalization.
No weight-prefetch overlap with the shared GEMV is claimed or required by this implementation.

Source-level count (per shared task, default 32 rows; a group below is a dependency phase, **not a measured memory
round trip**, and the unchanged shared-half stream is excluded): old sdown had three serial
`metadata -> h staging -> matrix stream` chains, **9 groups**, **6 per-slice barriers**, and **1,536 coherent scalar
metadata lane-loads** (3 slices x 2 words x 256 lanes). New sdown has **0 metadata global reads**, **1 batched h group**
before the shared stream, **1 matrix group** in source after it, and **1 added barrier** after the shared stream.
The three slice matrices have independent addresses, but the compiler may split their issue; the ISA counts below
are the actual static evidence. Old shgu had three serial metadata/GEMV chains; new shgu has one six-word broadcast
phase and one wave-distributed slice batch. Only **6 metadata scalar words per shgu task** are read globally.

Prior evidence was checked in Fleet's pinned `docs/STATUS.md` / `results/` and ETX design section 15: broad prefetch
regressed (4.69488 -> 4.77216 ms in that older Fleet session), and one versus three waves/SIMD measured 38.8 us for
both in the occupancy probe. This change is confined to the short slices; those old results are not timing evidence
for D4s. The original D4 trace was re-read using `ledger.load_etx`: median across layers 3..25, shared gate_up mean
body **33.660 us**, shared down ready **39.030 us**, last done **77.110 us**, mean body **35.734 us**; routed down last
done **65.570 us**, mean body **17.514 us**. The original default trace's down last done is **72.870 us**. These are
old GPU observations, not measurements of the new implementation.

Final ROCm 7.2.4 gfx942 static evidence (`examples/dsv2lite/results/levers/d4s.json`; device assembly + host device
pass + host syntax pass all exit 0, offline container):

| build | VGPR | AGPR | SGPR | scratch B/lane | LDS B | waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|
| default | 256 | 48 | 108 | 0 | 26,668 | 1 |
| alld4r0 before and after | 256 | 90 | 108 | 0 | 26,668 | 1 |
| old all + D4, 32 rows | 256 | 89 | 108 | 0 | 26,672 | 1 |
| optimized D4, 32 rows, no levers | 256 | 65 | 108 | 0 | 26,672 | 1 |
| alld4s | 256 | 104 | 108 | 0 | 26,672 | 1 |

Default plan SHA256 is `09011c207b1e92febf29a2c2efc2d944ef9ad7ac8a0803633579c8444d821f8a`; default ISA is
`a105cb16a414ee0f9f697c9de13205419d09674f5c69f6ee804cf36942377724`. **alld4r0's entire ISA is byte-identical**
before/after, SHA256 `f26a7ff3bdf99aa5d0f7ae1c7a50450bb5bfb801b089a6b3217455f3fb396bf5`; its plan is identical too.

In `build/d4s-final-slices/<all-five-levers>/mk.s`, metadata uses **two vector coherent-load instructions**, each
followed by a wait, with three active lanes (lines 112465 and 112471), then the workgroup barrier at 112478. An initial
thread-0-only implementation compiled to six serialized loads and was replaced before the final build. In shared
down the expert-base pointer fetch is at 108655; the three slice matrix-load sites are 108660, 108665 and 108668,
with a full wait at 108666: **two matrix issue batches**, not one latency round trip. The new all-lever build uses
15 more AGPRs than the old 32-row build; scratch, LDS and reported occupancy are unchanged. These counts do not
predict the GPU body time.

Proof commands (from this worktree, pinned export `~/code/etx-cache/fleet-3fb67a6`):

```sh
bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 default build/d4s-final-default
ETX_DSV2_D4=1 ETX_DSV2_D4_ROWS=0 ETX_CFLAGS=-DETX_DSV2_D4=1 bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live build/d4s-final-r0
ETX_DSV2_D4=1 ETX_CFLAGS=-DETX_DSV2_D4=1 bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 default build/d4s-final-d4
ETX_DSV2_D4=1 ETX_CFLAGS=-DETX_DSV2_D4=1 bash examples/dsv2lite/rocm_build_check.sh ~/code/etx-cache/fleet-3fb67a6 gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live build/d4s-final-slices
.venv/bin/python -m pytest -q tests/test_dsv2_*.py tests/test_coherence.py
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/ds_d4s.py --gpus 0 --dry-run
```

Required suite: **133 passed in 35.53 s, exit 0**. Tests include placement refusal, lowered same-worker queue
adjacency, exact row cover, 4,096 three-slice reduction cases with signed zero/cancellation, BF16 epilogue, KEEP
threshold boundaries, three-way order, correctness failure in any arm, interruption/truncation, and refusal to reuse
a completed report. Dry run: **7 jobs, exit 0**, all generated job commands pass `bash -n`; three builds and preflight
are hard dependencies of the sole paired GPU job. Full logs, per-invocation deadlines and silence probes use the
existing session runner. `ds_d4s.py` is prepared, not run, with a 1,800-second cap and no provisioning/download jobs.

Unknown: GPU correctness (32/32 + 27/27), the <=25 us shared-down / <=36.7 us shared-gate_up targets, shared-versus-routed
completion order, and the paired >=30 us gain against alld4r0. No new performance result, KEEP, or latency estimate
is claimed. GPU hours **0**, GPU cost **$0**. The initial metadata serialization was caught locally; a CPU test's
nonassociativity witness was corrected from 2^24 to 2^25 before the passing full suite. A trace lookup initially
used run2 for the default trace; the reused default trace is in run1. No GPU session failed or was provisioned.

## The paid hour, as built (2026-10-13): `examples/sessions/ds_hour.py`

D3 and D5 are coded and locally proven, so their slots are filled; both are measured on the five-lever arm `all`
(2-way, 59 s each), not on alld4r0: D4's expert bodies do not chunk (D3 + D4 is refused at build time) and D5's
prefetch was proven on `all`. Order: pairs_l1, pairs_alld4r0l1, pairs_alld3, pairs_alld5, trace_default,
trace_alld4r0, trace_alld4r0l1, then the optional pairs_alld4s, then `hash_session` (box-side SHA256SUMS). GPU total
~494 s [from the runner records]; cap ETX_SESS_MAX_SECONDS=1800. Local proof before the box (ROCm 7.2.4, 0 scratch
each): l1, alld4r0l1, alld3, alld5, alld4s, stamped alld4r0, stamped alld4r0l1; default plan 09011c20, default ISA
a105cb16, unstamped alld4r0 ISA f26a7ff3 unchanged.

Rule provenance for the hour's candidates: l1 and alld4r0l1 -- 8d69e52 (before the D2b code, d2e4d54); alld3 --
1f6e3cc (before the D3 code, 04c768c); alld5 -- written in the doc first but committed together with the code
(14ea3b7, Astra brief 06); alld4s -- likewise in one commit with its code (ff15c2c, Astra brief 05). For the last
two the order is the worker's statement, not a separate commit. Expectation recorded for alld5, not a rule: on `all`
every task's `buffer_inv sc1` follows the event wait the prefetch runs before, so warmed L2 lines may be dropped
before use; a DROP there says nothing about D5 on the D2b arm.

## Review 07 (Astra, independent reviewer, 2026-10-13): 11 findings, all addressed before the box

Report: `/tmp/etx-dsv2/answer-07-review.md` (reproductions under `/tmp/etx-dsv2/review07-*`). Fixes, each re-checked:
R01 `timeout --foreground` keeps the measured binary in the job's process group (Astra's repro: no longer survives the
hard stop); R02 gains rounded to 1e-6 us before any threshold (repro: the boundary case now NO-GO as written); R03 fetch
keeps the build records (plans, resources, generated kernels), only binaries / patched sources / objects stay; R04 the
box's SHA256SUMS is verified (`etx/tools/verify_sums.py`), never overwritten; R05 the hash excludes the runner's live
`startup.*` (repro: `sha256sum -c` OK); R06 the hash waits for every job (repro: waits); R07 each GPU job waits for ALL
earlier ones (repro: alld3 waits for l1); R08 staging sends exactly the tracked files, deletes the rest, and the box's
`source` job recomputes the tree hash; R09 preflight refuses inputs other than the pinned measured ones; R10
`examples/dsv2lite/verify_hour.py` re-derives every hour verdict and stage table from the raw files; R11 the D4 stage
split is documented in each D4 stages JSON (topk = routed XCDs only; totals and verdicts exact).

**FETCH + VERIFY in the hour:** `tools/session.sh fetch <ip> <dest>` (verifies the box SHA256SUMS), then
`.venv/bin/python -m examples.dsv2lite.verify_hour <dest>` must print `HOUR VERIFIED` before the box is destroyed.

## Review 08 (Astra re-check of f7b6279): R01-R07, R09-R11 confirmed fixed; R08 open, R12-R14 new -- all fixed

R08 staging cleanup and the box's source check now cover every non-directory entry, symlinks included (real staging
snippet on a used box: both untracked symlinks removed, `import yaml` resolves to the system package, tree hash
equal). R12 a fetch refuses a non-empty destination (postmortem re-fetches with ETX_FETCH_REUSE=1) and the verifier
fails on any fetched file the box did not list (Astra's repro: stale results now `NOT IN THE BOX MANIFEST`, `HOUR
NOT VERIFIED`). R13 verify_hour requires every result ds_hour.py promises: exit 0 = complete and verified, 3 =
verified but INCOMPLETE, 2 = not verified. R14 plain `timeout` again (its deadline kills a build's compiler
children: repro 0.30 s, no survivor), and the runner kills every descendant of a job at a stop -- /proc tree taken
before the kill -- so the group `timeout` creates cannot survive the hard stop (both R01 repros: False).

## Review 09 (Astra re-check of 2ef1e58): R12 still open, R15 new -- both fixed in 0a88e45

R12 (postmortem re-fetch over an earlier capture from a new box with no SHA256SUMS kept the old manifest and results,
and `verify_hour` certified them: HOUR VERIFIED, exit 0). Fetch now deletes the destination's SHA256SUMS and
SHA256SUMS.local before any transfer and mirrors the box with `rsync --delete --delete-excluded`. Astra's repro with a
failing transfer and with a working Linux rsync: both end `HOUR NOT VERIFIED`, exit 2; with the working transfer only
the new box's `startup.NEW` remains. R15 (the hard stop killed from a /proc snapshot while the tree could still fork,
so children forked during the kill escaped: Astra's repro 2/4/4 survivors). The runner now SIGSTOPs the job's group and
every descendant, re-scanning until no new pid appears, then SIGKILLs; it is a child subreaper (Linux prctl) so orphans
stay its descendants, and the hard stop sweeps its own tree last. Repro: 0/0/0 survivors. A Linux-only regression test
(`test_hard_stop_kills_children_forked_during_the_stop`) fails 3/3 on 2ef1e58 and passes 3/3 on 0a88e45. Local gates:
238 passed, 1 skipped (that test, macOS has no /proc); `ds_hour.py --dry-run` in the container unchanged.

## Review 10 (Astra re-check of 0a88e45): R15 and R01-R11, R13, R14 confirmed; R12 nested case -- fixed

Astra confirmed R15 (fork race: 0/0/0 survivors; ten extra two-lane per-job-timeout and hard-stop cases: no late
marker, no live or stopped survivor, no cross-lane kill) and both top-level R12 paths. Remaining: a failed re-fetch
removed only the top-level manifests, so an earlier capture nested as `session-*/` (the runner's layout when its log
directory is taken) still verified. Fetch now deletes every SHA256SUMS / SHA256SUMS.local under the destination before
the transfer. Astra's nested repro, failed transfer: `nested_old_box_manifest_survived False`, `HOUR NOT VERIFIED`,
exit 2 (was `HOUR VERIFIED`, exit 0); working transfer: `nested_old_session_removed True`, only `startup.NEW` left.
First fetch and golden-cache copies unchanged (`review10-fetch-first-and-golden`: identical, verified, preserved).

## Review 11 (Astra re-check of 18f4f51): R12 symlinked destination -- fixed

Nested and top-level R12 paths, path safety (outside targets, whitespace/newline paths, non-manifests) and first fetch /
golden cache confirmed. Remaining: `find "$DEST"` did not enter a destination that is itself a symlink (no trailing
slash), so a failed re-fetch left that capture certifiable. Now `find -H "$DEST" -mindepth 1 ...` (the starting path is
followed, links below it are not; DEST itself is never a target). Astra's symlink repro: `current` and
`current-trailing-slash` both `verify_hour_exit 2`, `old_box_manifest_survived False` (was exit 0, True); all PATH
SAFETY checks True; macOS `find` with the same flags leaves no manifest.

## K1: shared kv_a on the kept arm, and two lever ablations (thresholds written before the code, 2026-10-14)

**Why K1.** Kept arm `alld4r0l1`, retained stamped trace (`results/2026-10-13-hour/trace_alld4r0l1.*`), ledger
mean MoE layer 124.0 us [C]. The handoffs are tight: mean wait 0.5-2.6 us per task. The time is in bodies, and the
largest body excess without any lever, open or closed, is qkv: 21.5 us against a 3.5 us byte floor.
- Its stamps: norm 2.2 + gamma 1.2, then q GEMV 6.7, then kv_a GEMV 5.3 us, in sequence.
- Every XCD computes all 576 kv_a rows (OPS-dsv2.md:17,52,149: 18.9 MB requested for 2.36 MB unique).
  MODEL-dsv2.md:154 explained the qkv stream excess as issued bytes: a replay at 160-180 KB/task gave 12-16 us,
  and 14.6 us was stamped [M].
- The over-issue half was tried (gemv_no_terminal: dead, L2 hits). The replication half was never measured under ETX.
  Fleet's own default is `FLAG_KVA_SHARED`: kv_a split over the XCDs behind a global event (ETX_Technical_Design.md:1063).

**K1 change.** `ETX_DSV2_KVA=shared` (default `replicated` = today, byte-identical):
- each XCD's qkv tasks compute 1/8 of kv_a into one buffer (fleet's FLAG_KVA_SHARED path);
- the qkv event becomes global (1,), 296 arrivals; attention and q_absorb wait on it and read the single kv_a buffer.
Safety: attention's kv_a loads are already l1 agent-scope loads (levers/l1_loads.py), and a cross-XCD hand-off goes
through the last-arriver write-back (the loadscope cl1 case, 0 stale words) like o_proj -> router today.

**Expected, not promised.**
- qkv: kv_a issued bytes per task fall from ~98 KB to ~8-12 KB, so qkv stream -5..-7 us/layer [E].
- Costs: a global handoff instead of an XCD-local one, plus attention reading kv_a remotely (B2: a fresh remote
  64 KB costs 3.9-6.0 us; kv_a is 2.3 KB). +1..+4 us/layer [E].
- Net sign [U]. It decides on the GPU.

**Rule (K1):** arm `alld4r0l1kvs` vs `alld4r0l1`, five interleaved rounds, device time, no stamps.
**KEEP** if every pair is faster, median >= 30 us and > 2 x spread; otherwise **DROP**. 32/32 + 27/27 in every invocation.

**Ablations (new interaction question, not a 14 us assumption).** `alld4r0l1noQ` (without gemv_no_terminal_all) and
`alld4r0l1noA` (without argmax_tree), each vs `alld4r0l1`, same five rounds. **REMOVE** the lever from the kept
recipe if every pair is faster and median > 2 x spread; otherwise it stays. Low ceiling: argmax measures 5.1 us in
total. They ride the same box after K1, never instead of it.

### K1 stage-data note (from the retained kept trace; no new capture)
`python -m examples.dsv2lite.ledger --etx <hour>/b_alld4r0l1_stamps/plan.json <hour>/trace_alld4r0l1.trace.bin
<hour>/trace_alld4r0l1.stamps.bin`. The trace is stamped, so it is attribution only, not the uninstrumented baseline.
Mean MoE layer (layers 2-25) = 124.0 us [C]:
- handoff -19.1 / ramp 27.2: they cancel, because placement lets fast XCDs start early;
- staging 9.3, prologue 10.7, tail 9.0;
- bandwidth floor 38.4, stream above floor 48.5.
The per-type table (`trace_alld4r0l1.raw.log`) shows mean waits of only 0.5-2.6 us per task, and every phase starts
within ~0.5 us of the previous one's end. The time is in bodies [M]. Above-floor per layer:
- qkv 21.5 vs floor 3.5 (norm 2.2 + gamma 1.2, then q 6.7, then kv_a 5.3 us, in sequence);
- attention 18.1 vs 0.27;
- merge 11.4 vs 0.49 (W_uv GEMV 5.2 us per task);
- router 12.6 vs 0.06;
- gate_up 32.7 vs 21.5;
- down 18.1 vs 10.7.
These are accounting residuals, not measured stalls. K1 is the one body excess with a recorded cause
(MODEL-dsv2.md:154) and no lever, open or closed. merge's W_uv GEMV and attention have no measured cause yet [U].

### K1 local evidence (branch wf/dsv2-next; ROCm 7.2.4, `rocm_build_check.sh <fleet export> --arm <arm>`)
- **Default byte-identical.** Plan 09011c20 and kernel 751293fa unchanged (tests/test_dsv2_kva.py). The kept arm
  `alld4r0l1` gives plan 3379ec89 on main and on the branch, and its `mk.s` and `host_dev.s` are byte-identical.
  The K1 entry points sit behind `-DETX_DSV2_KVA_SHARED=1`; always-present entry points had moved the kept ISA in
  78 lines, so they were guarded.
- **K1 `alld4r0l1kvs`:**
  - plan f23f959e; 256 VGPR / 71 AGPR / 0 scratch / 1 wave;
  - the qkv event is DEVICE scope in all 27 layers;
  - `buffer_wbl2` 172 -> 226: the +54 is the per-layer release write-back K1 adds, its cost side in the code;
  - dispatch calls only the `_kvs` entry points.
- **Ablations:**
  - `alld4r0l1noQ`: 87 AGPR (+17 vs kept), static loads 3344 -> 3078;
  - `alld4r0l1noA`: 70 AGPR, barriers 252 -> 273 (+21, the argmax tree's 9 -> 2 reversed);
  - both 0 scratch, 1 wave. Static counts do not give issued loads; the GPU decides.

### K1 GPU queue (registered; needs the owner's approval; `examples/sessions/ds_k1.py`)
- **Order:** prep chain (fresh droplet: 3 min 59 s to preflight, 2026-10-13 [M]) with 5 parallel builds ->
  `pairs_alld4r0l1kvs` -> `pairs_alld4r0l1noQ` -> `pairs_alld4r0l1noA` -> `trace_alld4r0l1kvs` (attribution only).
- **Measurement:** five interleaved rounds per pairs job on one machine, deciding timings unstamped, 32/32 + 27/27
  in every invocation; rules as above.
- **GPU time [E]** from the 2026-10-13 runner records: 3 x 10 invocations x 5.9 s + 1 traced token ≈ 3.1 min. The
  runner hard stop is 1800 s. The paid hour on 2026-10-13 ran 10 min 53 s and was billed ≈ 50 min [M].
- **Proposed cap:** one MI300X box <= 1 h, about $2 at $1.99/h [E]. Delete the box at the end, fetch into a new
  directory, seal the record.
- **Expected vs unknown:** K1's net sign is [U] (-5..-7 us qkv against +1..+4 us handoff/attention per layer, both [E]).
  The ablations' ceilings are low: argmax is 5.1 us in total; terminal loads cost 14 us in an earlier session, on
  another combination. Nothing is promised.

## M1: merge's W_UV GEMV on all four waves (threshold written before the code, 2026-10-14)

**Why.** In the kept trace, merge takes 11.4 us per layer against a 0.49 us byte floor [C]. Its stamped stages are
coefficients 0.72, partial fold 1.90, then the W_UV GEMV 5.24 us per task [M, stamped]. That GEMV is
`gemv_rows(..., n_rows = 8, K = kv_lora = 512)`; K <= 512 picks `gemv_rows_pipelined<8, 1>`: 8 rows per wave per
group, groups interleaved over waves (`k = 8 * wave`). A task owns exactly 8 rows, so **wave 0 loads, reduces and
stores all eight while waves 1-3 sit idle** (OPS-dsv2.md:82, :146 noted it; no lever was ever made, open or closed).

**Change, lever `merge_uv_waves`:** merge calls `gemv_rows_pipelined<2, 1>` directly, so the 8 rows go 2 per wave.
- Every row is computed by the same operations: `fma8` of the lane's one 16-byte chunk, then `wave_sum`, then lane
  0's bf16 epilogue. The lane-to-chunk mapping depends on K only, so the outputs are **bit-identical** by construction.
- Default off. The kept recipe's other levers (including gemv_no_terminal_all's loop patch) apply as before.

**Expected [E], not promised.** Wave 0's serial chain shrinks from 8 row loads + 8 wave sums + 8 epilogue stores to
2 + 2 + 2. Estimate: 1-3 us per merge task. M3 ends each merge task, so that is up to 27-80 us/token. If the time
is a single memory round trip that both versions pay once, the gain is ~0 [U].

**Rule (M1):** arm `alld4r0l1mw` (kept + `merge_uv_waves`) vs `alld4r0l1`, five interleaved rounds, device time, no
stamps. **KEEP** if every pair is faster, median >= 30 us and > 2 x spread; otherwise **DROP**. 32/32 + 27/27 in every
invocation.

### M1 local evidence (branch wf/merge-waves2; ROCm 7.2.4)
- **Device build:** `rocm_build_check.sh <fleet export> --arm alld4r0l1mw`: plan 3379ec89 (= kept: the change is tile
  code only); 256 VGPR / 78 AGPR (+8 vs kept) / 0 scratch / 1 wave.
- **Probe:** `merge_and_uv` compiled alone with the kept recipe's GEMV + l1 levers, with and without M1 (lines after
  merge's last barrier = one wave's M3).
  - assembly lines 1082 -> 365;
  - weight loads 24 -> 6;
  - DPP/bpermute reduction ops 96 -> 24;
  - s_waitcnt 95 -> 30;
  - FMAs 144 -> 36;
  - the probe's VGPR 124 -> 64, 0 scratch.
- **Bit identity:** same per-row operations (fma8 on the lane's chunk, wave_sum, lane-0 epilogue; the lane -> chunk map
  is K's). Row coverage over waves is tested (each row once; 1 busy wave -> 4).
- **Queued** in `examples/sessions/ds_k1.py` right after K1, with the same kept baseline.

## KM: K1 and M1 together (threshold written before the code, 2026-10-14)

**Why.** On 2026-10-14 (`examples/dsv2lite/results/2026-10-14-k1`) K1 and M1 each passed KEEP against `alld4r0l1`,
measured separately: K1 +41 us (spread 8), M1 +66 us (spread 21) [M]. Their sum is not a measurement. K1 changes
where kv_a is computed and read (qkv -3.9, attention +1.7 us/layer in the stamped token); M1 changes merge's W_UV
GEMV, which runs inside the attention tasks that now read kv_a across XCDs. The two can interact.

**Arm `alld4r0l1kvsmw`:** the K1 build (`-DETX_DSV2_KVA_SHARED=1`, `ETX_DSV2_KVA=shared`) plus the lever
`merge_uv_waves`. No other change.

**Expected [E], not promised.** Additive: about +107 us vs `alld4r0l1`, about +41 us vs `alld4r0l1mw`. If M3's
shortened chain is hidden behind the cross-XCD kv_a read that K1 adds, the gain over `alld4r0l1mw` shrinks toward 0.

**Rule (KM):** five interleaved rounds, device time, no stamps; each round runs `alld4r0l1`, `alld4r0l1mw`,
`alld4r0l1kvsmw` (order reversed on even rounds, as every pairs study). 32/32 + 27/27 in every invocation.
- **KEEP KM** (the new kept arm is `alld4r0l1kvsmw`) if every pair is faster than `alld4r0l1mw`, median >= 30 us
  and > 2 x spread.
- **Otherwise** the new kept arm is `alld4r0l1mw` (M1 alone, KEEP on 2026-10-14), and K1 is recorded as not adding
  on top of M1.
- The gain vs `alld4r0l1` is reported for attribution only; it decides nothing.

### KM local evidence (ROCm 7.2.4, `rocm_build_check.sh --arm alld4r0l1kvsmw`)
- plan f23f959e (= K1's: M1 is tile code only); 256 VGPR / 79 AGPR (K1 71 + M1's 8) / 0 scratch / 1 wave.
- Session `examples/sessions/ds_km.py`: the prep chain (skips what the box holds), three builds, one pairs job.
