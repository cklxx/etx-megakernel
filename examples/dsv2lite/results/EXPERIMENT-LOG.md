# DeepSeek experiment log — four columns

Every experiment run on DeepSeek-V2-Lite / DeepSeek-Coder-V2-Lite, in both repositories.
`ETX` = the port under `examples/dsv2lite` (the target). `fleet` = `fleet-mi300x` (the reference kernel).
GPU column: the session that measured it. GPU numbers are paired medians (five interleaved rounds) against the arm's registered reference in the same session: `default`, `all` or `alld4r0`, as each row says. Milliseconds are never compared across sessions.

Certification correction, 2026-10-15: the historical B1/B2/B4 comparisons below lack the short-call vLLM gates and
producer-bound reference records now required. Their arithmetic is reproducible, but they are not certified by the
repaired protocol. B8 failed hidden-state checks; its later four-label diagnostic ran one binary four times and is
withdrawn. See [complete status](../../../docs/REPORT-STATUS-2026-10-15.md) and
[the next correctness experiment](../../../docs/NEXT-GPU-2026-10-15.md).

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| gemv_no_terminal | Drop the terminal dummy loads at the end of the GEMV K-loop. The byte account had counted them as issued HBM bytes, worth about 120 us on qkv | yes — session 2026-10-13, arm `Q` | −3 us. Dead. The removed loads re-read a row the workgroup had just loaded: L2 hits, not HBM bytes |
| gemv_no_terminal_all | Same change at every GEMV site, not only the last one, for about 310 us on the byte account | yes — arm `Qall` | −3 us alone, and −14 us when kept in the combination. Dead on its own. It was dropped from the 2026-10-13 run-1 follow-up, but the five-lever recipe the kept arm builds still contains it (`LEVERS_ALL`, examples/dsv2lite/arms.py) |
| topk_rank | One rank pass over the logits instead of repeated scans. The stage measures 128 us; the rank re-derivation said about 89 us | yes — arm `topk` | +66 us, every pair faster. 74% of the expectation. Kept |
| norm_tree_all | Do the norm reduction once for every consumer instead of once per consumer. in_norm measures 168 us against 128 us expected | yes — arm `Nall` | +37 us, spread 7. Kept |
| norm_tree | The same reduction sharing without the all-sites variant | yes — arm `N` | +30 us. Kept as attribution only |
| norm_gamma_early | Hoist the gamma into the earlier pass so the later pass disappears | locally only (CPU gates + ROCm resources) | PASS locally, 0 scratch, AGPR 72. Never measured alone |
| attn_prologue_live | Compute the attention prologue live rather than materialising it. prep measures 129 us against 65 us expected | yes — arm `prep` | +27 us, every pair faster. 41% of the expectation. Kept |
| argmax_tree | Tree-shaped argmax instead of a linear scan. The stage only measures 4.7 us | yes — arm `argmax` | +2 us with spread 21: inside the noise. Dead on its own, but still in the five-lever recipe the kept arm builds (`LEVERS_ALL`) |
| attn_cache_double_buffer | Double-buffer the attention cache to overlap its fill with use | locally only, three variants | REJECTED by the local gate every time: +160 B/lane scratch, +60 AGPR. Never went to GPU |
| stamps (STAMPS2) | Instrument each stage with one stamp per cache line, so every stage has a measured value rather than an inferred one | yes — trace builds in the same session | No speed verdict, by design. Produced the per-stage table the expectations are judged against. Paid hour 2026-10-13: one stamped token each, spans default 3997.8 us, alld4r0 3770.0 us, alld4r0l1 3491.3 us (`2026-10-13-hour/stages_*.json`) |
| combined 5-lever arm (`all`) | The five selected levers together (not every lever passed alone): topk_rank + norm_tree_all + attn_prologue_live + argmax_tree + gemv_no_terminal_all | yes — arm `all` | +92 us, all five pairs faster: GO. Then re-measured without the dead lever |
| `all` minus gemv_no_terminal_all | The same combination with the lever whose byte basis was wrong removed | yes — follow-up, 5 rounds | **+108 us**, spread 7, holds. The best ETX result of that session (2026-10-13 run 1); superseded by `alld4r0l1` below |
| Q-all inside the combination | Check whether a lever that is neutral alone still steals resources from the others | yes — follow-up, 5 rounds | Costs 14 us in the combination. Dropped |
| ROCm 7.2.3 against 7.14 | The box ran 7.14 while the local resource records were built on 7.2.4. Same generated sources, two compilers | yes — follow-up, 5 rounds | **7.14 is 62 us slower** (3.708 against 3.646 ms), every pair holds. The 7.2.3 build uses 48 AGPR |
| fleet F: per-XCD local expert fold | Do the expert fold where the data already sits instead of crossing XCDs, per the fleet trace | yes — fleet session 2026-10-12 | REJECTED: the QKV stage goes 367.1 → 374.6 us (+7.5) |
| fleet K: attention variant | Repay the gather and publication waits at the attention body-ready point | yes — fleet session 2026-10-12 | REJECTED: attention goes 210.55 → 219.62 us (+9.1) |
| fleet: six switches, all levers | Port the port's own deficits back into the reference kernel, one switch each, default OFF | locally only | 22/22 gates, 24/24 default artifacts byte-identical. GPU never measured; the repository was misdirected |
| fleet: cached working set | Keep the hot weight set resident instead of streaming it, on the theory that reuse beats a fresh read | yes — earlier fleet session | 18% slower. Closed, no further work |
| D1 safe acquire, cost attribution | The safe `buffer_inv sc1` acquire is the default; `relaxed` (the old sc0 acquire) prices what safety costs | yes — paid hour 2026-10-13 (`2026-10-13-hour`), `pairs_l1` rounds | Safe default costs **232 us** (spread 13) against relaxed. Attribution only: relaxed is unsafe, never a kept arm |
| D2b `l1`: agent-scope payload loads | Load payloads with `sc1` (past L1, from the XCD's L2): no per-task acquire invalidate and no added write-back | yes — paid hour 2026-10-13, arm `l1` (vs default and relaxed) | +330 us (spread 13) vs default; l1 is **99 us faster than relaxed** (spread 12). Registered **NO-GO**: part (b) required l1 within ±26 us of relaxed, and being faster also fails it. The verdict stands as registered; the favourable measurement is recorded, not reinterpreted |
| `alld4r0l1`: l1 on the five-lever + D4 zero-slice arm | Apply l1 to the best arm: `all` + D4 with 0 slice rows | yes — paid hour 2026-10-13, arm `alld4r0l1` (vs alld4r0 and default) | **KEEP**: +334 us (spread 19) vs alld4r0, +498 us (16) vs default, every pair faster. **3.238 ms device / 3.272 ms wall** (five-round medians: 309 / 306 tok/s), 32/32 tokens, 27/27 layers. The kept build of that session, superseded by KM below |
| D3 `alld3`: four 352-row K-chunks | Start `down` on gate_up's finished rows instead of after all of them | yes — paid hour 2026-10-13, arm `alld3` (vs all) | **NO-GO**: −741 us (spread 19), every pair slower (4.386 vs 3.647 ms). Measured on `all`, not on the kept arm |
| D5 `alld5`: compiler weight prefetch | Load immutable, routing-independent weights before the event wait | yes — paid hour 2026-10-13, arm `alld5` (vs all) | **DROP**: +2 us (spread 9), pairs [2, −1, 8, 0, 8] include a loss and a tie. Measured on `all` only; its interaction with l1 is unmeasured |
| `alld4s`: batched routed slices on the shared XCDs | Give the shared XCDs 32-row routed slices, batched, to remove serial staging | yes — paid hour 2026-10-13, arm `alld4s` (vs alld4r0 and default) | **NO-GO**: −13 us (spread 12) vs alld4r0, every deciding pair slower (+159 us vs default, which is not the deciding reference) |
| K1 `alld4r0l1kvs`: kv_a computed once | Every XCD computes all 576 kv_a rows (18.9 MB requested for 2.36 MB unique). Compute 1/8 per XCD behind a global qkv event (Fleet's own default), cutting qkv's issued bytes; attention then reads kv_a across XCDs | yes — session 2026-10-14 (`results/2026-10-14-k1`) | **+41 us device, KEEP** (pairs 40/40/41/48/44, spread 8; wall +41). Every run 32/32, 27/27. Stamped token vs the kept arm's 2026-10-13 trace: qkv 434.2 -> 329.8 us, attention 308.7 -> 354.0 us (cross-session, attribution only) |
| M1 `alld4r0l1mw`: merge's W_UV GEMV on all four waves | The merge task's 8 W_UV rows all land on wave 0 (`gemv_rows_pipelined<8, 1>`) while 3 waves idle; 5.24 us per task stamped. 2 rows per wave: same arithmetic, bit-identical outputs | yes — session 2026-10-14 (`results/2026-10-14-k1`) | **+66 us device, KEEP** (pairs 76/65/55/74/66, spread 21; wall +68). Every run 32/32, 27/27. Inside the 27-80 us estimate. Measured alone here; the subsequent KM row measures K1 + M1 together |
| KM `alld4r0l1kvsmw`: K1 and M1 together | K1 (+41) and M1 (+66) passed separately; M1 changes merge's GEMV inside the attention tasks that K1 makes read kv_a across XCDs, so their sum is not a measurement. Decided against M1 alone (rule bf92fd4, written first) | yes — session 2026-10-14 (`results/2026-10-14-km`) | **+52 us vs M1 alone, KEEP** (pairs 48/52/50/58/54, spread 10); +120 us vs alld4r0l1 (spread 6). Every run 32/32, 27/27. **New DeepSeek kept arm: `alld4r0l1kvsmw`, 3.068-3.075 ms device** |
| Batch 2 / 4 / 8 `x2` `x4` `x8` against vLLM at the same B (docs/BATCH-2026-10-14.md) | The streaming byte models estimate 9.18 ms for replication at B = 8 and 4.22 ms for weights-once [E]. Actual traffic is unmeasured, so these are conditional estimates, not physical bounds or a proven reason to build column-batched tiles. Throughput and per-request latency are decided separately | yes — sessions 2026-10-14 (`results/2026-10-14-batch-20261014T142757Z`, `-20261014T134920Z`) | Historical arithmetic, with the certification limits above: B = 1 **+32.98%** (274 vs 206 tok/s), B = 2 **+13.95%** (403 vs 354), B = 4 **−27.70%** (410 vs 567), every number's spread under 6.2%; ETX gates x2 and x4 PASS, **B = 8 not complete** (r5/r7 hidden prefixes 3/7 despite 32/32 tokens) |
| ablation `alld4r0l1noQ` | gemv_no_terminal_all was dead alone and cost 14 us inside `all`, yet it ships in the kept arm: does removing it help the new combination? | yes — session 2026-10-14 (`results/2026-10-14-k1`) | **STAYS**: removing it is slower, -29 us (pairs -29/-29/-23/-32/-22, spread 10). Inside the l1 + D4r0 recipe the lever now helps |
| ablation `alld4r0l1noA` | argmax_tree was +2 us (noise), yet it ships in the kept arm | yes — session 2026-10-14 (`results/2026-10-14-k1`) | **STAYS**: removing it is -13 us (pairs -13/-6/-21/-6/-14, spread 15): every pair slower, but within 2 x spread |

## The two accounting lines

- expected for the ETX combination: 531–533 us. Delivered: 92 us, then 108 us after dropping the dead lever. 17–20% realised.
- Fleet's older counter capture measured 5.20 GB/token; it is not ETX's current traffic. The ETX bulk-payload model at context 1024 is 4.934 GB/request, or 1.148 ms at an assumed 4.3 TB/s. Cache reuse, spills, dependencies and other work prevent treating this estimate as an achievable runtime or a measured HBM bound.
- fleet 3.568 ms = 280 tok/s was the best measured on the ETX box in an earlier session (2026-10-13 run 1). **The current kept ETX arm is `alld4r0l1kvsmw`: 3.068-3.075 ms device, 2026-10-14 (`results/2026-10-14-km`), five rounds, every run 32/32 and 27/27.** The 2026-10-13 paid hour measured its predecessor `alld4r0l1` at 3.238 ms device / 3.272 ms wall, so a tok/s figure quoted from that hour describes `alld4r0l1`, not the kept arm. Fleet was not timed alongside either arm, so no speedup over fleet is claimed across sessions.

## Registered preparation: two existing gaps, 2026-10-15

No GPU execution or spend. [Preparation and acceptance](../../../docs/DEEPSEEK-PREP-TWO-GAPS-2026-10-15.md).

- **X8 correctness discriminator:** `examples/sessions/x8_correctness.py`, 3600-second hard cap. Compare bound r5/r7
  on x1km (x8 numerical recipe, one request, no D4), then kept B1, then x8 twice. Strict token AND hidden-state gates.
  Go: a complete reproducible classification. No-go: incomplete provenance, metrics, or inconsistent repeats.
  A failure is diagnostic evidence, not a throughput result. Original lost payload recovery is not claimed.
- **B4 attribution and corrected baseline:** `examples/sessions/ds_b4_attribution.py`, separate 7200-second cap.
  Gated B1/B2/B4 traffic, body/wait spans, exact-prefix routing, five corrected vLLM pairs, and a local same-kernel graph eligibility gate.
  The cached-LDS handoff currently refuses separate graph launches; the two-baseline speed claim stays INCOMPLETE.
  Go: complete gates, exact identities and measured attribution inputs, then apply the existing registered rules.
  No-go: absent counters, trace/routing bindings, short-call gates, or incomplete pairs. No new kernel experiment.

Final SOL_CLOSE local acceptance: the remaining 14 graph tests passed; complete suite **2168 passed, 14 skipped in 508.28s (0:08:28)**, exit 0,
with no deselection. The independent final-code review status and exact raw evidence are in the linked preparation index.
DeepSeek implementation `a0c3e72` is unchanged by closure. A full-suite failure required the separately committed
process-group cleanup fix `fcb5f6`; before/after regressions and the failed first full run are retained. Both diagnostic queues remain unexecuted. DigitalOcean tariff,
billing minimum and spending approval remain unknown. Same-kernel replay is still UNAVAILABLE; no fusion claim.
