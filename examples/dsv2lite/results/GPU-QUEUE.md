# What still needs a GPU, and why nothing else on the list belongs here

Only device-only work. Everything with a CPU alternative is already done and is not repeated here.
One box, one session: `examples/dsv2lite/stage_inplace.sh <ip>`, then
`ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/ds_inplace.py`.
About 97 invocations at about 8 s each is about 13 minutes of GPU time; staging the model and weights on a
fresh box adds about 15 minutes. Cap 7200 s. Cost at $1.99/h: **one hour, $1.99**.

## E1 — the price of the safe acquire (D1)

- Measure: the safe sc1 build against the relaxed build, at stride 32, interleaved pairs.
- Verdict rule: record the paired median delta and its spread. Predicted **safe is 235±30 us slower**.
- Why GPU only: the cost is the per-task `buffer_inv sc1` and the producer `wbl2` traffic on a real L2 and
  across real XCDs. No CPU model reproduces either number.
- CPU part already done: the sessions now build safe by default (`2e7c20f`); the ISA and resource account
  are in the commit.

## E2 — coherent payload loads (D2)

- Measure: the coherence arm (`pairs_coh`) against relaxed, interleaved pairs.
- Verdict rule: **accept if coh lands within max(2 x spread, 20 us) of relaxed**, and every invocation
  passes 32/32 HF-greedy tokens and the 27/27 layer gate.
- Why GPU only: this is a memory-model change. The CPU side proves the emitted code (per-task invalidates
  dropped, `buffer_inv sc1` 276 -> 3, producer write-back kept) but cannot prove the data is correct or
  fast under real cross-XCD traffic.
- CPU part already done: 15 tests plus `verify_lever coherent_loads` (`6535047`).

## E3 — shared-XCD placement during routing (D4)

- Measure: the `d4` arm against default (`pairs_d4`) plus one `d4trace` stamped pass.
- Verdict rule: **keep if the median gain exceeds 30 us/token**; expected about 33 us from the per-row
  model, so this is close to the bar and may fail it. Numerics change, so 32/32 and 27/27 decide first.
- Why GPU only: the gain is the routing-to-body start timing, and the in-body wait only exists on the
  device. The placement refusal and its tests are CPU work and are already committed (`57277ed`).
- Note: 64 rows would lose; the 32-row configuration is the one under test.

## E4 — per-stage attribution at the new configuration

- Measure: the stamped (STAMPS2) trace for the accepted coh and d4 arms, against the stored default table.
- Why GPU only: the stamps are device clocks; the stage table is the only way to see which stage the
  change actually moved.
- This is what turns "the arm is faster" into "this stage moved by this much".

## E5 — which compiler version the record uses (ROCm 7.2.3 against 7.14)

- Measure: the same generated sources built by 7.2.3 and by 7.14, interleaved. Already measured once:
  **7.14 is 62 us slower** (3.708 against 3.646 ms), every pair, 48 AGPR against 60.
- Why GPU only: it is a codegen effect on the device; no CPU proxy.
- Decision this buys: which compiler every future arm is measured with, and whether the 2026-09-27 record
  (3.473 ms) is even comparable.

## E6 — ride-along only, 2 GPU minutes: routing persistence

- Measure: how often a sequence selects the same six experts twice in a row, from the stride-32 rebuild
  (`routing_stats.py --dump`).
- Why GPU only: it needs a real greedy trajectory.
- **This is not a reason to open a box.** The prediction study already ruled the idea out on the numbers
  (`results/EXPERT-PREFETCH.md`, NO-GO: the bar is 38.7% for a 6-expert predictor and the measured class
  reaches 14-18%). Attach it to a session that is happening anyway, or not at all.

## Not on this list, and why

- Anything with a CPU proof: the acquire flag change, the coherence plan, the placement refusal and its
  tests, the whole-experiment log, the reproduce entry point. All committed, all verified on CPU.
- The expert-prefetch lever: NO-GO on its own threshold before any code existed.
- D3 (four K-chunks) and D5 (early weight loading): parked. Both measured negative or neutral before, and
  neither is on the owner's priority list.
