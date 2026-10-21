# Kernel-comparison session pre-registration

This is the pre-registration of the kernel-comparison session, copied verbatim
from the original `o55-kernel-compare.json` result on 2026-10-09. Section 4
is the authoritative threshold block checked against the session docstring
and report output. Historical estimates below are not new measurements.

## 4. Thresholds, written before the run

Define the gain g = (ETX standalone median − best vendor standalone median), per layer; a positive g means the vendor kernel is faster.

- **Row closes, the tile is good enough:** g < 0.63 µs, which is 0.030 ms per token ÷ 48, the band an A/B cannot resolve (PLAN:154). For lm_head the limit is 30 µs.
- **Vendor wins:** g ≥ 0.63 µs in 3 of 3 pairs, and the median g is more than twice the spread (the rule at `examples/vllm_llm/pairs.py:33-34,138-143`). This is recorded only. Tuning the tile needs the owner's request (`AGENTS.md:7`).
- **Worth building link-mode import:** all three conditions must hold.
  - g ≥ 1.04 µs, which is 0.050 ms ÷ 48.
  - The vendor kernel's source is available.
  - The vendor time plus its import cost is at most the ETX in-situ body minus 1.04 µs. Measured import costs range from +1.04 µs (wvSplitK) to +6.62 µs (fused_moe up).

  Applied to the rows: qkv requires vendor + cost ≤ 8.30; o_proj ≤ 4.31; router ≤ 3.98; moe_up ≤ 16.53; moe_down ≤ 8.84; lm_head ≤ 106.7 per step [E].
- **Noise.** Session-to-session drift is at most 0.12 µs (A against C), so 0.63 µs is at least 5 times the drift, and 3 pairs is enough (the PLAN:154 minimum). A row with mixed signs closes unless the owner asks for 5 more pairs.

## 5. Where this runs

**Decision: (b), a separate micro-session**, `examples/sessions/kcmp.py`. It runs on the nvref VM only after nvref's `report` job has finished and its results have been fetched. `nvref.py` stays byte-unchanged, so its pre-registration holds and the risk to its 36 jobs is zero.

Option (a) would have to edit the job barriers at `nvref.py:110-140` and redo the P4-P6 checks (`docs/qwen/SPEC-nv-reference-2026-10-08.md:202-206`). It would also put a known rocprofv3 teardown segfault (`docs/qwen/RESULTS-2026-10-01.md:112`) next to the timing rounds.

Jobs, in ABBA order:
1. `kc_etx_1`
2. `kc_vnd_1`
3. `kc_vnd_2`
4. `kc_etx_2`
5. `kc_etx_3`
6. `kc_vnd_3`
7. `kc_report`, on the CPU.

Estimated cost:
- About 12 s per ETX job and about 90 s per vendor job, roughly 5 GPU minutes in total [E].
- About 10 extra minutes of VM life, about $0.50 at $2.99 per hour (SPEC:211). It is $0 extra if billing has a one-hour minimum and nvref ends near its 35-minute estimate.
- Keep nvref's hard stop of 55 minutes of VM life (SPEC:213). If nvref ends after 45 minutes, skip the micro-session and run it later on a fresh VM, about 20 minutes and $1.00 to $2.99 [E].
- The A and C traces are reused if the VM is the Hot Aisle bare-metal type. On the VF box, add `kc_vllm` (about 5.5 minutes, `prof.out:1` against `profrun.log:1`).

Local work before the session:
- The parser, tested against the A and B CSVs.
- A CPU dry run of the vendor script.
- `gpu_jobs --dry-run`.

**Smallest version:** `kc_etx_1` plus `kc_vnd_1`, about 2 GPU minutes. A single pair settles only the rows where g is clearly below 0 or above 2 µs. Three pairs cost only about 3 more minutes, so I recommend three.
