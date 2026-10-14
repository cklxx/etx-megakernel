# Qwen3-30B-A3B: readiness for the next GPU session (2026-10-14)

Labels: [M] measured, [E] estimate, [U] unknown. bf16 only. **No GPU has been used for anything in this file.**

**Update (later on 2026-10-14):** the box staging in section 3 now runs inside the session as jobs
(`examples/sessions/qwen_g6.py` stage_env / stage_vllm / stage_dl / stage_pack): the HF download is pinned to revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` before anything is fetched, closing section 3's open gap. The joint run with
DeepSeek and its budget are in docs/EXPERIMENTS-2026-10-14.md.
Conditions for every Qwen number: one MI300X VF, batch 1, context 1024, 32 tokens.

## Two questions, kept apart

1. **Does the common compiler mode WORK on Qwen?** This is the reuse criterion: one switch (`ETX_COHERENCE=l1`), two
   models, one rule (docs/INVENTORY-SYSTEM-2026-10-13.md 4.1). It needs `kept_l1` to build, to pass the same-build
   import certificate (positions 0 and 1040), and to reach 32/32 + CHECK-DSL. A NO-GO on speed does not undo this.
2. **Is it FASTER on Qwen?** That is G6's rule: KEEP if every pair is faster than `kept`, median >= 30 us and
   > 2 x spread, device time, 5 interleaved rounds.
   - Expect much less than DeepSeek's +330 us [E]. Qwen's DSL tiles are already coherent: `ETX_VL_COH`, -0.317 ms,
     in the kept build since 10-01 (docs/OPTIMISATIONS.md:16).
   - In Qwen the switch only moves 82 payload loads from system to agent scope. It removes 0 of the 98 acquires,
     which belong to imported vLLM kernels (ROCm 7.2.4 ISA of both builds).
   - Nothing here predicts Qwen >= 300 tok/s.

The session also answers a third, independent question: **Q2, row ownership between MoE workgroups** (section 4).

## 1. Already measured (Qwen's own sessions; none re-run)

| what | value | source |
|---|---|---|
| kept baseline, 2026-10-12 | 3.943 ms device / 4.078 ms wall = 254 tok/s [M]; raw logs lost with that box | docs/BASELINE-STATE-2026-10-12.md:7-11,19 |
| banked after it | router narrowing 0.0626 + norm hoist 0.0556 ms [M] | BASELINE-STATE:29-33 |
| coherent DSL tiles | -0.317 ms [M], in the kept build | docs/OPTIMISATIONS.md:16 |
| closed | MoE rows/task 24/16/16: -0.140/-0.356/-0.413 ms; prep-early neutral or negative; ARGPF/BAR2/ATTN_WIDE | BASELINE-STATE:37-41; OPTIMISATIONS.md:26; GAINS-LEDGER.md:62 |
| DeepSeek half of question 1/2 | `l1` vs default +330 us, spread 13, every pair faster: KEEP under G6's rule (re-derived, not re-run) | examples/dsv2lite/results/2026-10-13-hour/pairs_l1.json |

## 2. Ready locally (commits on wf/qwen-ready, from main 2c1d9dd)

- G6 arm registry, session and device-build check (67f8688):
  - `kept` vs `kept_l1` differ ONLY in `ETX_COHERENCE=l1` (tested);
  - ROCm 7.2.4: both rc 0 with an identical plan (60362 tasks, 436 events) and identical resources (128 VGPR,
    occupancy 4, the kept recipe's 7-VGPR / 32 B spill);
  - ISA: `sc0 sc1` loads 82 -> 0, `sc1` loads 1305 -> 1387, `buffer_inv` 98 -> 98, `buffer_wbl2` 50 -> 49.
  - Caveat: the local import source was vLLM v0.27.0-5; its `csrc/` is byte-identical to v0.27.1 (git diff empty).
- Preflight that refuses early (4139861):
  - the vLLM checkout must BE the v0.27.1 commit (before, any version passed);
  - the G6 session runs interpreter / vllm_source / packed_model / rocm_headers / gpu before anything, and both
    builds wait for it.
- Q2 microbenchmark `hw_bench moerows` (this commit):
  - ONE kernel for both ownership modes (91 VGPR, 0 scratch). Two per-mode templates had compiled to 47 vs 91 VGPR,
    which would have confounded the comparison.
  - Device and host passes compile with ROCm 7.2.4. The CPU oracle (`bench/hw/moerows_oracle.py`) and its 3 tests
    pass: every expert half covered exactly once per pass in both modes, and every measured read cold
    (>= 336 MB since that expert's last use, > 288 MB of MALL + all L2s).
- Session order (dry run): preflight, builds and hw_bench build (CPU) -> `moerows` -> `certify_kept` ->
  `certify_kept_l1` -> `pairs_kept_l1`. Targeted tests: 69 passed.

## 3. Missing inputs: exact list from a REAL preflight run on this Mac (2026-10-14)

`python -m examples.vllm_llm.g6_session preflight` gave: interpreter PASS, rocm_headers PASS, and three FAILs.

| item | state here | supply on the box (each printed by the preflight itself) |
|---|---|---|
| vLLM source at v0.27.1 | checkout at f46a9dfe (v0.27.0-5), not 6e448d0e | `git clone --depth 1 --branch v0.27.1 https://github.com/vllm-project/vllm.git ~/vllm` |
| packed model `~/llm/qwen3-30b-a3b/` (`vl.bin`, `vl.manifest`, `hf/config.json`, `golden_c1024.txt`) | absent: no packed Qwen model on any local disk | see below |
| GPU | none (a Mac) | the box |

Packed model supply:
1. Download the HF checkpoint `Qwen/Qwen3-30B-A3B` into `~/llm/qwen3-30b-a3b/hf`.
2. `python examples/vllm_llm/prep.py --src ~/llm/qwen3-30b-a3b/hf --out ~/llm/qwen3-30b-a3b --maxlen 1072`.
3. `cp examples/llm/results/golden_c1024_qwen3-30b-a3b.txt ~/llm/qwen3-30b-a3b/golden_c1024.txt`. The golden is
   pinned (sha256 bab2f1fb...), never regenerated.

Also needed on the box: `ETX_QWEN_G6_CONFIG` pointing at `{"model": "~/llm/qwen3-30b-a3b", "hf": ".../hf",
"golden_sha256": "bab2f1fb..."}` with absolute paths.

**Open gap, not fixed here:** the HF download takes no revision pin. The revision is only recorded afterwards
(`HF_REV`, docs/GAINS-LEDGER.md:31). The pinned golden + 32/32 would catch a changed checkpoint, not prevent one.

## 4. Q2 decision microbenchmark: rule written before any GPU run

- **Shape:** Qwen's moe_up phase exactly as `moegather`'s gather: 192 workgroups = 8 experts x 24 tasks, 32 rows of
  4 KB of gate + the same rows of up, experts rotating over the 805 MB block.
  - contiguous = rows j*32 .. j*32+31 (today's `dsl_gemv_rows`);
  - interleaved = rows j, j+24, ...
  - Same bytes (48 MB per pass), same experts, same instructions.
- **Correctness:** a position-dependent fill. Each pass's sum checksum must equal the host's sum over exactly the rows
  due, for both modes. Any mismatch -> **INCOMPLETE**, no decision.
- **Rule:** 5 launch pairs, order reversed on odd pairs, median span over passes 2..11.
  - **PASS** if interleaved is faster in every pair, the median gain is >= 3% of the contiguous span, and > 2 x spread.
  - **NO-GO** otherwise.
  - Why 3%: Qwen's MoE weight stream is 48 x (50.3 + 25.2) MB = 3.6 GB/token, about 1.0 ms at 3.55 TB/s [E].
    3% of that is about 30 us, the Qwen KEEP bar.
- **What follows:**
  - PASS -> build interleaved ownership as a switch in the shared GEMV op (`etx/ops/gemv.py`, every DSL model), then
    measure it on the kept build under the same KEEP rule.
  - NO-GO -> close Q2 for MoE.
  - INCOMPLETE -> fix the bench; do not build the op.
- **Cross-check:** `hw_bench moegather` runs in the same job; its gather span should match `moerows` contiguous.
- **Why the CPU cannot decide this:** throughput depends on HBM channel/bank interleaving, Infinity Cache admission
  and replacement, TLB reach and the hardware's request coalescing under each address pattern. None of these is
  modelled here. The model's own 30-60% "rows" gain was a MALL artifact, and its 0.50 ms is an old-build bound
  (docs/MODEL-vllm-kernels.md:579-582,646-649).
- **Caveat:** the coldness margin (336 vs 288 MB) assumes LRU-like replacement; the MALL policy is not documented [U].

## 5. Minimal GPU plan and cost

Box steps outside the runner:
- stage the repo;
- clone vLLM v0.27.1;
- download and pack the model. DeepSeek's 30 GB downloaded in 29 s on 10-13 [M]; Qwen's ~61 GB [E] about 1 min;
  packing time [U];
- copy the golden; write the config.

Then `tools/session.sh start <ip> examples/sessions/qwen_g6.py`. The runner's hard stop is **3600 s**.

| job | lane | cost |
|---|---|---|
| preflight, 2 builds, hw_bench build | CPU | build time on the box [U]; the hour session allows 1500 s each |
| moerows + moegather | GPU | 10 launches x 12 passes x ~15 us + fill: seconds [E]; job cap 600 s |
| certify kept, certify kept_l1 | GPU | 3 host invocations each, <= 240 s each (deadline) [E] |
| pairs_kept_l1 | GPU | 10 invocations, <= 240 s each [E] |

Measured history:
- Qwen box 2026-10-12: 3 h 33 min, about $7.1 at $1.99/h [M] (BASELINE-STATE:23-29).
- DeepSeek paid hour 2026-10-13: runner 10 min 53 s, billed about 50 min [M]
  (examples/dsv2lite/results/2026-10-13-hour/README.md).

**Cap proposed for approval:** one box for <= 1.5 h including setup, so <= $3.0 at $1.99/h [E]. Delete the box at the
end, fetch into a new directory, and seal the record.
