# Preflight: `examples/sessions/qwen300_hour.py` (74 min, one MI300X, $1.99/h)

**Verdict: GO-WITH-FIXES. Do not provision the session as committed: it is NO-GO unchanged.** As committed, staging refuses to start because the CPU gates are stale. If the gates are rerun, the session still cannot produce its deciding rows, because D cannot finish five rounds in 600 s.
**Must fix before provisioning:** fixes 1-5 below, all small (about 30 lines of code plus a rerun of the CPU gates). Fix 6 is an owner decision about the plan, not a code change.
**Top three findings:** (1) the CPU-gate evidence belongs to `bff0279`, and 17 hashed sources have changed since, including 168 inserted lines of device and runtime code. (2) D needs about 640-930 s [E] against a fixed 600 s, so D.fusion, D.system and D.target come out NOT-COMPLETE whenever NO_TOUCH qualifies. (3) S2 is fully decided by A's data (R2 is always 0), yet B's C1/C5 pairs run before it; a schedule at the protocol floor passes S1 and then must fail S2.

Reviewed tree: `wf/preflight` at `05d15a2`. The plan is `docs/PLAN-300TOKS-2026-10-11.md`, which is identical to the pinned revision `9f91f3a` (`git diff 9f91f3a HEAD -- docs/PLAN-300TOKS-2026-10-11.md` is empty). CPU only: no GPU, no provisioning, no edits to the session, plan, AGENTS.md, CLAUDE.md or any evidence document. All `qwen300_hour.py` line numbers refer to `examples/sessions/qwen300_hour.py`. [M] marks a measurement, [E] an estimate, [U] an unknown.

## Checks run for this review

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q tests/test_qwen300_hour.py` | 80 passed [M] |
| `python -m examples.sessions.qwen300_hour --dry-run` | budgets A=900, B=1320, C=1620, D=600; cap 4440 s [M] |
| `source_hashes()` at HEAD against `docs/evidence/qwen300-hour-cpu-2026-10-11.json` | 17 files differ [M] |
| `--analyze` on synthetic stamp records built from the real plan `build/qwen300-rocm-on/plan.json` (60,362 tasks, 1,403 event instances) | 9.7 s, about 1 GB RSS, 287 MB `edges.json` per run (Mac) [M] |
| The same analysis with every critical edge exactly at the 1.00 us floor | G = 0.15255 ms, S1 PASS [M on synthetic input] |
| Invocation times from prior session logs (`results_qwen300_2026-10-11` file mtimes; `results_vlbest2_2026-10-10`) | warm ETX host 17-28 s; vlbest2 `etx_r` 21-23 s; vLLM S3 73-97 s [M, other sessions] |

The synthetic generators are `/tmp/preflight_synth.py` (1.7 us per edge) and `/tmp/pf_floor.py` (per edge: pre 0.05, arrive 0.30, observation 0.50, setup 0.15 us, which sums to the 1.00 us floor). They are not committed.

## Q1. Thresholds

**These match the plan exactly:**
- S1: G ≥ 0.088 ms (`qwen300_hour.py:394`; plan 302).
- S2: G + R2 + H ≥ 0.324 ms (`qwen300_hour.py:421-422`; plan 308).
- KEEP: exactly 3 pairs, every gain > 0, median ≥ 0.030 ms and median > 2 × spread (`qwen300_hour.py:612-617`; plan 314).
- G atom counting: ≥ 0.625 us/layer (`qwen300_hour.py:385`; plan 301).
- Idle prices: observation 0.951 us (the conservative end of 0.642-0.951) and zero for arrive and setup (`qwen300_hour.py:367-369`; plan 301).
- C fusion: ≥ 1.042 us/layer, VGPR ≤ 128 and LDS ≤ 29,752 B (`qwen300_hour.py:710,720,852`).
- D target: ≤ 3.3333 ms (`qwen300_hour.py:972`; plan 318).

**These are in the code but not in plan section 7.1** (the session report registers them as "frozen task registration", `docs/qwen300-hour-2026-10-11.md:42-47`):
- A1 busy 0.100 / dead 0.050 and free 0.050 (`qwen300_hour.py:576-577`).
- A2 headroom 0.142 ms (`:581`).
- A3 0.086 ms (`:591`).
- A4 reset **> 0.030** (`:594`); plan 306 says **≥ 0.030** for X8.
- The isolation rule < 2.0 us (`:794`), whose arm is disabled.

None of these feeds S1 or S2, so none can change a stop decision.

**These are in the plan but missing from the code.** The report admits all of them (`docs/qwen300-hour-2026-10-11.md:335-351`):
1. OFF/ON validity and the 0.5 us-per-body perturbation limit (plan 299). A uses one ON snapshot of one step (`qwen300_hour.py:561-563`) and reports OFF walls only as a control (`:569-573`). Nothing invalidates G when stamps perturb the schedule.
2. The row-10 80% close rule (plan 300).
3. The P1 fetch and isolated P7 atoms, which are listed as `unobserved` (`:388`).
4. X5 and its R2 formula (plan 307), so R2 is always 0 (see Q2).
5. The touch rule, which needs ≥ 0.5 us/edge median shrink of the busy rows (plan 312). C.NO_TOUCH is decided by wall pairs alone (`:833,844`). `C_NO_TOUCH_edges.json` is produced (`:765-768`) but no rule reads it.
6. The X6 ≥ 0.625 us/layer exposure rule (plan 313).
7. The S3 0.847 ms target-attempt gate before D (plan 316). D runs whenever S2 passes (`:992-993`).

B runs C1/C5 (`qwen300_hour_arms.json` B), which the plan places in C (plan 310). The plan's X8/X5 are absent. A3 is the code's stand-in for X8.

**Is any threshold wrong?** No number is mistyped. The problem is S1, which is registered as the plan wrote it but cannot work as a stop (fix 6):
- Arrive and setup carry an idle price of zero (plan 301: "Zero is generous and so errs toward continuing").
- On the floor schedule, 339 counted critical edges give arrive 339 × 0.30 = 101.7 us = 0.1017 ms (2.12 us/layer, counted) and setup 339 × 0.15 = 0.0508 ms (1.06 us/layer, counted). Observation is 0.50 < 0.951, so it contributes 0.
- So **G = 0.15255 ms ≥ 0.088, and S1 passes when every edge is already at the protocol floor** [M on synthetic input].
- S1 stops only if arrive plus setup average below 0.088 ms / 339 = 0.26 us per edge [E]. A real schedule at or above the floor cannot meet that.
- On the 1.7 us/edge synthetic, G = 0.305 ms.

## Q2. Stop wiring

1. **R2 is structurally 0.** `attention_price` always returns `attainable_cut_ms=None` (`qwen300_hour.py:443`), and `second_stop` reads it as `or 0.` (`:407`). With H capped at 0.086 (`:416`), **S2 passes only if G ≥ 0.324 − 0.086 = 0.238 ms**.
   - G comes from A (`:583-584`) and H from A3 (`:586-595`). Tranche B (`:664-683`) writes neither, nor A2.
   - **S2 is therefore fully decided when A ends**, and B's 1320 s window runs before a decision it cannot affect.
   - On the floor schedule, G + H ≤ 0.15255 + 0.086 = 0.2386 < 0.324, so S2 is certain to STOP after B has been paid for.
   - When S2 stops, B's survivors feed nothing: they are used only by D (`:993`). The plan says intermediate candidates are not funded after a stop (plan 320). This is fix 3.
2. **Missing G closes the route.** If the stamps build fails certification, the stamp run fails, or the analysis fails, `gap_atoms` stays `None` (`:554,583-584`). `first_stop` then records `STOP-SPEND: close target route` (`:393-398`). The same happens when the kept certification fails (`:395`).
   - The plan treats a missing measurement as **NOT-COMPLETE** (plan 299). Ending the session is correct; recording a route closure is not.
   - The report states "Missing stamps stop" (`docs/qwen300-hour-2026-10-11.md:48`) but not that this is labelled as a closure. This is fix 4.
3. **A2 failure takes G down with it.** `attention_price` raises if any of the 48 layers lacks a critical `attn` or `reduce` phase (`:433-434`). It runs inside `stamped()` after A1 (`:579`), so the exception discards `result`, and with it `gap_atoms`, and S1 then records a STOP.
   - The phases are critical-path only (`etx/tools/inbody.py:132-163`).
   - Attention and reduce are serial in every layer, so a miss is unlikely. The synthetic run had all 48. But an A2 coverage failure has nothing to do with G. This is fix 5.
4. **H units are consistent despite the names.** The `named_host_ms` values are milliseconds, even under `_us` keys: `avg/1000` (`:462-463`). The 0.049 and the reset subtraction (`:413`) are in ms. No error.
5. **Stop order is otherwise correct.** A STOP at S1 marks B, C and D NOT-RUN (`:980-985`). A STOP at S2 marks C and D NOT-RUN (`:987-991`). Every registered row is emitted, including missing ones, as NOT-COMPLETE (`:219-221`).

## Q3. What tranche A really measures

- **Fresh correctness for three builds** (kept, stamps, stages). Each runs `--check 48` at positions 0 and 1040, plus one 32/32 run with CHECK-DSL (`qwen300_hour.py:490-504`). That is 9 host calls.
- **G and A1/A2 from one decode step, ctx 1024, of one ON-stamp run** (`:561-566`). The critical path is computed from the ON timestamps themselves (`etx/tools/inbody.py:127-135`).
  - No repetition and no OFF/ON validity check.
  - The kept OFF wall and the null-stamp wall are recorded for comparison only (`:569-573`).
  - G is therefore the stamped ON schedule's arrive and setup excess on 339 non-attention critical edges (`:370-381`). It is not an OFF-schedule quantity, and its perturbation is unbounded.
- **A3/A4 from one 32-step run of the profile build** (`:586-595`, `:446-469`).
- **Not what the plan asked for:** the plan's A is the committed `inbody.py` job unchanged, with five repetitions per arm, OFF and ON, and sparse MoE replay pairs (plan 297). The code measures a one-snapshot substitute.

## Q4. Ways to burn minutes

Invocation counts come from the code; durations are estimates from prior logs.

| Tranche | Calls | Window | Estimate, warm | Outcome |
|---|---|---|---|---|
| A | 11 host + analysis (`:555-595`) | 885 s (`:143`) | 11 × 22 + 10 ≈ 250 s, plus one cold 61 GB load [E] | Fits unless the cold load exceeds about 10 min |
| B | 2 arms × (3 cert + 6 timing) = 18 host (`:645-652`) | 1320 s (`:666`) | ≈ 400 s ≈ 6.6 min ≈ $0.22 [E] | Fits; spent before S2 (Q2.1) |
| C | NO_TOUCH: 3 cert + 1 stamp + analysis (diagnostic build) + 3 cert + 6 timing = 13 host (`:756-768,831-833`) | 1620 s (`:818`) | ≈ 300 s [E] | Fits; about 1300 s unused and **not passed on** |
| D | graph gate 1 + 5 × (3 host + 1 S3) = 16 host + 5 S3 (`:933-952`); kept certification is cached (`:492`) | **600 s** (`:926`) | 16 × 17-28 + 5 × 73-97 = **637-933 s**, central 16 × 22 + 5 × 85 = **777 s** [E] | **Does not fit** |

1. **D cannot complete (fix 2).**
   - A round costs about 3 × 22 + 85 = 151 s [E]. After the graph gate (about 22 s), the 600 s deadline falls inside round 4.
   - Both verdicts require all five rounds (`:956-957`). D.fusion, D.system and D.target are therefore never emitted, and `finish()` writes NOT-COMPLETE (`:221`).
   - With the harness's own expected values (120 s per host call, 180 s per S3), D needs 16 × 120 + 5 × 180 = 2820 s.
   - D gets more time only when C is skipped: +1620 s (`:813-817`). NO_TOUCH is enabled in `qwen300_hour_arms.json`, so if it qualifies, it is exactly what starves D.
   - The report admits that "20 cold D invocations may exceed ten minutes" (`docs/qwen300-hour-2026-10-11.md:328`). Under warm timings it still does not fit.
   - The whole session needs about 30-35 min [E] of its 74, so the cap is not the constraint. The per-tranche deadline is.
2. **B before S2.** On a stop path, B burns about 7 min [E] (up to 22 min) (Q2.1; fix 3).
3. **The cold first load lands in A.** The first host call reads about 61 GB from disk. CLAUDE.md puts 16 GB at 1-2 min, so this could take 4-8 min [E] of A's 885 s. The silence probe only probes; it does not kill (`:202-203`). The step watchdog arms only after `step` lines appear (`:194`). This is not fatal, but pre-warming the page cache before the clock removes the risk (fix 7).
4. **Staging cost is undeclared.** `--prepare` runs 9 builds, each with a 900 s timeout (`:94-108`). If that runs on the VM, it is billed outside the 74 minutes (plan 320: staging "must carry its own declared budget").
5. **The vLLM burst watchdog is a low risk.** It kills S3 after 10 s of log silence between `VLLM-GENERATE-BEGIN` and `END` (`:197-201`), and the warmup generate is inside that bracket (`examples/llm/vllm_bench.py:95-102`). In vlbest2, graph capture and torch.compile happened during engine init, outside the bracket. The first generate, including Triton JIT, finished within about 1 s (`results_vlbest2_2026-10-10/a_r1_S3.log:66-79`). The risk is small, as long as the container's cache directory persists (`-v $HOME:$HOME`, `:879`, with HOME=/root on that VM).
6. **Analysis output takes about 575 MB** of disk: two runs of about 287 MB `edges.json` (A and C). Each run took 9.7 s on a Mac against a 45 s timeout (`:566,767`). Fine.

## Q5. Correctness gates

These are sound, with no change needed:
- Each build gets fresh exhaustive import checks at positions 0 and 1040, validated against its exact instance lists (`qwen300_hour.py:496-499`; `examples/vllm_llm/import_certificate.py:131-177`).
- Every host output is checked for identity (`qwen300_hour.py:477`).
- Every timing sample must pass exactly one 32/32 run (`:481-487,600`) with all 31 post-first-token steps (`:605-606`).
- S3 needs its lever proof, a warmup gate and a 32/32 sample (`:893-900`).
- Partial populations raise NOT-COMPLETE (`:613`) and are never aggregated (`:956-975`).

One minor gap: the build identity hashes `vl.manifest` and the golden, not the weight bytes (`import_certificate.py:40`). Corrupted weights would still fail the 32/32 gate.

## Q6. NO_TOUCH identity and seal

**What is checked:**
- `verify_seal` (`qwen300_hour.py:121-133`) checks the seal hash, `source_hashes`, the commit, the condition (bf16, batch 1, ctx 1024, 32 tokens, KEY32) and the build digest.
- `verify_wait_variant` (`:726-753`) checks:
  - the identical full plan;
  - identical imports;
  - the overrides, ignoring the five path variables;
  - kept TOUCH on and isolation off;
  - the exact hipcc flags (`BASE_FLAGS`, plus `-DETX_NO_TOUCH`);
  - the stamp flag for the diagnostic build.
- `build.sh` is inside `source_hashes` (`:65-67`), so its internal flag logic is pinned by source.

That is adequate.

**Not verifiable here:**
- The report's `.text` evidence: TOUCH default against explicit ON is byte-identical, and NO_TOUCH and isolation differ from control (`docs/qwen300-hour-2026-10-11.md:476-491`). That evidence is said to remain under `results_toucharm_cpu_20261011/`.
- The earlier gate logs under `results_qwen300_hour_cpu_20261011/` (`:354`).

Neither directory exists in this worktree, in `~/code/etx`, or in any `~/code/.wt/*`. This is fix 8.

## Q7. Anything else

1. **The CPU gates are stale (fix 1, blocker).**
   - `docs/evidence/qwen300-hour-cpu-2026-10-11.json` records `prepared_from: bff0279` and `source_commit: null`.
   - Its `source_hashes` differ from HEAD in 17 files: `etx/codegen/kernel.py`, `etx/codegen/plan_header.py`, `etx/runtime/include/etx/{abi,host,inbody}.h`, `etx/tools/{inbody,spill_chain,verify_gaps,watchdog}.py`, `examples/sessions/qwen300_hour{.py,_arms.json}`, `examples/vllm_llm/import_certificate.py`, and five `examples/vllm_llm/*_cpu.py` files.
   - `git diff --stat bff0279 HEAD` over codegen, runtime and `host.hip` shows 168 insertions, from `57f186a` (isolation emission flag), `b4d3037` (arrival ledger and watchdog dump), `0cc6c1b`, `658beac` and `afe46bd`.
   - Both `prepare` (`:82`) and `preflight` (`:512`) refuse a mismatch, so this fails closed rather than producing a wrong result.
   - However, the gate file is a set of self-declared booleans (`:510-511`). Copying fresh hashes into it without rerunning the matrix and the native ROCm build checks would pass the harness.
   - The report says the matrix and native builds were not rerun after these commits (`docs/qwen300-hour-2026-10-11.md:421,532-533`). Its line 421 claim that "no kernel/host build source changed" is false relative to the gate commit.
2. **Report inconsistencies (fix 9).**
   - C's skip transfer is "exactly 1200 seconds" (`docs/qwen300-hour-2026-10-11.md:214`) and D is "30 when C is skipped" (`:216,330`). The code transfers 1620 s (`qwen300_hour.py:817`), which gives D 2220 s = 37 min.
   - The dry-run blocks at `:405,451` print `C=1200` and a 4020 s cap. They are superseded by `:517-518`, but they are still in the document.
3. **Cost.** At the 4440 s cap the session costs 74/60 × 1.99 = $2.45 [E]. The likely spend is about 30-35 min ≈ $1.00-1.16 [E], because tranches end early, but with D NOT-COMPLETE.

## What an unchanged run produces

- **As committed:** `--prepare` refuses with "CPU gates belong to different source" (`qwen300_hour.py:82`). If a prepared configuration is reused, `preflight` refuses with "CPU gates belong to another source" (`:512`). Nothing is measured. The only cost is VM time already spent provisioning and staging.
- **With the gates honestly rerun on HEAD and everything else unchanged** [E]:
  - A completes in about 5-15 min, and S1 PASSes; by Q1 it almost cannot fail on real data.
  - If G + H < 0.324, which is certain when the gaps sit near the floor: B spends about 7 min on C1/C5, then S2 STOPs. Total about 20 min, about $0.66, and the B spend bought nothing.
  - If G + H ≥ 0.324: B and C (NO_TOUCH) finish in about 12 min combined, then D runs out of its 600 s in round 4. D.fusion, D.system and D.target are NOT-COMPLETE. Total about 30-35 min, about $1.1, with no fusion verdict, no system verdict and no target verdict.
- **Edge case:** if the stamps build or A2 fails, the session ends after A with a false "close target route" record.

## Fixes

| # | Fix | Where | Size | Required |
|---|---|---|---|---|
| 1 | Rerun all six CPU gates on the final HEAD (pytest, gfx942 with `ETX_DEVLIBS`, dry run, shell, `local_matrix.sh` 16/16, `rocm_build_check.sh`), then record `source_hashes()` and `source_commit` from that run with its logs | evidence JSON plus logs | CPU about 1-2 h, $0 | **Yes (blocker)** |
| 2 | Give D the rest of the cap: `s.deadline = s.end` at `:926`, or carry the unused A/B/C time forward. Update the 4440 assertion and the tests that pin D=600 | `qwen300_hour.py:926,1010`; tests | about 3 lines plus tests | **Yes** |
| 3 | Evaluate S2 right after A: G, H and R2 = 0 are all known then. Skip B when it fails, or move B after S2 with B survivors entering D only | `:980-993` | about 10 lines plus tests | **Yes** |
| 4 | Label missing G, or a failed kept certification, as `NOT-COMPLETE` rather than `STOP-SPEND: close target route`, and still end the session | `:393-398` | about 4 lines | **Yes** |
| 5 | Run `attention_price` in its own `s.attempt`, after `stamped()` has returned `result`, so an A2 coverage failure cannot drop G | `:579-584` | about 6 lines | **Yes** |
| 6 | Owner decision: as registered, S1 passes at the 1.00 us floor (G = 0.15255). Either declare S1 a non-stop, or re-register arrive and setup against the floor (the plan says "zero ... errs toward continuing") | plan section 7.1 (owner) | doc-only | Decision before spend |
| 7 | Pre-warm the page cache for the model weights before the clock (e.g. `cat` the weight files into `/dev/null`) and declare the staging and prepare time as its own budget | session README/runbook | about 2 lines | Recommended |
| 8 | Restore or regenerate `results_toucharm_cpu_20261011/` and `results_qwen300_hour_cpu_20261011/` so that the `.text` identity and the gate logs can be audited | evidence dirs | CPU minutes | Recommended |
| 9 | Correct the report: 1200 s → 1620 s and D 30 → 37 min (`:214,216,330`). Mark the stale dry-run blocks (`:405,451`). Retract "no kernel/host build source changed" (`:421`) | `docs/qwen300-hour-2026-10-11.md` | doc-only | Recommended |
| 10 | Either implement the missing plan rules (OFF/ON validity and 0.5 us perturbation, the touch ≥ 0.5 us/edge rule, S3 0.847), or emit them as explicit NOT-RUN rows so that the report cannot read as plan-complete | `qwen300_hour.py` REGISTERED/rows | medium | Optional |

With fixes 1-5, D fits comfortably: about 780 s needed [E] against about 2400 s left [E]. S2 then costs only A when it stops. The remaining risks are the cold load in A and S1's inability to stop, which is a plan property and not a code defect.
