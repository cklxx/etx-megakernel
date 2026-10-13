# Session machinery: root causes and the structural fix (2026-10-13)

Why does every fix produce a new defect in the fix; is it the architecture? **Yes**: four root causes, each an
invariant no single component holds. Evidence: `file:line` at 2ef1e58, Astra's `/tmp/etx-dsv2/answer-0{7,8,9}-*.md`;
round 9 (`answer-09-recheck.md:9-10,41`, not yet in `docs/DSV2-D1-D5.md`) found R12 open, R15 new, **both in round-8 fixes**.

## 0. Why the fixes breed defects

Each invariant has several partial implementations; a fix adds one more guard where the reproduction pointed, so it
covers the reproduction, not the invariant, and the next review finds the sibling path. R08 cleaned regular files in
the reused `~/etx`; symlinks stayed (R08'). R12 refused a non-empty `fetch` destination (`tools/session.sh:92`);
`postmortem` sets `ETX_FETCH_REUSE=1` (`:57`) (R12'). R01 kept the GPU binary in the group (`timeout --foreground`);
the build's compiler children escaped (R14). R14 kills a `/proc` snapshot (`etx/tools/gpu_jobs.py:94-113,249,317`);
children forked after it survive (R15). Test for each cause below: one structural change makes its class impossible.

## 1. Root causes, ranked by the defects they explain

**RC1. Identity is a long-lived mutable directory reconciled by allow and deny lists** (R03-R06, R08, R08', R09, R10,
R12, R12', R13: 11 of 17 findings). *Source* `~/etx`: reused, rsynced, extras deleted by `find`/`comm`
(`examples/dsv2lite/stage_inplace.sh:30-37`), checked by a deny list (`examples/dsv2lite/inplace_session.py:217-228`)
plus `tree_hash` (`:207-214`); a second staging path has no identity (`tools/session.sh:20`). Fleet is already done
right: `git archive` into a fresh directory, content-hash checked (`examples/dsv2lite/pinned.py:51-56`). *Inputs*:
`SHA256SUMS` regenerated from whatever is on disk (`examples/sessions/ds_inplace.py:86-87`), so pinned hashes were
bolted on (`inplace_session.py:83-86,404-406`). *Session* `~/sess`: five writers, i.e. launcher (`session.sh:80`), runner
(`gpu_jobs.py:288-296,377-392`), invocations, builds, and a hash job that must run last and skip live files by name
(`examples/sessions/ds_hour.py:55-58`). *Fetch destination*: reused; two fetch paths (`session.sh:87-114`,
`examples/dsv2lite/reproduce.sh:93-107`), three verifiers (`etx/tools/verify_sums.py`, `examples/dsv2lite/verify_hour.py`,
`rederive.py`). "The files of this session" is written **seven times**: `ds_hour.py:55-56`, `session.sh:97,100`,
`reproduce.sh:97,101`, `verify_hour.py:23`, `verify_sums.py:55`. Provenance is policed afterwards: hypothesis 3 confirmed.

**RC2. Readiness is hand-written edges between job names, not derived from the artifacts a job reads.** Data
dependency, GPU priority and the end barrier share two fields (`gpu_jobs.py:63-64`). `after_any` counts a skip as done
(`:197-199`); list-order priority does not block a later ready job (`:26-27,350-363`), so priority was built from
`after_any` chains, by mutation: ten `JOBS[-1][...]` writes (`ds_inplace.py:101-127`; `after=` at `:113` is dead,
`:115` overwrites it); `ds_hour.py:27` clears another module's `JOBS`, `:50` chains, `:58` lists every name as the
barrier, `:34-43` names the builds apart from `arm_order` (`inplace_session.py:231-254`). Defects: the d1d5 loss
(2bb3597: pairs read `weights.bin.manifest` before `prep_pack` wrote it), R06 (barrier over a subset), R07 (priority one
link deep). Hypothesis 5 confirmed but secondary: immutable name edges would have had all three bugs.

**RC3. An arm and its rule have no single definition.** One arm lives in up to nine places: `ARMS`/`ARM_FLAGS`/
`ARM_ENV` (`inplace_session.py:33-78`), `RULES` + string dispatch (`:152-177`), `arm_order`, `build_report` (`:264-277`),
prints (`:465-479`), argparse lists (`:542,546-547,551`), `TRACEABLE` (`:81`), `ds_hour.py:34-43`. The build check
spells arms `lever+lever` and forwards ambient `ETX_DSV2_*` (`examples/dsv2lite/rocm_build_check.sh:21-30`); `Session.run`
strips every `ETX_` variable (`inplace_session.py:313`): local proof and box build of `alld4r0l1` cannot be shown to be
one build. `l1` is `ETX_DSV2_COH=l1` + a lever in the session (`:59-63`), `ETX_COHERENCE=l1` in `build.sh:20` and
`rocm_build_check.sh:51`. Rules are float code in four functions (`:102-147`); R02 was "fixed" by rounding at three
sites (`:105,112-114,126`). Defects: 2bb3597's `TypeError` (arm env given twice), R02, R10 (`rederive.py` has its own
arm list), R13, today's build-check naming. Hypothesis 4 confirmed.

**RC4. Nested supervisors; a process group is not a containment boundary** (R01, R14, R15). `gpu_jobs` contains a job
by process group (`gpu_jobs.py:227,251,258,319`); inside, `Session.run` is a second supervisor with its own deadline,
progress and silence probe (`inplace_session.py:294-347`), duplicating `gpu_jobs.py:66,70,265-281`. `timeout` leaves the
group (R01); `--foreground` keeps it but no longer kills grandchildren (R14); a snapshot races with `fork` (R15). Seven
more killpg supervisors: `etx/tools/cpu_check.py:146,156`, `examples/vllm_llm/{bench.py:42,59, bf16_300/launch.py:96,110}`,
`examples/sessions/{qwen300.py:139,159, moe_rpt.py:118,140, inbody.py:82,97, qwen300_hour.py:234,266}`. H6 confirmed.

**H1 (location): refuted as a cause**, a consequence of RC1/RC3: moving the file to `etx/` moves every defect with it
(the harm: tools bound to one session, `session.sh:47`, `verify_hour.py:57-58`). **H2 (six runners): confirmed, = RC1.**

## 2. The structural design: four components in `etx/session/`, one invariant each

**A. `proc.py`, the only supervisor (RC4).** The runner, and each job that runs invocations, is a Linux child
subreaper (`prctl(PR_SET_CHILD_SUBREAPER)`). Stop = SIGKILL every descendant, reap, repeat until `waitpid` returns
`ECHILD`; a child forked meanwhile re-parents to the subreaper and dies next pass, and `copy_process` refuses to fork
with a fatal signal pending, so the loop ends (cgroup v2 `cgroup.kill` where the box allows it [U]). Deadline, silence,
progress live here once. *Invariant: when a job leaves `running`, no process it started exists.* *Deletes*
`gpu_jobs.py:94-120,248-263,313-323`, `inplace_session.py:294-347`, the seven copies.

**B. `record.py`, the session record (RC1).** A write-once directory named by a session id (commit, UTC, nonce), owned
by the runner. Each artifact enters `MANIFEST` with its sha256 once its writer is reaped (A makes that true); the runner
owns its stdout and seals last: `SEAL` = hash(MANIFEST, source archive hash, input hashes, job plan). Source and inputs
are content-addressed as fleet already is (`git archive` into an empty `~/src/<archive sha>`; inputs under their pinned
hashes). Fetch writes `<results>/<session id>/`, which must not exist; verify recomputes `MANIFEST` and `SEAL`; unsealed
= NOT COMPLETE by type. *Invariant: a result is a sealed record whose every byte came from this session's reaped
processes.* *Deletes* `stage_inplace.sh`, `inplace_session.py:207-228,357-369`, `SOURCE_FILES`, `hash_session`,
`prep_sums`, `verify_sums.py`, `session.sh:53-64,78-82,87-114`, `reproduce.sh:93-124`, `gpu_jobs.py:286-296`.

**C. The artifact-derived graph (RC2).** A job declares `reads` and `writes` (record keys) and is ready when every read
is sealed; the GPU lane is a strict queue (a skip advances it); the end is the runner's seal, not a job. *Invariant: no
job starts before the bytes it reads are final; no skip reorders the GPU lane.* *Deletes* `after`/`after_any`/`gate`
(`gpu_jobs.py:63-65,189-208,419-426`) and every `JOBS[-1]` mutation (`ds_inplace.py:99-127`, `ds_hour.py:26-58`).

**D. Arm registry, rules as data (RC3).** `Arm(name, levers, flags, env, order, references, rule)`, one table per model
family; `Rule(basis, reference, min_median_us, spread_factor, strict)` evaluated by one function over integer
nanoseconds parsed from the logs' decimal text. Build, `rocm_build_check.sh --arm <name>`, jobs, verdicts and
re-derivation are generated from it. *Invariant: an arm name is one build and one rule everywhere.* *Deletes*
`inplace_session.py:32-81,102-177,231-283,461-479,541-553`, `rederive.py`, `verify_hour.py`.

**Net [E, from the ranges above]:** removed ≈ 1,010 lines (4 files outright, 354 lines; ≈ 520 inside files; ≈ 140 in
the seven supervisors), added ≈ 300 (A 50, B 120, C 40, D 90): **net ≈ -700 lines, -4 files**, guard tests included.

## 3. Migration order, go/no-go written first

1. **A on CPU, today**: wire `gpu_jobs.Runner` and `Session.run` to `proc.py`, nothing else. Proof: Astra's scripts,
   unchanged, in `etx-rocm22 --network=none`: `review09-runner-{timeout,deadline,kill}-repro.py` (R01, R14),
   `review09-runner-fork-race-repro.py` (R15; today `survivors_after_session_done` 2, 4, 4, its `.log:11,21,31`),
   `tests/test_gpu_jobs.py`. **GO: zero survivors in 3/3 runs of each; else stop and report.** A is first because
   B's seal holds only if no writer outlives the reap, and R15 shows one can today.
2. **B**: Astra's review 08/09 fetch, staging, gates fixtures (R03-R06, R08, R12 incl. postmortem, R13) against the new
   entry points all fail closed; the `ds_hour` dry run is unchanged in jobs and order.
3. **C**: `review09-runner-gates-repro.py` (R06, R07); a d1d5 replay (`build_coh` fails): pairs wait for the manifest.
4. **D**: every committed verdict re-derives identically (`reproduce.sh check` step 7: `all` 92.0 / 18.0 GO); R02's
   repro passes without rounding; `rocm_build_check.sh --arm X` gives build X's plan hash. Then Qwen moves onto A-D (G5).

## 4. The paid hour: keep the session as patched; do not delay it for the refactor

**Run `ds_hour.py` at 2ef1e58, code frozen, no more patches**; close the two open findings by procedure: **R12'**, fetch
only with `tools/session.sh fetch <ip> <new dir>`, never `postmortem` before `verify_hour` prints `HOUR VERIFIED`;
**R15**, a `HARD-STOP` in `runner.log` means NOT COMPLETE, and `ps` on the box must show no survivor before fetching.
Step 1 runs on CPU in parallel. Reasons: the measurement core (32/32 + 27/27 per invocation, five alternating rounds,
fixed references and rules) was found sound in all three reviews (`answer-07-review.md:15`, `answer-08-recheck.md:27`,
`answer-09-recheck.md:31`); the open defects sit in stop and fetch, where R12' needs a postmortem into a reused
destination and R15 a hard stop during a fork burst, against ~494 s GPU + ~4 min prep under an 1800 s cap
(`docs/DSV2-D1-D5.md:262,413`); an hour on the refactor would be the first run of unreviewed machinery, the very failure
described here. Raw logs are kept, so D's verifier can re-derive this hour's verdicts later.

**Cost of being wrong.** *Run now:* at most the 0.5 h cap, ≈ $1.50 at the $2.99/h planning rate (`ds_inplace.py:17`),
plus a rerun if an undiscovered defect corrupts a result (the verifier is hardened by three rounds, not proven).
*Delay:* the D2b rerun (3 of 5 rounds, l1 ≈ 0.1 ms ahead of relaxed, `docs/DSV2-D1-D5.md:238-243`) and the D3, D5,
alld4s verdicts wait four steps plus a review each [E: days], and the first refactored hour carries new-machinery risk.
