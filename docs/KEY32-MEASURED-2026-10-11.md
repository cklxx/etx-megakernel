# KEY32 measured on Qwen3-30B-A3B, 2026-10-11

**KEEP on the shipped unfolded plan.** Five pinned off/on fused pairs saved a
median **0.231 ms/step on host wall**, with **0.016 ms** spread and every saving
positive (`results_qwen300_2026-10-11/job2_run4.log:138`). This is a measurement
[M], replacing the ledger's `0–0.145 ms [E]` estimate. The measured median exceeds
that estimate's upper end by `(0.231 / 0.145 - 1) × 100 = 59.31%` (about 60%).

Conditions: Qwen3-30B-A3B, bf16, one MI300X, GPU_ID=0, batch 1, context 1024,
32 greedy tokens. Both `on` modes and both `off` modes have five pinned rounds.
The fused `on` arm is the **shipped-unfolded-plan** configuration; this result
does not qualify a folded plan. All source paths beginning `results_` refer to
the untracked local archive in `~/code/etx/`.

## Arm medians and bases

| KEY32 arm | Execution | Host wall median, ms/step [M] | Device median, ms/step [M] | Log lines |
|---|---|---:|---:|---|
| off | ETX fused | 4.410 | 4.279 | `job2_run4.log:139-140` |
| off | Same-kernel HIP graph | 4.596 | 4.562 | `job2_run4.log:141-142` |
| on | ETX fused, shipped unfolded plan | 4.180 | 4.045 | `job2_run4.log:144-145` |
| on | Same-kernel HIP graph | 4.357 | 4.323 | `job2_run4.log:146-147` |

The **wall** columns are host-wall per-step timing. The **device** columns are
device latency, not host throughput; do not substitute 4.045 for 4.180 when
computing the 300 tok/s gap. No absolute milliseconds are compared across
sessions. The log also prints a stored Job 1 vLLM reference
(`job2_run4.log:149-191`); this KEY32 decision is not a new strongest-vLLM system
verdict. The fused/graph summaries are explicitly labelled
`FUSION-DIAGNOSTIC (not adjacent fused/graph pairs)` in the source
(`job2_run4.log:143,148`), so they are not a new fusion-rule certificate.

## Five paired wall savings and the unchanged rule

Savings are `off fused wall - on fused wall` within each pinned round. In round
order, the source prints these full values (`job2_run4.log:138`):

```text
[0.23099999999999987, 0.23000000000000043, 0.2450000000000001,
 0.23399999999999999, 0.2290000000000001] ms
median=0.23099999999999987
spread=0.016000000000000014
KEEP=True
```

The pre-registered rule is **every wall saving > 0, median >= 0.03 ms, and
median > 2 × (max - min)** (`job2_run4.log:13`). It passes: the smallest saving
is 0.229 ms, the median is 0.231 ms, and `0.231 > 2 × 0.016 = 0.032 ms`.
These are rounded displays of the printed values, not changed measurements.
The paired-saving median is distinct from the difference of arm medians:
`4.410 - 4.180 = 0.230 ms`; the ledger records **0.231**, not 0.230.

Correctness evidence: both builds have fresh **97/97** imported-instance checks
at positions **0 and 1040** (`job2_run4.log:47,60` and each arm's
`certificate_<arm>/check_p0.txt` and `check_p1040.txt`). Local CPU validation of
both archived certificates with `timing=False` passes; its `eligible=False`
field is the checks-only result, not a timing/fusion qualification. All 20
timed runs' `r<1-5>_<off|on>_<fused|graph>.log` RESULT lines record
`matched=32 gen=32 matched_each=32`. Routing IDs and weights are bit-identical
across all 48 layers and eight replicas at both positions
(`job2_run4.log:67-68`). The run ends `JOB 2 COMPLETE` (`job2_run4.log:192`).

## Remaining gap to 300 tok/s

Using the measured fused `on` **host-wall median**, current throughput is
`1000 / 4.180 = 239.23 tok/s`. The requested target is **3.3333 ms/step**, so
the remaining cut is **`4.180 - 3.3333 = 0.8467 ms/step`**, about
`0.8467 / 4.180 × 100 = 20.26%` of the measured step time. Using the exact
`1000 / 300 = 3.333333...` target gives `0.846666... ms`, the same result at
four decimal places. **300 tok/s is not reached.** This gap is arithmetic from
this session's measurement, not an estimated additional KEY32 gain.

Section 5 of `docs/PLAN-300TOKS-2026-10-11.md` replaces the KEY32 row and
recomputes its dependent planning arithmetic: upper envelope
`0.814 - 0.145 + 0.231 = 0.900 ms [E]`, historical-anchor unpriced amount
`1.137667 - 0.900 = 0.237667 ms [E]`. That envelope transfers a paired saving
to the original planning anchor; it remains an estimate, distinct from the
**0.8467 ms** gap derived directly from this run. Other allowances remain
unmeasured. No original pass or pursuit rule is changed.

The machine was destroyed after this run. Re-measurement is unavailable; this
record uses the retained logs and CPU checks only. No GPU, provisioning,
network spend or reduced-precision path was used for this change.

## Retry policy and CPU regression evidence

`examples/sessions/qwen300.py` now moves each stale log before rerunning its
step to `log_history/<log stem>/attempt<N>/<filename>`, choosing the first
unused per-log attempt directory. The canonical log contains the latest run.
Both remain full and unfiltered under the same session directory. The policy
applies to every `step()` log, including `etx_env.log`, and to retained
`off`/`on` compiler logs and `compiler_version.txt`. Certificate and routing
directories use `mkdir(exist_ok=True)`; no other step's data is deleted or
borrowed. Commands, deadlines, timeouts, ordering and pass/fail rules are
unchanged. This policy is also in the module docstring and the session
description in `docs/XCD-TIMELINE-2026-10-11.md`.

`tests/test_qwen300_session.py::test_step_retry_keeps_both_full_logs` executes
real CPU subprocesses, retries after both success and exit 7, checks both
stdout and stderr, verifies byte-identical archived evidence, and executes a
third attempt without changing another step's log. Before modifying the
script at base `a4ca716`, running
`.venv/bin/python -m pytest -q tests/test_qwen300_session.py -k step_retry_keeps_both_full_logs`
returned **2 failed, 15 deselected**, both failures at the second execution:
`RuntimeError: refuse to overwrite .../retry.log` (`qwen300.py:114` at that
base). After the fix both cases pass. A separate mocked CPU test,
`test_job2_retry_keeps_directories_and_compiler_logs`, reaches both arms'
certificate/routing steps twice, preserves existing directory evidence, and
keeps both generations of all retained compiler outputs.

Validation before commit: `.venv/bin/python -m pytest -q` completed with
**1276 passed, 14 skipped** in 184.51 seconds; the CPU-only
`python -m examples.sessions.qwen300 --dry-run` also passed. A local AST
comparison against the base confirmed unchanged step/container/capture
commands and arguments, job definitions, pre-registration header, flags,
session caps, container deadlines, correctness gates and `main()`.

The source retry failures are recorded at
`attempt1/job2_rerun.log:71` (trace grid-name parser, already fixed by
`dbd78ad`) and `attempt2/job2_run3.log:41` (`certificate_off` already exists).
The first attempt's `etx_env.log` is still readable in `attempt1/`. Three GPU
sessions were consumed by the failures reported in the brief; this CPU fix
does not infer a dollar cost from those logs.
