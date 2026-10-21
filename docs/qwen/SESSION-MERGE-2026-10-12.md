# Qwen300: one 140-minute session, existing KEY32 evidence

This update supersedes the 74-minute budget and deferred-arm descriptions in
`docs/qwen/qwen300-hour-2026-10-11.md`; that earlier document retains its historical
CPU results. CPU only: no GPU, provisioning, Docker GPU run or push.

The budget/arm change is commit `e780405` (base `54ea65c`). The separate
KEY32 evidence-reuse change adds documentation, with no off/on timing tranche.
`docs/qwen/CLAIM-239-2026-10-12.md` is an **uncommitted owner-review draft**; it says
“owner approves before this goes into AGENTS.md”. AGENTS.md is unchanged.

| Setting | Old | New | Current source |
|---|---|---|---|
| Runner cap / job timeout | 4440 s | 8400 s; larger caps rejected | `examples/sessions/qwen300_hour.py:51` |
| A / first-stop command deadline | 900 / 885 s | unchanged | `examples/sessions/qwen300_hour.py:12` |
| B | 1320 s | unchanged | `examples/sessions/qwen300_hour.py:13` |
| C | 1620 s | 4020 s | `examples/sessions/qwen300_hour.py:13` |
| NO_TOUCH | enabled, 1620 s | unchanged, full diagnostic + 3 pairs | `examples/sessions/qwen300_hour_arms.json:8` |
| ISOLATE_POLLER | disabled, 600 s | enabled, 1200 s, 3 pairs + round-trip gate | `examples/sessions/qwen300_hour_arms.json:13` |
| C6 | disabled, unallocated | enabled, 1200 s, full 3 pairs if eligible | `examples/sessions/qwen300_hour_arms.json:18` |
| Environment guard | none | 240 s before A | `examples/sessions/qwen300_hour.py:52` |
| D | remaining cap, nominal 590 s after full A/B/C and report reserve | remaining cap, at least 1910 s after guard and full A/B/C | `examples/sessions/qwen300_hour.py:14` |
| Reporting reserve | 10 s | unchanged | `examples/sessions/qwen300_hour.py:53` |
| KEY32 timing tranche | none | none; reuse measured acceptance explicitly | `examples/sessions/qwen300_hour.py:24` |

Budget arithmetic: **240 + 900 + 1320 + 4020 + 1910 + 10 = 8400 s**.
C allocates **1620 + 1200 + 1200 = 4020 s**, against estimates of
**1575 + 1140 + 1085 = 3800 s**, leaving **220 s** of margin. Fresh import
checks at both positions, CHECK-DSL and 32/32 gates are included in the arm
costs. C6's new estimate is **9 × 120 + 5 = 1085 s** (three correctness host
calls, six paired timing calls, ancestry); it estimates measurement of a
ready prototype, not building/debugging an adapter. No arm was cut.

D keeps five adjacent fused/graph pairs and five adjacent fused/S3 pairs,
with alternating order, plus its graph correctness gate. The prewarmed
upper estimate is **16 × 28 + 5 × 97 = 933 s**, plus **3 × 28 = 84 s** for
fresh certification if the final selection is a previously untested
combination: **1017 s**, against **1910 s**, leaving **893 s**. B's warm upper
estimate is **18 × 28 = 504 s**, against 1320 s. These call timings come from
prior logs summarized in `docs/qwen/PREFLIGHT-QWEN300-HOUR-2026-10-11.md:95` and
are **estimates**, not new measurements or guaranteed run times. A retains its
15-minute S1 deadline, including preflight and correctness. The combined
planning estimate, reserving all 900 s of A, is **240 + 900 + 504 + 3800 + 1017 +
10 = 6471 s**, below the cap by **1929 s**; the stricter tranche allocations
above still reserve 8400 s in total. This is the honest prewarmed basis for
keeping every arm's complete population.

At the harness's deliberately conservative all-cold expected durations,
D alone would require **(16 + 3) × 120 + 5 × 180 = 3180 s**; this is not the
prewarmed budget basis. Preparation and cache prewarming must finish before
the runner clock. A miss still yields NOT-COMPLETE, with complete logs and
no shorter timing population substituted. The hard stop is 140 minutes,
10 minutes below 150 minutes; allocation/setup age must also fit the actual
machine life. No machine was allocated here.

## Eligibility and preserved guards

All five manifest arms are enabled. C6 remains **not runnable on this tree**:
there is a contract/hook, but no committed full-model adapter
(`docs/qwen/ISA-FIXES-2026-10-11.md:174`). Its existing sealed-build, prototype
certificate/commit, ancestry and native resource gates still apply. Merely
setting enabled does not confer eligibility; the current missing prototype
reports `C.eligibility.C6 NOT-COMPLETE`. The template includes its build path
but `--prepare` continues to build only integrated variants. No prototype
certificate, resource observation or seal was invented. Isolation similarly
requires its staged, sealed round-trip probe for acceptance. Reserving these
budgets does not certify that either prerequisite has been delivered.

S1 still closes the target route below 0.088 ms at 15 minutes. S2's unchanged
sum is G + R2 + H, with a 0.324 ms bar; it retains the existing early decision
after A and X5, before the B body pairs: X5 can refresh R2 before S2. No stop or
registered threshold was moved or weakened. Authorized arrive/setup prices
remain 0.30/0.15 us. Fresh-correctness, exact timing populations, fail-closed
NOT-COMPLETE, per-step/subprocess deadlines, silence probes and full logs are
unchanged. The historical AST comparison against `54ea65c` checked these functions and the
REGISTERED/THRESHOLD_SOURCES/BASE_FLAGS assignments.

## Existing 239 tok/s evidence; zero new KEY32 timing

The accepted record is `docs/qwen/KEY32-MEASURED-2026-10-11.md` and the retained
archive `/Users/ckl/code/etx/results_qwen300_2026-10-11/`.
Five pinned off/on fused pairs saved a median **0.231 ms**, spread **0.016 ms**,
minimum **0.229 ms**, with KEEP true (`job2_run4.log:138`). The unchanged rule
at `job2_run4.log:13` passes, including **0.231 > 2 × 0.016**. The measured
KEY32-on host-wall median **4.180 ms** yields **239.23 tok/s**
(`job2_run4.log:144`), not 300 tok/s. Same-kernel graph is **4.357 ms**
(`job2_run4.log:146`); the stored Job 1 S3 reference is **4.967 ms**
(`job2_run4.log:189`). These are not new adjacent-pair fusion or strongest-
system certificates. No absolute milliseconds are compared across sessions.

CPU reads revalidated both archived identity-bound 97/97 certificates at
positions 0 and 1040 (`job2_run4.log:47,60`) and all 20 timing logs' exact
32/32 RESULT lines (`r1_off_fused.log:48`, `r1_on_fused.log:48`, and the
other round/mode files listed in the draft). This reuses the completed
measurement and is not new GPU execution.

No generated-build input, builder flag or build recipe changed in these two
commits. Session source hashes/seals do change; old staging seals must not be
relabelled as fresh CPU gates. This is distinct from a binary build identity:
no equality between the older measured binaries and unrelated newer parent
changes is asserted. Any changed timed identity still receives its fresh
correctness in the tranche that already measures that candidate/diagnostic/
combination. An unchanged historical KEY32 acceptance buys no separate rerun.

## CPU validation

The full unfiltered logs are retained locally under
`results_session_merge_cpu_20261012/`. No shell file changed; the registered
job command passed `bash -n -c`, also exercised by the session dry run.
Focused tests passed **98/98**. The full-suite result is recorded below.
The AST guard audit and `git diff --check` passed. Archived checks-only
certificates correctly return `eligible=False`; that is not a timing verdict.

Historical session dry-run output (CPU-only, before the 240 s guard); operating numbers live in
`examples/sessions/qwen300_hour.py` and the corrected budget account above:

```text
{
  "budgets_s": {
    "A": 900,
    "B": 1320,
    "C": 4020,
    "D": "remaining session time (>=2150s)",
    "report": 10
  },
  "manifest": {
    "schema": 1,
    "B": [
      {
        "id": "C1",
        "build": "C1",
        "kind": "shape",
        "flags": {
          "ETX_VL_ROUTER_NARROW": "1"
        },
        "source": "docs/qwen/ISA-FIXES-2026-10-11.md:31",
        "overlap": "router-body",
        "enabled": true
      },
      {
        "id": "C5",
        "build": "C5",
        "kind": "shape",
        "defines": [
          "ETX_DSL_NORM_HOIST"
        ],
        "source": "docs/qwen/ISA-FIXES-2026-10-11.md:135",
        "overlap": "router-body",
        "enabled": true
      }
    ],
    "C": [
      {
        "id": "NO_TOUCH",
        "build": "NO_TOUCH",
        "diagnostic_build": "NO_TOUCH_stamps",
        "kind": "body",
        "enabled": true,
        "defines": [
          "ETX_NO_TOUCH"
        ],
        "flags": {
          "ETX_VL_TOUCH": "1",
          "ETX_PREFETCH_ISOLATE_POLLER": "0"
        },
        "budget_s": 1620,
        "estimated_required_s": 1575,
        "keep_threshold": {
          "complete_adjacent_pinned_pairs": 3,
          "every_wall_gain_ms_gt": 0,
          "median_wall_gain_ms_ge": 0.03,
          "median_gt_spread_multiplier": 2,
          "additional": null
        },
        "source": "docs/qwen/PLAN-300TOKS-2026-10-11.md:310,314; docs/qwen/XCD-PREFETCH-2026-10-11.md:71-76,125; etx/codegen/kernel.py:297"
      },
      {
        "id": "ISOLATE_POLLER",
        "build": "ISOLATE_POLLER",
        "kind": "body",
        "enabled": true,
        "flags": {
          "ETX_VL_TOUCH": "1",
          "ETX_PREFETCH_ISOLATE_POLLER": "1"
        },
        "budget_s": 1200,
        "estimated_required_s": 1140,
        "keep_threshold": {
          "complete_adjacent_pinned_pairs": 3,
          "every_wall_gain_ms_gt": 0,
          "median_wall_gain_ms_ge": 0.03,
          "median_gt_spread_multiplier": 2,
          "round_trip_median_us_lt": 2.0
        },
        "source": "docs/qwen/VERIFY-GAPS-2026-10-11.md:141-193; examples/vllm_llm/prefetch_isolation_cpu.py"
      },
      {
        "id": "C6",
        "build": "C6",
        "kind": "fusion",
        "enabled": true,
        "budget_s": 1200,
        "estimated_required_s": 1085,
        "prerequisites": "Contract/hook only today; requires a sealed committed full-model adapter, prototype_certificate, prototype_commit, expected_vgpr and expected_lds_bytes before qualification",
        "source": "docs/qwen/ISA-FIXES-2026-10-11.md:246"
      }
    ]
  },
  "thresholds": {
    "A1.busy": "instrument classification only: real >=0.100 ms; dead <0.050 ms",
    "A1.free": "instrument classification only: real >=0.050 ms",
    "A1.floor": "1.00 us/edge; retain below-floor count/min, never clip raw gaps",
    "A2.route": "available body headroom >=0.142 ms / 2.96 us/layer; estimate, no removal proof",
    "A3.host": "name stages accounting for unexplained 0.086 ms beyond priced 0.049 ms",
    "A4.reset": "pursue only >0.030 ms/step, host and device separately",
    "A.stop": "S1 at <=15 minutes: G >=0.088 ms (1.833 us/layer) in named above-floor gap atoms, else close target route",
    "B.stop": "S2 after A, before B: G + R2 + H >=0.324 ms (whole-target 6.743 us/layer), else close target route; unknowns count zero",
    "B.keep": "3 pairs, all faster, median >=0.030 ms, median >2*spread",
    "C.fusion": "net >=1.042 us/layer; VGPR <=128; LDS <=29752 B",
    "C.body": "3 pairs, all faster, median >=0.030 ms, median >2*spread",
    "C.NO_TOUCH": "3 complete adjacent pinned pairs; every wall gain >0; median >=0.030 ms; median >2*(max-min); no additional KEEP threshold",
    "C.ISOLATE_POLLER": "3 complete adjacent pinned pairs; every wall gain >0; median >=0.030 ms; median >2*(max-min); fresh measured round trip <2.0 us",
    "D.target": "5 complete rounds, host-wall step <=3.3333 ms, all fresh correctness gates",
    "D.fusion": "5 adjacent pinned pairs; every gain >0; median >2%; median >2*spread",
    "D.system": "5 adjacent pinned pairs; LEAD all deltas <0, median <=-0.050 ms"
  }
}
Session hard stop: 8400s from runner start (includes setup)
  0 qwen300_hour                     GPU 0    timeout  8400s  after: -

```

Actual full-suite output:

```text
.venv/bin/python -m pytest -q
1453 passed, 14 skipped in 183.64s (0:03:03)
```

The focused result was **98 passed in 1.56s**. The two new CPU regressions
exercise all C arms' complete correctness/pair work with an advancing budget
clock, and reject enabled C6 when its staged prototype is absent. Updated
regressions test 8400 acceptance / 8401 refusal, the 8390-second inner limit,
continued complete isolation/C6 measurements after a partial NO_TOUCH arm,
and unchanged three-/five-pair populations. These tests are not GPU evidence.
Both dry-run entry points passed; the second printed exactly:

```text
Session hard stop: 8400s from runner start (includes setup)
  0 qwen300_hour                     GPU 0    timeout  8400s  after: -
```

No new binary was built or timed. The full compiler/build-matrix staging gates
were not regenerated for this documentation/scheduling change, and no fresh
staging seal is claimed. All preserved staging gates still apply to a future
GPU session. GPU cost for this work is **0 hours / $0**.
