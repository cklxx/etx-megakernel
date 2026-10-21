# Strongest tested vLLM: 2026-10-09

## Result and evidence

**LEAD | CERT OK | FULL-CLAIM ELIGIBLE** is the session verdict (`S/report.log:6-16`). Strongest means strongest among tested, lever-proven, correct arms, not every legal vLLM configuration.

`R` = `~/code/etx/results_vlbest_2026-10-09` (outside git).
`S` = `R/session-iv3fdc3m`, the finished runner's directory. All `S/` citations below refer to that directory. Earlier attempts elsewhere in `R` are not this verdict's evidence.

## Conditions and protocol

**AMD Developer Cloud** is operator-supplied provenance. **MI300X VF**, **20** cores, kernel **6.8.0-138-generic**, and ROCm **7.14.0** are recorded in `S/record.log:7,18-20`.

The report's conditions, verbatim (`S/report.log:3`):

```text
SYSTEM  Qwen3-30B-A3B bf16 | MI300X VF | ROCm 7.14.0 | bs1 ctx1024 | host wall | 5 rounds | image vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7
```

**32 greedy tokens**, with the HF golden, are the arm command's protocol (`S/S0_r1.log:2`); the report records correctness for every final arm (`S/report.log:6-9`). vLLM uses host-wall generation samples; ETX uses host-wall step means in this report. Its column label “round medians” is preserved below, not used to rename the ETX statistic. The raw JSON retains the vLLM samples (`S/S0_r1.json:11-18`; corresponding JSON for each vLLM arm).

**Operator override:** `ETX_SESS_MAX_SECONDS=6000` is operator-supplied. The logged budget is copied verbatim (`S/budget.log:3`):

```text
BUDGET vm-life=150m runner-cap=100m override=ETX_SESS_MAX_SECONDS
```

This is the runner cap, not a measured VM lifetime or bill. The runner records `session done` (`S/runner.log:306`).

Correctness gates and the timed-build certificate precede screening (`S/runner.log:52-134`). Screening brackets each pass with S0 and reverses the candidate order (`S/runner.log:134-216`). `select_screen` records `"combination": null`; `select_final` selects S3 (`S/select_screen.log:3`, `S/select_final.log:3`). Combination jobs report `NOT-APPLIED C: fewer than two compatible winning axes` (`S/b_r1_C.log:3`, `S/b_r2_C.log:3`).

Final rounds alternate S0 → F → fused → unfused with the reverse order, on the same GPU (`S/runner.log:234-302`). **Five rounds** are the report basis (`S/report.log:3`). **Absolute milliseconds do not cross sessions.**

## Arm table

Host-wall milliseconds per token. Screening round medians below are copied from the `VLLM` log lines. The fetched logs do not print an aggregate screening median or spread; “not logged” preserves that limitation without recomputation. Final-arm aggregates appear verbatim in the next section.

| Screening arm | Configuration | Round medians | Median | Spread | Outcome / evidence |
|---|---|---|---|---|---|
| S0_open | default, opening bracket | 4.867, 4.871 | not logged | not logged | bracket; `S/a_r1_S0_open.log:74`, `S/a_r2_S0_open.log:73` |
| S0_close | default, closing bracket | 4.879, 4.860 | not logged | not logged | bracket; `S/a_r1_S0_close.log:73`, `S/a_r2_S0_close.log:73` |
| S2 | `attention_backend=TRITON_ATTN` | 4.934, 4.881 | not logged | not logged | NO-WINNER; `S/a_r1_S2.log:74`, `S/a_r2_S2.log:73` |
| S3 | `attention_backend=ROCM_AITER_UNIFIED_ATTN` | 4.772, 4.752 | not logged | not logged | WINNER; `S/a_r1_S3.log:79`, `S/a_r2_S3.log:78` |
| S7 | `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 4.872, 4.869 | not logged | not logged | NO-WINNER; `S/a_r1_S7.log:72`, `S/a_r2_S7.log:72` |
| S8 | `--no-detok` | 4.852, 4.867 | not logged | not logged | NO-WINNER; `S/a_r1_S8.log:73`, `S/a_r2_S8.log:73` |

Configurations are in each arm's command header (`S/a_r1_S2.log:2`, `S/a_r1_S3.log:2`, `S/a_r1_S7.log:2`, `S/a_r1_S8.log:2`). Selection and exclusion states are recorded in `S/select_screen.log:3` and `S/select_final.log:3`.

Every excluded search arm, with exact reason strings as serialized in `S/report.log:14`:

```text
SEARCH exclusions {"S1": "NOT-APPLIED S1: MOE-TUNE SKIPPED budget remaining=5756 reserve=3660", "S2": "NO-WINNER", "S4": "NOT-APPLIED S4: Enabled custom fusions: [^\\n]*qk_norm_rope_kvcache", "S5": "NOT-APPLIED S5: Enabled custom fusions: [^\\n]*qk_norm_rope_kvcache", "S6": "NOT-APPLIED S6: Enabled custom fusions: [^\\n]*qk_norm_rope_kvcache", "S7": "NO-WINNER", "S8": "NO-WINNER", "S9": "NOT-APPLIED S9: MOE-TUNE SKIPPED budget remaining=5756 reserve=3660"}
```

S1 and S9 are the dropped MoE levers: the copied reason names remaining and reserve seconds. The runner's exact gate reason is `gate moe_budget !~ '^MOE-TUNE GO$'`; dependent jobs say `a dependency failed` (`S/runner.log:128-133`).

S4, S5 and S6 ran, but the AITER `qk_norm_rope_kvcache` fusion never engaged: the required enabled-fusion proof line is absent. They are **NOT-APPLIED**, not valid timing observations for those levers (`S/report.log:13-14`; full arm logs `S/a_r1_S4.log`, `S/a_r2_S4.log`, `S/a_r1_S5.log`, `S/a_r2_S5.log`, `S/a_r1_S6.log`, `S/a_r2_S6.log`). Their raw JSON and log timings are retained in the data set but are not promoted to lever measurements here.

**Group 3 was never attempted.** The untested list is copied from `S/report.log:13`:

```text
['V2 model runner', 'AITER MoE dispatch policy', 'AITER fmoe tuning', 'block-size 32', 'compile_sizes=[1] + max_autotune', 'skinny GEMM off', 'shuffle KV cache', 'stream-interval >1']
```

## Final result lines, verbatim

The four arm rows below are `S/report.log:6-9`, with its header (`S/report.log:4-5`). Values and precision are unchanged.

| arm | config | round medians | median | spread | 32/32 | proof/cert |
|---|---|---|---|---|---|---|
| ETX fused | timed build | [4.44541935483871, 4.458548387096775, 4.452935483870967, 4.453612903225807, 4.446354838709677] | 4.453 | 0.013 | 5/5 | CERT OK |
| ETX unfused | same kernels, HIP graph | [4.643516129032258, 4.646870967741935, 4.653741935483871, 4.641451612903226, 4.6451935483870965] | 4.645 | 0.012 | 5/5 | CERT OK |
| vLLM default (S0) | none | [4.848, 4.856, 4.859, 4.868, 4.867] | 4.859 | 0.020 | 5/5 | proof OK |
| vLLM strongest tested (R) | {"components": ["S3"], "env": {}, "flags": [], "id": "S3", "kw": {"attention_backend": "ROCM_AITER_UNIFIED_ATTN"}} | [4.745, 4.763, 4.744, 4.744, 4.78] | 4.745 | 0.036 | 5/5 | proof OK |

FUSION, DELTA and SPREAD-BOUND, verbatim (`S/report.log:10-12`):

```text
FUSION  unfused−fused: gains [0.042660942417105975, 0.040526750570970285, 0.04314946002523124, 0.040469819647635234, 0.04280525829681738], median 4.27%, spread 0.27% → ESTABLISHED
DELTA   fused−R: [-0.29958064516129035, -0.3044516129032253, -0.2910645161290324, -0.29038709677419305, -0.33364516129032307] median -0.300 spread 0.043 → LEAD   (vs S0: [-0.4025806451612901, -0.3974516129032253, -0.4060645161290326, -0.4143870967741936, -0.4206451612903228] → LEAD)
SPREAD-BOUND fused−R: True (spread ≤ 0.050 ms and < |median|; diagnostic only)
```

## Registered predictions and outcomes

The fetched report records these predictions verbatim (`S/report.log:15`):

```text
Predictions S1 -0.05…-0.25; S5 -0.1…-0.4; S7/S8 within ±0.1 ms [E]. LEAD survives if F gains <0.40 ms on S0.
```

- **S1:** untested; the budget gate dropped the lever. Its prediction is unresolved, not falsified (`S/report.log:14`).
- **S5:** not applied; no enabled-fusion proof. Its prediction is unresolved, not a measured miss (`S/report.log:14`).
- **S7/S8:** consistent with the registered bound (inference from the copied screening round medians and S0 brackets above); both are `NO-WINNER` (`S/select_screen.log:3`). No numerical delta is recomputed here.
- **“LEAD survives if F gains < 0.40 ms on S0”: held for the selected arm.** The measured screening S3 `"gain": -0.10725000000000007` and `"deltas": [-0.10099999999999909, -0.11350000000000104]` are copied from `S/select_final.log:3`; this is the signed screening change against the S0 brackets, not a recomputed final-round gain. F selects that S3 configuration (`S/select_final.log:3`), and the final outcome is `LEAD` (`S/report.log:11`). The final report does not print an F-minus-S0 aggregate gain; none is invented here.
- **Spread bound:** `True`, as printed in `S/report.log:12`; diagnostic only.

## Timed-build certificate

**97 of 97 imported instances** were checked on the timed build at positions **0** and **1040** (`S/check_certificate.log:3`). The position records are `S/check_timed.log:6,215`. Both checks print **`CHECK OK: 0 launch(es) differ`** (`S/check_timed.log:210,436`).

The checks-only certificate is copied verbatim (`S/check_certificate.log:3`):

```text
IMPORT-CERTIFICATE {"build_sha256": "517b6920cde0904c1e17e0dc059bded765cd21de42a285473be54c1c4a1246ef", "checked": 97, "eligible": false, "planned": 97, "positions": [0, 1040]}
```

**`eligible: false` is expected in `--checks-only` mode, not a failure.** The command explicitly uses that mode (`S/check_certificate.log:2`): the validator initializes eligibility false, assigns it only when validating timing, and accepts checks-only success independently (`examples/vllm_llm/import_certificate.py:177,200,229-231`).

The before/after build identity is unchanged across checks (`S/check_timed.log:3,211-212,437`) and timing (`S/etx_r1.log:3,2121`; corresponding identity lines in each fused/unfused round). The report prints **CERT OK** for both ETX arms and **FULL-CLAIM ELIGIBLE** (`S/report.log:6-7,16`). Each final ETX arm records `matched=32 gen=32` (`S/etx_r1.log:2120` through `S/etx_r5.log:2120`, `S/etxu_r1.log:2120` through `S/etxu_r5.log:2120`); vLLM records `first_divergence=32/32` (`S/S0_r1.log:73` through `S/S0_r5.log:73`, `S/F_r1.log:78` through `S/F_r5.log:78`).

## What this does not cover

One VF, one session. The two dropped MoE levers remain unmeasured; the three AITER-fusion levers never engaged; group 3 remains untested (`S/report.log:13-14`). “Strongest tested” does not establish the strongest possible legal configuration. No bare-metal result, cross-session absolute-millisecond comparison, or VM billing claim follows.

## Data inventory

`R` remains outside git. Local file inventory: **201 regular files / 20134708 file-content bytes**. These are filesystem metadata, not benchmark measurements. No `SHA256SUMS` manifest is present; no manifest verification is claimed. `R` also retains earlier attempts; use `S` for the finished-session verdict.
