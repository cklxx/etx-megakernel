# More levers, priced, with their mechanism simulated on the CPU (2026-10-12)

**Condition.** One MI300X VF, Qwen3-30B-A3B, bf16, batch 1, context 1024, 32 generated tokens. ms per decode
step on host wall unless marked device. Anchor: KEY32-on fused **4.180 wall / 4.045 device [M]**
(`results_qwen300_2026-10-11/job2_run4.log:144-145`). Need for 300 tok/s: **0.847 ms**. Labels as in
`docs/PLAN-300-CERTAIN-2026-10-12.md` (CERT): [M] measured, [E] estimate with arithmetic, [R] residual, [U]
unknown (adds nothing). Results directories below live in `~/code/etx/` (outside git). No GPU was used.

## 1. The 0.80 ms anchor drift: not a source or flag change

**Verdict: the drift is the machine, not the build.** The two builds' device code is identical, and vLLM's own
kernels, compiled from byte-identical source by the same compiler, ran **13-19% slower** in the hour session.
It is not a lever at the source level. It is a session risk with a cheap guard (section 5).

**1a. Flags.** The block quoted in the brief (`raw/hour_build.json`) is the **stamps** build (`ETX_INBODY_BUILD=1`,
`ETX_OUT=.../stamps`); `raw/hour_build.json` and `session-wklfw6bc/stamps_hour_build.json` are byte-identical.
The timed **kept** build is `session-wklfw6bc/kept_hour_build.json`, the same block without `ETX_INBODY_BUILD`.
The 10-11 build environment, recovered from the session script that ran it (`git show
7c85efb:examples/sessions/qwen300.py:355-367`) and its log (`results_qwen300_2026-10-11/job2_run4.log:27-30`):

| Variable | 10-11 build (`b_on`) | Hour kept build | Effective difference |
|---|---|---|---|
| `ETX_HIPCC_FLAGS` | `-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH` | same + `-DETX_INBODY_REPORT` | host `printf` only (1c) |
| `ETX_TOPK_KEY32`, `ETX_VL_CHECK` | 1, 1 | 1, 1 | none |
| `ETX_VL_CU` | unset (all `ETX_VL_*` stripped) | 256 | none: default 256 (`examples/vllm_llm/model.py:72`) |
| `ETX_VL_REQUESTS` | unset | 1 | none: default 1 (`model.py:193`) |
| `ETX_VL_TOUCH` | unset | 1 | none: `_on()` defaults to on (`model.py:63-67,138`) |
| `ETX_DEV_RESET` | unset | 0 | none: default 0 (`model.py:154`) |
| `ETX_PREFETCH_ISOLATE_POLLER` | unset | 0 | none: default 0 (`etx/codegen/kernel.py:151` at e068e47) |
| `VLLM`, `ETX_TORCH_HEADERS` | `~/vllm`, `~/rocm-headers/pytorch` | unset | none: import outputs hash-identical (1b) |
| Runtime `ETX_VL_PROGRESS` | 64 | 1 (`qwen300_hour.py:177-178` at e068e47) | host `printf` outside the timer |
| Runtime `ETX_SYNC` | `''` | unset | none: only `block` switches (`host.h:186-187`) |

**1b. Generated code, regenerated on this Mac.** I re-ran `etx.ops.emit`, `etx.importer.adapter` and `etx compile`
from each certificate's `imports.json` at both trees. Every hashed artefact reproduces the VM's sha256 exactly:

| File (certificate `build_identity.json`) | 10-11 hash | Reproduced at | Hour hash | Reproduced at |
|---|---|---|---|---|
| `dsl_tiles.hip` | a0bcc2dc | 7c85efb, a4ca716, d7c1234 | 29e5f250 | e068e47 |
| `megakernel_d0.hip` (VM paths substituted) | 650781f6 | 7c85efb | bc4c8cd3 | e068e47 |
| `plan_data.h` | a3fb4817 | 7c85efb | 88a08466 | e068e47 |
| `vl_plan.txt` / `plan.json` | 046be64d / b32c3299 | both | 046be64d / b32c3299 | both |
| `imp_tiles.hip`, `imports.json`, `orig_*.hip` | d4bf266a, 04274de5, same | both | identical | both |

The 10-11 commit is therefore any of 7c85efb..d7c1234 (one equivalence class for every generated file). The diff:

- `megakernel_d0.hip`: **identical** except three `#include` paths.
- `dsl_tiles.hip` (50 lines): the staged-norm helper gains `template <int FixedK = 0, int FixedR = -1>` with
  `K = FixedK ? FixedK : d.K` (identity at the defaults) and an `#ifdef ETX_DSL_NORM_HOIST` block (C5; undefined).
- `plan_data.h`: one `#if defined(ETX_PRODUCER_LEDGER)` table (undefined). `abi.h`: fields under `#if ETX_INBODY` only.
- **Device IR**: both megakernels compiled to gfx942 LLVM IR locally (`hip_ir_local.sh`, same defines). After
  removing the HIP `cuid` and inline-asm `srcloc` line numbers, **0 differing lines** in 29.2 MB (section 4, S0).
  The VM's resource remarks agree (128 VGPR, 265 SGPR spills, 29,736 B LDS in both `hipcc.log`s).
- Host (`host.hip:714,725-728,758,796,824` at e068e47): `-DETX_INBODY_REPORT` prints one line before `w0`
  and others after the timer, and passes a 5 s watchdog instead of 60 s. The device number is the in-kernel
  `token_time` span (prologue to argmax), which host code cannot lengthen.

**1c. What did change: the machine and the data [M].** Standalone launches of vLLM's **original** kernels in
the import checks (`check_p0.txt`, `check_p1040.txt`, layers 1-47, median of 47 each; S0b in section 4):

| Session | attn original p0 / p1040 us | reduce original p0 | imported attn p0 | imported reduce p0 |
|---|---:|---:|---:|---:|
| vlbest 10-09, vlbest2 10-10 | 7.30 / 8.20 | 6.60 | 9.50 | 10.30 |
| qwen300 10-11 on / off | 7.50 / 8.30, 7.40 / 8.30 | 6.60, 6.50 | 9.30, 9.50 | 10.20, 10.30 |
| hour W kept / W stamps / J kept | 8.50 / 9.30, 8.50 / 9.20, 8.70 / 9.20 | 7.50, 7.10, 7.10 | 10.70-10.90 | 11.70-12.20 |

`orig_attention.hip` has the same sha256 (6735281a) in both builds, and both use the same compiler build
(46fcb339…). So a +13-16% rise in that kernel is not ETX code. The megakernel's +19.6% device rise
(4.840 / 4.045) has the same sign and size. Second, the data were not the same either. CHECK-DSL's layer-47
inputs at the final step are deterministic: four earlier gates (10-09, 10-10, 10-11 on and off) print identical
`topk dev ids [60 117 24 14 97 79 126 122] wts [0.288143 …]`. All four hour gates print `… 126 109] wts
[0.283657 …]` (`kept_token_gate.log`, `stamps_`, `stages_`, J). The golden (bab2f1fb) and `vl.manifest`
(d95a244f) hashes agree, so the staged `vl.bin` bytes or HF snapshot on the hour VM are the [U] candidate. The
hour runner does not pack weights; it uses what was staged (`qwen300_hour.py:98`). Neither data difference
explains a slower position-0 attention kernel. A VF clock or power state does [E].

**Cheapest GPU test (part of every next session, 4 GPU minutes, $0.13):** run `rocm-smi --showclocks
--showpower --showperflevel` and the import check's original-kernel medians first. Then run the kept gate from
both trees (7c85efb and e068e47 rebuilt, same flags) as two pinned pairs, and record `sha256sum vl.bin`.
**Decision:** if the two trees are within 0.030 ms of each other, the drift is environmental and closed. Then, if
original attention is above 7.9 us (+5% over 7.5), the VM is slow: record and reprovision, or rebase every row.
Tree A slower than tree B by ≥ 0.030 ms falsifies this section.
