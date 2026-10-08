# SPEC: ETX against vLLM on one MI300X, on one timing basis, beside the published NVIDIA rows

Specification only (2026-10-08, base `5f6b603`). The session author commits `examples/sessions/nvref.py`,
`examples/vllm_llm/nvref_report.py` and `tests/test_nvref_report.py` before provisioning. `~/code/etx/results_*` is outside git.

## 0. Why 227 tok/s does not answer the question yet
- 4.405 ms is a device span: `~/code/etx/results_tiles/ab2_all_x.log:8` reads `median_ms=4.405 wall_ms=4.548` (32/32;
  rounds 1 and 3 in `ab1_all_x.log:8`, `ab3_all_x.log:8`; the set is not yet in `docs/DATA.md`). vLLM's figures are
  host time (`docs/VLLM-LADDER-2026-10-01.md:30-31`): ETX read 0.045 ms (0.9%) behind on the device basis and 0.242 ms
  (5.0%) on the wall basis (`:49`), from 0.197 ms of per-step host work the device timer excludes (`:38,41`; 0.143 ms
  in the tile session). The ladder ran on an MI300X VF, ROCm 7.14 (`:3`), the tiles on a Hot Aisle box, and absolute
  ms/token does not transfer between machines (`docs/DATA.md:64`). Both arms therefore run in one session, on one GPU.

## 1. Timing basis (decided): host wall time per decode step, batch 1
It is the only basis vLLM's harness yields (`examples/llm/vllm_bench.py:4-5,74`), and it includes the per-step host
cost a user pays. **vLLM stays unchanged; ETX changes** from the device span (`median_ms`, kernel stamps,
`examples/vllm_llm/host.hip:693`) to host wall time, measured the same way:
- vLLM: `(t(32 new) - t(1 new)) / 31`, wall time around `llm.generate` (`vllm_bench.py:63-67,74`): a 31-step mean.
- ETX: the `std::chrono::steady_clock` window `host.hip:625` -> `:647` around `vl_set_params` + `hst.run()`
  (`run()` = reset, residency check, launch, wait: `etx/runtime/include/etx/host.h:424-429`). With `ETX_VL_PROGRESS=1`
  each step prints `  step <t>: <ms> ms wall` (`host.hip:649`) after the window closes. **The ETX figure is the mean of
  t = 1024..1054 in the last run**: the 31 steps after the first generated token, as in vLLM's difference (prompt 1024,
  steps 1055, `host.hip:207`; the default `--repeat 2` keeps the last run, `host.hip:116,621`).
- Cross-check: RESULT `wall_ms` (median of 32 steps, `host.hip:648,878,885`) within 0.020 ms of that mean, otherwise
  the report prints `BASIS-MISMATCH` and both values.
- Residual, not measured: vLLM's window also holds sampling, output processing and scheduling; ETX keeps tokens on
  the device until the end (`host.hip:655`), which favours ETX.

## 2. Prompt set (decided): the committed golden, not ShareGPT
- `examples/llm/results/golden_c1024_qwen3-30b-a3b.txt`, sha256
  `bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484`: line 1 `prompt:` + 1024 ids, line 2 `gen:` + 32
  HF-greedy ids, line 3 `margin:` + 32 values. Fleet's benchmark prompt (`docs/ETX_Technical_Design.md:939`).
- To the VM: `tools/session.sh push` rsyncs `~/code/etx` (`tools/session.sh:17-18`; the `results_*` exclusion does not
  match `examples/llm/results`). `prep` copies it to `~/llm/qwen3-30b-a3b/golden_c1024.txt` (`examples/sessions/tiles.py:50`).
  Job `golden_sha` (cpu, after `prep`; every gate and timing job lists it in `after`; expect `golden_c1024.txt: OK`):
  `echo "bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484  $HOME/llm/qwen3-30b-a3b/golden_c1024.txt" | sha256sum -c`
- Not ShareGPT: the ETX host checks one golden per request; each ShareGPT prompt would need a new HF golden (new GPU
  data). No internet: the prompt needs none, but `dl` (weights) and `docker_pull` do; no fallback. They fail, dependants
  are skipped (`etx/tools/gpu_jobs.py:10`); independent branches still finish. Missing required evidence makes the
  report NOT-COMPLETE. Delete the VM when the planned runnable work is done (or at the hard stop).
- Model `Qwen/Qwen3-30B-A3B`, not the third party's `-Instruct-2507`: only the former has a golden and prep path. Its
  HF `config.json` (fetched 2026-10-08) against `examples/llm/results/config_qwen3-30b-a3b.json:1`: every shape field
  equal; only `rope_theta` (1e7 vs 1e6) and `max_position_embeddings` (262144 vs 40960) differ. Same cost (estimate).
  `dl` also prints `HF_REV` from `huggingface_hub.model_info('Qwen/Qwen3-30B-A3B').sha`.

## 3. vLLM arm: `examples/llm/vllm_bench.py`, unchanged
Image `vllm/vllm-openai-rocm:v0.27.1`, vLLM 0.27.1 (`docs/DATA.md:69-70`). Default flags, AITER off, the strongest
measured: default 4.862 vs `--custom-ops all` 4.889 (`VLLM-LADDER:15-16`); AITER off 4.836 vs on 4.916
(`examples/llm/results/vllm_aiter0_qwen3-30b-a3b.json:6`, `vllm_aiter1_….json:6`). Batch 1 only; 32 greedy tokens,
`ignore_eos` (`vllm_bench.py:35,64`); warm-up `gen(32)` + `gen(1)` per invocation (`:69`); 5 runs, one invocation per
round, `--repeats 1`, as in `~/code/etx/results_2026-10-01/vllm_ladder/run.sh:11`. Job `vllm_r<R>` (GPU 0, timeout
420, quiet_s 240):
```
sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$GPU -v $HOME:$HOME -w $HOME/etx --entrypoint python3 \
  vllm/vllm-openai-rocm:v0.27.1 examples/llm/vllm_bench.py --model $HOME/llm/qwen3-30b-a3b/hf \
  --golden $HOME/llm/qwen3-30b-a3b/golden_c1024.txt --label default --batch 1 --tokens 32 --repeats 1 --json $HOME/sess/vllm_r<R>.json
```
Expected: `VLLM label=default aiter=0 … batch=1 version=0.27.1 context=1024 median_ms=<x> … first_divergence=32/32`
(`vllm_bench.py:79-82`); anything else is a failed run. `docker_pull` (cpu lane, timeout 1200, quiet_s 900) also prints
`IMAGE` + `docker inspect --format '{{index .RepoDigests 0}}'`.

## 4. ETX arm: `examples/sessions/nvref.py` under `etx/tools/gpu_jobs.py`
- Copy `tiles.py`: jobs `apt`, `etx_env`, `llm_env`, `dl`, `prep` verbatim (`:34-50`). One timing build, `all`, with
  `ETX_HIPCC_FLAGS='-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH'` (`:53,62-68`; the job prints the
  megakernel resource line and compares it with the previous session's spill 9 / scratch 40, informational only), on the shipped defaults: DSL GEMV, top-k, fold, attnprep, NT, VECNORM
  (`examples/vllm_llm/build.sh:31-35,106,110`). Job `record` (cpu): `rocm-smi --showproductname`,
  `cat /opt/rocm/.info/version`, `uname -r`, `nproc`, `lspci | grep -i mi300` (VF or bare metal).
- Correctness first, GPU 0. `gate_all` = `tiles.py:81-85` with b=all: needs `matched=32 gen=32` and `CHECK-DSL OK`.
  `gate_unf`: the same binary with `--unfused --repeat 1`, needs 32/32; it gates only the optional unfused arm.
- Two check builds, both with the same four DSL flags and `ETX_VL_CHECK=1`: `build_chk` keeps the default XCD
  replicas; `build_chk1` additionally sets `ETX_VL_XCD=0`, giving NR=1. `ETX_VL_XCD` is on by default
  (`model.py:63-68,164`), not uncertain. `check_imp` retains the `--check 2` demonstration of the documented
  per-XCD refusal (`host.hip:515`). `check_imp1` runs `--check 2` and requires
  `CHECK OK: 0 launch(es) differ` (`host.hip:603`); its process must also exit successfully.
  Both check jobs require only their own build, `prep`, and `golden_sha` to succeed; preceding gates/checks are
  completion barriers (`after_any`). Neither check gates timing jobs through `after`.
  The report prefers `check_imp1`; failure, timeout, a recorded skip, or unrecognized output refuses the report.
  Only when it is absent from both logs and status may `check_imp` supply legacy OK or the documented
  `IMPORT-CHECK n/a (host.hip:515)`. `CHECK FAILED` in the selected check always refuses.
- Timing success dependencies (`after`): `vllm_r1..5` needs `golden_sha`, `docker_pull`;
  `etx_r1..5` needs `golden_sha`, `build_all`, `gate_all`; `etxu_r1..5` adds `gate_unf` to those ETX dependencies.
- Timing, R = 1..5, GPU 0: odd rounds `vllm_r`, `etx_r`, `etxu_r`; even rounds reversed.
  Each job uses `after_any` for all preceding timing jobs and the initial gates/checks. This includes adjacent
  round barriers and prevents an immediately skipped neighbour from allowing a later round to overtake an
  unfinished earlier round. A failed unfused branch does not skip vLLM or fused ETX timing.
  ETX command (timeout 300, as `tiles.py:104`), `--unfused` only in `etxu_r` (same-tiles HIP graph, `host.hip:122,882`):
  `set -o pipefail; echo "ETX_ENV ETX_UNFUSED_SYNC=${ETX_UNFUSED_SYNC:-} ETX_UNFUSED_NOGRAPH=${ETX_UNFUSED_NOGRAPH:-}";
  ETX_VL_PROGRESS=1 stdbuf -oL $HOME/sess/b_all/run $HOME/llm/qwen3-30b-a3b $HOME/sess/b_all/vl_plan.txt [--unfused]`
  (the ETX_ENV line is what `examples/vllm_llm/pairs.py:6-10` parses).
- Last, nice to have (each waits for all preceding jobs via `after_any`; failures do not stop the tail): `hostprof` (fused run, `ETX_HOST_PROF=1`, prints `HOSTPROF steps …`, `host.h:387`), `etx_noprog`
  (fused run without `ETX_VL_PROGRESS`: print effect), `vllm_aiter_r1..3` (`-e VLLM_ROCM_USE_AITER=1`).
- `hostprof` and `etx_noprog` use the fused success dependencies; AITER uses the vLLM dependencies.
- `report` (cpu, only `after_any` barriers, including the last job): `.venv/bin/python -m examples.vllm_llm.nvref_report $HOME/sess`, then
  `python -m examples.vllm_llm.pairs <etx_r*/etxu_r* logs in run order> --pairs 5 --basis wall` for its table only. Failure of this optional table is logged and does not fail a successful primary report.

## 5. Report: required system comparison, optional fusion comparison
`nvref_report.py` validates everything before printing any performance number. vLLM and fused ETX each require
five valid runs, 32/32 on every run, and fused ETX requires exactly 31 measured steps in the last repeat.
The required gates, import check, golden checksum and provenance must also validate. Failure prints only
`REFUSED: <reason>. No performance numbers.` and `VERDICT NOT-COMPLETE`.

The unfused arm is optional: a missing/failed `gate_unf`, any missing/failed `etxu_r*`, wrong mode, non-32/32 result,
or invalid step count suppresses the unfused row and numerical FUSION line. Print `FUSION unavailable (<reason>)`
instead; vLLM, fused ETX, DELTA and VERDICT remain available. No partial unfused statistics are printed.
The unfused row and FUSION line below appear only with five fully valid unfused runs and a valid gate.
```
| source | system | hardware | basis | runs (ms/token) | median | mean | spread | tok/s | 32/32 |
| this session | vLLM 0.27.1 default     | MI300X <record> | host wall, (t32-t1)/31    | r1..r5 | m | a | max-min | 1000/m | 5/5 |
| this session | ETX fused, 4 tile flags | MI300X <record> | host wall, 31-step mean   | r1..r5 | … |
| this session | ETX unfused, HIP graph  | MI300X <record> | host wall, 31-step mean   | r1..r5 | … |
| this session | ETX fused (secondary)   | MI300X <record> | device span, not vs vLLM  | r1..r5 | … |
| third party eliovp.com 2025-09-26, not reproduced | vLLM 0.10.2, CUDA 13 | H200 | unknown | - | - | 5.28 | - | 189.5 | not checked |
| third party eliovp.com 2025-09-26, not reproduced | vLLM 0.10.2, CUDA 13 | B200 | unknown | - | - | 5.55 | - | 180.1 | not checked |
| third party eliovp.com 2025-09-26, not reproduced | Paiton, ROCm 6.4.1 | MI300X | unknown | - | - | 5.28 | - | 189.4 | not checked |
| third party eliovp.com 2025-09-26, not reproduced | vLLM 0.10.0        | MI300X | unknown | - | - | 6.16 | - | 162.2 | not checked |
DELTA r1..r5 (ETX fused - vLLM, same round) median spread | FUSION (unfused - fused) median | IMPORT-CHECK OK|n/a
VERDICT LEAD|TIE|LOSS|INSUFFICIENT|NOT-COMPLETE | image <digest> | HF_REV <sha> | machine <record>
```
Third-party rows: Instruct-2507, ShareGPT, 256 output tokens, mean of 5; ms/token = 1000 / tok/s. Unknown: whether
their tok/s includes prefill or TTFT (their methodology page could not be fetched for this spec).

## 6. Claims
- **Supported:** on this MI300X (as recorded), Qwen3-30B-A3B bf16, batch 1, 1024-token prompt, 32 greedy tokens, host
  wall per decode step, 5 interleaved runs: ETX X ms against vLLM 0.27.1 default Y ms, and, only when the optional arm is complete, ETX fused against unfused.
- **Not supported:** anything about H200 or B200, including "matches or beats NVIDIA": checkpoint, prompts, output
  length, software and basis differ, so no ratio chaining. Their stock-vLLM MI300X 6.16 against our vLLM 4.862
  (`VLLM-LADDER:15`) already shows protocol and version moving the number about 27% (uncontrolled observation). Also
  not: batch > 1, serving latency, other machines (`DATA.md:64`). The ETX GEMVs and MoE are ETX's own DSL tiles
  (`build.sh:31-35`; only attention, reduce, rms_norm, silu are vLLM's, `:36-39`), so a lead is an ETX-system result;
  the AGENTS.md section 1 claim rests only on the fused-against-unfused pair.

## 7. Go/no-go, fixed before the run
Δ_R = ETX fused 31-step mean - vLLM per-token, same round. Threshold 0.050 ms, over twice the largest observed spread
(vLLM 0.020, `VLLM-LADDER:15`; ETX wall 0.022, `results_tiles/ab1..3_all_x.log:8`).
- The primary verdict uses only fused ETX versus vLLM; unavailable fusion evidence does not change it.
- **LEAD:** all five Δ_R < 0, median <= -0.050. **LOSS:** all five > 0, median >= +0.050. **TIE:** |median| < 0.050.
  **INSUFFICIENT:** otherwise. If AITER's median is > 0.050 below default, recompute against AITER and print both.
- Prediction (estimate, across machines): Δ about -0.3 ms (ETX 4.548, `ab2_all_x.log:8`; vLLM 4.836, `vllm_aiter0…json:6`).
- LEAD: write `docs/NVREF-<date>.md`, index the set in `docs/DATA.md`, update `README.md:48-50`. NVIDIA stays open;
  the cheapest next step is `vllm_bench.py` with the same golden on one H200, which needs no CUDA port of ETX
  (`docs/ROADMAP.md:228`). Provider and cost unknown.
- TIE (the null result): the device-basis lead does not survive host time, and ETX only matches vLLM. Next: the
  per-step host work (`VLLM-LADDER:62-66`), ranked by `HOSTPROF`. LOSS or INSUFFICIENT: the same, plus a check of the
  machine record; no claim. A failed required gate/check or < 5 valid runs in either required arm: NOT-COMPLETE, no number.
  An unavailable unfused arm only removes the fusion claim; it does not invalidate the system comparison.

## 8. Before provisioning (CPU), then the session
P1 `.venv/bin/python -m pytest -q`, including `tests/test_nvref_report.py` (synthetic pass, 31/32 refusal, missing run,
n != 31). This worktree has no `.venv`; use `~/code/etx`'s or create one, and keep the worktree under `~/code`:
`rocm_build_check.sh:39-41` mounts only `$HOME/code` and refuses an outdir outside it. P2 `bash examples/vllm_llm/local_matrix.sh`:
16 builds (6 qwen3-8b + 6 qwen2.5-1.5b + 2 qwen3-30b-a3b + 2 mixtral-8x7b, `local_matrix.sh:81-87`), every one rc=0. P3 `ETX_CFLAGS='-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH' bash
examples/vllm_llm/rocm_build_check.sh qwen3-30b-a3b default`: a compile gate only (exit 0, LLVM 22 parses the IR).
Measured 2026-10-08: spill 7 / scratch 32 / LDS 29736, and the same numbers without `ETX_CFLAGS`, so the container
does not see these four flags on the resource report. The VM's own build with the same flags reported spill 9 /
scratch 40 (`results_tiles`), so the session compares against that and treats a mismatch as informational. P4 `python3 -m etx.tools.gpu_jobs
examples/sessions/nvref.py --dry-run`. P5 `shasum -a 256` of the golden equals section 2. P6 commit, and confirm
`~/code/etx` holds the commit (push copies from there, `session.sh:17`); balance at least $3.00. Then
`tools/hotaisle_grab.sh 1x`; `tools/session.sh push <ip>`; `tools/session.sh start <ip> examples/sessions/nvref.py`;
`status` every 5 min; `fetch <ip> ~/code/etx/results_nvref_<date>`; delete the VM; confirm "No virtual machines".

## 9. Cost (estimates from `results_tiles/runner.log` and `vllm_ladder/driver.log`)
Setup to the first gate 6.5 min (`runner.log:66`); image pull 5-10 min in parallel (`DATA.md:54`); a vLLM run about 71 s
(`driver.log:1-3`), an ETX run about 23 s (`runner.log` status table). Rounds 5 x (71 + 23 + 23) s = 9.8 min; gates
1.5 min; nice to have 4.5 min. GPU busy about 19 min, runner 25 min, VM life 35 min. At $2.99/h for one MI300X: $1.74
billed per minute, $2.99 under a 1-hour minimum (recorded for 2x: `tools/hotaisle_grab.sh:4`, `docs/PLAN-v3.md:178`;
unknown for 1x). Ceiling 60 min / $3.00. **Hard stop:** delete the VM at 55 min of VM life; nice-to-have jobs go first.

## 10. Risk → mitigation (local change; GPU validation still pending)
| Risk | Mitigation / evidence location |
|---|---|
| Default per-XCD replicas make byte comparison n/a | `build_chk1` sets `ETX_VL_CHECK=1 ETX_VL_XCD=0`; `check_imp1` requires the exact OK line; `build_chk` / `check_imp` retain the refusal demonstration. |
| A new check failure is hidden by legacy n/a | `nvref_report.py` selects `check_imp1` whenever its log or status exists; `test_check_imp1_failure_refuses` covers FAILED, timeout, refusal and skipped status. |
| Unmeasured unfused path kills the main comparison | Only `etxu_r1..5` has `gate_unf` in `after`; `etx_r1..5` and `vllm_r1..5` have independent success dependencies. |
| Skipped neighbours break round interleaving | All earlier jobs remain `after_any` barriers; `test_check_build_and_skipped_neighbour_barrier` checks the runner's actual dependency logic. |
| Missing/incorrect unfused evidence rejects a valid system verdict | `examples/vllm_llm/nvref_report.py:178` handles unavailable evidence and emits `FUSION unavailable (<reason>)`; `test_fusion_missing_still_verdict` and existing incorrect-run cases cover this. |
| Invalid required evidence leaks partial numbers | Validation precedes output; `tests/test_nvref_report.py:215` (`test_missing_required_with_missing_fusion_refuses`), required 32/32 and step-count tests retain refusal checks. |
| Failed nice-to-have job or pairs table loses the primary report | `hostprof`, `etx_noprog`, `vllm_aiter_r1..3`, `report` wait via `after_any`; `report` preserves the optional pairs output without requiring success. |

Reference corrections in this revision: section 4's `model.py:161` described replicas but did not establish the
default (`:63-68,164` do); `host.hip:588` is a per-launch comparison, while the final CHECK line is `:603`.
Section 1's host timing references were stale: `610→625`, `632→647`, `634→649`, `606→621`,
`633→648`, `864→878`, `870→885`, `640→655`, `678→693`; section 4's mode print moved `867→882`.
The report's legacy n/a citation was also stale (`host.hip:500→515`). Historical result files outside this
worktree were not revalidated. This revision does not provision, build on a VM, run a GPU, or push.
