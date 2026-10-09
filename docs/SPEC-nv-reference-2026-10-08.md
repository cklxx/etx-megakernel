# SPEC: ETX against vLLM on one MI300X, on one timing basis, beside the published NVIDIA rows

Session specification (2026-10-08, evidence fixes based on `423c040`). The session author commits `examples/sessions/nvref.py`,
`examples/vllm_llm/nvref_report.py` and `tests/test_nvref_report.py` before provisioning. `~/code/etx/results_*` is outside git.

## 0. Why 227 tok/s does not answer the question yet
- 4.405 ms is a device span: `~/code/etx/results_tiles/ab2_all_x.log:8` reads `median_ms=4.405 wall_ms=4.548` (32/32;
  rounds 1 and 3 in `ab1_all_x.log:8`, `ab3_all_x.log:8`; the set is archived in `results_tiles.tgz` and indexed in `docs/DATA.md`). vLLM's figures are
  host time (`docs/VLLM-LADDER-2026-10-01.md:30-31`): ETX read 0.045 ms (0.9%) behind on the device basis and 0.242 ms
  (5.0%) on the wall basis (`:49`), from 0.197 ms of per-step host work the device timer excludes (`:38,41`; 0.143 ms
  in the tile session). The ladder ran on an MI300X VF, ROCm 7.14 (`:3`), the tiles on a Hot Aisle box, and absolute
  ms/token does not transfer between machines (`docs/DATA.md:66`). Both arms therefore run in one session, on one GPU.

## 1. Timing basis (decided): host wall time per decode step, batch 1
It is the only basis vLLM's harness yields (`examples/llm/vllm_bench.py:4-5,74`), and it includes the per-step host
cost a user pays. **vLLM stays unchanged; ETX changes** from the device span (`median_ms`, kernel stamps,
`examples/vllm_llm/host.hip:704`) to host wall time, measured the same way:
- vLLM: `(t(32 new) - t(1 new)) / 31`, wall time around `llm.generate` (`vllm_bench.py:63-67,74`): a 31-step mean.
- ETX: the `std::chrono::steady_clock` window `host.hip:636` -> `:658` around `vl_set_params` + `hst.run()`
  (`run()` = reset, residency check, launch, wait: `etx/runtime/include/etx/host.h:424-429`). With `ETX_VL_PROGRESS=1`
  each step prints `  step <t>: <ms> ms wall` (`host.hip:660`) after the window closes. **The ETX figure is the mean of
  t = 1024..1054 in the last run**: the 31 steps after the first generated token, as in vLLM's difference (prompt 1024,
  steps 1055, `host.hip:222`; the default `--repeat 2` keeps the last run, `host.hip:117,632`).
- Cross-check: RESULT `wall_ms` (median of 32 steps, `host.hip:659,889-890,896-897`) within 0.020 ms of that mean, otherwise
  the report prints `BASIS-MISMATCH` and both values.
- Residual, not measured: vLLM's window also holds sampling, output processing and scheduling; ETX keeps tokens on
  the device until the end (`host.hip:666`), which favours ETX.

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
  The separate `record` job prints `HF_REV` from `huggingface_hub.model_info('Qwen/Qwen3-30B-A3B').sha`.

## 3. vLLM arm: `examples/llm/vllm_bench.py`, unchanged
Image `vllm/vllm-openai-rocm:v0.27.1`, vLLM 0.27.1 (`docs/DATA.md:71-72`). Default flags, AITER off, the strongest
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
  per-XCD refusal (`host.hip:515`).
  Validated without a GPU on 2026-10-08: the `chk1` variant of `examples/vllm_llm/rocm_build_check.sh`
  (`ETX_VL_CHECK=1 ETX_VL_XCD=0`) compiles in the ROCm 22 container, and its plan has `rep=1` on all 434
  instances, i.e. NR=1. The timing plan has `rep=8` on 146 instances, i.e. NR=8, which is why the refusal fired.
  `check_imp1` runs `--check 2` and requires the unchanged exact line
  `CHECK OK: 0 launch(es) differ`; its process must also exit successfully.
  Exports beginning exactly with `dsl_` are native DSL tiles (`model.py:459`, `plan_imports.py:15`),
  not imported kernels. `--check` executes them to preserve downstream inputs, skips their original lookup
  and byte comparison, and prints `CHECK NOT-IMPORTED: N instance(s): <first five names>, ...` separately.
  The count is of instances visited within the requested layers, not distinct exports. DSL correctness belongs
  to `CHECK-DSL OK` in `gate_all`. A non-DSL export with no original still dies; differing imported launches
  still end in `CHECK FAILED: N launch(es) differ`.
  Both check jobs require only their own build, `prep`, and `golden_sha` to succeed; preceding gates/checks are
  completion barriers (`after_any`). Neither check gates timing jobs through `after`.
  The report keeps two rules apart. A **differing launch** is a correctness failure: `CHECK FAILED` in *either* check
  refuses the report and prints no number. An **unavailable check** is not: when `check_imp1` printed no
  `CHECK OK` (build failure, timeout, skip) the report still prints every number, writes
  `IMPORT-CHECK NOT MET (check_imp1: <status>)`, and puts the same text in the verdict line, so a LEAD then makes
  no byte-identity claim. Skipped checks need no log: their status is sufficient to mark NOT MET.
  Only when `check_imp1` has neither a log nor a status entry may `check_imp` supply `OK` or
  `IMPORT-CHECK n/a (host.hip:515)`; a `check_imp` that passed never masks a `check_imp1` that did not.
- Timing success dependencies (`after`): `vllm_r1..5` needs `golden_sha`, `docker_pull`;
  `etx_r1..5` needs `golden_sha`, `build_all`, `gate_all`; `etxu_r1..5` adds `gate_unf` to those ETX dependencies.
- Timing, R = 1..5, GPU 0: odd rounds `vllm_r`, `etx_r`, `etxu_r`; even rounds reversed.
  Each job uses `after_any` for all preceding timing jobs and the initial gates/checks. This includes adjacent
  round barriers and prevents an immediately skipped neighbour from allowing a later round to overtake an
  unfinished earlier round. A failed unfused branch does not skip vLLM or fused ETX timing.
  ETX command (timeout 300, as `tiles.py:104`), `--unfused` only in `etxu_r` (same-tiles HIP graph, `host.hip:123,893`):
  `set -o pipefail; echo "ETX_ENV ETX_UNFUSED_SYNC=${ETX_UNFUSED_SYNC:-} ETX_UNFUSED_NOGRAPH=${ETX_UNFUSED_NOGRAPH:-}";
  ETX_VL_PROGRESS=1 stdbuf -oL $HOME/sess/b_all/run $HOME/llm/qwen3-30b-a3b $HOME/sess/b_all/vl_plan.txt [--unfused]`
  (the ETX_ENV line is what `examples/vllm_llm/pairs.py:6-10` parses).
- Last, nice to have (each waits for all preceding jobs via `after_any`; failures do not stop the tail): `hostprof` (fused run, `ETX_HOST_PROF=1`, prints `HOSTPROF steps …`, `host.h:387`), `etx_noprog`
  (fused run without `ETX_VL_PROGRESS`: print effect), `vllm_aiter_r1..3` (`-e VLLM_ROCM_USE_AITER=1`).
- `hostprof` and `etx_noprog` require only `golden_sha` and `build_all` in `after`; AITER requires only
  `golden_sha` and `docker_pull`. Gates and all preceding timings are completion barriers (`after_any`),
  never success dependencies for these diagnostics. A failed gate therefore does not suppress the tail.
- The single scheduler owns state transitions and atomically replaces `status.json` from a unique sibling
  temporary file at initialization and every start, finish, timeout, or skip, before starting dependent jobs.
  The in-session report sees actual terminal states; concurrent readers never see partially written JSON.
- `report` (cpu, only `after_any` barriers, including the last job): `.venv/bin/python -m examples.vllm_llm.nvref_report $HOME/sess`, then
  `python -m examples.vllm_llm.pairs <etx_r*/etxu_r* logs in run order> --pairs 5 --basis wall` for its table only. Failure of this optional table is logged and does not fail a successful primary report.

## 5. Report: required system comparison, optional fusion comparison
`nvref_report.py` validates everything before printing any performance number. vLLM and fused ETX each require
five valid runs, 32/32 on every run, and fused ETX requires exactly 31 measured steps in the last repeat.
The required fused gate, golden checksum and provenance must also validate. An actual `CHECK FAILED`
in either import check also refuses; an unavailable import check merely marks NOT MET as specified above.
Failure of required evidence prints only
`REFUSED: <reason>. No performance numbers.` and `VERDICT NOT-COMPLETE`.

The unfused arm is optional: a missing/failed `gate_unf`, any missing/failed `etxu_r*`, wrong mode, non-32/32 result,
or invalid step count suppresses the unfused row and numerical FUSION line. Print `FUSION unavailable (<reason>)`
instead; vLLM, fused ETX, DELTA and VERDICT remain available. No partial unfused statistics are printed.
The unfused row and numerical FUSION/HOST-EXCESS lines below appear only with five fully valid unfused runs
and a valid gate. `HOST-EXCESS` uses each round's RESULT `wall_ms - median_ms`, then takes each arm's median;
`excess(fused-unfused)` subtracts those medians. This difference of reported medians is a host-overhead proxy,
not a median of paired per-step overheads, and is separate from the 31-step-mean verdict basis.

Every nice-to-have (`hostprof`, `etx_noprog`, each AITER run) is validated independently. Missing logs,
failed/timeout/skipped jobs, malformed output, or non-32/32 results print `<job> unavailable (<reason>)`.
They never refuse the primary report, even if a failed job printed a RESULT. AITER statistics require all three
valid runs; partial statistics are not printed.
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
DELTA r1..r5 (ETX fused - vLLM, same round) <five deltas> median spread
RESOLUTION: paired delta spread X ms against a 0.050 ms threshold; |median|=... ms is ...x the spread [only when unresolved]
FUSION (unfused - fused) median=... ms gain median=... spread=... rule met|not met
HOST-EXCESS fused median=... unfused median=... excess(fused-unfused)=... ms
IMPORT-CHECK OK|NOT MET (<reason>)|n/a
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
  not: batch > 1, serving latency, other machines (`DATA.md:66`). The ETX GEMVs and MoE are ETX's own DSL tiles
  (GEMV, MoE, top-k, attention prologue), with vLLM's attention, attention-reduce and RMSNorm kernels; a lead against vLLM is an ETX-system result.
  AGENTS.md section 1 claims this mixed-kernel megakernel beats a HIP-graph replay of the same kernels on one MI300X, Qwen3-30B-A3B batch-1 decode: `examples/vllm_llm/pairs.py` must return ESTABLISHED on host wall time (five interleaved pairs in one session on one GPU, every gain positive, median above 2% and twice the spread), with every imported launch `host --check` OK and 32/32 tokens equal to HF greedy. Status 2026-10-09: the wall fusion rule passed once at 2.79% in one session (`results_nvref_2026-10-09/fusion_pairs.txt:7`) on an MI300X VF (`results_nvref_2026-10-09/record.log:7`), with 0.130 ms gain against an approximately 0.093 ms bar (2% of the unfused time; `results_nvref_2026-10-09/report.log:8,17`). This is not replicated; the import check covers only 5 of 97 imported instances (`results_nvref_2026-10-09/check_imp1.txt:2-6`, `results_nvref_2026-10-09/vlplan_all.txt:3-430`), in a different build from the timed one (`results_nvref_2026-10-09/build_chk1.log:2,30-32`, `results_nvref_2026-10-09/build_all.log:2,27-29`); the replicated check refused (`results_nvref_2026-10-09/check_imp.txt:2`). Megakernel-specific host excess of 0.094 ms (`results_nvref_2026-10-09/report.log:18`) reduces the device gain of 4.87% to 2.79% on the wall (`results_nvref_2026-10-09/fusion_pairs.txt:7-8`). vLLM-only fusion loses (-0.171 and -0.284 ms, `results_2026-10-01/round24/FINDING.md:27-28`) and is not claimed. Kernel quality is not ETX's claim; tuning ETX tiles to win still needs the owner's request.

## 7. Go/no-go, fixed before the run
Δ_R = ETX fused 31-step mean - vLLM per-token, same round. Threshold 0.050 ms is a decision rule, not an observed noise bound. The cited graph-enabled vLLM
spreads range from 0.020 to 0.083 ms (0.020 default and 0.052 custom ops, `docs/VLLM-LADDER-2026-10-01.md:15-16`;
0.083, `results_2026-09-27/vlnext/vllm_moe_b1.log:65`); ETX wall spread is 0.022 ms
(`results_tiles/ab1..3_all_x.log:8`). The ladder's non-graph controls have still larger spreads, 0.960 and 1.216 ms (`docs/VLLM-LADDER-2026-10-01.md:17-18`).
- The primary verdict uses only fused ETX versus vLLM; unavailable fusion evidence does not change it.
- **LEAD:** all five Δ_R < 0, median <= -0.050. **LOSS:** all five > 0, median >= +0.050. **TIE:** |median| < 0.050.
  **INSUFFICIENT:** otherwise. If AITER's median is > 0.050 below default, recompute against AITER and print both.
- Print the five paired deltas and their spread (max minus min).
  A paired delta spread above 0.050 ms, or a median inside that spread, prints the `RESOLUTION` line above with
  the ratio of the median to the spread. When the median is no larger than the spread the sign of the paired
  difference is not stable, so the verdict becomes `INSUFFICIENT` whatever the median says; otherwise the
  verdict stands and the ratio tells the reader how far it is from the noise.

For the actions below, F and U are the fused and unfused host-wall 31-step medians, and X is the
HOST-EXCESS excess(fused - unfused) defined in section 5. The restored host-work branch is
`(U - F + X) / U > 2%`: if removing X would clear that bar, investigate per-step host work using section 11;
otherwise return to the device levers (`docs/FLOORS-2026-10-08.md:288-293`). This is a counterfactual screening
rule, not the measured fusion verdict (`examples/vllm_llm/pairs.py:33-34`); unavailable U or X leaves it undecided.
- **LEAD action:** write `docs/NVREF-<date>.md`, index the evidence in `docs/DATA.md`, and update the README comparison;
  retain the separate fusion gate and leave NVIDIA validation open (`fff11e6:docs/SPEC-nv-reference-2026-10-08.md:116-118`).
- **TIE action:** report the null result and apply the host-work branch above, then section 11's diagnostic split
  (`fff11e6:docs/SPEC-nv-reference-2026-10-08.md:119-120`; `docs/VLLM-LADDER-2026-10-01.md:62-66`).
- **LOSS action:** report no lead, check the machine record, and apply the same host-work branch
  (`fff11e6:docs/SPEC-nv-reference-2026-10-08.md:120-121`).
- **INSUFFICIENT action:** make no lead/loss claim; inspect the machine record and paired spread before treating the
  host-work branch as actionable (`fff11e6:docs/SPEC-nv-reference-2026-10-08.md:120-121`; `docs/ROADMAP.md:146-149`).

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
Setup to the first gate 6.5 min (`runner.log:66`); image pull 5-10 min in parallel (`DATA.md:56`); a vLLM run about 71 s
(`driver.log:1-3`), an ETX run about 23 s (`runner.log` status table). Rounds 5 x (71 + 23 + 23) s = 9.8 min; gates
1.5 min; nice to have 4.5 min. GPU busy about 19 min, runner 25 min, VM life 35 min. At $2.99/h for one MI300X: $1.74
billed per minute, $2.99 under a 1-hour minimum (recorded for 2x: `tools/hotaisle_grab.sh:4`, `docs/PLAN-v3.md:178`;
unknown for 1x). Ceiling 60 min / $3.00. **Hard stop:** delete the VM at 55 min of VM life; nice-to-have jobs go first.

## 10. Risk → mitigation (local change; GPU validation still pending)
| Risk | Mitigation / evidence location |
|---|---|
| Default per-XCD replicas make byte comparison n/a | `build_chk1` sets `ETX_VL_CHECK=1 ETX_VL_XCD=0`; `check_imp1` requires the exact OK line; `build_chk` / `check_imp` retain the refusal demonstration. |
| A new check failure is hidden by legacy n/a | A `CHECK FAILED` in either check refuses (`test_legacy_check_failed_refuses_even_when_new_check_passed`); an unavailable `check_imp1` proceeds and marks `import-check NOT MET (check_imp1: <status>)` on the verdict line (`test_check_imp1_incomplete_proceeds_with_not_met`), so no number is ever printed without saying whether the gate was met. |
| Unmeasured unfused path kills the main comparison | Only `etxu_r1..5` has `gate_unf` in `after`; `etx_r1..5` and `vllm_r1..5` have independent success dependencies. |
| Skipped neighbours break round interleaving | All earlier jobs remain `after_any` barriers; `test_check_build_and_skipped_neighbour_barrier` checks the runner's actual dependency logic. |
| Missing/incorrect unfused evidence rejects a valid system verdict | `examples/vllm_llm/nvref_report.py:178` handles unavailable evidence and emits `FUSION unavailable (<reason>)`; `test_fusion_missing_still_verdict` and existing incorrect-run cases cover this. |
| Invalid required evidence leaks partial numbers | Validation precedes output; `tests/test_nvref_report.py:215` (`test_missing_required_with_missing_fusion_refuses`), required 32/32 and step-count tests retain refusal checks. |
| Failed nice-to-have job or pairs table loses the primary report | `hostprof`, `etx_noprog`, `vllm_aiter_r1..3`, `report` wait via `after_any`; `report` preserves the optional pairs output without requiring success; optional parsers catch invalid/incorrect results and print unavailable (`test_optional_failure_never_refuses`). |

Host line references describe `83f80b3`. The import-check change added eleven lines after its setup
relative to `423c040`. The final CHECK summary is `host.hip:614`, and the DSL predicate is
`host.hip:554`; the legacy per-XCD refusal remains at `:515`. The `chk1` compile observation above predates
this fix and does not establish that the modified GPU path runs correctly. Local CPU dispatch tests use stubs;
full GPU launch comparisons, CHECK-DSL and 32/32 model correctness remain session-time validation.
This revision does not provision, build on a VM, run a GPU, or push.

Evidence-fix risks and local checks (2026-10-08):

| Risk | Mitigation / evidence location |
|---|---|
| DSL exports have no original and abort `--check` | Exact `dsl_` prefix follows the planner; tiles execute but do not compare. `tests/test_host_import_check.py` compiles the actual dispatch with CPU stubs, including non-DSL failure cases. Real GPU byte identity remains unvalidated. |
| In-session report sees stale/default status or skipped check has no log | Atomic transition snapshots; `test_live_status_before_report_job`, `test_status_atomic_replace`, `test_skipped_check_without_log`. |
| Gate failure suppresses diagnostic evidence | Tail needs only builds/golden/image success; gates are barriers; `test_tail_needs_builds_not_gates`. |
| Noisy deltas look resolved; positive fusion median looks sufficient | `test_resolution` covers both resolution conditions; `test_fusion_rule` checks gain and spread rules via the shared pairs verdict. |
| Wall fusion margin hides megakernel-specific host cost | `test_host_excess_uses_result_wall` checks 0.197 / 0.036 / 0.161 ms using RESULT values distinct from the step mean; unavailable fusion also makes HOST-EXCESS unavailable. |

## 11. Pre-registered expectations
Written 2026-10-08 before provisioning. `[M]` derives from a logged measurement; `[E]` is an estimate. Each result is
checked against this section; a miss is a finding, not a reason to re-run.
- Gates. `gate_all`: `CHECK-DSL OK`, 32/32, device about 4.406, wall about 4.547 `[M, results_tiles/gate_all.txt:20,23]`.
  The six tile builds were bit-identical (`bitcmp.log:19-24`), so a failure means the binary or machine changed.
  `gate_unf`: 32/32 `[E]`; this build never ran unfused. `build_all`: spill 9 / scratch 40 `[M, buildlog_all.txt:27]`.
- `record`: rocm-smi product name, `/opt/rocm/.info/version`, `uname -r`, `nproc`, an `lspci` line and an `HF_REV` line.
  Expected MI300X, ROCm 7.2.4, bare metal `[M-derived, docs/DATA.md:62]` for the planned Hot Aisle box;
  VF / ROCm 7.14 is the historical alternate, not that expectation (`docs/DATA.md:63`). No sourced expectation
  for `uname -r`, `nproc` or the exact HF SHA; record them, do not invent values.
  The hub lookup falls back to `HF_REV unknown (...)`, so a hub failure cannot refuse the report.
- `IMAGE`: expected `vllm/vllm-openai-rocm@sha256:<digest>` `[E, examples/sessions/nvref.py:47-49]` from the
  pulled v0.27.1 image. No pinned digest is sourced; record the actual digest, not a guessed hash.
- `build_chk` / `build_chk1`: both print compiled resources and the informational 9 / 40 comparison
  (`examples/sessions/nvref.py:53-64`). Timing-build prior: VGPR 128, spill 9, scratch 40 B/lane, LDS 29736 B/block
  `[M, results_tiles/buildlog_all.txt:27]`. No measured resource expectation for either check variant is sourced;
  linking originals and removing XCD replicas need not preserve the timing build's resources. Section 8's
  container spill 7 / scratch 32 is a different build observation, not a VM check-build prediction.
- `check_imp`: expected refusal `--check needs a plan without per-XCD replicas (build without ETX_VL_XCD)`
  `[E, examples/vllm_llm/host.hip:515; examples/sessions/nvref.py:55-56,79-84]`; this is the NR>1 demonstration,
  not a successful byte comparison. The job accepts that refusal.
- `check_imp1`. `--check` is parsed at `host.hip:140` and the instance loop breaks at `:551`. Layers 0-1 hold 19
  instances `[M, results_tiles/vlplan_all.txt:3-21]`: 14 `dsl_*`, 5 imported (`rms_norm_2d`, `attn_mfma16_g8` x2,
  `attn_reduce` x2). `ETX_VL_XCD` changes only rep, stride and copyoff (`model.py:371-388`), so the NR=1 plan holds the
  same list. Expected: five `identical` lines (`host.hip:598`), then
  `CHECK NOT-IMPORTED: 14 instance(s): dsl_gemv, dsl_attnprep, dsl_gemv, dsl_gemv_fold, dsl_topk, ...` (`:613`), then
  `CHECK OK: 0 launch(es) differ` (`:614`), exit 0. The same three kernels were byte-identical on 2026-09-28
  (`results_2026-09-28/moe_check.log:54`). `CHECK FAILED` refuses. A count other than 14 means the plan differs from this
  section: the report prints NOT MET and a LEAD carries no byte-identity claim.
  Import-overhead control: per-launch `original / imported us` and per-kernel `(+%)` are printed at
  `examples/vllm_llm/host.hip:597-598,609-612`. Priors `[M]` for the surviving imported kernels are
  rms_norm_2d 766.2 / 8.9 us (-99%), attn_mfma16_g8 summed 2515.9 / 23.8 us (-99%), and attn_reduce
  summed 16.5 / 18.6 us (+13%) (`results_2026-09-28/moe_check.log:39-40,46`). Per-launch attention was
  2507.4 / 12.3 and 8.5 / 11.4 us; reduce 9.5 / 9.1 and 7.0 / 9.5 us (`:10-11,26-27`). These one-launch
  event timings include very large first-original costs and a different build: no defensible steady-state
  expected overhead range or pass threshold is sourced for chk1. They are diagnostic evidence toward the
  roadmap's import-overhead control (`docs/ROADMAP.md:139`), not its native-operators-in-a-Graph measurement.
- vLLM default: 4.83-4.91 per run `[M-derived: VLLM-LADDER:15, results_2026-09-28b/vllm1_b1_default.log:67,
  results_2026-09-27/vlnext/vllm_moe_b1.log:65; this image was never measured on this box]`.
- ETX fused wall: 4.53-4.56 `[M, ab1..3_all_x.log:8]`; device 4.39-4.42 `[M]`. Unfused: 4.61-4.65 `[E: 4.405 + device
  margin 0.172-0.200 (BUDGET:355, VLLM-LADDER:48) + unfused host 0.036-0.042 (VLLM-LADDER:39)]`.
- DELTA: median about -0.31, each round -0.38 to -0.27 `[E]`. Expected verdict LEAD. A median above -0.25 is a new fact
  about this box. The verdict is against vLLM default only: `enable_qk_norm_rope_fusion` ran 0.167 ms faster on the dev
  image (`vllm1_b1_qknr.log:68`) and this session does not run it.
- RESOLUTION: expected spread 0.02-0.05 `[E]`. Observed graph-enabled vLLM spreads range 0.020-0.083 ms
  (0.020 / 0.052, `docs/VLLM-LADDER-2026-10-01.md:15-16`; 0.083,
  `results_2026-09-27/vlnext/vllm_moe_b1.log:65`); the 0.050 ms decision rule is not an observed noise bound, so the line may print. The verdict becomes unusable only when spread >= |median|. Five rounds
  cannot bound the tails: one outlier sets the range and the sign test reaches p = 1/32 one-sided at best.
- FUSION: gain 1.4-2.1% `[E]`; round 18 measured 1.7% on the wall basis (`round18/FINDING.md:55`). Expected `not met`. **Failed prediction:** the wall rule passed (`results_nvref_2026-10-09/fusion_pairs.txt:7`).
  `met` requires unfused - fused > 0.093 ms at fused 4.548, all five gains positive and gain spread < median / 2
  (`pairs.py:33-34`). `not met` with a device margin near 0.2 ms leaves the claim open: fusion wins on the device and the
  megakernel's own host work eats the win on the wall. A median gain <= 0 refutes it for this build.
- `fusion_pairs.txt` device gains: about 3.8-4.3% `[E]`, from a 0.172-0.200 ms device margin
  (`docs/BUDGET-2026-10-01.md:355-356`, `docs/VLLM-LADDER-2026-10-01.md:48`) and fused device 4.405 ms
  (`results_tiles/ab2_all_x.log:8`), using margin / (4.405 + margin). This transfers an older margin to a new
  build, not a measurement. The table prints both bases (`examples/vllm_llm/pairs.py:152-155`;
  `examples/sessions/nvref.py:139-140`); it does not decide the primary system verdict.
- `BASIS-MISMATCH`: expected absent `[E]`, i.e. RESULT wall median within 0.020 ms of the 31-step mean
  (section 1's rule; `examples/vllm_llm/nvref_report.py:240-243`). No measured distribution of that difference
  is sourced. If present, retain both values; do not switch the verdict basis.
- HOST-EXCESS: fused 0.140 `[M, ab1..3 line 8: 0.140 / 0.143 / 0.139]`; unfused 0.036-0.042 `[E]`; excess about 0.10 `[E]`.
  Round 18 cut the fused figure from 0.276 to 0.135 (`round18/FINDING.md:34`); above 0.16 means that fix regressed here.
- HOSTPROF (stderr; `hostprof.log` only, the report does not parse it): steps 2110; reset about 64 us, residency 0.6,
  launch 18 `[M-derived, round18/FINDING.md:10-12]`; wait about device + 71 us `[E, :13]`.
- Report `hostprof wall_ms=... (31-step mean)`: about 4.53-4.56 ms `[E]`, transferring the unprofiled
  RESULT wall prior (`results_tiles/ab1..3_all_x.log:8`) to the diagnostic; no measured profiled 31-step mean
  for this build is sourced. The report prints it at `examples/vllm_llm/nvref_report.py:287-289`.
- AITER: median about 4.92 `[M, results_2026-09-27/extra/etx/examples/llm/results/vllm_aiter1_qwen3-30b-a3b.json: 4.916 against 4.836 with aiter off (vllm_aiter0_...json)]`: AITER measured *slower* on this model, so the 0.050 ms recompute rule does not fire. `etx_noprog`: RESULT wall within
  0.022 ms of the fused RESULT wall `[E]`; a larger gap means the progress print leaks into the timing basis.
- Next step, decided by this run. X = excess(fused - unfused); E_f = fused HOST-EXCESS; P + O = HOSTPROF reset +
  residency + launch + (wait - device). X < 0.050: no host work, return to the device levers (`FLOORS:288-293`).
  X >= 0.050 and P + O >= E_f - 0.050: async enqueue, done when X < 0.050. X >= 0.050 and P + O < E_f - 0.050: resident
  kernel (`ETX_Technical_Design.md:724-726`). The 0.050 is section 7's threshold; the repository fixes no other for this split.
  **This split is indicative, not a strict decision.** HOSTPROF averages all 2 x 1055 steps, including prompt
  steps and the slow first step (46.837 ms wall in `results_tiles/gate_all.txt:2`), because every profiled run
  accumulates into one counter (`etx/runtime/include/etx/host.h:431-441`). RESULT `median_ms` covers only the
  last repeat's 32 decode steps (`examples/vllm_llm/host.hip:703-705,889-890`). Thus O = wait - device mixes
  populations and mean with median. Its bias has unknown size and sign relative to 0.050 ms; do not silently
  reinterpret it as measured overshoot. The strict fix is for the host to print a per-step device mean over
  exactly the same steps as HOSTPROF, then subtract matching means. That requires a CPU code change, outside
  this documentation-only task; the existing rule and threshold remain unchanged.
- Waste checks: a failure of `record`, `golden_sha`, `gate_all`, `docker_pull` or `vllm_r1` makes the report
  NOT-COMPLETE and shows by about minute 12. Let `check_imp1`, `gate_unf` and `hostprof` finish, then delete the VM.

**Post-session observation, 2026-10-09 (pre-registered rule unchanged):** this machine's five-round result is
INSUFFICIENT (`results_nvref_2026-10-09/report.log:20`). Default vLLM returned 4.849, 3.840, 4.825,
4.809 and 4.800 ms, arm spread 1.009 ms (`results_nvref_2026-10-09/report.log:6`); the resulting paired
DELTA spread is 1.013 ms against the 0.050 ms threshold, exceeding the median magnitude 0.283 ms
(`results_nvref_2026-10-09/report.log:15-16`). The follow-up returned 4.811794, 4.829696, 4.812703,
5.375439, 4.839390 and 4.801095 ms (`results_nvref_2026-10-09/vs_summary.log:19-24`). Thus nine of eleven
same-configuration rounds cluster, with opposite outliers; the five-round rule did not resolve this sample.
A future session must either identify/control the trigger in the separately recorded full/single-token windows,
or pre-register more paired rounds with a rule that can tolerate exceptions and quantify uncertainty.
Merely adding rounds cannot outvote an exception under the unchanged unanimity/range rule. Conditional sample-count
calculations and evidence limits are in `docs/NVREF-2026-10-09.md` §4; none changes this session's verdict or the predictions above.


## 12. Pre-registration amendment: nvref2 (CPU preparation, 2026-10-09)

**Question.** Does ETX fused beat vLLM's best measured configuration on host wall time at batch 1 on one
MI300X, for Qwen3-30B-A3B bf16 and the same 1024-token prompt and 32 greedy tokens? Section 7's paired
rule is unchanged. This amendment applies only to the next run, `examples/sessions/nvref2.py`; section 11
and `examples/sessions/nvref.py` remain the record and reproducible protocol of the previous run.

**Measurement change and reason.** The primary default vLLM jobs use `--repeats 5` instead of `--repeats 1`
(`examples/sessions/nvref.py:107`; option: `examples/llm/vllm_bench.py:36`). Each round reports the median
of five `(gen(32) - gen(1)) / 31` samples (`examples/llm/vllm_bench.py:71-75`). The engine is constructed
once per invocation, outside the repeat loop, and warm-up remains once per invocation
(`examples/llm/vllm_bench.py:58-59,69-75`): engine start still happens once per round, not once per sample.
All other job commands, dependencies, lanes, deadlines, ordering and gates stay byte-identical, including
fused/unfused ETX, hostprof, noprog and the AITER tail (which retains `--repeats 1`).

The previous verdict was INSUFFICIENT (`results_nvref_2026-10-09/report.log:20`). Two of eleven default
rounds had opposite window excursions (`docs/NVREF-2026-10-09.md:100-130`): `one_ms` reached 48.57 ms
(`results_nvref_2026-10-09/vllm_r2.json:19`) against the usual 17.18-18.12 ms
(`results_nvref_2026-10-09/vs_r5.json:19`, `results_nvref_2026-10-09/vllm_r4.json:19`), while `full_ms`
reached 184.75 ms (`results_nvref_2026-10-09/vs_r4.json:16`) against the usual 166.6-167.7 ms
(`results_nvref_2026-10-09/vllm_r5.json:16`, `results_nvref_2026-10-09/vllm_r1.json:16`). With one sample,
either excursion determines the entire round. Independent audit: `/Users/ckl/.hermes/cache/scratch/o55-completion.json:1`,
`result`, sections 3 and 4; the raw windows establish the estimator's sensitivity, not the cause of the stalls.

**Prediction [E].** The vLLM per-round spread (max minus min across the five default round medians) should
fall from the nine non-outlier rounds' 0.049 ms (derived from `results_nvref_2026-10-09/vllm_r1.json:11`
and `results_nvref_2026-10-09/vllm_r5.json:11`; the complete set is cited at
`docs/NVREF-2026-10-09.md:100-116`) to **at most 0.020 ms [E]**. A median of five discards one extreme
stalled-call sample; it does not guarantee this spread if drift or repeated/correlated stalls persist.
The earlier 0.049 ms is the non-outlier range, not the complete previous arm's spread. Retain every raw
`full_ms`, `one_ms` and per-token sample, including outliers, and report whether this prediction held.

**MISSED — outcome, 2026-10-09.** The §12 prediction missed: **0.032 ms** against **≤0.020 ms**.
**4.920, 4.921, 4.911, 4.899, 4.889 ms** are the five reported default medians
(`results_nvref2_2026-10-09/session-7de5pood/vllm_r1.log` through `vllm_r5.log`).
**LEAD resolved this session:** paired-delta spread **0.036 ms**, inside §7's **0.050 ms** criterion
(`results_nvref2_2026-10-09/report.log`, re-rendered from the runner's own directory).
**NOT-COMPLETE** in the runner's report read the aborted attempt's status; the logdir fix is in review.
**0.032632 ms** is the full-precision JSON spread; the report uses rounded log medians, and both miss.
**ROCm 7.14.0** replaces the earlier **7.2.4** environment; **10 spills / 44 bytes scratch** replace **9 / 40**
(`results_nvref2_2026-10-09/session-7de5pood/record.log`, `build_all.log`; earlier `results_nvref_2026-10-09/build_all.log`).
**Clang 23** is owner-reported; this is not a rebuild of the same binary.
**One LEAD on a VF**, with import coverage unchanged, is the scope; see [reconciliation](NVREF2-2026-10-09.md).

**Go/no-go, unchanged.** Section 7's primary system rule, verbatim:

- The primary verdict uses only fused ETX versus vLLM; unavailable fusion evidence does not change it.
- **LEAD:** all five Δ_R < 0, median <= -0.050. **LOSS:** all five > 0, median >= +0.050. **TIE:** |median| < 0.050.
  **INSUFFICIENT:** otherwise. If AITER's median is > 0.050 below default, recompute against AITER and print both.
- Print the five paired deltas and their spread (max minus min).
  A paired delta spread above 0.050 ms, or a median inside that spread, prints the `RESOLUTION` line above with
  the ratio of the median to the spread. When the median is no larger than the spread the sign of the paired
  difference is not stable, so the verdict becomes `INSUFFICIENT` whatever the median says; otherwise the
  verdict stands and the ratio tells the reader how far it is from the noise.


The separate fusion requirement remains: "five interleaved pairs in one session on one GPU, every gain
positive, median gain above 2% and above twice the spread" (`CLAUDE.md:7`; constants:
`examples/vllm_llm/pairs.py:33-34`). This is the fused-versus-unfused rule, not section 7's vLLM LEAD rule;
no criterion or report implementation is changed by this amendment. Correctness gates remain required,
and the import-check limitation below remains explicit even if the timing rules pass.

Outcome branches (section 7, unchanged):

- **LEAD:** write `docs/NVREF-<date>.md`, index the evidence in `docs/DATA.md`, update the README comparison;
  retain the separate fusion gate and leave NVIDIA validation open. Limit the claim to tested configurations.
- **TIE:** report the null result; apply section 7's host-work branch, then section 11's diagnostic split.
- **LOSS:** report no lead, check the machine record and apply the same host-work branch.
- **INSUFFICIENT:** make no lead/loss claim; inspect the machine record and paired spread before treating
  the host-work branch as actionable. Record a failed spread prediction if applicable; do not relax the rule.

**Budget [E].** Five primary vLLM rounds at about 125 s of engine time each (about two minutes per round),
plus setup, ETX arms, gates, the unchanged diagnostic tail and fetch: about **30 GPU minutes / 0.5 hours**,
about **$1.50 at $2.99 per hour** (planning rate; estimate, not a new billing measurement). These are the
requested planning estimates, also proposed in `/Users/ckl/.hermes/cache/scratch/o55-completion.json:1`,
`result`, section 3. They are not a guaranteed runtime for the full unchanged job list. The operator's
**55-minute VM hard stop** remains binding (section 9); drop nice-to-have jobs first, fetch full unfiltered
logs into a fresh `results_nvref2_<date>` directory, then delete the VM by the hard stop. Do not overwrite
the preceding session's runner log or status. Before any later provisioning, complete section 8's remaining
preflight and balance checks and ensure the committed session is in the checkout copied by `session.sh`.
This preparation authorizes no GPU or VM use.

If cut short, preserve completed correctness checks, individual full/single-token samples and completed
paired host-wall observations, with their actual count and missing jobs. They are partial diagnostic
evidence only: fewer than five complete required pairs is NOT-COMPLETE, with no LEAD/TIE/LOSS verdict
and no substitution of a shorter run. If all required pairs and gates complete but only the optional tail
is cut, report the primary rule, explicitly mark missing fusion/AITER/hostprof/noprog evidence unavailable,
and limit any best-measured-configuration claim accordingly.

**Out of scope.** Import-check coverage remains 5 of 97 imported instances and in a different build from
the timed one (check evidence: `results_nvref_2026-10-09/check_imp1.txt:2-6,14-15`; full timed-plan inventory:
`results_nvref_2026-10-09/vlplan_all.txt:3-430`; audit count and build distinction:
`/Users/ckl/.hermes/cache/scratch/o55-completion.json:1`, `result`, sections 1 and 5). This session does not
establish byte identity for every timed imported launch. The question of vLLM's truly best configuration
also remains open: this is the best measured default/AITER comparison, not a configuration search
(`docs/NVREF-2026-10-09.md:82-84`; untested configuration: section 11). Neither limitation is fixed by
more samples, and a resolved timing verdict must retain both qualifications.


**Optional model diagnostics, pre-registered [E].** Append `trace1` (GPU 0), `floors_trace` (CPU),
then `evx` (GPU 0), each with a 300 s deadline and a successful-predecessor dependency starting at
`report`. They cannot affect any preceding measurement or the report; the rules above stay unchanged.
Sources below are `/Users/ckl/.hermes/cache/scratch/o55-model.json:1`, `result`, sections A and C
("Out of scope"). These are predictions for the same MI300X, batch 1, context 1024 build, not new results.

- **Trace:** one `--repeat 1` run saves `$HOME/sess/tr_nvref2.bin`; `floors_trace` uses the session's
  `$HOME/sess/b_all/plan.json` and saves and prints `$HOME/sess/floors_nvref2.txt`. Expect narrow-phase
  bodies of about **3.85-7.95 us [E]**, rather than the old width-scaled sub-microsecond costs
  (analysis A, "The three largest errors", item 2, from `results_tiles/tr_all.log:7-442`). The table
  separates mean/critical body, start delay and phase delta, testing the missing narrow-body term
  and the delay/tail contribution to the approximately 335 us residual. Cost: under one GPU minute,
  **about $0.05 at $2.99/hour [E]** (analysis A item 2 and C, "Out of scope"). The verbatim analysis
  command traces fused execution: `host.hip:888` suppresses phase reporting with `--unfused`.
  If these bodies do not reproduce, the profile's transferred narrow-body costs are not validated
  for this build; inspect the per-phase discrepancy before attributing the residual to event waits.
- **Event protocol:** compile `bench/hw/hw_bench.hip` using the VM's `hipcc` and run `evx`, using the
  form in `examples/sessions/moe_fix.py:66-74`. Expect the complete-event wait (acquire + barrier)
  to be about **0.52 us [E]**, a per-task cost missing from the simulator (analysis A, paragraph
  "What remains after all three", citing `docs/MODEL-vllm-kernels.md:330`; command in C,
  "Out of scope"). Cost: about five VM minutes, **about $0.25 at $2.99/hour [E]**. This tests the
  event-protocol explanation; it does not establish that multiplying one wait explains all 335 us,
  or independently isolate COH's effect. If it does not reproduce, leave that explanation unvalidated
  and retain the residual rather than fitting a replacement constant.

Both are optional under the unchanged 55-minute hard stop. Preserve full logs; a timeout, missing trace
or incomplete benchmark is NOT-COMPLETE for that diagnostic, not evidence for either expected value.
