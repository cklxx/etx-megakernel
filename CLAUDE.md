# ETX: how we work

These rules come from what went wrong in 2026-09 (weeks spent tuning our own tiles against the user's direction, building a whole system before measuring the one number that decides it, GPU sessions lost to silent hangs and untested scripts). Follow them in this order.

## 1. The claim decides the experiment

- **The claim ETX has to prove:** on one MI300X, Qwen3-30B-A3B decode at batch 1, the ETX megakernel built from ETX's DSL tiles (GEMV, MoE, top-k, attention prologue) and vLLM's attention, attention-reduce and RMSNorm kernels is faster than the same kernels replayed one by one through a HIP graph: `examples/vllm_llm/pairs.py` returns ESTABLISHED on host wall time (five interleaved pairs in one session on one GPU, every gain positive, median gain above 2% and above twice the spread), with `host --check` OK for every imported launch and 32/32 tokens equal to HF greedy. Status 2026-10-09: the wall fusion rule passed once at 2.79% in one session (`results_nvref_2026-10-09/fusion_pairs.txt:7`) on an MI300X VF (`results_nvref_2026-10-09/record.log:7`), with 0.130 ms gain against an approximately 0.093 ms bar (2% of the unfused time; `results_nvref_2026-10-09/report.log:8,17`). This is not replicated; the import check covers only 5 of 97 imported instances (`results_nvref_2026-10-09/check_imp1.txt:2-6`, `results_nvref_2026-10-09/vlplan_all.txt:3-430`), in a different build from the timed one (`results_nvref_2026-10-09/build_chk1.log:2,30-32`, `results_nvref_2026-10-09/build_all.log:2,27-29`); the replicated check refused (`results_nvref_2026-10-09/check_imp.txt:2`). Megakernel-specific host excess of 0.094 ms (`results_nvref_2026-10-09/report.log:18`) reduces the device gain of 4.87% to 2.79% on the wall (`results_nvref_2026-10-09/fusion_pairs.txt:7-8`). With vLLM's kernels only, fusion loses (`results_2026-10-01/round24/FINDING.md:27-28`, -0.171 and -0.284 ms) and is not claimed. Kernel quality is not ETX's claim; tuning our own tiles to win a comparison still needs the owner's request.
- Every comparison has two baselines: the same kernels launched one by one (isolates fusion), and the strongest real system (vLLM's best configuration, measured the same way). Report both.
- Follow the user's stated direction. If you think it is wrong, say so once with evidence; do not drift away from it silently.

## 2. Before building: ceiling, then the key unknown

1. **Ceiling first.** For every target: the bandwidth floor (bytes / about 4.3 TB/s achievable on MI300X), the baseline's time, and, from a kernel trace, how much of the baseline is gaps between kernels. That gap is the upper bound on what fusion can recover. Set the target from these numbers, never from hope.
2. **Measure the deciding unknown with a microbenchmark** (minutes of GPU time) before building anything that depends on it. Example: whether an ETX synchronisation is cheaper than a kernel boundary decides whether fusing vLLM's kernels can win at all.
3. **Write down the go/no-go threshold before the experiment**, and what happens next in each case.

4. **Check what was already measured.** Before proposing a lever, search `~/code/fleet-mi300x/docs/STATUS.md` and `results/` (the DeepSeek megakernel on the same GPU, sessions 1-9) and this repo's section 15: occupancy, uncached activations, prefetch, split worker groups, last-arriver folds and more were measured there. Missing this cost a plan its main lever on 2026-09-27.

5. **Reuse what is measured.** `docs/DATA.md` lists every measured data set (with checksums), what it answers and what uses it. A GPU run is for a new condition or to validate a prediction, never to regenerate data already there; keep reusable artefacts (fleet reference outputs, goldens) between sessions.

## 3. Correctness gates

- No performance number without correctness: every imported launch byte-identical to the original (`host --check`), the full model 32/32 tokens equal to HF greedy.
- An experiment that did not complete is reported as not complete, not as a number from a shorter run.

## 4. GPU sessions (money)

- **Before provisioning:** everything that can run locally has run: unit tests, local gfx942 compile with the ROCm device libraries (`ETX_DEVLIBS`), shell syntax, a dry run of the script's non-GPU parts (for the vLLM path: `examples/vllm_llm/local_matrix.sh`, all 12 builds), and the device build with the VM's own compiler (`examples/vllm_llm/rocm_build_check.sh <model> <variant>`: ROCm 7.2.4 clang in an x86 container; its VGPR / spill report matches the VM's, the local LLVM 23's does not). Check the balance; state the session budget (hours, dollars) up front.
- **One script per session**, prepared and committed beforehand; it keeps **full, unfiltered logs** under one directory. Never grep errors away. Sessions are job files for `etx/tools/gpu_jobs.py` (every GPU of the VM busy at once, CPU builds in parallel, pinned A/B pairs, deadlines, silence warnings; one runner per VM with `machine` tags when there are two VMs), driven by `tools/session.sh push|start|status|fetch` and provisioned with `tools/hotaisle_grab.sh [1x|2x|any]`. Example: `examples/sessions/ds_gap.py`.
- **Nothing runs silently.** Line-buffered output (`stdbuf -oL`), progress every N steps, a per-step deadline that prints the stuck state, timeouts sized to the expected time plus a margin, not 30 minutes blind. If a process shows no output for about 3 minutes past its expected time, probe it (ps, log, GPU use) instead of waiting.
- Order inside a session: microbenchmarks and correctness checks first, full runs second, nice-to-have captures last.
- Delete the VM the moment the planned work is done (no asking), verify "No virtual machines", record the balance in memory.
- Known traps: `pkill -f`/`pgrep -f` over ssh kills the ssh itself (kill by PID from `ps -eo pid,args | awk`); a re-imaged VM at a reused IP changes host key (`ssh-keygen -R <ip>`); ROCm ships `llvm-link`/`opt` but no `llvm-as`; clang links device libraries before `-mlink-builtin-bitcode` files (use `etx.importer.bundle`); `hipcc ... -x hip` makes later `.o` inputs parse as source (put `-x none` before them); 16 GB weight files take 1-2 minutes to load (not a hang); `ETX_COHERENT_DEVICE_DATA` (the `ETX_VL_UC` builds) is an unchecked promise that every tensor crossing XCDs is uncached: a new cached cross-XCD tensor silently reads stale data (the KV-cache part is enforced by `examples/vllm_llm/check_uc_plan.py`; audit the rest when adding buffers).

## 5. Reporting

- Numbers with their conditions (GPU, batch, context, what was measured: device span or wall time).
- Separate measured, estimated and unknown. Estimates are labelled as estimates.
- Say what failed and what it cost. Short, in the user's language; deliverable documents in English unless asked otherwise.

## 6. Where things are

- `etx/importer/`: kernel import (slice, IR import, adapters, bundle, vLLM manifest). `examples/vllm_llm/`: Qwen with vLLM's kernels; `vm_next.sh` is the prepared next session (microbenchmarks and gate first). `bench/sync/sync_bench.hip`: the synchronisation microbenchmarks. `bench/hw/hw_bench.hip` (lat / lf / launch / evx) and `bench/hw/imp_bench.hip`: every kernel-model parameter measured directly; `examples/vllm_llm/bench_params.py` maps them onto the model (docs/MODEL-vllm-kernels.md section 10).
- Schedule inspection: `python -m etx.tools.timeline <plan.json> <ETX_TRACE_RAW file> --bytes <spec> --grids <regex> --html out.html` (per-XCD occupancy, critical path, memory demand; also runs on a simulated schedule). DeepSeek-V2-Lite (fleet's tiles under ETX): `examples/dsv2lite/` (`bytes.py` per-task bytes, `whatif.py` layout estimates from measured phases). `bench/overlap/overlap_bench.hip`: Infinity Cache read rate and latency under a co-running stream.
- Plans and research: `docs/PLAN-v3.md`, `docs/RESEARCH-sync-hardware.md`, `docs/RESEARCH-moe-import.md`; results in `docs/ETX_Technical_Design.md` section 15.
- Tests: `.venv/bin/python -m pytest -q`. Local toolchain: `/opt/homebrew/opt/llvm/bin` (LLVM 23) with the ROCm 7.2.4 headers and device libraries in `~/code/rocm-headers`.
