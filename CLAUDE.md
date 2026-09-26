# ETX: how we work

These rules come from what went wrong in 2026-09 (weeks spent tuning our own tiles against the user's direction, building a whole system before measuring the one number that decides it, GPU sessions lost to silent hangs and untested scripts). Follow them in this order.

## 1. The claim decides the experiment

- The claim ETX has to prove: **the kernels production actually runs (vLLM's, or the best available), fused into one ETX megakernel, are faster than the same kernels launched one by one.** Kernel quality is not ETX's claim; do not tune our own tiles to win a comparison unless the user asks for it.
- Every comparison has two baselines: the same kernels launched one by one (isolates fusion), and the strongest real system (vLLM's best configuration, measured the same way). Report both.
- Follow the user's stated direction. If you think it is wrong, say so once with evidence; do not drift away from it silently.

## 2. Before building: ceiling, then the key unknown

1. **Ceiling first.** For every target: the bandwidth floor (bytes / about 4.3 TB/s achievable on MI300X), the baseline's time, and, from a kernel trace, how much of the baseline is gaps between kernels. That gap is the upper bound on what fusion can recover. Set the target from these numbers, never from hope.
2. **Measure the deciding unknown with a microbenchmark** (minutes of GPU time) before building anything that depends on it. Example: whether an ETX synchronisation is cheaper than a kernel boundary decides whether fusing vLLM's kernels can win at all.
3. **Write down the go/no-go threshold before the experiment**, and what happens next in each case.

## 3. Correctness gates

- No performance number without correctness: every imported launch byte-identical to the original (`host --check`), the full model 32/32 tokens equal to HF greedy.
- An experiment that did not complete is reported as not complete, not as a number from a shorter run.

## 4. GPU sessions (money)

- **Before provisioning:** everything that can run locally has run: unit tests, local gfx942 compile with the ROCm device libraries (`ETX_DEVLIBS`), shell syntax, a dry run of the script's non-GPU parts. Check the balance; state the session budget (hours, dollars) up front.
- **One script per session**, prepared and committed beforehand; it keeps **full, unfiltered logs** under one directory. Never grep errors away.
- **Nothing runs silently.** Line-buffered output (`stdbuf -oL`), progress every N steps, a per-step deadline that prints the stuck state, timeouts sized to the expected time plus a margin, not 30 minutes blind. If a process shows no output for about 3 minutes past its expected time, probe it (ps, log, GPU use) instead of waiting.
- Order inside a session: microbenchmarks and correctness checks first, full runs second, nice-to-have captures last.
- Delete the VM the moment the planned work is done (no asking), verify "No virtual machines", record the balance in memory.
- Known traps: `pkill -f`/`pgrep -f` over ssh kills the ssh itself (kill by PID from `ps -eo pid,args | awk`); a re-imaged VM at a reused IP changes host key (`ssh-keygen -R <ip>`); ROCm ships `llvm-link`/`opt` but no `llvm-as`; clang links device libraries before `-mlink-builtin-bitcode` files (use `etx.importer.bundle`); `hipcc ... -x hip` makes later `.o` inputs parse as source (put `-x none` before them); 16 GB weight files take 1-2 minutes to load (not a hang).

## 5. Reporting

- Numbers with their conditions (GPU, batch, context, what was measured: device span or wall time).
- Separate measured, estimated and unknown. Estimates are labelled as estimates.
- Say what failed and what it cost. Short, in the user's language; deliverable documents in English unless asked otherwise.

## 6. Where things are

- `etx/importer/`: kernel import (slice, IR import, adapters, bundle, vLLM manifest). `examples/vllm_llm/`: Qwen with vLLM's kernels; `vm_next.sh` is the prepared next session (microbenchmarks and gate first). `bench/sync/sync_bench.hip`: the synchronisation microbenchmarks.
- Plans and research: `docs/PLAN-v3.md`, `docs/RESEARCH-sync-hardware.md`, `docs/RESEARCH-moe-import.md`; results in `docs/ETX_Technical_Design.md` section 15.
- Tests: `.venv/bin/python -m pytest -q`. Local toolchain: `/opt/homebrew/opt/llvm/bin` (LLVM 23) with the ROCm 7.2.4 headers and device libraries in `~/code/rocm-headers`.
