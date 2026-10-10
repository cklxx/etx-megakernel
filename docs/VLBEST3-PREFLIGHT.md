# vlbest3: tuned Triton MoE on S3

This replaces the invalid AITER MoE + tuned-table design. M0 is an exact alias
of existing S3 (no `VLLM_ROCM_USE_AITER` switch). M1 adds only
`VLLM_TUNED_CONFIG_FOLDER=$HOME/moecfg/tuned` to S3. The adapter uses the pinned
image's `benchmarks/kernels/benchmark_moe.py`; it does not vendor or install vLLM.
No GPU, network, image pull, or provisioning is used for local acceptance.

## G0 before any GPU work

`python3 -m examples.sessions.vlbest3 --g0-command` prints the CPU-only command.
The session's `g0` job runs this inside the pinned image, with `--pull=never`,
`--network=none`, no device mounts, and a 60-second timeout. All GPU jobs depend
on successful G0, directly or transitively. Its full output is retained in the
runner's actual `$ETX_SESS_LOGDIR/g0.log`, including all four groups on failure:

- `torch.__version__`: require >=2.11. The source image recipe uses ROCm PyTorch
  release/2.12; the actual image value must be probed on the VM.
- `vllm.utils.torch_utils._USE_LAYERNAME`: require exactly true.
- Presence and resolved path of `benchmarks/kernels/benchmark_moe.py`.
- Resolved `VLLM_TUNED_CONFIG_FOLDER` and
  `E=128,N=768,device_name=AMD_Instinct_MI300X.json`.

G0 also checks loader/CLI contracts and whether the adapter can remove the
source's Ray dependencies. It executes only the pure filename function, with
the specified MI300X name, to avoid initializing a GPU. `record.log` supplies
actual hardware provenance. No image version disagreement is silently repaired.
The optional source-baseline test takes `ETX_VLLM_SOURCE_ROOT`; absent that
explicit input it skips and never assumes a checkout under the user's home.

## Adapter and search space

The command inside the GPU container is:

```sh
timeout -k 5 1170 python3 -m examples.vllm_llm.vlbest3_tuner \
  --model "$HOME/llm/qwen3-30b-a3b/hf" --tp-size 1 --batch-size 1 \
  --save-dir "$HOME/moecfg/tuned"
```

`--tuner-command --directory <logdir>` prints the entire Docker command,
including its deterministic session-specific container name and pinned digest.
The new adapter parses the upstream source, removes Ray imports and decoration,
sets the worker's local device index to zero, omits the Ray-driven main/CLI,
and invokes the original worker's `tune` method directly. ROCm dtype is bf16;
the original `is_fp16` branch already includes bf16 and is unchanged. It checks
(E, topk, N, H) = (128, 8, 768, 2048). The upstream worker and writer receive
1536 for the doubled intermediate projection, producing filename N=768.
The upstream writer is used with no dtype segment, verified before writing.

**The requested approximately 200 points cannot be obtained by pinning M alone.**
The source has 40,000 raw Cartesian points. Pinning `BLOCK_SIZE_M=16`, preserving
all other dimensions, leaves **8,000**. The unchanged upstream ROCm pruning for
batch 1, top-k 8, N=768, H=2048 leaves **704** candidates. These counts were
computed locally from the source; G0 source contracts and the VM tuner logs
provide image-side evidence. Additional reductions would change the requested
other dimensions, so none are introduced. The time cap may therefore stop
this search before it writes a table; such a run is NOT-COMPLETE.

The GPU job has `timeout=1200`; the daemon-owned container has a stricter
1170-second timeout plus five-second kill grace. The wrapper prints progress
every 60 seconds even if compilation stalls. A 2700-second hard deadline is
also checked by the wrapper, but the 1200-second runner cap normally stops it
first. The CPU gate runs after **any** tuning outcome, kills any remaining
adapter by PID (`ps -eo pid,args | awk` inside the named container), and removes
the container. Thus the 45-minute ceiling is superseded by an earlier cap,
not permission to retry or extend tuning. Cleanup has bounded subprocess waits.

A stale target/receipt is deleted before tuning. A successful tuner must write
a nonempty `"1"` row before a SHA-256 receipt is issued. CPU job `moe_gate`
prints resolved folder, filename, and key presence in `moe_gate.log`, then GO
or NOT-APPLIED. Missing table/key/receipt or an unsuccessful tuner is a red
NOT-COMPLETE stop, not an optional skipped lever.

## Screening, red stops, and finals

There are exactly eight screening runs: S0-open, S3, M1, S0-close in round one;
S0-open, M1, S3, S0-close in round two. Each has a 390-second runner cap.
Every vLLM run is checked immediately for five 32/32 samples; every ETX final
is checked for 32/32 and failed-repeat markers. All subsequent timing jobs
require successful completion of the previous job.

For M1 and selected S9 finals, the proof must contain the exact absolute path
from the echoed folder followed by the exact filename and ` for MoE layer.`.
No AITER MoE line may appear. Missing proof is NOT-APPLIED and stops the chain.
The existing default-config rejection also remains. S0 bracket drift above
0.050 ms is a NOT-COMPLETE stop.

GO requires both M1−S3 deltas <0 and their median <=−0.050 ms. On GO the selected
configuration is named S9 and becomes R. Otherwise S3 remains R and S9 is recorded as
NO-WINNER, NOT-APPLIED, tested-and-not-won. Selection is recomputed from logs
when reporting. S9 Triton tuning does not remove AITER MoE tuning from the
untested list: they are separate levers.

All five final rounds run afresh: S0, selected F, ETX fused, and the same-kernel
HIP graph, alternating order each round. The report also gives the S0 delta;
R stays the preselected S9 or S3 for this design. Imported-launch certificates transfer
only after the unchanged build-identity check. Historical captures without the
`vlbest3` design marker retain the original reporting path.

The session retains a 140-minute runner cap and 150-minute VM life cap (at most
$4.98 at $1.99/hour). The operator must delete the VM immediately when done or
at its deadline. This change does not rent, start, or delete hardware.

## Local acceptance (2026-10-10)

- `.venv/bin/python -m pytest -q`: **1070 passed, 16 skipped, 0 failed**,
  194.96 seconds. The optional local-source test skips without an injected path.
- With `ETX_VLLM_SOURCE_ROOT` explicitly supplied, the preserved G0 source test
  passed separately (1 passed, 68 deselected).
- `python -m examples.sessions.vlbest3 --dry-run`: passed; shell syntax for all
  job commands and failure propagation are also covered by the suite.
- Replay of `results_vlbest2_2026-10-10`: **−0.278 ms**, **0.015 ms** spread,
  **LEAD**, **4.015%** step-mean fusion and **3.98%** RESULT-wall fusion,
  unchanged. Both fusion rules remain ESTABLISHED.
- No network, image execution, hardware rental, tuning or performance run.
