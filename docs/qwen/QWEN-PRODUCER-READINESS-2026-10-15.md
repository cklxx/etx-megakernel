# Producer readiness fixes

All three confirmed readiness defects are fixed. This is a local preparation result.
No GPU was used. No claim about model correctness or speed is made.

## Changes

The runbook now uses `python3 -m tools.producer_start`. It runs the real fresh-archive
push, attaches an existing independent venv, probes Python, NumPy and PyYAML at the
staged root, and then starts the runner. Push and attachment share a 300-second
limit. The runner cap also fits the original allocation deadline, with a startup
margin. Failure blocks start. The persistent lease still prevents a budget reset.
A retry with a different cap is refused. The user must fetch and stop in that case.
There is no package install, image pull, download or provisioning step.

The resolved configured HF path must equal the resolved `model/hf` path before
preflight probes or compute. S3 uses that path and reads both its prompt and expected
tokens from the hashed golden file. Its raw log and JSON keep the actual launch path
and input digest. The offline report checks both records against the session input
identity. Equal tokens alone do not establish input identity.

The trace parser uses column 6 for ready. Column 5 is taken. Column 7 is body done,
before phase-three release and event arrival. The report names body completion and
the common producer-start-to-consumer-ready window. It leaves the phase-three
publication cost unknown. No new stamp was added. The untraced KEEP basis and all
numerical, import and token gates are unchanged.

## Evidence

Evidence is in
[`readiness-fixes/`](../../examples/vllm_llm/results/2026-10-15-producer-prep/readiness-fixes).
The source revision tested by the full suite is `5b8afb3`.

- `failing-before.log`: three failures against the original code.
- `focused-final.log`: 118 passed, one skipped in 101.21 seconds.
- `full-suite.log`: 2,061 passed, 18 skipped in 523.69 seconds, exit 0.
- `commands.json`: exact commands, return codes and elapsed times.
- `historical-evidence-audit.json`: all 183 original artifact hashes matched before receipt refresh.
- `manifest.json`: 328 current source hashes and the scope of reused compile evidence.

The tests execute a real git archive and the real push/stage/start shell path.
SSH, rsync, product probes and compute are labelled local substitutes. There is no
network provisioning. A separate archive test creates an independent venv with real
NumPy and PyYAML. Before attachment, the real command exits 127. After attachment,
the interpreter probe succeeds. A failed actual preflight skips compute and writes
a NOT-COMPLETE report through the real final job.

After receipt refresh, `receipt-probe.log` verifies a 3,102,720-byte committed
source-only archive. The real preflight accepts the current source receipt and
interpreter, then rejects the synthetic box before compute. The real report writes
NOT-COMPLETE. This small check excludes historical result trees; the full push/start
path was already exercised by both final test runs.

The S3 test runs the actual estimator with CPU torch/vLLM substitutes. It observes
the checkpoint argument and prompt tokens. It makes the former prompt and golden
paths unreadable. Offline tests reject changed checkpoint, golden, prompt, digest
and raw launch records. Trace tests accept taken < producer body done < ready and
retain the unresolved tail. Timing tests retain the exact 30 microsecond KEEP edge.

The first full-push fixture used a 45-second test limit. It timed out while archiving
the 2.5 GiB tree. The failure log is retained in `launch-integration.log`. The fixture
limit is now 120 seconds. It passed in both final test runs. Production staging is
still limited to 300 seconds. This failure cost local CPU time only.

Heavy validation ran in tmux with the shared `qwen-prep-heavy.lock`. It used private
scratch, a 1,800-second suite limit and a five-GiB free-space stop. Only completed,
owned archive fixtures were removed after their small records were retained.

## Limits and receipt scope

The 18 skips are listed verbatim in `full-suite.log`. They cover absent historical
data/builds, unconfigured toolchain test inputs, native hipcc and Linux `/proc`.
One more test is skipped than in the prior 2,052-pass run: this worktree does not
have the generated device build used by `test_emitted_plan_audit_actual_compiler_output`.
It is not counted as a pass.

No device emitter, kernel, model, host HIP source, build script, header, compiler
flag or kept recipe changed. The original matrix and compiler evidence is reused
only for these unchanged inputs. It is not a new compile run. The original full-suite
receipt does not certify the new Python sources. The new receipt binds those sources
to the new tests. The old receipt and checksum list are kept byte for byte under
`readiness-fixes/previous-*`. Original logs and the original manifest are unchanged.

Linux HIP executable link/load, actual-model replay, fresh imported-launch checks,
32/32 hardware tokens, memory visibility and performance remain pending. No local
substitute closes these gates. GPU spend for this repair is $0.
