# Producer-scoring GPU runbook

Status: prepared experiment. No GPU run is authorized by this preparation task.
The candidate is off by default. It has no real-model correctness or speed result.

## Local check

From a clean task checkout, use the repository Python environment:

```sh
.venv/bin/python -u -m examples.vllm_llm.producer_local \
  --out /absolute/new/evidence-directory
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/qwen_producer.py --dry-run
```

The local driver writes `examples/vllm_llm/results/2026-10-15-producer-prep/local-ready.json`.
It refuses to overwrite that file or its evidence directory.
Archive an older local receipt before a new acceptance run.
A source change invalidates the old receipt.
Commit the source and receipt before staging.
The final preparation record contains the exact commands, exit codes and skip reasons.

## Prerequisites for a later approved run

Use one MI300X GPU lane. Do not select a two-GPU VM for this job graph.
Use the existing provisioning and teardown workflow in `AGENTS.md`.
Check the balance and current price before a funded start.
No provisioning or price check was done in this task.

The box must already have these inputs:

- The bf16 packed Qwen checkpoint, `vl.manifest`, pinned HF files and `golden_c1024.txt`.
- A `qwen-g6.json` with `model`, `hf`, and `golden_sha256` paths or values.
- An independent Python >=3.11 venv outside all checkouts, with NumPy and PyYAML installed inside it.
- The existing build environment: HIP compiler, ROCm headers and vLLM source.
- The content-addressed vLLM image from `examples/vllm_llm/qwen300_config.py`.
- The committed local-ready receipt with matching source hashes.

The resolved `hf` path must equal the resolved `model/hf` path. A symlink to the same tree is valid.
S3 reads both its prompt and expected tokens from the hashed `model/golden_c1024.txt`.
The reference JSON and raw log retain the actual launch path and the input digest.
The offline report checks these bindings as well as the token and timing gates.

The preflight verifies these inputs. It does not install, download or pull them.
The session hashes the complete packed weight file and the HF reference files.
It retains source, image, hardware, compiler, plan and executable identities.
Each executable is separate. There is no shared `run` alias.

## Budget and command

Plan at most 60 GPU minutes, including staging and evidence collection.
Reserve five minutes for evidence. Use at most five minutes for staging.
The example caps the runner at 55 minutes.
At the historical, unverified assumption of $1.99/GPU-hour, 60 minutes is about $1.99.
This is a budget estimate, not a current price quote.
If setup exceeds its allowance, reduce the runner cap or stop the run.
The runner never extends the budget to finish a sample population.
Slow model loads can make this session NOT-COMPLETE.

Only after separate GPU authorization, stage the committed source and start:

```sh
python3 -m tools.producer_start IP \
  --venv /home/hotaisle/envs/etx \
  --config /home/hotaisle/qwen-g6.json \
  --allocation-deadline ORIGINAL_ALLOCATION_END_UNIX_SECONDS
```

Use the original allocation end time, no later than allocation start plus 3,600 seconds.
Use the correct absolute config and independent venv paths for the chosen box.
This wrapper runs the real `session.sh push` (fresh archive and dependency transfer),
then attaches the supplied venv at the new root and probes Python, NumPy and PyYAML.
It performs no install or download. The old checkout's `.venv` is not used.
Push and attachment share a 300-second deadline. A failure prevents start.
The runner cap is at most 3,300 seconds and also fits the remaining allocation time,
with a 15-second SSH startup margin. Keep the same allocation deadline on retry.
The persistent lease can refuse a retry whose cap differs; do not reset it.
Fetch and stop in that case. Setup cannot silently extend the funded budget.
The existing lifecycle owner must enforce the allocation deadline.
The runner cap starts at runner launch, not VM allocation.
The job file's absolute maximum is 3,600 seconds.
The example selects a smaller cap to leave staging time.

The optional runner lease is `~/.cache/etx-qwen-producer-session.lease` on the box.
The runner takes the lock before it creates an evidence directory or starts a job.
A second runner is refused.
A retry keeps the first deadline and creates a unique record.
It cannot change the cap, run a partial graph, or overwrite raw evidence.
Do not delete the lease to obtain another funded attempt.
A new budget requires a separate decision.

The phase order is preflight, parallel CPU builds, actual-model capture and correctness,
five interleaved wall rounds, optional traces, and a final report.
The session keeps full raw logs and all build and capture files.
Each command has a deadline. Each long command prints a progress line every 30 seconds.
A command silent for 180 seconds triggers a process-state probe and a failure.
A reserve check stops new commands before evidence time is consumed.

## Failure, fetch and teardown

```sh
tools/session.sh status IP
tools/session.sh fetch IP /absolute/local/evidence-directory
```

Read the `FAIL`, `EVIDENCE`, and `SAFE` lines first.
A failed actual-model routing or expert-input byte gate blocks all timing.
Raw router-logit changes are diagnostic. The existing numerical gate remains active.
A failed or missing certificate, token result, or CHECK-DSL result also blocks timing.
A timeout or incomplete sample set has no speed claim.
Use the fetched raw logs to fix the exact failed phase.
A retry restarts correctness and keeps the original session budget.
If the lease has expired, fetch and stop. Do not restart compute.

Delete the VM immediately after the planned work or the allocation deadline.
Use the existing provider workflow. Verify `No virtual machines` and record the balance.
This preparation does not add a new cloud lifecycle system.
The runner does not delete the VM.

For offline report reproduction, point `ETX_SESS_LOGDIR` at the fetched record:

```sh
ETX_SESS_LOGDIR=/absolute/fetched/record \
  .venv/bin/python -m examples.vllm_llm.producer_session report
```

The report file is exclusive. To recompute an existing report, preserve it first.
The report revalidates the evidence and refuses partial results.
