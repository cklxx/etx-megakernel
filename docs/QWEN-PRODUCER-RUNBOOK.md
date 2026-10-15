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
- The existing build environment: Python, NumPy, HIP compiler, ROCm headers and vLLM source.
- The content-addressed vLLM image from `examples/vllm_llm/qwen300_config.py`.
- The committed local-ready receipt with matching source hashes.

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
tools/session.sh push IP
ETX_QWEN_G6_CONFIG=/home/hotaisle/qwen-g6.json \
ETX_SESS_MAX_SECONDS=3300 \
  tools/session.sh start IP examples/sessions/qwen_producer.py 0 A
```

Use the correct absolute config path for the chosen box.
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
