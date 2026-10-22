# GLM baseline protocol

Read this file before any baseline run. It matches the Qwen and DeepSeek protocol
already in the repo. Do not invent another protocol.

## 1. Input

Each request uses one golden file.

Each golden file has exactly 1024 prompt tokens and 32 independent HF greedy tokens.

The goldens live in examples/glm53/goldens and are committed.

Batch N uses the first N goldens in file-name order. The order is the same for
every batch.

## 2. Golden generation

Generate the goldens once with examples/glm53/reference.py.

Use the same checkpoint, the same REVISION file and the same tokenizer as the run.

Record the checkpoint REVISION in the certificate.

## 3. Correctness gate

Run the gate before any timing run.

The gate passes when the ETX tokens equal the HF greedy tokens for all 32 positions
and the cosine is at or above the repo threshold.

If the gate fails, do not report a step time. Fix the path first.

## 4. Timing

Measure host wall time.

Start the clock before the launch and stop it before any device-to-host copy.

Report device time and host wall time in two separate columns. Never mix them.

## 5. Run order

Run the batches serially on one GPU pair.

Do not run batches in parallel on different GPU pairs. Those numbers are not
comparable.

## 6. Repetition

Run a warmup, then take the median of at least 20 measured steps.

Report the spread with the median.

## 7. Certificate

The certificate records: golden file names, checkpoint REVISION, build identity,
gate state, batch size, step count, host wall per token, device time per token.

## 8. Stop rule

If the same failure happens twice, stop and report.

Do not start a third attempt on the same cause.
