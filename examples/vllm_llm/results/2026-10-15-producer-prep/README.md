# Producer scoring preparation evidence

This directory contains CPU executions only. GPU spend is $0.
No file in this directory is a real-model producer correctness or speed result.

`final-v7/commands.json` is the final local acceptance command record.
Each entry has its actual exit code, elapsed time, command and full log path.
`local-ready.json` binds all checked source files after that record completes.
`validation.md` gives the final counts, skip reasons and compilation limits.
`SHA256SUMS` covers the retained files, excluding itself.

The numerical model calls `etx/ops/producer_math.h` through a compiled native library.
`simulation.json` is the first completed 32-case run.
The final driver also writes a fresh simulation in `final-v7/`.
Both use synthetic weights and synthetic activations.
They reuse the retained sparse witness. Actual-model replay is pending.

The first CPU compile used Apple's clang without C++ standard headers and failed.
The corrected run uses the installed LLVM compiler and passes.
The first ROCm candidate compile failed because the combined emitter removes includes.
The fix embeds the real shared header text in the emitted tile.
The next ROCm compile passed. Full error and compiler logs are retained.

Two earlier local acceptance attempts were interrupted during the full suite.
Review found that experts consume only the canonical normalized input.
The first interruption allowed the canonical-input correction.
The second allowed complete HF input fingerprints and exact timing arithmetic.
The candidate and capture format were corrected before final acceptance.
Neither interrupted run is a full-suite pass.
A later focused run caught a decimal timing type error. The corrected suite passes.
The first complete full-suite run found eight host-dispatch fixture setup errors.
The fixture now supplies the new capture buffers.
A new test executes the actual C++ capture writer with CPU memory-copy stubs.
It verifies written bytes, capture provenance, and overwrite refusal.
The repaired host-dispatch suite passes 9/9.
The final full-suite record follows that correction and the gate-policy audit.
The audit keeps raw router-logit changes diagnostic.
It keeps the registered exact routing and expert-input checks.
A successful full-suite and matrix run then found container include paths in the native compile.
The local driver now maps only those include paths to the same host-mounted files.
It preserves the original source and records the mapping.
An early native IR and assembly check verifies the fix before final acceptance.
The next full suite found one test portability violation.
The fixture now uses a temporary home directory. The focused suite includes the portability guard.
The final driver runs full tests and compiler checks in parallel and waits for both.
The earlier focused test failures and their corrected reruns are retained.

There was no GPU allocation, cloud mutation, remote execution, deployment,
credential output, precision change, default enablement, or AGENTS.md edit.
