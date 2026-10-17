# Final local acceptance evidence

Runtime implementation: c906b3bf8013b66c29b0932d82384ba1d3c79fe9.
Qwen compiler fix and B2/B4/B8 completion: 85900b2.

- integrated-full-clean.log: 2280 passed, 18 skipped, clean tree, exit 0.
- qwen-local-completion.json and qwen-batch-complete.log: all three exact-standard ROCm O3 batch builds + host checks.
- qwen-matrix-complete.json/log: all 16 model/configuration builds.
- glm53-acceptance-verified.log: complete committed GLM local acceptance script, exit 0, 43 focused tests.
- glm-cpu-b*.json: exact 32-step IDs, 480/480 total; synthetic scaled model, same native numerical body as the HIP tiles, independent HF reference.
- glm-full-shape-compile.json and compiler logs: actual full 45-layer/H4096/B8 code generation and both GPU kernels compiled; no checkpoint weights and no GPU execution.
- glm-arithmetic.json: 254 finite FP8 patterns and 10000 BF16 rounding cases.
- initial-* and older failure logs: original harness, top-k tie and conv-rounding failures, retained for causal attribution.

All compiler/CPU records explicitly distinguish local checks from hardware execution. No new GPU correctness/performance claim or spend. See report.md and ../../../QWEN-BATCH-GLM53.md for scope, numerical policy, exact commands and hardware-only gates.
