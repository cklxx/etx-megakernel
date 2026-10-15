# Executed validation and limits

CPU only, macOS arm64, Python 3.13.5, NumPy 2.5.3. Exact source/input hashes are in
`manifest.json`. No production kernel or default behavior changed.

- Full repository suite: `.venv/bin/python -m pytest -q -ra`, **2028 passed,
  17 skipped, zero failures**, exit 0, **471.24 seconds**. `full-suite.log` is unfiltered.
- Final focused regression suite: **14 passed**, zero skips, exit 0, **0.18 seconds**
  in `final-focused.log`. The full suite collected the original 12 study tests before
  two further tests and final helper integration were added. The final 14-test run,
  restored-guard run and byte-identical simulation replay cover the final study code.
  The whole suite was not rerun after those isolated study refinements. Do not add
  these overlapping test counts or call the 2028 count a final 2030-test execution.
- Three intentional child-process mutants: each exits 1 with the expected pytest
  assertion failure; restored final module: **14 passed**, zero skips, exit 0.
  See `guard-execution.log`, `mutant-*.log`, `guards-restored.log`.
- Original numerical/schedule execution and final replay: completed, exit 0. The
  final replay reproduces all four JSONs byte-for-byte (`replay-verification.log`).
- `python -m py_compile` for both new modules and `git diff --check`: pass.
- First parser run failed on `moe_up_0_x`; original failure retained. Actual name
  `moe_up_0` is handled and regression-tested. No failed run was used as a timing result.

## Full-suite skips (not passes)

The 17 skip records remain in the raw log:

- 2 acquire-contract checks lack retained ROCm22 Qwen/Mixtral builds.
- 1 bandwidth acceptance, 2 floors checks, 1 layer-budget and 1 round48 pair check
  lack archived inputs or specified ISA/vLLM paths (floors includes a vLLM checkout).
- 3 coherence-model and 2 phase-model checks lack measured plans.
- 1 codegen check lacks hipcc.
- 1 importer check lacks its configured vLLM/local toolchain inputs.
- 1 gpu_jobs descendant sweep requires Linux /proc.
- 1 vlbest-session check needs an injected vLLM source checkout.

These are environment/data/platform skips, not numerical-policy exceptions. The full
raw reasons and file/line attribution are in `full-suite.log`. No test was disabled
or tolerance widened for this study.

## Not executed / not established

No candidate GPU kernel exists: the proposed exact identity failed on CPU. Therefore
there is no candidate gfx942 resource report, 16-build local matrix, VM-compiler
build, actual-model router replay, fresh imported-launch certificate, 32-token
candidate correctness run, GPU timing or current-build gain estimate. The successor
P5 experiment is **design-only and blocked on local implementation/prerequisites**,
not queued or launch-ready. Existing production-kernel GPU evidence is not inherited
by a nonexistent candidate. All GPU-spend and cloud-mutation operations stayed unrun.
