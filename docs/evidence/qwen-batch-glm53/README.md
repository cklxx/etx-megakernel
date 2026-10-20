# Task evidence — CPU preparation only

Read `validation.json` and `../../QWEN-BATCH-GLM53.md` before interpreting any log.
No new GPU correctness or performance was measured. Planned import counts are
not checked counts; synthetic test process output is not model evidence.

Original `glm-config.json`, `glm-model-api.json`, `glm-recipe.yaml` and
`sources.json` from the interrupted attempt are preserved byte-for-byte.
`source-verification.json` adds pinned/public-source verification without replacing them.

`pytest.log` is the initial failing run (C++ fixture and dirty-tree refusal).
`pytest-final.log` is the successful clean-tree full run at aaab31d.
`focused-tests-final.log` covers subsequent supervisor/native-wrapper/preparation fixes.
`matrix/` contains all 16 unfiltered local build/compile/configuration logs.

`compiler-commands.json` records exact native/default-flag commands and versions.
`compiler_b*/artifacts.json` hashes generated inputs and useful local outputs;
large generated files remain under build/, rather than being added to git.
`qwen-oom-events.log` confirms local container OOMs. O1 was a compile-capacity
probe only and was not adopted. Native B2 compiler stages/object completed, but
its old enclosing wrapper failed after it was edited while active. Native B4
IR is retained; B4 ISA and B8 native checks are NOT-COMPLETE. The current native
wrapper is bounded and tested, including explicit synthetic compiler fixtures.

The GLM boundary compiler logs prove compilation only. `glm-model-blocker*`
intentionally exits 2; the probe is not GLM inference. Complete ROCm B4/B8
preflight with adequate local compiler memory before any GPU provisioning.
