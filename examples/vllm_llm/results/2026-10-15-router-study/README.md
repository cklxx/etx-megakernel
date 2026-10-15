# CPU-only Qwen router study evidence

See [the study and experiment design](../../../../docs/QWEN-ROUTER-FUSION-STUDY-2026-10-15.md).
No GPU execution or spend. No candidate GPU kernel exists.

- `tr_off.bin.gz`, `tr_off.log`: original October 11 Qwen trace, lossless gzip and unchanged log.
- `routing-originals.json.gz`: 384 original binary routing files encoded as hex with individual hashes; output-only evidence, not inputs to the candidate.
- `fleet-raw/`: original DeepSeek v0.16 published/per-task routing run JSONs, for historical classification only.
- `manifest.json`, `history-search.log`: source identities, input discovery limits, reachable-history search.
- `simulation.log`, `numerical.json`: seeded synthetic full-shape numerical study and sparse counterexample.
- `windows.json`, `costs.json`: per-layer replay of historical producer finish times and 216 assumed cost scenarios; no current wall prediction.
- `routing-audit.json`: reuse of actual historical off/on top-k output evidence, not a deferred-scale check.
- `initial-parser-failure.log`: first execution failed on an incorrect `_x` suffix; preserved without filtering.
- `mutant-*.log`: three real, intentional failing pytest executions, before restoring the guards.
- `guard-execution.log`, `guards-restored.log`, `final-focused.log`: focused test results.
- `replay.log`, `replay-verification.log`: independent rerun into another output directory; all four derived JSONs byte-identical.
- `full-suite.log`, `validation.md`: full-suite result, skips, timing and readiness limits.
- `SHA256SUMS`: all retained evidence except the checksum file itself.

Run the modules from repository root; no local cache path is required to replay the
committed inputs. Use a separate `--out` directory to preserve the original run.
The synthetic numerical study rejects nonfinite inputs. NumPy exp/rsqrt are not
ROCm intrinsics. The CPU publication model is a logical ordering check, not a GPU
memory-system simulation. Trace/log type alignment is checked, but the missing
historical plan prevents a fresh task-to-plan dependency identity check.
