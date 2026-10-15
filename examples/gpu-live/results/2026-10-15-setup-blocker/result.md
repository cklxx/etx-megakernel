# GPU execution — interrupted before experiments

The accepted source was successfully archived, transferred, checksum-verified and staged on root@<GPU_HOST>. Commit: 554133d744b4591c9cf87c68b6817df781137c9b. Archive SHA-256: abee09fcda357bb982cfba4d85bb387a6da97ec2385d3d4574f1f7cd4282d5da. Remote source receipt timestamp: 2026-10-15T15:26:30Z.

All five experiments remain NOT STARTED: Qwen producer, attention early merge, expert row merge, DeepSeek x8 correctness and B4 attribution. No benchmark runner or GPU command was launched. No correctness/performance result exists from this attempt.

The machine initially had Python 3.12.3, the pinned vLLM 0.27.1 container, ROCm 7.14 native hipcc/rocprofv3, but no torch/NumPy/model inputs. Independent venv setup first failed because ensurepip was absent. The second setup installed python3.12-venv and all requested Python dependencies, successfully attached /root/etx-python, then exited 1 on the native HIP header probe: hip/hip_runtime.h missing. The runbook native-header remedy is required after connectivity is restored. New dependency transfers and a separate read-only reconciliation returned SSH Connection refused, exit 255. Remote setup/process state is UNKNOWN and must be reconciled before any restart. No benchmark had been started by these commands.

Raw logs/exit receipts: /LOCAL_WORKSPACE/.hermes/cache/scratch/etx-gpu-live-20261015-evidence. stage.log and stage.exit establish successful staging; target-environment.log records prerequisites; python-setup*.log/.exit preserve both setup attempts; fleet/header/pinned-fleet transfer and ssh-reconcile-01 logs retain actual transport outcomes. SHA256SUMS binds exported evidence.

Amber handoff: restore/reconcile SSH on authorized target <GPU_HOST> (droplet <GPU_RESOURCE_ID>) to permit continuation. Fixed execution/export stop remains 2026-10-15T17:22:10Z; no reset or extension. No alternative machine, provider token retry, credential/account change, recurring monitor, or provisioning action was attempted. Amber owns eventual deletion; poweroff does not stop billing. Provider deletion is NOT verified.

DeepSeek same-kernel replay remains UNAVAILABLE due to LDS lifetime; the complete two-baseline/fusion claim remains INCOMPLETE. Existing 2190-pass/compiler preparation evidence was reused without redundant reruns. Frozen source was not edited or published.
