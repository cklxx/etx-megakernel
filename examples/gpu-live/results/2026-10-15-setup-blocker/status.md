# Live GPU status — BLOCKED on SSH, 2026-10-15T15:28Z

Authorized target: root@<GPU_HOST>, droplet <GPU_RESOURCE_ID>. Fixed execution/export deadline: 2026-10-15T17:22:10Z (unchanged). Amber owns resource supervision/deletion. No provider API/token retried; no monitor/cron restarted.

Frozen source successfully staged: 554133d744b4591c9cf87c68b6817df781137c9b, archive abee09fcda357bb982cfba4d85bb387a6da97ec2385d3d4574f1f7cd4282d5da, remote /root/src/abee09fcda357bb982cfba4d85bb387a6da97ec2385d3d4574f1f7cd4282d5da-20261015T152622Z, receipt staged 2026-10-15T15:26:30Z.

Evidence: /LOCAL_WORKSPACE/.hermes/cache/scratch/etx-gpu-live-20261015-evidence

- Producer: NOT STARTED. Python/model/pinned source prerequisites incomplete.
- Attention early merge: NOT STARTED. Same model/source prerequisites incomplete.
- Expert merge: NOT STARTED. Preparing independent Python; first attempt refused missing ensurepip. Second attempt successfully installed NumPy/PyYAML/HF dependencies and attached /root/etx-python to the staged source. Its final native compiler probe failed: hip/hip_runtime.h missing (exit 1). Native HIP header installation is required once SSH is restored. No runner launched.
- DeepSeek x8: NOT STARTED. Pinned Fleet transfer failed at SSH connection. Existing writable Fleet transfer completed but is not accepted pinned reference input and was never used.
- DeepSeek B4: NOT STARTED. Same inputs unavailable; replay baseline remains UNAVAILABLE (LDS lifetime), full fusion claim INCOMPLETE.

At approximately 15:27Z, new SSH connections returned Connection refused. A dedicated read-only reconciliation command also exited 255; remote process state is UNKNOWN. No restart attempted. Preserve all attempts, logs and exits. Next requires Amber to restore/reconcile SSH on this exact authorized target; do not substitute machines or reset deadline. Once reachable, inspect setup state before continuing to expert synthetic and reachable model experiments.
