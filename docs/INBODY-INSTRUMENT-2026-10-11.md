# In-body instrumentation — 2026-10-11

CPU-only work, based on 4a39f92, branch wf/inbody. No GPU measurements,
provisioning, downloads or network spend. Implementation and evidence are
checkpointed on this branch.

## Pre-registration

Target: explain the supplied 0.847 ms gap, not improve kernels in this change.
On one MI300X, Qwen3-30B-A3B bf16, batch 1, context 1024, KEY32 on:
success requires at least 90% of machinery + dependency gap + hop attributed
to named **exclusive** atomic items, and the largest removable item priced.
Missing data, overlapping issue intervals, or absent removability evidence
cannot count as attributed work. Failure means retain raw evidence and report
NOT-ESTABLISHED; do not propose a saving from an unpriced pool.

GPU perturbation, imported-launch checks and 32/32 tokens are NOT RUN here.
The forthcoming session must measure stamps on/off with counters on/off.
The original four trace words must retain their existing semantics.

Implementation details and CPU evidence will be added as sections land.
