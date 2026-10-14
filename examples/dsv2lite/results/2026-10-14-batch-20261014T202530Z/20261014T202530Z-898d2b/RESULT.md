# Batch 8 diagnostic: one arm ran four times, so this record has no recipe comparison (20261014T202530Z-898d2b)

Condition: one MI300X VF (192 GB, ROCm 7.14); four candidate arms built from one staged tree on the same machine;
references from the goldens set generated on the box (`GOLDENS OK 8 requests: prompts are the 97 r rotations, pairwise
distinct; collisions [[0, 4]]`).

What the record contains (raw lines):

    BUILD arm=x8p OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 96,  'lds': 26668, 'occupancy': 2} plan_sha256=1d19b7e…
    BUILD arm=x8m OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 96,  'lds': 26668, 'occupancy': 2} plan_sha256=1d19b7e…
    BUILD arm=x8k OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 144, 'lds': 26668, 'occupancy': 2} plan_sha256=6beb4b0…
    BUILD arm=x8  OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 144, 'lds': 26668, 'occupancy': 2} plan_sha256=6beb4b0…
    BGATE arm=x8p b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)
    BGATE arm=x8m b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)
    BGATE arm=x8k b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)
    BGATE arm=x8  b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)

The builds are four real builds and they differ by recipe, as their plans and resources show. The gates are not four
gates: `BatchSession.etx(b, name)` resolved the arm from the batch number, so all four invocations ran `b_x8/run` and the
four verdicts above are one measurement recorded once per label. **This record therefore cannot say whether the plain
replication arm shares the failure.** The claim drawn from it (that eight requests break in ETX regardless of recipe)
was withdrawn on 2026-10-15; the fix and its local test are in `examples/dsv2lite/batch_session.py` and
`tests/test_dsv2_batch.py::test_gate_runs_the_arm_it_names`.

Batch 8 remains without a verdict. One gate per arm, with the fix staged, is the next GPU step.

The record itself is complete and verified: `VERIFIED: session 20261014T202530Z-898d2b, 61 files match, 0 bad, 60 left
on the box by design; complete`. Two earlier attempts in this hunt are recorded as well (`20261014T185922Z-3e63d4`:
`goldens.json` absent; `20261014T201049Z-0b4dcc`: the gate refused the candidate names) — tooling faults, both fixed.
