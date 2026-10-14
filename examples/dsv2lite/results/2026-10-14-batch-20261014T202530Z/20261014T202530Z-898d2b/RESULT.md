# Batch 8: the failure is in ETX's eight-request path, not in the recipe (session 20261014T202530Z-898d2b)

Condition: one MI300X VF (192 GB, ROCm 7.14); four arms built from one staged tree on the same machine; references from
the goldens set generated on the box (`GOLDENS OK 8 requests: prompts are the 97 r rotations, pairwise distinct;
collisions [[0, 4]]`); one eight-request gate per arm, 27 layers of each request compared against vLLM's own tokens.

Raw verdicts:

    BUILD arm=x8p OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 96,  'lds': 26668, 'occupancy': 2}
    BUILD arm=x8m OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 96,  'lds': 26668, 'occupancy': 2}
    BUILD arm=x8k OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 144, 'lds': 26668, 'occupancy': 2}
    BUILD arm=x8  OK resources={'vgpr': 256, 'sgpr': 108, 'scratch': 144, 'lds': 26668, 'occupancy': 2}
    BGATE arm=x8p b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)
    BGATE arm=x8m b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)
    BGATE arm=x8k b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)
    BGATE arm=x8  b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)

`x8p` is the plain replication arm: it carries the five levers and `l1_loads`, and neither of the two levers this round
added. It fails exactly as the arm that carries both. So eight requests break in ETX itself, and the four builds exist
to separate that from a lever interaction — the separation is the result.

The record is complete: `VERIFIED: session 20261014T202530Z-898d2b, 61 files match, 0 bad, 60 left on the box by
design; complete`. The two earlier diagnostic attempts in this hunt are recorded too
(`20261014T185922Z-3e63d4`: `goldens.json` missing; `20261014T201049Z-0b4dcc`: the gate refused the candidate names);
both are tooling faults, both fixed and pushed, and neither says anything about batch 8.
