<!-- One experiment = one PR. The PR is squash-merged. See docs/EXPERIMENTS.md. -->

## Hypothesis

<!-- State one testable claim. Give the expected effect and the threshold that accepts or rejects it. -->

## Change

<!-- List the code and config changes. Give the base commit and the candidate commit. -->

## Control group

<!-- Use the same machine for the candidate and the control. Interleave A/B rounds (for example A B B A A B).
     Give the machine name, the GPU index, the number of rounds and the order. -->

- Machine:
- Control commit:
- Candidate commit:
- Rounds and order:

## Result

<!-- Report deltas and spread only. Do not report absolute times from different sessions or machines.
     Copy the conclusion table from the generated key.md. -->

| Metric | Median delta | Spread (max - min) | Rounds | Verdict |
|---|---|---|---|---|
|  |  |  |  |  |

Correctness gates:

## Conclusion

<!-- Accept, reject or inconclusive. Give the next step. -->

## Evidence location

<!-- Commit key.md and key.json. tools/extract_key.py generates them.
     Give the location and the SHA256 of each raw file that is not in git. -->

- key.md:
- key.json:
- Raw files not in git (location and SHA256):
- Registry entry in examples/gpu-live/results/registry.json: yes / no
- `python tools/check_results.py` passes: yes / no
