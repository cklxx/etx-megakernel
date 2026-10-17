# Experiment process

This document gives the rules for GPU experiments in this repository.

## One experiment is one PR

1. Open one branch and one PR for each experiment. Use `.github/pull_request_template.md`.
2. The PR contains the code change, the runner script change and the key result files.
3. Do not push experiment work directly to `main`. Do not make separate commits that only record acceptance, handoff or authorization.
4. Merge the PR with squash merge. One experiment then gives one commit on `main`.

## Measurement rules

1. Run the control and the candidate on the same machine.
2. Interleave the A/B rounds. Do not compare a number with a number from another session or another machine.
3. Report the median delta and the spread (max - min) of the paired rounds.
4. Report every correctness gate. A failed gate makes the verdict FAIL. A nonzero exit makes the verdict INCOMPLETE.

## Result files

Each experiment folder under `examples/gpu-live/results/<date>-<name>/` commits two generated files:

- `key.md`: a conclusion table and the extracted log lines. Each log line has its source file, its line number and the SHA256 of the source file.
- `key.json`: the same numbers for machine checks. It contains the config, the machine, the commit, the medians, the spread, the verdict and the correctness gates.

`tools/extract_key.py` writes both files. Do not edit them by hand. A hand-written `result.md` is optional.

The extractor keeps only the lines that a pattern in `PATTERNS` matches. The pattern kinds are:

| Kind | Examples |
|---|---|
| `resource` | Compiler remarks: VGPRs, AGPRs, ScratchSize, SGPRs Spill, VGPRs Spill, LDS Size, Occupancy |
| `build` | `BUILD arm=... resources={...}` |
| `exit` | Runner job results: `ok name rc=0`, `fail name rc=1` |
| `error` | `error:`, `fatal error`, `Traceback`, `...Error`, `Connection refused` |
| `gate` | `BGATE ... PASS`, `SEALED ... complete=True`, `B4-RESULT ...`, `N passed` |
| `timing` | `BREPORT ...`, `TRACE COMPLETE ...`, lines with `median` |
| `meta` | Commit hashes, GPU marketing name, `gfx942` |

When a new runner prints a new key line, add a pattern to `PATTERNS`. Then rerun the extractor.

## Registry

`examples/gpu-live/results/registry.json` is the data for the CI check. It has three lists:

- `key_extraction`: one entry per experiment folder. The entry gives the title, the config, the input log globs and the exit receipt globs. It can give the machine or the commit when the logs do not contain them. Each such value gives its source.
- `derived`: one entry per derived file. The entry gives the script, the committed inputs and the output.
- `not_reproducible`: derived files that the CI check cannot rerun, with the reason.

To add an experiment:

```sh
# 1. Add an entry to key_extraction in registry.json.
python tools/extract_key.py --write --only <dir>
# 2. Add derived entries for summaries that a CPU script makes from committed inputs.
python tools/check_results.py
```

## CI check

The `CPU acceptance` workflow runs `python tools/check_results.py`. The check does these steps on a CPU:

1. It reruns each `derived` entry from its committed inputs. It fails when the output bytes differ from the committed file.
2. It reruns `tools/extract_key.py --check`. It fails when `key.json` or `key.md` differs from a fresh extraction. A changed source log changes its SHA256, so an edited log also fails the check.
3. It prints each `not_reproducible` entry.

## Existing files

Existing raw logs stay in place. This process does not move or delete them. `key.json` and `key.md` are not in the folders' `SHA256SUMS` files. They carry the SHA256 of each source file themselves.
