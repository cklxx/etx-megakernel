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

### Same-machine A/B script

`tools/ab_same_box.py` applies these rules. It runs three arms on one machine: A (baseline), B (candidate) and C (a second copy of A, the control).

```sh
python tools/ab_same_box.py --a "<baseline run>" --b "<candidate run>" \
    --metric 'step_ms=([\d.]+)' --rounds 6 --min-gain 1 --label qkv_b4 \
    --identity-cmd "ssh <gpu host> 'hostname; rocm-smi --showserial --json'" \
    --out examples/gpu-live/results/<date>-<name>
```

- Warm-up: each arm runs once before round 0. The warm-up values do not count. `ab.log` records them as `ABWARM` lines.
- Order: the rounds cycle through the 6 orders ABC, CAB, BAC, CBA, ACB, BCA. Each arm is in each position equally often. The last arm of each round is the first arm of the next round. So, over the whole run sequence, each arm runs directly after each other arm equally often. `--rounds` must be a multiple of 6. The default is 6.
- The paired gain of each round is `A/B - 1`. The control gain is `A/C - 1`. It measures the noise of the machine.
- `p` is the exact sign-test p-value of the gains. It is one-sided in the observed direction (the sign of the median gain). 6 positive gains out of 6 give p = 0.5^6 = 0.0156.
- WIN: every gain is above 0, the median gain is above max(`--min-gain`, max |control|), and p <= 0.05. LOSS is the same rule with the opposite sign. Any other result is NO-DIFF. `--min-gain` is in percent. The default is 1. A PR that uses another value must give the value and the reason.
- INCOMPLETE: a run exits nonzero or times out, a run prints no value, a value is not a positive number, the identity command fails, or the machine identity changes. The script stops at the first problem. `ab.log` and `ab.json` keep the completed runs.
- Timeout: `--timeout` is the limit for one run. After every run, also after a normal exit, the script kills the process group of the run, so a background child cannot hold the GPU during the next arm. After a timeout kill the script waits at most 10 s for the output pipe, then records `rc=timeout`.
- Timeout over ssh: a local kill stops only the local ssh client. The remote GPU process can continue and disturb the next session. Use `ssh -tt` (the remote process gets a hangup when the client dies), or put a remote timeout around the GPU command, for example `ssh <host> 'timeout -k 10 1700 <gpu command>'` with a limit below `--timeout`.
- Background children: a child process that keeps stdout open makes an rc=0 run wait until `--timeout` (up to 1800 s by default). The run is then INCOMPLETE, so no wrong data is kept, but GPU time is lost. Redirect the output of every background child, for example `cmd > child.log 2>&1 &`.
- Warm-up order: B, C, A. It ends with A, the first arm of round 0.
- `--metric` must have exactly one group. Otherwise the script exits with code 2 before any run.
- Machine identity: the script reads it at the start, before every round and at the end. The default is the local host name and GPU serial. When the runs go over ssh, give `--identity-cmd`. Otherwise the check reads the controller, not the GPU host. An `ABWARN` line shows that no GPU serial was read.
- The commands run in a shell. Do not pass untrusted text. `--label` must not contain white space.
- The exit code is 0 for WIN, LOSS and NO-DIFF, 1 for INCOMPLETE and 2 for bad arguments. `run()` checks the same arguments and raises `ValueError` before any run.
- The script writes `ab.log`. Add `ab.log` to the experiment's `inputs` in `registry.json`. `tools/extract_key.py` then puts the result into `comparisons` and each run into `runs`. An INCOMPLETE session gives an `INCOMPLETE` comparison row and the verdict INCOMPLETE. `tools/check_results.py` prints each verdict and lists every experiment that is not COMPLETE on a `NOT COMPLETE` line.
- `tools/check_results.py` returns 0 for an INCOMPLETE experiment, because older INCOMPLETE results are committed on purpose. CI therefore does not block an INCOMPLETE A/B session. A reviewer must read the `NOT COMPLETE` line of the check output.

## Result files

Each experiment folder under `examples/gpu-live/results/<date>-<name>/` commits two generated files:

- `key.md`: a conclusion table and the extracted log lines. Each log line has its source file, its line number and the SHA256 of the source file.
- `key.json`: the same numbers for machine checks. It contains the config, the machine, the commit, the medians, the spread, the verdict and the correctness gates.

`tools/extract_key.py` writes both files. Do not edit them by hand. A hand-written `result.md` is optional.

The extractor keeps only the lines that a pattern in `PATTERNS` matches. The pattern kinds are:

| Kind | Examples |
|---|---|
| `ab` | `tools/ab_same_box.py` lines: `ABREPORT`, `ABRUN`, `ABWARM`, `ABMACHINE`, `ABWARN` |
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
