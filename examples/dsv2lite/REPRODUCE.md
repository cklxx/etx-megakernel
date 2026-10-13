# Reproducing the 2026-10-13 ETX DeepSeek lever session

One entry point, `examples/dsv2lite/reproduce.sh`, takes a fresh clone to the measured numbers. Steps 1-3 need no GPU
and re-derive every published verdict from the committed raw logs; steps 4-8 re-run the session on a box.

## What was measured

| | value | where |
|---|---|---|
| ETX commit that ran | `7f3dd89f8cc92648b4fa35f9000028059658cfee` (branch `dsv2-session`) | `results/2026-10-13-ds_inplace/source.json` |
| fleet source | `3fb67a6`, content hash `28f5cbb4f39524df…` (git archive, `pinned.py`) | same |
| inputs | weights.bin `ad7e8290b804…` (31,412,968,448 B), fleet_cache.bin `61f526227ca7…`, golden_tokens.txt `7ca30d0e843e…`, golden_hidden.bin `dccd2f4818ae…`, weights.bin.manifest `1f0f667dd8ab…` | `inputs.json` |
| job table | 33 jobs, hard stop 7200 s | `DRYRUN.txt` (= the 33 jobs `runner.log` started) |
| result | ETX all levers 3.616 ms vs default 3.712 ms: **median 92.0 us, spread 18.0, every pair faster, GO**; fleet 3.568 ms | `pairs_all.json`, `RESULT.md` |
| follow-up | all without Q-all 3.600 ms (-108 us vs default); ROCm 7.2.3 build of default 3.646 vs 7.14 3.708 ms | `followup/` |

Paths are relative to `examples/dsv2lite/`.

## Machine and toolchain (the measured box)

- GPU: one AMD Instinct MI300X VF, gfx942 (sramecc+, xnack-), 304 CUs (DigitalOcean MI300X droplet).
- Host: Intel Xeon Platinum 8568Y+, 20 vCPUs, 235 GB RAM, 697 GB disk (needs >= 100 GB free: weights 31 GB + model 31 GB).
- OS: Ubuntu 24.04 (apt installed python3.12-venv 3.12.3-1ubuntu0.17; sshd OpenSSH_9.6p1 Ubuntu-3ubuntu13.18), python
  3.12.3, no torch preinstalled (the session installs its own).
- ROCm 7.14: `HIP version: 7.14.60850-0000000`, `AMD clang version 23.0.0git (… 46fcb339fb61119b…)`, hipcc at
  `/opt/rocm/core-7.14/bin/hipcc`. The image shipped WITHOUT HIP headers: step 4 installs `amdrocm-runtime-dev7.14`.
- Network on the box: huggingface.co (model `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`), PyPI, download.pytorch.org.
- The box rate-limited new SSH connections (ufw limit): every script here uses one multiplexed connection.

Laptop: git, python >= 3.11, `gh` logged in with read access to `cklxx/etx-data` (archived data sets), a clone of
`github.com/cklxx/fleet-mi300x` at `ETX_FLEET_REPO` (default `~/code/fleet-mi300x`), an SSH key the box accepts.
`ETX_CACHE` (default `~/code/etx-cache`) receives the pinned fleet export and the data sets.

## Steps and their expected output

**0. Clone** this repository and `git checkout dsv2-session` (or any later commit of it: step 2.1 proves the box-side
code is still `7f3dd89`).

**1. `examples/dsv2lite/reproduce.sh setup`**: `.venv` with the dev extra.
```
OK   .venv ready: Python 3.1x.y
```

**2. `examples/dsv2lite/reproduce.sh check`** (no GPU, ~30 s once the caches exist). Expected, in order:
```
== 1. box-side code is the measured commit (7f3dd89); later commits may only ADD files
OK   no box-side file modified or removed since 7f3dd89
== 2. pinned fleet source (git archive 3fb67a6)
OK   <ETX_CACHE>/fleet-3fb67a6 = 3fb67a6, src sha256 28f5cbb4f39524df...
== 3. archived data sets (cklxx/etx-data release data-2026-09-27, tarball + per-file SHA256)
OK   <ETX_CACHE>/data-2026-09-27
== 4. CPU gates (tests/test_dsv2_*.py)
OK   63 passed in <t>s
== 5. dry run: the same 33-job table as the measured session
OK   33 jobs, identical to examples/dsv2lite/results/2026-10-13-ds_inplace/DRYRUN.txt
== 6. expected per-stage values regenerate byte for byte (stage_table.py: trace partition + replay)
OK   Summary, arithmetic (rows add): 3473.5 - 533.4 = 2940.1 us/token = 340.1 tok/s.
== 7. every verdict re-derived from the session's raw logs (incl. the combined arm's 92 us)
  all      vs default median    92.0 spread  18.0 every_positive=1 GO=True  device medians default=3.712 all=3.616 fleet=3.568
  ...      (every arm: baseline 139.0, Q -3.0, Qall -3.0, N 30.0, Nall 37.0, argmax 2.0, topk 66.0, prep 27.0)
STAGES default: span 3959.8 us, recomputed from trace + stamps, equal to stages_default.json
STAGES all: span 3890.7 us, recomputed from trace + stamps, equal to stages_all.json
FOLLOWUP e1 all-allnoQ: median 14.0 spread 10.0 ...   FOLLOWUP e2 default-default723: median 62.0 spread 14.0 ...
RESULT OK: every verdict re-derived from raw logs
OK   all vs default: median 92.0 us, spread 18.0, every pair faster, GO
```
The STAGES lines need the two STAMPS2 stamp files (2 x 112 MB, archived outside git in
`$ETX_CACHE/results_2026-10-13-ds_inplace/`, hashes in the session's `SHA256SUMS`); without them they print `skipped`.
Steps 1-3 re-derive the numbers from the recorded raw logs EXACTLY (to 1e-6 us). Steps 4-8 re-measure: a new session
gives new milliseconds (timing noise, and absolute ms are not compared across boxes); what must reproduce is the
decision under the same rule, with the same commit and the same input hashes.

**3. Provision** one MI300X box as above (`ETX_SESS_USER`, default `root`).

**4. `examples/dsv2lite/reproduce.sh box-setup <ip>`**
```
installing amdrocm-runtime-dev7.14 (the HIP headers of the installed ROCm)      # only if they are missing
HIP-HEADERS-OK
HIP version: 7.14.60850-0000000
```

**5. `examples/dsv2lite/reproduce.sh stage <ip>`** (refuses a dirty tree)
```
fleet export <ETX_CACHE>/fleet-3fb67a6 (verified against the pinned 3fb67a6 hash)
commit <HEAD>
branch dsv2-session
dirty 0
FLEET-EXPORT-OK
HIP-HEADERS-OK
```

**6. `examples/dsv2lite/reproduce.sh run <ip>`**, then `reproduce.sh status <ip>`. The measured run took ~16 min
(prep 4 min with builds in parallel, then 97 timed invocations of ~8 s). Key lines in `~/sess/*.log`:
```
SOURCE OK fleet export 28f5cbb4f39524df etx <commit>
      smallest top-2 margin over the run: 6.6250
wrote /root/dsv2-data/weights.bin (31.41 GB) ...
PREFLIGHT OK inputs are the pinned measured inputs
BUILD arm=default stamped=0 resources={'vgpr': 256, 'agpr': 56, 'sgpr': 108, 'scratch': 0, 'lds': 26668, 'occupancy': 1}
BUILD arm=all stamped=0 resources={'vgpr': 256, 'agpr': 104, 'sgpr': 108, 'scratch': 0, 'lds': 26668, 'occupancy': 1}
PAIRS arm=baseline device gain vs default median 139.0 us spread 20.0 every_positive=1 GO=None
PAIRS arm=all device gain vs default median 92.0 us spread 18.0 every_positive=1 GO=True
REPORT all: median 92.0 us spread 18.0 GO=True; stages measured: True; ...
```
(ROCm 7.14 resources; every arm 0 scratch, 26,668 B LDS, 1 wave/SIMD. A build with scratch > 0 fails its job.)

**7. `examples/dsv2lite/reproduce.sh fetch <ip> <dest>`**, then **`reproduce.sh rederive <dest>`**:
```
RESULT OK: every verdict re-derived from raw logs
inputs.json: SAME as the measured session
source.json: SAME as the measured session
```
`DIFFERENT` means a different experiment (e.g. a new HuggingFace revision of the model: the download does not pin
one; the weights hash is the identity). Fetch BEFORE destroying the box.

**8. Follow-up arms** (optional; same box after step 6):
`ETX_SESS_LOGDIR=~/sess_followup ETX_DSV2_FLEET=~/fleet-3fb67a6 ETX_DSV2_DATA=~/dsv2-data .venv/bin/python -u -m examples.dsv2lite.followup_2026_10_13 /root/sess`
on the box (needs the `rocm:latest` image with ROCm 7.2.3 for E2). Expected: `FOLLOWUP E1 all - allnoQ`,
`FOLLOWUP E1 default - allnoQ`, `FOLLOWUP E2 default - default723` lines and `followup.json`.

## Verified

2026-10-13, from `git clone` of this repository at `dsv2-session` (58f2f95) with an EMPTY `ETX_CACHE` and a new `.venv`
(Python 3.14.0): `reproduce.sh setup` OK; `reproduce.sh check` all 7 steps OK in 37 s (fleet export rebuilt from the
fleet repository and hash-checked, data sets downloaded and checksum-verified, 63 CPU gates, the 33-job table identical,
stage table byte-identical, plan regenerated = `09011c20`, all vs default 92.0 us / 18.0 / GO re-derived from the raw
logs). STAGES printed `skipped` there: the two STAMPS2 stamp files exist only in the measuring machine's
`ETX_CACHE/results_2026-10-13-ds_inplace/` until they are published to the data repository.

## Files

`reproduce.sh` (entry point), `../../tools/session.sh stage` (staging), `../sessions/ds_inplace.py` (job graph), `inplace_session.py`
(each job), `rederive.py` (verdicts from raw logs), `stage_table.py` + `replay.py` (expected values),
`pinned.py` (fleet export + data sets), `verify_lever.py` (per-lever local records), `followup_2026_10_13.py`.
