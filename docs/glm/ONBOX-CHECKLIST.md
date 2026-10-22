# GLM on-box checklist

One row per step. Same six fields in the same order. A step that cannot be filled
carries U and a one-line reason.

Box used on 2026-10-22: 8x MI325X, gfx942, 2 TB disk, 1259 GB RAM, $30.4 per hour.

## Before the run

| Step | Command | What it proves | PASS number | On failure | GPU minutes |
|---|---|---|---|---|---|
| 1 | `nproc; free -g; df -h /` | the box is the one expected | 160 cores, 1200 GB RAM, 1 TB free | stop, wrong box | 0 |
| 2 | `apt install -y linux-firmware-amd-graphics && reboot` | amdgpu can load | 8 cards in `ls /dev/dri` | do not proceed | 0 |
| 3 | `ls /opt/rocm/bin/hipcc || which hipcc; hipcc --version` | the compiler exists | HIP 7.1 or newer | install rocm-dev | 0 |
| 4 | `echo '#include <hip/hip_runtime.h>
__global__ void k(){}' > /tmp/t.hip && hipcc --offload-arch=gfx942 -c /tmp/t.hip -o /tmp/t.o` | gfx942 code generation works | an object file exists | stop | 0 |
| 5 | `sha256sum -c shards.sha256` in the snapshot | the weights match the pinned tree | every shard OK | re-download only the failed shard | 0 |
| 6 | `cat REVISION` | the checkpoint revision is recorded | matches the pinned 40-hex sha | stop | 0 |
| 7 | `mount -t tmpfs -o size=700G tmpfs /ram` | staging writes to RAM, not disk | `df -h /ram` shows 700G | use disk and expect 10x slower staging | 0 |
| 8 | `python -m examples.glm53.stage --src <snapshot> --out /ram/run/stage --gate production --trust-verified` | all 45 layers staged | `STAGE layer=44` present | stop | 0 |
| 9 | `python -m examples.glm53.prepare --require-model --stage /ram/run/stage --gate production --batch N --context 1024 --gpu-gib 256 --out /ram/run/prepare.json` | the plan fits the card | status is STAGED-MODEL-VALIDATED | lower batch or raise gpu-gib | 0 |
| 10 | `ETX_GLM_BATCH=N ETX_GLM_GATE=production bash examples/glm53/build_model.sh /ram/run/stage /ram/run/build` | the binary matches the stage | `GLM-MODEL-BUILD OK` | stop | 0 |

## The run

| Step | Command | What it proves | PASS number | On failure | GPU minutes |
|---|---|---|---|---|---|
| 11 | `run stage build/load.tsv prompts.txt prefix` | memory fits both ranks | `GLM-ALLOCATED used=` under 250 GB per rank | lower batch | 2 |
| 12 | the same run, one step | one step completes | `GLM-MODEL-STEP ms=` printed | stop, read stderr | 1 |
| 13 | compare `.tokens` against the golden | correctness | 32 of 32 tokens equal per request | no timing reported | 2 |
| 14 | 31 steps, median and spread | the step time | median with spread, both recorded | rerun once | 26 |
| 15 | copy logs to the Mac before stopping the box | the evidence survives | files present on the Mac | keep the box up | 0 |

## Stop rules

Two failures of the same cause: stop and report.
90 minutes without a passing gate: stop and report with the evidence.
Never delete the box before the logs are on the Mac.

## Measured on 2026-10-22, for reference

Staging with tmpfs and `--trust-verified`: 10 s per layer, 8 minutes for 45 layers.
Staging on disk with the double hash: 108 s per layer, 81 minutes for 45 layers.
Full model, batch 1, PP2: 162.2 GB on rank 0, 160.8 GB on rank 1, 51.0 s per token.
Fabric: barrier 5.46 us clean, 15.62 us with a dirty L2.
