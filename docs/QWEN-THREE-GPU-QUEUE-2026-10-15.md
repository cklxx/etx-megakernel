# GPU-only queue for the three Qwen preparations

NOT AUTHORIZED TO RUN. Exactly three existing paths; no aggregate runner or new
experiment. CPU simulations, local compiles, archives and dry-runs are preparation,
not GPU experiments. Review blockers must close before publication; a separate
GPU-spend decision must approve allocation and real inputs before any session.

Each session owns its original fixed deadline across retries, fresh output paths,
full raw logs, true exits and evidence reserve. Never extend an expired lease.
Run separately, on one MI300X with the pinned native toolchain/inputs. Keep every
GPU of an authorized VM busy under project policy, but do not invent another
experiment to fill a lane. Stop on missing gates, bad identity, incorrect payload,
resource failure, incomplete pairs or timeout. Do not report partial runs as numbers.

| Order | Existing job | Bound and estimated cost | Required inputs / next decision |
| --- | --- | --- | --- |
| 1 | `examples/sessions/qwen_producer.py` | Lifecycle <=60 min: staging <=300 s, runner <=3300 s including 300 s evidence reserve; job hard maximum 3600 s. Cost <=1 hour times actual hourly rate. | Independent Python >=3.11 env with NumPy/PyYAML, packed Qwen data + model/hf, golden, qwen-g6 JSON, pinned vLLM image/checkout. Full-model correctness then incremental wall/fusion/S3 gates. |
| 2 | `examples/sessions/attention_early_merge.py` | <=2400 s /40 min INCLUDING build/setup; cost <=2/3 hour times rate. | Independent Python env, native ROCm libs, pinned ETX_VLLM, fresh absolute C6_WORK, packed C6_MODEL, all four positions. Correctness/resource/residency then 50 us local screen; pass permits later full-model integration, not wall claim. |
| 3 | `examples/sessions/qwen_expert_merge.py` | <=1800 s /30 min PER ISOLATED INPUT; cost <=1/2 hour times rate. | Default synthetic screen. Optional validated EEM_INPUT pack; fresh ETX_SESS_LOGDIR and independent Python env. Pass synthetic before separately budgeted real inputs/capture. Eight real role/position inputs are not hidden in one 30 min cap. |

No hourly price was queried. These are per-session estimates, not an approved
combined allocation or spend. No GPU was rented or used by preparation.

After separate authorization and staging the committed source with
`tools/session.sh push`, attach the supplied independent environment as documented
in the producer runbook. Do not rely on a checkout-local `.venv` surviving archive.
Do not install or download dependencies during the bounded session.

Producer launch from an operator machine (ORIGINAL_UNIX is the unchanged allocation
deadline; no restart may replace it):

```sh
python3 -m tools.producer_start IP --venv /absolute/independent-env \
  --config /absolute/qwen-g6.json --allocation-deadline ORIGINAL_UNIX
```

C6 launch ON the staged GPU machine; tools/session.sh does not forward C6_WORK or
C6_MODEL. Export the supplied environment so `python` resolves there:

```sh
export PATH=/absolute/independent-env/bin:$PATH
ETX_VLLM=/absolute/pinned-vllm C6_WORK=/absolute/fresh/c6-work \
C6_MODEL=/absolute/packed/qwen python -m etx.tools.gpu_jobs \
  examples/sessions/attention_early_merge.py --gpus 0 --logdir /absolute/fresh/c6-logs
```

Expert launch ON the staged GPU machine (python3 must resolve in the supplied env):

```sh
export PATH=/absolute/independent-env/bin:$PATH
ETX_SESS_LOGDIR=/absolute/fresh/expert-logs python3 -m etx.tools.gpu_jobs \
  examples/sessions/qwen_expert_merge.py --gpus 0 --logdir /absolute/fresh/expert-logs
```

Set `EEM_INPUT=/absolute/validated/pack` only for an independently budgeted real
input. The capture/attest CLI accepts the strict staged SOURCE_IDENTITY receipt;
missing/malformed provenance must fail. Capture is not implicitly in a synthetic
screen's budget. Full model follow-up needs original-allocation checks, imports,
DSL, 32/32 tokens, same-kernel graph and strongest same-session vLLM baseline.

All three paths keep normal production defaults OFF. Delete a future rented VM
as soon as its approved plan ends and verify no VM remains. No such action is
performed by this document or local integration.
