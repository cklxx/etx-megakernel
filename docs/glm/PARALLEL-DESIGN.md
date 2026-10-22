# GLM parallel design review

## Contents
- [Steps](#steps)
- [Conditions](#conditions)
- [Commands](#commands)

## Steps
| step | command | what it proves | PASS number | failure action | GPU minutes |
| --- | --- | --- | --- | --- | --- |
| 1 Ownership | [C1](#c1) | Move destinations, immutable bytes, historical aliases, relative links and snapshot hashes | 5 tests passed | Stop; repair references without changing evidence bytes | 0 |
| 2 Configuration | [C2](#c2) | Pinned revision, layer counts, tensor dimensions and mixed-precision checkpoint bytes | 8 assertions; 328250014584 bytes [E] | Stop; reconcile the pinned configuration | 0 |
| 3 Analysis | [C3](#c3) | Opens the preserved comparison, capacity, critical-path and evolution reference for cc-o55 review | U (unverified): review acceptance has not been recorded | Record rejected claim and source line; do not substitute a GPU result | 0 |
| 4 Local contracts | [C4](#c4) | GLM contracts and ownership checks; not GPU qualification | 232 passed; 17 skipped in the recorded local run | Stop on failures; report skips without treating them as passes | 0 |
| 5 S0 PP2 | U (unverified): no qualified MI325X execution command is recorded here | Full-model PP2 correctness; rank memory fit and host-wall timing | U (unverified): 32/32 tokens per request at B1/2/4/8 is the target [E] | Stop GPU qualification; retain logs and the PP2 baseline | U (unverified): no measured runtime |
| 6 S1 Rank parallelism | U (unverified): tile decomposition is not implemented | Same-numerics compute scaling within a rank; exact weight, state and workspace ownership | U (unverified): paired compute and memory measurements are missing | Retain whole-layer tiles; reject peak-bandwidth division as evidence | U (unverified): no measured runtime |
| 7 S2 Collective gate | U (unverified): production-shaped collective benchmark is missing | Completed-transfer all-reduce and sparse all-to-all progress under concurrent compute on four GPUs | U (unverified): latency, bandwidth, skew and tail measurements are missing | Block S4; do not substitute the historical two-GPU seam | U (unverified): no measured runtime |
| 8 S3 Pipeline capacity | U (unverified): arbitrary PP rank support is not implemented | PP2/4/8 layer ownership, memory fit, token feedback and independent-request overlap | U (unverified): numerical and pipeline-bubble measurements are missing | Retain PP2; reject same-request next-token overlap | U (unverified): no measured runtime |
| 9 S4 TP and EP | U (unverified): TP and EP execution paths are not implemented | Tensor sharding and expert sharding with bounded buffers and in-kernel communication | U (unverified): correctness and paired performance measurements are missing | Retain smaller fitting groups; reject per-op host dispatch | U (unverified): no measured runtime |
| 10 S5 Combinations | U (unverified): multi-axis placement is not implemented | PP+TP, TP+EP and DP comparisons at identical global batch and latency objective | U (unverified): no measured B1 or B8 winner | Retain qualified configuration; reject unequal global-batch comparisons | U (unverified): no measured runtime |
| 11 S6 Service | U (unverified): eight-GPU service is not qualified | Oversized-model capacity, cache isolation, sustained throughput and reference correctness | U (unverified): service acceptance and runtime evidence are missing | Stop rollout; retain all-rank diagnostics | U (unverified): no measured runtime |

## Conditions
| scope | condition |
| --- | --- |
| All commands | Local checkout at `$HOME/code/etx`; commands are read-only except test temporary files; no GPU provisioning or execution |
| C1 and C4 | Existing `.venv/bin/python` with pytest; stop on a missing interpreter |
| C2 and C3 | `python3` available; C3 reads `docs/glm/PARALLEL-ANALYSIS.md`, the preserved reference, not this checklist |
| Evidence | [E] means estimate; U (unverified) means missing evidence; target counts are not observed passes |
| Terminology | PP means pipeline parallelism; TP means tensor parallelism; EP means expert parallelism; SP means sequence parallelism; DP means data parallelism |
| Recorded skips | 6 tests require accelerate; 11 tests require local gfx942 clang, ROCm headers and device libraries |
| Recorded baseline failures | Two focused DeepSeek kernel-hash tests fail identically before and after the ownership move; expected hashes remain unchanged |
| S0 | Exactly two visible GPUs; pinned release checkpoint; reference comparison; explicit GPU authorization and budget |
| S1 | S0 qualified; unchanged numerical contract; same-weight paired comparison |
| S2 | Explicit GPU authorization and budget; four-GPU collective measurement before S4; eight-GPU topology measurement before eight-way claims |
| S3 | Exact per-rank allocation reserve; independent requests for pipeline overlap; numerical checks across cuts |
| S4 | S2 completed; compute savings exceed communication, layout, imbalance and added host cost with measured margin |
| S5 | Fixed global B1/2/4/8, context and generated length; same latency objective; compare DP4×TP2 against TP8, EP8, TP2+EP4 and PP2+TP4 |
| S6 | All-rank memory fit and progress; pinned model identity; no hidden offload; same-kernel and strongest-system baselines |
| Historical calibration | 8.61 us is an intra-GPU barrier; 285 GiB/s is a workgroup-0 peer-store timing proxy; neither measures MI325X collectives |
| Capacity | 305.71 GiB [E] is mixed-precision checkpoint inventory; 256 GB equals 238.418579 GiB [E]; workspace and allocation reserve remain unverified |
| B1 and B8 choices | TP is the B1 candidate [E]; DP4×TP2 is the eight-GPU B8 candidate [E]; measured winners remain U (unverified) |

## Commands
### C1
```sh
(cd "$HOME/code/etx" && .venv/bin/python -m pytest -q tests/test_documentation_ownership.py)
```
### C2
```sh
(cd "$HOME/code/etx" && python3 - <<'PY'
import json
from pathlib import Path
root = Path('docs/evidence/qwen-batch-glm53')
t = json.loads((root / 'glm-config.json').read_text())['text_config']
a = json.loads((root / 'glm-model-api.json').read_text())
assert a['sha'] == '46a050bec13002ad29ee795c16e61cf4fa3c4a9c'
assert (t['num_hidden_layers'], t['hidden_size'], t['hc_mult']) == (45, 4096, 4)
assert (t['n_routed_experts'], t['num_experts_per_tok'], t['n_shared_experts']) == (288, 8, 1)
assert (t['intermediate_size'], t['moe_intermediate_size']) == (12288, 2048)
assert (t['layer_types'].count('linear_attention'), t['layer_types'].count('deepseek_sparse_attention')) == (34, 11)
assert (t['mlp_layer_types'].count('dense'), t['mlp_layer_types'].count('sparse')) == (3, 42)
assert (t['kv_lora_rank'], t['q_lora_rank'], t['num_attention_heads'], t['qk_nope_head_dim'], t['v_head_dim']) == (512, 1536, 64, 256, 256)
p = a['safetensors']['parameters']
weight_bytes = p['F8_E4M3'] + 2 * p['BF16'] + 4 * p['F32']
assert weight_bytes == 328250014584
print('PASS: 8 configuration assertions; weight_bytes =', weight_bytes, '[E]')
PY
)
```
### C3
```sh
(cd "$HOME/code/etx" && python3 - <<'PY'
from pathlib import Path
print(Path('docs/glm/PARALLEL-ANALYSIS.md').read_text(), end='')
PY
)
```
### C4
```sh
(cd "$HOME/code/etx" && .venv/bin/python -m pytest -q -rs tests/test_glm53_*.py tests/test_documentation_ownership.py)
```
