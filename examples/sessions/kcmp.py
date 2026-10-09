"""Per-kernel micro-session, only after nvref report and fetch.

Operator prerequisites: nvref report finished and fetched; retain its logs in sess.
Budget and VM-life limits: /Users/ckl/.hermes/cache/scratch/o55-kernel-compare.json:1,
result section 5: estimated five GPU minutes, ten VM minutes, $0.50; skip if
nvref ends after 45 minutes, and retain the operator's 55-minute VM hard stop.
This job file does not provision or extend the VM deadline. Historical traces on
VF hardware are not a matched baseline; the requested fixed job list excludes kc_vllm.

All numeric threshold statements below are verbatim from
/Users/ckl/.hermes/cache/scratch/o55-kernel-compare.json:1 (result, section 4).

Define the gain g = (ETX standalone median − best vendor standalone median), per layer; a positive g means the vendor kernel is faster.

- **Row closes, the tile is good enough:** g < 0.63 µs, which is 0.030 ms per token ÷ 48, the band an A/B cannot resolve (PLAN:154). For lm_head the limit is 30 µs.
- **Vendor wins:** g ≥ 0.63 µs in 3 of 3 pairs, and the median g is more than twice the spread (the rule at `examples/vllm_llm/pairs.py:33-34,138-143`). This is recorded only. Tuning the tile needs the owner's request (`AGENTS.md:7`).
- **Worth building link-mode import:** all three conditions must hold.
  - g ≥ 1.04 µs, which is 0.050 ms ÷ 48.
  - The vendor kernel's source is available.
  - The vendor time plus its import cost is at most the ETX in-situ body minus 1.04 µs. Measured import costs range from +1.04 µs (wvSplitK) to +6.62 µs (fused_moe up).

  Applied to the rows: qkv requires vendor + cost ≤ 8.30; o_proj ≤ 4.31; router ≤ 3.98; moe_up ≤ 16.53; moe_down ≤ 8.84; lm_head ≤ 106.7 per step [E].
- **Noise.** Session-to-session drift is at most 0.12 µs (A against C), so 0.63 µs is at least 5 times the drift, and 3 pairs is enough (the PLAN:154 minimum). A row with mixed signs closes unless the owner asks for 5 more pairs.

"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


S = '$HOME/sess'
LOGS = "${ETX_SESS_LOGDIR:-$HOME/sess}"
MODEL = '$HOME/llm/qwen3-30b-a3b'
IMAGE = 'vllm/vllm-openai-rocm:v0.27.1'
finished = []
for arm, rnd in [('etx', 1), ('vnd', 1), ('vnd', 2), ('etx', 2), ('etx', 3), ('vnd', 3)]:
    name = f'kc_{arm}_{rnd}'
    if arm == 'etx':
        cmd = (f'test ! -e {S}/{name} || {{ echo REFUSED: existing trace directory; exit 1; }}; '
               f'ETX_VL_PROGRESS=1 rocprofv3 --kernel-trace --output-format csv -d {S}/{name} -- '
               f'stdbuf -oL {S}/b_all/run {MODEL} {S}/b_all/vl_plan.txt --unfused --max-steps 8 --repeat 1')
    else:
        cmd = ('sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host '
               '--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 '
               f'-v $HOME:$HOME -w $HOME/etx --entrypoint python3 {IMAGE} '
               'examples/vllm_llm/vendor_gemv.py --lib hipblaslt,rocblas,tunable '
               f'--copies 48 --warmup 10 --iters 30 --check {S}/dump_all --model {MODEL} --out {S}/{name}')
    job(name=name, gpu=0, lane='gpu', after_any=list(finished), timeout=60 if arm == 'etx' else 180,
        quiet_s=60, cmd=cmd, summary=[r'^RESULT|^KC|REFUSED'])
    finished.append(name)
job(name='kc_report', lane='cpu', after_any=list(finished), timeout=60,
    cmd=f'.venv/bin/python -m examples.vllm_llm.kcmp_report {LOGS}', summary=[r'.'])
