"""Same-build import certification and host-wait diagnostic, one MI300X VF.

Qwen3-30B-A3B, batch 1, context 1024, 32 greedy tokens; numerics unchanged.
Pre-registration (before any GPU run):
* Every planned imported instance (expected 97, replicas included in each
  instance) must compare byte-identical at BOTH positions 0 and 1040. Zero
  missing/duplicate instances; 32/32 HF tokens and CHECK-DSL OK. Failure or
  refusal stops timing. Coverage is for these inputs/positions, not all prompts.
* The SAME CHECK executable must pass five interleaved fused/HIP-graph pairs:
  all wall gains positive, median > 2%, median > twice max-minus-min spread.
  Device span is secondary. Do not borrow timing from nvref/nvref2.
* Three separate polling-fused / blocking-fused / HIP-graph triplets diagnose
  ETX_SYNC=block; reverse the second round. Compare RESULT wall and device
  medians on the same decode population. A blocking wall saving > 0.050 ms,
  positive in all three rounds and > twice its spread, supports changing the
  host wait. Blocking host excess < 0.050 ms closes this host-cost question;
  otherwise it stays open. These are diagnostic thresholds, not fusion proof.
* A single optional vLLM default run is a contemporaneous system reference,
  not a five-round LEAD verdict or a search for vLLM's best configuration.
  Missing it leaves the real-system comparison NOT-COMPLETE.

Budget: 40 minutes maximum from runner start, enforced by MAX_SESSION_SECONDS,
including setup. At $1.99/GPU-hour that is about $1.33 (estimate, not a quote).
The operator must also stop/delete the VM by 40 minutes of allocation, including
any time BEFORE the runner starts, and delete immediately on completion. This
file does not provision/delete VMs. A cold setup can exhaust the budget; report
NOT-COMPLETE, never reduce the correctness gates or the five-pair requirement.

No warm artifacts required: dl/prep/golden_sha supply the model and golden;
build_chkx produces ~/sess/b_all, gate_chkx produces ~/sess/dump_all. Build
identity and check captures are archived in the runner's actual log directory.
Optional vLLM is LAST, enabled with ETX_NVCHK_VLLM=1; its image is pulled only
then, with enough budget left for a bounded container run. No image pull or GPU
access occurs on import or --dry-run.

Local inspection: python -m examples.sessions.nvchk --dry-run
"""
from copy import deepcopy

from examples.sessions import nvref2 as reference

MAX_SESSION_SECONDS = 40 * 60
LLM, M, S, LOGS = reference.LLM, reference.M, reference.S, reference.LOGS
V, ALL = reference.V, reference.ALL
CERT = '.venv/bin/python -m examples.vllm_llm.import_certificate'
BUILD = f'{S}/b_all'
MODEL = f'{LLM}/{M}'
JOBS = []


def job(**kw):
    kw.setdefault('machine', 'A')
    kw.setdefault('quiet_s', 180)
    JOBS.append(kw)


# Copy only inert setup definitions; never mutate either reference session.
for name in ('apt', 'etx_env', 'llm_env', 'dl', 'prep', 'record', 'golden_sha'):
    entry = deepcopy(next(j for j in reference.JOBS if j['name'] == name))
    job(**entry)

# ROCm images may install under core-<version>; missing metadata is informational.
record = next(j for j in JOBS if j['name'] == 'record')
record['cmd'] = record['cmd'].replace(
    'cat /opt/rocm/.info/version',
    'rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || '
    '{ newest=$(printf "%s\\n" /opt/rocm/core-*/.info/version | sort -V | tail -n 1); '
    'cat "$newest" 2>/dev/null; } || hipconfig --version 2>/dev/null || printf unknown); '
    'printf "ROCM-VERSION %s\\n" "$rocm_version"')

job(name='build_chkx', lane='cpu', reads=['etx_env', 'prep', 'golden_sha'],
    timeout=1200,
    cmd=f'set -euo pipefail; mkdir -p {S}; {V} ETX_VL_CHECK=1 '
        f"ETX_HIPCC_FLAGS='{ALL}' ETX_OUT={BUILD} bash examples/vllm_llm/build.sh {MODEL} "
        f'2>&1 | tee {LOGS}/build_chkx.txt; '
        f'{CERT} --export-build {BUILD} --model {MODEL} --session {LOGS}',
    summary=[r'tasks=|megakernel|built|REFUSED'])


def capture(extra='', sync=''):
    # Set the wait mode explicitly; an inherited ETX_SYNC must not contaminate
    # the primary experiment. Reject debug graph overrides through the parser.
    return (f'ETX_SYNC={sync} ETX_VL_PROGRESS=1 {CERT} '
            f'--run-build {BUILD} --model {MODEL} -- {extra}')


job(name='gate_chkx', gpu=0, reads=['build_chkx', 'record'], timeout=600,
    cmd=f'set -euo pipefail; rm -rf {S}/dump_all; mkdir -p {S}/dump_all; '
        f'ETX_DUMP={S}/dump_all ETX_DUMP_STEP=1040 ETX_TOKTRACE={S}/tok_all.txt '
        + capture('--repeat 1 --check-dsl') + f' 2>&1 | tee {LOGS}/gate_chkx.txt; '
        f"grep -q '^RESULT .*matched=32 gen=32' {LOGS}/gate_chkx.txt; "
        f"grep -q '^CHECK-DSL OK' {LOGS}/gate_chkx.txt; "
        f'test -s {S}/dump_all/logits.bin; echo GATE-OK', summary=[reference.RES])
job(name='gate_unf', gpu=0, reads='gate_chkx', timeout=600,
    cmd='set -euo pipefail; ' + capture('--unfused --repeat 1')
        + f' 2>&1 | tee {LOGS}/gate_unf.txt; '
        f"grep -q '^RESULT .*matched=32 gen=32' {LOGS}/gate_unf.txt; echo GATE-OK",
    summary=[reference.RES])
job(name='check_timed', gpu=0, reads='gate_unf', timeout=600,
    cmd='set -euo pipefail; ' + '; '.join(
        capture(f'--check 48 --check-pos {pos}') + f' 2>&1 | tee {LOGS}/check_p{pos}.txt'
        for pos in (0, 1040)),
    summary=[r'^CHECK|^IMPORT-CERTIFICATE'])

job(name='check_certificate', lane='cpu', reads='check_timed', timeout=120,
    cmd=f'{CERT} {LOGS} --require-positions 0,1040 --checks-only', summary=[r'.'])

finished = ['check_certificate']
pair_logs = []
for rnd in range(1, 6):
    for arm in (('etx', 'etxu') if rnd % 2 else ('etxu', 'etx')):
        name = f'{arm}_r{rnd}'
        job(name=name, gpu=0, reads=list(finished), timeout=300,
            cmd='set -euo pipefail; ' + capture('--unfused' if arm == 'etxu' else ''),
            summary=[r'^RESULT|BUILD-IDENTITY'])
        pair_logs.append(f'{LOGS}/{name}.log')
        finished = [name]

# Always report incomplete primary evidence; diagnostic jobs depend on the
# correctness check, not on whether the fusion performance threshold passed.
primary = [j['name'] for j in JOBS]
job(name='report', lane='cpu', reads=['?' + n for n in primary], timeout=120,
    cmd=f'set -o pipefail; {CERT} {LOGS} --require-positions 0,1040; cert_rc=$?; '
        'if [ "$cert_rc" -gt 1 ]; then exit "$cert_rc"; fi; '
        f".venv/bin/python -m examples.vllm_llm.pairs {' '.join(pair_logs)} --pairs 5 --basis wall; "
        'pairs_rc=$?; test "$cert_rc" -eq 0 && test "$pairs_rc" -eq 0', summary=[r'.'])
finished = ['report']
for rnd in range(1, 4):
    arms = ('poll', 'block', 'graph') if rnd % 2 else ('graph', 'block', 'poll')
    for arm in arms:
        name = f'{arm}_r{rnd}'
        job(name=name, gpu=0, reads=['check_certificate', *('?' + n for n in finished)], timeout=300,
            cmd='set -euo pipefail; ' + capture('--unfused' if arm == 'graph' else '',
                                              sync='block' if arm == 'block' else ''),
            summary=[r'^RESULT|BUILD-IDENTITY'])
        finished.append(name)
job(name='sync_report', lane='cpu', reads=['?' + n for n in finished], timeout=120,
    cmd=f'.venv/bin/python -m examples.sessions.nvchk --report-sync {LOGS}', summary=[r'.'])

# Reserve enough time BEFORE pulling. The timeout inside the container also
# bounds GPU work if its docker client dies; runner hard-stop cannot kill a
# daemon-owned container by killing the client's process group alone.
optional = reference.vllm('vllm_last').replace('--entrypoint python3', '--entrypoint timeout')
optional = optional.replace(' examples/llm/vllm_bench.py', ' -k 5 300 python3 examples/llm/vllm_bench.py')
optional = optional.replace(f'{S}/vllm_last.json', f'{LOGS}/vllm_last.json')
job(name='vllm_last', gpu=0, reads=['golden_sha', 'check_certificate', '?sync_report'],
    timeout=600,
    cmd='set -euo pipefail; '
        'if [ "${ETX_NVCHK_VLLM:-0}" != 1 ]; then echo "VLLM NOT-COMPLETE: optional arm disabled"; exit 0; fi; '
        'if [ "${ETX_SESS_REMAINING_SECONDS:-0}" -lt 610 ]; then '
        'echo "VLLM NOT-COMPLETE: insufficient remaining budget"; exit 0; fi; '
        f'timeout 180 sudo docker pull {reference.IMAGE}; '
        f'sudo docker inspect --format "{{{{index .RepoDigests 0}}}}" {reference.IMAGE}; '
        + optional, summary=[r'^VLLM|sha256:'])


def report_sync(path):
    """Read only this session's complete, same-build triplets."""
    import json
    from pathlib import Path
    from statistics import median
    from examples.vllm_llm import import_certificate as cert
    from examples.vllm_llm.nvref_report import correctness
    from examples.vllm_llm.pairs import Refusal, parse

    path = Path(path)
    try:
        cert.validate(path)  # validates all primary evidence, even if not eligible
        identity = json.loads((path / 'build_identity.json').read_text())
        statuses = {r['name']: r['status'] for r in json.loads((path / 'status.json').read_text())}
        rows = []
        gpus = set()
        for rnd in range(1, 4):
            runs = {}
            for arm in ('poll', 'block', 'graph'):
                name = f'{arm}_r{rnd}'
                if statuses.get(name) != 'ok':
                    raise Refusal(f'{name}: incomplete')
                text = (path / f'{name}.log').read_text()
                cert.stamped(text, identity)
                correctness(text, name)
                parsed = parse(text.splitlines()).runs
                if len(parsed) != 1 or parsed[0].arm != ('graph' if arm == 'graph' else 'fused'):
                    raise Refusal(f'{name}: wrong arm')
                runs[arm] = parsed[0]
                gpus.add(parsed[0].gpu)
            p, b, g = (runs[a] for a in ('poll', 'block', 'graph'))
            saving = p.wall_ms - b.wall_ms
            excess = (b.wall_ms - b.device_ms) - (g.wall_ms - g.device_ms)
            rows.append((saving, excess))
            print(f'SYNC round={rnd} wall_saving_ms={saving:.6f} block_host_excess_ms={excess:.6f} '
                  f'poll_device_ms={p.device_ms:.6f} block_device_ms={b.device_ms:.6f} graph_device_ms={g.device_ms:.6f}')
        if len(gpus) != 1 or None in gpus or 'None' in gpus:
            raise Refusal('triplets must share one GPU')
        savings, excesses = zip(*rows)
        med, spread, excess = median(savings), max(savings) - min(savings), median(excesses)
        supported = min(savings) > 0 and med > .050 and med > 2 * spread
        print(f'SYNC supported={supported} median_saving_ms={med:.6f} spread_ms={spread:.6f} '
              f'block_host_excess_ms={excess:.6f} host_question={"closed" if excess < .050 else "open"}')
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'SYNC NOT-COMPLETE: {exc}')
        return 2


if __name__ == '__main__':
    import argparse
    from etx.tools.gpu_jobs import main

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--report-sync')
    args = parser.parse_args()
    raise SystemExit(report_sync(args.report_sync) if args.report_sync else main([__file__, '--dry-run']))
