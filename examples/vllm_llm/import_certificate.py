"""Fail-closed import evidence for one executable and the exercised positions.

Build: --write-build B --model M (called by build.sh).
Capture: --run-build B --model M -- <host arguments>; fingerprints are verified
before AND after execution. Only successful runs receive the closing stamp.
Archive: --export-build B --model M --session S copies the plan/import metadata.
Validate: S --require-positions 0,1040 [--checks-only]. No historical timing is
borrowed. This certifies exercised launch instances/inputs, not all prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from examples.vllm_llm.pairs import Refusal, parse, pairs, verdict


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan_requests(build: Path):
    """The request count the plan header publishes (`total .. requests <n>`); plans without one serve one request.
    A missing plan is left to the fingerprint, which refuses it."""
    plan = build / 'vl_plan.txt'
    if not plan.is_file():
        return 1
    for line in plan.read_text().splitlines():
        if line.startswith('total '):
            words = line.split()
            if 'requests' in words:
                n = int(words[words.index('requests') + 1])
                if n < 1:
                    raise Refusal('plan header publishes no request')
                return n
            return 1
    return 1


def request_goldens(requests):
    """Request r's golden: the base prompt rotated by 97 r tokens (examples/llm/prep.py --rotate)."""
    return ['golden_c1024.txt'] + [f'golden_c1024_r{97 * r}.txt' for r in range(1, requests)]


def fingerprint(build: Path, model: Path):
    paths = {'executable': build / 'run', 'plan': build / 'vl_plan.txt',
             'imports': build / 'imports.bc', 'import_metadata': build / 'imp/imports.json',
             'plan_header': build / 'plan_data.h', 'plan_json': build / 'plan.json',
             'weights_manifest': model / 'vl.manifest', 'golden': model / 'golden_c1024.txt',
             'compiler_output': build / 'hipcc.log'}
    if (build / 'hipcc_command.json').exists():
        paths['compiler_command'] = build / 'hipcc_command.json'
    # A multi-request plan reads one golden and one HF prefill reference per request; one-request
    # identities are unchanged.
    requests = plan_requests(build)
    if requests > 1:
        for name in request_goldens(requests):
            paths[f'request_golden/{name}'] = model / name
            prefill = name.replace('golden', 'prefill', 1)
            paths[f'request_prefill/{prefill}'] = model / prefill
    sources = sorted(build.glob('*.hip'))
    for required in ('megakernel_d0.hip', 'dsl_tiles.hip', 'imp_tiles.hip'):
        if build / required not in sources:
            raise Refusal(f'missing generated source {required}')
    paths.update({f'device_source/{p.name}': p for p in sources})
    paths.update({f'original/{p.name}': p for p in sorted((build / 'imp').glob('orig*')) if p.is_file()})
    return {name: sha(path) for name, path in paths.items()}


def write_build(build, model):
    identity = dict(schema=1, check_enabled=os.environ.get('ETX_VL_CHECK') == '1',
                    hipcc_flags=os.environ.get('ETX_HIPCC_FLAGS', ''),
                    compiler=(build / 'compiler_version.txt').read_text().strip(),
                    overrides={k: v for k, v in os.environ.items()
                               if k.startswith(('ETX_VL_', 'ETX_DSL_', 'ETX_TOPK_'))
                               or k in ('ETX_PREFETCH_ISOLATE_POLLER', 'ETX_DEV_RESET')},
                    files=fingerprint(build, model))
    (build / 'build_identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    return identity


def verified_build(build, model):
    identity = json.loads((build / 'build_identity.json').read_text())
    if identity['files'] != fingerprint(build, model):
        raise Refusal('build/model fingerprint changed')
    return identity


def capture(build, model, args):
    # The wrapper fixes executable, plan, model and golden; alternate golden or
    # truncated/teacher-forced timing must not inherit their fingerprints.
    allowed = {'--check', '--check-pos', '--repeat', '--unfused', '--check-dsl', '--price-attention', '--check-coverage'}
    for arg in args:
        if arg.startswith('-') and arg not in allowed:
            raise Refusal(f'unsupported certified host option {arg}')
    identity = verified_build(build, model)
    # Multi-request plans: the wrapper, not the caller, names each request's fingerprinted golden.
    requests = plan_requests(build)
    if requests > 1:
        args = list(args)
        for r, name in enumerate(request_goldens(requests)):
            args += ['--golden' + (str(r) if r else ''), name]
    tag = digest(identity)
    print(f'BUILD-IDENTITY phase=before sha256={tag}', flush=True)
    print('ETX_ENV ETX_UNFUSED_SYNC=' + os.environ.get('ETX_UNFUSED_SYNC', '')
          + ' ETX_UNFUSED_NOGRAPH=' + os.environ.get('ETX_UNFUSED_NOGRAPH', ''), flush=True)
    rc = subprocess.call(['stdbuf', '-oL', str(build / 'run'), str(model), str(build / 'vl_plan.txt'), *args])
    if rc:
        return rc
    if verified_build(build, model) != identity:
        raise Refusal('build identity changed during run')
    print(f'BUILD-IDENTITY phase=after sha256={tag}', flush=True)
    return 0


def export_build(build, model, session):
    verified_build(build, model)
    session.mkdir(parents=True, exist_ok=True)
    for src, dst in [('build_identity.json', 'build_identity.json'), ('vl_plan.txt', 'vl_plan.txt'),
                     ('imp/imports.json', 'imports.json')]:
        shutil.copyfile(build / src, session / dst)


def expected_instances(plan, imports):
    rows = []
    all_ids = set()
    for line in plan.splitlines():
        if not line or line.startswith(('#', 'total ')):
            continue
        r = line.split()
        n = int(r[0])
        if n in all_ids:
            raise Refusal('duplicate plan instance ID')
        all_ids.add(n)
        if r[1].startswith('dsl_'):
            continue
        # The plan's trailing columns are rep stride copyoff req nb; old plans
        # without them belong to request 0.
        rows.append(dict(id=n, export=r[1], layer=int(r[2]), replicas=int(r[12]),
                         req=int(r[15]) if len(r) > 15 else 0,
                         grid=list(map(int, r[4:7])), block=imports[r[1]]['block']))
    if not rows:
        raise Refusal('zero planned imports')
    return rows


def stamped(text, identity):
    stamps = re.findall(r'^BUILD-IDENTITY phase=(\w+) sha256=(\S+)$', text, re.M)
    tag = digest(identity)
    if stamps != [('before', tag), ('after', tag)]:
        raise Refusal('missing or mismatched build fingerprints')
    lines = text.splitlines()
    before = lines.index(f'BUILD-IDENTITY phase=before sha256={tag}')
    after = lines.index(f'BUILD-IDENTITY phase=after sha256={tag}')
    if any(line.startswith(('CHECK', 'RESULT ')) for line in lines[:before] + lines[after + 1:]):
        raise Refusal('evidence outside fingerprint envelope')


def check_output(text, expected, position, identity):
    stamped(text, identity)
    records = [json.loads(line[len('CHECK-INSTANCE '):]) for line in text.splitlines()
               if line.startswith('CHECK-INSTANCE ')]
    if [r['id'] for r in records] != [r['id'] for r in expected]:
        raise Refusal(f'position {position}: incomplete/duplicate/out-of-order instances {len(records)}/{len(expected)}')
    for r, want in zip(records, expected):
        # Records written before host.hip published req belong to request 0.
        if 'req' not in r and want.get('req') == 0:
            want = {k: v for k, v in want.items() if k != 'req'}
        if any(r.get(k) != v for k, v in want.items()):
            raise Refusal(f'position {position}: instance metadata mismatch')
        if r.get('position') != position or type(r.get('bytes')) is not int or r['bytes'] <= 0:
            raise Refusal('invalid position/compared-byte count')
        if r.get('result') != 'identical' or r.get('first_byte') != -1:
            raise Refusal('failed byte comparison')
    summary = re.findall(r'^CHECK-IMPORTED checked=(\d+) planned=(\d+) replicas=(\d+) position=(\d+)$', text, re.M)
    want = tuple(map(str, (len(expected), len(expected), sum(r['replicas'] for r in expected), position)))
    if summary != [want]:
        raise Refusal('missing or inconsistent coverage summary')
    if re.findall(r'^CHECK (?:OK|FAILED):.*$', text, re.M) != ['CHECK OK: 0 launch(es) differ']:
        raise Refusal('missing or failed CHECK result')
    if re.search(r'DIFFERS|^ERROR:|^HANG ', text, re.M):
        raise Refusal('failed checker output')


def validate(session, require_positions=(0, 1040), timing=True):
    identity = json.loads((session / 'build_identity.json').read_text())
    required = {'executable', 'plan', 'imports', 'import_metadata', 'plan_header', 'plan_json',
                'weights_manifest', 'golden', 'compiler_output', 'device_source/megakernel_d0.hip',
                'device_source/dsl_tiles.hip', 'device_source/imp_tiles.hip'}
    files = identity.get('files', {})
    if (identity.get('schema') != 1 or identity.get('check_enabled') is not True
            or not identity.get('compiler') or not required <= files.keys()
            or any(not re.fullmatch('[0-9a-f]{64}', v) for v in files.values())):
        raise Refusal('incomplete CHECK build identity')
    if sha(session / 'vl_plan.txt') != files['plan'] or sha(session / 'imports.json') != files['import_metadata']:
        raise Refusal('archived plan/import metadata hash mismatch')
    expected = expected_instances((session / 'vl_plan.txt').read_text(), json.loads((session / 'imports.json').read_text()))
    if not require_positions or len(set(require_positions)) != len(require_positions) or min(require_positions) < 0:
        raise Refusal('invalid required positions')
    statuses = {r['name']: r['status'] for r in json.loads((session / 'status.json').read_text())} if (session / 'status.json').exists() else {}
    def read(name, suffix='.log'):
        if name in statuses and statuses[name] != 'ok':
            raise Refusal(f'{name}: job {statuses[name]}')
        return '\n'.join(s for s in (session / (name + suffix)).read_text().splitlines() if not s.startswith('#'))
    if 'check_timed' in statuses and statuses['check_timed'] != 'ok':
        raise Refusal('check_timed did not complete')
    for pos in require_positions:
        check_output(read(f'check_p{pos}', '.txt'), expected, pos, identity)
    result = dict(checked=len(expected), planned=len(expected), positions=list(require_positions),
                  build_sha256=digest(identity), eligible=False)
    if timing:
        # Import lazily: nvref_report also calls this validator.
        from examples.vllm_llm.nvref_report import correctness
        for name in ('gate_chkx', 'gate_unf'):
            text = read(name)
            stamped(text, identity)
            correctness(text, name)
            if name == 'gate_chkx' and not re.search(r'^CHECK-DSL OK\b', text, re.M):
                raise Refusal('missing CHECK-DSL gate')
        lines = []
        for rnd in range(1, 6):
            for arm, mode in [('etx', 'fused'), ('etxu', 'graph')]:
                name = f'{arm}_r{rnd}'
                text = read(name)
                stamped(text, identity)
                correctness(text, name)
                parsed = parse(text.splitlines())
                if len(parsed.runs) != 1 or parsed.runs[0].arm != mode:
                    raise Refusal('wrong certified timing arm')
                # Preserve original runner headers for the single-GPU check.
                lines += (session / f'{name}.log').read_text().splitlines()
        ps = pairs(parse(lines), 5)
        result['eligible'] = verdict([(u.wall_ms - f.wall_ms) / u.wall_ms for f, u in ps])['established']
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session', type=Path, nargs='?')
    ap.add_argument('--require-positions', default='0,1040')
    ap.add_argument('--checks-only', action='store_true')
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--write-build', type=Path)
    modes.add_argument('--run-build', type=Path)
    modes.add_argument('--export-build', type=Path)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--session', dest='export_session', type=Path)
    # Host flags follow -- and cannot be confused with certificate options.
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    host = args[args.index('--') + 1:] if '--' in args else []
    args = args[:args.index('--')] if '--' in args else args
    a = ap.parse_args(args)
    try:
        if a.write_build:
            write_build(a.write_build, a.model)
        elif a.run_build:
            return capture(a.run_build, a.model, host)
        elif a.export_build:
            export_build(a.export_build, a.model, a.export_session)
        else:
            result = validate(a.session, tuple(map(int, a.require_positions.split(','))), not a.checks_only)
            print('IMPORT-CERTIFICATE ' + json.dumps(result, sort_keys=True))
            return 0 if a.checks_only or result['eligible'] else 1
    except (Refusal, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f'IMPORT-CERTIFICATE REFUSED: {exc}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
