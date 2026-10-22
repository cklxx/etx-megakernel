"""Offline GLM Cartesian build sweep. COMPILED is not correct, fast or selected.

No GPU, containers, downloads, weights (by default), or performance measurements.
All compiler output is retained. Every rank must compile and report resources.
"""
import argparse
from dataclasses import fields
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import yaml

from examples.glm53 import model_build, options, stage

ROOT = Path(__file__).resolve().parents[2]
UNRECORDED_ENV = ('HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS',
                  'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'CCC_OVERRIDE_OPTIONS')


def parse_report(text, kernel):
    """Read AMDGPU metadata plus LLVM's *static* occupancy report, fail closed."""
    match = re.search(r'\.amdgpu_metadata\s*\n(.*?)\.end_amdgpu_metadata', text, re.S)
    if not match:
        raise ValueError('missing AMDGPU compiler metadata')
    data = yaml.safe_load(match.group(1).strip())
    records = [r for r in data.get('amdhsa.kernels', []) if r.get('.name') == kernel]
    if len(records) != 1:
        raise ValueError(f'missing/ambiguous compiler kernel {kernel}')
    r = records[0]
    names = {'vgpr': '.vgpr_count', 'sgpr': '.sgpr_count', 'vgpr_spills': '.vgpr_spill_count',
             'sgpr_spills': '.sgpr_spill_count', 'scratch_bytes_per_workitem': '.private_segment_fixed_size',
             'lds_bytes_per_workgroup': '.group_segment_fixed_size', 'wavefront_size': '.wavefront_size'}
    result = {'kernel': kernel, 'basis': 'actual compiler metadata; static resource estimate, NOT measured occupancy'}
    for name, key in names.items():
        if type(r.get(key)) is not int or r[key] < 0:
            raise ValueError(f'missing/invalid compiler field {key}')
        result[name] = r[key]
    # Kernel-info blocks follow .size and can precede other helper functions.
    blocks = re.split(r'; Kernel info:', text)
    occupancy = None
    for before, after in zip(blocks, blocks[1:]):
        sizes = re.findall(r'\.size\s+([\w.$]+)\s*,', before)
        if sizes and sizes[-1] == kernel:
            found = re.search(r';\s*Occupancy:\s*(\d+)', after)
            if found:
                occupancy = int(found.group(1))
    if occupancy is None:
        raise ValueError('missing compiler waves/SIMD occupancy report')
    result['compiler_waves_per_simd'] = occupancy
    return result


def feasibility(report, config, min_workgroups_per_cu=1):
    """Static limits only; LDS includes both native staging and ETX runtime LDS."""
    if report['wavefront_size'] != 64:
        raise ValueError('gfx942 wavefront must be 64')
    lds = report['lds_bytes_per_workgroup']
    if lds > 65536:
        raise ValueError(f'actual compiler LDS {lds} exceeds 65536 bytes/workgroup')
    if not 0 < report['compiler_waves_per_simd'] <= 8:
        raise ValueError('no feasible compiler wave occupancy')
    if report['vgpr'] > 512:
        raise ValueError('compiler VGPR count exceeds gfx942 register file')
    # A static upper bound, not actual achieved or measured occupancy. Include
    # the machine model max_workgroups_per_CU=2 and 256-byte LDS granularity.
    waves_per_wg = config.threads // 64
    limits = [2, report['compiler_waves_per_simd'] * 4 // waves_per_wg]
    if lds:
        limits.append(65536 // (((lds + 255) // 256) * 256))
    bound = min(limits)
    report['estimated_workgroups_per_cu_upper_bound'] = bound
    report['requested_waves_per_simd_max'] = config.waves_per_simd_max
    report['planner_vgpr_budget_estimate'] = config.vgpr_budget
    if bound < min_workgroups_per_cu:
        raise ValueError(f'estimated residency upper bound {bound} below required {min_workgroups_per_cu}')
    return report


def run_command(command, log, timeout, env):
    """Full unfiltered stdout/stderr with deadline; no shell or output filtering."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('compiler timeout must be finite and positive')
    print(f'COMPILE {log.name}: {shlex.join(map(str, command))}', flush=True)
    with log.open('w') as stream:
        stream.write('$ ' + shlex.join(map(str, command)) + '\n'); stream.flush()
        proc = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except BaseException as exc:
            import signal
            try:os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                stream.write(f'\nTIMEOUT after {timeout}s\n')
                raise RuntimeError(f'compiler timeout; full log {log}') from exc
            raise
    if rc:
        raise RuntimeError(f'compiler exit {rc}; full log {log}')


def compile_candidate(build, config, llvm, headers, devlibs, timeout=180, host_syntax=False, min_workgroups_per_cu=1):
    env = dict(os.environ)
    bad = [name for name in UNRECORDED_ENV if env.get(name)]
    if bad:
        raise ValueError(f'unrecorded compiler environment: {bad}')
    env.update(ETX_LLVM=str(llvm), ETX_ROCM_HEADERS=str(headers), ETX_DEVLIBS=str(devlibs))
    if not (devlibs/'ocml.bc').is_file() or not (devlibs/'ockl.bc').is_file():
        raise ValueError('ETX_DEVLIBS must contain ocml.bc and ockl.bc; no -nogpulib qualification')
    contract = json.loads((build/'build-resources.json').read_text())
    min_workgroups_per_cu = max(min_workgroups_per_cu, contract['required_workgroups_per_cu'])
    devices = contract['devices']
    generated = sorted(p.name for p in build.glob('megakernel_d*.hip'))
    if type(devices) is not int or devices not in (2, 4, 8) or generated != sorted(f'megakernel_d{d}.hip' for d in range(devices)):
        raise ValueError(f'build-resources.json devices={devices!r} does not match the generated kernels {generated}')
    reports = []
    for rank in range(devices):
        ir = build/f'mk{rank}.ll'; assembly = build/f'mk{rank}.s'
        run_command(['bash', ROOT/'etx/importer/hip_ir_local.sh', build/f'megakernel_d{rank}.hip', ir,
                     '-I'+str(ROOT/'etx/runtime/include'), '-I'+str(build), '-I'+str(ROOT)],
                    build/f'rank{rank}-clang.log', timeout, env)
        run_command([llvm/'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-O3', ir, '-o', assembly],
                    build/f'rank{rank}-llc.log', timeout, env)
        report = parse_report(assembly.read_text(), f'etx_megakernel_d{rank}')
        report['rank'] = rank
        report['required_workgroups_per_cu'] = min_workgroups_per_cu
        reports.append(report)
        try:
            feasibility(report, config, min_workgroups_per_cu)
        finally:
            (build/'compiler-resources.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(f'RESOURCES rank={rank} {json.dumps(report, sort_keys=True)}', flush=True)
    if host_syntax:
        command = [llvm/'clang++', '-x', 'hip', '--cuda-host-only', '-fsyntax-only', '-nogpuinc', '-nogpulib',
                   '--offload-arch=gfx942', '-std=c++17', '-isystem', headers/'clang', '-isystem', headers/'inc',
                   '-include', '__clang_hip_runtime_wrapper.h', '-D__HIP_PLATFORM_AMD__',
                   '-I'+str(ROOT/'etx/runtime/include'), '-I'+str(build), '-I'+str(ROOT),
                   '-include', 'glm53_options.h']
        if sys.platform == 'darwin':
            command += ['-isystem', ROOT/'etx/importer/shim/mac']
        run_command(command+[ROOT/'examples/glm53/model_host.hip'], build/'host-syntax.log', timeout, env)
    model_build.finalize(build, local=True)
    return reports


def enumerate_candidates(axes, batches):
    names = [f.name for f in fields(options.Options)]
    for batch in batches:
        for values in itertools.product(*(axes[n] for n in names)):
            yield batch, dict(zip(names, values))


def later_commands(candidate, batch, staged=None, gate='production'):
    """Commands are plans only; real stages/references and GPU correctness required."""
    config = candidate/'build/build-options.json'
    gpu = candidate/f'gpu-b{batch}'
    stage_arg = shlex.quote(str(staged)) if staged is not None else '"${STAGE:?real validated pinned stage required}"'
    common = f'--stage {stage_arg} --build {shlex.quote(str(gpu))} --gate {gate}'
    return [f'ETX_GLM_BATCH={batch} ETX_GLM_GATE={gate} bash examples/glm53/build_model.sh {stage_arg} {shlex.quote(str(gpu))} --options {shlex.quote(str(config))}',
            f'python -m examples.glm53.bench {common} --reference "${{REFERENCE_B{batch}:?HF reference for this batch required}}" --out {shlex.quote(str(candidate / f"benchmark-b{batch}"))}']


def sweep(spec, out, axes, batches, *, llvm, headers, devlibs, timeout=180, host_syntax=False,
          enumerate_only=False, min_workgroups_per_cu=1, staged=None, gate='production', max_candidates=256):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 1, 'status': 'NOT-COMPLETE', 'gpu_used': False, 'selected_winner': None,
               'performance_measured': False, 'candidates': [], 'survivors': [],
               'toolchain': {'llvm': str(llvm), 'headers': str(headers), 'devlibs': str(devlibs)},
               'spec_sha256': __import__('hashlib').sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}
    try:
        if max_candidates < 1:
            raise ValueError('--max-candidates must be positive')
        if len(batches) != len(set(batches)) or any(len(v) != len(set(v)) for v in axes.values()):
            raise ValueError('duplicate batch/axis values would repeat identical candidates')
        count = len(batches)
        for axis in axes.values():count *= len(axis)
        if count > max_candidates:
            raise ValueError(f'{count} candidates exceeds --max-candidates {max_candidates}; refusing truncation')
        for index, (batch, values) in enumerate(enumerate_candidates(axes, batches)):
            record = {'index': index, 'batch': batch, 'options': values, 'status': 'NOT-COMPLETE'}
            receipt['candidates'].append(record)
            candidate = out/f'candidate-{index:04d}-b{batch}'
            candidate.mkdir()
            started = time.monotonic()
            print(f'CANDIDATE {index} batch={batch} {json.dumps(values,sort_keys=True)}', flush=True)
            try:
                config = options.from_dict(values)
                record['options_sha256'] = config.digest()
                record['cache_comparison_basis'] = 'matched explicit payload loads (cache bit differs)' if config.cache_ab else 'endpoint settings only; NOT an isolated cache-bit comparison'
                model_build.generate(spec, candidate/'build', batch, config)
                record['source_identity_sha256'] = stage.file_sha(candidate/'build/source-identity.json')
                if enumerate_only:
                    record['status'] = 'ENUMERATED'
                else:
                    record['resources'] = compile_candidate(candidate/'build', config, llvm, headers, devlibs,
                                                            timeout, host_syntax, min_workgroups_per_cu)
                    record['status'] = 'COMPILED'
                    record['build_identity_sha256'] = stage.file_sha(candidate/'build/build-identity.json')
                    record['later_commands'] = later_commands(candidate, batch, staged, gate)
                    receipt['survivors'].append(str(candidate))
            except Exception as exc:
                record['status'] = 'REJECTED'
                record['error'] = f'{type(exc).__name__}: {exc}'
                print(f'REJECTED {index}: {record["error"]}', flush=True)
            finally:
                record['elapsed_seconds'] = time.monotonic()-started
                (candidate/'status.json').write_text(json.dumps(record, indent=2)+'\n')
                (out/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if not receipt['candidates']:
            receipt['error'] = 'zero candidates'
        elif any(r['status'] == 'REJECTED' for r in receipt['candidates']):
            receipt['status'] = 'PARTIAL' if receipt['survivors'] else 'FAILED'
        else:
            receipt['status'] = 'ENUMERATED' if enumerate_only else 'COMPILED'
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (out/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
        (out/'survivors.json').write_text(json.dumps(receipt['survivors'], indent=2)+'\n')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', type=Path, help='validated stage; absent uses weights-free production shape')
    ap.add_argument('--allow-fixture', action='store_true')
    ap.add_argument('--gate', choices=['production','module','slice'], default='production')
    ap.add_argument('--maxlen', type=int, default=1072)
    ap.add_argument('--batches', default='1')
    ap.add_argument('--options', type=Path, help='base options JSON; axes override these values')
    for field in fields(options.Options):
        ap.add_argument('--'+field.name.replace('_','-'), help='comma-separated Cartesian axis')
    ap.add_argument('--llvm', type=Path, default=Path(os.environ.get('ETX_LLVM','/opt/homebrew/opt/llvm/bin')))
    ap.add_argument('--headers', type=Path, default=Path(os.environ.get('ETX_ROCM_HEADERS',str(Path.home()/'code/rocm-headers'))))
    ap.add_argument('--devlibs', type=Path, default=Path(os.environ['ETX_DEVLIBS']) if os.environ.get('ETX_DEVLIBS') else None)
    ap.add_argument('--max-candidates', type=int, default=256, help='refuse larger Cartesian sets; never silently truncate')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--host-syntax', action='store_true')
    ap.add_argument('--enumerate-only', action='store_true', help='emit candidates without compiling; no COMPILED survivors')
    ap.add_argument('--min-workgroups-per-cu', type=int, choices=[1,2], default=1, help='required static upper bound, not measured residency')
    a = ap.parse_args()
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        ap.error('--timeout must be finite and positive')
    if not a.enumerate_only and a.devlibs is None:
        ap.error('--devlibs or ETX_DEVLIBS is required (directory containing ocml.bc)')
    try:
        base = options.load(a.options)
        axes = {f.name: [int(v) for v in getattr(a,f.name).split(',') if v.strip()] if getattr(a,f.name) is not None else [getattr(base,f.name)] for f in fields(options.Options)}
        batches = [int(v) for v in a.batches.split(',') if v.strip()]
        spec = stage.validate(a.stage,allow_fixture=a.allow_fixture,gate=a.gate) if a.stage else model_build.production_compile_spec(a.maxlen)
        result = sweep(spec,a.out,axes,batches,llvm=a.llvm.resolve(),headers=a.headers.resolve(),
                       devlibs=a.devlibs.resolve() if a.devlibs else None,timeout=a.timeout,host_syntax=a.host_syntax,
                       enumerate_only=a.enumerate_only,min_workgroups_per_cu=a.min_workgroups_per_cu,
                       staged=a.stage.resolve() if a.stage else None,gate=a.gate,max_candidates=a.max_candidates)
    except Exception as exc:
        print(f'SWEEP NOT-COMPLETE: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    print(f'SWEEP {result["status"]} candidates={len(result["candidates"])} compiled={len(result["survivors"])} selected=none measured=none',flush=True)
    return 0 if result['status'] in ('COMPILED','ENUMERATED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
