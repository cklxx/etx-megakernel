"""Host-only compile/dry-run entry point; never invokes a GPU or downloads inputs.

python -m examples.vllm_llm.bf16_300.build --compile
ETX_DEVLIBS and ETX_ROCM_HEADERS override only local toolchain locations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from etx.codegen import emit_kernel, emit_plan_json
from etx.codegen.lowering import emit_lowering_header
from etx.machine.model import load_machine
from etx.ops import attnprep, gemv, coh
from .protocol import dry_run, require
from . import dispatcher
from .source import PAIRS, current_source, probe_source, specialised_source
from . import paths

FLAGS = ['-DETX_DSL_COH', '-DETX_DSL_NT', '-DETX_DSL_VECNORM', '-DETX_DSL_WSUM',
         '-DETX_DSL_SCALAR', '-DETX_DSL_GU1', '-DETX_DSL_RBATCH',
         '-DETX_EMIT_DSL_BIGK=0', '-DETX_EMIT_DSL_EXT=0', '-DETX_EMIT_FOLD_TOPK=1']


def resources(assembly):
    entries = []
    for block in re.findall(r'(?ms)^  - \.agpr_count.*?(?=^  - \.agpr_count|\Z)', assembly):
        def field(key):
            match = re.search(rf'\.{key}:\s+(\d+)', block)
            if not match:
                raise ValueError(f'missing compiler metadata: {key}')
            return int(match[1])
        vgpr, agpr, lds = field('vgpr_count'), field('agpr_count'), field('group_segment_fixed_size')
        # Register-only capacity, plus LDS / thread capacity. This is a static
        # estimate, NOT an actual co-residency measurement on a VF.
        register_waves = min(8, 512 // max(1, ((vgpr + agpr + 3) // 4) * 4))
        blocks = min(register_waves * 4 // 16, 65536 // max(1, ((lds + 511) // 512) * 512), 2048 // 1024)
        entries.append({'kernel': re.search(r'\.name:\s+(\S+)', block)[1], 'vgpr': vgpr, 'agpr': agpr,
                        'vgpr_spills': field('vgpr_spill_count'), 'sgpr_spills': field('sgpr_spill_count'),
                        'scratch_bytes_lane': field('private_segment_fixed_size'), 'lds_bytes': lds,
                        'static_blocks_per_cu': blocks, 'actual_co_residency': 'UNVERIFIED_CPU_ONLY',
                        'numeric_gate': vgpr <= 128 and lds <= 29752 and blocks >= 1})
    require(bool(entries), 'compiler_metadata')
    return entries


def fusion_gate(report):
    require(all(r['numeric_gate'] for r in report), 'fusion_resource_gate')


def preamble(lds):
    return (f'#define ETX_IMPORT_LDS {lds}\n#define ETX_IMPORT_SLOTS 8\n' +
            '#include "etx_lowering.h"\n#include "etx/primitives.h"\n#include "etx/tile_util.h"\n')


def control_source(fixed=False):
    s = preamble(gemv.LDS_BYTES) + (specialised_source() if fixed else current_source())
    for e in ['dsl_moe_up', 'dsl_moe_down']:
        s += f'''\nextern "C" __global__ __launch_bounds__(1024, 1) void bf16_300_{e}(const unsigned char* a, int first) {{
  etx_import_init(); __syncthreads();
  dsl_run_{e}(a, first + blockIdx.x);
  ETX_RELEASE_DOMAIN(); __syncthreads();
}}\n'''
    return s


def run(command, log, env):
    print('CPU command:', ' '.join(map(str, command)), flush=True)
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), cwd=paths.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(), flush=True)
        raise RuntimeError(f'exit {result.returncode}: {log}')


def prepare(compile=False, task=0, wave=0, row=0):
    require(gemv.DSL_NB == 1 and not gemv.DSL_EXT, 'bf16_batch_identity')
    # Do not let ambient experiment flags leak into imports or emitted sources.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ETX_VL_', 'ETX_LLM_', 'ETX_DSL_', 'DSL_ABLATE_', 'BF16_'))}
    env.pop('ETX_DEV_RESET', None)
    llvm = Path(env.get('ETX_LLVM', '/opt/homebrew/opt/llvm/bin'))
    headers = Path(env.get('ETX_ROCM_HEADERS', str(Path.home() / 'code/rocm-headers')))
    libs = Path(env.get('ETX_DEVLIBS', str(headers / 'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    env.update(ETX_LLVM=str(llvm), ETX_ROCM_HEADERS=str(headers), ETX_DEVLIBS=str(libs))
    if compile:
        require((libs / 'ocml.bc').is_file(), 'local_device_libraries')
    result_dir = paths.ROOT / 'results_bf16_300_cpu_20261011'
    result_dir.mkdir(exist_ok=True)
    pack = paths.ROOT / 'packed/bf16_300'
    pack.mkdir(parents=True, exist_ok=True)
    # No model pack or KV allocation is fabricated; separate metadata explicitly
    # records missing data and prevents transfer of any historical certificate.
    (pack / 'cpu_manifest.json').write_text(json.dumps({'variant_family': 'bf16_300', 'precision': 'bf16',
        'batch': 1, 'context': 1024, 'weights': None, 'golden': None, 'kv': None,
        'timing_eligible': False, 'certificate': None, 'head_state_bytes': 40, 'domain_queue_bytes': 16,
        'queue_contract': 'five tickets per head-local XCD, owner first; reset only at launch quiescence'}, indent=2) + '\n')
    variants = ['moe_control', 'moe_fixed_up', 'fusion', 'fusion_check'] + ['timestamp_' + p for p in PAIRS] + ['dispatcher_control'] + ['dispatcher_' + p for p in dispatcher.PAIRS]
    reports = {}
    for lever in variants:
        out = paths.ROOT / 'build/bf16_300' / lever
        if not compile:
            out = out / 'dry_run'
        out.mkdir(parents=True, exist_ok=True)
        print(f'prepare bf16_300_{lever}', flush=True)
        (out / 'etx_lowering.h').write_text(emit_lowering_header(load_machine('gfx942')))
        if lever.startswith('fusion'):
            imports = out / 'imp'
            commands = [
                [sys.executable, '-m', 'etx.importer.vllm_kernels', '--vllm', env.get('ETX_VLLM', str(Path.home() / 'code/vllm')),
                 '--out', imports, '--only', 'attn_mfma16_g8,attn_reduce', '--slots', 'attn_mfma16_g8=1,attn_reduce=8'],
                [sys.executable, '-m', 'etx.importer.adapter', imports / 'imports.json', '--tiles', out / 'adapters.hip', '--header', out / 'imports.h'],
                [sys.executable, '-m', 'etx.importer.bundle', '--imports', imports, '--exports', 'attn_mfma16_g8,attn_reduce',
                 '--devlibs', libs, '-o', out / 'imports.bc']]
            if compile:
                for index, command in enumerate(commands):
                    run(command, out / f'prepare_{index}.log', env)
            s = ('#include "etx_lowering.h"\n#include "etx/primitives.h"\n#include "etx/tile_util.h"\n' +
                 '#include "adapters.hip"\n' + coh.HELPERS + attnprep.TILE + (paths.HERE / 'fusion.hip').read_text())
            extra = ['-Xclang', '-mlink-builtin-bitcode', '-Xclang', str(out / 'imports.bc')]
            if lever == 'fusion_check':
                extra += ['-DBF16_300_CHECK']
        else:
            commands, extra = [], []
            if lever.startswith('dispatcher_'):
                tile = out / 'tiles.hip'
                tile.write_text(preamble(gemv.LDS_BYTES) + current_source())
                plan = dispatcher.graph(tile)
                s = dispatcher.instrument(plan, lever[len('dispatcher_'):]) if lever != 'dispatcher_control' else emit_kernel(plan)
                (out / 'plan.json').write_text(emit_plan_json(plan))
                # Dispatcher uses DEVICE events; retain this graph's original acquires/releases.
            elif lever.startswith('timestamp_'):
                s = preamble(gemv.LDS_BYTES) + probe_source(lever[len('timestamp_'):], task, wave, row)
            else:
                s = control_source(lever == 'moe_fixed_up')
        source = out / 'probe.hip'
        source.write_text(s)
        variant_flags = [f for f in FLAGS if not (lever.startswith('dispatcher_') and (f == '-DETX_DSL_COH' or f.startswith('-DETX_EMIT_')))]
        command = ['bash', paths.ROOT / 'etx/importer/hip_ir_local.sh', source, out / 'probe.ll',
                   '-I' + str(paths.ROOT / 'etx/runtime/include'), '-I' + str(out), '-I' + str(paths.ROOT)] + variant_flags + extra
        commands += [command, [llvm / 'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-O3', out / 'probe.ll', '-o', out / 'probe.s']]
        identity = {'variant': 'bf16_300_' + lever, 'source_sha256': hashlib.sha256(s.encode()).hexdigest(),
                    'flags': variant_flags + extra, 'commands': [[str(v) for v in c] for c in commands],
                    'sampling': {'task': task, 'wave': wave, 'row': row} if lever.startswith('timestamp_') else None,
                    'gpu_checks': 'NOT_RUN', 'timing_eligible': False}
        (out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        if compile:
            for index, command in enumerate(commands[-2:]):
                run(command, out / f'compile_{index}.log', env)
            report = resources((out / 'probe.s').read_text())
            fusion_gate(report)
            identity['resources'] = report
            identity['compiler_version'] = subprocess.check_output([llvm / 'clang++', '--version'], text=True)
            identity['hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [source, out / 'probe.ll', out / 'probe.s', llvm / 'clang++', llvm / 'llc', libs / 'ocml.bc', libs / 'ockl.bc']}
            if lever.startswith('fusion'):
                identity['constituent_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / 'imp').glob('*.bc'))}
            (out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
            reports[lever] = report
            print(json.dumps({'variant': lever, 'resources': report}), flush=True)
    result = {'protocol': dry_run(), 'compiled': compile, 'variants': reports if compile else variants,
              'thresholds_us_layer': {'skeleton': 2.083, 'fusion_net': 1.042, 'unpriced_target': 6.743},
              'vendor_moe_swap': 'CLOSED_30.72us_LAYER', 'actual_co_residency': 'UNVERIFIED_CPU_ONLY',
              'fusion_status': 'STOP_BEFORE_MODEL_INTEGRATION_AND_HARDWARE'}
    (result_dir / ('compile.json' if compile else 'dry_run.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--wave', type=int, default=0)
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    prepare(args.compile, args.task, args.wave, args.row)


if __name__ == '__main__':
    main()
