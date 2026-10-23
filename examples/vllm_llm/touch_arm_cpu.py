"""Offline full-plan compile and TOUCH default/explicit-ON .text comparison.

Reuses the retained Qwen import closure; no host execution, downloads or GPU.
--native uses only the installed etx-rocm22 image (--network=none, --pull=never).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from etx.codegen import emit_kernel, emit_lowering_header, emit_plan_json
from etx.passes import compile_graph, PassOptions
from etx.passes.plan import environment_options
from etx.contracts.compiler import _load_example
from .verify_gaps_cpu import FLAGS, ROOT


def run(out, retained, native=False):
    out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    retained = retained.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ETX_VL_', 'ETX_LLM_', 'ETX_INBODY_', 'ETX_DSL_', 'DSL_ABLATE_'))
           and k not in ('ETX_PREFETCH_ISOLATE_POLLER', 'ETX_DEV_RESET')}
    env.update(ETX_VL_IMPORTS=str(retained/'imp/imports.json'),
               ETX_VL_DSL_TILES=str(retained/'dsl_tiles.hip'),
               ETX_VL_ADAPTERS=str(retained/'imp_tiles.hip'), ETX_VL_CU='256',
               ETX_VL_REQUESTS='1', ETX_DEV_RESET='0', ETX_PREFETCH_ISOLATE_POLLER='0',
               ETX_LLM_CONFIG=str(ROOT/'examples/llm/configs/qwen3-30b-a3b.json'))
    headers = Path.home()/'code/rocm-headers'
    libs = Path(os.environ.get('ETX_DEVLIBS', str(headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')))
    llvm = headers/'rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin' if native else Path('/opt/homebrew/opt/llvm/bin')
    env.update(ETX_DEVLIBS=str(libs), ETX_LLVM=str(llvm), ETX_ROCM_HEADERS=str(headers))
    prefix = (['docker', 'run', '--rm', '--pull=never', '--network=none', '--platform', 'linux/amd64',
               '-v', f'{Path.home()/"code"}:{Path.home()/"code"}:ro', '-v', f'{out}:{out}',
               '-w', str(ROOT), '-e', f'ETX_DEVLIBS={libs}', '-e', f'ETX_LLVM={llvm}',
               '-e', f'ETX_ROCM_HEADERS={headers}', 'etx-rocm22'] if native else [])
    variants = ('default', 'on') if native else ('default', 'on', 'NO_TOUCH', 'NO_TOUCH_stamps', 'ISOLATE_POLLER')
    sources = {}; plans = {}; hashes = {}
    for arm in variants:
        # Emission is CPU-only, even when the compiler runs in a container.
        os.environ.clear(); os.environ.update(env)
        if arm != 'default': os.environ['ETX_VL_TOUCH'] = '1'
        if arm == 'ISOLATE_POLLER': os.environ['ETX_PREFETCH_ISOLATE_POLLER'] = '1'
        if arm == 'NO_TOUCH_stamps': os.environ['ETX_INBODY_EMIT'] = '1'
        mod = _load_example(str(ROOT/'examples/vllm_llm/model.py'))
        bindings = mod.bindings()
        plan = compile_graph(mod.build(), 'gfx942', bindings, mod.runtime(bindings), PassOptions(inline_tiles=True, **environment_options(os.environ)))
        plans[arm] = emit_plan_json(plan)
        work = out/arm; work.mkdir(exist_ok=True)
        sources[arm] = emit_kernel(plan)
        (work/'kernel.hip').write_text(sources[arm])
        (work/'etx_lowering.h').write_text(emit_lowering_header(plan.machine))
        defines = ['-DETX_NO_TOUCH'] if arm.startswith('NO_TOUCH') else []
        if arm.endswith('_stamps'): defines.append('-DETX_INBODY=1')
        commands = [prefix+['bash', str(ROOT/'etx/importer/hip_ir_local.sh'), str(work/'kernel.hip'), str(work/'kernel.ll'),
                    *FLAGS, '-DETX_INBODY_REPORT', *defines, '-I'+str(work), '-I'+str(ROOT/'etx/runtime/include'),
                    '-I'+str(ROOT/'etx/runtime'), '-Xclang', '-mlink-builtin-bitcode', '-Xclang', str(retained/'imports.bc')],
                    prefix+[str(llvm/'llc'), '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-O3', str(work/'kernel.ll'), '-o', str(work/'kernel.s')],
                    prefix+[str(llvm/'llvm-mc'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx942', '-filetype=obj', str(work/'kernel.s'), '-o', str(work/'kernel.o')],
                    prefix+[str(llvm/'llvm-objcopy'), '-O', 'binary', '--only-section=.text', str(work/'kernel.o'), str(work/'text.bin')]]
        for stage, cmd in enumerate(commands):
            print(f'TOUCH CPU {arm} stage={stage} native={native} ETX_DEVLIBS={libs}', flush=True)
            with (work/f'stage-{stage}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        hashes[arm] = hashlib.sha256((work/'text.bin').read_bytes()).hexdigest()
        print(f'TOUCH CPU {arm} gfx942 PASS tasks={len(plan.tasks)} workers={plan.workers_per_device()}', flush=True)
    if sources['default'] != sources['on']: raise ValueError('explicit ON changed emitted source')
    if len(set(plans.values())) != 1: raise ValueError('arm plan changed')
    if (out/'default/text.bin').read_bytes() != (out/'on/text.bin').read_bytes():
        raise ValueError('explicit ON changed .text')
    for arm in ('NO_TOUCH', 'ISOLATE_POLLER'):
        if arm in hashes and hashes[arm] == hashes['default']:
            raise ValueError(arm+' did not change device .text')
    result = dict(default_on_source_identical=True, default_on_text_identical=True,
                  plans_identical=True, text_sha256=hashes, native=native,
                  flag='existing -DETX_NO_TOUCH; ON means define absent', gpu='NOT_RUN_CPU_ONLY')
    (out/'identity.json').write_text(json.dumps(result, indent=2)+'\n')
    print('TOUCH IDENTITY '+json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--retained', type=Path, required=True)
    p.add_argument('--native', action='store_true')
    a = p.parse_args(); run(a.out, a.retained, a.native)
