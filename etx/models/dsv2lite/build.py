"""DeepSeek-V2-Lite (fleet-mi300x port) build provider: the steps of the historical examples/dsv2lite/build.sh.

Variant fields (strings, as the shell read them from its environment): fleet (fleet-mi300x tree),
patches (ETX_FLEET_PATCH, space separated), isolate (ETX_DSV2_ISOLATE), coherence (ETX_COHERENCE),
inline (ETX_INLINE_TILES, default 1), relay (ETX_RELAY_PLAN, default on), rdc (ETX_RDC, default 0),
d5 (ETX_DSV2_D5), verbose (ETX_BUILD_VERBOSE). Lever patches and the coherence gate are declaration
code in examples/dsv2lite, run as the same `python -m` steps from the repository root.
"""
import dataclasses
from pathlib import Path
import shutil

from etx.api.models import BuildRefused
from etx.api.types import Configuration


STAMPS2 = ('-DETX_DSV2_STAMPS2', '-DETX_DSV2_STAMPS2=1')


def _variant(request):
    v = request.variant.to_dict()
    if not v.get('fleet'):
        raise ValueError('dsv2lite build requires variant.fleet (path to fleet-mi300x)')
    return v


def refine(request):
    v = _variant(request)
    stamps2 = any(flag in STAMPS2 for flag in request.flags)
    if stamps2 and v.get('isolate'):
        raise BuildRefused('ETX_DSV2_STAMPS2 rejects ETX_DSV2_ISOLATE: repeated tasks share prologue slots')
    patches = (v.get('patches') or '').split()
    # The build-wide coherence mode (l1) needs the fleet family's agent-scope accessors.
    if (v.get('coherence') or 'declared') == 'l1' and 'l1_loads' not in patches:
        patches.append('l1_loads')
    inline = (v.get('inline') or '1') == '1'
    relay = v.get('relay') or 'on'
    rdc = v.get('rdc') or '0'
    root, out = request.root, request.output
    fleet_src = f'{out}/fleet_src' if (patches or stamps2) else f'{v["fleet"]}/src'
    shim = () if inline else (f'{root}/examples/dsv2lite/fleet_d5.hip' if v.get('d5') == '1'
                              else f'{root}/examples/dsv2lite/fleet_shim.hip',)
    rdc_flag = () if (rdc == '0' and inline) else ('-fgpu-rdc',)
    options = dict(request.plan.options.to_dict(), relay=relay, inline_tiles=inline)
    if v.get('coherence'):
        options['coherence'] = v['coherence']
    plan = dataclasses.replace(request.plan, options=Configuration(options))
    flags = ('-O3', '-std=c++17', '--offload-arch=gfx942', *rdc_flag, '-DFLEET_NT_WEIGHTS=1', *request.flags)
    includes = (fleet_src, f'{root}/etx/runtime/include', out, root)
    return dataclasses.replace(request, plan=plan, flags=flags, includes=includes, extra_sources=shim,
                               variant=Configuration(dict(v, patches=' '.join(patches), stamps2=stamps2,
                                                          fleet_src=fleet_src)))


def prepare(request, steps):
    v = request.variant.to_dict()
    patches = v['patches'].split()
    if patches or v['stamps2']:
        # Measurement mode patches only the private source copy.
        copy = Path(request.output) / 'fleet_src'
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(Path(v['fleet']) / 'src', copy, symlinks=True)
        if patches:
            steps.python('patch-fleet', ('-m', 'examples.dsv2lite.patch_fleet', v['fleet_src'], *patches), exit_code=1)
        if v['stamps2'] and steps.failed is None:
            steps.python('patch-stamps', ('-m', 'examples.dsv2lite.patch_stamps', v['fleet_src']), exit_code=1)
    return request


def check(request, steps):
    # D2: the plan and the tiles must agree on coherence; refused before the compiler runs.
    steps.python('coh-check', ('-m', 'examples.dsv2lite.coh_check', f'{request.output}/megakernel_d0.hip',
                               request.variant['fleet_src']), exit_code=2)
