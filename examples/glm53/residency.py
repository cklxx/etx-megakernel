"""Offline co-residency gate for a linked GLM build, from the binary's own gfx942 code objects.

The persistent megakernel needs every launched workgroup resident at once
(etx_host::check_residency()). That check needs a GPU; this one reads the
same inputs from the linked `run`: each kernel's VGPR/AGPR count, LDS and
workgroup size from the code-object metadata, and ETX_N_LAUNCH from
plan_data.h. A compiler that misses the occupancy target (ROCm 6.4.1 uses
AGPRs as VGPR spill slots: 1 wave/SIMD) is refused at build time instead of
at launch. The device's own check remains the authority; this is a gate, not
a substitute.

gfx942 rules: 512 unified vector registers (VGPR + AGPR; the metadata .vgpr_count
is already that total) per SIMD lane, allocated in granules of 8,
at most 8 waves per SIMD, 4 SIMDs per CU, 64 KiB LDS per CU in 256-byte
granules, at most 2 workgroups per CU in the machine model.

usage: python -m examples.glm53.residency <build> [--cus 304] [--llvm /opt/rocm/llvm/bin]
"""
from __future__ import annotations
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
KEYS = ('name', 'vgpr_count', 'agpr_count', 'sgpr_count', 'vgpr_spill_count', 'group_segment_fixed_size',
        'private_segment_fixed_size', 'max_flat_workgroup_size', 'wavefront_size')
MAX_WG_PER_CU = 2


def bundles(fatbin):
    """Split a .hip_fatbin section into its concatenated offload bundles."""
    starts = [i for i in range(0, len(fatbin), 8) if fatbin.startswith(MAGIC, i)]
    return [fatbin[s:(starts[n + 1] if n + 1 < len(starts) else len(fatbin))] for n, s in enumerate(starts)]


def kernels(notes):
    """Kernel records from `llvm-readelf --notes` text (AMDGPU metadata, one block per kernel).

    Kernel items are `  - .key:` at two spaces, their own fields `    .key:` at four;
    deeper lines (the argument list, which has its own `.name`) are ignored."""
    blocks = []
    for line in notes.split('amdhsa.kernels:', 1)[-1].splitlines():
        m = re.match(r'^(  - |    )\.(\w+):\s+(\S+)\s*$', line)
        if not m:
            continue
        if m.group(1) == '  - ':
            blocks.append({})
        if blocks:
            blocks[-1][m.group(2)] = m.group(3)
    return [{k: (b[k] if k == 'name' else int(b.get(k, 0))) for k in KEYS}
            for b in blocks if 'name' in b and 'vgpr_count' in b]


def resident(kernel, cus):
    """Resident workgroups per device for one kernel under the gfx942 rules above."""
    # gfx90a/gfx942 metadata .vgpr_count is the unified total (architectural VGPRs + AGPRs).
    registers = math.ceil(kernel['vgpr_count'] / 8) * 8
    waves_per_simd = min(8, 512 // registers) if registers else 8
    waves_per_wg = math.ceil(kernel['max_flat_workgroup_size'] / kernel['wavefront_size'])
    by_registers = 4 * waves_per_simd // waves_per_wg
    lds = math.ceil(kernel['group_segment_fixed_size'] / 256) * 256
    by_lds = 65536 // lds if lds else MAX_WG_PER_CU
    per_cu = min(by_registers, by_lds, MAX_WG_PER_CU)
    return dict(waves_per_simd=waves_per_simd, wg_per_cu_registers=by_registers, wg_per_cu_lds=by_lds,
                wg_per_cu=per_cu, resident=per_cu * cus)


def launch_count(build):
    m = re.search(r'#define ETX_N_LAUNCH (\d+)', (build / 'plan_data.h').read_text())
    if not m:
        raise ValueError('plan_data.h has no ETX_N_LAUNCH')
    return int(m.group(1))


def check(build, cus=304, llvm=Path('/opt/rocm/llvm/bin')):
    build = Path(build)
    launch = launch_count(build)
    ranks = json.loads((build / 'build-resources.json').read_text())['devices']
    found = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([llvm / 'llvm-objcopy', '--dump-section', f'.hip_fatbin={tmp / "fatbin"}',
                        build / 'run', tmp / 'stripped'], check=True, capture_output=True)
        for n, bundle in enumerate(bundles((tmp / 'fatbin').read_bytes())):
            path = tmp / f'b{n}'; path.write_bytes(bundle)
            targets = subprocess.run([llvm / 'clang-offload-bundler', '--list', '--type=o', f'--input={path}'],
                                     check=True, capture_output=True, text=True).stdout.split()
            for target in (t for t in targets if t.endswith('gfx942') or ':gfx942' in t or 'gfx942' in t):
                co = tmp / f'b{n}.co'
                subprocess.run([llvm / 'clang-offload-bundler', '--unbundle', '--type=o', f'--input={path}',
                                f'--targets={target}', f'--output={co}'], check=True, capture_output=True)
                notes = subprocess.run([llvm / 'llvm-readelf', '--notes', co], check=True,
                                       capture_output=True, text=True).stdout
                found += kernels(notes)
    megakernels = {k['name']: k for k in found if k['name'].startswith('etx_megakernel_d')}
    expected = {f'etx_megakernel_d{d}' for d in range(ranks)}
    rows = [dict(kernel, **resident(kernel, cus), launch=launch) for kernel in found]
    failures = sorted(expected - set(megakernels))
    failures += [r['name'] for r in rows if r['name'] in expected and r['resident'] < launch]
    return dict(schema=1, status='PASS' if not failures else 'FAIL', launch=launch, cus=cus, ranks=ranks,
                failures=failures, kernels=rows,
                basis='computed from the linked code objects; the device check_residency() remains the authority')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('build', type=Path)
    ap.add_argument('--cus', type=int, default=304, help='compute units per device (MI300X/MI325X: 304)')
    ap.add_argument('--llvm', type=Path, default=Path('/opt/rocm/llvm/bin'))
    a = ap.parse_args(argv)
    result = check(a.build, a.cus, a.llvm)
    (a.build / 'residency.json').write_text(json.dumps(result, indent=2) + '\n')
    for r in result['kernels']:
        if r['name'].startswith('etx_megakernel_d'):
            print(f'GLM-RESIDENCY {r["name"]} vgpr={r["vgpr_count"]} agpr={r["agpr_count"]} '
                  f'spill={r["vgpr_spill_count"]} lds={r["group_segment_fixed_size"]} '
                  f'waves_per_simd={r["waves_per_simd"]} resident={r["resident"]} launch={r["launch"]}', flush=True)
    print(f'GLM-RESIDENCY {result["status"]} launch={result["launch"]} cus={a.cus} '
          f'failures={result["failures"]}', flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
