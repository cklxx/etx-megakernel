#!/usr/bin/env python3
"""Measure vendor linear calls; refusals never contain a timing.

Shapes, repetitions, copies, peak-bandwidth guard and MoE proxy:
/Users/ckl/.hermes/cache/scratch/o55-kernel-compare.json:1, result sections 1 and 3.
The nvref dump cannot validate folded qkv: host.hip:372,417,649-650,723-724.
Run --dry-run for CPU-only validation, before importing torch or starting a child.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SHAPES = {'qkv': (5120, 2048, 1), 'o_proj': (2048, 4096, 1),
          'router': (128, 2048, 1), 'lm_head': (151936, 2048, 1),
          'moe_up': (1536, 2048, 8), 'moe_down': (2048, 768, 8)}
LIBS = ('hipblaslt', 'rocblas', 'tunable')


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib', default=','.join(LIBS), help='hipblaslt|rocblas|tunable, or comma-separated arms')
    ap.add_argument('--copies', type=int, default=48)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--check', type=Path, help='nvref dump_all directory; required for measurement')
    ap.add_argument('--model', type=Path, default=Path.home() / 'llm/qwen3-30b-a3b')
    ap.add_argument('--out', type=Path, default=Path('kc_vendor'))
    ap.add_argument('--dry-run', action='store_true', help='validate configuration; no torch import, subprocess or GPU access')
    ap.add_argument('--worker', choices=('rocprof', 'graph-event'), help=argparse.SUPPRESS)
    return ap


def checked(torch, got, ref):
    # Match CHECK-DSL, including its cancellation floor (host.hip:773-780).
    ref = ref.to(torch.bfloat16).float().cpu()
    got = got.float().cpu()
    tol = torch.maximum(2 * ref.abs() / 128, ref.abs().max() / 4096)
    return bool(torch.isfinite(got).all() and torch.isfinite(ref).all() and ((got - ref).abs() <= tol).all())


def tensor_file(torch, path):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).clone()


def packed(torch, model, name, n, k):
    entries = [l.split() for l in (model / 'vl.manifest').read_text().splitlines()]
    entry = next(e for e in entries if e[:2] == ['t', name])
    if int(entry[3]) != n * k:
        raise ValueError(f'unexpected packed shape: {name}')
    with (model / 'vl.bin').open('rb') as f:
        f.seek(int(entry[2]))
        data = f.read(n * k * 2)
    if len(data) != n * k * 2:
        raise ValueError('short packed weight read')
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(n, k)


def worker(a):
    os.environ['PYTORCH_TUNABLEOP_ENABLED'] = str(int(a.lib == 'tunable'))
    os.environ['PYTORCH_TUNABLEOP_TUNING'] = str(int(a.lib == 'tunable'))
    import torch
    import torch.nn.functional as F
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU required')
    device = torch.cuda.get_device_name()
    if 'MI300X' not in device:
        raise RuntimeError(f'protocol requires MI300X, got {device}')
    print('KC_DEVICE', device, 'torch', torch.__version__, flush=True)
    preferred = getattr(torch.backends.cuda, 'preferred_blas_library', None)
    if preferred is None:
        raise RuntimeError('preferred_blas_library unavailable; cannot select requested arm')
    if a.lib == 'hipblaslt':
        try:
            preferred('hipblaslt')
        except (ValueError, RuntimeError):
            preferred('cublaslt')
    else:
        preferred('cublas')
    if hasattr(torch.cuda, 'tunable'):
        torch.cuda.tunable.enable(a.lib == 'tunable')
    if a.lib == 'tunable':
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
    print('KC_LIBRARY', a.lib, str(preferred()), flush=True)
    roctx = None
    if a.worker == 'rocprof':
        for lib in ('librocprofiler-sdk-roctx.so', '/opt/rocm/lib/librocprofiler-sdk-roctx.so', 'libroctx64.so'):
            try:
                roctx = ctypes.CDLL(lib)
                break
            except OSError:
                pass
        if roctx is None:
            raise RuntimeError('ROCTx unavailable; cannot delimit timed kernels')
        roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
        roctx.roctxRangePushA.restype = ctypes.c_int
        roctx.roctxRangePop.restype = ctypes.c_int
    metadata = {}
    # Scrub reads/writes twice the MALL capacity before each pass, outside ranges.
    # Capacity source: docs/RESEARCH-sync-hardware.md:98; router copies alone are too small.
    scrub = torch.zeros(2 * 256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    for shape, (n, k, calls) in SHAPES.items():
        count = 1 if shape == 'lm_head' else a.copies
        info = {'copies': count, 'iters': a.iters, 'bytes': calls * n * k * 2, 'check_ok': False}
        metadata[shape] = info
        print('KC_BEGIN', a.lib, shape, flush=True)
        x = torch.randn(calls, 1, k, dtype=torch.bfloat16)
        w = torch.randn(128 if calls > 1 else calls, n, k, dtype=torch.bfloat16)
        if shape in ('qkv', 'o_proj'):
            inp, expected, weight = ('hn', 'qkv', 'wqkv') if shape == 'qkv' else ('attn_out', 'o', 'wo')
            x = tensor_file(torch, a.check / f'{inp}.bin')[:k].reshape(1, 1, k)
            w = packed(torch, a.model, f'L47.{weight}', n, k).unsqueeze(0)
            ref = tensor_file(torch, a.check / f'{expected}.bin')[:n].reshape(1, 1, n)
        else:
            ref = torch.bmm(x.double(), w[:calls].double().transpose(1, 2))
        xd, wd = x.cuda(), w[:calls].cuda()
        got = torch.stack([F.linear(xd[e], wd[e]) for e in range(calls)])
        ok = checked(torch, got, ref)
        if shape == 'qkv':
            info['reason'] = ('hn -> qkv check ' + ('passed' if ok else 'failed') +
                              '; dump input is stale and qkv is post-prep (host.hip:372,417,649-650,723-724)')
        elif not ok:
            info['reason'] = 'CHECK-DSL tolerance failed'
        else:
            info['check_ok'] = True
        if not info['check_ok']:
            print('REFUSED', a.lib, shape, info['reason'], flush=True)
            continue
        info['check_basis'] = 'nvref layer-47 dump' if shape == 'o_proj' else 'CPU float64 linear reference (proxy only)'
        weights = [w[[(layer * calls + e) % 128 for e in range(calls)]].cuda()
                   if calls > 1 else wd.clone() for layer in range(count)]
        info['expert_ids'] = [[(layer * calls + e) % 128 for e in range(calls)] for layer in range(count)] if calls > 1 else None

        def row(layer):
            return [F.linear(xd[e], weights[layer][e]) for e in range(calls)]

        for it in range(a.warmup):
            scrub.add_(1)
            for layer in range(count):
                row(layer)
            torch.cuda.synchronize()
            print('KC_WARMUP', shape, it, flush=True)
        if a.lib == 'tunable':
            torch.cuda.tunable.tuning_enable(False)
        samples = []
        if a.worker == 'graph-event':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = [row(layer) for layer in range(count)]
            torch.cuda.synchronize()
        for it in range(a.iters):
            scrub.add_(1)
            torch.cuda.synchronize()
            if roctx is not None:
                for layer in range(count):
                    roctx.roctxRangePushA(f'kcmp:{shape}:{it}:{layer}'.encode())
                    row(layer)
                    torch.cuda.synchronize()
                    roctx.roctxRangePop()
            else:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / count)
            print('KC_PASS', a.lib, shape, it, flush=True)
        if samples:
            us = median(samples)
            if us < info['bytes'] / 5.3e6:
                info.update(check_ok=False, reason='cache-contaminated: below bytes / 5.3 TB/s')
            else:
                info.update(median_us=us, clock='graph-event', kernel_names=[],
                            names_unavailable='rocprof unavailable; graph event fallback', samples=len(samples))
        (a.out / 'metadata.json').write_text(json.dumps(metadata))
        del weights, wd, xd, got
        if a.worker == 'graph-event':
            del graph, outputs
        torch.cuda.empty_cache()
        if a.lib == 'tunable':
            torch.cuda.tunable.tuning_enable(True)
    (a.out / 'metadata.json').write_text(json.dumps(metadata))
    (a.out / 'complete').write_text('complete\n')


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    libs = a.lib.split(',')
    if any(x not in LIBS for x in libs) or len(set(libs)) != len(libs):
        ap.error('unknown or duplicate library')
    if (a.copies, a.warmup, a.iters) != (48, 10, 30):
        ap.error('pre-registered protocol requires copies=48, warmup=10, iters=30')
    if a.dry_run:
        print('CPU-ONLY: configuration validated; stopped before torch import, subprocess, and GPU access')
        print(json.dumps({'libraries': libs, 'shapes': SHAPES, 'copies': a.copies,
                          'warmup': a.warmup, 'iters': a.iters, 'measurements': None}))
        return 0
    if a.check is None:
        ap.error('--check is required; no unchecked timings')
    if a.worker:
        worker(a)
        return 0
    a.out.mkdir(parents=True, exist_ok=False)
    from etx.tools.ktrace import vendor
    results = {}
    for lib in libs:
        out = a.out / lib
        out.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', 'rocprof', '--lib', lib,
               '--check', str(a.check), '--model', str(a.model), '--out', str(out)]
        prof = ['rocprofv3', '--kernel-trace', '--marker-trace', '--output-format', 'csv', '-d', str(out / 'trace'), '--']
        try:
            rc = subprocess.run(prof + cmd, timeout=150).returncode
        except FileNotFoundError:
            rc = 127
        if rc and not (out / 'complete').exists():
            print('KC_FALLBACK', lib, 'rocprof failed; graph-event clock, not comparable with ETX', flush=True)
            cmd[cmd.index('rocprof')] = 'graph-event'
            rc = subprocess.run(cmd, timeout=150).returncode
            if rc:
                raise RuntimeError(f'vendor worker failed: {lib}')
            data = json.loads((out / 'metadata.json').read_text())
            results[lib] = {s: m if m.get('check_ok') else {'refused': m.get('reason')} for s, m in data.items()}
        else:
            if rc:
                print('KC_PROFILER_TEARDOWN', lib, rc, 'worker completed; checking full CSV coverage', flush=True)
            data = json.loads((out / 'metadata.json').read_text())
            results[lib] = vendor(out / 'trace', data)
        for shape, result in results[lib].items():
            print('KC_ROW', lib, shape, json.dumps(result), flush=True)
    payload = {'complete': True, 'arms': results}
    (a.out / 'result.json').write_text(json.dumps(payload, indent=2))
    print('KC_TOTAL', json.dumps(payload))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
