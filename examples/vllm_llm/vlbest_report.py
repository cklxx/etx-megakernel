"""Pre-registered strongest-tested search, section C (host wall, one session).

A missing proof is NOT-APPLIED, never a timing observation. No performance
output is printed until the complete five-round certificate is validated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from statistics import mean, median

from examples.llm.vllm_bench import CONFIG_PREFIX, decode_config
from examples.vllm_llm import import_certificate as cert
from examples.vllm_llm.nvref_report import Session, etx, one_line, verdict, vllm
from examples.vllm_llm.pairs import Refusal, verdict as fusion_verdict

IMAGE = 'vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7'
GROUP3 = ['V2 model runner', 'AITER MoE dispatch policy', 'AITER fmoe tuning',
          'block-size 32', 'compile_sizes=[1] + max_autotune', 'skinny GEMM off',
          'shuffle KV cache', 'stream-interval >1']
PREDICTIONS = 'S1 -0.05…-0.25; S5 -0.1…-0.4; S7/S8 within ±0.1 ms [E]. LEAD survives if F gains <0.40 ms on S0.'


def config(arm):
    env, kw, flags = {}, {}, []
    if arm in ('S4', 'S5', 'S6'):
        env['VLLM_ROCM_USE_AITER'] = '1'
        kw['attention_backend'] = 'ROCM_AITER_UNIFIED_ATTN' if arm == 'S6' else 'ROCM_AITER_FA'
        flags = ['--pass-config', '{"fuse_qk_norm_rope_kvcache":true}']
    if arm in ('S5', 'S6'):
        env.update({f'VLLM_ROCM_USE_AITER_{part}': '0' for part in ('MOE', 'TRITON_GEMM', 'RMSNORM')})
    if arm in ('S1', 'S9'):
        env['VLLM_TUNED_CONFIG_FOLDER'] = '$HOME/moecfg/' + ('mi308' if arm == 'S1' else 'tuned')
    if arm in ('S2', 'S3'):
        kw['attention_backend'] = 'TRITON_ATTN' if arm == 'S2' else 'ROCM_AITER_UNIFIED_ATTN'
    if arm == 'S7':
        env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    if arm == 'S8':
        flags = ['--no-detok']
    return dict(id=arm, env=env, kw=kw, flags=flags, components=[arm])


def proof(text, cfg):
    """Return the missing proof, without interpreting any timings."""
    echoed = decode_config(one_line(text, CONFIG_PREFIX, cfg['id']))
    for key in ('async_scheduling', 'cudagraph_mode', 'pass_config', 'kernel_config', 'attention_backend', 'env'):
        if key not in echoed:
            return f'missing config {key}'
    for key, value in cfg['env'].items():
        actual = echoed['env'].get(key, '')
        if not (actual.endswith(value.removeprefix('$HOME')) if value.startswith('$HOME/') else actual == value):
            return f'env {key}'
    backend = cfg['kw'].get('attention_backend')
    if backend and f'Using {backend} backend (selected via --attention-backend)' not in text:
        return f'backend {backend}'
    components = cfg['components']
    if any(c in components for c in ('S4', 'S5', 'S6')):
        for pattern in (r'Enabled custom fusions: [^\n]*qk_norm_rope_kvcache',
                        r'QK-Norm\+RoPE\+KVCache fusion: replaced 48 pattern\(s\)',
                        r'Capturing CUDA graphs \(decode, FULL\)'):
            if not re.search(pattern, text):
                return pattern
    if any(c in components for c in ('S5', 'S6')):
        for line in ('Using TritonExperts MoE backend', "rms_norm=['native']"):
            if line not in text:
                return line
    folder = cfg['env'].get('VLLM_TUNED_CONFIG_FOLDER')
    if folder:
        suffix = re.escape(folder.removeprefix('$HOME'))
        if not re.search(r'Using configuration from \S*' + suffix + r'/E=128,N=768,device_name=AMD_Instinct_MI300X.json[^\n]*for MoE layer', text) or 'Using default MoE config' in text:
            return 'tuned MoE file'
    if 'S7' in components and '(EngineCore pid=' in text:
        return 'in-process EngineCore'
    if 'S8' in components and 'VLLM-SAMPLING detok=0' not in text:
        return 'detok=0'
    return None


def observation(text, cfg, label=None):
    try:
        missing = proof(text, cfg)
    except (Refusal, ValueError, TypeError, KeyError):
        missing = 'config echo'
    if missing:
        raise Refusal(f"NOT-APPLIED {cfg['id']}: {missing}")
    samples = re.findall(r'^VLLM-SAMPLE sample=(\d+) first_divergence=(\d+/\d+)$', text, re.M)
    if samples != [(str(i), '32/32') for i in range(5)]:
        raise Refusal(f"failed 32/32 {cfg['id']}: every sample required")
    return vllm(text, cfg['id'], cfg['env'].get('VLLM_ROCM_USE_AITER', '0'), label or cfg['id']).wall


def screening(values, brackets):
    if len(values) != 2 or len(brackets) != 2:
        return 'INSUFFICIENT', []
    if any(abs(a - b) > .050 + 1e-12 for a, b in brackets):
        return 'INSUFFICIENT', []
    deltas = [v - mean(b) for v, b in zip(values, brackets)]
    win = all(d < 0 for d in deltas) and (mean(deltas) <= -.050 or math.isclose(mean(deltas), -.050, abs_tol=1e-12))
    return ('WINNER' if win else 'NO-WINNER'), deltas


def screen(directory, stage='a', candidates=None):
    session = Session(directory)
    candidates = candidates or [config(f'S{i}') for i in range(1, 10)]
    result = dict(winners={}, excluded={}, tested=['S0'], brackets=[])
    try:
        result['brackets'] = [[observation(session.read(f'{stage}_r{r}_S0_{end}'), config('S0'))
                               for end in ('open', 'close')] for r in (1, 2)]
    except (Refusal, OSError) as exc:
        result['excluded']['S0'] = str(exc)
        return result
    for cfg in candidates:
        arm = cfg['id']
        try:
            values = [observation(session.read(f'{stage}_r{r}_{arm}'), cfg) for r in (1, 2)]
            state, deltas = screening(values, result['brackets'])
            result['tested'].append(arm)
            if state == 'WINNER':
                result['winners'][arm] = dict(config=cfg, deltas=deltas, gain=mean(deltas))
            else:
                result['excluded'][arm] = state
        except (Refusal, OSError) as exc:
            result['excluded'][arm] = str(exc)
    return result


def combination(winners):
    axes = [('S1', 'S9'), ('S2', 'S3', 'S4', 'S5', 'S6'), ('S7', 'S8')]
    chosen = [min((winners[a] for a in axis if a in winners), key=lambda w: w['gain'], default=None) for axis in axes]
    if sum(c is not None for c in chosen) < 2:
        return None
    cfg = dict(id='C', env={}, kw={}, flags=[], components=[])
    # MoE last: its tuned Triton path overrides S4's AITER MoE.
    for c in (chosen[1], chosen[2], chosen[0]):
        if c:
            src = c['config']
            cfg['env'].update(src['env']); cfg['kw'].update(src['kw'])
            cfg['flags'] += src['flags']; cfg['components'] += src['components']
    if chosen[0] and cfg['env'].get('VLLM_ROCM_USE_AITER') == '1':
        cfg['env']['VLLM_ROCM_USE_AITER_MOE'] = '0'
    return cfg


def finalist(winners):
    return deepcopy(min(winners.values(), key=lambda w: w['gain'])['config']) if winners else config('S0')


def reference(default, final, final_id):
    if len(default) != 5 or len(final) != 5:
        raise Refusal('NOT-COMPLETE: five confirmation rounds required')
    return final_id if median(final) < median(default) else 'S0'


def paired(deltas):
    # Unchanged nvref_report verdict and resolution rule (lines 265–290).
    spread = max(deltas) - min(deltas)
    med_abs = abs(median(deltas))
    unresolved = med_abs < spread or spread > .050 + 1e-12
    return 'INSUFFICIENT' if unresolved and med_abs <= spread else verdict(deltas)


def spread_bound(deltas):
    spread = max(deltas) - min(deltas)
    return spread <= .050 + 1e-12 and spread < abs(median(deltas))


def report(directory):
    session = Session(directory)
    selected = json.loads((directory / 'final.json').read_text())
    cfg = selected['final']
    certificate = cert.validate(directory, (0, 1040))
    default = [observation(session.read(f'S0_r{r}'), config('S0')) for r in range(1, 6)]
    final = default if cfg['id'] == 'S0' else [observation(session.read(f'F_r{r}'), cfg) for r in range(1, 6)]
    rid = reference(default, final, cfg['id'])
    rv = default if rid == 'S0' else final
    fused = [etx(session.read(f'etx_r{r}'), f'etx_r{r}', 'fused').wall for r in range(1, 6)]
    unfused = [etx(session.read(f'etxu_r{r}'), f'etxu_r{r}', 'graph').wall for r in range(1, 6)]
    digest = one_line(session.read('docker_pull', 'None'), 'IMAGE ', 'docker_pull').removeprefix('IMAGE ').strip()
    if digest != IMAGE:
        raise Refusal('wrong image digest')
    if not re.search(r'(?:^|/)golden_c1024.txt: OK$', session.read('golden_sha', 'None'), re.M):
        raise Refusal('golden SHA not checked')
    print(f'SYSTEM  Qwen3-30B-A3B bf16 | MI300X VF | ROCm 7.14.0 | bs1 ctx1024 | host wall | 5 rounds | image {digest}')
    print('| arm | config | round medians | median | spread | 32/32 | proof/cert |')
    print('|---|---|---|---|---|---|---|')
    for arm, desc, vals, proof_text in (
        ('ETX fused', 'timed build', fused, 'CERT OK'),
        ('ETX unfused', 'same kernels, HIP graph', unfused, 'CERT OK'),
        ('vLLM default (S0)', 'none', default, 'proof OK'),
        ('vLLM strongest tested (R)', json.dumps(config('S0') if rid == 'S0' else cfg, sort_keys=True), rv, 'proof OK')):
        print(f'| {arm} | {desc} | {vals} | {median(vals):.3f} | {max(vals)-min(vals):.3f} | 5/5 | {proof_text} |')
    gains = [(u-f)/u for f, u in zip(fused, unfused)]
    fusion = fusion_verdict(gains)
    print(f"FUSION  unfused−fused: gains {gains}, median {fusion['median']:.2%}, spread {fusion['spread']:.2%} → {'ESTABLISHED' if fusion['established'] else 'NOT'}")
    ds, d0 = ([f-v for f, v in zip(fused, vv)] for vv in (rv, default))
    print(f'DELTA   fused−R: {ds} median {median(ds):+.3f} spread {max(ds)-min(ds):.3f} → {paired(ds)}   (vs S0: {d0} → {paired(d0)})')
    print(f'SPREAD-BOUND fused−R: {spread_bound(ds)} (spread ≤ 0.050 ms and < |median|; diagnostic only)')
    search = selected['search']
    print(f"SEARCH  tested {search['tested']} | winners {list(search['winners'])} | NOT-APPLIED "
          f"{ {k:v for k,v in search['excluded'].items() if 'NOT-APPLIED' in v} } | failed 32/32 "
          f"{ {k:v for k,v in search['excluded'].items() if '32/32' in v} } | untested: {GROUP3}")
    print('SEARCH exclusions ' + json.dumps(search['excluded'], sort_keys=True))
    print('Predictions ' + PREDICTIONS)
    print(f"FULL-CLAIM {'ELIGIBLE' if certificate['eligible'] else 'NOT ELIGIBLE'}; strongest only among tested, lever-proven, 32/32 arms.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--select', choices=['screen', 'final'])
    a = ap.parse_args(argv)
    try:
        if a.select == 'screen':
            search = screen(a.directory)
            data = dict(search=search, combination=combination(search['winners']))
            (a.directory / 'screen.json').write_text(json.dumps(data, indent=2))
            print(json.dumps(data, sort_keys=True))
        elif a.select == 'final':
            data = json.loads((a.directory / 'screen.json').read_text())
            search = data['search']
            if data['combination']:
                b = screen(a.directory, 'b', [data['combination']])
                for key in ('winners', 'excluded'):
                    search[key].update(b[key])
                search['tested'] += [x for x in b['tested'] if x != 'S0']
            data = dict(search=search, final=finalist(search['winners']))
            (a.directory / 'final.json').write_text(json.dumps(data, indent=2))
            print(json.dumps(data, sort_keys=True))
        else:
            report(a.directory)
    except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'NOT-COMPLETE: {exc}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
