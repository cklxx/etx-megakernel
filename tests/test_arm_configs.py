"""Offline v0.27.1 contract and captured proof replay; no vLLM import required."""
import argparse
import json
from pathlib import Path

import pytest

from examples.vllm_llm import vlbest_report as r
from test_vlbest_report import log, PLAN

FIXTURES = Path(__file__).parent / 'fixtures/vlbest'
FACTS = json.loads((FIXTURES / 'vllm_v0.27.1.json').read_text())
ARMS = [f'S{i}' for i in range(10)] + ['S4f', 'S5f', 'S6f']


def merged_compilation(cfg):
    # Same CLI overlay and right-biased merge as examples/llm/vllm_bench.py:71-75.
    parser = argparse.ArgumentParser()
    for flag in ('pass-config', 'custom-ops', 'cudagraph-mode'):
        parser.add_argument('--' + flag)
    parser.add_argument('--compile-mode', type=int)
    parser.add_argument('--no-detok', action='store_true')
    args = parser.parse_args(cfg['flags'])
    cc = ({'custom_ops': [args.custom_ops]} if args.custom_ops else {}) | (
        {'pass_config': json.loads(args.pass_config)} if args.pass_config else {}) | (
        {'cudagraph_mode': args.cudagraph_mode} if args.cudagraph_mode else {}) | (
        {'mode': args.compile_mode} if args.compile_mode is not None else {})
    return dict(cfg['kw'].get('compilation_config', {})) | cc


@pytest.mark.parametrize('arm', ARMS)
def test_schema_and_guard(arm):
    cfg = r.config(arm)
    cc = merged_compilation(cfg)
    assert set(cfg['kw'].get('compilation_config', {})) <= set(FACTS['fields']['CompilationConfig'])
    assert set(cc) <= set(FACTS['fields']['CompilationConfig'])
    passes = cc.get('pass_config', {})
    assert set(passes) <= set(FACTS['fields']['PassConfig'])
    effective = FACTS['guard']['defaults'] | cc
    for key, enabled in passes.items():
        assert not (key.startswith('fuse_') and enabled
                    and effective['splitting_ops'] is None
                    and not effective['use_inductor_graph_partition']), (
            f'ARM {arm}: {key} disabled by vllm@0.27.1 compilation.py:1173 guard; '
            'set splitting_ops=[] or use_inductor_graph_partition=True')


@pytest.mark.parametrize('arm', ARMS)
def test_proof_replay(arm):
    cfg = r.config(arm)
    historical = arm.removesuffix('f')
    captured = sorted(FIXTURES.glob(f'a_r*_{historical}*.log'))
    if historical not in ('S1', 'S9'):
        assert captured, f'UNCAPTURED {historical}'
    # S1/S9 were budget-dropped: no captured positive proof exists.
    for path in captured:
        expected = 'Disabling fuse_qk_norm_rope_kvcache' if historical in ('S4', 'S5', 'S6') else None
        missing = r.proof(path.read_text(), cfg)
        assert missing == expected, f"PROOF {missing!r} absent in {path}:1; expected refusal reason: {expected}"
    positive = log(arm).replace(
        'norm_quant, act_quant, qk_norm_rope_kvcache',
        'norm_quant, act_quant, mla_dual_rms_norm, qk_norm_rope_kvcache')
    missing = r.proof(positive, cfg, expected=r.expected_replacements(PLAN))
    assert missing is None, f"PROOF {missing!r} absent in synthetic {arm}:1; expected acceptance"
