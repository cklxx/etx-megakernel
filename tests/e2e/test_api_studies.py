"""Reduced phase-0 studies: external API, real simulator, retained legacy results."""
from pathlib import Path
import sys

import pytest

from etx.external import ProcessRequest, execute, run_protocol_study

ROOT = Path(__file__).resolve().parents[2]
# Measured on desktop from HEAD's study with n/B=2, seed unchanged (2026-10-23).
# Full precision capture: /tmp/o55-lead/API-STUDIES-RAW-BEFORE-20261023.log.
LEGACY = {
    'variance': [
        (0, (3.542, 16.4612, 6.6044)),
        (0.1, (3.8478340798586546, 16.324236227924374, 6.910234079858654)),
        (0.3, (4.459502239575963, 16.778702239575964, 7.521902239575963)),
        (0.6, (5.3770044791519265, 17.696204479151927, 8.439404479151927)),
        (1, (6.600340798586545, 18.919540798586546, 9.662740798586546)),
    ],
    'routing-skew': [
        (0, (47.69066472105026, 112.89506472105025, 61.5362370148727)),
        (0.5, (47.69066472105026, 112.89506472105025, 61.5362370148727)),
        (1, (47.71324898566199, 112.17564898566195, 61.55882127948443)),
        (2, (47.69066472105026, 110.95306472105025, 61.5362370148727)),
        (4, (47.71324898566199, 112.17564898566195, 61.55882127948443)),
    ],
}


@pytest.mark.parametrize('name', LEGACY)
def test_reduced_study_matches_captured_legacy_values(name):
    result = run_protocol_study(name, size=2)
    assert result['backend'] == 'simulation'
    assert result['timing_scope'] == 'simulated_protocol'
    assert result['tensor_correctness'] == 'not_executed'
    assert result['gpu_measurement'] is False
    assert result['schema_version'] == 1 and result['seed'] == 1 and result['size'] == 2
    for row, (value, times) in zip(result['rows'], LEGACY[name], strict=True):
        assert row['value'] == value
        assert tuple(row['makespan_us'][mode] for mode in ('static', 'dynamic', 'hybrid')) == times
        assert row['best'] == 'static'
        if name == 'routing-skew':
            assert row['max_mean'] == 4.0
    detached = result['rows']
    detached.clear()
    assert len(result['rows']) == 5


@pytest.mark.parametrize('name,kwargs', [
    ('missing', {}), ('variance', {'size': 0}), ('variance', {'size': True}),
    ('variance', {'values': []}), ('variance', {'values': [float('nan')]}),
    ('variance', {'values': [3]}), ('routing-skew', {'values': [5]}),
])
def test_invalid_study_refused(name, kwargs):
    with pytest.raises(ValueError):
        run_protocol_study(name, **kwargs)


def test_study_cli_uses_same_rows(tmp_path):
    result = execute(ProcessRequest((sys.executable, 'bench/sim_experiments.py', '--size', '2'),
                                   str(ROOT), 30, environment={'PATH': '/usr/bin:/bin'}),
                     output=tmp_path / 'command')
    assert result.returncode == 0
    output = (tmp_path / 'command/output.log').read_text()
    assert 'n=2 -> 10 tasks' in output and 'B=2 tokens' in output
    for name, cases in LEGACY.items():
        for value, (static, dynamic, hybrid) in cases:
            prefix = f'{value:5.2f}' if name == 'variance' else f'{value:5.1f} {4.0:9.2f}'
            assert f'{prefix} {static:9.1f} {dynamic:9.1f} {hybrid:9.1f}   static\n' in output
