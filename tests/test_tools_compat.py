"""Legacy tool imports keep their callable identities after concern splitting."""
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize('name,owner', [
    ('gpu_jobs', 'etx.session.gpu_jobs'),
    ('timeline', 'etx.runtime.trace'),
    ('isa_hops', 'etx.tools.isa.hops'),
    ('acquire_contract', 'etx.tools.contracts.acquire_contract'),
    ('source_control', 'etx.tools.contracts.source_control'),
    ('shared_state', 'etx.tools.memory.shared_state'),
    *[(name, f'etx.tools.costs.{name}') for name in
      ('floors', 'stage_sim', 'fresh_sim', 'fresh_handoff')],
    *[(name, f'etx.tools.tracing.{name}') for name in
      ('inbody', 'layer_budget', 'ktrace', 'watchdog')],
    *[(name, f'etx.tools.diagnostics.{name}') for name in
      ('verify_gaps', 'spill_chain', 'isa_phases')],
    *[(name, f'etx.tools.validation.{name}') for name in
      ('bench_sim', 'cpu_check', 'pytest_acceptance')],
    ('cli', 'etx.tools.commands.cli'),
])
def test_legacy_exports_are_the_implementation(name, owner):
    legacy = importlib.import_module(f'etx.tools.{name}')
    implementation = importlib.import_module(owner)
    exports = {name: value for name, value in vars(implementation).items()
               if not name.startswith('_') and callable(value)}
    assert exports
    for name, value in exports.items():
        assert getattr(legacy, name) is value


def test_tool_concerns_stay_bounded():
    root = Path(__file__).resolve().parents[1] / 'etx/tools'
    for package in root.iterdir():
        if package.is_dir() and package.name != '__pycache__':
            assert sum(len(p.read_text().splitlines()) for p in package.rglob('*.py')) <= 1500
    assert all(len(p.read_text().splitlines()) <= 1500 for p in root.rglob('*.py'))
