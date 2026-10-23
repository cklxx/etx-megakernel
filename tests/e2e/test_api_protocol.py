"""CPU protocol simulations through the caller facade, not device evidence."""
from dataclasses import replace
import pytest

from etx.external import PlanRequest, plan, simulate
from etx.testing import fixture


@pytest.mark.parametrize('devices', [1, 2, 8])
def test_protocol_execution_is_reproducible_and_explicitly_simulated(devices):
    planned = plan(PlanRequest(fixture(devices=devices, batch=2)))
    first = simulate(planned, seed=17)
    assert first == simulate(planned, seed=17)
    assert first.plan_identity == planned.identity
    assert first.backend == 'simulation'
    assert first.metrics['timing_scope'] == 'simulated_protocol'
    assert first.metrics['makespan_us'] > 0
    assert not first.deadlock
    with pytest.raises(ValueError, match='cannot claim a device backend'):
        replace(first, backend='gpu')


def test_forged_or_stale_plan_is_rejected():
    planned = plan(PlanRequest(fixture()))
    report = planned.report.to_dict()
    report['devices'] = 42
    with pytest.raises(ValueError, match='identity drift'):
        simulate(replace(planned, report=report))


def test_invalid_simulation_request():
    with pytest.raises(TypeError):
        simulate(fixture())
    with pytest.raises(ValueError):
        simulate(plan(PlanRequest(fixture())), seed=True)
