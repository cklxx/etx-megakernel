"""The external planner returns records, not mutable compiler plans."""
from dataclasses import FrozenInstanceError
import json

import pytest
from etx.external import PlanRequest, PlannedWorkload, kernel, KernelSpec, model, plan
from etx.testing import fixture


@pytest.mark.parametrize('devices', [1, 2, 8])
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
def test_graph_planning_is_deterministic_and_complete(devices, batch):
    workload = fixture(devices=devices, batch=batch)
    before = workload.to_dict()
    request = PlanRequest(workload)
    first, second = plan(request), plan(request)
    assert first.identity == second.identity
    assert workload.to_dict() == before
    assert first.report['devices'] == devices
    assert len(first.report['descs']) == devices * batch * 2
    assert {item['device'] for item in first.report['descs']} == set(range(devices))
    if devices > 1:
        assert {event['scope'] for event in first.report['events']} == {'SYSTEM'}
    record = json.loads(json.dumps(first.to_dict()))
    assert PlannedWorkload.from_dict(record) == first
    assert first.options['coherence'] == 'declared'


def test_plan_snapshots_do_not_expose_mutable_compiler_state():
    first = plan(PlanRequest(fixture()))
    identity = first.identity
    first.report['descs'].clear()
    first.machine['resources'].clear()
    assert first.identity == identity
    assert first.report['descs']
    with pytest.raises(FrozenInstanceError):
        first.schema_version = 2
    assert plan(PlanRequest(fixture(batch=2))).identity != identity


@pytest.mark.parametrize('workload', [
    model('unknown', config={}),
    kernel(KernelSpec('unknown', 'native', {})),
    kernel(KernelSpec('unknown', 'imported', {})),
    kernel(KernelSpec('unknown', 'graph', {})),
    kernel(KernelSpec('unknown', 'graph', {'arbitrary': True})),
])
def test_unsupported_providers_fail_before_emission(workload):
    with pytest.raises(ValueError):
        plan(PlanRequest(workload))


@pytest.mark.parametrize('options', [
    {'unknown': True}, {'event_elimination': 'false'}, {'force_mode': 'other'},
    {'coherence': 'other'}, {'relay': 'other'}, {'affinity_imbalance': 0},
])
def test_invalid_options_fail_closed(options):
    with pytest.raises(ValueError):
        plan(PlanRequest(fixture(), options=options))


def test_planner_options_are_explicit_not_ambient(monkeypatch):
    monkeypatch.setenv('ETX_COHERENCE', 'invalid-ambient-value')
    workload = fixture()
    assert plan(PlanRequest(workload)).options['coherence'] == 'declared'
    assert plan(PlanRequest(workload, options={'coherence': 'l1'})).options['coherence'] == 'l1'


def test_invalid_request_and_target():
    with pytest.raises(TypeError):
        plan(object())
    with pytest.raises(ValueError):
        plan(PlanRequest(fixture(), target='../../target.yaml'))
    with pytest.raises(ValueError):
        PlanRequest(fixture(), bindings={'B': True})
