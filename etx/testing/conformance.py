"""Internal usability probes exposed without leaking mutable compiler objects."""
from contextlib import contextmanager
from dataclasses import dataclass, FrozenInstanceError
from unittest.mock import patch

from etx.api import Configuration, PlanRequest, plan
from .fixtures import fixture


@contextmanager
def forbid_compilation():
    """Prove a public compiler request fails before reaching graph compilation."""
    from etx.contracts import compiler
    with patch.object(compiler, 'compile_graph',
                      side_effect=AssertionError('unsupported request reached compiler')):
        yield


@dataclass(frozen=True)
class ConformanceResult:
    seam: str
    checks: tuple[str, ...]
    evidence: Configuration


def conformance(seam, *, devices=1, batch=1):
    if seam == 'declarations':
        config = Configuration({'nested': [1]})
        try:
            config._json = '{}'
        except FrozenInstanceError:
            return ConformanceResult('declarations', ('frozen-storage',),
                                     Configuration({'backend': 'cpu'}))
        raise AssertionError('configuration storage is mutable')
    if seam != 'planning':
        raise ValueError(f'unknown conformance seam {seam!r}')
    from etx.contracts.planning import compile_request
    from etx.passes.plan import verify_plan

    workload = fixture(devices=devices, batch=batch)
    before = workload.to_dict()
    request = PlanRequest(workload)
    compiled = compile_request(request)
    if verify_plan(compiled):
        raise AssertionError('compiler returned an invalid plan')
    first = plan(request)
    compiled.graph.grids.clear()
    second = plan(request)
    if first != second or workload.to_dict() != before:
        raise AssertionError('planner requests share mutable state')
    descs = first.report['descs']
    if {desc['device'] for desc in descs} != set(range(devices)):
        raise AssertionError('rank ownership is incomplete')
    if devices > 1 and any(event['scope'] != 'SYSTEM' for event in first.report['events']):
        raise AssertionError('cross-rank handoff lacks SYSTEM scope')
    return ConformanceResult('planning', ('verified', 'isolated', 'ownership', 'scope'),
                             Configuration(dict(plan_identity=first.identity, devices=devices,
                                                batch=batch, tasks=len(descs), backend='cpu')))
