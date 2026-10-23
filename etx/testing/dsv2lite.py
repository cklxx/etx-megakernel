"""Fleet graph/planner conformance, not native model execution."""
import os
from unittest.mock import patch

from etx.api.types import Configuration


def dsv2lite_conformance(kind):
    from etx.ir import Scope
    from etx.passes import compile_graph, verify_plan, PassOptions
    from etx.sim import simulate
    from examples.dsv2lite import graph as dsv

    if kind == 'shape':
        plan = compile_graph(dsv.build(), 'gfx942', {}, {})
        assert verify_plan(plan) == []
        assert 50_000 < len(plan.tasks) < 60_000
        for task in plan.tasks:
            if task.grid.startswith(('qkv_', 'oproj_', 'router_', 'gate_up_', 'down_')):
                assert task.domain == task.coord[0]
        events = plan.events
        assert events['E_oproj_5'].scope == Scope.DEVICE and events['E_down_5'].scope == Scope.DEVICE
        for name in ('E_qkv_5', 'E_qabs_5', 'E_attn_5', 'E_merge_5', 'E_norm_5', 'E_gu_5'):
            assert events[name].scope == Scope.DOMAIN, name
        result = dict(tasks=len(plan.tasks), verified=True)
    elif kind == 'simulation':
        plan = compile_graph(dsv.build(layers=3), 'gfx942', {}, {})
        simulated = simulate(plan, seed=0)
        assert not simulated.deadlock, simulated.blocked
        per_layer = simulated.makespan_us / 3
        assert 80 < per_layer < 300, per_layer
        result = dict(deadlock=simulated.deadlock, simulated_us_per_layer=per_layer)
    elif kind == 'interleave':
        from etx.codegen import emit_kernel
        for couple in ('router', 'experts', 'none'):
            with patch.dict(os.environ, ETX_DSV2_COUPLE=couple):
                plan = compile_graph(dsv.build(layers=3, requests=2), 'gfx942', {}, {}, PassOptions(relay='off'))
                assert plan.wg_per_cu == 2 and plan.workers_per_domain == 76
                for task in plan.tasks:
                    rank = 1 if task.grid.startswith('r1_') else 0
                    assert rank * 38 <= task.worker - task.domain * 76 < rank * 38 + 38
                simulated = simulate(plan, seed=0)
                assert not simulated.deadlock, (couple, simulated.blocked[:3])
        source = emit_kernel(plan, 0)
        assert '#ifdef ETX_SLOT_BY_CU' in source and 'etx_amdgcn_cu_key()' in source and 'k < 2' in source
        with patch.dict(os.environ, ETX_DSV2_COUPLE='router'):
            graph = dsv.build(layers=3, requests=2)
        assert 'r0_E_norm_1' in graph.grid('r1_qkv_1').in_edges
        assert 'r1_E_norm_0' in graph.grid('r0_qkv_1').in_edges
        result = dict(couplings=3, workers_per_domain=plan.workers_per_domain, emitted=True)
    else:
        raise ValueError('unknown Fleet graph conformance probe')
    return Configuration(dict(backend='simulation', device_qualification=False, **result))
