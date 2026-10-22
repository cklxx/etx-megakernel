"""Codegen contracts for optional separate SYSTEM event storage (not GPU correctness)."""
import re

from etx.codegen import emit_kernel
from etx.codegen.plan_header import emit_plan_header
from etx.ir import Scope
from etx.passes import compile_graph
from examples import gemm_rs, moe_layer


SYSTEM_BASE = '(p.sys_events ? p.sys_events : p.events)'


def test_cross_device_wait_arrive_and_probe_share_the_system_buffer():
    plan = compile_graph(gemm_rs.build(2), 'gfx90a', gemm_rs.bindings(), {})
    sources = '\n'.join(emit_kernel(plan, d) for d in range(plan.n_devices))
    assert f'etx_wait_SYSTEM({SYSTEM_BASE} + ' in sources
    assert f'etx_arrive_SYSTEM({SYSTEM_BASE} + ' in sources
    assert f'ETX_POLL_SYSTEM({SYSTEM_BASE} + ' in sources
    assert not re.search(r'(?:etx_wait_SYSTEM|etx_arrive_SYSTEM|ETX_POLL_SYSTEM)\(p\.events', sources)


def test_runtime_system_count_initialization_uses_the_same_buffer():
    bindings = moe_layer.bindings()
    plan = compile_graph(moe_layer.build(), 'gfx942', bindings, moe_layer.runtime(bindings))
    # Isolate emitter behavior: the ordinary MoE fixture has runtime DEVICE counts.
    # Its scheduling/layout are unchanged; only this emitter input scope is varied.
    runtime_events = [e for e in plan.events.values() if e.runtime_init]
    assert runtime_events
    for e in runtime_events:
        e.scope = Scope.SYSTEM
    source = emit_kernel(plan)
    for e in runtime_events:
        line = next(line for line in source.splitlines() if f'// runtime init of {e.name}' in line)
        assert f'{SYSTEM_BASE}[ETX_EVW({e.offset} + i)]' in line


def test_plan_header_exposes_system_scope_for_counter_snapshots():
    plan = compile_graph(gemm_rs.build(2), 'gfx90a', gemm_rs.bindings(), {})
    header = emit_plan_header(plan)
    assert '#define ETX_HAS_EVENT_SCOPE 1' in header
    initializer = re.search(r'etx_ev_system\[[^]]+\]\s*=\s*\{([^}]+)\}', header)
    assert initializer
    scopes = [int(x.strip()) for x in initializer[1].split(',') if x.strip()]
    assert scopes == [int(e.scope == Scope.SYSTEM) for e in plan.events.values()]
