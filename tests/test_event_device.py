"""Per-event owner device for split-buffer watchdog dumps (codegen contract, not GPU-run)."""
from pathlib import Path
import re

from etx.codegen.plan_header import emit_plan_header
from etx.ir import Scope
from etx.passes import compile_graph
from etx.models.glm53 import model_build
from examples.glm53 import model
from examples.glm53.options import Options

ROOT = Path(__file__).resolve().parents[1]


def owners(header):
    body = re.search(r'etx_ev_device\[[^\]]*\]\s*=\s*\{([^}]*)\}', header).group(1)
    return [int(x) for x in body.split(',') if x.strip()]


def pp2_two_layer_plan():
    # One layer per rank: device-local phase events on both devices plus SYSTEM boundary events.
    spec = model_build.production_compile_spec()
    spec['config'] = dict(spec['config'], num_hidden_layers=2,
                          layer_types=['linear_attention', 'deepseek_sparse_attention'], mlp_layer_types=['dense', 'sparse'])
    spec['weights'] = {k: v for k, v in spec['weights'].items() if k in ('l0', 'l1', 'embedding', 'norm', 'head')}
    spec['ranks'] = [dict(rank=0, layers=[0]), dict(rank=1, layers=[1])]
    return compile_graph(model.build_from(spec, 1, Options()), 'gfx942', {'B': 1}, {})


def test_owner_is_the_single_touching_device_and_system_events_have_none():
    plan = pp2_two_layer_plan()
    header = emit_plan_header(plan)
    assert '#define ETX_HAS_EVENT_DEVICE 1' in header
    got = owners(header)
    events = list(plan.events.values())
    assert len(got) == len(events)
    for e, owner in zip(events, got):
        devices = {plan.task(t).device for table in (plan.inst.producers, plan.inst.consumers)
                   for (name, _), tids in table.items() if name == e.name for t in tids}
        if e.scope == Scope.SYSTEM or len(devices) != 1:
            assert owner == -1
        else:
            assert owner == next(iter(devices))
    # Each device holds local events the other device's dump must skip; SYSTEM events stay visible.
    assert {0, 1, -1} <= set(got)
    assert all(o == -1 for e, o in zip(events, got) if e.scope == Scope.SYSTEM)


def test_dump_skips_only_foreign_local_events_in_split_mode():
    text = (ROOT / 'etx/runtime/include/etx/host.h').read_text()
    dump = text[text.index('void dump_events()'):text.index('void dump_events()') + 1500]
    assert 'sys_events && sys_events != d_events && etx_ev_device[e] >= 0 && etx_ev_device[e] != device' in dump
    assert 'not shown (dump those devices)' in dump
    snapshot = text[text.index('void snapshot_events('):text.index('void dump_events()')]
    assert 'etx_ev_device' not in snapshot  # the producer ledger keeps the raw merged snapshot
