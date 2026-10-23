"""Declarative graph adapter; mutable compiler objects never cross the facade."""
from dataclasses import asdict, fields
import json

from etx.api.types import PlanRequest, PlannedWorkload
from etx.codegen import emit_plan_json
from etx.ir.types import Graph, Resource, TileBody
from etx.machine.model import list_archs, load_machine
from etx.passes import PassOptions, compile_graph
from etx.passes.plan import recorded_options


def _graph(workload):
    if workload.kind != 'graph':
        raise ValueError(f'no planning provider for workload kind {workload.kind!r}')
    config = workload.config.to_dict()
    unknown = set(config) - {'tensors', 'events', 'grids', 'codegen'}
    if unknown:
        raise ValueError(f'unknown graph configuration fields: {sorted(unknown)}')
    graph = Graph(workload.name, codegen=config.get('codegen', {}))
    for record in config.get('tensors', []):
        if record['name'] in graph.tensors:
            raise ValueError(f'duplicate tensor {record["name"]!r}')
        graph.tensor(**record)
    for record in config.get('events', []):
        if set(record) - {'name', 'shape', 'wait_count', 'runtime_init_by', 'runtime_count'}:
            raise ValueError('event declaration contains unsupported fields')
        if record['name'] in graph.events:
            raise ValueError(f'duplicate event {record["name"]!r}')
        graph.etensor(**record)
    for record in config.get('grids', []):
        record = dict(record)
        if 'prologue' in record:
            raise ValueError('prologue is compiler-owned')
        body = TileBody(**record.pop('body'))
        resource = Resource(**record.pop('resource', {}))
        graph.call_device(body=body, resource=resource, **record)
    if not graph.grids:
        raise ValueError('graph requires at least one grid')
    return graph


def compile_request(request):
    if not isinstance(request, PlanRequest):
        raise TypeError('plan requires PlanRequest')
    if request.target not in list_archs():
        raise ValueError(f'unknown packaged target {request.target!r}')
    options = request.options.to_dict()
    unknown = set(options) - {field.name for field in fields(PassOptions)}
    if unknown:
        raise ValueError(f'unknown planner options: {sorted(unknown)}')
    # The legacy compiler may read ETX_COHERENCE; the public request never does.
    options.setdefault('coherence', 'declared')
    opts = PassOptions(**options)
    for field in fields(PassOptions):
        if type(field.default) is bool and type(getattr(opts, field.name)) is not bool:
            raise ValueError(f'{field.name} requires a boolean')
    if type(opts.affinity_imbalance) not in (int, float) or opts.affinity_imbalance < 1:
        raise ValueError('affinity_imbalance must be at least one')
    if opts.coherence not in ('declared', 'l1'):
        raise ValueError('invalid coherence')
    if opts.relay not in ('off', 'on', 'auto'):
        raise ValueError('invalid relay')
    if opts.force_mode not in (None, 'static', 'dynamic', 'hybrid'):
        raise ValueError('invalid force_mode')
    machine = load_machine(request.target)
    if request.workload.kind == 'graph' and 'provider' in request.workload.config:
        from .providers import operator_inputs
        graph, bindings, runtime = operator_inputs(request)
    elif request.workload.kind == 'model':
        from .models import model_inputs
        graph, bindings, runtime = model_inputs(request)
    else:
        graph = _graph(request.workload)
        bindings, runtime = request.bindings.to_dict(), request.runtime.to_dict()
    return compile_graph(graph, machine, bindings, runtime, opts)


def plan(request):
    return snapshot(request, compile_request(request))


def recompile(planned):
    if not isinstance(planned, PlannedWorkload):
        raise TypeError('operation requires PlannedWorkload')
    compiled = compile_request(planned.request)
    if snapshot(planned.request, compiled).identity != planned.identity:
        raise ValueError('plan identity drift; replan before using this plan')
    return compiled


def snapshot(request, compiled):
    # JSON conversion normalizes enum/integer dictionary keys in compiler data.
    machine = json.loads(json.dumps(asdict(compiled.machine)))
    return PlannedWorkload(request, machine, recorded_options(compiled.options),
                           json.loads(emit_plan_json(compiled)), tuple(compiled.log))


def environment_options(environ):
    """Adapter onto the planner's environment translation (etx.passes.plan.environment_options)."""
    from etx.passes.plan import environment_options as options
    return options(environ)
