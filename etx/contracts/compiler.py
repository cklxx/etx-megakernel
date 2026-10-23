"""Compatibility provider for Python graph files; shared compiler/emission services.

Importing a graph file executes its Python code, exactly as the historical CLI.
Only an explicit compiler_command invocation performs that operation. Ambient
legacy options and custom architecture files are intentionally preserved here;
new declarative PlanRequest uses its stricter explicit configuration instead.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

from etx.api.compiler import CompilerRequest
from etx.frontends import collect_sources
from etx.machine import list_archs, load_machine
from etx.passes import PassOptions, compile_graph
from etx.sim import simulate
from .emission import emit_payloads


def _load_example(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Kept for compatibility with existing graph-file imports/dataclasses.
    sys.path.insert(0, str(path.parent.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan(request):
    module = _load_example(request.graph)
    graph = module.build()
    bindings = module.bindings(**request.bindings.to_dict())
    runtime = module.runtime(bindings)
    options = PassOptions(**request.options.to_dict())
    return compile_graph(graph, load_machine(request.target), bindings, runtime, options), graph


def compiler_command(request):
    if not isinstance(request, CompilerRequest):
        raise TypeError('compiler_command requires CompilerRequest')
    if request.command == 'archs':
        for name in list_archs():
            machine = load_machine(name)
            print(f"{name:8s} {machine.family:10s} domains={machine.num_domains} cus/domain={machine.cus_per_domain()} lds={machine.resources['lds_kb']}KB hbm={machine.bandwidth['hbm_tbs']}TB/s")
        return 0
    compiled, graph = _plan(request)
    if request.command == 'compile':
        output = Path(request.output)
        output.mkdir(parents=True, exist_ok=True)
        files, _, _ = emit_payloads(compiled, all_rank_headers=False, runtime_headers=False)
        for name, text in files.items():
            (output / name).write_text(text, encoding='utf-8')
        print(compiled.summary())
        print(f'sources to link: {collect_sources(compiled.graph)}')
        print(f'wrote {output}/')
        if request.simulate:
            result = simulate(compiled, seed=request.seed)
            print('sim:', result.summary())
            if result.deadlock:
                return 2
        return 0
    print(compiled.summary())
    print('--- decision log')
    for line in compiled.log:
        print(line)
    if request.simulate:
        for mode in ('static', 'dynamic', 'hybrid'):
            variant = replace(request, options={**request.options.to_dict(), 'force_mode': mode})
            candidate, _ = _plan(variant)
            result = simulate(candidate, seed=request.seed)
            print(f'sim[{mode:7s}] {result.summary()}')
    return 0
