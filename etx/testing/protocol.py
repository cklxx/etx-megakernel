"""Protocol seam probes; mutable plans and deliberate faults stay internal."""
import copy
from dataclasses import asdict, FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from etx.api.types import Configuration
from etx.experiments import protocol
from etx.sim import simulate


CASE = {'name': 'small', 'workload': 'splitk', 'size': 2, 'duration_cv': 0.1}


def protocol_conformance(*, mode='static', fault=None):
    if mode not in protocol.MODES:
        raise ValueError('unknown protocol mode')
    allowed = (None, 'missing', 'duplicate', 'nan', 'sync', 'deadlock', 'budget', 'inflated', 'last-producer')
    if fault not in allowed:
        raise ValueError('unknown protocol fault')
    case = {**CASE, 'size': 1, 'duration_cv': 0} if fault == 'last-producer' else CASE
    planned = protocol.make_plan(case, 'gfx942', 'dynamic' if fault == 'last-producer' else mode)
    if fault == 'last-producer':
        producers = [task for task in planned.tasks if task.grid == 'partial_sum']
        producers[0].duration_us = 1000
    result = asdict(simulate(planned, seed=7))
    spec = protocol.contract(planned)
    task = next(segment for segment in result['segments'] if segment['kind'] == 'task')
    if fault == 'missing':
        result['segments'].remove(task)
    elif fault == 'duplicate':
        result['segments'].append(copy.copy(task))
    elif fault == 'nan':
        result['makespan_us'] = float('nan')
    elif fault == 'sync':
        consumer_id = next(int(t) for t, deps in spec['dependencies'].items() if deps)
        next(s for s in result['segments'] if s['kind'] == 'task' and s['task'] == consumer_id)['start'] = 0
    elif fault == 'deadlock':
        result['deadlock'] = True
    elif fault == 'budget':
        result['busy_us'] += 1
    elif fault == 'inflated':
        result['makespan_us'] *= 100
    elif fault == 'last-producer':
        consumer = next(s for s in result['segments'] if s['kind'] == 'task' and planned.tasks[s['task']].grid == 'final_sum')
        latest = max(s['end'] for s in result['segments'] if s['kind'] == 'task' and planned.tasks[s['task']].grid == 'partial_sum')
        if consumer['start'] < latest + planned.machine.t_sync_us(planned.events['E'].scope):
            raise AssertionError('consumer started before the last producer completed')
    return Configuration({'backend': 'simulation', 'errors': protocol.check_sample(spec, result)})


def provenance_conformance(root, *, layout='git'):
    root = Path(root)
    if layout in ('copied', 'installed'):
        from etx.artifacts.identity import verify_source_snapshot
        source = root / 'etx/operators/fixture.py'
        source.parent.mkdir(parents=True)
        source.write_text('original\n')
        if layout == 'copied':
            for name in ('examples/moe_layer.py', 'examples/splitk_sum.py', 'pyproject.toml'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original\n')
        before = protocol.provenance(root)
        receipt = {key: value for key, value in before.items()
                   if key not in ('python', 'host', 'created_utc')}
        verify_source_snapshot(root, receipt)
        source.write_text('changed\n')
        try:
            verify_source_snapshot(root, receipt)
        except ValueError:
            refused = True
        else:
            raise AssertionError('source mutation was not refused')
        return Configuration(dict(scope=before['scope'], commit=before['commit'],
                                  dirty=before['dirty'], mutation_refused=refused))
    if layout != 'git':
        raise ValueError('unknown provenance layout')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    names = ['driver.sh', 'host.hip', 'runtime.h', 'model.tla', 'model.cfg', 'proof.lean', 'workflow.yml']
    for name in names:
        (root / name).write_text('original\n')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=test', '-c',
                    'user.email=test@example.com', 'commit', '-qm', 'fixture'], check=True)
    before = protocol.provenance(root)
    for name in names:
        (root / name).write_text('changed\n')
    after = protocol.provenance(root)
    if before['commit'] != after['commit'] or any(before['source_sha256'][name] == after['source_sha256'][name] for name in names):
        raise AssertionError('provenance failed to bind non-Python source changes')
    return Configuration({'non_python_sources': len(names), 'commit_unchanged': True})


def _provider_signature(mod):
    """Same size-2 signature as the retained pre-extraction capture script."""
    from etx.contracts.emission import emit_payloads
    from etx.passes import compile_graph, PassOptions

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    bindings = mod.bindings(**({'B': 2} if hasattr(mod, 'NE') else {'n': 2}))
    runtime = mod.runtime(bindings)
    graph = digest(json.dumps(asdict(mod.build()), sort_keys=True, default=str))
    plans = {}
    for mode in protocol.MODES:
        plan = compile_graph(mod.build(), 'gfx942', bindings, runtime, PassOptions(force_mode=mode))
        files, _, _ = emit_payloads(plan, all_rank_headers=False, runtime_headers=False)
        plans[mode] = dict(contract=digest(json.dumps(protocol.contract(plan), sort_keys=True)),
                           files={name: digest(text) for name, text in files.items()})
    value = dict(constants={key: getattr(mod, key) for key in ('D','H','HD','S','NE','K','FF','TPT','T') if hasattr(mod, key)},
                 bindings=bindings, runtime=runtime, graph=graph, plans=plans)
    return digest(json.dumps(value, sort_keys=True))


def operator_provider_conformance(*, small=False, kind='legacy'):
    """CPU-only provider identity, legacy reload, and example-free import probes."""
    if type(small) is not bool or kind not in ('legacy', 'library', 'isolated'):
        raise ValueError('invalid operator provider probe')
    if kind == 'isolated':
        script = '''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class NoExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'examples' or fullname.startswith('examples.'):
            raise ImportError('example import forbidden: ' + fullname)
sys.meta_path.insert(0, NoExamples())
from etx.testing.protocol import operator_provider_conformance
print(json.dumps(operator_provider_conformance(small=sys.argv[2] == '1', kind='library').to_dict()))
'''
        result = subprocess.run([sys.executable, '-I', '-c', script, str(protocol.ROOT),
                                 '1' if small else '0'], capture_output=True, text=True,
                                timeout=30, check=True)
        return Configuration(json.loads(result.stdout))

    from etx.operators import splitk_sum
    from etx.operators.moe_layer import provider
    from etx.experiments import studies
    moe = provider(small=small)
    signatures = {'splitk': _provider_signature(splitk_sum), 'moe': _provider_signature(moe)}
    try:
        moe.D = 1
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('provider dimensions must be immutable')
    with patch.dict(os.environ, {'ETX_MOE_SMALL': '1' if small else '0'}):
        study = studies.run('routing-skew', values=[0.0], size=2)
        case = dict(CASE, workload='moe')
        if small:
            try:
                protocol.make_plan(case, 'gfx942', 'hybrid')
            except ValueError as error:
                if 'unset ETX_MOE_SMALL' not in str(error):
                    raise
            else:
                raise AssertionError('frozen benchmark accepted small dimensions')
        else:
            frozen = protocol.make_plan(case, 'gfx942', 'hybrid')
            if frozen.graph.tensors['x'].shape != ('B', 1024):
                raise AssertionError('frozen benchmark dimensions changed')

    result = dict(signatures=signatures, immutable=True, study=study,
                  frozen_small_refused=small)
    if kind == 'legacy':
        from etx.contracts.compiler import _load_example
        # Mirror real graph-file loading without leaking loader sys.path/module
        # changes or environment switches into another conformance test.
        with patch.dict(os.environ), patch.dict(sys.modules), patch.object(sys, 'path', list(sys.path)):
            os.environ['ETX_MOE_SMALL'] = '1' if small else '0'
            for name, expected in (('splitk_sum', signatures['splitk']), ('moe_layer', signatures['moe'])):
                if _provider_signature(_load_example(protocol.ROOT / 'examples' / (name + '.py'))) != expected:
                    raise AssertionError('legacy/provider signature mismatch')
            loaded = []
            for flag in (False, True, False):
                os.environ['ETX_MOE_SMALL'] = '1' if flag else '0'
                loaded.append(_load_example(protocol.ROOT / 'examples/moe_layer.py'))
            dimensions = [mod.D for mod in loaded]
            if dimensions != [1024, 256, 1024]:
                raise AssertionError('cached provider poisoned fresh example loading')
            for mod, flag in zip(loaded, (False, True, False)):
                if _provider_signature(mod) != _provider_signature(provider(small=flag)):
                    raise AssertionError('previously loaded provider mutated after reload')
            result['reload_dimensions'] = dimensions
    return Configuration(result)
