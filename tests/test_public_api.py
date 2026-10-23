"""The facade exposes operations, with CPU-verifiable behavior."""
import importlib
import json
import subprocess
import sys
from types import ModuleType

import pytest

import etx
import etx.api as api
from examples import splitk_sum


def test_documented_surface_contains_only_operations():
    expected = {'compile', 'plan', 'run', 'load_model', 'serve'}
    assert set(api.__all__) == expected
    assert len(api.__all__) == len(expected)
    assert etx.__all__ == api.__all__
    namespace = {}
    exec('from etx.api import *', namespace)
    for name in api.__all__:
        operation = getattr(importlib.import_module('etx.api'), name)
        assert callable(operation) and operation.__doc__
        assert not isinstance(operation, ModuleType)
        assert not name.startswith('_')
        assert namespace[name] is operation is getattr(etx, name)
    assert {name for name in namespace if not name.startswith('__')} == expected


def test_plan_and_compile_generate_usable_metadata(tmp_path):
    scheduled = api.plan(splitk_sum.build(), 'gfx942', splitk_sum.bindings(), {})
    out = api.compile(scheduled, tmp_path / 'compiled')
    metadata = json.loads((out / 'plan.json').read_text())
    assert metadata['arch'] == 'gfx942'
    assert len(metadata['descs']) == len(scheduled.tasks)
    assert (out / 'megakernel_d0.hip').is_file()
    assert (out / 'plan_data.h').is_file()
    assert (out / 'etx_tiles.h').is_file()


def test_load_model_validates_and_resolves_contract(tmp_path):
    from tests.test_model_contract import contract
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(contract()))
    model = api.load_model(path)
    assert len(model['config_sha256']) == 64
    assert model['correctness']['golden'] == str(tmp_path / 'golden.json')
    assert model['arch']['ffn'] in ('moe', 'dense')


class StubRuntime:
    def __init__(self):
        self.calls = []

    def decode(self, state, count, eos=-1):
        self.calls.append((state, count, eos))
        if state == 'bad':
            raise RuntimeError('backend failed')
        return [state] * count


def test_run_and_serve_preserve_runtime_arguments_and_order():
    runtime = StubRuntime()
    assert api.run(runtime, 4, 2, eos=9) == [4, 4]
    results = api.serve(runtime, iter([5, 6]), 3, eos=7)
    assert runtime.calls == [(4, 2, 9)]
    assert list(results) == [[5, 5, 5], [6, 6, 6]]
    assert runtime.calls == [(4, 2, 9), (5, 3, 7), (6, 3, 7)]


def test_serve_propagates_backend_failure():
    runtime = StubRuntime()
    results = api.serve(runtime, [1, 'bad', 3])
    assert next(results) == [1]
    with pytest.raises(RuntimeError, match='backend failed'):
        next(results)
    assert runtime.calls == [(1, 1, -1), ('bad', 1, -1)]


@pytest.mark.parametrize('first', ['etx.api', 'examples.llm.model', 'etx.compiler.frontend.contract', 'etx.serve.kv'])
def test_import_order_does_not_require_example_initialization(first):
    code = (f'import {first}; import etx.api; import etx; '
            'assert etx.serve is etx.api.serve')
    if first == 'etx.api':
        code += '; import sys; assert "examples.llm.model" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
