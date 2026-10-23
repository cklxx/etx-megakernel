"""Policy conformance, not a workload E2E test."""
from pathlib import Path
import pytest
from etx.testing import check_external_sources, check_api_layers, check_migrated_entry_points


def test_public_and_external_layers_only_reach_supported_adapters():
    assert check_api_layers(Path(__file__).parent.parent / 'etx') == ()


@pytest.mark.parametrize('layer,source', [
    ('external', 'from etx.passes import compile_graph'),
    ('external', 'from ..contracts import planning'),
    ('api', 'from etx.external import model'),
    ('api', 'from ..passes import compile_graph'),
    ('api', 'from examples.glm53 import model'),
])
def test_layer_bypass_is_rejected(tmp_path, layer, source):
    for name in ('api', 'external'):
        (tmp_path / name).mkdir()
    (tmp_path / layer / '__init__.py').write_text(source)
    assert check_api_layers(tmp_path)


def test_migrated_commands_only_import_the_external_facade():
    assert check_migrated_entry_points(Path(__file__).parent.parent) == ()


@pytest.mark.parametrize('source', [
    'from etx.experiments.protocol import run',
    'from etx.api import run_cpu_acceptance',
    'from etx.testing import acceptance_probe',
    'import subprocess',
    'from .private import run',
])
def test_migrated_command_cannot_regain_private_lifecycle(tmp_path, source):
    for name in ('etx/tools/bench_sim.py', 'etx/tools/cpu_check.py', 'bench/sim_experiments.py',
                 'examples/experiment_program/stage.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('from etx.external import run_protocol_study\n')
    assert check_migrated_entry_points(tmp_path) == ()
    (tmp_path / 'etx/tools/bench_sim.py').write_text(source)
    assert check_migrated_entry_points(tmp_path)


def test_missing_migrated_command_is_not_a_pass(tmp_path):
    assert len(check_migrated_entry_points(tmp_path)) == 4


def test_external_tree_has_no_private_imports():
    assert check_external_sources(Path(__file__).parent / 'e2e') == ()


@pytest.mark.parametrize('source', [
    'from etx.passes import compile_graph',
    'import etx.codegen as codegen',
    'from etx.api import model',
    'from examples.glm53 import model',
    'from tests.test_glm53_model import fixture',
    'from .private_helper import fixture',
    'import importlib',
    'import subprocess',
    '__import__("etx.passes")',
    'exec("from etx.passes import compile_graph")',
    'import os\nos.system("python -c anything")',
    'from etx.external import _private',
    'from etx.testing import *',
    'from etx.testing.boundary import check_external_sources',
    'import etx.external._private',
    'from etx.external import internal_runner',
    'from etx.testing import boundary',
    'request.getfixturevalue("private")',
    'from os import system as launch\nlaunch("python anything")',
    'import os as operating\nlaunch = operating.system\nlaunch("anything")',
    'load = __import__\nload("etx.passes")',
    'import builtins as b\nb.eval("anything")',
    'from unittest.mock import patch as substitute\nsubstitute("etx.passes.compile_graph")',
    'import unittest.mock as mock\nmock.patch.object(target, "run")',
    'change = monkeypatch.setattr\nchange("etx.passes.compile_graph", replacement)',
    'getattr(facade, "_private")',
    'import etx.external as facade\nfacade.types',
    'import etx.external\netx.api.model("private", config={})',
    'import etx.external as facade\nfacade.model.__globals__',
    'import sys as system\nsystem.modules["etx.passes"]',
    'from tests.e2e.missing_helper import fixture',
    'import multiprocessing',
    'import asyncio\nasyncio.create_subprocess_exec("anything")',
])
def test_forbidden_helper_is_detected(tmp_path, source):
    helper = tmp_path / 'helpers' / 'fixture.py'
    helper.parent.mkdir()
    helper.write_text(source)
    violations = check_external_sources(tmp_path)
    assert violations
    assert all(str(helper) in item for item in violations)


def test_missing_scan_root_is_not_a_pass(tmp_path):
    with pytest.raises(ValueError, match='not a directory'):
        check_external_sources(tmp_path / 'missing')


def test_supported_imports(tmp_path):
    (tmp_path / 'test_example.py').write_text(
        'from etx.external import model\nfrom etx.testing import check_external_sources\n'
        'import json\nimport pytest\nfrom pathlib import Path\n')
    assert check_external_sources(tmp_path) == ()
