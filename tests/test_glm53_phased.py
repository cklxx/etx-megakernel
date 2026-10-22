"""Phased GLM layers (examples/glm53/phases.h): CPU identity with the one-workgroup body, HF, plan shape."""
import ctypes
import pytest

from examples.glm53 import cpu_check, model, model_build
from examples.glm53.options import Options
from etx.ir import Scope
from etx.passes import compile_graph, verify_plan


def _run(model_, ids, steps):
    out = []
    for _ in range(steps):
        logits, layers = model_(ids)
        out.append((logits, layers))
        ids = logits.argmax(-1).tolist()
    return out


@pytest.mark.parametrize('bf16,fp8,selection', [(False, False, False), (True, False, False), (True, True, False), (False, False, True)])
def test_phased_cpu_body_is_bit_identical_to_layer_body(tmp_path, bf16, fp8, selection):
    """Every tile of every phase grid, at the plan's tile counts and at counts that do not divide the rows."""
    import numpy as np
    pytest.importorskip('torch'); pytest.importorskip('transformers')
    out = tmp_path / 'fixture'
    cpu_check.fixture(out, bf16, fp8, selection=selection)
    ref = cpu_check.CPUModel(out); ref.reset(2)
    want = _run(ref, [3, 4], 20)
    for tiles in (None, 3, 7):
        cand = cpu_check.CPUModel(out, phased=True, tiles=tiles); cand.reset(2)
        for pos, ((a, la), (b, lb)) in enumerate(zip(want, _run(cand, [3, 4], 20))):
            assert np.array_equal(a, b), (tiles, pos)
            assert all(np.array_equal(x, y) for r in range(2) for x, y in zip(la[r], lb[r])), (tiles, pos)
        assert all(np.array_equal(s, t) for r in range(2) for s, t in zip(ref.states[r], cand.states[r]))


@pytest.mark.parametrize('batch,bf16,fp8', [(1, False, False), (2, True, True)])
def test_phased_cpu_body_matches_independent_hf(tmp_path, batch, bf16, fp8):
    pytest.importorskip('torch'); pytest.importorskip('transformers')
    result = cpu_check.check(tmp_path / 'fixture', batch, 32, bf16, fp8, phased=True, tiles=5)
    assert result['status'] == 'PASS' and result['matched'] == batch * 32 and result['phased']


def test_phased_production_schedule_matches_independent_hf(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('transformers')
    result = cpu_check.check(tmp_path / 'production', 1, 8, production_schedule=True, phased=True)
    assert result['status'] == 'PASS' and result['layers'] == 45 and result['experts'] == 288


def test_strides_and_shared_memory_match_the_header(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('transformers')
    out = tmp_path / 'fixture'
    spec, _, _ = cpu_check.fixture(out)
    lib = ctypes.CDLL(str(cpu_check.compile_cpu(out)))
    got = (ctypes.c_int * 8)(); lib.glm_strides(got)
    d = spec['dimensions']; s = model.strides(d)
    assert list(got) == [s[k] for k in ('hc', 'p1', 'p2', 'att', 'mid', 'rt', 'act', 'sel')]
    assert lib.glm_lds_floats() == model.lds_floats(d)


def test_production_phased_plan_spreads_layers_and_keeps_pp2_boundary():
    s = model_build.production_compile_spec(); d = s['dimensions']
    g = model.build_from(s, 1); p = compile_graph(g, 'gfx942', {'B': 1}, {})
    assert not verify_plan(p) and p.n_devices == 2 and p.wg_per_cu == 2
    assert 4 * model.lds_floats(d) <= 32 * 1024          # two workgroups per CU fit the 64 KiB LDS
    assert p.events['e_l23_down'].scope == Scope.SYSTEM and p.events['e_l22_down'].scope == Scope.DEVICE
    assert p.events['e_argmax'].scope == Scope.SYSTEM and p.tensor_placement['l23_out'] == 'fine_grained'
    per_grid = {gr.name: gr.grid[1] for gr in g.grids if len(gr.grid) == 2}
    assert max(per_grid.values()) == model.TILE_CAP and per_grid['l3_mla'] == d['NH'] and per_grid['l0_kda'] == d['LH']
    assert {t.domain for t in p.tasks if t.grid == 'l4_up'} == set(range(8))
    assert [gr.device for gr in g.grids if gr.name.endswith('_down')] == [0] * 24 + [1] * 21


def test_unphased_option_keeps_one_task_per_layer():
    s = model_build.production_compile_spec()
    p = compile_graph(model.build_from(s, 1, Options(phased=0)), 'gfx942', {'B': 1}, {})
    assert len(p.tasks) == 48


@pytest.mark.parametrize('devices', [4, 8])
def test_phased_pipeline_partition_and_system_boundaries(devices):
    s = model_build.production_compile_spec()
    g = model.build_from(s, options=Options(devices=devices))
    p = compile_graph(g, 'gfx942', {'B': 1}, {})
    assert not verify_plan(p) and p.n_devices == devices
    own = model.owners(s, devices)
    layers = s['config']['num_hidden_layers']
    sizes = [sum(own[f'l{i}'] == dev for i in range(layers)) for dev in range(devices)]
    assert min(sizes) > 0 and max(sizes) - min(sizes) <= 1
    assert [gr.device for gr in g.grids if gr.name.endswith('_down')] == [own[f'l{i}'] for i in range(layers)]
    for i in range(layers - 1):
        crosses = own[f'l{i}'] != own[f'l{i+1}']
        assert (p.events[f'e_l{i}_down'].scope == Scope.SYSTEM) == crosses
    assert p.events['e_argmax'].scope == Scope.SYSTEM
    assert next(gr.device for gr in g.grids if gr.name == 'head') == devices - 1
    assert next(gr.device for gr in g.grids if gr.name == 'feedback') == 0


def test_up_pairs_option_changes_only_up_phase_grids():
    s = model_build.production_compile_spec()
    baseline = model.build_from(s)
    candidate = model.build_from(s, options=Options(up_pairs=32))
    old = {gr.name: gr for gr in baseline.grids}
    changed = []
    for gr in candidate.grids:
        if gr.grid != old[gr.name].grid:
            changed.append(gr.name)
            assert gr.name.endswith('_up') and gr.grid[1] <= 608
        assert gr.device == old[gr.name].device
    assert changed
    assert any(gr.grid[1] == 576 for gr in candidate.grids if gr.name.endswith('_up'))
