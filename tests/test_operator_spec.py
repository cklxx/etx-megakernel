"""CPU checks of the generated host loader and the argument packer."""
import json
from pathlib import Path
import subprocess

import pytest

from etx.importer.adapter import emit_host_header
from etx.importer.argblock import pack
from etx.importer.spec import OPERATORS, emit_host_files
from etx.importer.triton_import import import_triton
from etx.importer.launch_spec import CAPTURE_ROOT

ROOT = Path(__file__).resolve().parents[1]


def gemv_info():
    types = ['i32']*6 + ['ptr']*4 + ['i32']*2
    params, off = [], 0
    for ty in types:
        size = 8 if ty == 'ptr' else 4
        off = (off+size-1)//size*size
        params.append(dict(ty=ty, offset=off))
        off += size
    return dict(export='etx_imp_wvsplitk_y2', params=params, grid_offset=off,
                arg_bytes=off+16, block=[64,16,1])


@pytest.fixture(scope='module')
def imports():
    result = {'wvsplitk_y2': gemv_info()}
    for name in ('fused_moe_up', 'fused_moe_down', 'fused_moe_up_e8', 'fused_moe_down_e8'):
        _, info, _, _ = import_triton(CAPTURE_ROOT/name, 'fused_moe_kernel', 'etx_imp_'+name, 1024, 1)
        result[name] = info.to_json()
    return result


def values_for(name):
    if name == 'wvsplitk_y2':
        return [4096,4096,4096,32000,1,1,0,0,0,0,16,256]
    spec = OPERATORS[name]
    values = [0]*len(next(iter(spec.arguments.values())))
    for index, _, expression in spec.checks:
        values[index] = int(expression)
    return values


@pytest.fixture(scope='module')
def loader(tmp_path_factory, imports):
    directory = tmp_path_factory.mktemp('operator-loader')
    (directory/'imports.h').write_text(emit_host_header(imports))
    host = (ROOT/'examples/vllm_llm/host.hip').read_text()
    args = host[host.index('struct Args {'):host.index('\nint main(')]
    # The old loader has the same ABI checks but no semantic check.
    old = args.replace('struct Args {', 'struct OldArgs {').replace('  Args(', '  OldArgs(').replace('Args&', 'OldArgs&')
    old = '\n'.join(line for line in old.splitlines() if 'etx_spec_validate' not in line)
    source = '''#include <string>
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "imports.h"
void die(std::string s) { throw std::runtime_error(s); }
''' + args + '\n' + old + '''
int main(int argc, char** argv) {
  const auto* d = etx_imp_find(argv[1]);
  std::vector<unsigned char> data(d->arg_bytes);
  FILE* f = fopen(argv[2], "rb"); fread(data.data(), 1, data.size(), f); fclose(f);
  int gx = atoi(argv[3]);
  if (argc > 5) {
    const char* error = etx_spec_model(argv[1], "moe_up", atoi(argv[5]), atoi(argv[6]),
        atoi(argv[7]), atoi(argv[8]), atoi(argv[9]), atoi(argv[10]), atoi(argv[11]), 16, 5, gx, 1, 1);
    if (error) { puts(error); return 1; }
    return 0;
  }
  try {
    if (argc > 4) { OldArgs a(argv[1], data.data()); a.i=d->nparams; a.grid(gx,1,1); }
    else { Args a(argv[1], data.data()); a.i=d->nparams; a.grid(gx,1,1); }
  } catch (const std::exception& e) { puts(e.what()); return 1; }
  return 0;
}
'''
    (directory/'loader.cpp').write_text(source)
    from etx.importer.bundle import llvm_tool
    compiler = llvm_tool('clang++')
    assert compiler, 'A C++ compiler is required for the loader regression test'
    result = subprocess.run([compiler, '-std=c++17', str(directory/'loader.cpp'), '-o', str(directory/'loader')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return directory/'loader'


@pytest.mark.parametrize('name', ['wvsplitk_y2', 'fused_moe_up', 'fused_moe_down', 'fused_moe_up_e8', 'fused_moe_down_e8'])
def test_host_accepts_captured_configuration(loader, imports, tmp_path, name):
    grid = (256,1,1) if name == 'wvsplitk_y2' else OPERATORS[name].grid()
    data = pack(imports[name], values_for(name), grid)
    path = tmp_path/'args.bin'; path.write_bytes(data)
    result = subprocess.run([str(loader), name, str(path), str(grid[0])], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize('name,bad_grid,parameter', [
    ('fused_moe_up_e8', 896, 'BLOCK_SIZE_N/MOE_BN'),
    ('wvsplitk_y2', 256, 'CuCount'),
])
def test_old_loader_accepts_but_new_loader_rejects(loader, imports, tmp_path, name, bad_grid, parameter):
    import struct
    values = values_for(name)
    good_grid = (256,1,1) if name == 'wvsplitk_y2' else OPERATORS[name].grid()
    data = bytearray(pack(imports[name], values, good_grid))
    if name == 'wvsplitk_y2':
        values[11] = 304
        struct.pack_into('<i', data, imports[name]['params'][11]['offset'], 304)
    path = tmp_path/'args.bin'; path.write_bytes(data)
    command = [str(loader), name, str(path), str(bad_grid)]
    assert subprocess.run(command+['old'], capture_output=True).returncode == 0
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and parameter in result.stdout
    with pytest.raises(ValueError, match=parameter):
        pack(imports[name], values, (bad_grid,1,1))


def test_shape_conflict_names_parameter(imports):
    values = values_for('fused_moe_up_e8'); values[6] = 2048
    with pytest.raises(ValueError, match='K conflicts'):
        pack(imports['fused_moe_up_e8'], values, (3584,1,1))


def test_every_operator_has_arguments_and_effects(tmp_path):
    for op in OPERATORS.values():
        assert op.arguments and op.template and op.supported_shapes
        assert op.reads and op.writes and op.layouts and op.dtypes
    emit_host_files(tmp_path)
    assert '@' not in (tmp_path/'operator_dispatch.inc').read_text()
    assert '@' not in (tmp_path/'operator_helpers.inc').read_text()


@pytest.mark.parametrize('name,xcd,digest', [
    ('qwen3-8b', '0', '5a7267ca8e61d799f214e978739f983b51a1d124c7cafe73c4eb3719abee5de9'),
    ('qwen3-8b', '1', '5a7267ca8e61d799f214e978739f983b51a1d124c7cafe73c4eb3719abee5de9'),
    ('qwen2.5-1.5b', '0', 'ca98012b2f7bc601979ec89ed613439ee74673611537c53dad25c63acd29e56d'),
    ('qwen2.5-1.5b', '1', 'ca98012b2f7bc601979ec89ed613439ee74673611537c53dad25c63acd29e56d'),
    ('qwen3-30b-a3b', '0', 'cc318c90db4e0817a70d484fbab9e83c114d6eeeaeb70773e06c5fe20712ece8'),
    ('qwen3-30b-a3b', '1', 'cc318c90db4e0817a70d484fbab9e83c114d6eeeaeb70773e06c5fe20712ece8'),
    ('mixtral-8x7b', '0', '137c17169ecb3d920ba91d8c49d1d2c7aadec6d943a03ce47f099b641700b1ec'),
    ('mixtral-8x7b', '1', '3d3ff5e648d2d76d4d29ec8314592214f0135e09beb5baf0da67dcd81f40b191'),
])
def test_launch_records_match_before_change(monkeypatch, name, xcd, digest):
    """These digests describe the launch records before the specification change."""
    import os
    import hashlib
    import importlib
    from examples.llm.model import load_dims
    for key in os.environ:
        if key.startswith(('ETX_VL_', 'ETX_DSL_', 'ETX_LLM_')):
            monkeypatch.delenv(key)
    monkeypatch.setenv('ETX_VL_IMPORTED', '1')
    monkeypatch.setenv('ETX_VL_XCD', xcd)
    import examples.vllm_llm.model as model
    model = importlib.reload(model)
    dims = load_dims(str(ROOT/f'examples/llm/configs/{name}.json'))
    result = json.dumps(model.instances(dims), sort_keys=True).encode()
    assert hashlib.sha256(result).hexdigest() == digest


def test_build_rejects_changed_binary_constants(imports):
    from dataclasses import asdict
    from etx.importer.spec import validate_import
    from etx.importer.launch_spec import registry
    name = 'fused_moe_up_e8'
    info = imports[name] | {'operator_spec': asdict(OPERATORS[name]), 'capture_spec': registry()[name]}
    validate_import(name, info)
    info = json.loads(json.dumps(info))
    info['operator_spec']['constants']['BLOCK_SIZE_N'] = 64
    with pytest.raises(ValueError, match='BLOCK_SIZE_N'):
        validate_import(name, info)


@pytest.mark.parametrize('name,shape,grid', [
    ('fused_moe_up', [2048,768,128,8,32,4,128], 192),
    ('fused_moe_up_e8', [4096,14336,8,2,32,8,128], 3584),
])
def test_loader_checks_model_shape(loader, imports, tmp_path, name, shape, grid):
    path = tmp_path/'args.bin'
    path.write_bytes(pack(imports[name], values_for(name), (grid,1,1)))
    command = [str(loader), name, str(path), str(grid), 'shape']
    result = subprocess.run(command+list(map(str,shape)), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    shape[0] += 8
    result = subprocess.run(command+list(map(str,shape)), capture_output=True, text=True)
    assert result.returncode == 1 and 'H conflicts' in result.stdout


@pytest.mark.parametrize('kind,digest', [
    ('dispatch', '0ed316382ed6f2e4631ef0d7fae1b8bfa195eaa3a12f737f2df0228d54723fcf'),
    ('helpers', 'ac7fb88fee3a2c3ee10ea431ad709ca69e09ca3d2e827110043a4ec9a18ab017'),
])
def test_host_argument_code_matches_before_change(tmp_path, kind, digest):
    import hashlib
    import re
    emit_host_files(tmp_path)
    source = (tmp_path/f'operator_{kind}.inc').read_text()
    assert hashlib.sha256(re.sub(r'\s+', '', source).encode()).hexdigest() == digest


def model_dims(**changes):
    from dataclasses import replace
    from examples.llm.model import load_dims
    dims = load_dims(str(ROOT/'examples/llm/configs/qwen3-30b-a3b.json'))
    return replace(dims, **changes)


def operator_info(name, block):
    types = {'ptr': 'ptr', 'i32': 'i32', 'i64': 'i64', 'f32': 'float', 'f64': 'double', 'b8': 'i1'}
    args = next(iter(OPERATORS[name].arguments.values()))
    return dict(block=list(block), params=[{'ty': types[ty]} for ty, _ in args])


def test_unused_export_does_not_stop_plan():
    from etx.importer.spec import validate_model_imports
    records = {'silu_and_mul': operator_info('silu_and_mul', (1024, 1, 1))}
    report = validate_model_imports(records, model_dims(), set())
    assert report['used'] == {}
    assert 'silu_and_mul' in report['unused']


@pytest.mark.parametrize('name,block,changes', [
    ('rms_norm_2d', (1024, 1, 1), {'H': 16384}),
    ('rms_norm_2d', (128, 1, 1), {}),
    ('rms_norm_3d', (32, 1, 1), {}),
    ('fused_add_rms_norm', (256, 1, 1), {}),
    ('silu_and_mul', (1024, 1, 1), {}),
    ('rope_neox', (256, 1, 1), {}),
    ('reshape_and_cache', (128, 1, 1), {}),
    ('moe_sum_k8', (128, 1, 1), {}),
])
def test_legal_block_range(tmp_path, name, block, changes):
    from etx.importer.spec import validate_import, validate_model_imports
    records = {name: operator_info(name, block)}
    validate_import(name, records[name])
    validate_model_imports(records, model_dims(**changes))
    result = host_model_check(tmp_path, records, model_dims(**changes), name)
    assert result.returncode == 0, result.stdout


def host_model_check(tmp_path, records, dims, name, *, parts=64):
    from etx.importer.spec import emit_model_validation
    from etx.importer.bundle import llvm_tool
    spec = OPERATORS[name]
    role = next(iter(spec.arguments))
    grid = spec.grid(CU=256, parts=parts, NKV=dims.NKV, NH=dims.NH,
                     heads=dims.NH if role == 'q_norm' else dims.NKV,
                     rows=dims.TOPK if role == 'moe_act' else 1)
    values = [dims.H, dims.MI, dims.E, dims.TOPK, dims.NH, dims.NKV, dims.HD, 16, parts, *grid]
    source = '#include <cstring>\n#include <cstdio>\n' + emit_model_validation(records)
    source += '\nint main() { const char* error = etx_spec_model("' + name + '", "' + role + '", '
    source += ', '.join(map(str, values)) + '); if (error) { puts(error); return 1; } return 0; }\n'
    path = tmp_path/'model.cpp'
    path.write_text(source)
    result = subprocess.run([llvm_tool('clang++'), '-std=c++17', str(path), '-o', str(tmp_path/'model')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return subprocess.run([str(tmp_path/'model')], capture_output=True, text=True)


@pytest.mark.parametrize('name,block', [
    ('rms_norm_2d', (0, 1, 1)),
    ('rms_norm_2d', (1025, 1, 1)),
    ('silu_and_mul', (96, 2, 1)),
    ('reshape_and_cache', (63, 1, 1)),
    ('wvsplitk_y2', (64, 8, 1)),
    ('attn_reduce', (64, 1, 1)),
])
def test_illegal_block_fails_in_python_and_host(tmp_path, name, block):
    from etx.importer.spec import validate_model_imports
    dims = model_dims()
    records = {name: operator_info(name, block)}
    with pytest.raises(ValueError, match='block'):
        validate_model_imports(records, dims, {name})
    result = host_model_check(tmp_path, records, dims, name)
    assert result.returncode == 1 and 'block' in result.stdout


def test_unused_fault_is_recorded_and_missing_export_fails():
    from etx.importer.spec import validate_model_imports
    records = {'silu_and_mul': operator_info('silu_and_mul', (1025, 1, 1))}
    report = validate_model_imports(records, model_dims(), set())
    assert 'block' in report['unused']['silu_and_mul']
    with pytest.raises(ValueError, match='silu_and_mul.*block'):
        validate_model_imports(records, model_dims(), {'silu_and_mul'})
    with pytest.raises(ValueError, match='missing.*rms_norm_2d'):
        validate_model_imports(records, model_dims(), {'rms_norm_2d'})


def test_block_rule_change_reaches_python_and_host(monkeypatch, tmp_path):
    from dataclasses import replace
    from etx.importer.spec import validate_model_imports
    name = 'silu_and_mul'
    monkeypatch.setitem(OPERATORS, name, replace(OPERATORS[name],
                       block_rule=(('32', '512'), ('1', '1'), ('1', '1'))))
    for threads in (31, 32, 512, 513):
        records = {name: operator_info(name, (threads, 1, 1))}
        valid = 32 <= threads <= 512
        if valid:
            validate_model_imports(records, model_dims())
        else:
            with pytest.raises(ValueError, match='block'):
                validate_model_imports(records, model_dims())
        result = host_model_check(tmp_path, records, model_dims(), name)
        assert result.returncode == (0 if valid else 1), result.stdout


@pytest.mark.parametrize('maxlen,valid', [(16384, True), (16385, False), (17000, False)])
def test_reduce_partition_limit(tmp_path, maxlen, valid):
    from etx.importer.spec import grid_for
    dims = model_dims()
    parts = (maxlen + 255)//256
    if valid:
        assert grid_for('attn_reduce', dims, role='reduce', cu=256, parts=parts) == (dims.NH, 1, 1)
    else:
        with pytest.raises(ValueError, match='MAX_PARTS'):
            grid_for('attn_reduce', dims, role='reduce', cu=256, parts=parts)
    records = {'attn_reduce': operator_info('attn_reduce', (128, 1, 1))}
    result = host_model_check(tmp_path, records, dims, 'attn_reduce', parts=parts)
    assert result.returncode == (0 if valid else 1), result.stdout
    if not valid:
        assert 'MAX_PARTS' in result.stdout


def test_old_block_guidance_does_not_change_compiled_abi():
    from dataclasses import asdict
    from etx.importer.spec import validate_import
    name = 'silu_and_mul'
    info = operator_info(name, (1024, 1, 1))
    info['operator_spec'] = asdict(OPERATORS[name]) | {
        'block_rule': ['min(MI//8,1024) if moe else 1024', '1', '1']}
    validate_import(name, info)


def test_double_digit_attention_group_constant():
    assert OPERATORS["attn_mfma16_g12"].constants["G"] == 12


@pytest.mark.parametrize("name,k,valid", [
    ("wvsplitk_y2", 16384, True), ("wvsplitk_y2", 16392, False),
    ("wvsplitk_l40_y2", 16392, True), ("wvsplitk_l40_y2", 20480, True),
    ("wvsplitk_l40_y2", 20488, False),
])
def test_wvsplitk_export_lds_cap(name, k, valid):
    from etx.importer.spec import emit_validation, validate_values
    values = values_for("wvsplitk_y2")
    values[0] = values[1] = values[2] = k
    if valid:
        validate_values(name, values, (256, 1, 1))
    else:
        with pytest.raises(ValueError, match="K exceeds supported shape"):
            validate_values(name, values, (256, 1, 1))
    cap = 20480 if name == "wvsplitk_l40_y2" else 16384
    assert f"p1 > {cap}" in emit_validation({name: gemv_info()})
