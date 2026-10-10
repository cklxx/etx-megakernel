"""Pinned-image, CPU-only G0. Source baseline: local vLLM f46a9dfe2.

Do not import fused_moe to resolve a target filename: its platform detection can
initialize a device. Execute only the source's pure filename function, supplying
the explicitly specified MI300X name. Actual hardware is recorded by record.log.
"""
import ast
import json
import os
from pathlib import Path
import re

from examples.vllm_llm.vlbest_report import MOE_FILENAME, MOE_PROOF, config

# benchmark_moe.py:697,1007 (writer); :1048-1072 (CLI), f46a9dfe2.
TUNER = 'benchmarks/kernels/benchmark_moe.py'
FOLDER_VAR = 'VLLM_TUNED_CONFIG_FOLDER'  # envs.py:249,1800; fused_moe.py:1127


def find_tuner(package):
    return next((p / TUNER for p in package.parents if (p / TUNER).is_file()), package.parent / TUNER)


def source_function(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<image-source>', 'exec'), namespace)
    return namespace[name]


def validate(values, sources):
    errors = []
    version = re.match(r'(\d+)\.(\d+)', str(values['torch']))
    if not version or tuple(map(int, version.groups())) < (2, 11):
        errors.append('torch < 2.11 or unknown')
    if values['layername'] is not True:
        errors.append('_USE_LAYERNAME is not True')
    if not values['tuner_exists']:
        errors.append(f'missing {TUNER}')
    if values['folder'] != str(Path.home() / 'moecfg/tuned'):
        errors.append(f'{FOLDER_VAR} value mismatch')
    if values['filename'] != MOE_FILENAME:
        errors.append('config filename mismatch')
    for name in config('M1')['env']:
        if name not in sources['envs']:
            errors.append(f'missing environment variable {name}')
    for literal in (FOLDER_VAR, MOE_PROOF):
        if literal not in sources['moe']:
            errors.append(f'missing loader string {literal}')
    # Verify adapter and CLI assumptions before GPU work. No source edits on VM.
    for literal in ('def resolve_dtype(', 'dtype = resolve_dtype(config)',
                    'def get_model_params(', 'def main(', 'def save_configs(',
                    '--model', '--tp-size', '--batch-size', '--dtype', '--save-dir', '--tune'):
        if literal not in sources['tuner']:
            errors.append(f'tuner contract changed: {literal}')
    try:
        from examples.vllm_llm.vlbest3_tuner import adapted_source
        adapted_source(sources['tuner'])
    except (ValueError, SyntaxError, StopIteration) as exc:
        errors.append(f'tuner adapter source contract: {exc}')
    return errors


def probe():
    values = dict(torch=None, layername=None, tuner_exists=False, tuner_path=None,
                  folder=None, filename=None)
    sources = dict(envs='', moe='', unquantized='', tuner='')
    errors = []
    try:
        import torch
        values['torch'] = str(torch.__version__)
        import vllm
        from vllm import envs
        from vllm.utils import torch_utils
        values['layername'] = torch_utils._USE_LAYERNAME
        values['folder'] = getattr(envs, FOLDER_VAR)
        package = Path(vllm.__file__).resolve().parent
        tuner = find_tuner(package)
        values.update(tuner_path=str(tuner), tuner_exists=tuner.is_file())
        paths = dict(envs=package / 'envs.py',
                     moe=package / 'model_executor/layers/fused_moe/fused_moe.py',
                     unquantized=package / 'model_executor/layers/fused_moe/oracle/unquantized.py',
                     tuner=tuner)
        for key, path in paths.items():
            if path.is_file():
                sources[key] = path.read_text()
        fn = source_function(sources['moe'], 'get_config_file_name',
                             {'get_device_name_as_file_name': lambda: 'AMD_Instinct_MI300X'})
        values['filename'] = fn(128, 768, None, None)
        for key, path in paths.items():
            print(f'G0 source {key}={path}', flush=True)
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    # Always record all four groups, including failures, before returning red.
    print('G0 torch.__version__=' + json.dumps(values['torch']), flush=True)
    print('G0 vllm.utils.torch_utils._USE_LAYERNAME=' + json.dumps(values['layername']), flush=True)
    print('G0 tuner=' + json.dumps({k: values[k] for k in ('tuner_exists', 'tuner_path')}), flush=True)
    print('G0 config=' + json.dumps({FOLDER_VAR: values['folder'], 'filename': values['filename']}), flush=True)
    errors += validate(values, sources)
    for error in errors:
        print(f'PREFLIGHT FAIL g0: {error}', flush=True)
    if not errors:
        print('PREFLIGHT OK g0', flush=True)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(probe())
