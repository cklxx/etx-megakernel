"""Single-process adapter for the pinned image's benchmark_moe.py (no Ray).

Only BLOCK_SIZE_M is pinned; upstream ROCm validity pruning is retained.
Source contracts fail closed before executing the adapted module.
"""
import argparse
import ast
from itertools import product
from pathlib import Path
import time

from examples.vllm_llm.vlbest3_g0 import find_tuner
from examples.vllm_llm.vlbest_report import MOE_FILENAME


def adapted_source(source):
    tree = ast.parse(source)
    body = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == 'ray' for a in node.names):
            found.add('ray'); continue
        if isinstance(node, ast.ImportFrom) and node.module == 'ray.experimental.tqdm_ray':
            found.add('tqdm'); continue
        if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            found.add('main'); continue
        if isinstance(node, ast.ClassDef) and node.name == 'BenchmarkWorker':
            if len(node.decorator_list) != 1 or 'ray.remote' not in ast.unparse(node.decorator_list[0]):
                raise ValueError('BenchmarkWorker decorator changed')
            node.decorator_list = []
            init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            assign = next(n for n in init.body if isinstance(n, ast.Assign) and
                          any(ast.unparse(t) == 'self.device_id' for t in n.targets))
            if ast.unparse(assign.value) != 'int(ray.get_gpu_ids()[0])':
                raise ValueError('worker device contract changed')
            assign.value = ast.Constant(0)
            found.add('worker')
        body.append(node)
    if found != {'ray', 'tqdm', 'main', 'worker'}:
        raise ValueError(f'tuner source contract changed: {found}')
    tree.body = body
    ast.fix_missing_locations(tree)
    if any(isinstance(n, ast.Name) and n.id == 'ray' for n in ast.walk(tree)):
        raise ValueError('unhandled Ray dependency')
    return tree


def tuning_space(original):
    ranges = original(True)
    ranges['BLOCK_SIZE_M'] = [16]
    return [dict(zip(ranges, values)) for values in product(*ranges.values())]


def progress(items):
    start = last = time.monotonic()
    print(f'TUNER pruned_points={len(items)}', flush=True)
    for index, item in enumerate(items):
        now = time.monotonic()
        if now - last >= 60:
            print(f'TUNER point={index}/{len(items)} elapsed={now-start:.0f}s', flush=True)
            last = now
        yield item


def run(args, namespace):
    torch = namespace['torch']
    config = namespace['get_config'](model=args.model, trust_remote_code=False)
    shape = namespace['get_model_params'](config)
    if shape != (128, 8, 768, 2048) or args.tp_size != 1 or args.batch_size != 1:
        raise ValueError(f'unexpected tuning shape/TP/batch: {shape}')
    if not namespace['current_platform'].is_rocm():
        raise ValueError('ROCm required')
    # Explicit bf16 override, including callers in the loaded module.
    namespace['resolve_dtype'] = lambda config: torch.bfloat16
    dtype = namespace['resolve_dtype'](config)
    space = tuning_space(namespace['get_rocm_tuning_space'])
    print(f'TUNER dtype=bf16 E=128 N=768 H=2048 topk=8 batch=1 tp=1 raw_points={len(space)}', flush=True)
    filename = namespace['get_config_file_name'](128, 768, None, None)
    if filename != MOE_FILENAME:
        raise ValueError(f'filename mismatch: {filename}')
    worker = namespace['BenchmarkWorker'](0)
    # Upstream tune computes is_fp16 from the three quantization switches;
    # all false includes bf16. shard_intermediate_size includes both projections.
    winner = worker.tune(1, 128, 1536, 2048, 8, dtype, False, False, False, space, None, False)
    if not winner:
        raise ValueError('no tuning winner')
    row = namespace['sort_config'](winner)
    folder = Path(args.save_dir)
    folder.mkdir(parents=True, exist_ok=True)
    # Use the upstream writer, but verify its bf16 filename before writing.
    dtype_str = namespace['_get_config_dtype_str'](dtype, use_int8_w8a16=False,
        use_fp8_w8a8=False, use_int4_w4a16=False)
    if dtype_str is not None:
        raise ValueError(f'bf16 must have no dtype filename segment: {dtype_str}')
    namespace['save_configs']({1: row}, 128, 1536, 2048, 8, dtype,
                              False, False, False, None, str(folder))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tp-size', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--save-dir', required=True)
    args = parser.parse_args(argv)
    import vllm
    path = find_tuner(Path(vllm.__file__).resolve().parent)
    namespace = {'__name__': 'vlbest3_adapted_tuner', '__file__': str(path), 'tqdm': progress}
    exec(compile(adapted_source(path.read_text()), str(path), 'exec'), namespace)
    return run(args, namespace)


if __name__ == '__main__':
    raise SystemExit(main())
