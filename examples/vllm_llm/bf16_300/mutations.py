"""Remove each real guard/token, prove landing BEFORE pytest, then restore.

Host only. Tests fail under mutation; a final full suite is a separate gate.
Never run concurrently with pytest/builds in this worktree.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results_bf16_300_cpu_20261011/mutations'


def guard_mutations():
    for name in ('protocol', 'check', 'source', 'build', 'dispatcher', 'attribution'):
        path = HERE / (name + '.py')
        text = path.read_text()
        lines = text.splitlines(keepends=True)
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'require' and
                    len(node.args) == 2 and isinstance(node.args[1], ast.Constant)):
                token = ast.get_source_segment(text, node)
                invariant = node.args[1].value
                yield path, invariant, token, f'require(True, {invariant!r})', f'test_guard_rejection[{invariant}]'


def device_mutations():
    # Read test data without executing tests. It is intentionally separate from
    # implementation, so restoring/removing a device invariant cannot change it.
    import runpy
    tests = runpy.run_path(str(ROOT / 'tests/test_bf16_300.py'))
    for number, token in enumerate(tests['DEVICE_TOKENS']):
        # Token replacement may remove a call prefix. Source correspondence is
        # the relevant CPU check; it does NOT pretend to execute mutant GPU code.
        yield HERE / 'fusion.hip', f'device_{number:02d}', token, '/* MUTATED_DEVICE_TOKEN */', 'test_device_contract_source'
    for number, token in enumerate(tests['FIXED_TOKENS']):
        yield HERE / 'source.py', f'fixed_{number:02d}', token, '/* MUTATED_FIXED_TOKEN */', 'test_fixed_shape_keeps_down_original_and_up_math_order'
    for number, token in enumerate(tests['PROBE_TOKENS']):
        replacement = "'', # MUTATED_PROBE_BODY_WINDOW\n" if token == "mark('body_begin')," else '/* MUTATED_PROBE_TOKEN */'
        yield HERE / 'source.py', f'probe_{number:02d}', token, replacement, 'test_probe_bounded_state_contract'


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = []
    mutations = list(guard_mutations()) + list(device_mutations())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for path, name, token, replacement, test in mutations:
        original = path.read_text()
        before = original.count(token)
        assert before == 1, (name, before)
        changed = original.replace(token, replacement, 1)
        try:
            path.write_text(changed)
            # Read BACK the actual edited file. Landing counts are persisted and
            # printed before invoking or reading any test verdict.
            landed = path.read_text()
            after, replacement_count = landed.count(token), landed.count(replacement)
            assert after == 0 and replacement_count == 1, (name, after, replacement_count)
            landing = {'mutation': name, 'path': str(path.relative_to(ROOT)), 'before': before,
                       'after': after, 'replacement_count': replacement_count, 'token': token}
            (RESULTS / (name + '.landing.json')).write_text(json.dumps(landing, indent=2) + '\n')
            print(json.dumps(landing), flush=True)
            command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_bf16_300.py::' + test] if '[' in test else [sys.executable, '-m', 'pytest', '-q', 'tests/test_bf16_300.py', '-k', test]
            with (RESULTS / (name + '.pytest.log')).open('w') as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            verdict = (RESULTS / (name + '.pytest.log')).read_text()
            assert completed.returncode == 1 and 'failed' in verdict, (name, completed.returncode, verdict)
            results.append(dict(landing, exit_code=completed.returncode, verdict=verdict.strip().splitlines()[-1]))
            print(f'{name}: RED {results[-1]["verdict"]}', flush=True)
        finally:
            path.write_text(original)
    (RESULTS / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'MUTATIONS {len(results)}/{len(mutations)} RED; all source files restored', flush=True)


if __name__ == '__main__':
    main()
