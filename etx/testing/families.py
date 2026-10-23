"""Executable census of the fourteen scaffold sets (API review section 7).

A set is on the single scaffold when every one of its entry points is a thin
translator over the external API and the installed-package probe executes or
replays its lifecycle. Static checks here; the installed probe keys are asserted
by tests/e2e/test_api_installed.py, so a counted set is backed by both tests.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ('etx.external', 'etx.api')
# Process spawning and dynamic loading belong to the API's executors and providers.
PRIVATE_MECHANISMS = ('subprocess', 'importlib', 'runpy', 'multiprocessing')
# A translator may name its repository root, any number of directories up, then exec an etx module.
ROOT_PROLOGUE_RE = re.compile(r'ROOT="\$\(cd "\$\(dirname "\$0"\)(/\.\.)+" && pwd\)"')


@dataclass(frozen=True)
class Family:
    id: str
    kind: str
    title: str
    entries: tuple
    installed: tuple = ()
    declarations: tuple = ()


FAMILIES = (
    Family('S1', 'top', 'generic graph/operator compiler', ('etx/tools/cli.py', 'tools/cpu_device_check.sh'),
           installed=('rows', 'operators', 'gemm_ranks')),
    Family('S2', 'top', 'GLM full model', ('examples/glm53/run_all.py', 'examples/glm53/model_build.py'),
           installed=('glm_run_all_dry_run', 'glm_model_build_refusal')),
    Family('S3', 'top', 'legacy LLM', ('examples/llm/build.sh', 'examples/llm/prep.py'),
           installed=('llm_build', 'llm_prep')),
    Family('S4', 'top', 'vLLM-kernel LLM', ('examples/vllm_llm/decode.py', 'examples/vllm_llm/build.sh'),
           installed=('vllm_decode_dry_run', 'vllm_build_refusal')),
    Family('S5', 'top', 'Fleet DeepSeek', ('examples/dsv2lite/build.sh', 'examples/dsv2lite/inplace_session.py'),
           installed=('dsv2lite_build', 'dsv2lite_report')),
    Family('S6', 'top', 'sealed experiment program', ('examples/experiment_program/session.py',),
           installed=('catalog_records', 'sealed_preflight_resume', 'sealed_export')),
    Family('S7', 'top', 'GPU scheduler/transport', ('etx/tools/gpu_jobs.py', 'tools/session.sh'),
           installed=('job_session', 'job_record_verified', 'transport_status')),
    Family('S8', 'top', 'CPU protocol experiment/replay',
           ('etx/tools/bench_sim.py', 'etx/tools/cpu_check.py', 'bench/sim_experiments.py'),
           installed=('protocol_replay', 'studies')),
    Family('S9', 'top', 'native microbenchmarks', ('bench/calib/run_calib.sh', 'etx/tools/fresh_handoff.py'),
           installed=('calibration_run', 'fresh_handoff_check')),
    Family('S2a', 'nested', 'GLM boundary', ('examples/glm53/build_boundary.sh',), installed=('glm_boundary_build',)),
    Family('S2b', 'nested', 'GLM seam', ('examples/glm53/seam.py',), declarations=('examples.glm53.seam_graph',),
           installed=('glm_seam_plan', 'glm_seam_summary')),
    Family('S4a', 'nested', 'Qwen BF16 probe', ('examples/vllm_llm/bf16_300/launch_build.py',),
           installed=('bf16_protocol', 'bf16_build_dry_run')),
    Family('S4b', 'nested', 'Qwen expert merge', ('examples/vllm_llm/expert_merge/runner.py',),
           installed=('expert_merge_protocol', 'expert_merge_status')),
    Family('S4c', 'nested', 'Qwen attention early merge', ('examples/vllm_llm/attention_early_merge/session.py',),
           installed=('attention_merge_simulate', 'attention_merge_status')),
)


def _python_violations(path, family):
    violations = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ''] if not node.level else ['.' * node.level + (node.module or '')]
        else:
            continue
        for name in names:
            root = name.split('.')[0]
            if name.startswith('.'):
                violations.append(f'{node.lineno}: relative import {name}')
            elif root == 'etx' and not any(name == p or name.startswith(p + '.') for p in PUBLIC):
                violations.append(f'{node.lineno}: private import {name}')
            elif root == 'examples' and name not in family.declarations:
                violations.append(f'{node.lineno}: example implementation import {name}')
            elif root in PRIVATE_MECHANISMS:
                violations.append(f'{node.lineno}: private mechanism {name}')
    return sorted(violations, key=lambda reason: int(reason.split(':')[0]))


def _shell_violations(path):
    violations = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text or text.startswith('#') or text in ('set -euo pipefail', 'set -eu') or ROOT_PROLOGUE_RE.fullmatch(text):
            continue
        if text.startswith('exec ') and ' -m etx.' in text:
            continue
        violations.append(f'{number}: shell implementation')
    return violations


def entry_violations(family, root=ROOT):
    """Return entry:line reasons why a set is not yet a thin external translator."""
    violations = []
    for entry in family.entries:
        path = Path(root) / entry
        if not path.is_file():
            violations.append(f'{entry}: missing')
            continue
        found = _python_violations(path, family) if path.suffix == '.py' else _shell_violations(path)
        violations.extend(f'{entry}:{reason}' for reason in found)
    return violations


def census(root=ROOT):
    """One row per set: thin entry points, installed coverage and whether it counts."""
    rows = []
    for family in FAMILIES:
        violations = entry_violations(family, root)
        rows.append(dict(id=family.id, kind=family.kind, title=family.title,
                         violations=len(violations), first=violations[:3],
                         installed=family.installed,
                         counted=not violations and bool(family.installed)))
    return rows


def inline_compilers(root=ROOT):
    """Session declarations and example shells that still invoke a compiler directly, not through the API."""
    root = Path(root)
    paths = sorted((root / 'examples/sessions').glob('*.py')) + sorted((root / 'examples').rglob('*.sh'))
    return [str(path.relative_to(root)) for path in paths
            if any(word in path.read_text() for word in ('hipcc', 'clang++', 'llvm-link'))]


def main():
    rows = census()
    for row in rows:
        print(f"{row['id']:<4} {row['kind']:<6} {'ON ' if row['counted'] else 'OFF'} "
              f"violations={row['violations']:<4} installed={','.join(row['installed']) or '-'} "
              f"{row['title']}; {'; '.join(row['first'])}")
    top = sum(row['counted'] for row in rows if row['kind'] == 'top')
    nested = sum(row['counted'] for row in rows if row['kind'] == 'nested')
    print(f'CENSUS top={top}/9 nested={nested}/5 total={top + nested}/14')
    inline = inline_compilers()
    print(f'INLINE-COMPILERS files={len(inline)} (session declarations and example shells calling a compiler directly)')


if __name__ == '__main__':
    main()
