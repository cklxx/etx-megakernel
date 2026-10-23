"""examples/vllm_llm/batch_native.sh, moved from tests/test_qwen_batch_correctness.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_native_script_supervises_both_compiler_stages(tmp_path):
    toolchain = tmp_path / 'synthetic-compiler'
    toolchain.mkdir()
    # Only transport/orchestration is tested here; these are NOT GPU compilers.
    for name in ('clang++', 'llc'):
        p = toolchain / name
        p.write_text(f'#!{sys.executable}\nimport pathlib, sys\n'
                     'pathlib.Path(sys.argv[sys.argv.index("-o")+1]).write_text("synthetic compiler fixture\\n")\n')
        p.chmod(0o755)
    out = tmp_path / 'batch'
    d = out / 'b2'
    d.mkdir(parents=True)
    (d / 'megakernel_d0.hip').write_text('#include "/code/fixture.h"\n')
    (d / 'imports.bc').write_bytes(b'synthetic')
    run = run_command(['bash', 'examples/vllm_llm/batch_native.sh', str(out), '2'], cwd=ROOT, timeout=20,
                      env=dict(os.environ, ETX_LLVM=str(toolchain), ETX_QWEN_NATIVE_SECONDS='10'))
    assert run.returncode == 0, run.stdout
    receipt = json.loads((d / 'native-check.json').read_text())
    assert receipt['status'] == 'COMPILED' and receipt['gpu_correctness'] == 'unmeasured'
    assert all((d / name).exists() for name in ('native-ir.log', 'native-isa.log'))
