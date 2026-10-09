"""Portable guard-removal proofs; restore each source in finally."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
session = root / 'examples/sessions/vlbest.py'
report = root / 'examples/vllm_llm/vlbest_report.py'
tests = 'tests/test_vlbest_session.py'
cases = [
    ('failed tune cannot mark complete', session,
     "cmd='set -euo pipefail; ' + container_prefix() +",
     "cmd='' + container_prefix() +", 'test_failed_tune_never_marks_complete'),
    ('ray probe reports absence', session, "available = False", "available = True",
     'test_ray_probe'),
    ('tune admission budget', session, "if remaining < cost + reserve:",
     "if stage != 'moe' and remaining < cost + reserve:", 'test_budget_guard'),
    ('stage B admission budget', session, "if remaining < cost + reserve:",
     "if stage != 'b' and remaining < cost + reserve:", 'test_budget_guard'),
    ('missing budget fails closed', session,
     "os.environ.get('ETX_SESS_REMAINING_SECONDS', '0')",
     "os.environ.get('ETX_SESS_REMAINING_SECONDS', '99999')", 'test_budget_guard'),
    ('ray gate omits tune/S9', session,
     "gate=('docker_pull', r'^MOE-TUNE ray=yes$'),", "gate=None,", 'test_ray_omits_tune_and_s9'),
    ('budget gate omits configs/S1/S9', session,
     "gate=('moe_budget', r'^MOE-TUNE GO$'),", "gate=None,", 'test_budget_omits_configs_s1_s9'),
    ('completion gate rejects partial tune', session,
     "gate=('moe_tune', r'^MOE-TUNE COMPLETE$') if arm == 'S9' else None,",
     "gate=None,", 'test_incomplete_tune_never_screens_s9'),
    ('stage B gate preserves stage C', session,
     "gate=('b_budget', r'^STAGE-B GO$'),", "gate=None,", 'test_stage_b_reserves_c'),
    ('report retains ray reason', report,
     "if 'MOE-TUNE ray=no' in lines('docker_pull'):",
     "if False:", 'test_ray_omits_tune_and_s9'),
    ('report retains tune budget reason', report,
     "if line.startswith('MOE-TUNE SKIPPED budget '):",
     "if False:", 'test_budget_omits_configs_s1_s9'),
    ('report retains stage B reason', report,
     "if line.startswith('STAGE-B SKIPPED budget '):",
     "if False:", 'test_stage_b_reserves_c'),
    ('excluded arm cannot become timed', report,
     "if arm in result['excluded']:", "if False:", 'test_omission_overrides_arm_logs'),
]
out = Path(__file__).with_name('round3-mutations.txt')
with out.open('w') as log:
    for name, path, old, new, test in cases:
        original = path.read_text()
        assert original.count(old) == 1, (name, old)
        try:
            path.write_text(original.replace(old, new))
            proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', f'{tests}::{test}'],
                                  cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(f'=== REMOVED: {name} ===\n{proc.stdout}exit={proc.returncode}\n')
            assert proc.returncode == 1, (name, proc.stdout)
        finally:
            path.write_text(original)
        proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', f'{tests}::{test}'],
                              cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(f'=== RESTORED: {name} ===\n{proc.stdout}exit={proc.returncode}\n')
        log.flush()
        assert proc.returncode == 0, (name, proc.stdout)
        print(f'{name}: removed RED; restored GREEN', flush=True)
