from pathlib import Path
import subprocess
import importlib.util
cases = [
 ('prefix-plus-one', 'examples/llm/vllm_bench.py', 'CONFIG_PREFIX = "VLLM-CONFIG "', 'CONFIG_PREFIX = "VLLM-CONFIG  "', 'test_harness_flags_and_all_samples'),
 ('prefix-minus-one', 'examples/llm/vllm_bench.py', 'CONFIG_PREFIX = "VLLM-CONFIG "', 'CONFIG_PREFIX = "VLLM-CONFIG"', 'test_harness_flags_and_all_samples'),
 ('spread-cap-removed', 'examples/vllm_llm/vlbest_report.py', 'spread <= .050 + 1e-12 and spread <', 'spread <', 'test_spread_bound_diagnostic'),
 ('spread-strictness-removed', 'examples/vllm_llm/vlbest_report.py', 'spread < abs(median(deltas))', 'spread <= abs(median(deltas))', 'test_spread_bound_diagnostic'),
 ('parser-offset-plus-one', 'examples/llm/vllm_bench.py', 'line[len(CONFIG_PREFIX):]', 'line[len(CONFIG_PREFIX)+1:]', 'test_harness_flags_and_all_samples'),
 ('diagnostic-removed', 'examples/vllm_llm/vlbest_report.py', "print(f'SPREAD-BOUND", "print(f'DISABLED-SPREAD", 'test_report_spread_diagnostic_roundtrip'),
 ('diagnostic-changes-verdict', 'examples/vllm_llm/vlbest_report.py', "return 'INSUFFICIENT' if unresolved and med_abs <= spread else verdict(deltas)", "return 'INSUFFICIENT' if not spread_bound(deltas) else verdict(deltas)", 'test_spread_bound_does_not_change_verdict'),
 ('portable-home-guard', 'tests/test_vlbest_report.py', 'import json', 'import json\nfrom pathlib import Path\nNONPORTABLE = Path.home()', 'tests/test_test_paths.py::test_test_modules_have_no_home_dependencies'),
 ('budget-vm-life', 'examples/sessions/vlbest.py', 'VM_LIFE_SECONDS = 150 * 60', 'VM_LIFE_SECONDS = 151 * 60', 'tests/test_vlbest_session.py::test_budget_and_graph'),
 ('budget-setup-allowance', 'examples/sessions/vlbest.py', 'SETUP_ALLOWANCE_SECONDS = 10 * 60', 'SETUP_ALLOWANCE_SECONDS = 9 * 60', 'tests/test_vlbest_session.py::test_budget_and_graph'),
 ('budget-runner-cap', 'examples/sessions/vlbest.py', 'MAX_SESSION_SECONDS = VM_LIFE_SECONDS - SETUP_ALLOWANCE_SECONDS', 'MAX_SESSION_SECONDS = VM_LIFE_SECONDS', 'tests/test_vlbest_session.py::test_budget_and_graph'),
 ('donor-ambiguity', 'examples/sessions/vlbest.py', 'if len(candidates) != 1:', 'if False:', 'tests/test_vlbest_session.py::test_donor_ambiguity_refuses'),
 ('null-combination', 'examples/sessions/vlbest.py', 'if cfg is None:', 'if False:', 'tests/test_vlbest_session.py::test_no_combination_skips_without_process'),
 ('certificate-dependency', 'examples/sessions/vlbest.py', "after=['select_final', 'check_certificate']", "after=['select_final']", 'tests/test_vlbest_session.py::test_correctness_and_screening_dependencies'),
 ('vendor-byte-changed', 'examples/vllm_llm/vendor/benchmark_moe.py', '# SPDX', '# mutated SPDX', 'test_vendor_pinned_bytes'),
]
out = Path('examples/vllm_llm/results/2026-10-09-vlbest/stage-t-mutations.txt')
with out.open('w') as log:
 for label, filename, old, new, test in cases:
  p = Path(filename); original = p.read_text(); assert old in original
  try:
   p.write_text(original.replace(old,new,1))
   Path(importlib.util.cache_from_source(str(p))).unlink(missing_ok=True)
   run = subprocess.run(['.venv/bin/python','-m','pytest','-q',test if '::' in test else f'tests/test_vlbest_report.py::{test}'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
   log.write(f'MUTATION {label} exit={run.returncode}\n{run.stdout}\n'); log.flush()
   assert run.returncode == 1, label
  finally:
   p.write_text(original)
   Path(importlib.util.cache_from_source(str(p))).unlink(missing_ok=True)
print(out.read_text())
