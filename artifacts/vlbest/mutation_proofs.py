from pathlib import Path
import subprocess
cases = [
 ('prefix-plus-one', 'examples/llm/vllm_bench.py', 'CONFIG_PREFIX = "VLLM-CONFIG "', 'CONFIG_PREFIX = "VLLM-CONFIG  "', 'test_harness_flags_and_all_samples'),
 ('prefix-minus-one', 'examples/llm/vllm_bench.py', 'CONFIG_PREFIX = "VLLM-CONFIG "', 'CONFIG_PREFIX = "VLLM-CONFIG"', 'test_harness_flags_and_all_samples'),
 ('spread-cap-removed', 'examples/vllm_llm/vlbest_report.py', 'spread <= .050 + 1e-12 and spread <', 'spread <', 'test_spread_bound_diagnostic'),
 ('spread-strictness-removed', 'examples/vllm_llm/vlbest_report.py', 'spread < abs(median(deltas))', 'spread <= abs(median(deltas))', 'test_spread_bound_diagnostic'),
 ('vendor-byte-changed', 'examples/vllm_llm/vendor/benchmark_moe.py', '# SPDX', '# mutated SPDX', 'test_vendor_pinned_bytes'),
]
out = Path('artifacts/vlbest/stage-h-mutations.txt')
with out.open('w') as log:
 for label, filename, old, new, test in cases:
  p = Path(filename); original = p.read_text(); assert old in original
  try:
   p.write_text(original.replace(old,new,1))
   run = subprocess.run(['.venv/bin/python','-m','pytest','-q',f'tests/test_vlbest_report.py::{test}'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
   log.write(f'MUTATION {label} exit={run.returncode}\n{run.stdout}\n'); log.flush()
   assert run.returncode == 1, label
  finally:
   p.write_text(original)
print(out.read_text())
