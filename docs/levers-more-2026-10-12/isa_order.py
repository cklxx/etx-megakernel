#!/usr/bin/env python3
"""Prologue order of global loads, vmcnt waits and barriers in the kept build's DSL tiles (local LLVM 23, [E]).

  .venv/bin/python docs/levers-more-2026-10-12/isa_order.py > docs/levers-more-2026-10-12/isa_order.txt

Reuses docs/master-table-2026-10-11/isa.py's wrapper and flags. ETX_BUILD points at the certificate build's
tile sources (default ~/code/etx/build/qwen300-rocm-on; dsl_tiles.hip sha256 a0bcc2dc...)."""
import hashlib, importlib.util, os, re, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('isa', HERE.parent / 'master-table-2026-10-11/isa.py')
isa = importlib.util.module_from_spec(spec); spec.loader.exec_module(isa)
isa.B = Path(os.environ.get('ETX_BUILD', Path.home() / 'code/etx/build/qwen300-rocm-on')); isa.SRC = isa.B / 'dsl_tiles.hip'
print('dsl_tiles.hip sha256', hashlib.sha256(isa.SRC.read_bytes()).hexdigest()[:8], '| flags', ' '.join(isa.FLAGS))
src = isa.SRC.read_text().split('\n')
if os.environ.get('ISA_S'):   # a .s compiled elsewhere, e.g. by ROCm 7.2.4 clang 22 (see isa_order_rocm22.sh)
    s = Path(os.environ['ISA_S']).read_text().split('\n')
    print('assembly', os.environ['ISA_S'], 'sha256', hashlib.sha256('\n'.join(s).encode()).hexdigest()[:8])
else:
    with tempfile.TemporaryDirectory() as w:
        s = isa.compile_s(Path(w)).read_text().split('\n')
dsl = {int(m.group(1)) for m in (re.match(r'\s*\.file\s+(\d+)\s+".*?"\s+"(.*?)"', l) or re.match(r'\s*\.file\s+(\d+)\s+"(.*?)"', l) for l in s) if m and m.group(2).endswith('dsl_tiles.hip')}
for fn in ('k_topk_rep', 'k_moe_up_x_fold', 'k_moe_down_x_fold'):
    st = next(i for i, l in enumerate(s) if l.startswith(fn + ':'))
    line, out = None, []
    for l in s[st:]:
        m = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)', l)
        if m: line = int(m.group(2)) if int(m.group(1)) in dsl else -int(m.group(2)); continue
        t = l.strip()
        if t.startswith('s_endpgm') or ('load_dwordx4' in t and ' nt' in t): break   # stop at the weight stream
        if re.match(r'(global|flat)_(load|store)|s_waitcnt\s+vmcnt|s_barrier', t):
            out.append(f'  L{abs(line) if line else 0}: {t.split(",")[0][:44]:44s} | {src[line - 1].strip()[:70] if line and line > 0 else f"(other file, line {-line})" if line else ""}')
    print(f'== {fn}: global loads, vmcnt waits and barriers before the first NT weight load (static order)')
    print('\n'.join(out[:24]))
