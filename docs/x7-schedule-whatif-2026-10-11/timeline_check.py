#!/usr/bin/env python3
"""Run the requested timeline CLI without leaving inflated input copies in git."""
import gzip,subprocess,sys,tempfile
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
with tempfile.TemporaryDirectory(prefix='x7-timeline-') as temp:
 T=Path(temp)
 for label,plan,raw in [('measured','original-plan.json.gz','measured.bin.gz'),('best','oracle-best-plan.json.gz','oracle-best.bin.gz')]:
  (T/'plan.json').write_bytes(gzip.decompress((O/plan).read_bytes()))
  (T/'trace.bin').write_bytes(gzip.decompress((O/raw).read_bytes()))
  with (O/f'timeline-{label}.txt').open('w') as f:
   subprocess.run([sys.executable,'-m','etx.tools.timeline',str(T/'plan.json'),str(T/'trace.bin'),'--bytes',str(ROOT/'docs/xcd-timeline-2026-10-11/bytes.json'),'--json',str(O/f'timeline-{label}.json')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  with (O/f'timeline-{label}-layer24.txt').open('w') as f:
   subprocess.run([sys.executable,'-m','etx.tools.timeline',str(T/'plan.json'),str(T/'trace.bin'),'--grids','_24$','--bytes',str(ROOT/'docs/xcd-timeline-2026-10-11/bytes.json'),'--html',str(O/f'timeline-{label}-layer24.html')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('timeline CLI completed: measured and best, full step plus layer 24',flush=True)
