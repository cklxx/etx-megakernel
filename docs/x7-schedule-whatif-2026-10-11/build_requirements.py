#!/usr/bin/env python3
"""Attach concrete future-build requirements to every timed case."""
import json
from pathlib import Path
O=Path(__file__).resolve().parent
rows=[]
for source in ('results.json','secondary-results.json','branch-results.json','oracle-results.json','finalist-results.json'):
 for r in json.loads((O/source).read_text()):
  if 'span_us' not in r:continue
  changes=['Preserve original tile functions, argument blocks, coordinates, bf16 and all dependency edges.',
           'Replace TaskInst.worker and Plan.static_queues before emit_plan_header; verify queue order and worker coverage.',
           'Regenerate plan_data.h and plan/kernel output with kept flags; rerun full same-build import checks and 32/32 greedy tokens.']
  mapping=r.get('map','original')
  if mapping=='rotate_heads':changes.insert(1,'Set prep/attn/reduce physical domains to (original_domain + layer%8)%8; intact head groups keep DOMAIN scope.')
  if mapping=='dense_balance':changes.insert(1,'Override dense qkv/o_proj/router/lm_head domain placement with the equal-count median-cost partition; keep coordinates and DEVICE barriers.')
  if mapping=='attention_wide':changes.insert(1,'Use existing ETX_VL_ATTN_WIDE=1 (model.py:144-145,591); derive DEVICE scopes and last-arriver domain shares from new placement before materializing queues.')
  if r['id']=='one_xcd_domain':changes.insert(1,'Place all tasks on XCD0, 38 workers; derive all scopes DOMAIN; no new payload data or fewer byte transfers.')
  if r['id'] in ('narrow6','narrow14'):changes.insert(1,'Use only the first 6 or 14 logical worker slots per XCD for the queues; retain every original task, event and persistent-worker capacity.')
  if r['id'].startswith('serial_head'):changes=['New composite call wrapper: invoke original prep, five attn tasks then reduce serially with original contexts; external qkv and o_proj gates retained.','Remove internal event/fetch operations only after deriving actual wrapper overhead; current optimistic bound is not an implementable timing certificate.','Full import-call checks and 32/32 greedy tokens; resource footprint and wrapper overhead need a real compiler build.']
  if r['id'].startswith('owner_prep'):changes=['New prep+owner-attn wrapper, same original calls and contexts; publish prep event after wrapper completion, explicitly delaying all sibling partitions.','Retain all attention partial producers and reduce gate; no early mid-body signal in current whole-body plan.','Full import-call checks, token checks, and timestamps pricing delayed siblings and wrapper overhead.']
  if r.get('clairvoyant'):changes.insert(1,'Use retained oracle-best-layout.json/apply_layout.py for nominal winner, or regenerate model-specific oracle table from original inputs; table is frozen, runtime does not observe future durations.')
  rows.append(dict(source=source,candidate=r['id'],model=r.get('model',r.get('local_floor_us','additive')),build_changes=changes))
(O/'build-requirements.json').write_text(json.dumps(rows,separators=(',',':'),sort_keys=True)+'\n')
print(f'Concrete build requirements for {len(rows)} timed cases; wave-only proposal is in wave-scope.json')
