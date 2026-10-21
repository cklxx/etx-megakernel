"""Fail-closed contracts for isolated column experiments (no device execution)."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
from statistics import median

class ProtocolError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise ProtocolError(message)

def stable_groups(picks):
    """First-encounter expert order; original request/slot order remains the fold order."""
    groups = {}
    require(1 <= len(picks) <= 4, 'one to four requests required')
    width = len(picks[0])
    require(width > 0 and all(len(row) == width for row in picks), 'ragged picks')
    for request, row in enumerate(picks):
        require(len(set(row)) == len(row), 'duplicate expert within a request')
        for slot, expert in enumerate(row):
            require(isinstance(expert, int) and expert >= 0, 'invalid expert identity')
            groups.setdefault(expert, []).append((request, slot))
    return [{'expert': expert, 'slots': slots, 'singleton': len(slots) == 1}
            for expert, slots in groups.items()]

REVISIONS = {'fleet': '554133d744b4591c9cf87c68b6817df781137c9b',
             'dsl': 'b5eaa67', 'dsl-expert': 'ff9ef92'}
DAG = {'fleet': 'fleet-fma8-wave-xor32-to1-bf16',
       'dsl': 'dsl-dot-segment-wave-tree-ordered-wave-fold-bf16',
       'dsl-expert': 'dsl-dot-segment-wave-tree-ordered-wave-fold-bf16'}

def sha256(path):
    h = hashlib.sha256()
    with resolve_path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20): h.update(chunk)
    return h.hexdigest()

def validate_capture(c):
    require(c.get('schema') == 1 and c.get('record') in ('G02','G03','G05','G13'), 'unknown capture schema/record')
    family = c.get('family')
    require(family in REVISIONS and c.get('source_revision') == REVISIONS[family], 'source revision mismatch')
    require(c.get('per_column_dag') == DAG[family], 'per-column arithmetic DAG mismatch')
    require(c.get('input_provenance') == 'native-capture', 'native capture required; model goldens are not tile inputs')
    require(c.get('source_bound') is True and c.get('placement_bound') is True, 'source and placement must be bound')
    require(c.get('capture_complete') is True, 'incomplete capture')
    require(c.get('precision') == 'bf16' and c.get('context') == 1024, 'model context/precision mismatch')
    require(c.get('positions') == ([1024,1040] if family=='fleet' else [0,1040]), 'both decode positions required')
    require(c.get('batch') in (2,4), 'batch must be two or four')
    for key in ('N','K','ld','workers','xcds','threads','epoch'):
        require(type(c.get(key)) is int and c[key] > 0, f'invalid {key}')
    require(c['ld'] >= c['K'] and c['K'] % 8 == 0, 'unaligned shape')
    require(c['threads'] == (256 if family == 'fleet' else 1024), 'native worker shape changed')
    require(c['workers'] == (37 if family=='fleet' else 38) and c['xcds'] == 8, 'native worker budget changed')
    arrivals = c.get('arrivals_ticks', [])
    require(len(arrivals) == c['batch'] and all(type(v) is int and v >= 0 for v in arrivals)
            and min(arrivals) == 0, 'invalid readiness population')
    require(c.get('role') in ('qkv','gate_up','down','expert_up_down'), 'unsupported native role')
    for key in ('source_sha256','golden_sha256'):
        digest = c.get(key, '')
        require(len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest), f'invalid {key}')
    payload = c.get('payload', {})
    path = resolve_path(payload.get('path', '/nonexistent'))
    require(path.is_file(), 'native payload missing')
    require(path.stat().st_size == payload.get('bytes'), 'truncated/extended native payload')
    require(sha256(path) == payload.get('sha256'), 'native payload digest mismatch')
    spans = []
    names = set()
    for seg in c.get('segments', []):
        name, offset, n = seg.get('name'), seg.get('offset'), seg.get('bytes')
        require(name not in names, 'duplicate capture segment'); names.add(name)
        require(type(offset) is int and type(n) is int and offset >= 0 and n > 0
                and offset + n <= payload['bytes'], 'segment outside capture')
        require(seg.get('dtype') in ('bf16','fp32','int32'), 'unsupported native dtype')
        require(offset % 4 == 0, 'unaligned native segment')
        require(not any(offset < end and begin < offset+n for begin,end in spans), 'overlapping native segments')
        spans.append((offset,offset+n))
    require({'weights','inputs'} <= names, 'missing weights/activation inputs')
    sizes = {s['name']: s['bytes'] for s in c['segments']}
    require(sizes['weights'] >= c['N'] * c['ld'] * (4 if c['role']=='gate_up' else 2), 'weight shape mismatch')
    require(sizes['inputs'] >= c['batch'] * c['K'] * (4 if family == 'fleet' else 2), 'activation shape mismatch')
    if c['record'] == 'G05':
        groups = stable_groups(c.get('picks', []))
        require(any(not group['singleton'] for group in groups), 'no observed duplicated expert group')
        require(len(c['picks']) == c['batch'], 'pick/request mismatch')
        require({'mix','golden_slots'} <= names, 'missing exact mix/slot outputs')
    return c

class EpochCompletion:
    """Host reachability model, not a substitute for native visibility checks."""
    def __init__(self, tasks, waves, epoch):
        require(tasks > 0 and waves > 0 and 0 < epoch < 2**32, 'invalid completion geometry')
        self.tasks, self.waves, self.epoch = tasks,waves,epoch
        self.done = set()
    def publish(self, task, wave, epoch, *, drained):
        require(epoch == self.epoch, 'stale epoch')
        require(0 <= task < self.tasks and 0 <= wave < self.waves, 'invalid producer')
        require(drained, 'writing wave not drained')
        require((task,wave) not in self.done, 'duplicate completion')
        self.done.add((task,wave))
    def ready(self, epoch):
        require(epoch == self.epoch, 'stale consumer epoch')
        return len(self.done) == self.tasks * self.waves
    def reset(self, epoch):
        require(self.ready(self.epoch), 'early release/reset')
        require(self.epoch < epoch < 2**32, 'epoch rollover requires quiescent reallocation')
        self.epoch = epoch; self.done.clear()

def queue_reachable(queues, dependencies):
    flat = [task for queue in queues for task in queue]
    if len(set(flat)) != len(flat): return False
    done = set(); pending = [list(q) for q in queues]
    while any(pending):
        progressed = False
        for queue in pending:
            if queue and all(p in done for p in dependencies.get(queue[0], [])):
                done.add(queue.pop(0)); progressed = True
        if not progressed: return False
    return True

def decide_diagnostic(control, simultaneous, restored):
    require(all(len(x) == 5 for x in (control,simultaneous,restored)), 'five complete pairs required')
    require(all(math.isfinite(v) and v > 0 for x in (control,simultaneous,restored) for v in x), 'invalid timing')
    deficit = [a-b for a,b in zip(control,simultaneous)]
    recovery = [a-b for a,b in zip(control,restored)]
    dominant = all(d > 0 and r > d/2 for d,r in zip(deficit,recovery)) and median(recovery) > 2*(max(recovery)-min(recovery))
    return {'verdict': 'DOMINANT_NEW_ISOLATED' if dominant else 'PARTIAL_OR_UNKNOWN',
            'historical_attribution':'UNKNOWN', 'deficit':deficit, 'recovery':recovery}

def mechanism_source():
    return Path(__file__).with_name('kernels.hip').read_text()
