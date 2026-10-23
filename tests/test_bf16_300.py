"""CPU gates for the isolated preparatory prototypes; no GPU certificate."""
from dataclasses import replace
from pathlib import Path
import random
import struct

import numpy as np
import pytest

from etx.models.vllm_llm.bf16_300.protocol import Head, ProtocolError, dry_run, queue_liveness, slot_participation, reset_queues
from etx.models.vllm_llm.bf16_300 import build, check, source, dispatcher, attribution, paths

HERE = paths.HERE   # declarations and native assets (fusion.hip)


def ready():
    h = Head()
    h.claim(0)
    for w in range(h.waves):
        h.drain_prep(w)
    h.publish_prep()
    return h


def arrived():
    h = ready()
    for p in range(h.partitions):
        h.start(p, h.epoch)
        for w in range(h.waves):
            h.drain_partial(p, w)
        h.arrive(p)
    return h


def reduced():
    h = arrived()
    h.acquire_partials()
    h.reduce(range(5))
    return h


def reject_case(name):
    h = ready()
    if name == 'exact_once_owner': h.claim(0)
    elif name == 'prep_writer': Head().drain_prep(0)
    elif name == 'all_prep_waves': Head(claimed=True, prep_drained={0}).publish_prep()
    elif name == 'epoch_acquire': h.start(0, h.epoch - 1)
    elif name == 'early_publication': Head().start(1, 1)
    elif name == 'partition_once': h.start(0, 1); h.start(0, 1)
    elif name == 'partial_writer': h.drain_partial(0, 0)
    elif name == 'arrival_once': h = arrived(); h.arrive(0)
    elif name == 'all_partial_waves': h.start(0, 1); h.drain_partial(0, 0); h.arrive(0)
    elif name == 'completion_fanin': h.acquire_partials()
    elif name == 'all_partial_acquires': arrived().reduce(range(5))
    elif name == 'partition_order': h = arrived(); h.acquire_partials(); h.reduce([4, 3, 2, 1, 0])
    elif name == 'reduce_once': reduced().reduce(range(5))
    elif name == 'all_reduce_waves': reduced().publish_output()
    elif name == 'queue_epoch_reset': reset_queues([5], [reduced()])
    elif name == 'worker_retirement': reduced().retire(4)
    elif name == 'quiescent_reset':
        h = reduced(); h.output_drained.update(range(16)); h.publish_output(); h.reset(2)
    elif name == 'idle_slot_participation': slot_participation(16, range(4))
    elif name == 'queue_exact_once': queue_liveness([['a'], ['a']], [])
    elif name == 'queue_endpoints': queue_liveness([['a']], [('z', 'a')])
    elif name == 'queue_progress':
        # A removed liveness guard is allowed to spin in the implementation;
        # the checking harness has its own short deadline and goes red.
        import subprocess, sys
        result = subprocess.run([sys.executable, '-c',
            "from etx.models.vllm_llm.bf16_300.protocol import queue_liveness; queue_liveness([['waiter', 'producer']], [('producer', 'waiter')])"],
            capture_output=True, text=True, timeout=0.5)
        assert 'ProtocolError: queue_progress' in result.stderr
        raise ProtocolError('queue_progress')
    elif name in ['snapshot_coverage', 'snapshot_identity', 'snapshot_bytes']:
        ref = check.capture('attention', 0, b'original', [b'payload'])
        got = (replace(ref, partition=1) if name == 'snapshot_identity' else replace(ref, payload=b'wrong'))
        check.compare([ref], [] if name == 'snapshot_coverage' else [got])
    elif name == 'source_anchor': source.replace_one('aa', 'a', 'b')
    elif name == 'specialisation_anchors':
        old = source.current_source
        try:
            source.current_source = lambda: old().replace('if (d.K % (ETX_WAVE_SIZE * 8) == 0 && 2 * gn <= DSL_RCH', 'if (false && 2 * gn <= DSL_RCH', 1)
            source.specialised_source()
        finally: source.current_source = old
    elif name == 'probe_selection': source.probe_source('flush', wave=16)
    elif name == 'compiler_metadata': build.resources('')
    elif name == 'dispatcher_selection': dispatcher.instrument(None, 'bad')
    elif name == 'attribution_gate':
        a, b, c = attribution_fixture(); a['last_task'] = 2; attribution.attribution(a, b, c)
    elif name == 'fusion_resource_gate': build.fusion_gate([{'numeric_gate': False}])
    elif name == 'local_device_libraries':
        import os
        before = os.environ.get('ETX_DEVLIBS')
        try:
            os.environ['ETX_DEVLIBS'] = '/nonexistent/cpu-only-libraries'
            build.prepare(compile=True)
        finally:
            if before is None: os.environ.pop('ETX_DEVLIBS', None)
            else: os.environ['ETX_DEVLIBS'] = before
    elif name == 'bf16_batch_identity':
        before = source.gemv.DSL_NB
        try:
            source.gemv.DSL_NB = 2
            build.prepare()
        finally: source.gemv.DSL_NB = before
    else: raise AssertionError(name)


GUARDS = ['exact_once_owner', 'prep_writer', 'all_prep_waves', 'epoch_acquire', 'early_publication',
          'partition_once', 'partial_writer', 'arrival_once', 'all_partial_waves', 'completion_fanin',
          'all_partial_acquires', 'partition_order', 'reduce_once', 'all_reduce_waves', 'quiescent_reset',
          'queue_epoch_reset', 'worker_retirement', 'idle_slot_participation', 'queue_exact_once', 'queue_endpoints', 'queue_progress',
          'snapshot_coverage', 'snapshot_identity', 'snapshot_bytes', 'source_anchor',
          'specialisation_anchors', 'probe_selection', 'compiler_metadata', 'bf16_batch_identity',
          'dispatcher_selection', 'attribution_gate', 'fusion_resource_gate', 'local_device_libraries']


@pytest.mark.parametrize('name', GUARDS)
def test_guard_rejection(name):
    with pytest.raises(ProtocolError, match=name):
        reject_case(name)


def test_permutations_rollover_early_owner_and_exact_reduction():
    assert dry_run()['orders'] == 120
    h = ready()
    assert not h.started  # ready is not postponed until owner attention finishes
    h.start(4, 1)
    assert h.prep_count == 1


def test_random_wave_retirement_and_cacheline_reuse():
    for seed in range(50):
        rng = random.Random(seed)
        h = ready()
        for epoch in range(1, 4):
            order = list(range(5)); rng.shuffle(order)
            for p in order:
                h.start(p, epoch)
            writes = [(p, w) for p in range(5) for w in range(16)]; rng.shuffle(writes)
            remaining = {p: 16 for p in range(5)}
            for p, w in writes:
                h.drain_partial(p, w); remaining[p] -= 1
                if not remaining[p]: h.arrive(p)
            h.acquire_partials()
            assert h.reduce(range(5)) == 12.0
            h.output_drained.update(range(16)); h.publish_output()
            for p in range(5): h.retire(p)
            h.reset(epoch + 1)
            assert not h.partials and not h.arrived
            h.claim(0)
            for w in range(16): h.drain_prep(w)
            h.publish_prep()


def test_snapshot_before_inplace_successor():
    qkv = bytearray(b'pre-rope-qkv')
    ref = check.capture('qknorm', -1, b'original-qknorm', [qkv])
    got = check.capture('qknorm', -1, b'original-qknorm', [qkv])
    qkv[:] = b'post-rope-qkv'
    check.compare([ref], [got])
    assert ref.payload == b'pre-rope-qkv'
    with pytest.raises(ProtocolError, match='snapshot_bytes'):
        check.compare([ref], [check.capture('qknorm', -1, b'original-qknorm', [qkv])])


def test_control_source_is_unchanged_and_probe_has_no_extra_fences():
    control = source.current_source()
    assert control == source.gemv.emit_tiles()
    assert 'Bf16Probe' not in control and 'BF16_PROBE' not in control
    for pair in source.PAIRS:
        probe = source.probe_source(pair)
        assert 's_memrealtime' in probe
        assert probe.count('ETX_RELEASE_DOMAIN()') == 2  # retained output publication, one per body
        assert 'buffer_inv' not in probe
        assert probe.index('probe.save(records') > probe.index('ETX_RELEASE_DOMAIN()')


# These are implementation-to-model correspondence checks, not hardware proof.
# Mutation removes the ACTUAL device token, while the model rejection tests above
# exercise its semantic reason separately.
DEVICE_TOKENS = [
    'domain >= ETX_NUM_DOMAINS', 'q->count != 5', 'ticket >= q->count', 'a->partition != ticket',
    '__hip_atomic_fetch_add(&q->next', 'queues + domain',
    'a->epoch != s->epoch', 'a->partitions != 5', 'a->partition >= 5',
    'a->prep.coord[0] != (int)s->head', 'a->attention.coord[0] != (int)(5 * s->head + a->partition)',
    'a->reduce.coord[0] != (int)s->head', 'ETX_DOMAIN_ID() != s->xcd',
    'if (a->partition == 0)', '__hip_atomic_exchange(&s->claimed',
    'ETX_RELEASE_DOMAIN();  // EACH producing wave',
    '__hip_atomic_store(&s->ready, (unsigned long long)a->epoch + 1ull, __ATOMIC_RELEASE',
    '__hip_atomic_load(&s->ready, __ATOMIC_ACQUIRE', '++n >= a->spin_limit',
    'ETX_ACQUIRE_DOMAIN();  // every consumer wave',
    '__hip_atomic_fetch_or(&s->mask', '__hip_atomic_fetch_add(&s->arrived, 1u, __ATOMIC_ACQ_REL',
    'ETX_ACQUIRE_DOMAIN();\n    { const etx_ctx c = a->reduce;',
    '__hip_atomic_store(&s->output, (unsigned long long)a->epoch + 1ull, __ATOMIC_RELEASE',
    'bf16_snapshot(snapshots[3 * (5 * s->head + a->partition)]);',
    'bf16_snapshot(snapshots[3 * (5 * s->head + a->partition) + 1]);',
    'bf16_snapshot(snapshots[3 * (5 * s->head + a->partition) + 2]);',
]


@pytest.mark.parametrize('token', DEVICE_TOKENS)
def test_device_contract_source(token):
    s = (HERE / 'fusion.hip').read_text()
    assert s.count(token) == 1
    assert s.index('__hip_atomic_store(&s->ready') < s.index('etx_tile_attn_mfma16_g8(&c)')
    assert s.index('etx_tile_attn_mfma16_g8(&c)') < s.index('etx_tile_attn_reduce(&c)')


def test_fixed_shape_keeps_down_original_and_up_math_order():
    old, new = source.current_source(), source.specialised_source()
    a = old.index('// ---- dsl_moe_down:'); b = old.index('// ---- dsl_gemv_fold:', a)
    assert old[a:b] == new[new.index('// ---- dsl_moe_down:'):new.index('// ---- dsl_gemv_fold:')]
    assert 'd.K == 2048 && gn == 32 && d.rpt == 32 && gnw == 16 && blockDim.x == 1024' in new
    assert 'static_assert(EPL == 8 && NB == 1' in new
    assert '__syncthreads(); // all gate/up results visible before SiLU' in new
    assert 'z += w2 == wv ? (0.f + v) : 0.f' in new
    assert 'g[q < gn ? q : d.rpt + q - gn] = dsl_bfr(v[0])' in new
    assert 'dsl_silu(g[r - r0]) * g[d.rpt + r - r0]' in new
    # Model the EXACT flat segment ownership, not a guess that all up/down rows
    # fit. A single-owner proof must fail for a different number of rows.
    def owners(nrows, k):
        spr = k // 512; per = (nrows * spr + 15) // 16
        return [{sg // per for sg in range(r * spr, (r + 1) * spr)} for r in range(nrows)]
    assert all(len(o) == 1 for o in owners(64, 2048))
    assert any(len(o) > 1 for o in owners(60, 2048))
    for wave in range(16):
        for value in [np.float32(-0.), np.float32(0.), np.float32(1e-30), np.float32(-1e30), np.float32('inf')]:
            partial = np.float32(np.float32(0.) + value)
            baseline = np.float32(0.)
            for w in range(16): baseline = np.float32(baseline + (partial if w == wave else np.float32(0.)))
            direct = np.float32(0.)
            for w in range(16): direct = np.float32(direct + (np.float32(np.float32(0.) + value) if w == wave else np.float32(0.)))
            assert struct.pack('f', direct) == struct.pack('f', baseline)


def attribution_fixture():
    import copy
    a = {'begin': 100, 'end': 110, 'task': 1, 'last_task': 1, 'wave': 0, 'last_wave': 0,
         'body_us': 20.0, 'tokens_matched': 32, 'all_imports_byte_equal': True,
         'session': 'S', 'compiler': 'C', 'math_identity': 'M',
         'resources': {'vgpr': 64, 'agpr': 0, 'vgpr_spills': 0, 'sgpr_spills': 0, 'scratch_bytes_lane': 0,
                       'lds_bytes': 21256, 'static_blocks_per_cu': 2}}
    return a, copy.deepcopy(a), copy.deepcopy(a)


@pytest.mark.parametrize('bad', ['wave', 'perturbation', 'resources', 'sgpr', 'imports', 'tokens', 'session', 'compiler', 'math', 'empty'])
def test_attribution_rejects_unqualified_evidence(bad):
    a, b, c = attribution_fixture()
    if bad == 'wave': a['last_wave'] = 1
    elif bad == 'perturbation': a['body_us'] += 0.501
    elif bad == 'resources': c['resources']['vgpr'] += 4
    elif bad == 'sgpr': c['resources']['sgpr_spills'] += 1
    elif bad == 'imports': a['all_imports_byte_equal'] = False
    elif bad == 'tokens': a['tokens_matched'] = 31
    elif bad == 'session': b['session'] = 'different'
    elif bad == 'compiler': b['compiler'] = 'different'
    elif bad == 'math': b['math_identity'] = 'different'
    elif bad == 'empty': c['begin'] = 0
    with pytest.raises(ProtocolError, match='attribution_gate'): attribution.attribution(a, b, c)


def test_attribution_reports_issue_only_not_a_booked_gain():
    a, b, c = attribution_fixture()
    a['body_us'] += 0.5
    result = attribution.attribution(a, b, c)
    assert result['issue_interval_us'] == 0.1 and result['empty_pair_us'] == 0.1
    assert result['exclusive_removable_us_layer'] is None


def test_sparse_dispatcher_slices(tmp_path):
    tile = tmp_path / 'tile.hip'; tile.write_text(source.current_source())
    plan = dispatcher.graph(tile)
    from etx.codegen import emit_kernel
    original = emit_kernel(plan)
    assert f'#include "{tile}"' in original
    for pair in dispatcher.PAIRS:
        instrumented = dispatcher.instrument(plan, pair)
        assert instrumented.count('ETX_RELEASE_DOMAIN()') == original.count('ETX_RELEASE_DOMAIN()')
        assert instrumented.count('__syncthreads()') == original.count('__syncthreads()')
        assert 'etx_call_body(sym, &ctx)' in instrumented
        assert 'bf_begin' in instrumented and 'bf_end' in instrumented
    assert 'bf_begin' not in original


FIXED_TOKENS = [
    'd.K == 2048 && gn == 32 && d.rpt == 32 && gnw == 16 && blockDim.x == 1024',
    'static_assert(EPL == 8 && NB == 1',
    '__syncthreads(); // all gate/up results visible before SiLU',
    'z += w2 == wv ? (0.f + v) : 0.f',
]
PROBE_TOKENS = [
    "mark('body_begin'),",
    'blk = etx_uniform(blk);',
    'const int nblk = *(const int DSL_ARG_AS*)(uintptr_t)(a + DSL_GRID_OFF + 12);',
    'selected && !started', 'selected && started && !ended',
    'selected && threadIdx.x % 64 == 0',
    'probe.selected = blk == {task} && etx_wave() == {wave}',
]


def test_probe_bounded_state_contract():
    s = Path(source.__file__).read_text()
    for token in PROBE_TOKENS: assert s.count(token) == 1
    assert 'unsigned long long begin = 0, end = 0;' in source.HELPERS
    assert '"memory"' in source.HELPERS and 's_waitcnt' not in source.HELPERS


def test_full_body_probe_includes_argument_reads():
    s = source.probe_source('body')
    s = s[s.index('void bf16_300_probe_dsl_moe_up'):]
    assert s.index('probe.first();') < s.index('const int nblk') < s.index('const DslArgs d')
    assert s.index('ETX_RELEASE_DOMAIN()') < s.index('probe.last();') < s.index('probe.save(records')


def test_probe_keeps_original_scalar_argument_path():
    s = source.probe_source('arguments')
    s = s[s.index('void bf16_300_probe_dsl_moe_up'):]
    assert 'blk = etx_uniform(blk);' in s
    assert 'const int nblk = *(const int DSL_ARG_AS*)(uintptr_t)(a + DSL_GRID_OFF + 12);' in s
    assert s.index('blk = etx_uniform(blk);') < s.index('const DslArgs d = dsl_args(a);')
