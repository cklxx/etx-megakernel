"""Synthetic gfx942 snippets for etx/tools/isa_hops.py."""
from etx.tools import isa_hops as ih


def _an(body, **kw):
    kw.setdefault('end', 'regs')
    text = 'k:\n' + body.strip('\n') + '\n.Lfunc_end0:\n'
    return ih.analyse(ih.region(text, func='k'), **kw)


def test_dependent_loads_are_hops():
    r = _an('''
\ts_load_dwordx2 s[0:1], s[4:5], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_mov_b64_e32 v[2:3], s[0:1]
\tglobal_load_dword v1, v[2:3], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tv_lshlrev_b32_e32 v1, 2, v1
\tglobal_load_dword v4, v1, s[0:1]
''')
    assert r['hops']['scalar'] == 1 and r['hops']['coherent'] == 1 and r['hops']['plain'] == 1
    assert [h['line'] for h in r['chain']] == [2, 5, 8]


def test_loads_in_flight_together_are_one_hop():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off
\tglobal_load_dword v4, v[2:3], off offset:4
\ts_waitcnt vmcnt(0)
\tv_add_u32_e32 v5, v1, v4
\tglobal_load_dword v6, v5, s[8:9]
''')
    assert r['hops']['plain'] == 2


def test_wait_without_data_dependency_is_not_a_hop():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v4, v[6:7], off
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v8, v[10:11], off
''')
    assert r['hops']['plain'] == 1


def test_branch_on_loaded_value_is_control():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tv_cmp_lt_i32_e32 vcc, 0, v1
\ts_and_saveexec_b64 s[4:5], vcc
\ts_cbranch_execz .LBB0_2
; %bb.1:
\tglobal_load_dword v4, v[6:7], off
.LBB0_2:
\ts_or_b64 exec, exec, s[4:5]
\tglobal_load_dword v8, v[10:11], off
''')
    assert r['hops']['coherent'] == 1 and r['hops']['plain'] == 1


def test_scratch_reload_continues_the_chain():
    r = _an('''
\tglobal_load_dwordx2 v[2:3], v[0:1], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tscratch_store_dwordx2 off, v[2:3], off offset:8 ; 8-byte Folded Spill
\tv_mov_b32_e32 v2, 0
\tv_mov_b32_e32 v3, 0
\tscratch_load_dwordx2 v[8:9], off, off offset:8 ; 8-byte Folded Reload
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v4, v[8:9], off
''')
    assert r['hops'] == {'coherent': 1, 'plain': 1, 'scalar': 0, 'scratch': 1, 'store': 0, 'lds': 0}
    assert r['scratch_reloads_on_chain'] == 1
    assert [h['class'] for h in r['chain']] == ['coherent', 'scratch', 'plain']


def test_scratch_slots_pair_by_offset():
    r = _an('''
\tglobal_load_dwordx2 v[2:3], v[0:1], off
\ts_waitcnt vmcnt(0)
\tscratch_store_dwordx2 off, v[2:3], off offset:8
\tscratch_store_dwordx2 off, v[20:21], off offset:16
\tscratch_load_dwordx2 v[8:9], off, off offset:16
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v4, v[8:9], off
''')
    assert r['hops']['plain'] == 1 and r['hops']['scratch'] == 1


def test_sgpr_spill_lanes_keep_the_value():
    r = _an('''
\ts_load_dword s4, s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_writelane_b32 v127, s4, 7
\ts_mov_b32 s4, 0
\tv_readlane_b32 s6, v127, 7
\tglobal_load_dword v1, v0, s[6:7]
''')
    assert r['hops']['scalar'] == 1 and r['hops']['plain'] == 1


def test_lds_is_not_a_global_hop():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off
\ts_waitcnt vmcnt(0)
\tds_write_b32 v0, v1
\ts_waitcnt lgkmcnt(0)
\ts_barrier
\tds_read_b32 v5, v0
\ts_waitcnt lgkmcnt(0)
\tglobal_load_dword v6, v5, s[8:9]
''')
    assert r['hops']['plain'] == 2 and r['hops']['lds'] == 1


def test_store_drain_counts_once_at_the_release():
    body = '''
\tglobal_load_dword v1, v[2:3], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[4:5], v1, off sc0 sc1
\ts_waitcnt vmcnt(0)
'''
    assert _an(body)['hops'] == {'coherent': 1, 'plain': 0, 'scalar': 0, 'scratch': 0, 'store': 1, 'lds': 0}
    assert _an(body, with_drain=False)['hops']['store'] == 0


def test_store_then_barrier_then_load_is_two_hops_only_with_barrier_drain():
    body = ('''
\tglobal_load_dword v1, v[2:3], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[4:5], v1, off sc0 sc1
\ts_waitcnt vmcnt(0)
\ts_barrier
\tglobal_load_dword v6, v[8:9], off sc0 sc1
''')
    r = _an(body, with_drain=False, barrier_drain=True)
    assert [h['class'] for h in r['chain']] == ['coherent', 'store', 'coherent']
    assert _an(body, with_drain=False)['hops']['coherent'] == 1     # a vmcnt(0) drain alone is not a hop


def test_loop_with_carried_chain_reports_hops_per_trip():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off
\ts_mov_b32 s2, 8
.LBB0_1:
\ts_waitcnt vmcnt(0)
\tv_lshlrev_b32_e32 v1, 2, v1
\tglobal_load_dword v1, v1, s[8:9]
\ts_add_i32 s2, s2, -1
\ts_cmp_lg_u32 s2, 0
\ts_cbranch_scc1 .LBB0_1
; %bb.2:
\ts_waitcnt vmcnt(0)
''')
    assert r['hops']['plain'] == 2
    (lp,) = r['loops']
    assert lp['hops_per_extra_trip']['plain'] == 1 and lp['linear']
    assert lp['trip_count'].startswith('runtime') or lp['trip_count'].startswith('constant 0')


def test_loop_without_carried_chain_adds_nothing():
    r = _an('''
\ts_mov_b32 s2, 8
.LBB0_1:
\tglobal_load_dwordx4 v[4:7], v[0:1], off
\tv_add_co_u32_e32 v0, vcc, 16, v0
\tv_addc_co_u32_e32 v1, vcc, 0, v1, vcc
\ts_waitcnt vmcnt(0)
\tv_add_f32_e32 v10, v10, v4
\ts_add_i32 s2, s2, -1
\ts_cmp_lg_u32 s2, 0
\ts_cbranch_scc1 .LBB0_1
''')
    assert r['hops']['plain'] == 1 and r['loops'] == [] and r['loops_without_carried_hops'] == 1


def test_publish_end_ignores_dangling_loads():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[4:5], v1, off sc0 sc1
\tglobal_load_dword v6, v1, s[8:9]
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v7, v6, s[8:9]
''', end='publish')
    assert r['hops'] == {'coherent': 1, 'plain': 0, 'scalar': 0, 'scratch': 0, 'store': 1, 'lds': 0}


def test_loop_entered_from_a_later_block():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off
\ts_branch .LBB0_2
.LBB0_1:
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v1, v1, s[8:9]
.LBB0_2:
\ts_add_i32 s2, s2, -1
\ts_cmp_lg_u32 s2, 0
\ts_cbranch_scc1 .LBB0_1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[4:5], v1, off sc0 sc1
''', end='publish')
    assert r['hops']['plain'] == 1 and r['hops']['store'] == 1
    (lp,) = r['loops']
    assert lp['hops_per_extra_trip']['plain'] == 1


def test_control_ends_at_reconvergence():
    r = _an('''
\tglobal_load_dword v1, v[2:3], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tv_cmp_lt_i32_e32 vcc, 0, v1
\ts_and_saveexec_b64 s[4:5], vcc
\ts_cbranch_execz .LBB0_2
; %bb.1:
\tglobal_load_dword v4, v[6:7], off
.LBB0_2:
\ts_or_b64 exec, exec, s[4:5]
\tglobal_load_dword v8, v[10:11], off
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v8, off sc0 sc1
''', end='publish')
    assert r['hops'] == {'coherent': 0, 'plain': 1, 'scalar': 0, 'scratch': 0, 'store': 1, 'lds': 0}


def test_scalar_branch_scope():
    body = '''
\ts_load_dword s4, s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\ts_cmp_eq_u32 s4, 0
\ts_cbranch_scc1 .LBB0_2
; %bb.1:
\tglobal_load_dword v4, v[6:7], off
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v4, off sc0 sc1
.LBB0_2:
\tglobal_load_dword v8, v[10:11], off
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[14:15], v8, off sc0 sc1
'''
    r = _an(body, end='publish')
    assert [h['class'] for h in r['chain']] == ['scalar', 'plain', 'store']


def test_exclusive_copies_do_not_chain():
    copy = '''
\tv_cmp_eq_u32_e32 vcc, {k}, v0
\ts_and_saveexec_b64 s[{a}:{b}], vcc
\ts_cbranch_execz .LBB0_{k}
; %bb.c{k}:
\tglobal_load_dword v1, v[2:3], off
\ts_waitcnt vmcnt(0)
\tv_cmp_lt_i32_e32 vcc, 0, v1
\ts_and_saveexec_b64 s[{c}:{d}], vcc
\ts_cbranch_execz .LBB0_i{k}
; %bb.d{k}:
\tglobal_load_dword v4, v[6:7], off
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[8:9], v4, off
.LBB0_i{k}:
\ts_or_b64 exec, exec, s[{c}:{d}]
.LBB0_{k}:
\ts_or_b64 exec, exec, s[{a}:{b}]
'''
    body = ''.join(copy.format(k=k, a=10 + 4 * k, b=11 + 4 * k, c=12 + 4 * k, d=13 + 4 * k) for k in range(4))
    r = _an(body, end='publish')
    assert r['hops']['plain'] == 2 and r['hops']['store'] == 1


def test_watch_gives_the_chain_before_a_stream():
    r = _an('''
\ts_load_dwordx2 s[0:1], s[4:5], 0x0
\ts_waitcnt lgkmcnt(0)
\tglobal_load_dword v1, v0, s[0:1] sc0 sc1
\ts_waitcnt vmcnt(0)
\tv_lshlrev_b32_e32 v2, 4, v1
\tglobal_load_dwordx4 v[4:7], v2, s[8:9]
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[8:9], v4, off sc0 sc1
''', end='publish', watch={7})
    assert r['before_watch']['hops'] == {'coherent': 1, 'plain': 0, 'scalar': 1, 'scratch': 0, 'store': 0, 'lds': 0}


def test_else_lanes_keep_the_old_value():
    # structured if/else: the then lanes overwrite v4 with a load; the else lanes use their old v4 (a pointer)
    r = _an('''
\tv_cmp_gt_i32_e32 vcc, 4, v0
\ts_and_saveexec_b64 s[2:3], vcc
\ts_xor_b64 s[2:3], exec, s[2:3]
\ts_cbranch_execz .LBB0_2
\tglobal_load_dwordx2 v[4:5], v[8:9], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[10:11], v4, off sc0 sc1
.LBB0_2:
\ts_andn2_saveexec_b64 s[2:3], s[2:3]
\ts_cbranch_execz .LBB0_4
\tglobal_load_dword v6, v[4:5], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v6, off sc0 sc1
.LBB0_4:
\ts_or_b64 exec, exec, s[2:3]
''', end='publish')
    assert r['hops']['coherent'] == 1


def test_value_after_reconvergence_sees_both_lane_sets():
    # after the branch joins, v4 holds the load in the then lanes and the old value in the others
    r = _an('''
\tv_cmp_gt_i32_e32 vcc, 4, v0
\ts_and_saveexec_b64 s[2:3], vcc
\ts_cbranch_execz .LBB0_2
\tglobal_load_dwordx2 v[4:5], v[8:9], off sc0 sc1
.LBB0_2:
\ts_or_b64 exec, exec, s[2:3]
\ts_waitcnt vmcnt(0)
\tglobal_load_dword v6, v[4:5], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v6, off sc0 sc1
''', end='publish')
    assert r['hops']['coherent'] == 2


def test_nested_write_keeps_the_narrow_lanes_at_the_join():
    # v4 is written at the outer level under one branch, then again inside the then part of an inner branch
    r = _an('''
\tv_cmp_gt_i32_e32 vcc, 8, v0
\ts_and_saveexec_b64 s[6:7], vcc
\tv_mov_b32_e32 v4, v20
\tv_mov_b32_e32 v5, v21
\tv_cmp_gt_i32_e32 vcc, 4, v0
\ts_and_saveexec_b64 s[2:3], vcc
\ts_xor_b64 s[2:3], exec, s[2:3]
\ts_cbranch_execz .LBB0_2
\tglobal_load_dwordx2 v[4:5], v[8:9], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[10:11], v4, off sc0 sc1
.LBB0_2:
\ts_andn2_saveexec_b64 s[2:3], s[2:3]
\ts_cbranch_execz .LBB0_4
\tglobal_load_dword v6, v[4:5], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v6, off sc0 sc1
.LBB0_4:
\ts_or_b64 exec, exec, s[2:3]
\ts_or_b64 exec, exec, s[6:7]
''', end='publish')
    assert r['hops']['coherent'] == 1


def test_join_order_does_not_matter_for_else_lanes():
    # the skip edge reaches the join first (laid out above the then part)
    r = _an('''
\tv_cmp_gt_i32_e32 vcc, 8, v0
\ts_and_saveexec_b64 s[6:7], vcc
\tv_mov_b32_e32 v4, v20
\tv_mov_b32_e32 v5, v21
\tv_cmp_gt_i32_e32 vcc, 4, v0
\ts_and_saveexec_b64 s[2:3], vcc
\ts_xor_b64 s[2:3], exec, s[2:3]
\ts_cbranch_execnz .LBB0_5
.LBB0_2:
\ts_andn2_saveexec_b64 s[2:3], s[2:3]
\ts_cbranch_execz .LBB0_4
\tglobal_load_dword v6, v[4:5], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v6, off sc0 sc1
.LBB0_4:
\ts_or_b64 exec, exec, s[2:3]
\ts_or_b64 exec, exec, s[6:7]
\ts_endpgm
.LBB0_5:
\tglobal_load_dwordx2 v[4:5], v[8:9], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[10:11], v4, off sc0 sc1
\ts_branch .LBB0_2
''', end='publish')
    assert r['hops']['coherent'] == 1


def test_implicit_def_ends_a_loop_carried_lane_value():
    # in a loop, v4 is set only in a branch; the compiler marks the other lanes undefined
    r = _an('''
\ts_mov_b64 s[4:5], 0
.LBB0_1:
\t; implicit-def: $vgpr4_vgpr5
\tv_cmp_gt_i32_e32 vcc, 4, v0
\ts_and_saveexec_b64 s[2:3], vcc
\ts_cbranch_execz .LBB0_2
\tv_mov_b32_e32 v4, v20
\tv_mov_b32_e32 v5, v21
.LBB0_2:
\ts_or_b64 exec, exec, s[2:3]
\tglobal_load_dword v20, v[4:5], off sc0 sc1
\ts_waitcnt vmcnt(0)
\ts_add_u32 s4, s4, 1
\ts_cmp_lt_u32 s4, s6
\ts_cbranch_scc1 .LBB0_1
\tglobal_store_dword v[12:13], v20, off sc0 sc1
''', end='publish')
    assert r['loops'][0]['hops_per_extra_trip']['coherent'] == 1     # the load feeds the next v4 in the branch lanes


def test_implicit_def_comment_parses():
    from etx.tools.isa_hops import parse
    ins = parse('\t; implicit-def: $vgpr2_vgpr3_vgpr4_vgpr5, $sgpr0\n')
    assert ins[0].mnem == 'implicit_def' and ins[0].ops == ['v2', 'v3', 'v4', 'v5', 's0']


def test_mad_u64_carry_out_is_not_a_source():
    r = _an('''
\tglobal_load_dword v1, v[8:9], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tv_cmp_gt_i32_e64 s[4:5], v1, 0
\tv_mad_u64_u32 v[6:7], s[4:5], v15, v20, v[6:7]
\tglobal_load_dword v2, v[6:7], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v2, off sc0 sc1
''', end='publish')
    assert r['hops']['coherent'] == 1


def test_if_else_restore_gives_back_the_outer_exec_value():
    # an if/else on a loaded value; a load after the join does not depend on that value
    r = _an('''
\tglobal_load_dword v1, v[8:9], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tv_cmp_gt_i32_e32 vcc, 4, v1
\ts_and_saveexec_b64 s[0:1], vcc
\ts_xor_b64 s[14:15], exec, s[0:1]
\tv_mov_b32_e32 v2, 1
.LBB0_1:
\ts_andn2_saveexec_b64 s[0:1], s[14:15]
\tv_mov_b32_e32 v2, 2
\ts_or_b64 exec, exec, s[0:1]
\tglobal_load_dword v3, v[10:11], off sc0 sc1
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v[12:13], v3, off sc0 sc1
''', end='publish')
    assert r['hops']['coherent'] == 1      # v3 only: the join restores the exec of before the branch on v1
