"""Metadata-only EP/TP contracts; no torch, compiler, GPU or stage files needed."""
from copy import deepcopy
import hashlib
import json
import struct

import pytest

from examples.glm53 import parallel_layout, stage


ABI = struct.Struct('<QQiiii')


def make_spec(inter=256, dtype='torch.float8_e4m3fn', payload=False, attention=False,
              attention_dims=(128, 128), attention_fp8_slots=None):
    """A dense and sparse layer; dimensions cross real FP8 block boundaries."""
    config = dict(num_hidden_layers=2, n_routed_experts=3, hidden_size=256,
                  intermediate_size=inter, moe_intermediate_size=inter,
                  mlp_layer_types=['dense', 'sparse'])
    if attention:
        kd, ad = attention_dims
        config.update(hc_mult=2, layer_types=['linear_attention', 'deepseek_sparse_attention'],
                      linear_attn_config=dict(num_heads=8, head_dim=kd, short_conv_kernel_size=3),
                      num_attention_heads=8, qk_nope_head_dim=ad, q_lora_rank=128,
                      kv_lora_rank=128, index_n_heads=2, index_head_dim=8, index_kpool=4)
    slots = stage.slot_names(config); blobs = {}
    spec = dict(schema=1, model='synthetic', revision='pinned', fixture=True,
                config=config, weights={})

    def blob(name, shapes, layer=True):
        cursor = 0; entries = {}; data = bytearray()
        for number, key in enumerate(slots if layer else shapes):
            if key not in shapes:
                continue
            shape, dt = shapes[key]
            rows = shape[0]; cols = 1
            for value in shape[1:]:
                cols *= value
            itemsize = {'torch.float8_e4m3fn': 1, 'torch.float32': 4, 'torch.bfloat16': 2}[dt]
            cursor = (cursor+127)//128*128; offset = cursor
            size = rows*cols*itemsize; cursor += size; scale_offset = 0
            if dt == 'torch.float8_e4m3fn':
                cursor = (cursor+3)//4*4; scale_offset = cursor
                cursor += 4*((rows+127)//128)*((cols+127)//128)
            entries[key] = dict(shape=list(shape), dtype=dt, offset=offset, scale_offset=scale_offset)
            if payload:
                data.extend(b'\0'*(offset-len(data)))
                pattern = bytes((x+number) % 256 for x in range(251))
                length = cursor-offset
                data.extend((pattern*((length+250)//251))[:length])
        blobs[name+'.bin'] = bytes(data)
        return dict(blob=name+'.bin', meta=name+'.meta', blob_bytes=cursor,
                    meta_bytes=32*(len(slots) if layer else len(shapes)), tensors=entries)

    for layer in range(2):
        shapes = {'input_layernorm.weight': ((256,), 'torch.bfloat16'),
                  'self_attn.conv1d.weight': ((6, 1, 3), 'torch.float32')}
        if attention:
            # Deliberately independent of stage.shapes / derive's contract.
            common = [(8,512), (8,), (3,), (8,512), (8,), (3,), (256,), (256,)]
            shapes = {key: (shape, 'torch.float32') for key, shape in zip(stage.SLOTS[:8], common)}
            if layer == 0:
                qd = 8*kd
                values = [(qd,256), (qd,256), (qd,256), (3*qd,1,3),
                          (kd,256), (qd,kd), (qd,), (8,), (8,256),
                          (kd,256), (qd,kd), (kd,)]
                shapes.update({key: (shape, 'torch.float32') for key, shape in zip(stage.SLOTS[8:20], values)})
                fp8_slots = {8,9,10,13,18,20} if attention_fp8_slots is None else set(attention_fp8_slots)
            else:
                values = [(128,256), (128,), (8*ad,128), (128,256), (128,),
                          (16*ad,128), (16,128), (8,256), (8,), (8,),
                          (2,256), (8,256), (4,8)]
                shapes.update({key: (shape, 'torch.float32') for key, shape in zip(stage.SLOTS[21:34], values)})
                fp8_slots = {23,26,20} if attention_fp8_slots is None else set(attention_fp8_slots)
            shapes[stage.SLOTS[20]] = ((256,8*(kd if layer == 0 else ad)), 'torch.float32')
            for slot in fp8_slots:
                key = stage.SLOTS[slot]
                if key in shapes:
                    shapes[key] = (shapes[key][0], dtype)
        names = stage.SLOTS[34:37] if layer == 0 else stage.SLOTS[39:42]
        for key, shape in zip(names, ((inter, 256), (inter, 256), (256, inter))):
            shapes[key] = (shape, dtype)
        if layer:
            shapes['mlp.gate.weight'] = ((3, 256), 'torch.float32')
            shapes['mlp.gate.e_score_correction_bias'] = ((3,), 'torch.float32')
            for expert in range(3):
                for projection, shape in zip(('gate', 'up', 'down'), ((inter, 256), (inter, 256), (256, inter))):
                    shapes[f'mlp.experts.{expert}.{projection}_proj.weight'] = (shape, dtype)
        spec['weights'][f'l{layer}'] = blob(f'l{layer}', shapes)
    for name in ('embedding', 'norm', 'head'):
        spec['weights'][name] = blob(name, {name: ((256,), 'torch.bfloat16')}, False)
    return spec, blobs


def replay(layout, sources):
    dest = {name+'_weights': bytearray(w['blob_bytes']) for name, w in layout['weights'].items()}
    for r in layout['ranges']:
        src = sources[r['srcfile']]; dst = dest[r['tensor']]
        assert r['rows'] > 0 and 0 < r['width'] <= r['srcstride']
        assert r['width'] <= r['dststride']
        for row in range(r['rows']):
            a = r['srcoff']+row*r['srcstride']; b = r['destoff']+row*r['dststride']
            assert a+r['width'] <= len(src) and b+r['width'] <= len(dst)
            # Emulate bounded host chunks, including widths larger than the bound.
            for offset in range(0, r['width'], 73):
                size = min(73, r['width']-offset)
                dst[b+offset:b+offset+size] = src[a+offset:a+offset+size]
    return dest


def descriptor(layout, name, key):
    return layout['weights'][name]['descriptors'][layout['slot_names'].index(key)]


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('devices', [2, 4, 8])
def test_metadata_only_contract_worlds_and_source_identity(mode, devices):
    spec, _ = make_spec(inter=1024); before = deepcopy(spec)
    result = parallel_layout.derive(spec, mode, devices)
    assert spec == before
    assert json.loads(json.dumps(result)) == result
    expected = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert result['source_spec_sha256'] == expected
    assert result['devices'] == devices and result['mode'] == mode
    assert len(result['weights']) == 2*(devices+1)
    assert result['slot_names'] == stage.slot_names(spec['config'])
    for name, w in result['weights'].items():
        assert len(w['descriptors'])*32 == w['meta_bytes']
        assert result['graph_tensors'][name+'_weights']['shape'] == [w['blob_bytes']]
        assert result['graph_tensors'][name+'_meta']['rank'] == w['rank']
    outer = sum(spec['weights'][n]['blob_bytes']+spec['weights'][n]['meta_bytes'] for n in ('embedding', 'norm', 'head'))
    for m in result['rank_memory']:
        local = [w for w in result['weights'].values() if w['rank'] == m['rank']]
        assert m['blob_bytes'] == sum(w['blob_bytes'] for w in local)
        assert m['meta_bytes'] == sum(w['meta_bytes'] for w in local)
        assert m['outer_bytes'] == (outer if m['rank'] == 0 else 0)
        assert m['total_bytes'] == m['blob_bytes']+m['meta_bytes']+m['outer_bytes']
    assert not any(r['srcfile'].startswith(('embedding', 'norm', 'head')) for r in result['ranges'])
    assert len(result['ranges']) <= 2*sum(len(w['tensors'])*devices for name, w in spec['weights'].items() if name.startswith('l'))
    # Expert 2 wraps back to root in EP2, but occupies a separate rank in EP4/8.
    if mode == 'ep':
        key = 'mlp.experts.2.gate_proj.weight'
        for rank in range(devices):
            assert bool(descriptor(result, f'l1_r{rank}', key)[2]) == (rank == 2 % devices)


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('dtype', ['torch.float8_e4m3fn', 'torch.bfloat16', 'torch.float32'])
def test_range_replay_matches_independent_payload_and_scale_slices(mode, dtype):
    spec, sources = make_spec(dtype=dtype, payload=True)
    layout = parallel_layout.derive(spec, mode, 2); blobs = replay(layout, sources)
    copies = {}
    for name, w in layout['weights'].items():
        layer = name.split('_')[0]; source = spec['weights'][layer]
        data = sources[source['blob']]; dest = blobs[name+'_weights']
        for key, local in w['tensors'].items():
            original = source['tensors'][key]
            shape = original['shape']; rows = shape[0]; cols = 1
            for dimension in shape[1:]:
                cols *= dimension
            off, scale, nr, nc, kind, block = descriptor(layout, name, key)
            itemsize = {1: 2, 2: 4, 3: 1}[kind]
            axis = local['shard_axis']; start = local['source_start']
            expected = b''.join(data[original['offset']+((r+(start if axis == 0 else 0))*cols+(start if axis == 1 else 0))*itemsize:
                                       original['offset']+((r+(start if axis == 0 else 0))*cols+(start if axis == 1 else 0)+nc)*itemsize]
                                for r in range(nr))
            assert dest[off:off+nr*nc*itemsize] == expected
            assert block == (128 if kind == 3 else 0)
            if kind == 3:
                stride = (cols+127)//128; dr, dc = (nr+127)//128, (nc+127)//128
                origin = original['scale_offset']+((start//128)*stride if axis == 0 else start//128 if axis == 1 else 0)*4
                expected_scale = b''.join(data[origin+r*stride*4:origin+(r*stride+dc)*4] for r in range(dr))
                assert dest[scale:scale+dr*dc*4] == expected_scale
            else:
                assert scale == 0
            copies.setdefault((layer, key), []).append((name, local))
    for (layer, key), owners in copies.items():
        ffn = key in stage.SLOTS[34:37] or key in stage.SLOTS[39:42] or key.startswith('mlp.experts.')
        if mode == 'tp' and ffn:
            assert [v['source_start'] for _, v in owners] == [0, 128]
            assert {n for n, _ in owners} == {layer+'_r0', layer+'_r1'}
        else:
            assert len(owners) == 1


@pytest.mark.parametrize('mode', ['tp', 'ep'])
def test_every_source_payload_and_scale_byte_copied_once(mode):
    spec, _ = make_spec()
    layout = parallel_layout.derive(spec, mode, 2)
    # Compare interval unions with no overlap; padding must never be copied.
    for name in ('l0', 'l1'):
        w = spec['weights'][name]; actual = []
        for r in layout['ranges']:
            if r['srcfile'] == w['blob']:
                actual.extend((r['srcoff']+i*r['srcstride'], r['srcoff']+i*r['srcstride']+r['width']) for i in range(r['rows']))
        expected = []
        for tensor in w['tensors'].values():
            rows = tensor['shape'][0]; cols = 1
            for d in tensor['shape'][1:]:
                cols *= d
            size = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float8_e4m3fn': 1}[tensor['dtype']]
            expected.append((tensor['offset'], tensor['offset']+rows*cols*size))
            if tensor['dtype'] == 'torch.float8_e4m3fn':
                expected.append((tensor['scale_offset'], tensor['scale_offset']+4*((rows+127)//128)*((cols+127)//128)))

        def merge(intervals):
            merged = []
            for a, b in sorted(intervals):
                if merged and a == merged[-1][1]:
                    merged[-1] = (merged[-1][0], b)
                else:
                    assert not merged or a >= merged[-1][1]
                    merged.append((a, b))
            return merged
        assert merge(actual) == merge(expected)


def test_ep_dense_common_shared_local_and_expert_modulo_global_slots():
    spec, _ = make_spec(); layout = parallel_layout.derive(spec, 'ep', 4)
    for key in stage.SLOTS[34:37]:
        assert descriptor(layout, 'l0_common', key)[2] > 0
        assert all(descriptor(layout, f'l0_r{r}', key) == [0]*6 for r in range(4))
    for key in stage.SLOTS[39:42]:
        assert descriptor(layout, 'l1_r0', key)[2] > 0
        assert descriptor(layout, 'l1_common', key) == [0]*6
    for expert in range(3):
        key = f'mlp.experts.{expert}.gate_proj.weight'
        for rank in range(4):
            assert bool(descriptor(layout, f'l1_r{rank}', key)[2]) == (rank == expert % 4)
    for layer in range(2):
        for rank in range(4):
            assert descriptor(layout, f'l{layer}_r{rank}', 'input_layernorm.weight') == [0]*6
    assert layout['weights']['l1_r3']['blob_bytes'] == 1
    assert all(r['tensor'] != 'l1_r3_weights' for r in layout['ranges'])
    # Flattened conv ABI, without changing its source shape metadata.
    assert descriptor(layout, 'l0_common', 'self_attn.conv1d.weight')[2:4] == [6, 3]


def test_row_shards_collapse_but_down_columns_remain_strided():
    spec, _ = make_spec(); layout = parallel_layout.derive(spec, 'tp', 2)
    for rank in range(2):
        w = layout['weights'][f'l0_r{rank}']; ranges = [r for r in layout['ranges'] if r['tensor'] == f'l0_r{rank}_weights']
        for key in stage.SLOTS[34:36]:
            offset = w['tensors'][key]['offset']
            r = next(r for r in ranges if r['destoff'] == offset)
            assert r['rows'] == 1 and r['width'] == 128*256
        offset = w['tensors'][stage.SLOTS[36]]['offset']
        r = next(r for r in ranges if r['destoff'] == offset)
        assert (r['rows'], r['srcstride'], r['width'], r['dststride']) == (256, 256, 128, 128)


@pytest.mark.parametrize('inter', [128, 255, 384])
def test_tp_rejects_non_block128_fp8_intermediate_boundaries(inter):
    spec, _ = make_spec(inter=inter)
    with pytest.raises(ValueError, match='boundary'):
        parallel_layout.derive(spec, 'tp', 2)


def test_non_fp8_tp_can_shard_at_non_block_boundaries():
    spec, _ = make_spec(inter=12, dtype='torch.float32')
    result = parallel_layout.derive(spec, 'tp', 4)
    assert descriptor(result, 'l0_r3', stage.SLOTS[34])[2:4] == [3, 256]


@pytest.mark.parametrize('devices', [0, 1, 3, -2, True, 2.0, '2'])
def test_invalid_world_rejected(devices):
    spec, _ = make_spec()
    with pytest.raises(ValueError, match='devices'):
        parallel_layout.derive(spec, 'tp', devices)


@pytest.mark.parametrize('mode', ['pp', '', None])
def test_invalid_mode_rejected(mode):
    spec, _ = make_spec()
    with pytest.raises(ValueError, match='mode'):
        parallel_layout.derive(spec, mode, 2)


@pytest.mark.parametrize('filename', ['../l0.bin', '/tmp/l0.bin', 'a/b', 'a\\b', '.', '..', 'bad\tname', 'bad\x00name'])
def test_unsafe_source_basename_rejected(filename):
    spec, _ = make_spec(); spec['weights']['l0']['blob'] = filename
    with pytest.raises(ValueError, match='filename'):
        parallel_layout.derive(spec, 'ep', 2)


@pytest.mark.parametrize('change', ['negative', 'unaligned', 'bounds', 'scale_bounds', 'scale_overlap', 'scale_alignment',
                                  'nonfp8_scale', 'dtype', 'zero_dimension', 'overflow_dimension', 'ffn_shape', 'missing', 'unknown', 'meta_size'])
def test_malformed_metadata_rejected(change):
    spec, _ = make_spec(); w = spec['weights']['l0']; x = w['tensors'][stage.SLOTS[34]]
    if change == 'negative': x['offset'] = -128
    elif change == 'unaligned': x['offset'] += 1
    elif change == 'bounds': x['offset'] = (w['blob_bytes']+127)//128*128
    elif change == 'scale_bounds': x['scale_offset'] = (w['blob_bytes']+3)//4*4
    elif change == 'scale_overlap': x['scale_offset'] = x['offset']
    elif change == 'scale_alignment': x['scale_offset'] += 1
    elif change == 'nonfp8_scale': w['tensors']['input_layernorm.weight']['scale_offset'] = 4
    elif change == 'dtype': x['dtype'] = 'torch.float16'
    elif change == 'zero_dimension': x['shape'][0] = 0
    elif change == 'overflow_dimension': x['shape'][0] = 2**31
    elif change == 'ffn_shape': x['shape'] = [128, 256]
    elif change == 'missing': del w['tensors'][stage.SLOTS[34]]
    elif change == 'unknown': w['tensors']['mlp.experts.999.gate_proj.weight'] = deepcopy(x)
    elif change == 'meta_size': w['meta_bytes'] -= 32
    with pytest.raises(ValueError):
        parallel_layout.derive(spec, 'ep', 2)


@pytest.mark.parametrize('attention', [False, True])
def test_write_layout_only_emits_small_meta_json_and_tsv(tmp_path, attention):
    spec, _ = make_spec(attention=attention)
    result = parallel_layout.derive(spec, 'tp', 2, attention_tp=attention)
    assert parallel_layout.write_layout(result, tmp_path) is result
    assert json.loads((tmp_path/'parallel-layout.json').read_text()) == result
    assert not list(tmp_path.glob('*.bin'))
    assert len(list(tmp_path.glob('*.meta'))) == len(result['weights'])
    for name, w in result['weights'].items():
        raw = (tmp_path/w['meta']).read_bytes()
        assert len(raw) == w['meta_bytes']
        assert [list(record) for record in struct.iter_unpack('<QQiiii', raw)] == w['descriptors']
    lines = (tmp_path/'ranges.tsv').read_text().splitlines()
    assert len(lines) == len(result['ranges'])
    for line, r in zip(lines, result['ranges']):
        assert line.split('\t') == [str(r[k]) for k in ('tensor', 'rank', 'srcfile', 'srcoff', 'destoff', 'rows', 'srcstride', 'width', 'dststride')]
    changed = deepcopy(result); changed['weights']['l0_common']['meta'] = '../escape.meta'
    with pytest.raises(ValueError, match='filename'):
        parallel_layout.write_layout(changed, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('devices', [2, 4, 8])
def test_attention_tp_ownership_head_ranges_and_unchanged_ffn(mode, devices):
    spec, _ = make_spec(inter=1024, attention=True); before = deepcopy(spec)
    old = parallel_layout.derive(spec, mode, devices)
    result = parallel_layout.derive(spec, mode, devices, attention_tp=True)
    assert spec == before
    assert result['attention_tp'] is True and old['attention_tp'] is False
    assert 'attention_head_ranges' not in old
    assert json.loads(json.dumps(result)) == result
    assert result['source_spec_sha256'] == old['source_spec_sha256']
    assert len(result['weights']) == 2*(2*devices+1)
    assert result['slot_names'] == stage.slot_names(spec['config'])
    for layer, kind in enumerate(('kda', 'mla')):
        name = f'l{layer}'
        assert result['attention_head_ranges'][name] == dict(
            kind=kind, heads=8, head_dim=128,
            ranks=[dict(rank=r, start=r*8//devices, end=(r+1)*8//devices) for r in range(devices)])
        attention_keys = set(spec['weights'][name]['tensors']) & set(stage.SLOTS[8:34])
        for key in attention_keys:
            assert descriptor(result, name+'_common', key) == [0]*6
            assert descriptor(old, name+'_common', key)[2] > 0
        for key in stage.SLOTS[:8]:
            assert descriptor(result, name+'_common', key) == descriptor(old, name+'_common', key)
        for rank in range(devices):
            # Adding attention TP must not silently change the FFN policy.
            assert result['weights'][f'{name}_r{rank}'] == old['weights'][f'{name}_r{rank}']
            local = result['weights'][f'{name}_a{rank}']
            assert local['rank'] == rank and set(local['tensors']) == attention_keys
            assert all(descriptor(result, f'{name}_a{rank}', key) == [0]*6 for key in stage.SLOTS[:8])
            assert all(record == [0]*6 for record in local['descriptors'][34:])
            assert len(local['descriptors']) == len(stage.slot_names(spec['config']))
            assert result['graph_tensors'][f'{name}_a{rank}_weights'] == dict(
                shape=[local['blob_bytes']], dtype='uint8', bytes_per_elem=1, role='weight', rank=rank)
            assert result['graph_tensors'][f'{name}_a{rank}_meta']['shape'] == [local['meta_bytes']]
        if mode == 'ep' and layer == 0:
            for key in stage.SLOTS[34:37]:
                assert descriptor(result, name+'_common', key)[2] > 0
    for memory in result['rank_memory']:
        weights = [w for w in result['weights'].values() if w['rank'] == memory['rank']]
        assert memory['blob_bytes'] == sum(w['blob_bytes'] for w in weights)
        assert memory['meta_bytes'] == sum(w['meta_bytes'] for w in weights)
        assert memory['outer_bytes'] == old['rank_memory'][memory['rank']]['outer_bytes']
        assert memory['total_bytes'] == sum(memory[k] for k in ('blob_bytes', 'meta_bytes', 'outer_bytes'))
    assert all(r['rank'] == result['weights'][r['tensor'].removesuffix('_weights')]['rank'] for r in result['ranges'])


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('devices', [2, 4, 8])
def test_attention_payloads_scales_and_multipart_conv(mode, devices):
    spec, sources = make_spec(inter=1024, attention=True, payload=True)
    layout = parallel_layout.derive(spec, mode, devices, attention_tp=True)
    blobs = replay(layout, sources)
    # Slot numbers intentionally mirror native.h independently of derive's axes.
    row_shards = ({8,9,10,13,14,15,16,18}, {23,26})
    for layer in range(2):
        source = spec['weights'][f'l{layer}']; data = sources[source['blob']]
        for rank in range(devices):
            name = f'l{layer}_a{rank}'; local = layout['weights'][name]
            dest = blobs[name+'_weights']
            for key, tensor in local['tensors'].items():
                slot = stage.SLOTS.index(key); original = source['tensors'][key]
                shape = original['shape']; rows = shape[0]
                cols = 1
                for value in shape[1:]: cols *= value
                off, scale, nr, nc, kind, block = descriptor(layout, name, key)
                itemsize = {1: 2, 2: 4, 3: 1}[kind]
                if slot == 11:
                    count = 1024//devices; start = rank*count
                    indices = [group*1024+start+r for group in range(3) for r in range(count)]
                    expected_ranges = [dict(start=group*1024+start, rows=count) for group in range(3)]
                    assert tensor['source_ranges'] == expected_ranges
                    assert tensor['shape'] == [3*count, 1, 3]
                    assert (nr, nc) == (3*count, 3)
                    copies = [r for r in layout['ranges'] if r['tensor'] == name+'_weights' and off <= r['destoff'] < off+nr*nc*itemsize]
                    assert len(copies) == 3
                    for group, copy in enumerate(copies):
                        assert copy['srcoff'] == original['offset']+(group*1024+start)*3*4
                        assert copy['destoff'] == off+group*count*3*4
                        assert copy['rows'] == 1 and copy['width'] == count*3*4
                    cstart = 0
                elif slot in row_shards[layer]:
                    assert nr == rows//devices and nc == cols
                    indices = range(rank*(rows//devices), (rank+1)*(rows//devices)); cstart = 0
                    assert tensor['shard_axis'] == 0 and tensor['source_start'] == rank*nr
                elif slot == 20:
                    assert nr == rows and nc == cols//devices
                    indices = range(rows); cstart = rank*nc
                    assert tensor['shard_axis'] == 1 and tensor['source_start'] == cstart
                else:
                    assert (nr, nc) == (rows, cols)
                    indices = range(rows); cstart = 0
                    assert tensor['shard_axis'] is None and tensor['source_start'] == 0
                expected = b''.join(data[original['offset']+(r*cols+cstart)*itemsize:
                                           original['offset']+(r*cols+cstart+nc)*itemsize] for r in indices)
                assert dest[off:off+nr*nc*itemsize] == expected
                assert block == (128 if kind == 3 else 0)
                if kind == 3:
                    stride = (cols+127)//128
                    rstart = rank*(rows//devices) if slot in row_shards[layer] else 0
                    origin = original['scale_offset']+((rstart//128)*stride+cstart//128)*4
                    dr, dc = (nr+127)//128, (nc+127)//128
                    expected_scale = b''.join(data[origin+r*stride*4:origin+(r*stride+dc)*4] for r in range(dr))
                    assert dest[scale:scale+dr*dc*4] == expected_scale
                    if slot == 20:
                        # Column scales must keep the original row stride, not
                        # accidentally copy a contiguous scale-grid prefix.
                        copies = [r for r in layout['ranges'] if r['tensor'] == name+'_weights' and r['destoff'] == scale]
                        assert len(copies) == 1
                        assert (copies[0]['rows'], copies[0]['srcstride'], copies[0]['width'], copies[0]['dststride']) == (dr, stride*4, dc*4, dc*4)
                else:
                    assert scale == 0


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('layer,slot', [(0,8), (0,11), (0,12), (0,15), (0,19), (0,20),
                                      (0,0), (0,6), (1,21), (1,23), (1,26), (1,27), (1,33)])
def test_attention_tp_fails_closed_on_incomplete_descriptors(mode, layer, slot):
    spec, _ = make_spec(attention=True)
    del spec['weights'][f'l{layer}']['tensors'][stage.SLOTS[slot]]
    with pytest.raises(ValueError, match='attention slots'):
        parallel_layout.derive(spec, mode, 2, attention_tp=True)


@pytest.mark.parametrize('field', ['layer_types', 'hc_mult', 'num_attention_heads', 'q_lora_rank'])
def test_attention_tp_fails_closed_on_missing_config(field):
    spec, _ = make_spec(attention=True)
    del spec['config'][field]
    with pytest.raises(ValueError):
        parallel_layout.derive(spec, 'ep', 2, attention_tp=True)


@pytest.mark.parametrize('schedule', [None, [], ['linear_attention'],
                                      ['linear_attention', 'unknown_attention']])
def test_attention_tp_rejects_invalid_layer_schedule(schedule):
    spec, _ = make_spec(attention=True)
    spec['config']['layer_types'] = schedule
    with pytest.raises(ValueError, match='attention layer schedule'):
        parallel_layout.derive(spec, 'ep', 2, attention_tp=True)


def test_attention_tp_accepts_full_attention_fixture_alias():
    spec, _ = make_spec(attention=True)
    spec['config']['layer_types'][1] = 'full_attention'
    result = parallel_layout.derive(spec, 'ep', 2, attention_tp=True)
    assert result['attention_head_ranges']['l1']['kind'] == 'mla'


def test_attention_tp_rejects_unexpected_cross_attention_slots():
    spec, _ = make_spec(attention=True)
    # Reuse a valid nonoverlapping descriptor under the wrong attention family.
    source = spec['weights']['l0']
    source['tensors'][stage.SLOTS[21]] = source['tensors'].pop(stage.SLOTS[12])
    with pytest.raises(ValueError, match='attention slots'):
        parallel_layout.derive(spec, 'ep', 2, attention_tp=True)


def test_attention_tp_rejects_legacy_partial_attention_fixture():
    spec, _ = make_spec()
    with pytest.raises(ValueError, match='attention'):
        parallel_layout.derive(spec, 'ep', 2, attention_tp=True)


@pytest.mark.parametrize('kind', ['kda', 'mla'])
@pytest.mark.parametrize('devices', [2, 4, 8])
def test_attention_tp_rejects_indivisible_heads(kind, devices):
    spec, _ = make_spec(inter=1024, attention=True)
    if kind == 'kda':
        spec['config']['linear_attn_config']['num_heads'] = devices+1
    else:
        spec['config']['num_attention_heads'] = devices+1
    with pytest.raises(ValueError, match='attention head count'):
        parallel_layout.derive(spec, 'ep', devices, attention_tp=True)


@pytest.mark.parametrize('slot', [8,11,15,20,23,26,27,33])
def test_attention_tp_rejects_wrong_descriptor_shapes(slot):
    spec, _ = make_spec(attention=True)
    layer = 0 if slot < 21 else 1
    tensor = spec['weights'][f'l{layer}']['tensors'][stage.SLOTS[slot]]
    tensor['shape'][0] -= 1
    with pytest.raises(ValueError, match='attention descriptor dimensions'):
        parallel_layout.derive(spec, 'ep', 2, attention_tp=True)


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('devices', [2, 4, 8])
@pytest.mark.parametrize('slot,dims', [(8,(72,128)), (13,(72,128)), (18,(72,128)),
                                      (20,(72,128)), (23,(128,72)), (26,(128,36))])
def test_attention_tp_rejects_misaligned_fp8_row_and_column_shards(mode, devices, slot, dims):
    # Select dimensions whose local width/height is never a whole FP8 block.
    # Other projections are FP32 so the rejection targets this exact slot.
    spec, _ = make_spec(inter=1024, attention=True, attention_dims=dims,
                        attention_fp8_slots={slot})
    with pytest.raises(ValueError, match='attention shard boundary'):
        parallel_layout.derive(spec, mode, devices, attention_tp=True)


@pytest.mark.parametrize('mode', ['tp', 'ep'])
@pytest.mark.parametrize('devices', [2, 4, 8])
def test_mla_output_column_fp8_boundary_is_checked_independently(mode, devices):
    spec, _ = make_spec(inter=1024, attention=True, attention_dims=(128,72),
                        attention_fp8_slots={20})
    with pytest.raises(ValueError, match='attention shard boundary.*o_proj'):
        parallel_layout.derive(spec, mode, devices, attention_tp=True)


def test_non_fp8_attention_accepts_unaligned_but_head_divisible_slices():
    spec, _ = make_spec(inter=1024, attention=True, attention_dims=(3,5),
                        attention_fp8_slots=set())
    result = parallel_layout.derive(spec, 'ep', 8, attention_tp=True)
    assert descriptor(result, 'l0_a7', stage.SLOTS[8])[2:4] == [3,256]
    assert descriptor(result, 'l0_a7', stage.SLOTS[11])[2:4] == [9,3]
    assert descriptor(result, 'l0_a7', stage.SLOTS[20])[2:4] == [256,3]
    assert descriptor(result, 'l1_a7', stage.SLOTS[23])[2:4] == [5,128]
    assert descriptor(result, 'l1_a7', stage.SLOTS[26])[2:4] == [10,128]
    assert descriptor(result, 'l1_a7', stage.SLOTS[20])[2:4] == [256,5]


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_attention_tp_flag_requires_bool(value):
    spec, _ = make_spec()
    with pytest.raises(ValueError, match='attention_tp'):
        parallel_layout.derive(spec, 'tp', 2, attention_tp=value)
