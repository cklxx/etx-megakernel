"""Compact EP/TP FFN layouts, derived without opening staged weight payloads.

By default attention, norms and routing stay on root (rank 0). EP assigns routed
expert ``e`` to ``e % devices``; shared experts stay in rank-0 FFN, while dense
FFNs stay in common (the original dense-up/down root path). TP shards every
dense, shared and routed FFN: gate/up rows and down columns use the same
contiguous intermediate interval. Optional attention TP adds separate ``aR``
weights for each rank, leaving mHC/input normalization on root. Native global
descriptor slots are never renumbered.

Host contract: allocate each graph tensor, zero its padding, load generated meta
files from the build directory, then execute ranges.tsv (no header), columns::

    tensor rank srcfile srcoff destoff rows srcstride width dststride

Each record copies ``width`` bytes for each row, at the given byte strides.
Sources are basenames relative to the original stage directory. A host can split
any row into bounded chunks; no full source/destination blob mapping is required.
Only JSON, this compact TSV and small descriptor files are generated, never new
weight blobs. Embedding/norm/head retain the parent's ordinary root load.tsv.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

from examples.glm53 import stage

_ABI = struct.Struct('<QQiiii')
_DTYPES = {'torch.bfloat16': (1, 2), 'torch.float32': (2, 4),
           'torch.float8_e4m3fn': (3, 1)}
_ZERO = [0, 0, 0, 0, 0, 0]
_U64 = 2**64 - 1
_I32 = 2**31 - 1
_RANGE_FIELDS = ('tensor', 'rank', 'srcfile', 'srcoff', 'destoff', 'rows',
                 'srcstride', 'width', 'dststride')
_DENSE = set(stage.SLOTS[34:37])
_SHARED = set(stage.SLOTS[39:42])


def _int(value, what, minimum=0, maximum=_U64):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'invalid {what}: {value!r}')
    return value


def _basename(value):
    if (not isinstance(value, str) or not value or value in ('.', '..')
            or Path(value).name != value or '/' in value or '\\' in value
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f'unsafe source filename: {value!r}')
    return value


def _align(value, block):
    return (value + block - 1) // block * block


def _validated(source, slots):
    """Reconstruct and validate source descriptors using JSON alone."""
    _basename(source['blob']); _basename(source['meta'])
    size = _int(source['blob_bytes'], 'blob_bytes', 1)
    if _int(source['meta_bytes'], 'meta_bytes') != _ABI.size * len(slots):
        raise ValueError('invalid source meta_bytes / global slot count')
    tensors = source['tensors']
    if not isinstance(tensors, dict) or set(tensors) - set(slots):
        raise ValueError('unknown source tensor slots')
    result = {}; occupied = []
    for key, tensor in tensors.items():
        shape = tensor['shape']
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError(f'invalid descriptor shape: {key}')
        for dim in shape:
            _int(dim, 'descriptor dimension', 1, _I32)
        rows = shape[0]; cols = math.prod(shape[1:]) if len(shape) > 1 else 1
        _int(cols, 'descriptor columns', 1, _I32)
        if tensor['dtype'] not in _DTYPES:
            raise ValueError(f'unsupported dtype: {tensor["dtype"]}')
        kind, itemsize = _DTYPES[tensor['dtype']]
        off = _int(tensor['offset'], 'tensor offset')
        scale = _int(tensor['scale_offset'], 'scale offset')
        if off % 128:
            raise ValueError('unaligned source tensor offset')
        occupied.append((off, off + rows * cols * itemsize))
        if kind == 3:
            if len(shape) != 2 or scale % 4:
                raise ValueError('invalid FP8 matrix / scale alignment')
            occupied.append((scale, scale + 4 * ((rows + 127)//128) * ((cols + 127)//128)))
        elif scale:
            raise ValueError('unexpected scale on non-FP8 tensor')
        result[key] = (rows, cols, kind, itemsize, off, scale)
    occupied.sort()
    if (any(a < 0 or b > size or b > _U64 or b <= a for a, b in occupied)
            or any(a[1] > b[0] for a, b in zip(occupied, occupied[1:]))):
        raise ValueError('overlapping/out-of-bounds tensor or scale extent')
    return result


def _ffn_axis(key):
    if key in _DENSE or key in _SHARED or key.startswith('mlp.experts.'):
        return 1 if key.endswith('down_proj.weight') else 0
    return None


def _attention_contract(t, layer, source, tensors, devices):
    """Validate the complete native attention ABI and return its slicing axes."""
    schedule = t.get('layer_types')
    if (not isinstance(schedule, list) or len(schedule) != t['num_hidden_layers']
            or any(kind not in ('linear_attention', 'deepseek_sparse_attention', 'full_attention') for kind in schedule)):
        raise ValueError('invalid attention layer schedule')
    h = _int(t['hidden_size'], 'hidden size', 1, _I32)
    hc = _int(t.get('hc_mult'), 'attention hc_mult', 1, _I32)
    shapes = dict(zip(stage.SLOTS[:8], [(hc*(hc+2), hc*h), (hc*(hc+2),), (3,),
                                      (hc*(hc+2), hc*h), (hc*(hc+2),), (3,), (h,), (h,)]))
    axes = {}
    if schedule[layer] == 'linear_attention':
        la = t.get('linear_attn_config', {})
        heads = _int(la.get('num_heads', t.get('linear_num_heads', 64)), 'KDA heads', 1, _I32)
        dim = _int(la.get('head_dim', t.get('linear_head_dim', 128)), 'KDA head dimension', 1, _I32)
        conv = _int(la.get('short_conv_kernel_size', t.get('linear_conv_kernel_dim', 4)), 'KDA convolution width', 1, _I32)
        qd = heads*dim
        values = [(qd,h), (qd,h), (qd,h), (3*qd,1,conv), (dim,h), (qd,dim),
                  (qd,), (heads,), (heads,h), (dim,h), (qd,dim), (dim,), (h,qd)]
        shapes.update(zip(stage.SLOTS[8:21], values))
        axes = {stage.SLOTS[i]: 0 for i in (8, 9, 10, 11, 13, 14, 15, 16, 18)}
        axes[stage.SLOTS[20]] = 1
        kind = 'kda'
    else:
        heads = _int(t.get('num_attention_heads'), 'MLA heads', 1, _I32)
        dim = _int(t.get('qk_nope_head_dim'), 'MLA head dimension', 1, _I32)
        qr = _int(t.get('q_lora_rank'), 'MLA Q rank', 1, _I32)
        kvr = _int(t.get('kv_lora_rank'), 'MLA KV rank', 1, _I32)
        ih = _int(t.get('index_n_heads'), 'indexer heads', 1, _I32)
        ident = _int(t.get('index_head_dim'), 'indexer head dimension', 1, _I32)
        pool = _int(t.get('index_kpool'), 'indexer pool', 1, _I32)
        values = [(qr,h), (qr,), (heads*dim,qr), (kvr,h), (kvr,),
                  (heads*2*dim,kvr), (ih*ident,qr), (ident,h), (ident,), (ident,),
                  (ih,h), (ident,h), (pool,ident)]
        shapes.update(zip(stage.SLOTS[21:34], values))
        shapes[stage.SLOTS[20]] = (h, heads*dim)
        axes = {stage.SLOTS[23]: 0, stage.SLOTS[26]: 0, stage.SLOTS[20]: 1}
        kind = 'mla'
    if heads % devices:
        raise ValueError(f'attention head count must divide evenly by devices: l{layer}')
    actual = {key for key in tensors if key in stage.SLOTS[:34]}
    if actual != set(shapes):
        raise ValueError(f'incomplete or unexpected attention slots: l{layer}')
    for key, shape in shapes.items():
        if tuple(source['tensors'][key]['shape']) != shape:
            raise ValueError(f'invalid attention descriptor dimensions: l{layer}/{key}')
    summary = dict(kind=kind, heads=heads, head_dim=dim,
                   ranks=[dict(rank=r, start=r*(heads//devices), end=(r+1)*(heads//devices))
                          for r in range(devices)])
    return axes, summary


def derive(spec, mode, devices, attention_tp=False):
    """Return a JSON-serializable layout; do not read files or mutate ``spec``.

    ``weights[name]`` contains rank, logical blob/meta basenames, byte counts,
    full global-slot ``descriptors`` (<QQiiii records), and local ``tensors``.
    ``graph_tensors`` provides the corresponding uint8 graph declarations.
    ``rank_memory`` includes root's unchanged embedding/norm/head allocations.
    ``source_spec_sha256`` hashes canonical JSON of the entire source spec.
    ``attention_tp`` independently shards attention heads in either FFN mode;
    its per-layer ``attention_head_ranges`` use half-open global head intervals.
    Attention helpers retain global-index activation/state buffers, but read the
    compact local descriptors (CONV packs Q, K and V in three local row groups).
    """
    if type(attention_tp) is not bool:
        raise ValueError('attention_tp must be bool')
    if mode not in ('tp', 'ep'):
        raise ValueError('mode must be tp or ep')
    if type(devices) is not int or devices not in (2, 4, 8):
        raise ValueError('devices must be 2, 4 or 8')
    t = spec['config']; slots = stage.slot_names(t)
    n = _int(t['num_hidden_layers'], 'layer count', 1, _I32)
    experts = _int(t['n_routed_experts'], 'expert count', 1, _I32)
    h = _int(t['hidden_size'], 'hidden size', 1, _I32)
    if len(t['mlp_layer_types']) != n or any(x not in ('dense', 'sparse') for x in t['mlp_layer_types']):
        raise ValueError('invalid FFN layer schedule')
    expected_weights = {f'l{i}' for i in range(n)} | {'embedding', 'norm', 'head'}
    if set(spec['weights']) != expected_weights:
        raise ValueError('incomplete source weights')
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    layout = dict(schema=1, mode=mode, devices=devices, attention_tp=attention_tp, source_spec_sha256=digest,
                  source={k: spec[k] for k in ('schema', 'model', 'revision', 'config_sha256', 'fixture', 'slice') if k in spec},
                  slot_names=slots, weights={}, ranges=[], graph_tensors={},
                  rank_memory=[dict(rank=r, blob_bytes=0, meta_bytes=0, outer_bytes=0, total_bytes=0) for r in range(devices)])
    if attention_tp:
        layout['attention_head_ranges'] = {}

    def add_range(name, source, src, dest, rows, srcstride, width, dststride):
        # Collapse contiguous row copies (notably TP gate/up and row scales).
        if srcstride == width == dststride:
            width *= rows; rows = 1; srcstride = dststride = width
        if (min(rows, width) <= 0 or min(src, dest) < 0 or srcstride < width or dststride < width
                or src + (rows-1)*srcstride + width > source['blob_bytes']
                or dest + (rows-1)*dststride + width > _U64):
            raise ValueError('out-of-bounds strided copy')
        layout['ranges'].append(dict(tensor=name+'_weights', rank=layout['weights'][name]['rank'],
                                     srcfile=source['blob'], srcoff=src, destoff=dest, rows=rows,
                                     srcstride=srcstride, width=width, dststride=dststride))

    for layer in range(n):
        layer_name = f'l{layer}'; source = spec['weights'][layer_name]
        tensors = _validated(source, slots)
        attention_axes = {}
        if attention_tp:
            attention_axes, summary = _attention_contract(t, layer, source, tensors, devices)
            layout['attention_head_ranges'][layer_name] = summary
        dense = t['mlp_layer_types'][layer] == 'dense'
        required = _DENSE if dense else _SHARED | set(slots[42:]) | set(stage.SLOTS[37:39])
        present_ffn = {key for key in tensors if _ffn_axis(key) is not None or key in stage.SLOTS[37:39]}
        if present_ffn != required:
            raise ValueError(f'incomplete or unexpected FFN slots: {layer_name}')
        inter = _int(t['intermediate_size' if dense else 'moe_intermediate_size'], 'intermediate size', 1, _I32)
        for key in required:
            axis = _ffn_axis(key)
            if axis is not None:
                rows, cols, *_ = tensors[key]
                expected = (h, inter) if axis == 1 else (inter, h)
                if len(source['tensors'][key]['shape']) != 2 or (rows, cols) != expected:
                    raise ValueError(f'invalid FFN descriptor dimensions: {layer_name}/{key}')
            elif key == stage.SLOTS[37] and tensors[key][:2] != (experts, h):
                raise ValueError('invalid router descriptor dimensions')
            elif key == stage.SLOTS[38] and tensors[key][:2] != (experts, 1):
                raise ValueError('invalid router bias descriptor dimensions')
        names = [layer_name+'_common'] + [f'{layer_name}_r{rank}' for rank in range(devices)]
        attention_names = [f'{layer_name}_a{rank}' for rank in range(devices)] if attention_tp else []
        for name in names + attention_names:
            rank = 0 if name == names[0] else int(name.rsplit('_', 1)[1][1:])
            layout['weights'][name] = dict(rank=rank, blob=name+'.bin', meta=name+'.meta',
                                          blob_bytes=0, meta_bytes=32*len(slots),
                                          descriptors=[_ZERO.copy() for _ in slots], tensors={})
        for slot, key in enumerate(slots):
            if key not in tensors:
                continue
            rows, cols, kind, itemsize, off, scale = tensors[key]
            axis = _ffn_axis(key)
            is_attention = attention_tp and key in stage.SLOTS[8:34]
            if is_attention:
                axis = attention_axes.get(key)
                owners = attention_names
            elif axis is None or (mode == 'ep' and key in _DENSE):
                owners = [names[0]]
            elif mode == 'tp':
                owners = names[1:]
            else:
                rank = int(key.split('.')[2]) % devices if key.startswith('mlp.experts.') else 0
                owners = [names[1+rank]]
            shard = axis is not None and (is_attention or mode == 'tp')
            multipart = is_attention and key == stage.SLOTS[11]
            # KDA convolution rows are [all Q, all K, all V], not one
            # contiguous attention-head interval across the entire descriptor.
            extent = (cols if axis == 1 else rows//3 if multipart else rows) if shard else 0
            if shard and (extent % devices or (kind == 3 and (extent // devices) % 128)):
                family = 'attention' if is_attention else 'FFN'
                raise ValueError(f'{family} shard boundary must divide evenly and be FP8 block128 aligned: {layer_name}/{key}')
            for name in owners:
                w = layout['weights'][name]; rank = w['rank']
                start = rank * (extent // devices) if shard else 0
                nr = rows // devices if shard and axis == 0 else rows
                nc = cols // devices if shard and axis == 1 else cols
                dst = _align(w['blob_bytes'], 128)
                row_ranges = [(group*extent+start, extent//devices) for group in range(3)] if multipart else [(start if shard and axis == 0 else 0, nr)]
                destrow = 0
                for srcrow, count in row_ranges:
                    src = off + (srcrow*cols + (start if shard and axis == 1 else 0))*itemsize
                    add_range(name, source, src, dst+destrow*nc*itemsize, count, cols*itemsize, nc*itemsize, nc*itemsize)
                    destrow += count
                end = dst + nr*nc*itemsize; ds = 0
                if kind == 3:
                    ds = _align(end, 4)
                    sc = (cols+127)//128
                    dr, dc = (nr+127)//128, (nc+127)//128
                    destrow = 0
                    for srcrow, count in row_ranges:
                        scale_rows = (count+127)//128
                        srcscale = scale + ((srcrow//128)*sc + (start//128 if shard and axis == 1 else 0))*4
                        add_range(name, source, srcscale, ds+destrow*dc*4, scale_rows, sc*4, dc*4, dc*4)
                        destrow += scale_rows
                    end = ds + dr*dc*4
                w['blob_bytes'] = end
                w['descriptors'][slot] = [dst, ds, nr, nc, kind, 128 if kind == 3 else 0]
                source_shape = list(source['tensors'][key]['shape'])
                local_shape = source_shape.copy()
                if shard:
                    local_shape[axis] //= devices
                w['tensors'][key] = dict(slot=slot, shape=local_shape, dtype=source['tensors'][key]['dtype'],
                                         offset=dst, scale_offset=ds, source_start=start,
                                         shard_axis=axis if shard else None)
                if multipart:
                    w['tensors'][key]['source_ranges'] = [dict(start=a, rows=b) for a, b in row_ranges]
        for name in names + attention_names:
            w = layout['weights'][name]
            w['blob_bytes'] = max(1, w['blob_bytes'])  # empty EP ranks still have valid arguments
            _int(w['blob_bytes'], 'destination blob_bytes', 1)
            memory = layout['rank_memory'][w['rank']]
            for kind, suffix in (('blob', 'weights'), ('meta', 'meta')):
                size = w[kind+'_bytes']; memory[kind+'_bytes'] += size
                layout['graph_tensors'][name+'_'+suffix] = dict(shape=[size], dtype='uint8', bytes_per_elem=1, role='weight', rank=w['rank'])
    for name in ('embedding', 'norm', 'head'):
        source = spec['weights'][name]
        _validated(source, list(source['tensors']))
        layout['rank_memory'][0]['outer_bytes'] += source['blob_bytes'] + source['meta_bytes']
    for memory in layout['rank_memory']:
        memory['total_bytes'] = memory['blob_bytes'] + memory['meta_bytes'] + memory['outer_bytes']
    return layout


def write_layout(layout, out):
    """Emit only ranges.tsv, parallel-layout.json and small native .meta files.

    Source files are never opened. Logical ``blob`` names are allocation labels,
    not files to load from the build directory. Returns ``layout`` unchanged.
    """
    out = Path(out)
    # Validate metadata before emitting any files (and refuse unsafe output names).
    metadata = {}
    for w in layout['weights'].values():
        filename = _basename(w['meta'])
        if filename in metadata:
            raise ValueError('duplicate derived meta filename')
        raw = b''.join(_ABI.pack(*record) for record in w['descriptors'])
        if len(raw) != w['meta_bytes']:
            raise ValueError('derived metadata size mismatch')
        metadata[filename] = raw
    text = ''.join('\t'.join(str(record[key]) for key in _RANGE_FIELDS)+'\n' for record in layout['ranges'])
    out.mkdir(parents=True, exist_ok=True)
    for filename, raw in metadata.items():
        (out/filename).write_bytes(raw)
    (out/'ranges.tsv').write_text(text)
    (out/'parallel-layout.json').write_text(json.dumps(layout, indent=2, allow_nan=False)+'\n')
    return layout
