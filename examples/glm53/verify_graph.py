"""Causal k+1-position GLM verification graph, with weights-free CPU execution.

Inputs are the last accepted token followed by k draft tokens (teacher forced).
Every position returns full logits and an argmax. KDA heads scan the block;
DSA runs its existing per-position causal index top-k; MoE up and down rows
share each expert weight across positions. State advances through the block: callers
must restore/replay state after a rejected suffix. No speculative rollback or
GPU performance/correctness qualification is implied by the CPU check.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess

from etx.frontends import hip_link
from etx.ir import Graph
from examples.glm53 import graph_cpu, model
from examples.glm53.options import Options

SOURCE = 'examples/glm53/verify_phases.h'


def _positions(k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer draft count')
    return k + 1


def build_from(stage, k, batch=1, options=None):
    """Build one verify step. Weights and runtime state are shared across positions.

    Blocks support 2 through 6 positions (k=1 through 5).
    This first verify path supports phased pipeline placement; TP/EP layouts
    need their own multi-position dispatch and are explicitly refused.
    """
    count = _positions(k)
    if count > 6:
        raise ValueError('verify read-once MoE supports at most 6 positions (k <= 5)')
    options = Options() if options is None else options
    if not options.phased or options.parallel or options.attention_tp:
        raise ValueError('verify requires phased serial attention and nonparallel FFN')
    base = model.build_from(stage, batch, options)
    graph = Graph(f'glm53_verify_k{k}_pp{options.devices}')
    graph.symbols = list(base.symbols)

    def local(name, p):
        if name not in base.tensors:
            return f'p{p}_{name}'
        t = base.tensors[name]
        return name if t.role in ('weight', 'runtime') else f'p{p}_{name}'

    for name, tensor in base.tensors.items():
        for p in range(1 if tensor.role in ('weight', 'runtime') else count):
            n = local(name, p)
            graph.tensors[n] = replace(tensor, name=n)
    produced, readers = {}, {}
    resource = replace(base.grid('l0_p1').resource)

    def emit(name, body, device, n, args, reads, writes, consts, res=resource):
        # Include WAR as well as RAW/WAW: e.g. a later cache publication must
        # not overtake an earlier position's attention on an asynchronous GPU.
        waits = {produced[x] for x in reads + writes if x in produced}
        for x in writes:
            waits.update(readers.get(x, set()))
        ev = 'e_' + name
        graph.etensor(ev, ('B',), wait_count=n)
        graph.call_device(name, ('B', n), body, device=device, resource=res,
                          args=args, reads=reads, writes=writes, consts=consts,
                          in_edges={e: 'bt->b' for e in sorted(waits)},
                          out_edges={ev: 'bt->b'})
        for x in writes:
            produced[x] = ev
            readers[x] = set()
        for x in reads:
            readers.setdefault(x, set()).add(ev)

    def copy(name, p):
        g = base.grid(name)
        rename = lambda names: [local(x, p) for x in names]
        n = g.grid[1] if len(g.grid) == 2 else 1
        emit(f'p{p}_{name}', replace(g.body, source=SOURCE), g.device, n, rename(g.args),
             rename(g.reads), rename(g.writes), g.consts, replace(g.resource,
                 lds_bytes=max(g.resource.lds_bytes, resource.lds_bytes)))

    def batched(name, symbol, device, shared, rows, reads, writes, flag=0):
        args = list(shared)
        rd, wr = list(shared), []
        for p in range(count):
            args.extend(local(x, p) for x in rows)
            rd.extend(local(x, p) for x in reads)
            wr.extend(local(x, p) for x in writes)
        emit(name, hip_link(SOURCE, symbol), device, 1, args, rd, wr,
             (1, flag, count, 0))

    batched('embedding_rows', 'glm53_verify_embed', base.grid('embedding').device,
            ['embedding_weights', 'embedding_meta'], ['step', 'embedded'],
            ['step'], ['embedded'])
    for i, kind in enumerate(stage['config']['layer_types']):
        layer = f'l{i}'
        linear = kind == 'linear_attention'
        d = stage['dimensions']
        # Per-row activations and reusable coefficients, never copied weights.
        hc = (2*d['HC'] + d['HC']*d['HC']+3)//4*4
        work_size = (2*hc + d['H'] + 5*d['QDIM'] + 2*d['NH']*d['AD'] + 2*d['ID'] + d['QR'] + d['KVR'])
        for p in range(count):
            for suffix in ('vwork', 'scanwork', 'fwork'):
                graph.tensor(local(layer+'_'+suffix, p), ('B', work_size), role='scratch',
                             dtype='float32', bytes_per_elem=4)
        src = base.grid(layer+'_p1')
        previous = src.args[2]
        batched(layer+'_prologue', 'glm53_verify_prologue', src.device,
                [layer+'_weights', layer+'_meta'],
                [previous, layer+'_hca', layer+'_p1', layer+'_vwork'],
                [previous], [layer+'_hca', layer+'_p1', layer+'_vwork'], int(linear))
        common_size = d['LD'] + d['LH'] + d['QDIM'] + d['POOL']*d['ID']
        graph.tensor(layer+'_vcommon', ('B', common_size), role='scratch',
                     dtype='float32', bytes_per_elem=4)
        if linear:
            src = base.grid(layer + '_kda')
            args = [layer + '_weights', layer + '_meta', layer + '_state', layer+'_vcommon']
            inputs, outputs = [], []
            for p in range(count):
                inputs.append(local(layer + '_p1', p))
                outputs.append(local(layer + '_att', p))
                args.extend([inputs[-1], outputs[-1], local(layer+'_scanwork', p)])
            emit(layer + '_kda_scan', hip_link(SOURCE, 'glm53_verify_kda_scan'),
                 src.device, 1, args, args[:3] + inputs,
                 [layer + '_state', layer+'_vcommon'] + outputs +
                 [local(layer+'_scanwork', p) for p in range(count)],
                 (1, 1, count, src.consts[3]))
        else:
            # Projection results remain private until the scan publishes each
            # row. No future KV entries become visible to earlier attention.
            src = base.grid(layer + '_mla')
            args = [layer+'_weights', layer+'_meta', layer+'_state', layer+'_vcommon']
            reads, writes = args[:3], [layer+'_state', layer+'_vcommon']
            for p in range(count):
                names = [local(n, p) for n in ('step', layer+'_p1', layer+'_p2',
                                              layer+'_att', layer+'_scanwork')]
                args.extend(names)
                reads.extend([names[0], names[1]])
                writes.extend(names[2:])
            emit(layer+'_dsa_scan', hip_link(SOURCE, 'glm53_verify_dsa_scan'),
                 src.device, 1, args, reads, writes, (1, 0, count, src.consts[3]))
        batched(layer+'_output', 'glm53_verify_output', src.device,
                [layer+'_weights', layer+'_meta'],
                [previous, layer+'_vwork', layer+'_att', layer+'_mid'],
                [previous, layer+'_vwork', layer+'_att'], [layer+'_mid'])
        moe = stage['config']['mlp_layer_types'][i] == 'sparse'
        batched(layer+'_ffn_input', 'glm53_verify_ffn_input', src.device,
                [layer+'_weights', layer+'_meta'],
                [layer+'_mid', layer+'_hcf', layer+'_rt', layer+'_fwork'],
                [layer+'_mid'], [layer+'_hcf', layer+'_rt', layer+'_fwork'], int(moe))
        if moe:
            for p in range(count):
                graph.tensor(f'p{p}_{layer}_partial',
                             ('B', (stage['dimensions']['TOPK'] + 1) * stage['dimensions']['H']),
                             role='scratch', dtype='float32', bytes_per_elem=4)
            src = base.grid(layer + '_up')
            batched(layer+'_route', 'glm53_verify_route', src.device,
                    [layer+'_weights', layer+'_meta'], [layer+'_rt', layer+'_sel'],
                    [layer+'_rt'], [layer+'_sel'])
            args = [layer + '_weights', layer + '_meta']
            reads, writes = args.copy(), []
            for p in range(count):
                rt, act, sel = [local(layer + '_' + s, p) for s in ('rt', 'act', 'sel')]
                args.extend([rt, act, sel])
                reads.extend([rt, sel])
                writes.append(act)
            n = model.tiles(stage['config']['moe_intermediate_size'], options.effective_up_pairs)
            total = n * (stage['dimensions']['EXPERTS'] + 1)
            emit(layer + '_expert_up', hip_link(SOURCE, 'glm53_verify_expert_up'),
                 src.device, total, args, reads, writes, (total, n, count, 0))
            args = [layer + '_weights', layer + '_meta']
            reads, writes = args.copy(), []
            for p in range(count):
                act, sel, partial = [local(layer + '_' + s, p)
                                     for s in ('act', 'sel', 'partial')]
                args.extend([act, sel, partial])
                reads.extend([act, sel])
                writes.append(partial)
            # Partition the existing down task budget over experts; each lane
            # loops over its output-row range in the read-once primitive.
            mix_tiles = base.grid(layer + '_down').grid[1]
            experts = stage['dimensions']['EXPERTS'] + 1
            n = max(1, (mix_tiles + experts - 1) // experts)
            total = n * experts
            emit(layer + '_expert_down', hip_link(SOURCE, 'glm53_verify_expert_down'),
                 src.device, total, args, reads, writes, (total, n, count, 0))
            for p in range(count):
                names = [layer + '_weights', layer + '_meta'] + [local(layer + '_' + s, p)
                         for s in ('partial', 'sel', 'mid', 'fwork', 'out')]
                emit(f'p{p}_{layer}_mix', hip_link(SOURCE, 'glm53_verify_expert_mix'),
                     src.device, mix_tiles, names, names[:-1], names[-1:],
                     (mix_tiles, 0, 0, 0))
        else:
            batched(layer+'_dense_up', 'glm53_verify_dense_up', src.device,
                    [layer+'_weights', layer+'_meta'], [layer+'_rt', layer+'_act'],
                    [layer+'_rt'], [layer+'_act'])
            batched(layer+'_dense_down', 'glm53_verify_dense_down', src.device,
                    [layer+'_weights', layer+'_meta'],
                    [layer+'_act', layer+'_mid', layer+'_fwork', layer+'_out'],
                    [layer+'_act', layer+'_mid', layer+'_fwork'], [layer+'_out'])
    # Normalize all rows before streaming each LM-head weight through every
    # position accumulator. Head scratch holds the normalized input between
    # the preparation and projection boundaries.
    src = base.grid('head')
    args = ['norm_weights', 'norm_meta', 'head_weights', 'head_meta']
    reads, writes = args.copy(), []
    for p in range(count):
        inp, out = local(src.args[0], p), local('head_scratch', p)
        args.extend([inp, out])
        reads.append(inp)
        writes.append(out)
    emit('head_rows', hip_link(SOURCE, 'glm53_verify_head'), src.device,
         1, args, reads, writes, (1, 0, count, 0))
    for phase in ('argmax', 'feedback'):
        for p in range(count):
            copy(phase, p)
    return graph



def instrumented_shim(work):
    """Instrument the scalar CPU reader, without modifying shared tile sources.

    Each event identifies a blob, descriptor offset, byte range and repetition
    count. Identical ranges are compressed in C++; the caller labels the step.
    Scales remain separate events, including every repeated scale lookup.
    """
    native = (graph_cpu.ROOT / 'examples/glm53/native.h').read_text()
    anchor = '  uint64_t i=(uint64_t)r*w.cols+c;'
    if native.count(anchor) != 1:
        raise RuntimeError('CPU weight reader instrumentation anchor changed')
    native = native.replace('#include "native_options.h"',
                            '#include "examples/glm53/native_options.h"')
    native = native.replace(anchor, anchor + r"""
  if(glm_observe){
    int size=w.kind==1?2:w.kind==2?4:1;
    glm_record({(uintptr_t)blob,w.offset,w.offset+i*size,(uint64_t)size,0});
    if(w.kind==3)glm_record({(uintptr_t)blob,w.offset,
      w.scale+4*((r/w.block)*((w.cols+w.block-1)/w.block)+c/w.block),4,1});
  }
""")
    import re
    emitted = set()

    def mirror(relative):
        target = Path(work) / ('observed_' + Path(relative).name)
        if relative in emitted:
            return target.name
        emitted.add(relative)
        text = native if relative.endswith('/native.h') else (graph_cpu.ROOT / relative).read_text()

        def include(match):
            dependency = match[1]
            if dependency.startswith('examples/glm53/'):
                return '#include "' + mirror(dependency) + '"'
            return match[0]

        target.write_text(re.sub(r'#include "([^"]+)"', include, text))
        return target.name

    entry = mirror(SOURCE)
    prefix = r"""
#include <map>
#include <array>
#include <stdint.h>
static bool glm_observe=false;
using ReadKey=std::array<uint64_t,5>;
static std::map<ReadKey,uint64_t> glm_reads;
static ReadKey pending[2];
static uint64_t multiplicity[2]={0,0};
static void glm_flush(int stream){
  if(multiplicity[stream])glm_reads[pending[stream]]+=multiplicity[stream];
  multiplicity[stream]=0;
}
static void glm_record(ReadKey key){
  int stream=key[4];auto &last=pending[stream];auto &count=multiplicity[stream];
  if(count && last==key){++count;return;}
  if(count==1 && last[0]==key[0] && last[1]==key[1]
     && last[2]+last[3]==key[2]){last[3]+=key[3];return;}
  glm_flush(stream);last=key;count=1;
}
extern "C" void glm_reads_begin(){glm_reads.clear();multiplicity[0]=multiplicity[1]=0;glm_observe=true;}
extern "C" void glm_reads_end(){glm_flush(0);glm_flush(1);glm_observe=false;}
extern "C" uint64_t glm_reads_size(){return glm_reads.size();}
extern "C" uint64_t glm_reads_copy(uint64_t* out){
  uint64_t n=0;
  for(auto &entry:glm_reads){
    const auto &k=entry.first;
    uint64_t* prev=n?out+6*(n-1):nullptr;
    if(prev && prev[0]==k[0] && prev[1]==k[1] && prev[4]==k[4]
       && prev[5]==entry.second && prev[2]+prev[3]==k[2])prev[3]+=k[3];
    else {uint64_t* dst=out+6*n++;for(auto v:k)*dst++=v;*dst=entry.second;}
  }
  return n;
}
"""
    return prefix + graph_cpu.SHIM.replace(
        '#include "examples/glm53/tiles.hip"', f'#include "{entry}"')


def observed_step(cpu, spec, pos, tokens, sink):
    """Execute decode or verify and deliver actual reader ranges to a sink.

    Works with a decode GraphCPU compiled using instrumented_shim as well.
    Events carry the invocation's starting position, rather than claiming a
    single position for reads shared across the verify block.
    """
    lib = cpu.lib
    lib.glm_reads_size.restype = ctypes.c_uint64
    lib.glm_reads_copy.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.glm_reads_begin()
    try:
        result = cpu.step(pos, tokens)
    finally:
        lib.glm_reads_end()
    n = lib.glm_reads_size()
    events = (ctypes.c_uint64 * (6*n))()
    lib.glm_reads_copy.restype = ctypes.c_uint64
    used = lib.glm_reads_copy(events)
    lookup = {(cpu.buffers[group+'_weights'].ctypes.data, rec['offset']): (group, key)
              for group, blob in spec['weights'].items()
              for key, rec in blob['tensors'].items()}
    for i in range(0, 6*used, 6):
        blob, descriptor, start, size, scale, repeats = events[i:i+6]
        group, tensor = lookup[blob, descriptor]
        sink.read_range(group, tensor, start=start, size=size, step=pos,
                        scale=bool(scale), repeats=repeats)
    return result


class VerifyGraphCPU:
    """Run the verify graph's actual C++ tiles using graph_cpu's CPU shim.

    step(pos, tokens) accepts shape (k+1, batch), returns tokens and full logits
    in position-major order. Buffers (including shared caches) persist across
    calls. The input block is validated before any state is changed.
    """
    def __init__(self, stage_dir, k, batch, gate, work, options=None, poison=False, observe=False):
        import numpy as np
        from examples.glm53 import stage
        self.np, self.batch, self.count = np, batch, (1 if observe and k == 0 else _positions(k))
        options = Options() if options is None else options
        stage_dir, work = Path(stage_dir), Path(work)
        spec = stage.validate(stage_dir, allow_fixture=True, gate=gate)
        self.d = spec['dimensions']
        self.graph = (model.build_from(spec, batch, options) if observe and k == 0
                      else build_from(spec, k, batch, options))
        work.mkdir(parents=True, exist_ok=True)
        (work / 'glm53_options.h').write_text(options.header())
        # Compile a separate TU, leaving graph_cpu and the decode tiles intact.
        shim = graph_cpu.SHIM.replace('#include "examples/glm53/tiles.hip"',
                                      f'#include "{SOURCE}"')
        if observe:
            shim = instrumented_shim(work)
        (work / 'verify_cpu.cpp').write_text(shim)
        lib = work / 'verify_cpu.so'
        subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
                        '-I', str(graph_cpu.ROOT), '-I', str(graph_cpu.ROOT / 'etx/runtime/include'),
                        '-I', str(stage_dir), '-I', str(work), '-include', 'glm53_options.h',
                        str(work / 'verify_cpu.cpp'), '-o', str(lib)], check=True, timeout=600)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.glm_run_grid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.glm_run_grid.restype = None
        placed = graph_cpu.weights(stage_dir, spec, None)
        self.buffers = {}
        for name, tensor in self.graph.tensors.items():
            size = tensor.nbytes({'B': batch})
            buf = graph_cpu._aligned(max(size, 1), np)
            if name in placed:
                if placed[name].size != size:
                    raise ValueError(f'{name}: weight size differs from graph')
                buf[:] = placed[name]
            elif poison and tensor.role == 'scratch':
                buf.view(np.float32)[:] = np.nan
            self.buffers[name] = buf
        self.lds_floats = model.lds_floats(self.d)
        self.lds = np.zeros(self.lds_floats, dtype=np.float32)
        self.program = []
        for g in self.graph.grids:
            args = (ctypes.c_void_p * len(g.args))(*[self.buffers[a].ctypes.data for a in g.args])
            cst = (ctypes.c_int * 4)(*(tuple(g.consts) + (0, 0, 0, 0))[:4])
            fn = ctypes.cast(getattr(self.lib, g.body.symbol), ctypes.c_void_p)
            self.program.append((g.name, fn, g.grid[1] if len(g.grid) > 1 else 1, args, cst))

    def step(self, pos, tokens):
        np = self.np
        tokens = np.asarray(tokens)
        if (tokens.shape != (self.count, self.batch) or tokens.dtype.kind not in 'iu'
                or np.any(tokens < 0) or np.any(tokens >= self.d['VOCAB'])):
            raise ValueError('tokens must be integer IDs shaped (k+1, batch) within vocabulary')
        if (isinstance(pos, bool) or not isinstance(pos, int) or pos < 0
                or pos + self.count > self.d['MAXLEN']):
            raise ValueError('position block exceeds cache capacity')
        for p in range(self.count):
            step = self.buffers['step' if self.count == 1 else f'p{p}_step'].view(np.int32).reshape(self.batch, 2)
            step[:, 0], step[:, 1] = pos + p, tokens[p]
        for _, fn, tiles, args, cst in self.program:
            self.lib.glm_run_grid(fn, self.batch, tiles, args, cst,
                                 self.lds.ctypes.data, self.lds_floats)
        returned, logits = [], []
        for p in range(self.count):
            returned.append(self.buffers['returned' if self.count == 1 else f'p{p}_returned'].view(np.int32).copy())
            head = self.buffers['head_scratch' if self.count == 1 else f'p{p}_head_scratch'].view(np.float32)
            logits.append(head.reshape(self.batch, -1)[:, 2*self.d['H']:].copy())
        return np.stack(returned), np.stack(logits)
