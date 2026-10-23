"""Causal k+1-position GLM verification graph, with weights-free CPU execution.

Inputs are the last accepted token followed by k draft tokens (teacher forced).
Every position returns full logits and an argmax. KDA heads scan the block;
DSA runs its existing per-position causal index top-k; MoE up rows are grouped
by expert across positions. State advances through the entire block: callers
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

    This first verify path supports phased pipeline placement; TP/EP layouts
    need their own multi-position dispatch and are explicitly refused.
    """
    count = _positions(k)
    options = Options() if options is None else options
    if not options.phased or options.parallel or options.attention_tp:
        raise ValueError('verify requires phased serial attention and nonparallel FFN')
    base = model.build_from(stage, batch, options)
    graph = Graph(f'glm53_verify_k{k}_pp{options.devices}')
    graph.symbols = list(base.symbols)

    def local(name, p):
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

    for p in range(count):
        copy('embedding', p)
    for i, kind in enumerate(stage['config']['layer_types']):
        layer = f'l{i}'
        linear = kind == 'linear_attention'
        for phase in ('hc', 'p1'):
            for p in range(count):
                copy(f'{layer}_{phase}', p)
        if linear:
            src = base.grid(layer + '_kda')
            args = [layer + '_weights', layer + '_meta', layer + '_state']
            inputs, outputs = [], []
            for p in range(count):
                inputs.append(local(layer + '_p1', p))
                outputs.append(local(layer + '_att', p))
                args.extend([inputs[-1], outputs[-1]])
            emit(layer + '_kda_scan', hip_link(SOURCE, 'glm53_verify_kda_scan'),
                 src.device, src.grid[1], args, args[:3] + inputs,
                 [layer + '_state'] + outputs,
                 (src.grid[1], 1, count, src.consts[3]))
        else:
            # A single writer owns the KV cache. The current IR requires one
            # writer grid per tensor, so serialize DSA positions within a
            # request; requests still execute independently.
            src = base.grid(layer + '_mla')
            args = [layer + '_weights', layer + '_meta', layer + '_state']
            reads, writes = args.copy(), [layer + '_state']
            for p in range(count):
                step, p1, p2, att = [local(n, p) for n in
                    ('step', layer + '_p1', layer + '_p2', layer + '_att')]
                args.extend([step, p1, p2, att])
                reads.extend([step, p1])
                writes.extend([p2, att])
            emit(layer + '_dsa_scan', hip_link(SOURCE, 'glm53_verify_dsa_scan'),
                 src.device, 1, args, reads, writes, (1, 0, count, src.consts[3]))
        for phase in ('o', 'hcf', 'rt'):
            for p in range(count):
                copy(f'{layer}_{phase}', p)
        moe = stage['config']['mlp_layer_types'][i] == 'sparse'
        if moe:
            src = base.grid(layer + '_up')
            for p in range(count):
                args = [layer + '_weights', layer + '_meta',
                        local(layer + '_rt', p), local(layer + '_sel', p)]
                emit(f'p{p}_{layer}_route', hip_link(SOURCE, 'glm53_parallel_route'),
                     src.device, 1, args, args[:3], args[3:], (1, 1, 0, 0))
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
        else:
            for p in range(count):
                copy(layer + '_up', p)
        for p in range(count):
            copy(layer + '_down', p)
    for phase in ('head', 'argmax', 'feedback'):
        for p in range(count):
            copy(phase, p)
    return graph


class VerifyGraphCPU:
    """Run the verify graph's actual C++ tiles using graph_cpu's CPU shim.

    step(pos, tokens) accepts shape (k+1, batch), returns tokens and full logits
    in position-major order. Buffers (including shared caches) persist across
    calls. The input block is validated before any state is changed.
    """
    def __init__(self, stage_dir, k, batch, gate, work, options=None, poison=False):
        import numpy as np
        from examples.glm53 import stage
        self.np, self.batch, self.count = np, batch, _positions(k)
        options = Options() if options is None else options
        stage_dir, work = Path(stage_dir), Path(work)
        spec = stage.validate(stage_dir, allow_fixture=True, gate=gate)
        self.d = spec['dimensions']
        self.graph = build_from(spec, k, batch, options)
        work.mkdir(parents=True, exist_ok=True)
        (work / 'glm53_options.h').write_text(options.header())
        # Compile a separate TU, leaving graph_cpu and the decode tiles intact.
        shim = graph_cpu.SHIM.replace('#include "examples/glm53/tiles.hip"',
                                      f'#include "{SOURCE}"')
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
            self.program.append((g.name, fn, g.grid[1], args, cst))

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
            step = self.buffers[f'p{p}_step'].view(np.int32).reshape(self.batch, 2)
            step[:, 0], step[:, 1] = pos + p, tokens[p]
        for _, fn, tiles, args, cst in self.program:
            self.lib.glm_run_grid(fn, self.batch, tiles, args, cst,
                                 self.lds.ctypes.data, self.lds_floats)
        returned, logits = [], []
        for p in range(self.count):
            returned.append(self.buffers[f'p{p}_returned'].view(np.int32).copy())
            head = self.buffers[f'p{p}_head_scratch'].view(np.float32)
            logits.append(head.reshape(self.batch, -1)[:, 2*self.d['H']:].copy())
        return np.stack(returned), np.stack(logits)
