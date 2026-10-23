"""Correctness-first transactional verification, using the actual native tiles.

Checkpoint j is the state before input j (after j consumed inputs). On a
rejection there, commit(j) retains exactly that prefix; the replacement token
must next be decoded at pos+j. Counts may differ by request. All-accepted
blocks commit count inputs. No GPU correctness or performance is claimed.
Full DSA snapshots restore rejected KV bytes as well as the visible prefix.
"""
import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess

from etx.frontends import hip_link
from examples.glm53 import graph_cpu
from examples.glm53.verify_graph import VerifyGraphCPU, build_from as verify_build

SOURCE = 'examples/glm53/verify_state.h'


def build_from(stage, k, batch=1, options=None):
    graph = verify_build(stage, k, batch, options)
    for grid in graph.grids:
        if grid.body.symbol not in ('glm53_verify_kda_scan', 'glm53_verify_dsa_scan'):
            continue
        state = grid.args[2]
        saved = state + '_checkpoints'
        tensor = graph.tensors[state]
        graph.tensors[saved] = replace(tensor, name=saved, role='scratch',
                                      shape=('B', k + 2, tensor.shape[1]))
        linear = grid.body.symbol == 'glm53_verify_kda_scan'
        grid.body = hip_link(SOURCE, 'glm53_checkpoint_kda' if linear else 'glm53_checkpoint_dsa')
        grid.args.append(saved)
        grid.writes.append(saved)
        grid.grid = ('B', 1)
        for event in grid.out_edges:
            graph.events[event].wait_count = 1
    return graph


class VerifyStateCPU(VerifyGraphCPU):
    """step opens a transaction; commit selects consumed inputs per request.

    A second step is refused until commit. Invalid input/counts do not mutate
    state. step(..., accepted=j) verifies and commits in one call. Checkpoints
    include the initial state, every verified position, and the full block.
    """
    def __init__(self, stage_dir, k, batch, gate, work, options=None, poison=False):
        from examples.glm53 import stage
        super().__init__(stage_dir, k, batch, gate, work, options, poison)
        spec = stage.validate(Path(stage_dir), allow_fixture=True, gate=gate)
        self.graph = build_from(spec, k, batch, options)
        work = Path(work)
        shim = graph_cpu.SHIM.replace('#include "examples/glm53/tiles.hip"',
                                     f'#include "{SOURCE}"')
        (work / 'state_cpu.cpp').write_text(shim)
        lib = work / 'state_cpu.so'
        subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
                        '-I', str(graph_cpu.ROOT), '-I', str(graph_cpu.ROOT / 'etx/runtime/include'),
                        '-I', str(stage_dir), '-I', str(work), '-include', 'glm53_options.h',
                        str(work / 'state_cpu.cpp'), '-o', str(lib)], check=True, timeout=600)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.glm_run_grid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.glm_run_grid.restype = None
        self.states = [n for n in self.buffers if n.endswith('_state')]
        for name in self.states:
            tensor = self.graph.tensors[name + '_checkpoints']
            self.buffers[tensor.name] = graph_cpu._aligned(tensor.nbytes({'B': batch}), self.np)
        self.program = []
        for g in self.graph.grids:
            args = (ctypes.c_void_p * len(g.args))(*[self.buffers[a].ctypes.data for a in g.args])
            cst = (ctypes.c_int * 4)(*(tuple(g.consts) + (0, 0, 0, 0))[:4])
            fn = ctypes.cast(getattr(self.lib, g.body.symbol), ctypes.c_void_p)
            self.program.append((g.name, fn, g.grid[1], args, cst))
        self.pending = False

    def _counts(self, accepted):
        np = self.np
        counts = np.asarray(accepted)
        if counts.ndim == 0:
            counts = np.repeat(counts, self.batch)
        if (counts.shape != (self.batch,) or counts.dtype.kind not in 'iu'
                or np.any(counts < 0) or np.any(counts > self.count)):
            raise ValueError('accepted must be integer prefix counts in [0, k+1] per request')
        return counts

    def step(self, pos, tokens, accepted=None):
        if self.pending:
            raise RuntimeError('commit the pending verification before another step')
        counts = None if accepted is None else self._counts(accepted)
        result = super().step(pos, tokens)
        self.pending = True
        if counts is not None:
            self.commit(counts)
        return result

    def commit(self, accepted):
        if not self.pending:
            raise RuntimeError('no pending verification')
        counts = self._counts(accepted)
        for name in self.states:
            dst = self.buffers[name].reshape(self.batch, -1)
            saved = self.buffers[name + '_checkpoints'].reshape(self.batch, self.count + 1, -1)
            for r, count in enumerate(counts):
                dst[r] = saved[r, count]
        self.pending = False
