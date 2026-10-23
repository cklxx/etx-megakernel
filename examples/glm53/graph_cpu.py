"""The planned GLM graph executed on the CPU: a weights-free HF check for every build option set.

Builds the same graph a GPU build compiles (examples.glm53.model.build_from with the given Options and,
for TP/EP, the derived parallel layout), compiles examples/glm53/tiles.hip for the host (GLM53_CPU, one
lane per workgroup), and runs every grid's tiles serially in program order (= topological order) for
each decode step, exactly like the launcher's step loop: step = (pos, token), argmax, feedback.

Weights are placed as the launcher places them: stage blobs and metas for the outer/serial tensors,
and for TP/EP every rank-local blob is packed from the stage through the layout's ranges.tsv records,
with its meta from the layout's descriptors. All buffers start zero (as hipMemset in the launcher);
--poison fills scratch with NaN instead, so a phase reading anything its producers did not write
fails. Every rank's tensors are separate buffers: data crosses ranks only through the graph's own
dispatch/combine grids, as on the GPUs.

Judged like examples.glm53.module_gate.predict: TOKENS-EQUAL against the independent HF reference,
then the module rule on the 32 per-step full-logit errors. This is CPU execution of the device code
on a synthetic checkpoint: it proves the arm's graph, layout and kernels compute the model; it says
nothing about GPU memory ordering, residency, capacity or speed.

  python -m examples.glm53.graph_cpu --gate DIR --options '{"devices":8,"parallel":2}' --out OUT.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SHIM = r'''#define GLM53_CPU 1
#define __device__
#define __forceinline__ inline
#define __hip_atomic_load(p, order, scope) (*(p))
#include "etx/abi.h"
#include "examples/glm53/tiles.hip"
typedef void (*glm_tile)(const etx_ctx*);
// One grid: every (request, tile) coordinate in order, LDS poisoned before each tile.
extern "C" void glm_run_grid(glm_tile f,int batch,int tiles,void* const* args,const int* cst,float* lds,int lds_floats){
  etx_ctx c={};c.args=args;c.lds=lds;for(int k=0;k<4;++k)c.cst[k]=cst[k];
  for(int r=0;r<batch;++r)for(int t=0;t<tiles;++t){
    for(int i=0;i<lds_floats;++i)lds[i]=NAN;
    c.coord[0]=r;c.coord[1]=t;f(&c);
  }
}
'''


def compile_tiles(stage_dir, options, work):
    """tiles.hip for the host with the build's options header and the stage's glm53_spec.h."""
    work.mkdir(parents=True, exist_ok=True)
    (work / 'glm53_options.h').write_text(options.header())
    (work / 'graph_cpu.cpp').write_text(SHIM)
    lib = work / 'graph_cpu.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC', '-I', str(ROOT),
                    '-I', str(ROOT / 'etx/runtime/include'), '-I', str(stage_dir), '-I', str(work),
                    '-include', 'glm53_options.h', str(work / 'graph_cpu.cpp'), '-o', str(lib)],
                   check=True, timeout=600)
    return ctypes.CDLL(str(lib))


def _aligned(nbytes, np):
    raw = np.zeros(nbytes + 256, dtype=np.uint8)
    offset = (-raw.ctypes.data) % 256
    return raw[offset:offset + nbytes]


def weights(stage_dir, spec, layout):
    """name -> bytes for every *_weights/*_meta tensor, placed as the launcher places them."""
    import numpy as np
    from examples.glm53.parallel_layout import _ABI
    out = {}
    for name, w in spec['weights'].items():
        if layout is not None and name.startswith('l') and name[1:].isdigit():
            continue
        out[name + '_weights'] = np.fromfile(stage_dir / w['blob'], dtype=np.uint8)
        out[name + '_meta'] = np.fromfile(stage_dir / w['meta'], dtype=np.uint8)
    if layout is not None:
        sources = {}
        for name, w in layout['weights'].items():
            out[name + '_weights'] = np.zeros(w['blob_bytes'], dtype=np.uint8)
            out[name + '_meta'] = np.frombuffer(b''.join(_ABI.pack(*d) for d in w['descriptors']), dtype=np.uint8)
        for r in layout['ranges']:
            src = sources.get(r['srcfile'])
            if src is None:
                src = sources[r['srcfile']] = np.fromfile(stage_dir / r['srcfile'], dtype=np.uint8)
            rows = np.lib.stride_tricks.as_strided(src[r['srcoff']:], shape=(r['rows'], r['width']),
                                                   strides=(r['srcstride'], 1))
            dst = out[r['tensor']]
            for i in range(r['rows']):
                dst[r['destoff'] + i * r['dststride']:r['destoff'] + i * r['dststride'] + r['width']] = rows[i]
    return out


class GraphCPU:
    def __init__(self, stage_dir, options, batch, gate, work, poison=False):
        import numpy as np
        from examples.glm53 import model, stage
        self.np = np
        spec = stage.validate(stage_dir, allow_fixture=True, gate=gate)
        self.d = spec['dimensions']
        layout = None
        if options.parallel:
            from examples.glm53.parallel_layout import derive
            layout = derive(spec, 'tp' if options.parallel == 1 else 'ep', options.devices,
                            attention_tp=bool(options.attention_tp))
        self.graph = model.build_from(spec, batch, options, layout)
        self.batch = batch
        self.lib = compile_tiles(stage_dir, options, work)
        self.lib.glm_run_grid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.glm_run_grid.restype = None
        placed = weights(stage_dir, spec, layout)
        self.buffers = {}
        for name, t in self.graph.tensors.items():
            nbytes = t.nbytes({'B': batch})
            buf = _aligned(max(nbytes, 1), np)
            if name in placed:
                if placed[name].size != nbytes:
                    raise ValueError(f'{name}: placed {placed[name].size} bytes, graph declares {nbytes}')
                buf[:] = placed[name]
            elif poison and t.role == 'scratch':
                buf.view(np.float32)[:] = np.nan if nbytes % 4 == 0 else 0
            self.buffers[name] = buf
        self.lds_floats = model.lds_floats(self.d)
        self.lds = np.zeros(max(self.lds_floats, 1), dtype=np.float32)
        self.program = []
        for g in self.graph.grids:
            tiles = 1 if len(g.grid) == 1 else int(g.grid[1])
            consts = (tuple(g.consts) + (0, 0, 0, 0))[:4] if g.consts else (0, 0, 0, 0)
            args = (ctypes.c_void_p * max(len(g.args), 1))(*[self.buffers[a].ctypes.data for a in g.args])
            cst = (ctypes.c_int * 4)(*consts)
            fn = ctypes.cast(getattr(self.lib, g.body.symbol), ctypes.c_void_p)
            self.program.append((g.name, fn, tiles, args, cst))
        self.layout = layout

    def step(self, pos, tokens):
        np = self.np
        step = self.buffers['step'].view(np.int32)
        for r, token in enumerate(tokens):
            step[2 * r], step[2 * r + 1] = pos, token
        for _, fn, tiles, args, cst in self.program:
            self.lib.glm_run_grid(fn, self.batch, tiles, args, cst, self.lds.ctypes.data, self.lds_floats)
        d = self.d
        head = self.buffers['head_scratch'].view(np.float32).reshape(self.batch, 2 * d['H'] + d['VOCAB'])
        return self.buffers['returned'].view(np.int32).copy(), head[:, 2 * d['H']:].copy()


def check(gate_dir, options, gate, out, poison=False, work=None):
    """Run the 32-step decode for the gate's prompts and judge it against the HF reference."""
    import numpy as np
    from examples.glm53 import module_gate, oracle, reference
    from examples.glm53.run import compare_logits
    gate_dir = Path(gate_dir)
    ref_path = gate_dir / 'reference.json'
    ref = json.loads(ref_path.read_text())
    prompts = [r['prompt_ids'] for r in ref['requests']]
    work = Path(work or tempfile.mkdtemp(prefix='glm53-graph-cpu-'))
    model = GraphCPU(gate_dir / 'stage', options, len(prompts), gate, work, poison)
    context = len(prompts[0])
    returned = [0] * len(prompts)
    generated = [[] for _ in prompts]
    logits = []
    for pos in range(context + 31):
        got, head = model.step(pos, [row[pos] if pos < context else returned[r] for r, row in enumerate(prompts)])
        returned = [int(x) for x in got]
        if any(int(np.argmax(head[r])) != returned[r] for r in range(len(prompts))):
            raise ValueError(f'argmax grid differs from the head logits at pos {pos}')
        if pos >= context - 1:
            logits.append(head.astype('<f4'))
            for r, token in enumerate(returned):
                generated[r].append(token)
    backend = oracle.build_backend(options.to_dict())
    candidate = dict(reference.identity(gate), backend=backend, batch=len(prompts),
                     requests=[dict(prompt_ids=row, generated_ids=g) for row, g in zip(prompts, generated)])
    verdict = oracle.compare(ref, candidate, gate, backend)
    expected = np.fromfile(ref_path.with_suffix('.logits'), dtype='<f4').reshape(32, len(prompts), -1)
    verdict.update(compare_logits(expected, np.stack(logits)))
    if verdict['status'] == 'TOKENS-EQUAL':
        ok, median = module_gate.module_rule(verdict)
        verdict.update(module_rule=dict(median_logit_max_abs=median, bound=module_gate.MEDIAN_LOGIT_BOUND, ok=ok))
        if not ok:
            verdict['status'] = 'FAILED'
    verdict.update(options=options.to_dict(), grids=len(model.program), poison=poison,
                   execution='CPU, planned graph (examples/glm53/tiles.hip, one lane per workgroup)',
                   gpu_correctness='unmeasured')
    Path(out).write_text(json.dumps(verdict, indent=2) + '\n')
    return verdict


def main():
    from examples.glm53.options import Options
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--gate-dir', type=Path, required=True, help='module_gate prepare output (stage, reference)')
    ap.add_argument('--gate', default='module')
    ap.add_argument('--options', default='{}', help='Options fields as JSON')
    ap.add_argument('--poison', action='store_true', help='NaN-fill scratch instead of zero')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    options = Options(**json.loads(a.options))
    v = check(a.gate_dir, options, a.gate, a.out, a.poison)
    matched = v.get('matched_each')
    rule = v.get('module_rule', {})
    print(f"GLM-GRAPH-CPU {v['status']} backend={oracle_backend(options)} options={json.dumps(options.to_dict(), sort_keys=True)} "
          f"matched={matched} median_logit={rule.get('median_logit_max_abs')} max_logit={v.get('logit_max_abs')}", flush=True)
    return 0 if v['status'] == 'TOKENS-EQUAL' else 2


def oracle_backend(options):
    from examples.glm53 import oracle
    return oracle.build_backend(options.to_dict())


if __name__ == '__main__':
    raise SystemExit(main())
