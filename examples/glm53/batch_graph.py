"""Independent-request GLM decode with union-of-experts weight reads.

Generated tiles use the portable MoE primitives (including CUDA warp32).
CPU execution checks arithmetic and graph boundaries, not GPU speed or ordering.
The ledger counts dequantized weight elements, excluding scale-cache traffic.
"""
from pathlib import Path
import ctypes
import subprocess
import tempfile

from etx.ir import Graph
from etx.frontends import hip_link
from examples.glm53 import graph_cpu, model
from examples.glm53.options import Options


SOURCE = r'''
#include "examples/glm53/tiles.hip"
#include "examples/glm53/verify_moe.h"
#ifndef GLM_BATCH
#define GLM_BATCH 2
#endif
#ifdef GLM53_CPU
static unsigned long long reads=0;
extern "C" unsigned long long glm_batch_reads(){return reads;}
extern "C" void glm_batch_reset(){reads=0;}
struct BatchRead {
 float operator()(const uint8_t* b,const glm_weight& w,int r,int col) const {
   ++reads;return glm_w(b,w,r,col);
 }
};
#else
using BatchRead=glm_verify_moe_reader;
#endif
extern "C" __device__ void glm_batch_up(const etx_ctx* c){
 if(c->coord[0])return;
 const float *x[GLM_BATCH],*sel[GLM_BATCH];float* act[GLM_BATCH];
 for(int p=0;p<GLM_BATCH;++p){
   x[p]=(const float*)c->args[2]+p*GLM_S_RT+GLM_R4(GLM_EXPERTS);
   act[p]=(float*)c->args[3]+p*GLM_S_ACT;
   sel[p]=(const float*)c->args[4]+p*GLM_S_SEL;
 }
 glm_verify_moe_up<GLM_BATCH>(GLM_PB(0),GLM_PW(1),x,sel,act,
   c->coord[1]/c->cst[1],c->coord[1]%c->cst[1],c->cst[1],BatchRead{});
}
extern "C" __device__ void glm_batch_down(const etx_ctx* c){
 if(c->coord[0])return;
 const float *act[GLM_BATCH],*sel[GLM_BATCH];float* part[GLM_BATCH];
 for(int p=0;p<GLM_BATCH;++p){
   act[p]=(const float*)c->args[2]+p*GLM_S_ACT;
   sel[p]=(const float*)c->args[3]+p*GLM_S_SEL;
   part[p]=(float*)c->args[4]+p*(GLM_TOPK+1)*GLM_H;
 }
 glm_verify_moe_down<GLM_BATCH>(GLM_PB(0),GLM_PW(1),act,sel,part,
   c->coord[1]/c->cst[1],c->coord[1]%c->cst[1],c->cst[1],BatchRead{});
}
extern "C" __device__ void glm_batch_mix(const etx_ctx* c){
 const float* part=GLM_PR(2,const float*,(GLM_TOPK+1)*GLM_H);
 const float* sel=GLM_PR(3,const float*,GLM_S_SEL);
 const float* mid=GLM_PR(4,const float*,GLM_S_MID);
 const float* hcf=GLM_PR(5,const float*,GLM_S_HC);
 glm_act* out=GLM_PR(6,glm_act*,GLM_HC*GLM_H);
 float* lds=(float*)c->lds;float* hc=lds+16;
 glm_hc_coeffs(GLM_PB(0),GLM_PW(1),F_FN,hcf,hc,lds+GLM_LDS_HEAD);
 for(int j=GLM_LANE;j<GLM_H;j+=GLM_LANES){
   float y=glm_verify_moe_mix(part,sel,j);
   for(int k=0;k<GLM_HC;++k)out[k*GLM_H+j]=glm_put(glm_mix1(hc,[&](int i){return mid[i];},y,k,j));
 }
}
'''


def _batch(batch):
    if type(batch) is not int or batch not in (1, 2, 4):
        raise ValueError('batch must be 1, 2 or 4')


def build_from(stage, batch=2, options=None, work=None):
    """Build phased PP decode; global phase barriers protect cross-row reads.

    Source is materialized in work and must remain there until compilation.
    TP/EP and head-sharded attention are deliberately refused.
    """
    _batch(batch)
    options = Options() if options is None else options
    if not options.phased or options.parallel or options.attention_tp:
        raise ValueError('batch union requires phased PP with serial attention')
    work = Path(work or tempfile.mkdtemp(prefix='glm-batch-'))
    work.mkdir(parents=True, exist_ok=True)
    source = work.resolve() / 'batch_tiles.hip'
    source.write_text(f'#define GLM_BATCH {batch}\n' + SOURCE)
    base = model.build_from(stage, batch, options)
    g = Graph(f'glm53_batch_union_b{batch}')
    g.symbols = list(base.symbols)
    g.tensors.update(base.tensors)
    previous = None

    def emit(name, grid, body, original, args, reads, writes, consts):
        nonlocal previous
        event = 'batch_e_' + name
        count = batch * grid[1]
        g.etensor(event, (1,), wait_count=count)
        g.call_device(name, grid, body, device=original.device, resource=original.resource,
                      args=args, reads=reads, writes=writes, consts=consts,
                      in_edges={} if previous is None else {previous: 'bt->(0)'},
                      out_edges={event: 'bt->(0)'})
        previous = event

    sparse = {f'l{i}' for i, kind in enumerate(stage['config']['mlp_layer_types']) if kind == 'sparse'}
    for old in base.grids:
        layer = old.name.rsplit('_', 1)[0]
        args = list(old.args)
        if layer in sparse and old.name.endswith('_up'):
            W, M, rt, act, sel = args
            emit(layer+'_route', ('B', 1), hip_link(str(source), 'glm53_parallel_route'),
                 old, [W,M,rt,sel], [W,M,rt], [sel], (1,1,0,0))
            n = model.tiles(stage['config']['moe_intermediate_size'], options.effective_up_pairs)
            total = n*(stage['dimensions']['EXPERTS']+1)
            emit(old.name, ('B',total), hip_link(str(source),'glm_batch_up'), old,
                 args, [W,M,rt,sel], [act], (total,n,0,0))
        elif layer in sparse and old.name.endswith('_down'):
            W,M,act,sel,mid,hcf,out = args
            part = layer+'_batch_partial'
            g.tensor(part, ('B',(stage['dimensions']['TOPK']+1)*stage['dimensions']['H']),
                     role='scratch',dtype='float32',bytes_per_elem=4)
            total = stage['dimensions']['EXPERTS']+1
            emit(layer+'_expert_down', ('B',total), hip_link(str(source),'glm_batch_down'), old,
                 [W,M,act,sel,part], [W,M,act,sel], [part], (total,1,0,0))
            emit(old.name, ('B',1), hip_link(str(source),'glm_batch_mix'), old,
                 [W,M,part,sel,mid,hcf,out], [W,M,part,sel,mid,hcf], [out], (1,0,0,0))
        else:
            n = old.grid[1] if len(old.grid)==2 else 1
            emit(old.name, ('B',n), hip_link(str(source),old.body.symbol), old,
                 args, list(old.reads), list(old.writes), old.consts)
    return g


class BatchGraphCPU(graph_cpu.GraphCPU):
    """Execute exactly the generated batch tiles with poisoned scratch support."""
    def __init__(self, stage_dir, batch, gate, work, options=None, poison=True):
        import numpy as np
        from examples.glm53 import stage
        _batch(batch)
        options = Options() if options is None else options
        stage_dir, work = Path(stage_dir), Path(work)
        spec = stage.validate(stage_dir, allow_fixture=True, gate=gate)
        self.np, self.batch, self.d = np, batch, spec['dimensions']
        self.graph = build_from(spec,batch,options,work)
        source = work.resolve()/'batch_tiles.hip'
        (work/'glm53_options.h').write_text(options.header())
        shim = graph_cpu.SHIM.replace('#include "examples/glm53/tiles.hip"', f'#include "{source}"')
        (work/'cpu.cpp').write_text(shim)
        lib = work.resolve()/'cpu.so'
        subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',
                        '-I',str(graph_cpu.ROOT),'-I',str(graph_cpu.ROOT/'etx/runtime/include'),
                        '-I',str(stage_dir.resolve()),'-I',str(work.resolve()),'-include','glm53_options.h',
                        str(work/'cpu.cpp'),'-o',str(lib)],check=True,timeout=180)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.glm_run_grid.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
            ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        self.lib.glm_run_grid.restype = None
        self.lib.glm_batch_reset.restype = None
        self.lib.glm_batch_reads.restype = ctypes.c_ulonglong
        placed = graph_cpu.weights(stage_dir,spec,None)
        self.buffers = {}
        for name,tensor in self.graph.tensors.items():
            size = tensor.nbytes({'B':batch})
            buf = graph_cpu._aligned(max(1,size),np)
            if name in placed:
                buf[:] = placed[name]
            elif poison and tensor.role=='scratch':
                buf.view(np.float32)[:] = np.nan
            self.buffers[name] = buf
        self.lds_floats = model.lds_floats(self.d)
        self.lds = np.zeros(self.lds_floats,dtype=np.float32)
        self.program = []
        for grid in self.graph.grids:
            args = (ctypes.c_void_p*len(grid.args))(*[self.buffers[a].ctypes.data for a in grid.args])
            cst = (ctypes.c_int*4)(*(tuple(grid.consts)+(0,0,0,0))[:4])
            fn = ctypes.cast(getattr(self.lib,grid.body.symbol),ctypes.c_void_p)
            self.program.append((grid.name,fn,grid.grid[1],args,cst))

    def step(self,pos,tokens):
        np = self.np
        tokens = np.asarray(tokens)
        if tokens.shape!=(self.batch,) or tokens.dtype.kind not in 'iu' or np.any(tokens<0) or np.any(tokens>=self.d['VOCAB']):
            raise ValueError('tokens must be one valid integer ID per request')
        if type(pos) is not int or not 0<=pos<self.d['MAXLEN']:
            raise ValueError('position exceeds cache capacity')
        self.lib.glm_batch_reset()
        return super().step(pos,tokens)

    @property
    def weight_reads(self):
        return self.lib.glm_batch_reads()
