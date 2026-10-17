"""Actual text-only GLM model in ETX: embedding, all hybrid/MoE layers, head.

Correctness-first complete-layer tiles, PP2 ownership, native FP8 storage.
ETX owns execution, cache ordering and SYSTEM peer publication. No HF calls.
"""
import json
import os
from pathlib import Path
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from examples.glm53.stage import validate

TILES='examples/glm53/tiles.hip'


def build_from(stage, batch=1):
    if batch not in (1,2,4,8):raise ValueError('batch must be 1,2,4,8')
    d=stage['dimensions'];t=stage['config'];g=Graph('glm53_flash_pp2')
    g.symbols=['B'];a_bytes=2 if d['BF16'] else 4
    g.tensor('step',('B',2),dtype='int32',bytes_per_elem=4)
    g.tensor('tokens',('B',),dtype='int32',bytes_per_elem=4)
    g.tensor('returned',('B',),dtype='int32',bytes_per_elem=4)
    for name,w in stage['weights'].items():
        g.tensor(name+'_weights',(w['blob_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
        g.tensor(name+'_meta',(w['meta_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
    g.tensor('head_scratch',('B',2*d['H']+d['VOCAB']),role='scratch',dtype='float32',bytes_per_elem=4)
    resource=Resource(threads=256,vgpr=64)
    previous='embedded';event='e_embed'
    g.tensor(previous,('B',d['HC'],d['H']),bytes_per_elem=a_bytes)
    g.etensor(event,('B',),wait_count=1)
    g.call_device('embedding',('B',),hip_link(TILES,'glm53_embed'),device=0,resource=resource,
                  args=['step','embedding_weights','embedding_meta',previous],reads=['step','embedding_weights','embedding_meta'],
                  writes=[previous],out_edges={event:'i->i'})
    owners={i:r['rank'] for r in stage['ranks'] for i in r['layers']}
    for i in range(t['num_hidden_layers']):
        rank=owners[i];linear=t['layer_types'][i]=='linear_attention';moe=t['mlp_layer_types'][i]=='sparse'
        nstate=d['STATE_KDA'] if linear else d['STATE_MLA']
        if nstate>2**31-1:raise ValueError('state exceeds the native int32 per-request stride ABI')
        name=f'l{i}';out=name+'_out';ev='e_'+name
        g.tensor(out,('B',d['HC'],d['H']),bytes_per_elem=a_bytes)
        g.tensor(name+'_scratch',('B',d['SCRATCH']),role='scratch',dtype='float32',bytes_per_elem=4)
        g.tensor(name+'_state',('B',nstate),role='runtime',dtype='float32',bytes_per_elem=4)
        g.etensor(ev,('B',),wait_count=1)
        args=['step',name+'_weights',name+'_meta',previous,out,name+'_state',name+'_scratch']
        g.call_device(name,('B',),hip_link(TILES,'glm53_layer'),device=rank,resource=resource,
                      args=args,reads=args[:4]+args[5:],writes=args[4:],
                      in_edges={event:'i->i'},out_edges={ev:'i->i'},consts=(int(linear),int(moe),nstate))
        previous,event=out,ev
    g.etensor('e_head',('B',),wait_count=1)
    args=[previous,'norm_weights','norm_meta','head_weights','head_meta','head_scratch','tokens']
    g.call_device('head',('B',),hip_link(TILES,'glm53_head'),device=1,resource=resource,args=args,
                  reads=args[:5],writes=args[5:],in_edges={event:'i->i'},out_edges={'e_head':'i->i'})
    g.call_device('feedback',('B',),hip_link(TILES,'glm53_feedback'),device=0,resource=resource,
                  args=['tokens','returned'],reads=['tokens'],writes=['returned'],in_edges={'e_head':'i->i'})
    return g


def build():
    path=Path(os.environ['ETX_GLM_STAGE'])
    return build_from(validate(path,allow_fixture=os.environ.get('ETX_GLM_FIXTURE')=='1'))

def bindings(**overrides):
    b=dict(B=int(os.environ.get('ETX_GLM_BATCH','1')));b.update(overrides)
    if b['B'] not in (1,2,4,8):raise ValueError('unsupported batch')
    return b

def runtime(bindings,rng=None):return {}
