"""Actual text-only GLM model in ETX: embedding, all hybrid/MoE layers, head.

Contiguous PP2/PP4/PP8 or FFN-only TP / routed-expert EP, native FP8 storage.
TP/EP can retain serial attention or enable head-sharded KDA/MLA with attention_tp.
ETX owns execution, cache ordering and
SYSTEM peer publication. No HF calls. Default (options.phased=1): every layer
is a chain of phase grids whose tiles cover all CUs (examples/glm53/phases.h).
phased=0 keeps the original one-workgroup-per-layer correctness tiles.
"""
import json
import os
from pathlib import Path
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from examples.glm53.stage import validate
from examples.glm53.options import Options, load as load_options

TILES='examples/glm53/tiles.hip'


def build_from(stage, batch=1, options=None, layout=None):
    options = Options() if options is None else options
    if batch not in (1,2,4,8):raise ValueError('batch must be 1,2,4,8')
    if options.parallel:
        if layout is None:
            from examples.glm53.parallel_layout import derive
            layout=derive(stage,'tp' if options.parallel==1 else 'ep',options.devices,attention_tp=bool(options.attention_tp))
        expected_mode='tp' if options.parallel==1 else 'ep'
        if (layout['mode']!=expected_mode or layout['devices']!=options.devices
                or bool(layout.get('attention_tp',False))!=bool(options.attention_tp)):
            raise ValueError('parallel layout differs from requested build options')
        return build_phased(stage,batch,options,layout)
    if options.phased:return build_phased(stage,batch,options)
    d=stage['dimensions'];t=stage['config'];g=Graph('glm53_flash_pp2')
    g.symbols=['B'];a_bytes=2 if d['BF16'] else 4
    g.tensor('step',('B',2),dtype='int32',bytes_per_elem=4)
    g.tensor('tokens',('B',),dtype='int32',bytes_per_elem=4)
    g.tensor('returned',('B',),dtype='int32',bytes_per_elem=4)
    for name,w in stage['weights'].items():
        g.tensor(name+'_weights',(w['blob_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
        g.tensor(name+'_meta',(w['meta_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
    g.tensor('head_scratch',('B',2*d['H']+d['VOCAB']),role='scratch',dtype='float32',bytes_per_elem=4)
    # Native static LDS staging is additional to scheduler LDS, not ctx.lds.
    resource=Resource(threads=options.threads,vgpr=options.vgpr_budget)
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


TILE_CAP=512          # measured phase-grid default; up can use all 608 resident workers


def tiles(units,per,cap=TILE_CAP):
    return max(1,min(cap,-(-units//per)))


def r4(n):
    return (n+3)//4*4


def strides(d):
    """Per-request float strides of the phase tensors; phases.h GLM_S_* mirrors these."""
    hcn=2*d['HC']+d['HC']**2
    return dict(hc=r4(hcn),p1=r4(max(3*d['QDIM']+2*d['LD']+d['LH'],d['QR']+d['KVR']+2*d['ID']+d['IH'])),
                p2=r4(d['NH']*d['AD']+d['IH']*d['ID']),att=r4(max(d['QDIM'],d['NH']*d['AD'])),mid=d['HC']*d['H'],
                rt=r4(d['EXPERTS'])+d['H'],act=(d['TOPK']+1)*d['INTERMAX'],sel=r4(2*d['TOPK']))


def lds_floats(d):
    """phases.h GLM_LDS_FLOATS."""
    return 48+max(d['H']+2*r4(d['TOPK'])+2*r4(d['EXPERTS']),r4(d['QR'])+r4(d['KVR']),r4(3*d['MAXLEN']+d['ID'])+5*r4(d['AD']),10*r4(d['LD']))


def phase_tiles(d,t,linear,moe,up_pairs=36):
    """Tile count of every phase grid of one layer, from its rows (payload rows, heads or columns)."""
    hcn=2*d['HC']+d['HC']**2
    mi=t['moe_intermediate_size'] if moe else t['intermediate_size']
    n=dict(hc=hcn,hcf=hcn,o=tiles(d['H'],8),rt=tiles(d['EXPERTS'],16) if moe else 1,
           up=tiles((d['TOPK']+1 if moe else 1)*mi,up_pairs,cap=608),down=tiles(d['H'],8))
    if linear:n.update(p1=tiles(3*d['QDIM']+2*d['LD']+d['LH'],48),att=d['LH'])
    else:n.update(p1=tiles(d['QR']+d['KVR']+2*d['ID']+d['IH'],32),p2=tiles(d['NH']*d['AD']*3+d['IH']*d['ID'],128),att=d['NH'])
    return n


def owners(stage,devices=2):
    """Pipeline owner of every layer and of embedding/norm/head. devices=2 keeps the staged split;
    4 and 8 cut the layers into contiguous blocks whose sizes differ by at most one."""
    n=stage['config']['num_hidden_layers']
    if devices==2:own={f'l{i}':r['rank'] for r in stage['ranks'] for i in r['layers']}
    else:own={f'l{i}':i*devices//n for i in range(n)}
    own.update(embedding=0,norm=devices-1,head=devices-1)
    return own


def build_phased(stage,batch,options,layout=None):
    d=stage['dimensions'];t=stage['config']
    attention='head_tp' if options.attention_tp else 'serial_attention'
    name=f'glm53_phased_pp{options.devices}' if layout is None else f'glm53_{layout["mode"]}{options.devices}_{attention}'
    g=Graph(name)
    g.symbols=['B'];a_bytes=2 if d['BF16'] else 4;st=strides(d)
    g.tensor('step',('B',2),dtype='int32',bytes_per_elem=4)
    g.tensor('tokens',('B',),dtype='int32',bytes_per_elem=4)
    g.tensor('returned',('B',),dtype='int32',bytes_per_elem=4)
    for name,w in stage['weights'].items():
        if layout is not None and name.startswith('l') and name[1:].isdigit():continue
        g.tensor(name+'_weights',(w['blob_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
        g.tensor(name+'_meta',(w['meta_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
    if layout is not None:
        for name,w in layout['weights'].items():
            g.tensor(name+'_weights',(w['blob_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
            g.tensor(name+'_meta',(w['meta_bytes'],),role='weight',dtype='uint8',bytes_per_elem=1)
    g.tensor('head_scratch',('B',2*d['H']+d['VOCAB']),role='scratch',dtype='float32',bytes_per_elem=4)
    resource=Resource(threads=options.threads,vgpr=128,lds_bytes=4*lds_floats(d))
    small=Resource(threads=options.threads,vgpr=options.vgpr_budget)
    produced={}          # tensor -> event of the grid that wrote it this step

    def grid(name,device,symbol,n,args,reads,writes,consts=(0,0,0,0)):
        ev='e_'+name;g.etensor(ev,('B',),wait_count=n)
        waits={produced[x]:'bt->b' for x in reads if x in produced}
        g.call_device(name,('B',n),hip_link(TILES,symbol),device=device,resource=resource,args=args,reads=reads,
                      writes=writes,in_edges=waits,out_edges={ev:'bt->b'},consts=(n,)+tuple(consts[1:]))
        for x in writes:produced[x]=ev

    previous='embedded'
    g.tensor(previous,('B',d['HC'],d['H']),bytes_per_elem=a_bytes)
    g.etensor('e_embed',('B',),wait_count=1)
    g.call_device('embedding',('B',),hip_link(TILES,'glm53_embed'),device=0,resource=small,
                  args=['step','embedding_weights','embedding_meta',previous],reads=['step','embedding_weights','embedding_meta'],
                  writes=[previous],out_edges={'e_embed':'i->i'})
    produced[previous]='e_embed'
    own=owners(stage,options.devices) if layout is None else {name:0 for name in stage['weights']}
    last=options.devices-1 if layout is None else 0
    for i in range(t['num_hidden_layers']):
        dev=own[f'l{i}'];linear=t['layer_types'][i]=='linear_attention';moe=t['mlp_layer_types'][i]=='sparse'
        nstate=d['STATE_KDA'] if linear else d['STATE_MLA']
        if nstate>2**31-1:raise ValueError('state exceeds the native int32 per-request stride ABI')
        L=f'l{i}';prefix=L if layout is None else L+'_common'
        W,M=prefix+'_weights',prefix+'_meta';n=phase_tiles(d,t,linear,moe,options.up_pairs);cst=(0,int(linear),int(moe),nstate)
        for name,size in [('hca',st['hc']),('p1',st['p1']),('att',st['att']),('mid',st['mid']),('hcf',st['hc']),
                          ('rt',st['rt']),('act',st['act']),('sel',st['sel'])]+([] if linear else [('p2',st['p2'])]):
            g.tensor(f'{L}_{name}',('B',size),role='scratch',dtype='float32',bytes_per_elem=4)
        if not options.attention_tp:g.tensor(L+'_state',('B',nstate),role='runtime',dtype='float32',bytes_per_elem=4)
        g.tensor(L+'_out',('B',d['HC'],d['H']),bytes_per_elem=a_bytes)
        x=lambda name:f'{L}_{name}'
        grid(x('hc'),dev,'glm53_hc',n['hc'],[W,M,previous,x('hca')],[W,M,previous],[x('hca')])
        if options.attention_tp:
            from examples.glm53.attention_graph import add_attention
            add_attention(g,grid,L,d,t,options,n,W,M,previous,linear,nstate)
        else:
            grid(x('p1'),dev,'glm53_p1',n['p1'],[W,M,previous,x('hca'),x('p1')],[W,M,previous,x('hca')],[x('p1')],cst)
            if linear:
                grid(x('kda'),dev,'glm53_kda',n['att'],[W,M,x('p1'),x('state'),x('att')],[W,M,x('p1'),x('state')],[x('state'),x('att')],cst)
            else:
                grid(x('p2'),dev,'glm53_p2',n['p2'],['step',W,M,x('p1'),x('p2'),x('state')],['step',W,M,x('p1'),x('state')],[x('p2'),x('state')],cst)
                grid(x('mla'),dev,'glm53_mla',n['att'],['step',W,M,x('p1'),x('p2'),x('state'),x('att')],
                     ['step',W,M,x('p1'),x('p2'),x('state')],[x('att')],cst)
            grid(x('o'),dev,'glm53_o',n['o'],[W,M,previous,x('hca'),x('att'),x('mid')],[W,M,previous,x('hca'),x('att')],[x('mid')])
        grid(x('hcf'),dev,'glm53_hcf',n['hcf'],[W,M,x('mid'),x('hcf')],[W,M,x('mid')],[x('hcf')])
        grid(x('rt'),dev,'glm53_rt',n['rt'],[W,M,x('mid'),x('hcf'),x('rt')],[W,M,x('mid'),x('hcf')],[x('rt')],cst)
        if layout is not None and (options.parallel==1 or moe):
            from examples.glm53.parallel_graph import add_ffn
            add_ffn(g,grid,L,d,t,options,n,W,M,moe)
        else:
            grid(x('up'),dev,'glm53_up',n['up'],[W,M,x('rt'),x('act'),x('sel')],[W,M,x('rt')],[x('act'),x('sel')],cst)
            grid(x('down'),dev,'glm53_down',n['down'],[W,M,x('act'),x('sel'),x('mid'),x('hcf'),x('out')],
                 [W,M,x('act'),x('sel'),x('mid'),x('hcf')],[x('out')],cst)
        previous=x('out')
    head=tiles(d['VOCAB'],256)
    args=[previous,'norm_weights','norm_meta','head_weights','head_meta','head_scratch']
    grid('head',last,'glm53_lm',head,args,args[:5],['head_scratch'])
    g.etensor('e_argmax',('B',),wait_count=1)
    g.call_device('argmax',('B',),hip_link(TILES,'glm53_argmax'),device=last,resource=small,args=['head_scratch','tokens'],
                  reads=['head_scratch'],writes=['tokens'],in_edges={produced['head_scratch']:'i->i'},out_edges={'e_argmax':'i->i'})
    g.call_device('feedback',('B',),hip_link(TILES,'glm53_feedback'),device=0,resource=small,
                  args=['tokens','returned'],reads=['tokens'],writes=['returned'],in_edges={'e_argmax':'i->i'})
    return g


def build():
    path=Path(os.environ['ETX_GLM_STAGE'])
    return build_from(validate(path,allow_fixture=os.environ.get('ETX_GLM_FIXTURE')=='1',gate=os.environ.get('ETX_GLM_GATE','production')), options=load_options(os.environ.get('ETX_GLM_OPTIONS')))

def bindings(**overrides):
    b=dict(B=int(os.environ.get('ETX_GLM_BATCH','1')));b.update(overrides)
    if b['B'] not in (1,2,4,8):raise ValueError('unsupported batch')
    return b

def runtime(bindings,rng=None):return {}
