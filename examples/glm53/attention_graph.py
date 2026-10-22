"""Head-sharded KDA/MLA with local weights/state and root ordered O reduction.

Global-index scratch/state strides simplify the existing head kernels; only the
rank's head interval is accessed. Low-rank MLA/indexer projections are replicated,
not split across ranks. This deliberately avoids a KV gather on every token.
"""
from examples.glm53.model import strides,tiles


def add_attention(g,grid,layer,dimensions,config,options,counts,weights,meta,previous,linear,nstate):
    d=dimensions;L=layer;s=strides(d);world=options.devices
    x=L+'_attention_input'
    g.tensor(x,('B',d['H']),role='scratch',dtype='float32',bytes_per_elem=4)
    grid(L+'_attention_input',0,'glm53_attention_input',1,
         [weights,meta,previous,L+'_hca',x],[weights,meta,previous,L+'_hca'],[x])
    heads=d['LH'] if linear else d['NH']
    if heads%world:raise ValueError('attention head count must divide devices')
    outputs=[]
    for rank in range(world):
        prefix=f'{L}_a{rank}';W=prefix+'_weights';M=prefix+'_meta'
        inp=prefix+'_input';p1=prefix+'_p1';p2=prefix+'_p2';att=prefix+'_att'
        state=prefix+'_state';partial=prefix+'_partial'
        for name,size,role in [(inp,d['H'],'scratch'),(p1,s['p1'],'scratch'),
                               (att,s['att'],'scratch'),(state,nstate,'runtime'),
                               (partial,d['H'],'scratch')]+([] if linear else [(p2,s['p2'],'scratch')]):
            g.tensor(name,('B',size),role=role,dtype='float32',bytes_per_elem=4)
        grid(prefix+'_dispatch',rank,'glm53_attention_dispatch',1,[x,inp],[x],[inp])
        cst=(0,int(linear),rank,world)
        # Replicated MLA low-rank and indexer rows retain their original grid;
        # KDA Q/K/V and beta rows are head-sharded, FA/GA remain replicated.
        local_p1=max(1,(counts['p1']+world-1)//world) if linear else counts['p1']
        grid(prefix+'_project',rank,'glm53_attention_p1',local_p1,
             [W,M,inp,p1],[W,M,inp],[p1],cst)
        if not linear:
            grid(prefix+'_expand',rank,'glm53_attention_p2',max(1,(counts['p2']+world-1)//world),
                 ['step',W,M,p1,p2,state],['step',W,M,p1,state],[p2,state],cst)
        args=['step',W,M,p1,p1 if linear else p2,state,att]
        grid(prefix+'_heads',rank,'glm53_attention_heads',heads//world,
             args,args[:-1],[att,state] if linear else [att],cst)
        grid(prefix+'_output',rank,'glm53_attention_o',counts['o'],
             [W,M,att,partial],[W,M,att],[partial],cst)
        outputs.append(partial)
    args=[weights,meta,previous,L+'_hca',L+'_mid']+outputs
    grid(L+'_attention_combine',0,'glm53_attention_combine',tiles(d['H'],8),
         args,args[:4]+outputs,[L+'_mid'],(0,world,0,0))
