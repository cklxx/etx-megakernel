"""Real FFN TP / routed-expert EP graph fragments; attention remains on rank 0.

Every rank has local weight shards and private activation/output tensors. Explicit
packet copies and SYSTEM event edges carry the input/selection and returned dots;
no worker streams a peer's expert weights. This is not full attention TP.
"""
from examples.glm53.model import tiles


def add_ffn(g, grid, layer, dimensions, config, options, counts, weights, meta, moe):
    d=dimensions;L=layer;x=lambda suffix:f'{L}_{suffix}'
    world=options.devices;tp=options.parallel==1
    grid(x('route'),0,'glm53_parallel_route',1,[weights,meta,x('rt'),x('sel')],
         [weights,meta,x('rt')],[x('sel')],(0,int(moe),0,0))
    outputs=[]
    for rank in range(world):
        prefix=f'{L}_r{rank}';W=prefix+'_weights';M=prefix+'_meta'
        packet=prefix+'_packet';act=prefix+'_act';out=prefix+'_dots'
        rt=((d['EXPERTS']+3)//4*4)+d['H'];sel=((2*d['TOPK']+3)//4*4)
        g.tensor(packet,('B',rt+sel),role='scratch',dtype='float32',bytes_per_elem=4)
        g.tensor(act,('B',(d['TOPK']+1)*d['INTERMAX']),role='scratch',dtype='float32',bytes_per_elem=4)
        g.tensor(out,('B',(d['TOPK']+1)*d['H']),role='scratch',dtype='float32',bytes_per_elem=4)
        grid(prefix+'_dispatch',rank,'glm53_parallel_dispatch',1,[x('rt'),x('sel'),packet],
             [x('rt'),x('sel')],[packet])
        mi=config['moe_intermediate_size'] if moe else config['intermediate_size']
        local_mi=mi//world
        up=tiles((d['TOPK']+1 if moe else 1)*local_mi,options.up_pairs,cap=608)
        cst=(0,int(moe),rank,world)
        grid(prefix+'_up',rank,'glm53_parallel_up',up,[W,M,packet,act],
             [W,M,packet],[act],cst)
        grid(prefix+'_down',rank,'glm53_parallel_down',counts['down'],[W,M,packet,act,out],
             [W,M,packet,act],[out],cst)
        outputs.append(out)
    args=[weights,meta,x('sel'),x('mid'),x('hcf'),x('out')]+outputs
    grid(x('combine'),0,'glm53_parallel_combine',counts['down'],args,
         args[:5]+outputs,[x('out')],(0,int(moe),world,0))
