"""Execute the actual GLM HIP numerical body locally against independent HF.

Scaled fixtures exercise both ranks, every operator, kpool selection beyond its
budget, all expert routes, recurrent state across steps, and free greedy decode.
Fixtures are explicitly rejected by production staging/qualification.
"""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path
import subprocess

from examples.glm53 import stage
from examples.glm53.prepare import metadata, MODEL, EVIDENCE, sha


def tiny_config():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    return Glm5NextTextConfig(vocab_size=48,hidden_size=8,intermediate_size=12,moe_intermediate_size=6,
      num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2,n_routed_experts=4,
      num_experts_per_tok=2,n_shared_experts=1,q_lora_rank=4,kv_lora_rank=4,qk_nope_head_dim=4,
      qk_rope_head_dim=0,v_head_dim=4,linear_head_dim=4,linear_num_heads=2,linear_conv_kernel_dim=4,
      hc_mult=4,hc_sinkhorn_iters=20,index_head_dim=4,index_n_heads=2,index_topk=64,index_kpool=2,
      index_kpool_always_select_tail=True,pad_token_id=0,max_position_embeddings=128,
      layer_types=['linear_attention','indexed_attention','linear_attention','indexed_attention'],
      mlp_layer_types=['dense','sparse','sparse','sparse'],indexer_types=['full']*4,
      attention_dropout=0.,rms_norm_eps=1e-5,routed_scaling_factor=2.5,swiglu_limit=10.)


def fixture(out, bf16=False, fp8=False, selection=False, fp32_conv=True):
    import torch
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    torch.manual_seed(713)
    cfg=tiny_config();cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    if selection:cfg.index_topk=2
    hf=Glm5NextTextModel(cfg).eval()
    if selection:
        with torch.no_grad():
            for layer in hf.layers:
                if layer.block_type=='linear_attention':continue
                a=layer.self_attn;a.q_a_proj.weight.zero_();a.q_a_proj.weight[0].fill_(1)
                idx=a.indexer;idx.wq_b.weight.zero_();idx.wq_b.weight[0,0]=1;idx.wq_b.weight[5,0]=1
                idx.wk.weight.zero_();idx.wk.weight[0].fill_(1);idx.wk.weight[1,0]=1;idx.wk.weight[2,1]=1
                idx.weights_proj.weight.fill_(1)
    head=torch.nn.Linear(cfg.hidden_size,cfg.vocab_size,bias=False).eval()
    if bf16:
        hf=hf.bfloat16();head=head.bfloat16()
        if fp32_conv:
            for layer in hf.layers:
                if layer.block_type=='linear_attention':
                    layer.self_attn.conv1d.float()
                    layer.self_attn.forget_gate.dt_bias.data=layer.self_attn.forget_gate.dt_bias.data.float()
                    layer.self_attn.forget_gate.A_log.data=layer.self_attn.forget_gate.A_log.data.float()
    out.mkdir(parents=True,exist_ok=False)
    t=cfg.to_dict();d=stage.dimensions(t,64,bf16)
    _,api=metadata()
    r=dict(schema=1,model=MODEL,revision=api['sha'],config_sha256=sha(EVIDENCE/'glm-config.json'),fixture=True,
      config=t,dimensions=d,ranks=[dict(rank=0,layers=[0,1]),dict(rank=1,layers=[2,3])],weights={})
    def prepare_tensor(value):
        value=value.detach().cpu()
        if fp8 and value.ndim==2 and min(value.shape)>1:
            # Exercise FP8 ABI with one partial 128x128 block; mirror dequantized
            # values into HF before any reference output is computed.
            scale=torch.full((1,1),max(value.abs().max().item()/448,1e-8))
            q=(value.float()/scale.item()).to(torch.float8_e4m3fn)
            value.copy_((q.float()*scale.item()).to(value.dtype))
            return q,scale
        return value,None
    for i,layer in enumerate(hf.layers):
        def get(key,layer=layer):
            if key.startswith('mlp.experts.'):
                parts=key.split('.');e=int(parts[2]);op=parts[3]
                value=layer.mlp.experts.down_proj[e] if op=='down_proj' else layer.mlp.experts.gate_up_proj[e].chunk(2,dim=0)[op=='up_proj']
            else:value=layer.state_dict()[key]
            return prepare_tensor(value)
        r['weights'][f'l{i}']=stage.write_blob(out,f'l{i}',stage.shapes(t,i),get,stage.slot_names(t))
    for name,value in [('embedding',hf.embed_tokens.weight),('norm',hf.norm.weight),('head',head.weight)]:
        # Keep embedding and head BF16/F32; quantize only layer projections.
        r['weights'][name]=stage.write_blob(out,name,{name:tuple(value.shape)},lambda k,v=value:(v.detach(),None),[name])
    (out/'stage.json').write_text(json.dumps(r,indent=2)+'\n');(out/'glm53_spec.h').write_text(stage.header(d))
    stage.validate(out,allow_fixture=True)
    return r,hf,head


def compile_cpu(out):
    root=Path(__file__).resolve().parents[2]
    source=out/'cpu.cpp'
    source.write_text('''#define GLM53_CPU 1
#include "examples/glm53/native.h"
extern "C" void run_layer(const void* b,const void* w,const void* x,void* y,void* state,void* tmp,int pos,int lin,int moe){
 glm_layer((const uint8_t*)b,(const glm_weight*)w,(const glm_act*)x,(glm_act*)y,(float*)state,(float*)tmp,pos,lin,moe);
}
extern "C" void run_mla(const void* b,const void* w,const void* x,void* y,void* state,void* tmp,int pos){
 glm_mla((const uint8_t*)b,(const glm_weight*)w,(const float*)x,(float*)y,(float*)state,(float*)tmp,pos);
}
extern "C" void run_mlp(const void* b,const void* w,const void* x,void* y,void* tmp){
 glm_mlp((const uint8_t*)b,(const glm_weight*)w,MG,MU,MD,(const float*)x,(float*)y,(float*)tmp);
}
extern "C" void run_embed(const void* b,const void* w,int id,void* out){
 auto m=(const glm_weight*)w;auto y=(glm_act*)out;
 for(int j=0;j<GLM_HC*GLM_H;++j)y[j]=glm_put(glm_w((const uint8_t*)b,m[0],id,j%GLM_H));
}
extern "C" void run_head(const void* in,const void* nb,const void* nw,const void* hb,const void* hw,void* tmp){
 const glm_act* a=(const glm_act*)in;float* x=(float*)tmp;float* y=x+GLM_H;float* logits=y+GLM_H;
 for(int j=0;j<GLM_H;++j){float s=0;for(int k=0;k<GLM_HC;++k)s+=glm_get(a[k*GLM_H+j]);x[j]=glm_bf(s/GLM_HC);}
 glm_norm((const uint8_t*)nb,((const glm_weight*)nw)[0],x,y,GLM_H);
 glm_mv((const uint8_t*)hb,((const glm_weight*)hw)[0],y,logits);
}
''')
    lib=out/'cpu.dylib'
    subprocess.run(['c++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC','-I',str(root),'-I',str(out),str(source),'-o',str(lib)],check=True,timeout=120)
    return lib


class CPUModel:
    def __init__(self,out):
        import numpy as np
        self.np=np;self.spec=stage.validate(out,allow_fixture=True);self.d=self.spec['dimensions']
        self.lib=ctypes.CDLL(str(compile_cpu(out)))
        ptr=ctypes.c_void_p;integer=ctypes.c_int
        self.lib.run_layer.argtypes=[ptr]*6+[integer]*3
        self.lib.run_embed.argtypes=[ptr,ptr,integer,ptr]
        self.lib.run_mla.argtypes=[ptr]*6+[integer]
        self.lib.run_mlp.argtypes=[ptr]*5
        self.lib.run_head.argtypes=[ptr]*6
        self.weights={k:(np.fromfile(out/v['blob'],dtype=np.uint8),np.fromfile(out/v['meta'],dtype=np.uint8)) for k,v in self.spec['weights'].items()}
        for name in ('run_layer','run_embed','run_head','run_mla','run_mlp'):getattr(self.lib,name).restype=None
        self.reset()
    def reset(self,batch=1):
        d=self.d;np=self.np;self.batch=batch
        self.states=[[np.zeros(d['STATE_KDA'] if kind=='linear_attention' else d['STATE_MLA'],dtype=np.float32) for kind in self.spec['config']['layer_types']] for _ in range(batch)]
        self.state_guards=[]
        for request in self.states:
            for i,state in enumerate(request):
                backing=np.full(len(state)+16,np.float32(123456.25));backing[:len(state)]=0
                request[i]=backing[:-16];self.state_guards.append(backing)
        self.position=0
    def __call__(self,ids):
        import torch
        np=self.np;d=self.d
        if len(ids)!=self.batch or self.position>=d['MAXLEN']:raise ValueError('invalid batch/position')
        logits=[];layers=[]
        for r,token in enumerate(ids):
            if not 0<=token<d['VOCAB']:raise ValueError('invalid token')
            dtype=np.uint16 if d['BF16'] else np.float32
            x=np.zeros((d['HC'],d['H']),dtype=dtype)
            addr=lambda v:ctypes.c_void_p(v.ctypes.data)
            blob,meta=self.weights['embedding'];self.lib.run_embed(addr(blob),addr(meta),token,addr(x))
            backing=np.full(d['SCRATCH']+16,np.float32(123456.25));tmp=backing[:-16];tmp.fill(0);outputs=[]
            for i,kind in enumerate(self.spec['config']['layer_types']):
                blob,meta=self.weights[f'l{i}'];y=np.zeros_like(x)
                self.lib.run_layer(addr(blob),addr(meta),addr(x),addr(y),addr(self.states[r][i]),addr(tmp),self.position,int(kind=='linear_attention'),int(self.spec['config']['mlp_layer_types'][i]=='sparse'))
                if not np.all(backing[-16:]==np.float32(123456.25)) or any(not np.all(s[-16:]==np.float32(123456.25)) for s in self.state_guards):raise ValueError('native workspace/state overrun')
                x=y;outputs.append(x.copy())
            nb,nw=self.weights['norm'];hb,hw=self.weights['head'];final=np.zeros(2*d['H']+d['VOCAB'],dtype=np.float32)
            self.lib.run_head(addr(x),addr(nb),addr(nw),addr(hb),addr(hw),addr(final))
            logits.append(final[2*d['H']:]);layers.append(outputs)
        self.last_scratch=tmp.copy()
        self.position+=1
        return np.stack(logits),layers


def check(out,batch=2,steps=32,bf16=False,fp8=False):
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    spec,hf,head=fixture(out,bf16,fp8);candidate=CPUModel(out);candidate.reset(batch)
    cache=DynamicCache(config=hf.config);ids=torch.tensor([[3+r] for r in range(batch)])
    errors=[];state_errors=[];matched=0
    with torch.inference_mode():
        for pos in range(steps):
            captured=[];selected=[]
            ih=[l.self_attn.indexer.register_forward_hook(lambda mod,args,res:selected.append(res.detach().cpu().numpy())) for l in hf.layers if l.block_type!='linear_attention']
            handles=[l.register_forward_hook(lambda mod,args,res:captured.append(res[0].detach().float().cpu())) for l in hf.layers]
            ref=hf(input_ids=ids,past_key_values=cache,use_cache=True).last_hidden_state
            for h in handles+ih:h.remove()
            target=head(ref)[:,0].float();got,layer_out=candidate(ids[:,0].tolist())
            diff=float(np.max(np.abs(got-target.numpy())));errors.append(diff)
            for i in range(4):
                actual=np.stack([layer_out[r][i] for r in range(batch)])
                if bf16:actual=torch.from_numpy(actual).view(torch.bfloat16).float().numpy()
                expected=captured[i][:,0].numpy()
                if not np.allclose(actual,expected,atol=0.012 if bf16 else 3e-5,rtol=0.025 if bf16 else 1e-4):
                    
                    np.savez(out/'failure.npz',actual=actual,expected=expected,ids=ids.numpy(),scratch=candidate.last_scratch,selected=selected[-1],state=candidate.states[-1][-1]);raise ValueError(f'layer mismatch pos={pos} layer={i} max_abs={np.max(np.abs(actual-expected))}')
                if hf.config.layer_types[i]=='linear_attention':
                    n=spec['dimensions']['QDIM']*3*spec['dimensions']['CONV']
                    state=np.stack([candidate.states[r][i][n:].reshape(2,4,4) for r in range(batch)])
                    refstate=cache.layers[i].recurrent_states[0].float().numpy()
                    state_errors.append(float(np.max(np.abs(state-refstate))))
                    if not np.allclose(state,refstate,atol=0.002 if bf16 else 2e-5,rtol=0.04 if bf16 else 2e-4):raise ValueError(f'KDA state mismatch pos={pos} layer={i}')
            reference=target.argmax(-1).numpy();actual=got.argmax(-1)
            if not np.array_equal(actual,reference):raise ValueError(f'greedy mismatch pos={pos}: {actual} != {reference}')
            matched+=batch;ids=torch.tensor(actual[:,None])
            if pos%8==0:print(f'GLM-CPU step={pos} batch={batch} logit_max_abs={diff:.8g}',flush=True)
    result=dict(schema=1,status='PASS',fixture=True,batch=batch,steps=steps,matched=matched,
                max_logit_abs=max(errors),max_kda_state_abs=max(state_errors),bf16=bf16,fp8=fp8,
                implementation='same C++ numerical body as ETX HIP tiles',reference='independent Transformers GLM5Next text model',
                full_checkpoint=False,gpu_correctness='unmeasured',performance='unmeasured')
    (out/'cpu-check.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch',type=int,default=2);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--bf16',action='store_true');ap.add_argument('--fp8',action='store_true')
    a=ap.parse_args();print(json.dumps(check(a.out.resolve(),a.batch,a.steps,a.bf16,a.fp8),indent=2))



def check_prefill(out,batch=2,context=8,bf16=False,fp8=False,fp32_conv=True):
    """HF chunk prefill versus native sequential state, then exact free decode."""
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    spec,hf,head=fixture(out,bf16,fp8,fp32_conv=fp32_conv);candidate=CPUModel(out);candidate.reset(batch)
    prompts=[[1+(i+3*r)%47 for i in range(context)] for r in range(batch)]
    cache=DynamicCache(config=hf.config);errors=[];matched=0
    with torch.inference_mode():
        for pos in range(context):got,_=candidate([row[pos] for row in prompts])
        ref=head(hf(input_ids=torch.tensor(prompts),past_key_values=cache,use_cache=True).last_hidden_state[:,-1]).float().numpy()
        for step in range(32):
            if not np.array_equal(got.argmax(-1),ref.argmax(-1)):raise ValueError(f'chunk-prefill greedy mismatch step={step}')
            errors.append(float(abs(got-ref).max()));matched+=batch
            if step<31:
                ids=got.argmax(-1);got,_=candidate(ids.tolist())
                ref=head(hf(input_ids=torch.tensor(ids[:,None]),past_key_values=cache,use_cache=True).last_hidden_state[:,0]).float().numpy()
    result=dict(status='PASS',fixture=True,batch=batch,context=context,steps=32,matched=matched,
                max_logit_abs=max(errors),bf16=bf16,fp8=fp8,gpu_correctness='unmeasured')
    (out/'prefill-check.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':main()
