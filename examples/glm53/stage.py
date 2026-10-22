"""Offline GLM native-FP8 staging. Never downloads weights or instantiates HF.

ABI: one raw aligned blob + fixed 32-byte weight descriptors per complete layer.
FP8 matrices remain E4M3FN with 128x128 FP32 scales. Only one matrix is mapped
at a time. Vision/MTP tensors are excluded; every requested text tensor is strict.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from examples.glm53.prepare import (gate_identity, metadata, partition, sha,
                                    check_slice, parse_layers, slice_config)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


SLOTS = ['attn_hc.fn','attn_hc.base','attn_hc.scale','ffn_hc.fn','ffn_hc.base','ffn_hc.scale',
 'input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_proj.weight',
 'self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.conv1d.weight',
 'self_attn.forget_gate.f_a_proj.weight','self_attn.forget_gate.f_b_proj.weight',
 'self_attn.forget_gate.dt_bias','self_attn.forget_gate.A_log','self_attn.b_proj.weight',
 'self_attn.g_a_proj.weight','self_attn.g_b_proj.weight','self_attn.o_norm.weight',
 'self_attn.o_proj.weight','self_attn.q_a_proj.weight','self_attn.q_a_layernorm.weight',
 'self_attn.q_b_proj.weight','self_attn.kv_a_proj_with_mqa.weight','self_attn.kv_a_layernorm.weight',
 'self_attn.kv_b_proj.weight','self_attn.indexer.wq_b.weight','self_attn.indexer.wk.weight',
 'self_attn.indexer.k_norm.weight','self_attn.indexer.k_norm.bias','self_attn.indexer.weights_proj.weight',
 'self_attn.indexer.index_kpool_compress_gate','self_attn.indexer.index_kpool_compress_ape',
 'mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','mlp.gate.weight',
 'mlp.gate.e_score_correction_bias','mlp.shared_experts.gate_proj.weight',
 'mlp.shared_experts.up_proj.weight','mlp.shared_experts.down_proj.weight']


def dimensions(t, maxlen=1072, bf16=True):
    la=t.get('linear_attn_config', {})
    h=t['hidden_size']; lh=la.get('num_heads', t.get('linear_num_heads',64)); ld=la.get('head_dim',t.get('linear_head_dim',128))
    d=dict(H=h, HC=t['hc_mult'], EPS=t['rms_norm_eps'], HCEPS=t['hc_eps'], SINK=t['hc_sinkhorn_iters'],
      LH=lh, LD=ld, QDIM=lh*ld, CONV=la.get('short_conv_kernel_size',t.get('linear_conv_kernel_dim',4)),
      LOWER=la.get('gate_lower_bound',t.get('linear_lower_bound',-5.)), NH=t['num_attention_heads'], AD=t['qk_nope_head_dim'],
      QR=t['q_lora_rank'], KVR=t['kv_lora_rank'], IH=t['index_n_heads'], ID=t['index_head_dim'],
      POOL=t['index_kpool'], INDEX_TOPK=t['index_topk'], EXPERTS=t['n_routed_experts'], TOPK=t['num_experts_per_tok'],
      ROUTE_SCALE=t['routed_scaling_factor'], INTERMAX=max(t['intermediate_size'],t['moe_intermediate_size']),
      CLIP=t['swiglu_limit'], VOCAB=t['vocab_size'], MAXLEN=maxlen, BF16=int(bf16))
    if (t['qk_rope_head_dim'] or t['v_head_dim']!=d['AD'] or t.get('n_group',1)!=1 or
        t.get('n_shared_experts',1)!=1 or not t.get('norm_topk_prob',True) or
        not t.get('index_kpool_always_select_tail',True) or d['LOWER'] is None or
        not 1<=maxlen<=t['max_position_embeddings'] or d['INDEX_TOPK']<d['POOL']):
        raise ValueError('unsupported GLM numerical contract (NoPE, equal KV head dims, one group/shared expert, normalized routing, safe KDA, tail required)')
    if not 1<=d['TOPK']<=d['EXPERTS'] or len(t['layer_types'])!=t['num_hidden_layers']:
        raise ValueError('invalid layer or expert schedule')
    d['STATE_KDA']=3*d['QDIM']*d['CONV']+lh*ld*ld
    d['STATE_MLA']=maxlen*(2*d['NH']*d['AD']+2*d['ID'])
    kda=7*d['QDIM']+d['LD']+d['LH']
    mla=d['QR']+d['NH']*d['AD']+d['KVR']+2*d['NH']*d['AD']+d['IH']*d['ID']+2*d['ID']+d['IH']+d['NH']*d['AD']+2*maxlen+d['ID']+d['NH']*maxlen
    ffn=2*d['INTERMAX']+d['EXPERTS']+2*d['TOPK']+h
    d['SCRATCH']=2*d['HC']*h+3*h+2*d['HC']+d['HC']**2+max(kda,mla,ffn)
    return d


def header(d):
    return '#pragma once\n'+''.join(f'#define GLM_{k} {v if isinstance(v,int) else format(v,".9g")+"f" if "." in format(v,".9g") or "e" in format(v,".9g") else str(v)+"f"}\n' for k,v in d.items())


def shapes(t, layer):
    d=dimensions(t,min(1072,t['max_position_embeddings'])); h=d['H']; hc=d['HC']; n=d['QDIM']; ld=d['LD']; nh=d['NH']; ad=d['AD']
    s={k:v for k,v in zip(SLOTS[:8],[(hc*(hc+2),hc*h),(hc*(hc+2),),(3,),
        (hc*(hc+2),hc*h),(hc*(hc+2),),(3,),(h,),(h,)])}
    if t['layer_types'][layer]=='linear_attention':
        values=[(n,h),(n,h),(n,h),(3*n,1,d['CONV']),(ld,h),(n,ld),(n,),(d['LH'],),
                (d['LH'],h),(ld,h),(n,ld),(ld,)]
        s.update(zip(SLOTS[8:20],values));s[SLOTS[20]]=(h,n)
    else:
        values=[(d['QR'],h),(d['QR'],),(nh*ad,d['QR']),(d['KVR'],h),(d['KVR'],),
                (nh*2*ad,d['KVR']),(d['IH']*d['ID'],d['QR']),(d['ID'],h),
                (d['ID'],),(d['ID'],),(d['IH'],h),(d['ID'],h),(d['POOL'],d['ID'])]
        s.update(zip(SLOTS[21:34],values));s[SLOTS[20]]=(h,nh*ad)
    if t['mlp_layer_types'][layer]=='dense':
        inter=t['intermediate_size'];s.update(zip(SLOTS[34:37],[(inter,h),(inter,h),(h,inter)]))
    else:
        inter=t['moe_intermediate_size'];s.update(zip(SLOTS[37:42],[(d['EXPERTS'],h),(d['EXPERTS'],),(inter,h),(inter,h),(h,inter)]))
        for e in range(d['EXPERTS']):
            for k,shape in zip(('gate_proj','up_proj','down_proj'),((inter,h),(inter,h),(h,inter))):s[f'mlp.experts.{e}.{k}.weight']=shape
    return s


def slot_names(t):
    return SLOTS+[f'mlp.experts.{e}.{k}.weight' for e in range(t['n_routed_experts']) for k in ('gate_proj','up_proj','down_proj')]


TRUST=[False]


def write_blob(out, name, expected, tensors, slots):
    """tensors(key) -> (torch Tensor, optional FP32 block scale); bounded mapping."""
    import torch
    blob=out/f'{name}.bin'; desc=[];entries={}
    with blob.open('xb') as f:
        for key in slots:
            if key not in expected:
                desc.append(struct.pack('<QQiiii',0,0,0,0,0,0));continue
            value,scale=tensors(key)
            if tuple(value.shape)!=tuple(expected[key]):raise ValueError(f'{name}/{key}: shape {tuple(value.shape)} expected {expected[key]}')
            dtype=value.dtype; kind={torch.bfloat16:1,torch.float32:2,torch.float8_e4m3fn:3}.get(dtype)
            if kind is None:raise ValueError(f'unsupported dtype {dtype}: {key}')
            if kind==3 and (value.ndim!=2 or scale is None):raise ValueError('FP8 requires a matrix and explicit block scales')
            if kind!=3 and scale is not None:raise ValueError('unexpected scale on non-FP8 tensor')
            if not TRUST[0] and not torch.isfinite(value.float()).all():raise ValueError(f'nonfinite weight {key}')
            rows,cols=(value.numel(),1) if value.ndim==1 else (value.shape[0],math.prod(value.shape[1:]))
            f.write(b'\0'*(-f.tell()%128));off=f.tell()
            raw=value.contiguous().view(torch.uint8).numpy().tobytes();f.write(raw)
            soff=0
            if scale is not None:
                if tuple(scale.shape)!=((rows+127)//128,(cols+127)//128) or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError(f'invalid FP8 block scales {key}')
                f.write(b'\0'*(-f.tell()%4));soff=f.tell();f.write(scale.float().contiguous().numpy().tobytes())
            desc.append(struct.pack('<QQiiii',off,soff,rows,cols,kind,128 if kind==3 else 0))
            entries[key]=dict(shape=list(value.shape),dtype=str(dtype),offset=off,scale_offset=soff)
    meta=out/f'{name}.meta';meta.write_bytes(b''.join(desc))
    return dict(blob=blob.name,meta=meta.name,blob_bytes=blob.stat().st_size,meta_bytes=meta.stat().st_size,
                blob_sha256=file_sha(blob),meta_sha256=file_sha(meta),tensors=entries)


class Snapshot:
    def __init__(self, src, trust=False):
        self.trust=trust;TRUST[0]=trust
        self.src=src
        self.index_sha256=file_sha(src/'model.safetensors.index.json')
        self.index=json.loads((src/'model.safetensors.index.json').read_text())['weight_map']
        self.shard_hashes={}
    def __call__(self, key):
        from safetensors import safe_open
        import torch
        # Native checkpoint names (HF converter names are accepted for local fixtures).
        raw=key
        for old,new in [('attn_hc.fn','hc_attn_fn'),('attn_hc.base','hc_attn_base'),('attn_hc.scale','hc_attn_scale'),
                        ('ffn_hc.fn','hc_ffn_fn'),('ffn_hc.base','hc_ffn_base'),('ffn_hc.scale','hc_ffn_scale')]:raw=raw.replace(old,new)
        raw=raw.replace('self_attn.forget_gate.','self_attn.')
        if '.self_attn.conv1d.weight' in key:
            values=[self(key.replace('conv1d','q_conv1d'))[0],self(key.replace('conv1d','k_conv1d'))[0],self(key.replace('conv1d','v_conv1d'))[0]]
            return torch.cat(values,dim=0),None
        if raw not in self.index and key in self.index:raw=key
        aliases=[raw]
        if raw.startswith('model.'):
            aliases += [raw.replace('model.','model.language_model.',1),raw.replace('model.','language_model.model.',1)]
        hits=[n for n in aliases if n in self.index]
        if len(hits)>1:raise ValueError(f'ambiguous checkpoint names: {hits}')
        if hits:raw=hits[0]
        if raw not in self.index:raise ValueError(f'missing checkpoint tensor: {raw}')
        path=self.shard(raw)
        with safe_open(path,framework='pt',device='cpu') as f:value=f.get_tensor(raw)
        scale=None
        if value.dtype==torch.float8_e4m3fn:
            scale_key=raw.removesuffix('.weight')+'.weight_scale_inv'
            if scale_key not in self.index:raise ValueError(f'missing native FP8 scale {scale_key}')
            with safe_open(self.shard(scale_key),framework='pt',device='cpu') as f:scale=f.get_tensor(scale_key)
        return value,scale
    def shard(self,key):
        name=self.index[key]
        if Path(name).name!=name or name in ('.','..'):raise ValueError('unsafe checkpoint shard path')
        # HF cache snapshots legitimately symlink to their sibling blob store.
        path=self.src/name
        if name not in self.shard_hashes:
            print(f'SNAPSHOT-SHARD hash_before={name}',flush=True)
            self.shard_hashes[name]=file_sha(path)
        return path
    def verified_inputs(self):
        if not self.trust and file_sha(self.src/'model.safetensors.index.json')!=self.index_sha256:raise ValueError('checkpoint index changed during staging')
        if not self.trust:
            for name,digest in self.shard_hashes.items():
                if file_sha(self.src/name)!=digest:raise ValueError(f'checkpoint shard changed during staging: {name}')
        return dict(index_sha256=self.index_sha256,shards=self.shard_hashes)


def stage(src, out, maxlen=1072, gate='production', layers=None, trust=False):
    """Stage the release checkpoint, or (--gate slice --layers 22-25) one contiguous
    window of it, re-indexed from layer 0. A slice must straddle the PP2 split."""
    evidence,model=gate_identity(gate);cfg,api=metadata(evidence)
    if gate=='slice' and layers is None:raise ValueError('the slice gate requires --layers')
    if gate!='slice' and layers is not None:raise ValueError('--layers is only valid with --gate slice')
    # Refuse a window that misses the PP2 split before touching the checkpoint.
    provenance=check_slice(cfg['text_config'],layers) if layers is not None else None
    if sha(src/'config.json')!=sha(evidence/'glm-config.json') or (src/'REVISION').read_text().strip()!=api['sha']:
        raise ValueError(f'snapshot is not the exact pinned GLM checkpoint/config ({gate} gate)')
    t=slice_config(cfg['text_config'],layers) if provenance else cfg['text_config']
    out.mkdir(parents=True,exist_ok=False)
    d=dimensions(t,maxlen);ranks=partition(t);reader=Snapshot(src,trust=trust)
    result=dict(schema=1,model=model,revision=api['sha'],config_sha256=sha(src/'config.json'),fixture=False,
                config=t,dimensions=d,ranks=ranks,weights={},numerics='native FP8 -> BF16 per block; FP32 accumulators; no activation quantization')
    if provenance is not None:
        result['slice']=provenance
        print(f'STAGE-SLICE layers={provenance["layers"]} split={provenance["split"]}',flush=True)
    for j,src_layer in enumerate(provenance['layers'] if provenance else range(t['num_hidden_layers'])):
        result['weights'][f'l{j}']=write_blob(out,f'l{j}',shapes(t,j),lambda k,j=j,src_layer=src_layer:reader(f'model.layers.{src_layer}.{k}'),slot_names(t))
        print(f'STAGE layer={j} source_layer={src_layer} bytes={result["weights"][f"l{j}"]["blob_bytes"]}',flush=True)
    outer={'embedding':('model.embed_tokens.weight',(t['vocab_size'],t['hidden_size'])),
           'norm':('model.norm.weight',(t['hidden_size'],)), 'head':('lm_head.weight',(t['vocab_size'],t['hidden_size']))}
    for name,(key,shape) in outer.items():result['weights'][name]=write_blob(out,name,{key:shape},reader,[key])
    result['snapshot_inputs']=reader.verified_inputs()
    (out/'stage.json').write_text(json.dumps(result,indent=2)+'\n');(out/'glm53_spec.h').write_text(header(d))
    validate(out,gate=gate)
    return result


def validate(out, allow_fixture=False, gate='production'):
    """The one staging validator. `gate` selects which pinned identity the stage must carry
    (the release, or the synthetic module-gate checkpoint); every other check is the same."""
    evidence,model=gate_identity(gate)
    r=json.loads((out/'stage.json').read_text());cfg,api=metadata(evidence)
    if r.get('schema')!=1 or r.get('model')!=model or r.get('revision')!=api['sha'] or r.get('config_sha256')!=sha(evidence/'glm-config.json'):raise ValueError(f'wrong staged model identity for the {gate} gate')
    if r.get('fixture') and not allow_fixture:raise ValueError('synthetic fixtures cannot qualify the production model')
    if not r.get('fixture') and r['dimensions']['BF16']!=1:raise ValueError('production activations must be BF16')
    if gate=='slice':
        sl=r.get('slice')
        if not isinstance(sl,dict):raise ValueError('the slice gate needs a --layers stage')
        canonical=check_slice(cfg['text_config'],tuple(sl.get('layers') or ()))
        if sl!=canonical:raise ValueError('changed slice provenance')
        if r['config']!=slice_config(cfg['text_config'],tuple(sl['layers'])):raise ValueError('changed slice config')
    elif r.get('slice') is not None:
        raise ValueError(f'a sliced stage cannot qualify the {gate} gate')
    elif not r.get('fixture') and r['config']!=cfg['text_config']:raise ValueError('changed production config')
    if r['dimensions']!=dimensions(r['config'],r['dimensions']['MAXLEN'],r['dimensions']['BF16']==1):raise ValueError('inconsistent compiled dimensions')
    n=r['config']['num_hidden_layers']
    if set(r['weights'])!={*(f'l{i}' for i in range(n)),'embedding','norm','head'}:raise ValueError('incomplete staged model')
    owned=[i for rank in r['ranks'] for i in rank['layers']]
    if sorted(owned)!=list(range(n)) or len(set(owned))!=n or len(r['ranks'])!=2:raise ValueError('invalid layer ownership')
    for name,spec in r['weights'].items():
        validate_descriptors(out,name,spec,r['config'])
        for kind in ('blob','meta'):
            p=out/spec[kind]
            if p.resolve().parent!=out.resolve() or p.stat().st_size!=spec[f'{kind}_bytes'] or file_sha(p)!=spec[f'{kind}_sha256']:raise ValueError('changed/unsafe staged tensor file')
    if (out/'glm53_spec.h').read_text()!=header(r['dimensions']):raise ValueError('changed compiled specification')
    return r


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--maxlen',type=int,default=1072)
    ap.add_argument('--gate',choices=['production','module','slice'],default='production')
    ap.add_argument('--trust-verified',action='store_true',help='reuse the hashes taken during staging instead of hashing every shard a second time')
    ap.add_argument('--layers',help='slice gate only: contiguous source layers that straddle the PP2 split, e.g. 22-25')
    a=ap.parse_args();stage(a.src,a.out,a.maxlen,a.gate,parse_layers(a.layers) if a.layers else None,trust=a.trust_verified)



def validate_descriptors(out, name, spec, t):
    """Refuse mismatched tensor schemas and all out-of-bounds native descriptors."""
    expected=shapes(t,int(name[1:])) if name.startswith('l') and name[1:].isdigit() else {
        {'embedding':'model.embed_tokens.weight','norm':'model.norm.weight','head':'lm_head.weight'}[name]:
        (t['hidden_size'],) if name=='norm' else (t['vocab_size'],t['hidden_size'])}
    # Outer fixture names are deliberately different from production checkpoints.
    if len(spec['tensors'])==1 and name in spec['tensors'] and name in ('embedding','norm','head'):
        expected={name:next(iter(expected.values()))}
    slots=slot_names(t) if name.startswith('l') and name[1:].isdigit() else list(expected)
    if set(spec['tensors'])!=set(expected) or spec['meta_bytes']!=32*len(slots):raise ValueError(f'incomplete tensor schema {name}')
    raw=(out/spec['meta']).read_bytes();ranges=[]
    for i,key in enumerate(slots):
        off,scale,rows,cols,kind,block=struct.unpack_from('<QQiiii',raw,i*32)
        if key not in expected:
            if any((off,scale,rows,cols,kind,block)):raise ValueError('nonempty unused descriptor')
            continue
        shape=expected[key];erows,ecols=(math.prod(shape),1) if len(shape)==1 else (shape[0],math.prod(shape[1:]))
        if rows!=erows or cols!=ecols or kind not in (1,2,3) or tuple(spec['tensors'][key]['shape'])!=tuple(shape):raise ValueError(f'incompatible descriptor {name}/{key}')
        if off%128 or off!=spec['tensors'][key]['offset'] or scale!=spec['tensors'][key]['scale_offset']:raise ValueError('unaligned/inconsistent descriptor')
        ranges.append((off,off+rows*cols*{1:2,2:4,3:1}[kind]))
        if kind==3:
            if block!=128 or len(shape)!=2 or scale%4:raise ValueError('invalid FP8 block ABI')
            ranges.append((scale,scale+4*((rows+127)//128)*((cols+127)//128)))
        elif scale or block:raise ValueError('unexpected scale descriptor')
    ranges.sort()
    if any(a<0 or b>spec['blob_bytes'] or b<=a for a,b in ranges) or any(ranges[i][1]>ranges[i+1][0] for i in range(len(ranges)-1)):raise ValueError('overlapping/out-of-bounds tensor descriptor')

if __name__=='__main__':main()
