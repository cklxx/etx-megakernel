"""Strict independent offline HF oracle for the exact GLM checkpoint.

Dequantizes native FP8 to BF16, eager attention/experts, no speculative decoding.
Can use Accelerate disk offload for the BF16 reference, which does NOT fit on
2x192 GiB. Requires explicitly supplied staging/offload paths; no downloads.

--gate slice --layers 22-25 writes a derived snapshot of only those source
layers (re-indexed from 0, outer tensors kept) and runs the same forward on it.
The window must straddle the PP2 split or the run is refused: a window on one
side cannot gate the two-rank pipeline.
"""
import argparse
import json
from pathlib import Path
from examples.glm53.prepare import (gate_identity,metadata,sha,check_slice,
                                    parse_layers,slice_config)


LAYER_PREFIXES=('model.layers.','model.language_model.layers.','language_model.model.layers.')


def identity(gate='production',layers=None):
    evidence,model=gate_identity(gate);cfg,api=metadata(evidence)
    out=dict(schema=1,model=model,revision=api['sha'],config_sha256=sha(evidence/'glm-config.json'),
             mode='greedy',speculative=False)
    if layers is not None:out['slice']=check_slice(cfg['text_config'],layers)
    return out


def prompts(path,gate='production'):
    cfg,_=metadata(gate_identity(gate)[0]);p=json.loads(path.read_text())
    if not isinstance(p,list) or len(p) not in (1,2,4,8) or len({tuple(x) for x in p})!=len(p):raise ValueError('distinct prompts required for B1/2/4/8')
    if len({len(x) for x in p})!=1 or not p[0] or len(p[0])+32>cfg['text_config']['max_position_embeddings']:raise ValueError('equal nonempty prompt lengths required')
    if any(type(x) is not int or not 0<=x<cfg['text_config']['vocab_size'] for row in p for x in row):raise ValueError('invalid prompt ID')
    return p


def split_layer_key(key):
    """(checkpoint prefix, source layer index, tensor name inside the layer) or None."""
    for prefix in LAYER_PREFIXES:
        if key.startswith(prefix):
            index,_,rest=key[len(prefix):].partition('.')
            if index.isdigit() and rest:return prefix,int(index),rest
    return None


def slice_weight_map(weight_map,layers):
    """Checkpoint key map for a contiguous source window.

    Layer keys are renamed to 0..len(layers)-1; every non-layer key keeps its
    name. Returns (rename, groups) where groups[j] lists (new_key, shard, old_key)
    for derived layer j and groups[None] holds the outer tensors.
    """
    order={source:j for j,source in enumerate(layers)}
    rename={};groups={j:[] for j in range(len(layers))};groups[None]=[]
    for key,shard in weight_map.items():
        hit=split_layer_key(key)
        if hit is None:
            rename[key]=key;groups[None].append((key,shard,key));continue
        prefix,index,rest=hit
        j=order.get(index)
        if j is None:continue
        new=f'{prefix}{j}.{rest}';rename[key]=new;groups[j].append((new,shard,key))
    empty=[j for j in range(len(layers)) if not groups[j]]
    if empty:raise ValueError(f'checkpoint has no tensors for slice position(s) {empty}')
    return rename,groups


def write_slice_snapshot(src,dst,layers):
    """Materialise a derived snapshot HF can load: one safetensors per selected
    layer plus the outer tensors, an index, and a truncated config. Consumes
    roughly one layer of extra disk beyond what is copied (~29 GB for 22-25)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    layers=tuple(layers)
    if dst.exists():raise ValueError('slice snapshot destination must be new')
    index=json.loads((src/'model.safetensors.index.json').read_text())
    _,groups=slice_weight_map(index['weight_map'],layers)
    cfg=json.loads((src/'config.json').read_text());cfg['text_config']=slice_config(cfg['text_config'],layers)
    dst.mkdir(parents=True,exist_ok=False)
    (dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (dst/'REVISION').write_text((src/'REVISION').read_text())
    weight_map={};total=0
    for group,items in groups.items():
        name='outer.safetensors' if group is None else f'model-{group:05d}.safetensors'
        by_shard={}
        for new,shard,old in items:
            if Path(shard).name!=shard:raise ValueError('unsafe checkpoint shard path')
            by_shard.setdefault(shard,[]).append((new,old))
        tensors={}
        for shard,pairs in sorted(by_shard.items()):
            with safe_open(src/shard,framework='pt',device='cpu') as f:
                for new,old in pairs:tensors[new]=f.get_tensor(old)
        save_file(tensors,dst/name,metadata={'format':'pt'})
        for key,value in tensors.items():
            weight_map[key]=name;total+=value.numel()*value.element_size()
        print(f'GLM-SLICE-SHARD {name} tensors={len(tensors)} bytes={total}',flush=True)
        del tensors
    (dst/'model.safetensors.index.json').write_text(json.dumps(
        {'metadata':{'total_size':total},'weight_map':weight_map},indent=2,sort_keys=True)+'\n')


def run(src,prompt_path,out,offload,max_memory,gate='production',layers=None):
    import torch
    import transformers
    from transformers import AutoConfig,AutoModelForImageTextToText,FineGrainedFP8Config
    evidence,_=gate_identity(gate);cfg,api=metadata(evidence)
    if out.exists() or out.with_suffix('.logits').exists():raise ValueError('reference output must be new')
    if gate=='slice' and layers is None:raise ValueError('the slice gate requires --layers')
    if gate!='slice' and layers is not None:raise ValueError('--layers is only valid with --gate slice')
    # Refuse a window that misses the PP2 split before touching the checkpoint.
    window=check_slice(cfg['text_config'],layers) if layers is not None else None
    if sha(src/'config.json')!=sha(evidence/'glm-config.json') or (src/'REVISION').read_text().strip()!=api['sha']:raise ValueError('wrong checkpoint')
    from examples.glm53.stage import Snapshot
    snapshot=Snapshot(src)
    if window is None:
        for key in snapshot.index:snapshot.shard(key)
    else:
        # Hash only the source shards the slice actually reads; the full checkpoint is ~305 GiB.
        _,groups=slice_weight_map(snapshot.index,window['layers'])
        for items in groups.values():
            for _,_,old in items:snapshot.shard(old)
    p=prompts(prompt_path,gate);record=dict(identity(gate,window['layers'] if window else None),backend='hf',batch=len(p),requests=[])
    load_src=src
    if window is not None:
        load_src=out.parent/(out.name+'.slice')
        write_slice_snapshot(src,load_src,window['layers'])
        record['slice_snapshot']=dict(path=str(load_src),index_sha256=sha(load_src/'model.safetensors.index.json'))
        print(f'GLM-SLICE-SNAPSHOT {load_src} layers={window["layers"]}',flush=True)
    # The saved release config names DSA differently from the library cache ABI.
    config=AutoConfig.from_pretrained(load_src,local_files_only=True,trust_remote_code=False)
    config.text_config.layer_types=['indexed_attention' if x=='deepseek_sparse_attention' else x for x in config.text_config.layer_types]
    config.text_config._experts_implementation='eager'
    # Placement/offload only: the release does not fit and needs Accelerate; a checkpoint
    # that fits one device (the module gate) loads without them (max_memory=None).
    placement={} if max_memory is None else dict(device_map='auto',max_memory=max_memory,offload_folder=str(offload))
    model,info=AutoModelForImageTextToText.from_pretrained(load_src,config=config,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,quantization_config=FineGrainedFP8Config(dequantize=True),
        attn_implementation='eager',output_loading_info=True,**placement)
    model.eval()
    # A tensor HF could not map is silently random-initialized: refuse instead. The
    # module-gate checkpoint carries no vision tower (text-only gate), so only there
    # may the vision weights be missing.
    missing=[k for k in info['missing_keys'] if not (gate=='module' and k.startswith('model.visual.'))]
    bad=dict(missing=missing,unexpected=list(info['unexpected_keys']),mismatched=list(info['mismatched_keys']),errors=list(info['error_msgs']))
    if any(bad.values()):raise ValueError(f'HF did not load the checkpoint exactly: {bad}')
    device=model.get_input_embeddings().weight.device
    if device.type=='meta':device=torch.device('cuda:0')
    ids=torch.tensor(p,device=device);cache=None;generated=[[] for _ in p];logits=[]
    with torch.inference_mode():
        for step in range(32):
            result=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values;last=result.logits[:,-1].float();ids=last.argmax(-1,keepdim=True)
            logits.append(last.cpu().numpy())
            for r,token in enumerate(ids[:,0].tolist()):generated[r].append(token)
            print(f'GLM-HF-STEP {step+1}/32 batch={len(p)}',flush=True)
    import numpy as np
    np.stack(logits).astype('<f4').tofile(out.with_suffix('.logits'))
    source=snapshot.verified_inputs()
    record.update(requests=[dict(prompt_ids=row,generated_ids=g) for row,g in zip(p,generated)],
                  provenance=dict(transformers=transformers.__version__,quantization='FP8 block dequantize -> BF16',attention='eager',experts='eager',
                                  config_source_sha256=sha(src/'config.json'),index_sha256=source['index_sha256'],source_shards=source['shards']))
    out.write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--src',type=Path,required=True);ap.add_argument('--prompts',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--offload',type=Path,required=True);ap.add_argument('--max-memory',required=True,help='JSON map, e.g. {"0":"80GiB","1":"80GiB","cpu":"32GiB"}')
    ap.add_argument('--gate',choices=['production','module','slice'],default='production')
    ap.add_argument('--layers',help='slice gate only: contiguous source layers that straddle the PP2 split, e.g. 22-25')
    a=ap.parse_args();m={int(k) if k.isdigit() else k:v for k,v in json.loads(a.max_memory).items()}
    run(a.src,a.prompts,a.out,a.offload,m,a.gate,parse_layers(a.layers) if a.layers else None)

if __name__=='__main__':main()
