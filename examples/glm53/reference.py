"""Strict independent offline HF oracle for the exact GLM checkpoint.

Dequantizes native FP8 to BF16, eager attention/experts, no speculative decoding.
Can use Accelerate disk offload for the BF16 reference, which does NOT fit on
2x192 GiB. Requires explicitly supplied staging/offload paths; no downloads.
"""
import argparse
import json
from pathlib import Path
from examples.glm53.prepare import gate_identity,metadata,sha


def identity(gate='production'):
    evidence,model=gate_identity(gate);_,api=metadata(evidence)
    return dict(schema=1,model=model,revision=api['sha'],config_sha256=sha(evidence/'glm-config.json'),
                mode='greedy',speculative=False)


def prompts(path,gate='production'):
    cfg,_=metadata(gate_identity(gate)[0]);p=json.loads(path.read_text())
    if not isinstance(p,list) or len(p) not in (1,2,4,8) or len({tuple(x) for x in p})!=len(p):raise ValueError('distinct prompts required for B1/2/4/8')
    if len({len(x) for x in p})!=1 or not p[0] or len(p[0])+32>cfg['text_config']['max_position_embeddings']:raise ValueError('equal nonempty prompt lengths required')
    if any(type(x) is not int or not 0<=x<cfg['text_config']['vocab_size'] for row in p for x in row):raise ValueError('invalid prompt ID')
    return p


def run(src,prompt_path,out,offload,max_memory,gate='production'):
    import torch
    import transformers
    from transformers import AutoConfig,AutoModelForImageTextToText,FineGrainedFP8Config
    evidence,_=gate_identity(gate);cfg,api=metadata(evidence)
    if out.exists() or out.with_suffix('.logits').exists():raise ValueError('reference output must be new')
    if sha(src/'config.json')!=sha(evidence/'glm-config.json') or (src/'REVISION').read_text().strip()!=api['sha']:raise ValueError('wrong checkpoint')
    from examples.glm53.stage import Snapshot
    snapshot=Snapshot(src)
    for key in snapshot.index:snapshot.shard(key)
    p=prompts(prompt_path,gate);record=dict(identity(gate),backend='hf',batch=len(p),requests=[])
    # The saved release config names DSA differently from the library cache ABI.
    config=AutoConfig.from_pretrained(src,local_files_only=True,trust_remote_code=False)
    config.text_config.layer_types=['indexed_attention' if x=='deepseek_sparse_attention' else x for x in config.text_config.layer_types]
    config.text_config._experts_implementation='eager'
    # Placement/offload only: the release does not fit and needs Accelerate; a checkpoint
    # that fits one device (the module gate) loads without them (max_memory=None).
    placement={} if max_memory is None else dict(device_map='auto',max_memory=max_memory,offload_folder=str(offload))
    model,info=AutoModelForImageTextToText.from_pretrained(src,config=config,local_files_only=True,trust_remote_code=False,
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
    ap.add_argument('--gate',choices=['production','module'],default='production')
    a=ap.parse_args();m={int(k) if k.isdigit() else k:v for k,v in json.loads(a.max_memory).items()};run(a.src,a.prompts,a.out,a.offload,m,a.gate)

if __name__=='__main__':main()
