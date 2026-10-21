"""Bind retained historical native DSL bodies and host to true ready/output capture.

The outputs are NEW source-bound captures. They do not recreate missing old
run/tensor identities. All patched files are written to a fresh private output.
"""
from __future__ import annotations
import argparse,importlib.util,json,os,re
from pathlib import Path
from .build import SOURCES
from .protocol import require,sha256
from .patch_capture import once

def patch_tiles(text):
    text=once(text,'#include "etx/tile_util.h"','#include "etx/tile_util.h"\n#define ETX_HIST_DEVICE_DEFINITION 1\n#include "examples/experiment_program/columns/historical_capture.h"')
    forms=['dsl_gemv','dsl_moe_up','dsl_moe_down','dsl_gemv_fold','dsl_moe_up_fold','dsl_moe_down_fold']
    for name in forms:
        marker=f'void dsl_run_{name}(const unsigned char* a, int blk) {{'
        text=once(text,marker,f'void dsl_run_{name}(const unsigned char* a, int blk,const etx_ctx* capture_ctx) {{')
        begin=text.index(f'void dsl_run_{name}(');end=text.index('\nextern "C"',begin)
        body=text[begin:end]
        if 'gemv' in name:
            lines=body.splitlines();stage=next(i for i,line in enumerate(lines) if 'else dsl_stage_nb(d.x,' in line)
            lines.insert(stage+1,'  EtxColumnRow* column_row=hist_ready(capture_ctx,a,blk,xs,d.K,KS,DSL_NB,d.W,d.M,d.K,-1,0,-1,d.estride);')
            body='\n'.join(lines);finish='  hist_finish(column_row,d.out,d.sout?d.sout:d.M,DSL_NB,r0,r1);'
        else:
            if 'int s = S0, rq = 0;' not in body:body=body.replace('  const DslArgs d = dsl_args(a);','  const DslArgs d = dsl_args(a);\n  const int rq=0;')
            lines=body.splitlines();stage=next(i for i,line in enumerate(lines) if line.strip().startswith('dsl_stage(d.x'))
            # Down captures the exact arithmetic operand, including batched
            # routing-weight stride. Up has no d.wts operand in the unfurled
            # original; only a folded top-k's actual LDS weight overrides the
            # original host-published routing metadata.
            if 'down' in name:
                match=re.search(r'const float ws = (.*);',body)
                require(match is not None,'native down weight expression absent')
                mix=match.group(1);use='1'
            else:mix='d.mode >= 2 ? lwts[s] : 0.f';use='d.mode >= 2'
            lines.insert(stage+1,f'  EtxColumnRow* column_row=hist_ready(capture_ctx,a,blk,xs,d.K,d.K,1,We,d.M,d.K,s,rq,(int)((We-d.W)/d.estride),d.estride,{mix},{use});')
            body='\n'.join(lines);finish='  hist_finish(column_row,d.out+(size_t)rq*d.sout+(size_t)s*d.M,d.M,1,r0,r1);'
        require(body.endswith('\n}'),'native function boundary changed')
        body=body[:-2]+'\n'+finish+'\n}'
        text=text[:begin]+body+text[end:]
    lines=[]
    for line in text.splitlines():
        if 'dsl_run_' in line and 'etx_imp_argblock' in line and line.rstrip().endswith(');'):line=line.rstrip()[:-2]+', c);'
        lines.append(line)
    return '\n'.join(lines)+'\n'

ALLOC=r'''
  const char* hist_dir=getenv("ETX_HIST_CAPTURE_DIR");
  const int hist_layer=getenv("ETX_HIST_CAPTURE_LAYER")?atoi(getenv("ETX_HIST_CAPTURE_LAYER")):11;
  const std::string hist_role=getenv("ETX_HIST_CAPTURE_ROLE")?getenv("ETX_HIST_CAPTURE_ROLE"):"qkv";
  const int hist_position=getenv("ETX_HIST_CAPTURE_POSITION")?atoi(getenv("ETX_HIST_CAPTURE_POSITION")):1040;
  if(hist_dir && (nreq!=2 || repeat!=2 || gen!=32 || teacher || unfused || max_steps>0 || (hist_position!=0&&hist_position!=1040)))
    die("historical capture requires full B2/32greedy/repeat2 source, position0/1040; no shortened/token-only certificate");
  std::vector<EtxHistConfig> hist_configs;std::vector<std::string> hist_names;
  if(hist_dir)for(const Inst& x:insts)if(x.layer==hist_layer && (x.role==hist_role || (hist_role=="expert_up_down"&&(x.role=="moe_up"||x.role=="moe_down"))) && x.exp.find("dsl_")==0){
    const int per=x.tasks/x.rep;if(x.tasks%x.rep)die("native historical replica task geometry");
    for(int rr=0;rr<x.rep;++rr){EtxHistConfig cfg{};cfg.blocks=per;cfg.base_block=rr*per;cfg.request=x.req;
      cfg.layer=x.layer;cfg.role=x.role=="qkv"?1:x.role=="moe_up"?2:3;cfg.selected_position=hist_position;
      cfg.position=reqs[x.req].positions;cfg.mix=reqs[x.req].topk_w+(size_t)rr*TOPK;cfg.mix_stride=n_tk;
      cfg.argument=(const unsigned char*)hst.args[etx_host::arg_index("impargs")]+x.off+(size_t)rr*x.stride;
      ETX_CHECK(hipMalloc((void**)&cfg.rows,per*sizeof(EtxColumnRow)));ETX_CHECK(hipMemset(cfg.rows,0,per*sizeof(EtxColumnRow)));
      hist_configs.push_back(cfg);hist_names.push_back("native_n"+std::to_string(x.n)+"_rep"+std::to_string(rr));
    }}
  if(hist_dir){if(hist_configs.empty()||hist_configs.size()>32)die("native historical capture target population");etx_hist_configure(hist_configs.data(),hist_configs.size(),hst.stream);}
'''
EXPORT=r'''
  if(hist_dir){
    int clock=0;ETX_CHECK(hipDeviceGetAttribute(&clock,hipDeviceAttributeWallClockRate,0));
    if(mall!=32)die("historical capture refuses non-exact per-request greedy outputs");
    for(size_t j=0;j<hist_configs.size();++j){const auto& cfg=hist_configs[j];std::vector<EtxColumnRow> rows(cfg.blocks);
      ETX_CHECK(hipMemcpy(rows.data(),cfg.rows,rows.size()*sizeof(EtxColumnRow),hipMemcpyDeviceToHost));
      for(auto& row:rows)if(row.h[0]!=0x455458434f4c3031ull||row.h[23]!=1||row.h[22]!=16||row.h[25]||row.h[4]!=(unsigned)hist_position)die("native historical capture incomplete/duplicate/stale");
      for(auto& row:rows){row.h[31]=clock;if(row.h[15]<(uint64_t)dw)die("historical native weight pointer outside blob");row.h[30]=row.h[15]-(uint64_t)dw;}
      std::ofstream out(std::string(hist_dir)+"/"+hist_names[j]+".bin",std::ios::binary);out.write((const char*)rows.data(),rows.size()*sizeof(EtxColumnRow));out.close();if(!out)die("native historical raw export failed");
      ETX_CHECK(hipFree(cfg.rows));
    }
    printf("HISTORICAL NEW CAPTURE COMPLETE targets=%zu native source/token/task bytes retained; missing historical run identity remains missing\n",hist_configs.size());
  }
'''
def patch_host(text):
    text=once(text,'struct Inst {','#include "examples/experiment_program/columns/historical_capture.h"\n\nstruct Inst {')
    text=once(text,'  std::vector<int> matched(nreq, 0);','  std::vector<int> matched(nreq, 0);\n'+ALLOC)
    text=once(text,'  for (int run = 0; run < repeat; ++run) {','  for (int run = 0; run < repeat; ++run) {\n    if(hist_dir)etx_hist_enable(run==repeat-1,hst.stream);')
    marker='      printf("run %d%s: %d/%d generated tokens match HF greedy%s", run, nreq > 1 ? (" request " + std::to_string(r)).c_str() : "",'
    text=once(text,marker,'      if(hist_dir && matched[r]!=32)die("historical native capture warmup/greedy per-request gate failed");\n'+marker)
    text=once(text,'  if (check_dsl && nreq > 1)', '  if(hist_dir)etx_hist_enable(0,hst.stream);\n  if (check_dsl && nreq > 1)')
    text=once(text,'  return 0;\n}',EXPORT+'\n  return 0;\n}')
    return text

def prepare(source,host,out,revision,nb):
    require(sha256(source)==SOURCES[revision],'historical source changed')
    require(nb in (1,2),'historical source NB must be declared')
    out=Path(out);require(not out.exists(),'historical capture output occupied');out.mkdir(parents=True)
    old=os.environ.get('ETX_VL_NB');os.environ['ETX_VL_NB']=str(nb)
    try:
        spec=importlib.util.spec_from_file_location('etx_exact_historical_gemv',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        tiles=module.emit_tiles()
    finally:
        if old is None:os.environ.pop('ETX_VL_NB',None)
        else:os.environ['ETX_VL_NB']=old
    (out/'dsl_tiles.hip').write_text(patch_tiles(tiles));(out/'host.hip').write_text(patch_host(Path(host).read_text()))
    receipt={'source':str(source),'source_sha256':sha256(source),'host_input_sha256':sha256(host),'revision':revision,'nb':nb,
             'label':'NEW_CAPTURE_NOT_HISTORICAL_RUN_RECONSTRUCTION','outputs':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}}
    probe=['#define ETX_IMPORT_LDS 50280','#define ETX_IMPORT_SLOTS 16','#include "etx_lowering.h"','#include "etx/primitives.h"','#include "dsl_tiles.hip"']
    for name in ('dsl_gemv','dsl_gemv_fold','dsl_moe_up','dsl_moe_up_fold','dsl_moe_down','dsl_moe_down_fold'):
        probe.append(f'extern "C" __global__ __launch_bounds__(1024,1) void captured_{name}(etx_ctx ctx,const unsigned char* args) {{ etx_import_init(); __syncthreads(); dsl_run_{name}(args,blockIdx.x,&ctx); }}')
    (out/'capture-probe.hip').write_text('\n'.join(probe)+'\n');receipt['outputs']['capture-probe.hip']=sha256(out/'capture-probe.hip')
    (out/'identity.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--host',type=Path,required=True)
    p.add_argument('--revision',choices=SOURCES,required=True);p.add_argument('--nb',type=int,choices=[1,2],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.host,a.out,a.revision,a.nb),indent=2))
if __name__=='__main__':main()
