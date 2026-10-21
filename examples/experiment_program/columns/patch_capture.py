"""Create opt-in capture source copies bound to original Fleet arithmetic sites.

No working source is modified. Generated host/shim/kernel paths must be supplied
explicitly to the original ETX compile/link path; source hashes are retained.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .protocol import require,sha256

def once(text,needle,replacement):
    require(text.count(needle)==1,'capture source anchor changed/duplicated: '+needle[:70])
    return text.replace(needle,replacement,1)

def patch_kernel(text):
    text=once(text,'struct TokenCtx {','struct TokenCtx {\n    EtxColumnSink* column_capture = nullptr;')
    text=once(text,'#include "gemv.h"','#include "gemv.h"\n#include "examples/experiment_program/columns/native_capture.h"')
    # Each call is after the original exact prologue and before the first GEMV load.
    marker='            const int q_rows = heads_per_xcd * d.q_head_dim;'
    ready='''            etx_column_ready(tc.column_capture,L,tc.tok,tc.pos,tc.epoch,1,xcd,worker,s.vec,d.hidden,
              d.heads*d.q_head_dim+d.kv_lora+d.qk_rope,d.hidden,w.layer_qkv[L],
              (uint64_t)(d.heads*d.q_head_dim+d.kv_lora+d.qk_rope)*d.hidden*2,-1,-1,1.f,t.kind,t.flags);
'''
    require(text.count(marker)==1,'QKV operand-ready anchor changed');text=text.replace(marker,ready+marker,1)
    marker='            expert_gate_up(u, s.vec, act.expert_h + (int64_t)xcd * d.moe_inter,'
    ready='''            etx_column_ready(tc.column_capture,L,tc.tok,tc.pos,tc.epoch,2,xcd,worker,s.vec,d.hidden,
              u.inter,d.hidden,u.gate_up,(uint64_t)2*u.inter*d.hidden*2,t.expert_slot,
              t.expert_slot>=0?ids[t.expert_slot]:t.expert_slot,u.weight,t.kind,t.flags,t.expert_slot>=0?ws[t.expert_slot]:1.f);
'''
    require(text.count(marker)==1,'expert-up operand-ready anchor changed');text=text.replace(marker,ready+marker,1)
    marker='                expert_down_chunk(u, s.vec, s.part, out, d.hidden, worker, nw,'
    ready='''                if(n_chunks==1) etx_column_ready(tc.column_capture,L,tc.tok,tc.pos,tc.epoch,3,xcd,worker,s.vec,kc,
                  d.hidden,u.down_ld,u.down,((uint64_t)(d.hidden-1)*u.down_ld+kc)*2,t.expert_slot,
                  t.expert_slot>=0?ids[t.expert_slot]:t.expert_slot,u.weight,t.kind,t.flags,t.expert_slot>=0?ws[t.expert_slot]:1.f);
'''
    require(text.count(marker)==1,'expert-down operand-ready anchor changed');text=text.replace(marker,ready+marker,1)
    return text

FINISH=r'''
  if(kind==fleet::TASK_QKV_FUSED) {
    const int qn=fp->d.heads*fp->d.q_head_dim,kvn=fp->d.kv_lora+fp->d.qk_rope;
    if(!(flags & fleet::FLAG_KVA_SHARED)) return; // qualified frozen K1 only
    const int qb=(int)c->domain*(qn/8),kb=(int)c->domain*((kvn+7)/8);
    etx_column_finish(fp->column_capture,L,fp->tok,fp->epoch,1,c->domain,worker,
      act.q,qb+worker,qb+qn/8,fleet::kWorkersPerXCD,false,
      act.kv_a,kb+worker,min(kvn,kb+(kvn+7)/8),fleet::kWorkersPerXCD,qn);
  } else if(kind==fleet::TASK_EXPERT_GATE_UP || kind==fleet::TASK_EXPERT_DOWN) {
    int n=kind==fleet::TASK_EXPERT_GATE_UP?fp->d.moe_inter:fp->d.hidden;
    const void* output=kind==fleet::TASK_EXPERT_GATE_UP?(const void*)(act.expert_h+(int64_t)c->domain*fp->d.moe_inter):
      (const void*)(act.expert_out+(int64_t)c->domain*fp->d.hidden);
    etx_column_finish(fp->column_capture,L,fp->tok,fp->epoch,kind==fleet::TASK_EXPERT_GATE_UP?2:3,
      c->domain,worker,output,worker,n,fleet::kWorkersPerXCD,kind==fleet::TASK_EXPERT_DOWN);
  }
'''
def patch_shim(text):
    text=once(text,'#include "kernels/fleet_kernel.hip"','#include "kernel.hip"')
    text=once(text,'struct FleetParams {','struct FleetParams {\n  EtxColumnSink* column_capture;')
    # All wrappers that construct TokenCtx have an fp in scope. Non-capture
    # contexts remain nullptr by TokenCtx's initializer.
    text=text.replace('tc.trace = false;','tc.trace = false; tc.column_capture=fp->column_capture;')
    marker='  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, worker);'
    require(text.count(marker)==1,'common Fleet original call anchor changed')
    return text.replace(marker,marker+'\n'+FINISH,1)

HOST_ALLOC=r'''
  const char* column_dir=getenv("ETX_COLUMN_CAPTURE_DIR");
  const int column_layer=getenv("ETX_COLUMN_CAPTURE_LAYER")?atoi(getenv("ETX_COLUMN_CAPTURE_LAYER")):1;
  const int column_token=getenv("ETX_COLUMN_CAPTURE_TOKEN")?atoi(getenv("ETX_COLUMN_CAPTURE_TOKEN")):16;
  const int column_role=getenv("ETX_COLUMN_CAPTURE_ROLE")?atoi(getenv("ETX_COLUMN_CAPTURE_ROLE")):1;
  if(column_dir && (R<2 || R>4 || context!=1024 || tokens!=32 || repeat!=2 || teacher || unfused ||
     column_layer<1 || column_layer>=d.layers || column_token<0 || column_token>=32 || column_role<1 || column_role>3))
    die("column capture requires source-bound B2/B4,context1024,32greedy,repeat2 and valid role/layer/token");
  int column_clock=0;if(column_dir)HIP_CHECK(hipDeviceGetAttribute(&column_clock,hipDeviceAttributeWallClockRate,0));
  std::vector<EtxColumnRow*> column_rows(R,nullptr);std::vector<EtxColumnSink*> column_sinks(R,nullptr);
  if(column_dir)for(int r=0;r<R;++r){
    HIP_CHECK(hipMalloc((void**)&column_rows[r],8*37*sizeof(EtxColumnRow)));
    HIP_CHECK(hipMemset(column_rows[r],0,8*37*sizeof(EtxColumnRow)));
    EtxColumnSink sink{column_rows[r],r,column_layer,column_token,column_role,37};
    HIP_CHECK(hipMalloc((void**)&column_sinks[r],sizeof(sink)));HIP_CHECK(hipMemcpy(column_sinks[r],&sink,sizeof(sink),hipMemcpyHostToDevice));
  }
'''
HOST_EXPORT=r'''
  if(column_dir){
    for(int r=0;r<R;++r){const Req& q=reqs[r];
      if(q.matched!=32 || q.compared!=32 || q.warm_matched!=32 || q.warm_compared!=32 || q.layers_ok!=27)
        die("native column capture refuses export: per-request golden gates failed");
    }
    for(int r=0;r<R;++r){
      std::vector<EtxColumnRow> rows(8*37);HIP_CHECK(hipMemcpy(rows.data(),column_rows[r],rows.size()*sizeof(EtxColumnRow),hipMemcpyDeviceToHost));
      for(int x=0;x<8;++x)for(int w0=0;w0<37;++w0){auto& row=rows[x*37+w0];row.h[31]=column_clock;if(row.h[15]<(uint64_t)blob || row.h[15]+row.h[16]>(uint64_t)blob+blob_bytes)die("native capture weight pointer outside original blob");row.h[30]=row.h[15]-(uint64_t)blob;
        if(row.h[0]!=0x455458434f4c3031ull || row.h[2]!=(unsigned)r || row.h[7]!=(unsigned)x || row.h[8]!=(unsigned)w0 ||
           row.h[1]!=row.h[23] || row.h[22]!=4 || row.h[25] || row.h[10]>row.h[11])die("native column capture incomplete/duplicate/stale");
      }
      const std::string stem=std::string(column_dir)+"/native_r"+std::to_string(r);
      std::ofstream f(stem+".bin",std::ios::binary);f.write((const char*)rows.data(),rows.size()*sizeof(EtxColumnRow));f.close();if(!f)die("native capture raw export failed");
      // Weights remain in the sealed original checkpoint; every native pointer
      // is checked above and recorded as a byte offset. No duplicate weights.
      HIP_CHECK(hipFree(column_rows[r]));HIP_CHECK(hipFree(column_sinks[r]));
    }
    std::printf("COLUMN CAPTURE COMPLETE role=%d layer=%d token=%d requests=%d tasks_per_request=296\n",column_role,column_layer,column_token,R);
  }
'''
def patch_host(text):
    text=once(text,'using namespace fleet;','#include "examples/experiment_program/columns/native_capture.h"\nusing namespace fleet;')
    text=once(text,'struct FleetParams {','struct FleetParams {\n  EtxColumnSink* column_capture = nullptr;')
    marker='  double wall_total = 0; std::vector<double> step_ms;'
    text=once(text,marker,HOST_ALLOC+'\n'+marker)
    marker='        fp.tok = t; fp.epoch = (uint32_t)(run * tokens + t + 1); fp.pos = context + t; fp.dump = (last && t == 0) ? 1 : 0;'
    text=once(text,marker,marker+'\n        fp.column_capture=column_dir && last?column_sinks[r]:nullptr;')
    text=once(text,'  return all_ok ? 0 : 3;',HOST_EXPORT+'\n  return all_ok ? 0 : 3;')
    return text

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fleet-src',type=Path,required=True)
    p.add_argument('--shim',type=Path,required=True);p.add_argument('--host',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,choices=(2,4),default=2)
    a=p.parse_args(argv);require(not a.out.exists(),'capture patch destination must be new');a.out.mkdir(parents=True)
    inputs={'kernel':a.fleet_src/'kernels/fleet_kernel.hip','shim':a.shim,'host':a.host}
    result={}
    for name,path in inputs.items():
        text=path.read_text();new={'kernel':patch_kernel,'shim':patch_shim,'host':patch_host}[name](text)
        dest=a.out/(name+'.hip');dest.write_text(new);result[name]={'input':str(path.resolve()),'input_sha256':sha256(path),'output':str(dest),'output_sha256':sha256(dest)}
    graph=a.out/'model.py'
    graph.write_text('from examples.dsv2lite.graph import bindings,runtime\nfrom examples.experiment_program.columns.integration import frozen_graph\n'
                     f'def build():\n    g=frozen_graph({a.batch})\n    for grid in g.grids:\n        if grid.body.source=="examples/dsv2lite/fleet_shim.hip":grid.body.source={str((a.out/"shim.hip").resolve())!r}\n    return g\n')
    result['native_graph']={'path':str(graph),'sha256':sha256(graph),'batch':a.batch,'kind':'original native graph with capture-bound shim; no arithmetic intervention'}
    (a.out/'identity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PATCHED_NOT_EXECUTED','out':str(a.out)}))
if __name__=='__main__':main()
