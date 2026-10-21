"""G11 only: current down -> next QKV subgraph, unchanged arithmetic/queues.

Publication issue/forced-retirement bounds and all consumer landmarks share
s_memrealtime in one invocation. Exposure is never removable work or speedup.
"""
import json,re,shutil
from pathlib import Path
from .protocol import require,resolve_path,sha,verify_files
LAYERS=(1,24,47)
PHASES=('residual_read','slot_read','ordered_slot_fold','residual_round_fold','norm')
MODES=('control','probe','empty')
HELPERS=r'''
#if defined(ETX_INBODY) && ETX_INBODY && !G11_CONTROL
#include "etx/inbody.h"
__shared__ uint64_t* g11_phase;
static __device__ __forceinline__ void g11_mark(int point) {
  if (g11_phase && etx_lane()==0) {
    uint64_t* row=g11_phase+etx_wave()*ETX_LOAD_RECORD_WORDS;
    if(row[0]!=5)return; // host selects exactly one actual consumer wave
    unsigned hw; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
    row[3+(point/2)*3]=((uint64_t)etx_amdgcn_xcc_id()<<32)|hw;
#if G11_EMPTY
    if (!(point&1)) { row[1+(point/2)*3]=etx_inbody_tick(); row[2+(point/2)*3]=etx_inbody_tick(); }
#else
    row[1+(point/2)*3+(point&1)]=etx_inbody_tick();
#endif
  }
}
static __device__ __forceinline__ void g11_fence_z(float (&z)[8]) {
  asm volatile("" : "+v"(z[0]),"+v"(z[1]),"+v"(z[2]),"+v"(z[3]),"+v"(z[4]),"+v"(z[5]),"+v"(z[6]),"+v"(z[7]) :: "memory");
}
static __device__ __forceinline__ void g11_fence_ri(dsl_u32x4 ri) {
  asm volatile("" :: "v"(ri[0]),"v"(ri[1]),"v"(ri[2]),"v"(ri[3]) : "memory");
}
#else
static __device__ __forceinline__ void g11_fence_z(float (&)[8]) {}
static __device__ __forceinline__ void g11_fence_ri(dsl_u32x4) {}
static __device__ __forceinline__ void g11_mark(int) {}
#endif
'''

def tiles(original):
    """Instrument only retained RBATCH norm; never alter math or stage lifetime."""
    require(original.count('template <int FixedK = 0, int FixedR = -1>')==1,'current norm source drift')
    s=original.replace('template <int FixedK = 0, int FixedR = -1>',HELPERS+'\ntemplate <int FixedK = 0, int FixedR = -1>',1)
    start=s.index('template <int FixedK = 0, int FixedR = -1>');end=s.index('static __device__ __forceinline__ void dsl_stage_norm(',start);body=s[start:end]
    def edit(a,b):
        nonlocal body
        require(body.count(a)==1,'G11 norm landmark anchor drift: '+a);body=body.replace(a,b)
    edit('    const dsl_u32x4 ri = dsl_a_ld128(rin + 8 * i);\n    for (int s0', '    g11_mark(0); // residual read issue\n    const dsl_u32x4 ri = dsl_a_ld128(rin + 8 * i);\n#if ETX_INBODY && !G11_CONTROL\n    asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // LABELED forced-read-drain diagnostic\n#endif\n    g11_fence_ri(ri); g11_mark(1);\n    for (int s0')
    edit('      dsl_u32x4 rv[DSL_RB];','      g11_mark(2); // slot reads issue\n      dsl_u32x4 rv[DSL_RB];')
    anchor='#pragma unroll\n      for (int u = 0; u < DSL_RB; ++u) if (s0 + u < R) {'
    edit(anchor,'#if ETX_INBODY && !G11_CONTROL\n      asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // LABELED forced-read-drain diagnostic\n#endif\n      g11_mark(3); g11_mark(4); g11_fence_z(z); // disjoint slot-read end / fold begin\n'+anchor)
    edit('    dsl_u32x4 ro;','    g11_fence_z(z); g11_mark(5); g11_mark(6); g11_fence_z(z); // ordered slot fold end / residual fold begin\n    dsl_u32x4 ro;')
    require(body.count('#ifdef ETX_DSL_NORM_HOIST')==2,'current two norm-hoist anchors')
    body=body.replace('#ifdef ETX_DSL_NORM_HOIST','  g11_mark(7); g11_mark(8); // residual/round/store end / norm begin\n#ifdef ETX_DSL_NORM_HOIST',1)
    edit('  __syncthreads();\n}','  __syncthreads();\n  g11_mark(9); // norm-ready after retained norm barrier\n}')
    s=s[:start]+body+s[end:]
    anchor='  dsl_run_dsl_gemv_fold(etx_imp_argblock(c->args, c->cst[0]), c->coord[0]);'
    require(s.count(anchor)==1,'current QKV entry drift')
    s=s.replace(anchor,'''#if ETX_INBODY && !G11_CONTROL
  if(threadIdx.x==0)g11_phase=c->inbody_load;
  __syncthreads(); // LABELED diagnostic pointer setup; never aliases norm scratch
  // Selection sentinel is initialized by the host, never broadened here.
#endif
'''+anchor)
    return s


def extract(plan,layer):
    """Bind all original producers/consumers, workers and restricted queue edges."""
    require(layer in LAYERS,'G11 representative layer');names={v['id']:n for n,v in plan['types'].items()}
    consumers=[d for d in plan['descs'] if names[d['type']]==f'qkv_{layer}'];events={e for d in consumers for e in plan['task_in'][d['id']]}
    require(len(consumers)==256 and len(events)==1,'current complete QKV/event population')
    event=next(iter(events));producer_ids=plan['ev_inst'][event][2];producers=[plan['descs'][i] for i in producer_ids]
    require(len(producers)==256 and len(set(producer_ids))==256 and all(names[d['type']]==f'moe_down_{layer-1}' for d in producers),'current complete final predecessor population')
    require(plan['workers_per_domain']==38 and plan['domains']==8,'current worker geometry')
    ids={d['id'] for d in producers+consumers};queues={k:[i for i in q if i in ids] for k,q in plan['static_queues'].items()};queues={k:q for k,q in queues.items() if q}
    for q in queues.values():require(all(i in producer_ids for i in q[:sum(i in producer_ids for i in q)]),'restricted queue can block producer')
    return dict(layer=layer,producers=producers,consumers=consumers,event=plan['ev_inst'][event],queues=queues,source_plan_event=event,removed_prefix='original independently captured operands; not a timing/availability claim about removed tasks')


def graph(tile,context):
    from etx.ir import Graph,Resource
    from etx.frontends import hip_link
    g=Graph('g11_current_down_qkv');g.codegen.update(coh=True,nb=1,requests=1,fold_topk=False,dsl_bigk=False,dsl_ext=False);g.tensor('down_args',(1280,),role='runtime',bytes_per_elem=1);g.tensor('fold_args',(160,),role='runtime',bytes_per_elem=1);g.etensor('publication',(1,),wait_count=256)
    down=context['producers'];fold=context['consumers'];do=[next(d['worker']%38 for d in down if d['domain']==x) for x in range(8)];co=[next(d['worker']%38 for d in fold if d['domain']==x) for x in range(8)]
    packed=lambda offsets:sum(v*64**x for x,v in enumerate(offsets))
    require(all(d['domain']==d['coord'][0] and d['worker']==38*d['domain']+(d['coord'][1]+do[d['domain']])%38 for d in down),'current down worker map changed')
    require(all(d['domain']==d['coord'][0]//32 and d['worker']==38*d['domain']+(d['coord'][0]%32+co[d['domain']])%38 for d in fold),'current fold worker map changed')
    r=Resource(threads=1024,vgpr=128,agpr=0,lds_bytes=8448)
    link=lambda symbol,pre,touch=None:hip_link(str(tile),symbol,pre_wait=pre,touch=touch,coherence='coherent',coherence_caps=('coherent_stores',),arg_bytes=160)
    g.call_device('down',(8,32),link('etx_tile_x_dsl_moe_down_fold','etx_pre_x_dsl_moe_down_fold'),resource=r,args=['down_args'],consts=(0,160,32,0),domain_map='xw->x',worker_map=f'xw->((w+(({packed(do)}/(64**x))%64))%38)',out_edges={'publication':'xw->(0)'})
    g.call_device('fold',(256,),link('etx_tile_dsl_gemv_fold','etx_pre_dsl_gemv_fold','etx_touch_dsl_gemv_fold'),resource=r,args=['fold_args'],consts=(0,),domain_map='i->(i/32)',worker_map=f'i->((i%32+(({packed(co)}/(64**(i/32)))%64))%38)',in_edges={'publication':'i->(0)'})
    return g


def prepare(native,out):
    from etx.passes import compile_graph,PassOptions
    from etx.codegen import emit_kernel,emit_plan_json,emit_lowering_header
    from etx.codegen.plan_header import emit_plan_header
    from etx.codegen.inbody import instrument
    from .build_support import ROOT
    native=resolve_path(native);out=resolve_path(out);require(not out.exists(),'preserve current subgraph attempt');require(shutil.disk_usage(out.parent).free>(2<<30)+(8<<20),'G11 serial subgraph requires8MiB emission budget above2GiB')
    out.mkdir();p=json.loads((native/'plan.json').read_text());old=json.loads((native/'build-identity.json').read_text());verify_files(old['hashes'])
    original=(native/'dsl_tiles.hip').read_text();(out/'tiles.hip').write_text(tiles(original));contexts={str(l):extract(p,l) for l in LAYERS}
    plans={}
    for layer,context in contexts.items():
        plan=compile_graph(graph(out/'tiles.hip',context),'gfx942',{}, {},PassOptions(inline_tiles=True,relay='off',prefetch=False,event_elimination=False))
        raw=json.loads(emit_plan_json(plan));require(len(raw['descs'])==512 and raw['workers_per_domain']==38,'current subgraph task/worker budget')
        # The source ABI and task coordinate/type rules are identical in all six
        # shape cases; host-loaded queues retain each layer's original pins.
        for d in raw['descs']:
            original_rows=context['producers'] if d['type']==0 else context['consumers'];match=[o for o in original_rows if o['coord']==d['coord']]
            require(len(match)==1 and (d['worker'],d['domain'])==(match[0]['worker'],match[0]['domain']),'emitted subgraph original pin mismatch')
        (out/f'plan-L{layer}.json').write_text(emit_plan_json(plan));(out/f'plan-L{layer}.h').write_text(emit_plan_header(plan,0));plans[layer]=raw
        if layer=='1':
            (out/'etx_lowering.h').write_text(emit_lowering_header(plan.machine));code=emit_kernel(plan,0)
            code=code.replace('ctx.cst[0] = ctx.cst[1] = ctx.cst[2] = ctx.cst[3] = 0;', 'ctx.cst[0] = ctx.cst[1] = ctx.cst[2] = ctx.cst[3] = 0;\n#if ETX_INBODY\n  ctx.inbody_load=nullptr;\n#endif');placement='''
#if ETX_INBODY
  if(p.inbody_wave && etx_lane()==0) {
    unsigned hw;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
    p.inbody_wave[((size_t)tid*16+etx_wave())*3+2]=((uint64_t)etx_amdgcn_xcc_id()<<32)|hw;
  }
#endif
'''
            anchor='  etx_call_body(sym, &ctx);   // phase 2: the only call site of each body'
            require(code.count(anchor)==1,'G11 placement boundary drift');code=code.replace(anchor,anchor+placement)
            (out/'control.hip').write_text(code)
            probe=instrument(code).replace('threadIdx.x / ETX_WAVE_SIZE) * 2;', 'threadIdx.x / ETX_WAVE_SIZE) * 3;')
            (out/'probe.hip').write_text(probe);(out/'empty.hip').write_text(probe)
    private=out/'include/etx';private.mkdir(parents=True);s=(ROOT/'etx/runtime/include/etx/inbody.h').read_text();anchor='  uint64_t e = (p).inbody_event ? etx_inbody_tick() : 0; \\'
    require(s.count(anchor)==1,'original publication anchor drift')
    s=s.replace(anchor,'  asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); /* LABELED original atomic forced retirement bound */ \\\n'+anchor)
    (private/'inbody.h').write_text(s)
    contract=dict(schema='etx.g11.subgraph.preparation.v1',native=str(native),current_native_identity_sha256=sha(native/'build-identity.json'),source_plan_sha256=sha(native/'plan.json'),original_dsl_source_sha256=sha(native/'dsl_tiles.hip'),contexts=contexts,plans=plans,modes=list(MODES),coverage=dict(producer_tasks=256,consumer_tasks=256,waves_per_task=16,layers=list(LAYERS),positions=[0,1040],sparse_samples_per_case=256*16,phase_names=list(PHASES),active_read_fold_waves=[0,1,2,3],all_wave_norm_and_body=True),clock='same invocation s_memrealtime; actual native frequency required',publication='original final-arrival issue/forced-retirement bounds; consumers never wait for stamps',phase_semantics='disjoint residual-read,slot-read,ordered-slot-fold,residual-round-fold,norm; inactive operand waves explicitly N/A',forced_drains='read completion and original atomic retirement are labeled diagnostic; no removable work/speedup',host_ready=False,qualified=False,gpu_executed=False,flags=old['flags'],files={str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'preparation.json').write_text(json.dumps(contract,indent=2)+'\n');return contract


def resource_gate(profiles,physical):
    """Match actual descriptor allocations; retain compiler spill counts as evidence."""
    keys=('agpr','scratch_bytes_lane')
    require(set(profiles)==set(physical)==set(MODES),'complete control/probe/empty resource population')
    equal=all(all(profiles[m][k]==profiles['control'][k] for k in keys) and physical[m]==physical['control'] for m in MODES)
    equal=equal and all(profiles[m]['static_blocks_per_cu']>=1 for m in MODES)
    return dict(status='RESOURCE_MATCHED' if equal else 'CLOSED_NATIVE_DIAGNOSTIC_RESOURCE',resources_equal=equal,admitted=equal,profiles=profiles,physical_resources=physical,gpu_qualification='UNMEASURED',exact_atomic_point_required=False,whole_model_instrumented_build_required=False,owner_protocol_change_required=False)


def admission(identity):
    """A shape-specific path must prove coverage/resources before any execution."""
    from examples.vllm_llm.bf16_300.build import resources
    from .row_ops.elf import descriptors,allocation_signature
    p=resolve_path(identity);c=json.loads(p.read_text());verify_files(c['files'])
    require(c['schema']=='etx.g11.subgraph.preparation.v1','canonical current-subgraph identity required')
    native=resolve_path(c['native']);require(sha(native/'build-identity.json')==c['current_native_identity_sha256'] and sha(native/'plan.json')==c['source_plan_sha256'] and sha(native/'dsl_tiles.hip')==c['original_dsl_source_sha256'],'current subgraph source identity changed')
    require((p.parent/'tiles.hip').read_text()==tiles((native/'dsl_tiles.hip').read_text()),'G11 phase source differs from declared original')
    plan=json.loads((native/'plan.json').read_text());require(c['contexts']=={str(l):extract(plan,l) for l in LAYERS},'complete actual producer/consumer/queue coverage changed')
    expected=dict(producer_tasks=256,consumer_tasks=256,waves_per_task=16,layers=list(LAYERS),positions=[0,1040],sparse_samples_per_case=4096,phase_names=list(PHASES),active_read_fold_waves=[0,1,2,3],all_wave_norm_and_body=True)
    require(c['coverage']==expected and c['modes']==list(MODES),'complete declared G11 cases/task/wave/matched-mode population required')
    require(c['flags']==json.loads((native/'build-identity.json').read_text())['flags'],'current math/compiler flags differ')
    if 'path' in c['compiler']:verify_files({c['compiler']['path']:c['compiler']['sha256']})
    for layer in LAYERS:require(c['plans'][str(layer)]==json.loads((p.parent/f'plan-L{layer}.json').read_text()),'selected layer plan differs from source-bound preparation')
    actual={};physical={}
    for mode in MODES:
        require('#define ETX_DSL_COH 1' in (p.parent/(mode+'.hip')).read_text(),'current coherent feature manifest absent')
        rs=resources((p.parent/(mode+'.s')).read_text());r=next(x for x in rs if x['kernel']=='etx_megakernel_d0')
        ds=descriptors(p.parent/(mode+'.hsaco'));actual[mode]=r;physical[mode]=allocation_signature(ds[r['kernel']],r['agpr'])
        require(rs==c['resources'][mode] and physical[mode]==c['physical_resources'][mode][r['kernel']],'actual native resource metadata changed')
    if c.get('raw_build_identity'):
        from .match_diagnostics import text_bytes
        raw=resolve_path(c['raw_build_identity']['path']);require(sha(raw)==c['raw_build_identity']['sha256'],'raw compiler identity changed');verify_files(json.loads(raw.read_text())['files'])
        for mode in MODES:require(text_bytes(raw.parent/(mode+'.hsaco'))==text_bytes(p.parent/(mode+'.hsaco')),'descriptor reserve changed instructions')
    gate=resource_gate(actual,physical)
    from .publication_run import verify_interface
    interface=verify_interface(c) if gate['resources_equal'] and c.get('interface') else None
    gate.update(scope='CURRENT_DOWN_QKV_SUBGRAPH',coverage=expected,local_resource_admitted=gate['admitted'],admitted=bool(interface),exact_publication_timestamp_available=False,publication_measurement='same-invocation original atomic issue/forced-retirement bounds and disjoint consumer phase marks',remaining_native_gates=['exact outputs/intermediates','all512 task/all16wave and final-consumer coverage','native placement','complete4096 sparse selectors per each of6 cases','<=0.5us perturbation','same active G12 trace/fold decision'],gpu_executed=False,qualified=False,runnable_interface_prepared=bool(interface))
    if gate['resources_equal']:gate['status']='PREPARED_REQUIRES_NATIVE_QUALIFICATION' if interface else 'INCOMPLETE_SUBGRAPH_RUNNER_AND_RAW_ADMISSION'
    return gate
