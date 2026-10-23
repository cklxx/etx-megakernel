"""Bounded offline installed-package checks, without repository example imports."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from etx.api.types import Configuration


_BOOTSTRAP = '''import runpy,sys
from pathlib import Path
installed=Path(sys.argv[1]).resolve()
workflow=sys.argv[2]
# Declarations may come only from the repository the workflow names explicitly (argv[4]), never ambiently.
declared=Path(sys.argv[4]).resolve() if len(sys.argv)>4 and sys.argv[4] else None
sys.argv=[workflow,*sys.argv[3:]]
sys.path.insert(0,str(installed))
runpy.run_path(workflow,run_name='__main__')
for name,module in tuple(sys.modules.items()):
    if name == 'examples' or name.startswith('examples.'):
        path=getattr(module,'__file__',None) or next(iter(getattr(module,'__path__',[])),None)
        assert declared is not None and path is not None and Path(path).resolve().is_relative_to(declared/'examples'), (name,path)
for name,module in tuple(sys.modules.items()):
    if name == 'etx' or name.startswith('etx.'):
        path=getattr(module,'__file__',None)
        if path is not None:
            assert Path(path).resolve().is_relative_to(installed), (name,path)
'''


_PROBE = '''import json,os,sys
from pathlib import Path
assert not any(key.startswith(('ETX_', 'PYTHON')) for key in os.environ)
from etx.external import PlanRequest, plan, simulate, emit, save_sources, load_sources, SealedSessionRequest, run_sealed_session
from etx.testing import fixture, sealed_original_host
rows=[]
for devices in (1,2,8):
    for batch in (1,2,4,8):
        planned=plan(PlanRequest(fixture('handoff',devices=devices,batch=batch)))
        result=simulate(planned,seed=17)
        assert not result.deadlock
        source=emit(planned)
        directory=Path(sys.argv[1])/f'sources-{devices}-{batch}'
        save_sources(source,directory)
        loaded=load_sources(directory)
        assert loaded.identity==source.identity and len(loaded.rank_sources)==devices
        rows.append(dict(devices=devices,batch=batch,status=loaded.status))
from etx.external import KernelSpec, kernel
operators=[]
for name,small in (('splitk_sum',False),('moe_layer',False),('moe_layer',True)):
    for size in (1,2,4,8):
        config=dict(provider=name)
        if name=='moe_layer':
            config['small']=small
        workload=kernel(KernelSpec(name,'graph',config))
        planned=plan(PlanRequest(workload,bindings={'n' if name=='splitk_sum' else 'B':size}))
        assert not simulate(planned,seed=17).deadlock
        source=emit(planned)
        assert source.status=='SOURCES' and not source.external_sources
        native=json.loads(source.files['etx_native_inputs.json'])
        tile='examples/tiles/'+('splitk.hip' if name=='splitk_sum' else 'moe.hip')
        defines={} if name=='splitk_sum' else (dict(D=256,H=2,S=128,FF=128) if small else dict(D=1024,H=8,S=512,FF=512))
        assert native==dict(schema_version=1,kind='native-tile-inputs',units=[dict(source=tile,language='hip',defines=defines)])
        assert source.files[tile] and source.files[tile.replace('/tiles/','/hosts/')]
        assert source.files['include/etx/abi.h'] and source.files['include/etx/tile_util.h']
        directory=Path(sys.argv[1])/f'operator-{name}-{small}-{size}'
        save_sources(source,directory)
        assert load_sources(directory)==source
        operators.append(dict(name=name,small=small,size=size,status=source.status))
gemm=emit(plan(PlanRequest(kernel(KernelSpec('gemm','graph',dict(provider='gemm_rs'))),bindings=dict(M=256,K=64,N=128))))
assert not gemm.external_sources and len(gemm.rank_sources)==2
assert gemm.files['examples/tiles/gemm_rs.hip'] and gemm.files['examples/hosts/gemm_rs.hip']
save_sources(gemm,Path(sys.argv[1])/'gemm')
assert load_sources(Path(sys.argv[1])/'gemm')==gemm
assert sealed_original_host()['historical_only']
catalog=run_sealed_session(SealedSessionRequest('list'))['records']
assert len(catalog)==13 and all(not row['enabled'] for row in catalog)
from etx.external import ProcessRequest, execute, read_process_result
from etx.external import run_protocol_experiment, report_protocol_experiment, run_protocol_study
import contextlib, shutil
work=Path(sys.argv[1])
request=ProcessRequest((sys.executable,'-I','-c','print("installed CPU child")'),str(work),5.0)
with contextlib.redirect_stdout(sys.stderr):
    process=execute(request,output=work/'process')
assert process.status=='complete' and process.backend=='host-process'
shutil.move(work/'process',work/'relocated-process')
assert read_process_result(work/'relocated-process',request=request)==process
assert (work/'relocated-process'/'output.log').read_text()=='installed CPU child\\n'
config=dict(schema_version=1,arch='gfx942',rounds=2,seed=7,cases=[
    dict(name=name,workload=name,size=2,duration_cv=0.1) for name in ('splitk','moe')])
with contextlib.redirect_stdout(sys.stderr):
    assert run_protocol_experiment(config,output=work/'protocol')==0
    summary=(work/'protocol'/'summary.json').read_bytes()
    assert report_protocol_experiment(work/'protocol')==0
assert (work/'protocol'/'summary.json').read_bytes()==summary
manifest=json.loads((work/'protocol'/'manifest.json').read_text())
assert not manifest['gpu_measurement'] and manifest['tensor_correctness']=='not_executed'
assert manifest['provenance']['commit'] is None
studies={name:run_protocol_study(name,size=2).to_dict() for name in ('variance','routing-skew')}
assert all(value['backend']=='simulation' and not value['gpu_measurement'] for value in studies.values())
assert all(len(value['rows'])==5 for value in studies.values())
from etx.external import JobSpec, JobSession, JobSessionRequest, run_jobs, verify_job_record
session=JobSession((JobSpec('installed-cpu','echo installed job',lane='cpu',cwd=str(work)),))
with contextlib.redirect_stdout(sys.stderr):
    jobs=run_jobs(JobSessionRequest(session,logdir=str(work/'jobs'),gpus=(),poll_s=.01))
assert jobs.returncode==0 and verify_job_record(jobs.record_directory)['returncode']==0
assert 'installed job' in Path(jobs.jobs[0].log).read_text()
from etx.external import SessionTransportRequest, run_session_transport
# Transport status against this machine as the box: ssh runs its remote command locally.
stub=work/'transport-bin'; stub.mkdir()
(stub/'ssh').write_text('#!/bin/bash\\nshift 5; exec bash -c "$*"\\n'); (stub/'ssh').chmod(0o755)
(work/'box/sess').mkdir(parents=True)
saved=dict(os.environ); os.environ.update(PATH=str(stub)+':'+os.environ['PATH'],HOME=str(work/'box'))
out=os.dup(1); capture=os.open(work/'transport-status.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL)
try:
    os.dup2(capture,1); status=run_session_transport(SessionTransportRequest(('status','localhost'),str(work)))
finally:
    os.dup2(out,1); os.close(capture); os.environ.clear(); os.environ.update(saved)
assert status==0 and (work/'transport-status.log').read_text().splitlines()[0]=='runner not running'
import hashlib
root=Path(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2] else None
fleet=sys.argv[3] if len(sys.argv)>3 and sys.argv[3] else None
s5={}
if root is not None and fleet is not None:
    from etx.external import ModelBuildRequest, PlanRequest, build_model, model, ExperimentSessionRequest, run_experiment_session
    # Recording compiler: the build's argv and the files it saw; no device code is produced.
    fake=work/'s5-bin'; fake.mkdir()
    (fake/'hipcc').write_text('#!/bin/bash\\nprintf "%s\\\\n" "$@" > "$(dirname "$0")/argv"\\nout=; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done\\nprintf "#!/bin/sh\\\\n" > "$out"; chmod +x "$out"\\n')
    (fake/'hipcc').chmod(0o755)
    graph=root/'examples/dsv2lite/graph.py'
    request=ModelBuildRequest(PlanRequest(model('dsv2lite',config=dict(declaration=str(graph),declaration_sha256=hashlib.sha256(graph.read_bytes()).hexdigest()))),
                              str(root),str(work/'s5-build'),str(root/'examples/dsv2lite/host.hip'),provider='dsv2lite',variant=dict(fleet=fleet))
    saved=dict(os.environ); os.environ['PATH']=str(fake)+':'+os.environ['PATH']
    try:
        with contextlib.redirect_stdout(sys.stderr):
            built=build_model(request)
    finally:
        os.environ.clear(); os.environ.update(saved)
    steps=[step['step'] for step in built['steps']]
    assert built['status']=='EXECUTABLE' and steps==['plan','coh-check','link'], (built['status'],steps)
    argv=(fake/'argv').read_text().splitlines()
    assert argv[:5]==['-O3','-std=c++17','--offload-arch=gfx942','-DFLEET_NT_WEIGHTS=1','-I'] and argv[5]==fleet+'/src', argv[:6]
    s5['dsv2lite_build']=True
    logs=work/'s5-session'; logs.mkdir()
    (logs/'pairs_all.json').write_text(json.dumps(dict(device_ms=dict(default=dict(median_us=-120.0,spread_us=4.0)),verdict=dict(outcome='SYNTHETIC'))))
    with contextlib.redirect_stdout(sys.stderr):
        run_experiment_session(ExperimentSessionRequest('dsv2lite','report',str(logs),str(root),dict(fleet=fleet,data=str(work/'no-data'))))
    report=json.loads((logs/'report.json').read_text())
    assert report['all']['verdict']['outcome']=='SYNTHETIC' and report['stages'] and all('measured_saving_us' not in row for row in report['stages'])
    s5['dsv2lite_report']=True
if root is not None:
    from etx.external import NativeProbeRequest, run_native_probe
    # Calibration with a recording compiler whose "benches" print a line: header, compile, run and tee steps.
    fake=work/'s9-bin'; fake.mkdir()
    (fake/'hipcc').write_text('#!/bin/bash\\nout=; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done\\nprintf "#!/bin/sh\\\\necho calibrated\\\\n" > "$out"; chmod +x "$out"\\n')
    (fake/'hipcc').chmod(0o755)
    saved=dict(os.environ); os.environ['PATH']=str(fake)+':'+os.environ['PATH']
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code=run_native_probe(NativeProbeRequest('calibration','run',str(root),dict(output=str(work/'calib'))))
    finally:
        os.environ.clear(); os.environ.update(saved)
    assert code==0 and (work/'calib/etx_lowering.h').is_file()
    assert all((work/'calib'/(b+'.txt')).read_text()=='calibrated\\n' for b in ('atomic_pingpong','flag_latency','queue_contention'))
    s5['calibration_run']=True
    import io
    plan_text=io.StringIO()
    with contextlib.redirect_stdout(plan_text):
        assert run_native_probe(NativeProbeRequest('fresh-handoff','plan',str(root)))==0
    # The frozen plan with no case evidence never passes the mechanism gate.
    (work/'no-run').mkdir(); (work/'no-run/manifest.json').write_text(json.dumps(dict(plan=json.loads(plan_text.getvalue()))))
    report=io.StringIO()
    with contextlib.redirect_stdout(report):
        assert run_native_probe(NativeProbeRequest('fresh-handoff','analyze',str(root),dict(directory=str(work/'no-run'))))==1
    analysis=json.loads(report.getvalue())
    assert analysis['mechanism_gate']=='not_passed' and all(case['state']=='fail' for case in analysis['cases'])
    s5['fresh_handoff_check']=True
if root is not None:
    # Legacy LLM: build through the host-link provider with a recording compiler, and pack a tiny synthetic checkpoint.
    from etx.external import ModelBuildRequest, PlanRequest, build_model, model, AssetPreparationRequest, prepare_model_assets
    fake=work/'s3-bin'; fake.mkdir()
    (fake/'hipcc').write_text('#!/bin/bash\\nout=; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done\\nprintf "#!/bin/sh\\\\n" > "$out"; chmod +x "$out"\\n')
    (fake/'hipcc').chmod(0o755)
    declaration=root/'examples/llm/model.py'
    workload=model('llm',config=dict(declaration=str(declaration),declaration_sha256=hashlib.sha256(declaration.read_bytes()).hexdigest(),
                                     environment={'ETX_LLM_CONFIG':str(root/'examples/llm/configs/phi-synth-a.json')}))
    saved=dict(os.environ); os.environ['PATH']=str(fake)+':'+os.environ['PATH']
    try:
        built=build_model(ModelBuildRequest(PlanRequest(workload,options=dict(inline_tiles=True)),str(root),str(work/'s3-build'),
                                            str(root/'examples/llm/host.hip'),('-O3',),(str(root/'etx/runtime/include'),str(work/'s3-build'),str(root)),True))
    finally:
        os.environ.clear(); os.environ.update(saved)
    assert built['status']=='EXECUTABLE' and [step['step'] for step in built['steps']]==['plan','link']
    s5['llm_build']=True
    from etx.testing import synthetic_llm_checkpoint
    src=synthetic_llm_checkpoint(work/'s3-src',root)
    packed=work/'s3-packed'; packed.mkdir()
    # An existing golden file: no Hugging Face/GPU reference generation.
    (packed/'golden.txt').write_text('prompt: 1 2 3\\ngen: 4 5\\nmargin: 0.1 0.2\\n')
    with contextlib.redirect_stdout(sys.stderr):
        prepare_model_assets(AssetPreparationRequest('llm',str(root),dict(model='synthetic/tiny',out=str(packed),gen=32,prompt='x',src=str(src),
                             context=0,sliced=True,rotate=0,golden_only=False,prefill_only=False)))
    manifest=(packed/'llm.manifest').read_text()
    assert 'cfg layout 2' in manifest and (packed/'weights.bin').stat().st_size>0 and (packed/'sliced.manifest').is_file()
    s5['llm_prep']=True
if root is not None:
    # Qwen expert-merge harness: its CPU protocol and arithmetic commands, and a fail-closed status.
    from etx.external import HarnessCommandRequest, run_harness_command
    harness=lambda module,*argv: run_harness_command(HarnessCommandRequest('qwen-expert-merge',module,tuple(argv),str(root)))
    # (The `cpu` arithmetic command fails to compile on glibc systems that declare rsqrtf: a known, unchanged defect.)
    with contextlib.redirect_stdout(sys.stderr):
        assert harness('protocol','--tokens','16')==0
        assert harness('runner','status',str(work/'em-session'))==2
    s5['expert_merge_protocol']=True; s5['expert_merge_status']=True
    attention=lambda module,*argv: run_harness_command(HarnessCommandRequest('qwen-attention-early-merge',module,tuple(argv),str(root)))
    with contextlib.redirect_stdout(sys.stderr):
        assert attention('simulate','--out',str(work/'aem-simulate.json'))==0
        assert attention('session','--work',str(work/'aem-work'),'--model',str(work/'no-model'),'status')==0
    assert json.loads((work/'aem-simulate.json').read_text())
    s5['attention_merge_simulate']=True; s5['attention_merge_status']=True
    bf16=lambda module,*argv: run_harness_command(HarnessCommandRequest('qwen-bf16-probe',module,tuple(argv),str(root)))
    with contextlib.redirect_stdout(sys.stderr):
        assert bf16('protocol')==0
        assert bf16('build')==0
    assert json.loads((root/'results_bf16_300_cpu_20261011/dry_run.json').read_text())
    s5['bf16_protocol']=True; s5['bf16_build_dry_run']=True
    # GLM seam: the graph declaration through the model planner, and the fail-closed receipt parser.
    seam_graph=root/'examples/glm53/seam_graph.py'
    seam=emit(plan(PlanRequest(model('glm53-seam',config=dict(declaration=str(seam_graph),
        declaration_sha256=hashlib.sha256(seam_graph.read_bytes()).hexdigest())),bindings=dict(B=2),options=dict(inline_tiles=True))))
    assert len(seam.rank_sources)==2 and seam.status=='SOURCES'
    with contextlib.redirect_stdout(sys.stderr):
        assert run_harness_command(HarnessCommandRequest('glm53-seam','seam',('summary',str(work/'absent.log')),str(root)))==2
    s5['glm_seam_plan']=True; s5['glm_seam_summary']=True
    # GLM boundary transport build with a recording compiler: plan both ranks, link with the boundary host.
    fake=work/'s2a-bin'; fake.mkdir()
    (fake/'hipcc').write_text('#!/bin/bash\\nout=; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done\\nprintf "#!/bin/sh\\\\n" > "$out"; chmod +x "$out"\\n')
    (fake/'hipcc').chmod(0o755)
    saved=dict(os.environ); os.environ['PATH']=str(fake)+':'+os.environ['PATH']
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code=run_harness_command(HarnessCommandRequest('glm53-boundary','boundary_build',(str(work/'s2a-build'),),str(root)))
    finally:
        os.environ.clear(); os.environ.update(saved)
    assert code==0 and (work/'s2a-build/run').is_file() and (work/'s2a-build/megakernel_d1.hip').is_file()
    s5['glm_boundary_build']=True
    # GLM full model: run_all plans every step and refuses missing inputs (exit 3); model_build refuses a missing stage.
    import io
    dry=io.StringIO()
    with contextlib.redirect_stdout(dry):
        code=run_harness_command(HarnessCommandRequest('glm53-run-all','run_all',('--dry-run','--model-root',str(work/'no-model'),'--out',str(work/'glm-out')),str(root)))
    assert code==3 and 'PLAN ' in dry.getvalue() and not (work/'glm-out').exists()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            refused=run_harness_command(HarnessCommandRequest('glm53-model-build','model_build',('--stage',str(work/'no-stage'),'--out',str(work/'glm-mb')),str(root)))
        except (ValueError, OSError):
            refused=1
    assert refused!=0 and not (work/'glm-mb').exists()
    s5['glm_run_all_dry_run']=True; s5['glm_model_build_refusal']=True
    # Qwen with vLLM kernels: decode's selection and step plan (dry run), and the build's refusal before importing.
    vllm=lambda module,*argv: run_harness_command(HarnessCommandRequest('qwen-vllm',module,tuple(argv),str(root)))
    selection=io.StringIO()
    with contextlib.redirect_stdout(selection):
        assert vllm('decode','qwen3-30b-a3b','--dry-run','--work',str(work/'qwen-work'))==0
    assert 'selection: budget' in selection.getvalue() and '[build] bash' in selection.getvalue()
    (work/'qwen-packed').mkdir(); (work/'qwen-packed/config.json').write_text((root/'examples/llm/configs/qwen3-30b-a3b.json').read_text())
    saved=dict(os.environ); os.environ['ETX_VL_BATCH']='1'; os.environ['ETX_OUT']=str(work/'qwen-build')
    try:
        assert vllm('vllm_build',str(work/'qwen-packed'))==1
    finally:
        os.environ.clear(); os.environ.update(saved)
    s5['vllm_decode_dry_run']=True; s5['vllm_build_refusal']=True
# Sealed session lifecycle with the public scripted stage backend: providers are not executed here.
from etx.testing import synthetic_sealed_stages
anchor=work/'sealed-input.txt'; anchor.write_text('Synthetic CPU operational fixture, not native input or performance evidence.\\n')
seal={str(anchor): hashlib.sha256(anchor.read_bytes()).hexdigest()}
command=[sys.executable,'-m','examples.experiment_program.row_ops','execute','--dry-run']
sealed_config=work/'sealed-session.json'
sealed_config.write_text(json.dumps(dict(base='b573664faa4e7988dff353bc03aca4ce8561a729',source_snapshot=seal,budget_seconds=10,
    records=dict(G09=dict(preflight=command,command=command,input_files=seal,timeout_seconds=2,decision_file='G09.json')))))
sealed_out=work/'sealed-out'
with contextlib.redirect_stdout(sys.stderr), synthetic_sealed_stages() as calls:
    run_sealed_session(SealedSessionRequest('preflight',config=str(sealed_config),output=str(sealed_out)))
    before=(sealed_out/'G09-preflight.log').read_bytes()
    run_sealed_session(SealedSessionRequest('preflight',config=str(sealed_config),output=str(sealed_out),resume=True))
assert len(calls)==1 and (sealed_out/'G09-preflight.log').read_bytes()==before
assert json.loads((sealed_out/'session.json').read_text())['stages']['G09']['status']=='PREFLIGHT_PASS'
exported=run_sealed_session(SealedSessionRequest('export',output=str(sealed_out)))
assert len(exported['sha256'])==64 and sealed_out.with_suffix('.tar').is_file()
s5['sealed_preflight_resume']=True; s5['sealed_export']=True
result=dict(**s5,rows=rows,operators=operators,gemm_ranks=len(gemm.rank_sources),catalog_records=len(catalog),historical_fixture=True,example_imports=False,relocated_process=True,
            protocol_cases=2,protocol_replay=True,studies=studies,job_session=True,job_record_verified=True,
            transport_status=True,device_qualification=False)
(work/'probe-result.json').write_text(json.dumps(result))
print(json.dumps(result))
'''


def installed_package_probe(output, *, declarations=None, fleet=None):
    """Build/install without dependency fetching, then run fixed external CPU probes.

    With `declarations` (a repository root) and `fleet` (a pinned fleet-mi300x export), the wheel also
    builds the DeepSeek port with a recording compiler and replays a session report from that tree."""
    root = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = output / 'project'
    project.mkdir()
    shutil.copy2(root / 'pyproject.toml', project / 'pyproject.toml')
    shutil.copytree(root / 'etx', project / 'etx',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('ETX_', 'PYTHON', 'PIP_'))}
    environment.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')

    def command(argv, name, cwd):
        log = output / (name + '.log')
        print('PACKAGE START', name, flush=True)
        with log.open('xb') as stream:
            result = subprocess.run(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=90)
        print('PACKAGE EXIT', name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; full log retained at {log}:\n{log.read_text()}')

    wheel_directory = output / 'wheel'
    command([sys.executable, '-m', 'build', '--wheel', '--no-isolation', '--outdir',
             str(wheel_directory)], 'build', project)
    wheels = list(wheel_directory.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('expected exactly one built wheel')
    with zipfile.ZipFile(wheels[0]) as archive:
        contents = sorted(archive.namelist())
    installed = output / 'installed'
    command([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target',
             str(installed), str(wheels[0])], 'install', output)
    from etx.testing.boundary import check_external_sources
    workflow_directory = output / 'workflow'
    workflow_directory.mkdir()
    workflow = workflow_directory / 'installed.py'
    workflow.write_text(_PROBE)
    violations = check_external_sources(workflow_directory)
    if violations:
        raise ValueError('installed workload violates external API policy: ' + '; '.join(violations))
    extra = [str(declarations) if declarations else '', str(fleet) if fleet else '']
    command([sys.executable, '-I', '-c', _BOOTSTRAP, str(installed), str(workflow),
             str(output), *extra], 'probe', output)
    return Configuration(dict(wheel_files=contents, probe=json.loads((output / 'probe-result.json').read_text()),
                              logs=[str(output / (name + '.log')) for name in ('build', 'install', 'probe')]))
