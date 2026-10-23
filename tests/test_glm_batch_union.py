"""CPU execution of portable batch tiles; no GPU or downloaded weights."""
import numpy as np
import pytest

from etx.passes import compile_graph, verify_plan
from examples.glm53 import graph_cpu, module_gate, stage
from examples.glm53.options import Options
from examples.glm53.batch_graph import BatchGraphCPU, build_from


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('batch-union')
    module_gate.checkpoint(root/'checkpoint', variant='module8')
    stage.stage(root/'checkpoint',root/'stage',maxlen=96,gate='module8')
    return root


@pytest.mark.parametrize('batch',[2,4])
def test_logits_state_and_union_ledger(fixture,tmp_path,batch):
    spec = stage.validate(fixture/'stage',allow_fixture=True,gate='module8')
    cpu = BatchGraphCPU(fixture/'stage',batch,'module8',tmp_path/'batch')
    singles = [graph_cpu.GraphCPU(fixture/'stage',Options(),1,'module8',tmp_path/f'row{r}',True)
               for r in range(batch)]
    plan = compile_graph(cpu.graph,'sm_90',{'B':batch},{})
    assert not verify_plan(plan)
    # Distinct histories, including DSA's index top-k pruning after position 64.
    rng = np.random.default_rng(231023)
    saved = 0
    for pos in range(68):
        tokens = rng.integers(0,cpu.d['VOCAB'],size=batch)
        if pos==0:
            tokens[:] = 3  # identical routes must amortize across all requests
        ids,logits = cpu.step(pos,tokens)
        expected = [single.step(pos,[int(token)]) for single,token in zip(singles,tokens)]
        np.testing.assert_array_equal(ids,np.concatenate([x[0] for x in expected]))
        np.testing.assert_array_equal(logits,np.concatenate([x[1] for x in expected]))
        assert np.isfinite(logits).all()
        union_reads = independent_reads = 0
        for i,kind in enumerate(spec['config']['mlp_layer_types']):
            if kind!='sparse':
                continue
            layer = f'l{i}'
            selections = cpu.buffers[layer+'_sel'].view(np.float32).reshape(batch,-1)[:,:cpu.d['TOPK']].astype(int)
            # Descriptor order is the native ABI: routed triples follow EXPERTS.
            from examples.glm53.parallel_layout import _ABI
            meta = cpu.buffers[layer+'_meta'].tobytes()
            desc = list(_ABI.iter_unpack(meta))
            def elements(start):
                return sum(desc[start+m][2]*desc[start+m][3] for m in range(3))
            shared = elements(stage.SLOTS.index('mlp.shared_experts.gate_proj.weight'))
            routed = len(stage.SLOTS)
            union_reads += shared + sum(elements(routed+3*e) for e in set(selections.flat))
            independent_reads += batch*shared + sum(elements(routed+3*e) for e in selections.flat)
        assert cpu.weight_reads == union_reads
        assert union_reads < independent_reads
        saved += independent_reads-union_reads
        for name,tensor in cpu.graph.tensors.items():
            if tensor.role=='runtime':
                rows = cpu.buffers[name].reshape(batch,-1)
                for r,single in enumerate(singles):
                    np.testing.assert_array_equal(rows[r],single.buffers[name],err_msg=name)
    assert saved>0
    before = {n:b.copy() for n,b in cpu.buffers.items()}
    for pos,tokens in [(-1,[1]*batch),(96,[1]*batch),(0,[1]),(0,[1.0]*batch),(0,[-1]*batch)]:
        with pytest.raises(ValueError):
            cpu.step(pos,tokens)
    for name,buf in before.items():
        np.testing.assert_array_equal(cpu.buffers[name],buf)


@pytest.mark.parametrize('batch',[0,3,8,True,2.0])
def test_invalid_batch(batch):
    with pytest.raises(ValueError,match='batch'):
        build_from({},batch)


@pytest.mark.parametrize('options',[Options(phased=0),Options(parallel=1),Options(parallel=1,attention_tp=1)])
def test_unsupported_placement(options):
    with pytest.raises(ValueError,match='phased PP'):
        build_from({},2,options)


@pytest.mark.parametrize('batch',[2,4])
def test_nvidia_tiles_compile_without_gpu(fixture,tmp_path,batch):
    import os
    from pathlib import Path
    import subprocess
    nvcc = Path(os.environ.get('CUDA','/opt/cuda'))/'bin/nvcc'
    if not nvcc.is_file():
        pytest.skip('CUDA nvcc unavailable')
    spec = stage.validate(fixture/'stage',allow_fixture=True,gate='module8')
    build_from(spec,batch,work=tmp_path)
    (tmp_path/'glm53_options.h').write_text(Options().header())
    source = tmp_path/'compile.cu'
    source.write_text('#include <cuda_runtime.h>\n#include "etx/abi.h"\n#include "batch_tiles.hip"\n')
    result = subprocess.run([str(nvcc),'-arch=sm_90','-std=c++17','--fmad=false',
                             '-I',str(graph_cpu.ROOT),'-I',str(graph_cpu.ROOT/'etx/runtime/include'),
                             '-I',str(fixture/'stage'),'-I',str(tmp_path),
                             '-include','glm53_options.h','-dc',str(source),'-o',str(tmp_path/'tiles.o')],
                            text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'tiles.o').stat().st_size>0
