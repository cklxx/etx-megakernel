"""CPU contract and byte-loading checks; no GPU runtime is initialized."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest
from etx.codegen.manifest import io_roles_manifest

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / 'etx/runtime/host'


@pytest.fixture(scope='module')
def harness(tmp_path_factory):
    path = tmp_path_factory.mktemp('loader')
    source = path / 'main.cc'
    source.write_text(r'''
#include "loader.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
struct cpu: etx::loader_transport {
 std::set<void*> live; int peers=0, allocations=0;
 void peer(int r,int o) override {if(r==o)std::abort();++peers;}
 void* allocate(int,size_t n,bool) override {++allocations;auto p=std::malloc(n);live.insert(p);return p;}
 void release(int,void* p) noexcept override {live.erase(p);std::free(p);}
 void zero(int,void* p,size_t n) override {std::memset(p,0,n);}
 void copy(int,void* p,const void* s,size_t n) override {if(std::getenv("ETX_TEST_FAIL_COPY"))throw std::runtime_error("copy failure");std::memcpy(p,s,n);}
};
int main(int argc,char** argv){cpu c;
 try { {etx::loaded_plan p(argv[1],argv[2],argv[3],c);
 for(auto name:{"left","right","ctrl"}){auto b=(unsigned char*)p.argument(name);
 std::cout<<name<<":";for(int i=0;i<4;++i)std::cout<<int(b[i])<<",";std::cout<<"\n";}}
 if(!c.live.empty())return 3;std::cout<<"peers="<<c.peers<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<" allocations="<<c.allocations;
 if(!c.live.empty())return 3;return 2;}}
''')
    exe = path / 'loader'
    subprocess.run([shutil.which('c++'), '-std=c++17', '-Wall', '-Wextra',
                    '-DETX_LOADER_CPU_ONLY', '-I'+str(HOST), str(source),
                    str(HOST/'loader.cc'), '-o', str(exe)], check=True, capture_output=True)
    return exe


def fixture(tmp_path, ranks):
    args = {'left': 0, 'right': 1, 'ctrl': 2}
    roles = {n: dict(bytes=4, owner=(1 if n == 'right' and ranks == 2 else 0),
                     role='runtime' if n == 'ctrl' else 'weight',
                     memory='system' if n == 'ctrl' else 'device') for n in args}
    manifest = io_roles_manifest(args, ranks, roles)
    (tmp_path/'blob').write_bytes(bytes(range(12)))
    ranges = [dict(arg='left', rank=0, file='blob', src=0, dst=0, rows=1, stride=4, width=4),
              dict(arg='right', rank=roles['right']['owner'], file='blob', src=4,
                   dst=0, rows=2, stride=4, width=2)]
    return dict(devices=ranks, args=args), manifest, dict(ranges=ranges)


def invoke(harness, tmp_path, data):
    paths = [tmp_path/n for n in ('plan.json', 'manifest.json', 'weights.json')]
    for p, d in zip(paths, data):
        p.write_text(json.dumps(d))
    return subprocess.run([str(harness), *map(str, paths)], capture_output=True, text=True)


@pytest.mark.parametrize('model,ranks', [('GLM_PP2', 2), ('Qwen', 1)])
def test_fixture_load(harness, tmp_path, model, ranks):
    result = invoke(harness, tmp_path, fixture(tmp_path, ranks))
    assert result.returncode == 0, result.stderr
    assert 'left:0,1,2,3,' in result.stdout
    assert 'right:4,5,8,9,' in result.stdout
    assert 'ctrl:0,0,0,0,' in result.stdout
    assert f'peers={ranks*(ranks-1)}' in result.stdout


@pytest.mark.parametrize('bad', ['missing', 'overlap', 'gap', 'rank', 'overflow',
                                  'truncated', 'path', 'unknown', 'nonweight',
                                  'index', 'size', 'role', 'memory'])
def test_contract_rejected_before_allocation(harness, tmp_path, bad):
    plan, manifest, weights = fixture(tmp_path, 2)
    r = weights['ranges'][0]
    if bad == 'missing': weights['ranges'].pop()
    elif bad == 'overlap': weights['ranges'].append(dict(r))
    elif bad == 'gap': r.update(dst=1, width=3)
    elif bad == 'rank': r['rank'] = 1
    elif bad == 'overflow': r.update(src=2**64-1)
    elif bad == 'truncated': r['src'] = 12
    elif bad == 'path': r['file'] = '../blob'
    elif bad == 'unknown': r['arg'] = 'absent'
    elif bad == 'nonweight': r['arg'] = 'ctrl'
    elif bad == 'index': plan['args']['right'] = 0
    elif bad == 'size': manifest['io_roles']['left']['bytes'] = 0
    elif bad == 'role': manifest['io_roles']['left']['role'] = 'guess'
    elif bad == 'memory': manifest['io_roles']['ctrl']['memory'] = 'managed'
    result = invoke(harness, tmp_path, (plan, manifest, weights))
    assert result.returncode == 2, result.stderr
    assert 'allocations=0' in result.stderr


@pytest.mark.parametrize('backend,env', [('CUDA', 'CUDA'), ('ROCM', 'ROCM')])
def test_sdk_compile(tmp_path, backend, env):
    # SDK roots or compiler commands accepted. Compile only: never run/link.
    value = os.environ.get(env)
    compiler = (str(Path(value)/'bin'/('nvcc' if backend == 'CUDA' else 'hipcc'))
                if value and Path(value).is_dir() else value)
    compiler = compiler or shutil.which('nvcc' if backend == 'CUDA' else 'hipcc')
    if not compiler:
        pytest.skip(f'{env} SDK unavailable')
    flags = shlex.split(os.environ.get(f'ETX_HOST_{backend}_FLAGS', ''))
    cmd = [compiler, '-std=c++17', *flags]
    if backend == 'CUDA': cmd += ['-DETX_HOST_CUDA']
    cmd += ['-c', str(HOST/'loader.cc'), '-o', str(tmp_path/'loader.o')]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_model_names():
    for path in HOST.glob('*'):
        assert not any(n in path.read_text() for n in ('glm', 'qwen'))


@pytest.mark.parametrize('field,value', [('bytes', -1), ('bytes', True),
                                        ('owner', 2), ('role', 'unknown'),
                                        ('memory', 'managed')])
def test_manifest_validation(tmp_path, field, value):
    plan, manifest, _ = fixture(tmp_path, 2)
    manifest['io_roles']['left'][field] = value
    with pytest.raises(ValueError):
        io_roles_manifest(plan['args'], 2, manifest['io_roles'])


def test_bad_json(harness, tmp_path):
    data = fixture(tmp_path, 1)
    invoke(harness, tmp_path, data)
    (tmp_path/'plan.json').write_text('{"devices":1,"devices":2}')
    result = subprocess.run([str(harness), str(tmp_path/'plan.json'),
                             str(tmp_path/'manifest.json'), str(tmp_path/'weights.json')],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'duplicate JSON key' in result.stderr
    assert 'allocations=0' in result.stderr


def test_transport_failure_releases_allocations(harness, tmp_path):
    invoke(harness, tmp_path, fixture(tmp_path, 2))
    result = subprocess.run([str(harness), *[str(tmp_path/n) for n in
                             ('plan.json', 'manifest.json', 'weights.json')]],
                            env={**os.environ, 'ETX_TEST_FAIL_COPY': '1'},
                            capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert 'copy failure allocations=3' in result.stderr
