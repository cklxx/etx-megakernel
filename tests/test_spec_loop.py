"""Compile/run the actual C ABI and device helper against a CPU-only backend."""
from pathlib import Path
import shutil
import subprocess

import pytest

HOST = Path(__file__).resolve().parents[1] / 'etx/runtime/host'

SOURCE = r'''
#include "spec.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Prefix-sensitive seeded target: a bad rollback changes all later predictions.
int32_t predict(const std::vector<int32_t>& prefix) {
  uint32_t x = 0x12345678;
  for (auto token : prefix) x = (x ^ uint32_t(token)) * 1664525u + 1013904223u;
  return int32_t(x % 97);
}
struct Stub {
  std::vector<int32_t> history{5};
  std::vector<char> events;
  etx_spec_request request{};
  int32_t* output = nullptr;
  etx_spec_result* result = nullptr;
  bool pending = false;
  unsigned calls = 0;
  int fail = 0;
  unsigned fail_at = 0;
  int malformed = 0;
  static int submit(void* ctx, const etx_spec_request* q, int32_t* out,
                    etx_spec_result* r) {
    auto& s = *static_cast<Stub*>(ctx);
    assert(!s.pending);
    s.events.push_back('s');
    assert(q->position == int64_t(s.history.size()-1));
    assert(q->token == s.history.back() && q->timeout_s == 2.0);
    if (s.fail_at == s.calls && s.fail == 1) return 1;
    if (s.fail_at == s.calls && s.fail == 3) throw std::runtime_error("submit");
    s.request = *q; s.output = out; s.result = r; s.pending = true;
    return 0; // No host-visible result until wait: enqueued work only.
  }
  static int wait(void* ctx, double timeout) {
    auto& s = *static_cast<Stub*>(ctx);
    assert(s.pending && timeout == 2.0);
    s.pending = false; s.events.push_back('w');
    if (s.fail_at == s.calls && s.fail == 2) return 1;
    if (s.fail_at == s.calls && s.fail == 4) throw std::runtime_error("wait");
    const auto& q = s.request;
    auto speculative = s.history;
    std::vector<int32_t> draft(q.k), target(q.k+1);
    unsigned prefix = s.calls % (q.k+1);
    for (unsigned i=0; i<q.k; ++i) {
      target[i] = predict(speculative);
      draft[i] = i < prefix ? target[i] : (target[i]+1+int(i%7))%97;
      speculative.push_back(draft[i]);
    }
    target[q.k] = predict(speculative);
    auto r = etx_spec_accept_greedy(draft.data(), target.data(), q.k,
                                   q.capacity, q.eos, s.output);
    // Device rollback discards the entire speculative suffix, then commits
    // only emitted tokens. No speculative state is exposed to the host loop.
    speculative.resize(s.history.size());
    speculative.insert(speculative.end(), s.output, s.output+r.L);
    s.history = speculative;
    *s.result = r;
    switch(s.malformed) {
      case 1: s.result->L=0; break;
      case 2: s.result->L=q.capacity+1; break;
      case 3: s.result->accepted=q.k+1; break;
      case 4: s.result->terminated=2; break;
      case 5: s.output[0]=-1; break;
      case 6: s.result->terminated=1; break;
    }
    ++s.calls;
    return 0;
  }
};
const etx_spec_backend backend{Stub::submit, Stub::wait};
etx_spec_descriptor descriptor(Stub& s, unsigned k=4) {
  return {ETX_SPEC_ABI_VERSION, sizeof(etx_spec_descriptor), &backend,
          &s, k, 0, 2.0, 0, 5, {0,0,0}};
}
void helper() {
  const int32_t target[]{10,11,12,13,14};
  for (unsigned prefix=0; prefix<=4; ++prefix) {
    int32_t draft[]{10,11,12,13}, out[6]{-9,-9,-9,-9,-9,-9};
    if (prefix<4) draft[prefix]=99;
    auto r=etx_spec_accept_greedy(draft,target,4,5,-1,out);
    assert(r.L==prefix+1 && r.accepted==prefix && !r.terminated);
    for(unsigned i=0;i<r.L;++i) assert(out[i]==target[i]);
    assert(out[r.L]==-9);
    // EOS in accepted prefix, correction, bonus; EOS after rejection ignored.
    for(int eos=10;eos<=14;++eos) {
      r=etx_spec_accept_greedy(draft,target,4,5,eos,out);
      unsigned n=std::min(prefix+1,unsigned(eos-10+1));
      assert(r.L==n && r.accepted==std::min(prefix,n));
      assert(r.terminated==unsigned(eos-10<int(prefix+1)));
    }
    for(unsigned cap=1;cap<=5;++cap) {
      r=etx_spec_accept_greedy(draft,target,4,cap,-1,out);
      assert(r.L==std::min(cap,prefix+1));
      assert(r.accepted==std::min(cap,prefix));
    }
  }
  int32_t out=-1;
  auto r=etx_spec_accept_greedy(nullptr,target,0,1,-1,&out);
  assert(r.L==1 && !r.accepted && out==10);
}
void greedy() {
  for(unsigned k : {0u,1u,4u,8u}) for(unsigned budget : {0u,1u,2u,32u,101u}) {
    Stub s; auto d=descriptor(s,k);
    std::vector<int32_t> tokens(budget+1,-1), reference{5};
    std::vector<uint32_t> lengths(budget+1,0);
    uint64_t count=999, steps=999;
    assert(etx_spec_decode(&d,budget,-1,tokens.data(),lengths.data(),&count,&steps)==ETX_OK);
    for(unsigned i=0;i<budget;++i) reference.push_back(predict(reference));
    assert(s.history==reference && count==budget && d.position==budget);
    assert(std::equal(reference.begin()+1,reference.end(),tokens.begin()));
    unsigned total=0;
    for(unsigned i=0;i<steps;++i) {
      unsigned expected=std::min(budget-total,i%(k+1)+1);
      assert(lengths[i]==expected); total+=expected;
      assert(s.events[2*i]=='s' && s.events[2*i+1]=='w');
    }
    assert(total==budget && s.events.size()==2*steps && !s.pending);
    assert(tokens[budget]==-1);
    if(steps) assert(d.last.L==lengths[steps-1]);
  }
  // EOS is emitted exactly once and stops both device acceptance and host loop.
  Stub s; auto d=descriptor(s);
  std::vector<int32_t> ref{5};
  for(unsigned i=0;i<7;++i) ref.push_back(predict(ref));
  int32_t eos=ref.back(), out[32]; uint32_t lengths[32]; uint64_t n, steps;
  auto first=std::find(ref.begin()+1,ref.end(),eos);
  assert(etx_spec_decode(&d,32,eos,out,lengths,&n,&steps)==ETX_OK);
  assert(n==uint64_t(first-ref.begin()) && d.last.terminated && out[n-1]==eos);
  assert(std::equal(ref.begin()+1,first+1,out));
}
void errors() {
  for(int failure=1;failure<=4;++failure) {
    Stub s; s.fail=failure; s.fail_at=2; auto d=descriptor(s);
    int32_t out[32]; uint32_t lengths[32]; uint64_t n,steps;
    assert(etx_spec_decode(&d,32,-1,out,lengths,&n,&steps)==ETX_BACKEND_ERROR);
    assert(n==3 && steps==2 && d.position==3 && d.last.L==2 && !s.pending);
    assert(lengths[0]==1 && lengths[1]==2 && d.token==out[2]);
    assert(s.events.size()==unsigned(failure==1 || failure==3 ? 5 : 6));
  }
  for(int malformed=1;malformed<=6;++malformed) {
    Stub s; s.malformed=malformed; auto d=descriptor(s);
    int32_t out[5];
    assert(etx_spec_step(&d,5,-1,out)==ETX_BACKEND_ERROR);
    assert(d.position==0 && d.token==5 && d.last.L==0 && !s.pending);
  }
}
void invalid() {
  for(int field=0;field<12;++field) {
    Stub s; auto d=descriptor(s); int32_t out[5];
    switch(field) {
      case 0:d.abi_version=99;break; case 1:d.struct_size=0;break;
      case 2:d.backend=nullptr;break; case 3:d.reserved=1;break;
      case 4:d.k=UINT32_MAX;break; case 5:d.timeout_s=0;break;
      case 6:d.timeout_s=NAN;break; case 7:d.timeout_s=INFINITY;break;
      case 8:d.position=-1;break; case 9:d.position=INT64_MAX;break;
      case 10:d.position=INT64_MAX-3;break;
      case 11:d.timeout_s=-1;break;
    }
    assert(etx_spec_step(&d,5,-1,out)==ETX_INVALID && s.events.empty());
  }
  Stub s; auto d=descriptor(s); int32_t out[5]; uint64_t n=99,steps=99;
  assert(etx_spec_step(nullptr,1,-1,out)==ETX_INVALID);
  assert(etx_spec_step(&d,0,-1,out)==ETX_INVALID);
  assert(etx_spec_step(&d,1,-1,nullptr)==ETX_INVALID);
  assert(etx_spec_decode(&d,0,-1,nullptr,nullptr,&n,&steps)==ETX_OK && !n && !steps);
  assert(etx_spec_decode(&d,1,-1,out,nullptr,&n,&steps)==ETX_INVALID && !n && !steps);
  n=99;
  assert(etx_spec_decode(&d,0,-1,nullptr,nullptr,&n,nullptr)==ETX_INVALID && !n);
  assert(etx_spec_decode(&d,0,-1,nullptr,nullptr,&n,&n)==ETX_INVALID);
  assert(etx_spec_decode(&d,UINT64_MAX,-1,out,nullptr,&n,&steps)==ETX_INVALID);
  assert(s.events.empty());
}
int main(int argc,char** argv) {
  assert(argc==2); std::string name=argv[1];
  if(name=="helper") helper(); else if(name=="greedy") greedy();
  else if(name=="errors") errors(); else if(name=="invalid") invalid();
  else return 1;
}
'''


@pytest.fixture(scope='module')
def executable(tmp_path_factory):
    work = tmp_path_factory.mktemp('spec-loop')
    source = work / 'stub.cc'
    source.write_text(SOURCE)
    binary = work / 'stub'
    result = subprocess.run(
        [shutil.which('c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
         '-I', str(HOST), str(source), str(HOST / 'spec.cc'), '-o', str(binary)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize('case', ['helper', 'greedy', 'errors', 'invalid'])
def test_spec_loop(executable, case):
    result = subprocess.run([str(executable), case], capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_c_header(tmp_path):
    source = tmp_path / 'header.c'
    source.write_text('#include "spec.h"\nint main(void) {\n'
                      'int32_t target=7, output;\n'
                      'etx_spec_result r=etx_spec_accept_greedy(0,&target,0,1,-1,&output);\n'
                      'return !(r.L==1 && output==7);\n}\n')
    binary = tmp_path / 'header'
    compiled = subprocess.run(
        [shutil.which('cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
         '-I', str(HOST), str(source), '-o', str(binary)],
        capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    subprocess.run([str(binary)], check=True, timeout=10)
