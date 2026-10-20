"""Execute the host's actual check dispatch on CPU with GPU calls stubbed out."""
from pathlib import Path
import subprocess
import os

import pytest


@pytest.fixture(scope="module")
def check_dispatch(tmp_path_factory):
    host = Path("examples/vllm_llm/host.hip").read_text()
    start = host.index('      // Same DSL namespace')
    end = host.index('      const int xtasks', start)
    dispatch = host[start:end]
    summary_start = host.index('    printf("CHECK NOT-IMPORTED:')
    return_start = host.index("    return bad ? 4 : 0;", summary_start)
    summary = host[summary_start:return_start]
    return_line = host[return_start:host.index("\n", return_start)]
    src = r'''
#include <string>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
struct etx_imp_desc {};
using hipFunction_t = void*;
using hipModule_t = void*;
struct { int stream; } hst;
int launches = 0, lookups = 0, last_launch = -1;
int topk_ids[64] = {17}, n_tk = 64, check_pos = 1040, NR = 8;
float topk_w[64] = {0.5f};
constexpr int H=2048,E=128;
unsigned short o[H]={0x3f80},resid[H]={0x4000},gamma[H]={0x3f80};
unsigned short router_weights[E*H]={0x3f00},ox[8*H]={0x4040},resid2[H]={0x4080},rlog[E]={0x40a0};
int tei[64]={3};
const void* W(const std::string& name) {
  return name.find(".ln2")!=std::string::npos ? (const void*)gamma : (const void*)router_weights;
}
constexpr int hipMemcpyDeviceToHost = 1;
int hipMemcpy(void* dst, const void* src, size_t n, int) { memcpy(dst, src, n); return 0; }
void launch_imported(int launch) { ++launches; last_launch = launch; }
bool wait_stream(int, double) { return true; }
void die(const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); exit(1); }
const etx_imp_desc* etx_imp_find(const char*) { return nullptr; }
const void* etx_orig_find(const char*) { ++lookups; return nullptr; }
bool etx_orig_triton(const char*, const char**, const char**, int*) { return false; }
int hipModuleLoad(hipModule_t*, const char*) { return 0; }
int hipModuleGetFunction(hipFunction_t*, hipModule_t, const char*) { return 0; }
#define ETX_CHECK(x) (void)(x)
int main(int argc, char** argv) {
  const int nreq = getenv("CHECK_TEST_REQUESTS") ? atoi(getenv("CHECK_TEST_REQUESTS")) : 1;
  int bad = getenv("CHECK_TEST_BAD") ? 1 : 0, not_imported = 0;
  std::string not_imported_names;
  struct Inst { std::string exp; int n; std::string role; int req; int layer; };
  for (int n = 1; n < argc; ++n) {
    Inst x{argv[n], n, std::string(argv[n]) == "dsl_topk" ? "topk" : "other", 0, n};
    if (x.exp=="dsl_gemv_producer") {x.role="o_proj";x.layer=0;}
''' + dispatch + r'''
  }
''' + summary + r'''
  printf("launches=%d lookups=%d last_launch=%d\n", launches, lookups, last_launch);
''' + return_line + r'''
}
'''
    directory = tmp_path_factory.mktemp("check_dispatch")
    source, binary = directory / "check.cc", directory / "check"
    source.write_text(src)
    # This repo's local toolchain supplies libc++ where Apple's c++ may not.
    llvm = Path("/opt/homebrew/opt/llvm/bin/clang++")
    compiler = os.environ.get("CXX", str(llvm) if llvm.exists() else "c++")
    build = subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return binary


def test_dsl_skips_comparison_but_executes(check_dispatch):
    run = subprocess.run([str(check_dispatch), "dsl_gemv", "dsl_topk", "dsl_gemv"], capture_output=True, text=True)
    assert run.returncode == 0
    assert "CHECK NOT-IMPORTED: 3 instance(s): dsl_gemv, dsl_topk, dsl_gemv\n" in run.stdout
    assert "CHECK OK: 0 launch(es) differ\n" in run.stdout
    assert "launches=3 lookups=0" in run.stdout


@pytest.mark.parametrize('requests', [1, 2, 4, 8])
def test_import_check_launch_offset_accounts_for_all_prologues(check_dispatch, requests):
    run = subprocess.run([str(check_dispatch), 'dsl_gemv', 'dsl_topk'],
                         env={**os.environ, 'CHECK_TEST_REQUESTS': str(requests)}, capture_output=True, text=True)
    assert run.returncode == 0
    assert f'launches=2 lookups=0 last_launch={requests+2}' in run.stdout


@pytest.mark.parametrize("export", ["rms_norm_2d", "not_dsl_gemv", "dsl", "DSL_gemv"])
def test_non_dsl_without_original_still_dies(check_dispatch, export):
    run = subprocess.run([str(check_dispatch), export], capture_output=True, text=True)
    assert run.returncode == 1
    assert run.stderr == f"no original for {export}\n"
    assert "CHECK OK" not in run.stdout


def test_not_imported_count_and_name_limit(check_dispatch):
    run = subprocess.run([str(check_dispatch)] + ["dsl_gemv"] * 7, capture_output=True, text=True)
    assert run.returncode == 0
    assert "CHECK NOT-IMPORTED: 7 instance(s): " + ", ".join(["dsl_gemv"] * 5) + ", ...\n" in run.stdout


def test_difference_summary_remains_failed(check_dispatch):
    run = subprocess.run([str(check_dispatch), "dsl_gemv"], env={**os.environ, "CHECK_TEST_BAD": "1"},
                         capture_output=True, text=True)
    assert run.returncode == 4
    assert "CHECK FAILED: 1 launch(es) differ\n" in run.stdout
    assert "CHECK OK" not in run.stdout


def test_routing_audit_copies_actual_layer_bytes(check_dispatch, tmp_path):
    import struct
    run = subprocess.run([str(check_dispatch), "dsl_gemv", "dsl_topk"],
                         env={**os.environ, "ETX_CHECK_TOPK_DUMP": str(tmp_path)}, capture_output=True, text=True)
    assert run.returncode == 0
    assert "ROUTING-AUDIT layer=2 position=1040 replicas=8" in run.stdout
    assert (tmp_path / "L2_ids.bin").read_bytes() == struct.pack("64i", 17, *([0]*63))
    assert (tmp_path / "L2_weights.bin").read_bytes() == struct.pack("64f", .5, *([0.]*63))
    assert len(list(tmp_path.iterdir())) == 2


def test_producer_capture_executes_host_writer_and_refuses_overwrite(check_dispatch,tmp_path):
    from examples.vllm_llm import producer_replay as replay
    import struct
    env={**os.environ,"ETX_PRODUCER_CAPTURE":str(tmp_path)}
    run=subprocess.run([str(check_dispatch),"dsl_gemv_producer"],env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    for kind,size in replay.KINDS.items():
        assert (tmp_path/f"L0_{kind}.bin").stat().st_size==size
    assert (tmp_path/"L0_o.bin").read_bytes()[:2]==struct.pack("<H",0x3f80)
    assert (tmp_path/"L0_norm.bin").read_bytes()[:2]==struct.pack("<H",0x4040)
    assert (tmp_path/"L0_mix.bin").read_bytes()[:4]==struct.pack("<f",.5)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    retry=subprocess.run([str(check_dispatch),"dsl_gemv_producer"],env=env,capture_output=True,text=True)
    assert retry.returncode==1 and "capture exists or cannot create" in retry.stderr
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    replay.seal(tmp_path,identity="CPU-host-dispatch-fixture",provenance="synthetic-test-fixture",positions=[1040],layers=1)
    replay.read_capture(tmp_path)
    with pytest.raises(ValueError,match="provenance"):replay.read_capture(tmp_path,actual=True)
