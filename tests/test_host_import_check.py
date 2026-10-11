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
int launches = 0, lookups = 0;
int topk_ids[64] = {17}, n_tk = 64, check_pos = 1040, NR = 8;
float topk_w[64] = {0.5f};
constexpr int hipMemcpyDeviceToHost = 1;
int hipMemcpy(void* dst, const void* src, size_t n, int) { memcpy(dst, src, n); return 0; }
void launch_imported(int) { ++launches; }
bool wait_stream(int, double) { return true; }
void die(const std::string& s) { fprintf(stderr, "%s\n", s.c_str()); exit(1); }
const etx_imp_desc* etx_imp_find(const char*) { return nullptr; }
const void* etx_orig_find(const char*) { ++lookups; return nullptr; }
bool etx_orig_triton(const char*, const char**, const char**, int*) { return false; }
int hipModuleLoad(hipModule_t*, const char*) { return 0; }
int hipModuleGetFunction(hipFunction_t*, hipModule_t, const char*) { return 0; }
#define ETX_CHECK(x) (void)(x)
int main(int argc, char** argv) {
  int bad = getenv("CHECK_TEST_BAD") ? 1 : 0, not_imported = 0;
  std::string not_imported_names;
  struct Inst { std::string exp; int n; std::string role; int req; int layer; };
  for (int n = 1; n < argc; ++n) {
    Inst x{argv[n], n, std::string(argv[n]) == "dsl_topk" ? "topk" : "other", 0, n};
''' + dispatch + r'''
  }
''' + summary + r'''
  printf("launches=%d lookups=%d\n", launches, lookups);
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
