"""CPU proofs for lever Q (gemv_no_terminal) on the pinned fleet source; no device required."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import gemv_no_terminal as q


def test_terminal_pipeline_executes_same_consumed_groups(tmp_path):
    """Compile the actual replacement loop with logging macros, not a rewrite.

    All shapes exercise odd/even tails and many full pairs. Each consumed
    padded row's exact clamped address and sequence must remain unchanged;
    precisely one final issued group per active wave disappears.
    """
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("host C++ compiler not installed")
    prelude = r"""
extern "C" int printf(const char*, ...);
extern "C" void abort();
#define assert(ok) do { if (!(ok)) abort(); } while(0)
struct List {
    int data[4096], size=0;
    void push_back(int v) { assert(size<4096); data[size++]=v; }
    void pop_back() { --size; }
    bool empty() const { return size==0; }
    bool operator==(const List& b) const {
        if (size!=b.size) return false;
        for (int i=0; i<size; ++i) if (data[i]!=b.data[i]) return false;
        return true;
    }
};
struct Trace { List issued, consumed, rows; };
#define FLEET_ISSUE_GROUP(BUF, KK) (BUF)=(KK); t.issued.push_back(KK);
#define FLEET_REDUCE_GROUP(BUF, KK) \
    assert((BUF)==(KK)); \
    t.consumed.push_back(KK); \
    for (int row=0; row<RPI; ++row) t.rows.push_back((KK)+row<cnt ? (KK)+row : cnt-1);
"""
    def wrapped(name, body):
        return f"""
Trace {name}(int RPI, int cnt, int wave) {{
    Trace t; int k=RPI*wave, stride=RPI*4, bufA=-1, bufB=-1;
    if (k>=cnt) return t;
{body}
    return t;
}}
"""
    source = prelude + wrapped("old_loop", q.PIPELINE_OLD) + wrapped("new_loop", q.PIPELINE_NO_TERMINAL) + r"""
int main() {
    int cases=0;
    for (int rpi=2; rpi<=8; rpi*=2) for (int cnt=0; cnt<=2048; ++cnt) for (int wave=0; wave<4; ++wave) {
        auto a=old_loop(rpi,cnt,wave), b=new_loop(rpi,cnt,wave);
        assert(a.consumed==b.consumed && a.rows==b.rows);
        assert(b.issued==b.consumed);
        if (!a.issued.empty()) { a.issued.pop_back(); assert(a.issued==b.issued); }
        ++cases;
    }
    printf("pipeline consumed/padded rows equal; one terminal group removed: %d cases\n", cases);
}
"""
    cpp, exe = tmp_path / "pipeline.cpp", tmp_path / "pipeline"
    cpp.write_text(source)
    subprocess.run([cxx, "-O2", "-std=c++17", str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    assert "24588 cases" in out


def test_q_scope_restores_the_pinned_source_byte_for_byte(tmp_path, pinned_src):
    """Q only adds a private helper/dispatch copy and retargets the three qkv calls: removing exactly those edits
    gives back the pinned gemv.h and fleet_kernel.hip, so every other GEMV call site is untouched."""
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    old_g = (copy / "kernels/gemv.h").read_text()
    old_k = (copy / "kernels/fleet_kernel.hip").read_text()
    pf.apply(copy, "gemv_no_terminal")
    g = (copy / "kernels/gemv.h").read_text()
    k = (copy / "kernels/fleet_kernel.hip").read_text()
    clone_pipe = pf._function(g, "template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_pipelined_no_terminal(")
    clone_dispatch = pf._function(g, "__device__ inline void gemv_rows_no_terminal(")
    assert g.replace("\n\n" + clone_pipe, "").replace("\n\n" + clone_dispatch, "") == old_g
    assert k.count("gemv_rows_no_terminal(") == 3
    assert k.replace("gemv_rows_no_terminal(", "gemv_rows(") == old_k
    assert q.PIPELINE_NO_TERMINAL in clone_pipe and q.PIPELINE_OLD not in clone_pipe
