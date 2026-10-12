"""CPU proofs for lever Q-all (gemv_no_terminal_all) on the pinned fleet source; no device required."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from examples.dsv2lite import patch_fleet as pf
from examples.dsv2lite.levers import gemv_no_terminal_all as qa


def test_every_loop_reduces_and_signals_the_same_groups_and_issues_none_past_the_end(tmp_path):
    """Compile the exact original and peeled loop texts of all three helpers with logging macros (no rewrite).

    For every RPI-like stride, cnt 0..2048 and start wave 0..3: the reduced groups, their buffer pairing and gate_up's
    CHUNKS_DONE signals are identical in order; the peeled loop issues exactly the groups it reduces; the original
    issues exactly one more (the terminal group)."""
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("host C++ compiler not installed")
    prelude = r"""
extern "C" int printf(const char*, ...);
extern "C" void abort();
#define assert(ok) do { if (!(ok)) abort(); } while(0)
struct List {
    int data[4096], size = 0;
    void push_back(int v) { assert(size < 4096); data[size++] = v; }
    void pop_back() { --size; }
    bool operator==(const List& b) const {
        if (size != b.size) return false;
        for (int i = 0; i < size; ++i) if (data[i] != b.data[i]) return false;
        return true;
    }
};
struct Trace { List issued, reduced, done; };
#define ISSUE(BUF, KK) (BUF) = (KK); t.issued.push_back(KK);
#define REDUCE(BUF, KK) assert((BUF) == (KK)); t.reduced.push_back(KK);
#define FLEET_ISSUE_GROUP ISSUE
#define FLEET_REDUCE_GROUP REDUCE
#define FLEET_HISSUE ISSUE
#define FLEET_HREDUCE REDUCE
#define FLEET_ISSUE_PAIR ISSUE
#define FLEET_REDUCE_PAIR REDUCE
#define FLEET_CHUNKS_DONE(KK) t.done.push_back(KK);
"""
    funcs, names = [], []
    for helper, (old, new) in qa.LOOPS.items():
        for tag, body in (("old", old), ("new", new)):
            name = f"{helper}_{tag}"
            names.append(name)
            funcs.append(f"""
Trace {name}(int stride, int cnt, int wave) {{
    Trace t; const int kWaves = stride; int k = (stride / 4) * wave, bufA = -1, bufB = -1;
    if (k >= cnt) return t;
{body}
    return t;
}}""")
    pairs = ", ".join(f"{{{h}_old, {h}_new}}" for h in qa.LOOPS)
    main = r"""
typedef Trace (*Loop)(int, int, int);
int main() {
    Loop loops[][2] = {""" + pairs + r"""};
    int cases = 0;
    for (auto& pair : loops)
        for (int stride = 4; stride <= 32; stride *= 2) for (int cnt = 0; cnt <= 2048; ++cnt) for (int wave = 0; wave < 4; ++wave) {
            Trace a = pair[0](stride, cnt, wave), b = pair[1](stride, cnt, wave);
            assert(a.reduced == b.reduced && a.done == b.done);
            assert(b.issued == b.reduced);
            if (a.issued.size) { assert(a.issued.size == a.reduced.size + 1); a.issued.pop_back(); assert(a.issued == b.issued); }
            ++cases;
        }
    printf("%d cases\n", cases);
}
"""
    cpp, exe = tmp_path / "loops.cpp", tmp_path / "loops"
    cpp.write_text(prelude + "".join(funcs) + main)
    subprocess.run([cxx, "-O2", "-std=c++17", str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    assert out.strip() == f"{3 * 4 * 2049 * 4} cases"


def test_q_all_only_rewrites_the_three_loops(tmp_path, pinned_src):
    copy = tmp_path / "src"
    shutil.copytree(pinned_src, copy)
    before = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file()}
    pf.apply(copy, "gemv_no_terminal_all")
    after = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file()}
    changed = [p for p in before if before[p] != after[p]]
    assert [str(p) for p in changed] == ["kernels/gemv.h"]
    g = after[changed[0]].decode()
    for old, new in qa.LOOPS.values():
        assert g.count(new) == 1 and old not in g
        g = g.replace(new, old)
    assert g.encode() == before[changed[0]]
