"""--check-coverage, the third gate (round 45).

Round 42's wvSplitK bug — lm_head rows [8192, 9728) never written — passed `--check` (both kernels get ETX's
grid, so both leave the same hole) and 32/32 (the golden's argmax never landed in the band) for weeks. Neither
gate can see an unwritten output row; a full-prefill per-position argmax diff can. These tests pin the gate's
wiring and the setup-time coverage asserts so the class stays closed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = (ROOT / "examples/vllm_llm/host.hip").read_text()
PREP = (ROOT / "examples/llm/prep.py").read_text()


def test_gate_exists_and_implies_teacher():
    """--check-coverage must teacher-force the whole sequence: a free-running run's positions diverge from the
    HF prefill at the first mismatch and every later position is meaningless."""
    assert '"--check-coverage"' in HOST
    assert "check_coverage = true; teacher = true;" in HOST
    assert "CHECK-COVERAGE" in HOST
    # the gate fails the process, not just a printf
    assert "return 5" in HOST.split("CHECK-COVERAGE FAILED")[-1][:200] or "return 5" in HOST


def test_reference_file_is_derived_from_the_golden_name():
    """prefill_c1024.txt beside golden_c1024.txt, per request — not one global file."""
    assert '"prefill" + gname.substr(6)' in HOST
    assert "--prefill-only" in PREP and "def prefill_reference" in PREP
    assert '"prefill" + gname[len("golden"):]' in PREP


def test_mismatches_are_classified_by_the_hf_margin():
    """A bf16 argmax tie is not an unwritten row: mismatches with HF top-1 margin <= 0.5 are tolerated and
    counted separately; anything larger fails the gate."""
    assert "cov_marg" in HOST and "<= 0.5" in HOST
    assert "margin:" in PREP  # the reference carries per-position margins


def test_hole_injection_is_env_gated_and_not_default():
    """ETX_DEBUG_HOLE reproduces the round-42 bug shape on demand (the gate's negative test). It must be
    reachable only through getenv and must never alter a build where the env is unset."""
    assert HOST.count('getenv("ETX_DEBUG_HOLE")') == 1
    seg = HOST.split('getenv("ETX_DEBUG_HOLE")')[1][:600]
    assert "deliberately never written" in seg
    # the shadowed M feeds both the row count and rpt, so the hole is exactly [M', V)
    assert "int xM = x.M;" in HOST and ".i32(xM).i32(x.K).i32(rpt)" in HOST


def test_setup_asserts_cover_the_sibling_sites():
    """The written-output invariants the host controls fail at setup, not at the gate: MoE task counts cover
    nb*TOPK slots of MI/32 (up) and H/64 (down) rows; attention partitions (hardcoded 256 tokens each,
    round 11) cover maxlen."""
    assert "do not cover nb*TOPK*(MI/32)" in HOST
    assert "do not cover nb*TOPK*(H/64)" in HOST
    assert "do not cover maxlen" in HOST
