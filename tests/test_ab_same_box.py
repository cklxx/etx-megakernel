"""tools/ab_same_box.py: same-machine interleaved A/B with an A/A control, and its lines in tools/extract_key.py."""
import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_same_box  # noqa: E402
import extract_key  # noqa: E402

FAKE = '''
import sys, pathlib
arm, base, step = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c = pathlib.Path(sys.argv[4]); n = int(c.read_text()) if c.exists() else 0; c.write_text(str(n + 1))
if len(sys.argv) > 5 and sys.argv[5] == "fail": sys.exit(3)
print(f"step_ms={base + step * (n % 2):.4f}")
'''


def fake(tmp_path, arm, base, step=0.0, extra=""):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    return f"{sys.executable} {script} {arm} {base} {step} {tmp_path / ('n_' + arm)} {extra}"


def args(tmp_path, a, b, c=None, rounds=5):
    return argparse.Namespace(a=a, b=b, control=c, metric=r"step_ms=([\d.]+)", rounds=rounds, timeout=60,
                              label="qkv_b4", out=str(tmp_path / "out"))


def extract(tmp_path):
    return extract_key.extract(tmp_path / "out", dict(dir="out", title="t", inputs=["ab.log"]))


def test_rotation_puts_every_arm_in_every_position():
    seen = [ab_same_box.order(r) for r in range(3)]
    for pos in range(3):
        assert {o[pos] for o in seen} == set(ab_same_box.ARMS)


def test_faster_candidate_is_a_win_and_the_extractor_reads_it(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0, 0.01), fake(tmp_path, "B", 9.0, 0.01),
                             fake(tmp_path, "C", 10.0, 0.01)), identity=lambda: "box1")
    assert r["verdict"] == "WIN"
    assert r["gain"]["median"] == pytest.approx(10.0 / 9.0 - 1, rel=1e-2)
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "COMPLETE"
    assert len(key["runs"]) == 15 and all(x["rc"] == 0 for x in key["runs"])
    (cmp_,) = key["comparisons"]
    assert cmp_["outcome"] == "WIN" and cmp_["batch"] == "qkv_b4"
    assert cmp_["median_pct"] == pytest.approx(r["gain"]["median"] * 100, abs=1e-5)


def test_slower_candidate_is_a_loss(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 9.0), fake(tmp_path, "B", 10.0)), identity=lambda: "m")
    assert r["verdict"] == "LOSS"


def test_effect_inside_the_control_noise_is_no_diff(tmp_path):
    # A alternates 10.0 / 10.5, so A against its own copy moves by up to 5 %; B is 1 % faster.
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0, 0.5), fake(tmp_path, "B", 9.9),
                             fake(tmp_path, "C", 10.25)), identity=lambda: "m")
    assert r["verdict"] == "NO-DIFF"


def test_failed_run_is_incomplete_in_both_tools(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0, extra="fail")),
                        identity=lambda: "m")
    assert r["verdict"] == "INCOMPLETE" and "rc=3" in r["rule"]
    assert extract(tmp_path)["verdict"]["value"] == "INCOMPLETE"
    assert (tmp_path / "out/runs/r0_B.log").exists()


def test_machine_change_is_incomplete(tmp_path):
    ids = iter(["box1", "box2"])
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0)), identity=lambda: next(ids))
    assert r["verdict"] == "INCOMPLETE" and "machine identity changed" in r["rule"]


def test_fewer_than_three_rounds_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        ab_same_box.main(["--a", "true", "--b", "true", "--metric", "x", "--rounds", "2", "--out", str(tmp_path)])
