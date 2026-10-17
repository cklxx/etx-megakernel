"""tools/ab_same_box.py: same-machine interleaved A/B with an A/A control, and its lines in tools/extract_key.py."""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_same_box  # noqa: E402
import check_results  # noqa: E402
import extract_key  # noqa: E402

# A fake run. argv: arm base step counter [mode]. The value alternates base, base+step by call count.
# mode "fail" exits 3; "garbage" prints step_ms=1.2.3; "zero" prints step_ms=0; "none" prints nothing;
# "sleep" sleeps 30 s; "failwarm" exits 3 on the first call only.
FAKE = '''
import sys, pathlib, time
arm, base, step = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c = pathlib.Path(sys.argv[4]); n = int(c.read_text()) if c.exists() else 0; c.write_text(str(n + 1))
mode = sys.argv[5] if len(sys.argv) > 5 else ""
if mode == "fail" or (mode == "failwarm" and n == 0): sys.exit(3)
if mode == "garbage": print("step_ms=1.2.3"); sys.exit(0)
if mode == "zero": print("step_ms=0"); sys.exit(0)
if mode == "none": print("no timing here"); sys.exit(0)
if mode == "sleep": time.sleep(30)
print(f"step_ms={base + step * (n % 2):.4f}")
'''
METRIC = r"step_ms=([\d.]+)"


def fake(tmp_path, arm, base, step=0.0, extra=""):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    return f"{sys.executable} {script} {arm} {base} {step} {tmp_path / ('n_' + arm)} {extra}"


def args(tmp_path, a, b, c=None, rounds=6, timeout=60, min_gain=1.0, metric=METRIC, identity_cmd=None):
    return argparse.Namespace(a=a, b=b, control=c, metric=metric, rounds=rounds, timeout=timeout,
                              label="qkv_b4", out=str(tmp_path / "out"), min_gain=min_gain,
                              identity_cmd=identity_cmd)


def extract(tmp_path):
    return extract_key.extract(tmp_path / "out", dict(dir="out", title="t", inputs=["ab.log"]))


def ab_lines(tmp_path, prefix):
    return [l for l in (tmp_path / "out/ab.log").read_text().splitlines() if l.startswith(prefix)]


# M1: all 6 orders, position and carry-over balance.
def test_six_orders_balance_position_and_carry_over():
    seen = [ab_same_box.order(r) for r in range(6)]
    assert sorted(seen) == sorted(itertools.permutations(ab_same_box.ARMS))
    for pos in range(3):
        counts = {arm: sum(o[pos] == arm for o in seen) for arm in ab_same_box.ARMS}
        assert set(counts.values()) == {2}
    pairs = {}
    for o in seen:
        for x, y in zip(o, o[1:]):
            pairs[(x, y)] = pairs.get((x, y), 0) + 1
    assert len(pairs) == 6 and set(pairs.values()) == {2}  # C does not always follow B
    assert ab_same_box.order(6) == ab_same_box.order(0)


@pytest.mark.parametrize("rounds", ["2", "3", "5", "7", "9"])
def test_rounds_must_be_a_multiple_of_six(tmp_path, rounds):
    with pytest.raises(SystemExit) as e:
        ab_same_box.main(["--a", "true", "--b", "true", "--metric", METRIC, "--rounds", rounds, "--out", str(tmp_path)])
    assert e.value.code == 2


def test_faster_candidate_is_a_win_and_the_extractor_reads_it(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0, 0.01), fake(tmp_path, "B", 9.0, 0.01),
                             fake(tmp_path, "C", 10.0, 0.01)), identity=lambda: "box1")
    assert r["verdict"] == "WIN"
    assert r["p"] == pytest.approx(0.5 ** 6)
    assert r["gain"]["median"] == pytest.approx(10.0 / 9.0 - 1, rel=1e-2)
    # M2: one warm-up run per arm, logged as ABWARM and not counted.
    assert len(ab_lines(tmp_path, "ABWARM")) == 3 and len(ab_lines(tmp_path, "ABRUN")) == 18
    assert all(len(v) == 6 for v in r["values"].values()) and set(r["warmup"]) == {"A", "B", "C"}
    (report,) = ab_lines(tmp_path, "ABREPORT")
    assert " p=0.015625 " in report and "min_gain=1%" in report and report.endswith("verdict=WIN")
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "COMPLETE"
    assert len(key["runs"]) == 21 and all(x["rc"] == 0 for x in key["runs"])
    (cmp_,) = key["comparisons"]
    assert cmp_["outcome"] == "WIN" and cmp_["batch"] == "qkv_b4"
    assert "p=0.015625" in cmp_["metric"]
    assert cmp_["median_pct"] == pytest.approx(r["gain"]["median"] * 100, abs=1e-5)
    assert {l["kind"] for l in key["lines"]} == {"ab"}


def test_slower_candidate_is_a_loss(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 9.0), fake(tmp_path, "B", 10.0)), identity=lambda: "m")
    assert r["verdict"] == "LOSS" and r["p"] == pytest.approx(0.5 ** 6)
    assert extract(tmp_path)["comparisons"][0]["outcome"] == "LOSS"


def test_effect_inside_the_control_noise_is_no_diff(tmp_path):
    # A alternates 10.0 / 10.5 and C stays 10.0, so max |control| is 5 %. B (9.9) gives gains of 1 % and 6 %:
    # every gain is positive, but the median (3.5 %) is inside the noise.
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0, 0.5), fake(tmp_path, "B", 9.9),
                             fake(tmp_path, "C", 10.0)), identity=lambda: "m")
    assert all(g > 0 for g in r["gains"]) and r["max_control"] == pytest.approx(0.05)
    assert r["verdict"] == "NO-DIFF"


# M3: minimum effect size and the sign test.
def test_min_gain_blocks_a_tiny_deterministic_win(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.999), min_gain=0.0),
                        identity=lambda: "m")
    assert r["verdict"] == "WIN"  # controls are all 0, so 0.01 % wins with --min-gain 0
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.999)), identity=lambda: "m")
    assert r["verdict"] == "NO-DIFF" and "min_gain=1%" in ab_lines(tmp_path, "ABREPORT")[0]


def test_cli_default_min_gain_is_one_percent(tmp_path):
    rc = ab_same_box.main(["--a", fake(tmp_path, "A", 10.0), "--b", fake(tmp_path, "B", 9.95), "--metric", METRIC,
                           "--identity-cmd", "echo m", "--out", str(tmp_path / "out")])
    assert rc == 0
    (report,) = ab_lines(tmp_path, "ABREPORT")
    assert "min_gain=1%" in report and report.endswith("verdict=NO-DIFF")  # a 0.5 % gain is below 1 %


def test_sign_test_p_values():
    assert ab_same_box.sign_test_p([0.1] * 6, positive=True) == pytest.approx(0.015625)
    assert ab_same_box.sign_test_p([0.1] * 3, positive=True) == pytest.approx(0.125)
    assert ab_same_box.sign_test_p([0.1] * 5 + [-0.1], positive=True) == pytest.approx(7 / 64)
    assert ab_same_box.sign_test_p([-0.1] * 6, positive=False) == pytest.approx(0.015625)
    assert ab_same_box.sign_test_p([0.0] * 6, positive=True) == 1.0


def test_p_above_alpha_is_no_diff():
    # 3 positive gains far above the noise: p = 0.125 > 0.05.
    v, _, p = ab_same_box.verdict([0.1, 0.1, 0.1], [0.0, 0.0, 0.0])
    assert v == "NO-DIFF" and p == pytest.approx(0.125)


def test_failed_run_is_incomplete_in_both_tools(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0, extra="fail")),
                        identity=lambda: "m")
    assert r["verdict"] == "INCOMPLETE" and "rc=3" in r["rule"]
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "INCOMPLETE"
    assert key["comparisons"][0]["outcome"] == "INCOMPLETE"
    assert (tmp_path / "out/runs/warm_B.log").exists()


def test_failed_warm_up_is_incomplete(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0, extra="failwarm")),
                        identity=lambda: "m")
    assert r["verdict"] == "INCOMPLETE" and "warm-up arm B rc=3" in r["rule"]
    assert not ab_lines(tmp_path, "ABRUN")


# Review 2, N1: the whole run sequence, warm-up included.
EXPECTED_ROUNDS = "ABC" "CAB" "BAC" "CBA" "ACB" "BCA"


@pytest.mark.parametrize("rounds", [6, 12])
def test_full_run_sequence_with_warm_up(tmp_path, rounds):
    ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0), rounds=rounds),
                    identity=lambda: "m")
    arms = [l.split("arm=")[1][0] for l in (tmp_path / "out/ab.log").read_text().splitlines()
            if l.startswith(("ABWARM", "ABRUN"))]
    seq = "".join(arms)
    assert seq == "ABC" + EXPECTED_ROUNDS * (rounds // 6)
    body = seq[3:]
    for k in range(rounds):  # each round is a permutation; the last arm of a round starts the next round
        assert sorted(body[3 * k:3 * k + 3]) == ["A", "B", "C"]
        if k + 1 < rounds:
            assert body[3 * k + 2] == body[3 * k + 3]
    pairs = {}
    for x, y in zip(body, body[1:]):
        if x != y:
            pairs[(x, y)] = pairs.get((x, y), 0) + 1
    assert len(pairs) == 6 and set(pairs.values()) == {2 * rounds // 6}
    for pos in range(3):
        assert {a: sum(body[3 * k + pos] == a for k in range(rounds)) for a in "ABC"} == {a: rounds // 3 for a in "ABC"}


# Review 2, N2: the process group is killed after every run, and the wait after a kill is bounded.
def _alive(pid):
    try:
        state = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):  # the process can exit during the read (ESRCH)
        return False
    return "\nState:\tZ" not in state and "\nState:\tX" not in state


def test_background_child_is_killed_after_a_normal_run(tmp_path):
    pidf = tmp_path / "pid"
    rc, text = ab_same_box.run_command(f"sleep 60 >/dev/null 2>&1 & echo $! > {pidf}; echo step_ms=1", timeout=20)
    assert rc == 0 and "step_ms=1" in text
    pid = int(pidf.read_text())
    deadline = time.monotonic() + 5
    dead = False
    while not dead and time.monotonic() < deadline:
        dead = not _alive(pid)  # an exiting process can read as gone and then as present once; one gone is enough
        time.sleep(0.05)
    assert dead


def test_wait_after_a_timeout_kill_is_bounded(tmp_path):
    pidf = tmp_path / "pid"
    # setsid puts the child outside the process group; it keeps the output pipe open.
    cmd = f"setsid sh -c 'echo $$ > {pidf}; exec sleep 60' & sleep 60"
    t0 = time.monotonic()
    try:
        rc, text = ab_same_box.run_command(cmd, timeout=1, kill_wait=1)
        assert rc == "timeout" and "output lost" in text
        assert time.monotonic() - t0 < 8
    finally:
        if pidf.exists():
            try:
                os.kill(int(pidf.read_text()), 9)
            except ProcessLookupError:
                pass


# Review 2, N4: run() checks its arguments.
@pytest.mark.parametrize("field,value,msg", [
    ("rounds", 0, "--rounds"), ("rounds", 5, "--rounds"), ("rounds", -6, "--rounds"), ("rounds", 6.0, "--rounds"),
    ("timeout", 0, "--timeout"), ("timeout", -1, "--timeout"), ("timeout", float("inf"), "--timeout"),
    ("min_gain", -1, "--min-gain"), ("min_gain", float("nan"), "--min-gain"),
    ("label", "a b", "--label"), ("label", "", "--label"),
    ("metric", r"x", "exactly 1 group"), ("metric", r"(", "not a valid regex"),
    ("a", "", "--a"), ("b", None, "--b")])
def test_run_rejects_bad_arguments(tmp_path, field, value, msg):
    a = args(tmp_path, "true", "true")
    setattr(a, field, value)
    with pytest.raises(ValueError, match=msg.replace("-", r"\-")):
        ab_same_box.run(a, identity=lambda: "m")
    assert not (tmp_path / "out").exists()  # nothing runs, nothing is written


@pytest.mark.parametrize("extra", [["--timeout", "0"], ["--min-gain", "-1"], ["--rounds", "0"]])
def test_cli_maps_bad_arguments_to_exit_2(tmp_path, extra):
    with pytest.raises(SystemExit) as e:
        ab_same_box.main(["--a", "true", "--b", "true", "--metric", METRIC, "--out", str(tmp_path / "o"), *extra])
    assert e.value.code == 2


# H1: every INCOMPLETE form gives INCOMPLETE in the extractor, with a comparison row.
@pytest.mark.parametrize("mode,reason", [("none", "printed no value"), ("garbage", "not a number"),
                                         ("zero", "not a positive finite number")])
def test_bad_or_missing_value_is_incomplete_in_both_tools(tmp_path, mode, reason):
    # H3: these values crashed the session before (ValueError, ZeroDivisionError).
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0, extra=mode)),
                        identity=lambda: "m")
    assert r["verdict"] == "INCOMPLETE" and reason in r["rule"]
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "INCOMPLETE" and "ABREPORT" in key["verdict"]["rule"]
    (cmp_,) = key["comparisons"]
    assert cmp_["outcome"] == "INCOMPLETE" and cmp_["median_pct"] is None and reason in cmp_["metric"]


def test_machine_change_is_incomplete_in_both_tools(tmp_path):
    ids = iter(["box1", "box1", "box1", "box2"])  # start, warm-up done, round 0, round 1 -> changed
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0)),
                        identity=lambda: next(ids, "box2"))
    assert r["verdict"] == "INCOMPLETE" and "machine identity changed (before round 2)" in r["rule"]
    assert len(ab_lines(tmp_path, "ABRUN")) == 6  # the session stops; completed rounds stay in ab.log
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "INCOMPLETE" and key["comparisons"][0]["outcome"] == "INCOMPLETE"


def test_extractor_flags_ab_runs_without_a_report(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out/ab.log").write_text("ABRUN round=0 arm=A rc=0 value=1.0 src=runs/r0_A.log\n")
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "INCOMPLETE" and "no ABREPORT" in key["verdict"]["rule"]


def test_check_results_flags_an_incomplete_ab_session(tmp_path, capsys):
    ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0, extra="none")),
                    identity=lambda: "m")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(dict(derived=[], key_extraction=[dict(dir="out", title="t", inputs=["ab.log"])])))
    assert extract_key.main(["--registry", str(registry), "--write"]) == 0
    capsys.readouterr()
    assert check_results.main(["--registry", str(registry)]) == 0
    text = capsys.readouterr().out
    assert "VERDICT INCOMPLETE out:" in text and "NOT COMPLETE (1): out=INCOMPLETE" in text
    # A key.json from the old extractor (COMPLETE with an INCOMPLETE A/B row) fails the check.
    key = json.loads((tmp_path / "out/key.json").read_text())
    key["verdict"] = dict(value="COMPLETE", rule="old extractor")
    (tmp_path / "out/key.json").write_text(json.dumps(key))
    assert check_results.main(["--registry", str(registry)]) == 1


# H2: a timeout does not crash the session.
def test_timeout_is_incomplete_and_keeps_the_completed_runs(tmp_path):
    calls = {"n": 0}
    b_cmd = f"{fake(tmp_path, 'B', 9.0)}; if [ -f {tmp_path}/hang ]; then {fake(tmp_path, 'H', 9.0, extra='sleep')}; fi"
    a = args(tmp_path, fake(tmp_path, "A", 10.0), b_cmd, timeout=2)

    def identity():
        calls["n"] += 1
        if calls["n"] == 3:  # before round 1: make the next B run hang (a child keeps the pipe open)
            (tmp_path / "hang").write_text("1")
        return "m"
    t0 = time.monotonic()
    r = ab_same_box.run(a, identity=identity)
    assert time.monotonic() - t0 < 20
    assert r["verdict"] == "INCOMPLETE" and "round 1 arm B rc=timeout" in r["rule"]
    runs = ab_lines(tmp_path, "ABRUN")
    assert runs[-1].startswith("ABRUN round=1 arm=B rc=timeout")
    assert sum(l.startswith("ABRUN round=0 ") for l in runs) == 3
    assert json.loads((tmp_path / "out/ab.json").read_text())["verdict"] == "INCOMPLETE"
    key = extract(tmp_path)
    assert key["verdict"]["value"] == "INCOMPLETE"
    assert key["runs"][-1]["status"] == "timeout"


# H3: the metric regex must have exactly one group.
@pytest.mark.parametrize("metric", [r"step_ms=[\d.]+", r"(step)_ms=([\d.]+)", r"step_ms=([\d.]+"])
def test_bad_metric_regex_exits_2(tmp_path, metric):
    with pytest.raises(SystemExit) as e:
        ab_same_box.main(["--a", "true", "--b", "true", "--metric", metric, "--out", str(tmp_path)])
    assert e.value.code == 2


def test_label_with_white_space_exits_2(tmp_path):
    with pytest.raises(SystemExit) as e:
        ab_same_box.main(["--a", "true", "--b", "true", "--metric", METRIC, "--label", "a b", "--out", str(tmp_path)])
    assert e.value.code == 2


# H4: --identity-cmd is read at the start, before every round and at the end.
def test_identity_cmd_change_is_incomplete(tmp_path):
    counter = tmp_path / "idn"
    cmd = (f"{sys.executable} -c \"import pathlib; p=pathlib.Path('{counter}'); "
           f"n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); "
           f"print('gpu-host serial', 'X' if n < 4 else 'Y')\"")
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0), identity_cmd=cmd))
    assert r["machine"] == "gpu-host serial X"
    assert r["verdict"] == "INCOMPLETE" and "machine identity changed (before round 3)" in r["rule"]
    assert int(counter.read_text()) == 5  # start + rounds 0..3
    assert not ab_lines(tmp_path, "ABWARN")


def test_identity_cmd_failure_is_incomplete(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0), identity_cmd="exit 7"))
    assert r["verdict"] == "INCOMPLETE" and "--identity-cmd rc=7" in r["rule"]
    assert extract(tmp_path)["verdict"]["value"] == "INCOMPLETE"


def test_identity_cmd_stable_gives_a_verdict(tmp_path):
    r = ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0),
                             identity_cmd="echo gpu1; echo serial 123"))
    assert r["verdict"] == "WIN" and r["machine"] == "gpu1 serial 123"


def test_host_name_only_identity_writes_a_warning(tmp_path):
    ab_same_box.run(args(tmp_path, fake(tmp_path, "A", 10.0), fake(tmp_path, "B", 9.0)), identity=lambda: "laptop")
    assert ab_lines(tmp_path, "ABWARN")


def test_main_end_to_end(tmp_path, capsys):
    rc = ab_same_box.main(["--a", fake(tmp_path, "A", 10.0), "--b", fake(tmp_path, "B", 9.0), "--metric", METRIC,
                           "--identity-cmd", "echo m", "--label", "x", "--out", str(tmp_path / "out")])
    assert rc == 0 and "verdict=WIN" in capsys.readouterr().out
