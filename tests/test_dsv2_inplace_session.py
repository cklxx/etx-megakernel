"""Local gates for the prepared ds_inplace GPU session: the job graph, the decision rule, and every GPU-side operation
run end to end against stand-in binaries that replay RECORDED outputs (archived ETX and fleet logs, the stamped 09-27b
trace), so the parsing, pairing, stage and report code paths are exercised on real data before any box exists."""
from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from etx.experiments.jobs import load_session
from etx.models.dsv2lite import session as ins
from examples.dsv2lite import inplace_session as cli
from etx.session.experiment import decide, ns, report, verify
from examples.dsv2lite import arms as registry
from examples.dsv2lite.arms import ARMS, LEVERS_ALL, SINGLE, full_result

SESSION = Path("examples/sessions/ds_inplace.py")
STAMPED = ("default", "all")


def rounds_for(arm, gains_us):
    return [{**{r: {"device_ms": "3.000", "wall_ms": "3.030"} for r in arm.references},
             arm.cand: {"device_ms": str(Decimal("3.000") - Decimal(g) / 1000),
                        "wall_ms": str(Decimal("3.030") - Decimal(g) / 1000)}} for g in gains_us]


def good_log(fleet=False):
    warm = "7.000 ms total, " if fleet else ""
    wall = ("wall time of the launches: 96.000 ms for 32 tokens = 3.000 ms/token (333 tok/s), 1 launch(es)"
            if fleet else "step wall (launch to end) 3.000 ms/token")
    return (f"run 0 (warm-up): {warm}32/32 tokens match\n"
            "  consecutive layers within the gate: 27 of 27\n"
            "per-token latency (embed -> argmax on device): median 2.900 ms mean 2.9 ms; " + wall + "\n"
            "tokens: 32/32 match HF greedy\n")


@pytest.mark.parametrize("fleet", [False, True])
def test_both_log_formats_require_all_gates(fleet):
    assert full_result(good_log(fleet))["wall_ms"] == "3.000"
    for old, new in [("27 of 27", "26 of 27"), ("tokens: 32/32", "tokens: 31/32"),
                     ("32/32 tokens match", "31/32 tokens match"), ("median 2.900", "median 0.000")]:
        with pytest.raises(ValueError):
            full_result(good_log(fleet).replace(old, new))


@pytest.mark.parametrize("gains,go", [
    ([75, 76, 74, 77, 75], True),             # every gain positive, median 75 >= 70, spread 3 < 37.5
    ([75, 76, -1, 77, 75], False),            # one negative pair
    ([65, 66, 64, 67, 65], False),            # median below 70
    ([75, 160, 40, 100, 71], False),          # median 75 but spread 120 > median / 2
])
def test_decision_rule(gains, go):
    assert decide(ARMS["all"].rule, rounds_for(ARMS["all"], gains), "all")["passed"] is go


def test_job_graph_order_and_inputs():
    jobs, limit = load_session(SESSION)
    by = {j.name: j for j in jobs}
    assert limit == 7200
    assert all(j.gpu == 0 for j in jobs if j.lane == "gpu")
    gpu = [j.name for j in jobs if j.lane == "gpu"]
    assert gpu[:5] == ["prep_ref", "baseline", "pairs_all", "trace_default", "trace_all"]   # decision before attribution
    assert gpu[5:8] == ["pairs_coh", "pairs_d4", "d4trace"]   # D2, D4: new conditions before the 10-13 attributions
    assert set(gpu[8:]) == {f"pairs_{a}" for a in SINGLE}
    assert by["pairs_coh"].reads == ["inputs.json", "b_default/run", "b_relaxed/run", "b_coh/run"]
    assert by["pairs_all"].reads == ["inputs.json", "b_default/run", "b_all/run"]       # fleet: context only
    assert by["baseline"].reads == ["inputs.json", "b_default/run", "b_fleet/run"]
    assert by["d4trace"].reads == ["inputs.json", "b_d4/run"]
    # the 2026-10-13 d1d5 loss: every GPU measurement reads inputs.json, which only a passed preflight writes, and
    # preflight reads the five data files their prep jobs write
    assert by["preflight"].writes == ["inputs.json"] and len(by["preflight"].reads) == len(ins.DATA_FILES)
    assert set(by["preflight"].reads) == set(by["prep_ref"].writes) | set(by["prep_pack"].writes)
    for j in jobs:
        if j.lane == "gpu" and j.name != "prep_ref":
            assert j.reads[0] == "inputs.json" and all(r.endswith("/run") for r in j.reads[1:]), j.name
        if j.name.startswith("build_"):
            assert j.reads == ["source.json"] and j.writes == [f"b_{j.name[6:]}/run"], j.name
    for arm in STAMPED:
        assert "--stamped" in by[f"build_{arm}_stamps"].cmd
    assert by["report"].lane == "final" and by["report"].reads == ["pairs_all.json"]
    assert all(j.env["ETX_DSV2_DATA"] and j.env["ETX_DSV2_FLEET"] for j in jobs)
    assert all("fleet-mi300x" not in j.cmd for j in jobs)


def test_every_arm_names_levers_present_and_locally_passed():
    from examples.dsv2lite.patch_fleet import levers
    root = Path("examples/dsv2lite/results/levers")
    for arm, spec in ARMS.items():
        for name in spec.levers:
            assert name in levers(), (arm, name)
            record = json.loads((root / f"{name}.json").read_text())
            assert record.get("verdict", "PASS") == "PASS", (arm, name)
    combined = root / ("+".join(LEVERS_ALL) + ".json")
    rec = json.loads(combined.read_text())
    assert rec["arm_build"]["resources"]["scratch"] == 0


@pytest.mark.parametrize("name,identity,ok", [
    ("A-1", "commit " + "a" * 40 + "\narchive A\n", True),
    ("B-1", "commit " + "a" * 40 + "\narchive A\n", False),        # not the directory that archive was extracted to
    ("A-1", "commit " + "a" * 40 + "\n", False),                    # no archive hash
    ("A-1", None, False),
])
def test_source_identity_is_the_staged_archive(tmp_path, name, identity, ok):
    root = tmp_path / name.replace("A", "a" * 64).replace("B", "b" * 64)
    root.mkdir()
    if identity is not None:
        (root / "SOURCE_IDENTITY").write_text(identity.replace("archive A", "archive " + "a" * 64))
    s = object.__new__(ins.Session)
    s.root = root
    if ok:
        assert s.source_identity().startswith("commit ")
    else:
        with pytest.raises(RuntimeError, match="not a staged archive"):
            s.source_identity()


def _exe(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A session directory whose arm binaries replay archived outputs, and GNU shims for timeout/stdbuf (macOS)."""
    from examples.dsv2lite.pinned import data_sets
    try:
        data = data_sets()
    except Exception as exc:
        pytest.skip(f"archived data sets unavailable: {exc}")
    b, d = data / "results_2026-09-27b", data / "results_2026-09-27d"
    logs = tmp_path / "logs"
    shims = tmp_path / "shims"
    _exe(shims / "timeout", 'while [[ "$1" == --* ]]; do shift; done; shift; exec "$@"\n')
    _exe(shims / "stdbuf", 'while [[ "$1" == -* ]]; do shift; done; exec "$@"\n')
    monkeypatch.setenv("PATH", f"{shims}:{os.environ['PATH']}")
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(logs))
    monkeypatch.setenv("ETX_DSV2_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ETX_DSV2_FLEET", str(tmp_path / "fleet"))
    etx_log, fleet_log = d / "sess2/t1_x1s.log", b / "t1_fleet.log"
    for arm in ARMS:
        _exe(logs / f"b_{arm}" / "run", f'cat "{etx_log}"\n')
    _exe(logs / "b_fleet" / "run", f'cat "{fleet_log}"\n')
    for arm in STAMPED:           # the stamped builds replay the stamped 09-27b token (trace + prologue stamps)
        out = logs / f"b_{arm}_stamps"
        _exe(out / "run", f'cp "{b / "etx_trace.bin"}" "$ETX_TRACE_RAW"; cp "{b / "etx_pro.bin"}" "$ETX_DS_PROLOGUE"; '
                          f'cat "{b / "t1_relaxed.log"}"\n')
        shutil.copy(b / "etx_plan.json", out / "plan.json")
    return logs, etx_log


def test_gpu_operations_run_end_to_end_on_recorded_outputs(recorded):
    recorded, _ = recorded
    s = ins.Session()
    s.pairs("baseline")
    s.pairs("all")
    for arm in STAMPED:
        s.trace(arm)
    for arm in SINGLE:
        s.pairs(arm)
    s.report()
    report = json.loads((recorded / "report.json").read_text())
    # identical recorded logs on both sides: every paired gain is exactly zero, so no GO
    assert report["all"]["device_ms"]["default"]["gains_us"] == [0.0] * 5
    assert report["all"]["decision"]["passed"] is False
    assert report["baseline"]["device_ms"]["default"]["median_us"] == -71.0
    assert all(report[a]["verdict"]["outcome"] == "NO RULE (attribution)" for a in SINGLE)
    stages = {r["stage"]: r for r in report["stages"]}
    assert len(stages) == 14 and all(r["measured_saving_us"] == 0 for r in stages.values())
    st = json.loads((recorded / "stages_default.json").read_text())
    assert abs(sum(st["stages_us"].values()) - st["span_us"]) < 1e-6          # the stamped partition is exact
    # every invocation kept its full log; nothing was overwritten
    assert len(list(recorded.glob("*.raw.log"))) == 10 + 15 + 2 + 10 * len(SINGLE)


def test_pairs_refuses_a_failed_correctness_gate(recorded):
    logs, etx_log = recorded
    _exe(logs / "b_all" / "run", f'sed "s/27 of 27/26 of 27/" "{etx_log}"\n')
    with pytest.raises(ValueError, match="27/27"):
        ins.Session().pairs("all")


def test_build_env_lets_an_arm_switch_its_plan_on(tmp_path, monkeypatch):
    """2026-10-13 d1d5 run: build coh / build d4 died with "multiple values for keyword argument" before compiling."""
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path / k))
    seen = {}

    class Stop(Exception):
        pass

    def fake_run(self, name, command, *, env=None, timeout=180):
        seen[name] = env
        raise Stop

    monkeypatch.setattr(ins.Session, "run", fake_run)
    for arm in ("default", "coh", "d4", "relaxed", "all", "alld5"):
        with pytest.raises(Stop):
            ins.Session().build(arm)
    assert seen["compile_default"]["ETX_DSV2_COH"] == "0" and seen["compile_default"]["ETX_DSV2_D4"] == "0"
    assert seen["compile_coh"]["ETX_DSV2_COH"] == "1" and seen["compile_coh"]["ETX_FLEET_PATCH"] == "coherent_loads"
    assert seen["compile_d4"]["ETX_DSV2_D4"] == "1" and seen["compile_d4"]["ETX_DSV2_COH"] == "0"
    assert seen["compile_default"]["ETX_DSV2_D5"] == seen["compile_all"]["ETX_DSV2_D5"] == "0"
    assert seen["compile_alld5"]["ETX_DSV2_D5"] == "1"
    assert seen["compile_alld5"]["ETX_FLEET_PATCH"] == seen["compile_all"]["ETX_FLEET_PATCH"] == " ".join(LEVERS_ALL)


def test_every_decided_arm_reports_one_verdict():
    assert {"all", "coh", "l1", "d4", "alld4r0", "alld4r0l1", "alld3", "alld4s", "alld5"} <= {a for a in ARMS if ARMS[a].rule}
    for arm in ARMS.values():
        if arm.rule:
            rounds = rounds_for(arm, [100] * 5)
            if arm.rule.equal_to:
                for r in rounds:
                    r[arm.rule.equal_to] = r[arm.cand]
            rep = report(arm, rounds, registry.CONDITION)
            assert rep["verdict"] == dict(rule=arm.rule.text, outcome=arm.rule.labels[0], passed=True)
            assert report(arm, rounds_for(arm, [0] * 5), registry.CONDITION)["verdict"]["passed"] is False
    arm = ARMS["d4r0"]
    assert report(arm, rounds_for(arm, [61] * 5), "test")["verdict"]["outcome"] == "NO RULE (attribution)"


@pytest.mark.parametrize("arm", ["all", "coh", "l1", "d4", "alld4r0", "alld4r0l1", "alld4s", "alld3", "alld5", "relaxed"])
def test_pairs_runs_end_to_end_for_every_arm_shape(tmp_path, monkeypatch, arm):
    """pairs() with stubbed invocations: every arm writes its JSON, a verdict and its PAIRS line (2026-10-13: an
    arm-shape bug after five GPU rounds would lose the arm)."""
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path / k))
    ms = {"default": "3.67", "fleet": "3.54", "all": "3.56", "relaxed": "3.44", "alld4r0": "3.50"}
    def fake(self, a, name):
        return {"device_ms": ms.get(a, "3.40"), "wall_ms": str(Decimal(ms.get(a, "3.40")) + Decimal("0.03"))}
    monkeypatch.setattr(ins.Session, "full_invocation", fake)
    monkeypatch.setattr(ins.Session, "out", lambda self, a, stamped=False: tmp_path / f"b_{a}")
    (tmp_path / "b_fleet").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b_fleet" / "run").write_text("")
    ins.Session().pairs(arm)
    rep = json.loads((tmp_path / "ETX_SESS_LOGDIR" / f"pairs_{arm}.json").read_text())
    assert "verdict" in rep and rep["verdict"]["outcome"]


@pytest.mark.parametrize("gains,outcome", [
    ([30, 30, 30, 30, 30], "KEEP"),      # equality at the minimum median is allowed
    ([29, 29, 29, 29, 29], "DROP"),      # positive and stable, but below 30 us
    ([20, 30, 30, 30, 35], "DROP"),      # median == 2 x spread is insufficient
    ([20, 30, 30, 30, 34], "KEEP"),
    ([0, 0, 0, 0, 0], "DROP"),
    ([10, 30, 30, 30, 40], "DROP"),       # |median| == spread is explicitly neutral
    ([-1, 40, 40, 40, 40], "DROP"),        # any slower pair drops the arm
    ([0, 40, 40, 40, 40], "DROP"),        # a tie cannot satisfy every pair faster
    ([-40, -40, -40, -40, -40], "DROP"),
])
def test_d5_registered_keep_and_drop_rule(gains, outcome):
    arm = ARMS["alld5"]
    rep = report(arm, rounds_for(arm, gains), registry.CONDITION)
    assert rep["decision"]["passed"] is (outcome == "KEEP")
    assert rep["verdict"] == dict(rule=arm.rule.text, outcome=outcome, passed=outcome == "KEEP")


@pytest.mark.parametrize("failure", [None, "layers", "tokens", "warmup", "incomplete"])
def test_d5_two_way_pairs_require_every_invocation_complete(tmp_path, monkeypatch, failure):
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path / k))
    calls = []

    def fake_run(self, name, command, **kwargs):
        arm = Path(command[0]).parent.name.removeprefix("b_")
        calls.append((arm, name))
        log = good_log().replace("median 2.900", "median 2.850" if arm == "alld5" else "median 2.900")
        if len(calls) == 10:
            if failure == "incomplete":
                raise RuntimeError("invocation deadline: NOT COMPLETE")
            old, new = {"layers": ("27 of 27", "26 of 27"), "tokens": ("tokens: 32/32", "tokens: 31/32"),
                        "warmup": ("32/32 tokens match", "31/32 tokens match")}.get(failure, ("", ""))
            if old:
                log = log.replace(old, new)
        return log

    monkeypatch.setattr(ins.Session, "run", fake_run)
    path = Path(os.environ["ETX_SESS_LOGDIR"]) / "pairs_alld5.json"
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            cli.main(["pairs", "alld5"])
        assert not path.exists()                 # no verdict from nine correct invocations
    else:
        cli.main(["pairs", "alld5"])
        report = json.loads(path.read_text())
        assert set(report["device_ms"]) == set(report["wall_ms"]) == {"all"}
        assert report["device_ms"]["all"]["median_us"] == pytest.approx(50)
        assert "go" not in report["device_ms"]["all"]
        assert report["verdict"]["outcome"] == "KEEP"
        assert all(set(r) == {"all", "alld5"} for r in report["rounds"])
        assert all(v["tokens"] == 32 and v["layers"] == 27 for r in report["rounds"] for v in r.values())
    assert calls == [(arm, f"alld5_r{i}_{arm}") for i in range(5)
                     for arm in (("all", "alld5") if i % 2 == 0 else ("alld5", "all"))]


def test_the_paid_hour_session_order_and_gates():
    """examples/sessions/ds_hour.py: the D2b rerun first, D3 / D5 next, the stamped record, the optional arm last; every
    GPU job reads inputs.json and each binary it runs; the GPU lane keeps this order (R07); the runner seals."""
    jobs, limit = load_session(Path("examples/sessions/ds_hour.py"))
    assert limit == 1800
    gpu = [j for j in jobs if j.lane == "gpu" and j.name != "prep_ref"]
    assert [j.name for j in gpu] == ["pairs_l1", "pairs_alld4r0l1", "pairs_alld3", "pairs_alld5", "trace_default",
                                     "trace_alld4r0", "trace_alld4r0l1", "pairs_alld4s"]
    writers = {w for j in jobs for w in j.writes}
    for j in gpu:
        assert j.reads[0] == "inputs.json" and set(j.reads) <= writers, j.name
    by = {j.name: j for j in jobs}
    assert by["pairs_alld3"].reads == ["inputs.json", "b_all/run", "b_alld3/run"]
    assert by["trace_alld4r0l1"].reads == ["inputs.json", "b_alld4r0l1_stamps/run"]
    assert "hash_session" not in by and "prep_sums" not in by
    for arm in ("l1", "alld4r0l1", "alld3", "alld5", "alld4s"):
        assert ARMS[arm].rule is not None, arm                    # every candidate measured in the hour has a rule


def _hour_dir(tmp_path, monkeypatch, arm, ms):
    """Run the real pairs() with a stubbed binary that writes genuine raw logs, then seal the record as the runner does."""
    from etx.session import record
    logs = tmp_path / "sess"
    for k, v in (("ETX_SESS_LOGDIR", logs), ("ETX_DSV2_FLEET", tmp_path / "f"), ("ETX_DSV2_DATA", tmp_path / "d")):
        monkeypatch.setenv(k, str(v))

    def fake_run(self, name, command, **kw):
        build = Path(command[0]).parent.name.removeprefix("b_")
        text = good_log().replace("median 2.900", f"median {ms[build]}")
        (self.logs / f"{name}.raw.log").write_text(text)
        return text
    monkeypatch.setattr(ins.Session, "run", fake_run)
    ins.Session().pairs(arm)
    record.seal(logs, jobs={f"pairs_{arm}": "ok"}, plan="test")
    return logs


def test_verify_hour_rederives_every_verdict_and_catches_a_changed_log(tmp_path, monkeypatch):
    """review 07, R10: the hour's own verifier -- same parser, same report code, on the raw logs."""
    session = Path("examples/sessions/ds_hour.py")
    d = _hour_dir(tmp_path, monkeypatch, "alld3", {"all": "3.500", "alld3": "3.440"})
    # one arm of the eight promised results: what exists verifies, but the hour is INCOMPLETE (review 08, R13)
    rc, lines = verify(d, registry, session=session)
    assert rc == 3 and len([line for line in lines if line.startswith("MISSING ")]) == 7
    assert any(line.startswith("MISSING stages_alld4r0l1.json") for line in lines)
    assert json.loads((d / "pairs_alld3.json").read_text())["verdict"]["outcome"] == "KEEP"
    raw = d / "alld3_r2_alld3.raw.log"
    raw.write_text(raw.read_text().replace("median 3.440", "median 3.300"))
    assert verify(d, registry, session=session)[0] == 2                  # both the box hash and the re-derived pairs disagree


def test_a_stale_file_in_the_record_copy_fails_verification(tmp_path, monkeypatch):
    """review 08, R12: a file the sealed MANIFEST does not list is not this session's."""
    from etx.session.record import verify
    d = _hour_dir(tmp_path, monkeypatch, "alld3", {"all": "3.500", "alld3": "3.440"})
    assert verify(d)[0] == 0
    (d / "pairs_alld5.json").write_text("{}")                # left over from an earlier fetch
    rc, lines = verify(d)
    assert rc == 2 and any("NOT IN THE SEALED MANIFEST pairs_alld5.json" in line for line in lines)


@pytest.mark.parametrize("ref,cand,keep", [
    (["3.500"] * 5, ["3.440", "3.440", "3.440", "3.440", "3.410"], False),   # gains 60 x4, 90: median 60 == 2 x spread 30 -> fail
    (["3.500"] * 5, ["3.470"] * 5, True),                            # 30 us exactly, spread 0: passes the >= 30 bar
])
def test_exact_boundaries_decide_as_written(ref, cand, keep):
    """review 07, R02: binary floating point must not move an inclusive or strict boundary."""
    rounds = [{"default": {"device_ms": r}, "d4": {"device_ms": c}} for r, c in zip(ref, cand)]
    assert decide(ARMS["d4"].rule, rounds, "d4")["passed"] is keep


def test_rule_uses_declared_basis_strictness_and_exact_nanoseconds():
    arm = ARMS["d4"]
    rounds = rounds_for(arm, [30, 30, 30, 30, 45])
    assert not decide(arm.rule, rounds, "d4")["passed"]
    assert decide(replace(arm.rule, strict=False), rounds, "d4")["passed"]
    for r in rounds:
        r["d4"]["wall_ms"] = r["default"]["wall_ms"]
    assert not decide(replace(arm.rule, basis="wall_ms", strict=False), rounds, "d4")["passed"]
    assert ns("3.000001") == 3000001
    assert not decide(arm.rule, rounds_for(arm, ["29.999"] * 5), "d4")["passed"]
    for text in ("3.0000001", "3.499999999999999999999999999999999999999", "3.500000000000000000000000000000000000001"):
        with pytest.raises(ValueError, match="exact integer nanosecond"):
            ns(text)
    with pytest.raises(TypeError, match="never floats"):
        ns(3.0)


def test_preflight_refuses_inputs_that_are_not_the_measured_ones(tmp_path, monkeypatch):
    """review 07, R09: a consistent but different input set (even with its own SHA256SUMS) must not pass preflight."""
    import hashlib
    data = tmp_path / "data"
    data.mkdir()
    for k, v in (("ETX_SESS_LOGDIR", tmp_path / "s"), ("ETX_DSV2_FLEET", tmp_path / "f"), ("ETX_DSV2_DATA", data)):
        monkeypatch.setenv(k, str(v))
    contents = {"weights.bin": b"w" * 64, "weights.bin.manifest": b"m", "fleet_cache.bin": b"c",
                "golden_tokens.txt": " ".join(["1"] * 33).encode(), "golden_hidden.bin": b"\0" * (27 * 2048 * 4)}
    for name, body in contents.items():
        (data / name).write_bytes(body)
    (data / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in contents.items()))
    with pytest.raises(RuntimeError, match="is not the measured one"):
        ins.Session().preflight()
    assert set(ins.PINNED_INPUTS) == set(ins.DATA_FILES)
