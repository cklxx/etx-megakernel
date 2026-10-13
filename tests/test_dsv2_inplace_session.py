"""Local gates for the prepared ds_inplace GPU session: the job graph, the decision rule, and every GPU-side operation
run end to end against stand-in binaries that replay RECORDED outputs (archived ETX and fleet logs, the stamped 09-27b
trace), so the parsing, pairing, stage and report code paths are exercised on real data before any box exists."""
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from etx.tools.gpu_jobs import load_session
from examples.dsv2lite import inplace_session as ins
from examples.dsv2lite.inplace_session import full_result, paired

SESSION = Path("examples/sessions/ds_inplace.py")


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
    assert full_result(good_log(fleet))["wall_ms"] == 3.0
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
    assert paired(gains)["go"] is go


def test_job_graph_order_and_inputs():
    jobs, limit = load_session(SESSION)
    by = {j.name: j for j in jobs}
    assert limit == 7200
    assert all(j.gpu == 0 for j in jobs if j.lane == "gpu")
    gpu = [j.name for j in jobs if j.lane == "gpu"]
    assert gpu[:5] == ["prep_ref", "baseline", "pairs_all", "trace_default", "trace_all"]   # decision before attribution
    assert gpu[5:8] == ["pairs_coh", "pairs_d4", "d4trace"]   # D2, D4: new conditions before the 10-13 attributions
    assert by["pairs_coh"].after == ["preflight", "build_default", "build_coh", "build_relaxed"]
    assert by["pairs_coh"].after_any == ["trace_all"]
    assert by["pairs_d4"].after == ["preflight", "build_default", "build_d4"] and by["d4trace"].after == ["preflight", "build_d4"]
    # a skipped after_any predecessor releases its successor at once: every GPU measurement gates HARD on preflight
    # and on every binary it runs (2026-10-13 d1d5 run: the singles started before the data existed)
    for j in jobs:
        if j.lane == "gpu" and j.name != "prep_ref":
            assert "preflight" in j.after, j.name
        if j.name.startswith("pairs_"):
            assert "build_default" in j.after and f"build_{j.name[6:]}" in j.after, j.name
    assert set(gpu[8:]) == {f"pairs_{a}" for a in ins.SINGLE}
    for arm in (*(a for a in ins.ARMS if a not in ("allrelaxed", "d4r0", "alld4r0", "l1", "alld4r0l1", "alld3", "alld4s", "alld5")), "fleet"):   # follow-up session files
        assert f"build_{arm}" in by and by[f"build_{arm}"].after == ["source"]
    for arm in ins.STAMPED:
        assert f"--stamped" in by[f"build_{arm}_stamps"].cmd
    assert "preflight" in by["baseline"].after and "preflight" in by["pairs_all"].after
    assert by["pairs_all"].after_any == ["baseline"] and by["trace_default"].after_any == ["pairs_all"]
    assert "pairs_all" in by["report"].after
    assert all("ETX_DSV2_DATA:=$HOME/dsv2-data" in j.cmd and "ETX_DSV2_FLEET:=$HOME/fleet-3fb67a6" in j.cmd for j in jobs)
    assert all("fleet-mi300x" not in j.cmd for j in jobs)


def test_every_arm_names_levers_present_and_locally_passed():
    from examples.dsv2lite.patch_fleet import levers
    root = Path("examples/dsv2lite/results/levers")
    for arm, names in ins.ARMS.items():
        for name in names:
            assert name in levers(), (arm, name)
            record = json.loads((root / f"{name}.json").read_text())
            assert record.get("verdict", "PASS") == "PASS", (arm, name)
    combined = root / ("+".join(ins.LEVERS_ALL) + ".json")
    rec = json.loads(combined.read_text())
    assert rec["arm_build"]["resources"]["scratch"] == 0


@pytest.mark.parametrize("identity,ok", [
    ("commit " + "a" * 40 + "\nbranch b\ndirty 0\ntree " + "TREE" + "\n", True),
    ("commit " + "a" * 40 + "\nbranch b\ndirty 0\n", False),          # no tree hash: refused (review 07, R08)
    ("commit " + "a" * 40 + "\nbranch b\ndirty 1\n", False),
    ("branch b\ndirty 0\n", False),
    (None, False),
])
def test_source_identity_must_name_a_clean_commit(tmp_path, monkeypatch, identity, ok):
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ETX_DSV2_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ETX_DSV2_FLEET", str(tmp_path / "fleet"))
    session = ins.Session()
    session.root = tmp_path
    (tmp_path / "SOURCE_FILES").write_text("")              # an empty staged tree: its hash is well defined
    if identity is not None:
        (tmp_path / "SOURCE_IDENTITY").write_text(identity.replace("TREE", ins.tree_hash(tmp_path)))
    if ok:
        assert session.source_identity().startswith("commit ")
    else:
        with pytest.raises(RuntimeError, match="SOURCE_IDENTITY|staged tree"):
            session.source_identity()


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
    for arm in ins.ARMS:
        _exe(logs / f"b_{arm}" / "run", f'cat "{etx_log}"\n')
    _exe(logs / "b_fleet" / "run", f'cat "{fleet_log}"\n')
    for arm in ins.STAMPED:           # the stamped builds replay the stamped 09-27b token (trace + prologue stamps)
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
    for arm in ins.STAMPED:
        s.trace(arm)
    for arm in ins.SINGLE:
        s.pairs(arm)
    s.report()
    report = json.loads((recorded / "report.json").read_text())
    # identical recorded logs on both sides: every paired gain is exactly zero, so no GO
    assert report["all"]["device_ms"]["default"]["gains_us"] == [0.0] * 5
    assert report["all"]["device_ms"]["default"]["go"] is False
    assert report["baseline"]["device_ms"]["default"]["median_us"] == pytest.approx(1000 * (3.474 - 3.545))
    assert all(report[a]["device_ms"]["default"]["go"] is None for a in ins.SINGLE)
    stages = {r["stage"]: r for r in report["stages"]}
    assert len(stages) == 14 and all(r["measured_saving_us"] == 0 for r in stages.values())
    st = json.loads((recorded / "stages_default.json").read_text())
    assert abs(sum(st["stages_us"].values()) - st["span_us"]) < 1e-6          # the stamped partition is exact
    # every invocation kept its full log; nothing was overwritten
    assert len(list(recorded.glob("*.raw.log"))) == 10 + 15 + 2 + 10 * len(ins.SINGLE)


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
    assert seen["compile_alld5"]["ETX_FLEET_PATCH"] == seen["compile_all"]["ETX_FLEET_PATCH"] == " ".join(ins.LEVERS_ALL)


def test_every_decided_arm_reports_one_verdict():
    """A decided arm must never look undecided: pairs JSONs carry `verdict`, re-derivable from the old fields."""
    from examples.dsv2lite.inplace_session import RULES, verdict
    keep = {"every_positive": True, "median_us": 61.0, "spread_us": 8.0, "keep": True}
    rep = {"device_ms": {"default": dict(keep, go=None), "all": dict(keep, go=None)}, "keep_vs_all": keep}
    assert verdict("alld4r0", rep) == dict(rule=RULES["alld4r0"][0], outcome="KEEP", passed=True)
    assert verdict("alld4r0", dict(rep, keep_vs_all=dict(keep, keep=False)))["outcome"] == "NO-GO"
    assert verdict("d4r0", rep)["outcome"] == "NO RULE (attribution)"
    assert {"all", "coh", "l1", "d4", "alld4r0", "alld4r0l1", "alld5"} <= set(RULES)


@pytest.mark.parametrize("arm", ["all", "coh", "l1", "d4", "alld4r0", "alld4r0l1", "alld4s", "alld3", "alld5", "relaxed"])
def test_pairs_runs_end_to_end_for_every_arm_shape(tmp_path, monkeypatch, arm):
    """pairs() with stubbed invocations: every arm writes its JSON, a verdict and its PAIRS line (2026-10-13: an
    arm-shape bug after five GPU rounds would lose the arm)."""
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path / k))
    ms = {"default": 3.67, "fleet": 3.54, "all": 3.56, "relaxed": 3.44, "alld4r0": 3.50}
    def fake(self, a, name):
        return {"device_ms": ms.get(a, 3.40), "wall_ms": ms.get(a, 3.40) + 0.03}
    monkeypatch.setattr(ins.Session, "full_invocation", fake)
    monkeypatch.setattr(ins.Session, "out", lambda self, a, stamped=False: tmp_path / f"b_{a}")
    (tmp_path / "b_fleet").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b_fleet" / "run").write_text("")
    ins.Session().pairs(arm)
    rep = json.loads((tmp_path / "ETX_SESS_LOGDIR" / f"pairs_{arm}.json").read_text())
    assert "verdict" in rep and rep["verdict"]["outcome"]




@pytest.mark.parametrize("gains,outcome,neutral,any_slower", [
    ([30, 30, 30, 30, 30], "KEEP", False, False),      # equality at the minimum median is allowed
    ([29, 29, 29, 29, 29], "DROP", False, False),      # positive and stable, but below 30 us
    ([20, 30, 30, 30, 35], "DROP", False, False),      # median == 2 x spread is insufficient
    ([20, 30, 30, 30, 34], "KEEP", False, False),
    ([0, 0, 0, 0, 0], "DROP", True, False),
    ([10, 30, 30, 30, 40], "DROP", True, False),       # |median| == spread is explicitly neutral
    ([-1, 40, 40, 40, 40], "DROP", True, True),        # any slower pair drops the arm
    ([0, 40, 40, 40, 40], "DROP", True, False),        # a tie cannot satisfy every pair faster
    ([-40, -40, -40, -40, -40], "DROP", False, True),
])
def test_d5_registered_keep_and_drop_rule(gains, outcome, neutral, any_slower):
    decision = ins.d5_verdict(paired(gains))
    assert decision["neutral"] is neutral and decision["any_slower"] is any_slower
    assert decision["keep"] is (outcome == "KEEP")
    assert decision["drop"] is (outcome == "DROP")
    assert ins.verdict("alld5", {"d5": decision}) == dict(rule=ins.RULES["alld5"][0], outcome=outcome,
                                                        passed=outcome == "KEEP")


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
            ins.main(["pairs", "alld5"])
        assert not path.exists()                 # no verdict from nine correct invocations
    else:
        ins.main(["pairs", "alld5"])
        report = json.loads(path.read_text())
        assert set(report["device_ms"]) == set(report["wall_ms"]) == {"all"}
        assert report["device_ms"]["all"]["median_us"] == pytest.approx(50)
        assert report["device_ms"]["all"]["go"] is None
        assert report["verdict"]["outcome"] == "KEEP"
        assert all(set(r) == {"all", "alld5"} for r in report["rounds"])
        assert all(v["tokens"] == 32 and v["layers"] == 27 for r in report["rounds"] for v in r.values())
    assert calls == [(arm, f"alld5_r{i}_{arm}") for i in range(5)
                     for arm in (("all", "alld5") if i % 2 == 0 else ("alld5", "all"))]


def test_the_paid_hour_session_order_and_gates():
    """examples/sessions/ds_hour.py: the D2b rerun first, D3 / D5 next, the stamped record, the optional arm last, a
    box-side hash job at the end; every GPU job gated hard on preflight and on each binary it runs."""
    jobs, limit = load_session(Path("examples/sessions/ds_hour.py"))
    assert limit == 1800
    gpu = [j for j in jobs if j.lane == "gpu" and j.name != "prep_ref"]
    assert [j.name for j in gpu] == ["pairs_l1", "pairs_alld4r0l1", "pairs_alld3", "pairs_alld5", "trace_default",
                                     "trace_alld4r0", "trace_alld4r0l1", "pairs_alld4s"]
    by = {j.name: j for j in jobs}
    for j in gpu:
        assert "preflight" in j.after and all(dep in by for dep in j.after), j.name
    assert by["pairs_alld3"].after == ["preflight", "build_all", "build_alld3"]
    assert by["trace_alld4r0l1"].after == ["preflight", "build_alld4r0l1_stamps"]
    # review 07: the hash waits for EVERY other job (R06); each GPU job waits for ALL earlier ones (R07)
    assert by["hash_session"].lane == "cpu" and set(by["hash_session"].after_any) == {j.name for j in jobs} - {"hash_session"}
    for k, j in enumerate(gpu):
        assert set(j.after_any) == {g.name for g in gpu[:k]}, j.name
    assert "'startup.*'" in by["hash_session"].cmd                              # R05: the runner's live file
    for arm in ("l1", "alld4r0l1", "alld3", "alld5", "alld4s"):
        assert arm in ins.RULES, arm                    # every candidate measured in the hour has a rule


def _hour_dir(tmp_path, monkeypatch, arm, ms):
    """Run the real pairs() with a stubbed binary that writes genuine raw logs; box-side SHA256SUMS like hash_session."""
    import hashlib
    logs = tmp_path / "sess"
    for k, v in (("ETX_SESS_LOGDIR", logs), ("ETX_DSV2_FLEET", tmp_path / "f"), ("ETX_DSV2_DATA", tmp_path / "d")):
        monkeypatch.setenv(k, str(v))

    def fake_run(self, name, command, **kw):
        build = Path(command[0]).parent.name.removeprefix("b_")
        text = good_log().replace("median 2.900", f"median {ms[build]:.3f}")
        (self.logs / f"{name}.raw.log").write_text(text)
        return text
    monkeypatch.setattr(ins.Session, "run", fake_run)
    ins.Session().pairs(arm)
    rows = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(logs)}"
            for p in sorted(logs.rglob("*")) if p.is_file()]
    (logs / "SHA256SUMS").write_text("\n".join(rows) + "\n")
    return logs


def test_verify_hour_rederives_every_verdict_and_catches_a_changed_log(tmp_path, monkeypatch):
    """review 07, R10: the hour's own verifier -- same parser, same report code, on the raw logs."""
    from examples.dsv2lite import verify_hour
    d = _hour_dir(tmp_path, monkeypatch, "alld3", {"all": 3.500, "alld3": 3.440})
    assert verify_hour.main([str(d)]) == 0
    assert json.loads((d / "pairs_alld3.json").read_text())["verdict"]["outcome"] == "KEEP"
    raw = d / "alld3_r2_alld3.raw.log"
    raw.write_text(raw.read_text().replace("median 3.440", "median 3.300"))
    assert verify_hour.main([str(d)]) == 2                  # both the box hash and the re-derived pairs disagree


@pytest.mark.parametrize("ref,cand,keep", [
    ([3.500] * 5, [3.440, 3.440, 3.440, 3.440, 3.410], False),   # gains 60 x4, 90: median 60 == 2 x spread 30 -> fail
    ([3.500] * 5, [3.470] * 5, True),                            # 30 us exactly, spread 0: passes the >= 30 bar
])
def test_exact_boundaries_decide_as_written(ref, cand, keep):
    """review 07, R02: binary floating point must not move an inclusive or strict boundary."""
    from examples.dsv2lite.inplace_session import d4_verdict, gain_us
    v = d4_verdict(paired([gain_us(r, c) for r, c in zip(ref, cand)]))
    assert v["keep"] is keep


def test_tree_hash_matches_the_staging_pipeline_and_catches_a_change(tmp_path):
    """review 07, R08: the box's source check recomputes the staged tree's hash; it must equal stage_inplace.sh's
    `git ls-files -z | xargs -0 shasum -a 256 | shasum -a 256`, and a changed tracked file must be refused."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    shell = subprocess.run("git ls-files -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-64", shell=True,
                           cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    stage = tmp_path / "etx"
    for name in files.splitlines()[:40]:                    # a staged tree: a slice of the real files
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        (stage / name).write_bytes((root / name).read_bytes())
    (stage / "SOURCE_FILES").write_text("\n".join(files.splitlines()[:40]) + "\n")
    full = tmp_path / "full"
    full.mkdir()
    (full / "SOURCE_FILES").write_text(files)
    for name in files.splitlines():
        (full / name).parent.mkdir(parents=True, exist_ok=True)
        (full / name).symlink_to(root / name)
    assert ins.tree_hash(full) == shell
    h = ins.tree_hash(stage)
    s = object.__new__(ins.Session)
    s.root = stage
    (stage / "SOURCE_IDENTITY").write_text(f"commit {'a' * 40}\nbranch b\ndirty 0\ntree {h}\nstaged t\n")
    assert s.source_identity()
    victim = stage / files.splitlines()[3]
    victim.write_bytes(victim.read_bytes() + b"\n# changed on the box\n")
    with pytest.raises(RuntimeError, match="not the staged tree"):
        s.source_identity()
    (stage / "SOURCE_IDENTITY").write_text(f"commit {'a' * 40}\nbranch b\ndirty 0\nstaged t\n")   # no tree line
    with pytest.raises(RuntimeError, match="not the staged tree"):
        s.source_identity()


def test_preflight_refuses_inputs_that_are_not_the_measured_ones(tmp_path, monkeypatch):
    """review 07, R09: a consistent but different input set passes its own SHA256SUMS; it must not pass preflight."""
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
