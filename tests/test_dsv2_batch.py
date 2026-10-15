"""Batch 2 / 4 / 8 on DeepSeek-V2-Lite (docs/BATCH-2026-10-14.md): the multi-request graph's queues, the per-request
gates and rules (etx/session/batch.py), the vLLM batch mode's lines, the golden checks and the session's job graph."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import pytest

from etx.session import batch
from examples.dsv2lite.arms import ARMS, BATCH_ARMS, BATCHES, LEVERS_ALL

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("binding", ["requests=8", "requestsz=8"])
def test_deepseek_cli_rejects_unsupported_bindings_before_compilation(tmp_path, monkeypatch, binding):
    from etx.tools import cli
    monkeypatch.setattr(cli, "compile_graph", lambda *args, **kwargs: pytest.fail("unsupported binding reached compiler"))
    with pytest.raises(ValueError, match="does not accept --bind.*" + binding.split("=")[0]):
        cli.main(["compile", str(ROOT / "examples/dsv2lite/graph.py"), "--out", str(tmp_path), "--bind", binding])
    from examples.dsv2lite import graph
    assert graph.bindings() == {}


# ---------------------------------------------------------------- the graph: slots, queue order, LDS hand-offs
@pytest.fixture(scope="module", params=[2, 4, 8])
def lowered(request):
    import os
    from etx.passes import PassOptions, compile_graph
    from examples.dsv2lite import graph
    old = {k: os.environ.get(k) for k in ("ETX_DSV2_COUPLE", "ETX_DSV2_COH")}
    os.environ.update(ETX_DSV2_COUPLE="none", ETX_DSV2_COH="l1")
    try:
        g = graph.build(layers=3, requests=request.param, routing_cached=True)
        plan = compile_graph(g, "gfx942", {}, {}, PassOptions(relay="on"))
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return request.param, g, plan


def _req(name):
    return int(name.split("_", 1)[0][1:])


def test_every_request_runs_on_its_slot_two_workgroups_per_cu(lowered):
    n, g, plan = lowered
    assert plan.wg_per_cu == 2
    per = plan.workers_per_domain
    for t in plan.tasks:
        r, local = _req(t.grid), t.worker - t.domain * per
        assert (r % 2) * 38 <= local < (r % 2) * 38 + 37, (t.grid, t.worker, t.domain)


def test_static_queues_keep_creation_order_so_in_body_waits_cannot_deadlock(lowered):
    """Fleet's attention waits for its q_absorb INSIDE its body; with requests sharing a slot that is safe only if every
    queue runs tasks in grid creation order and each qabs grid is created before its attention grid."""
    n, g, plan = lowered
    order = {gr.name: i for i, gr in enumerate(g.grids)}
    tasks = {t.id: t for t in plan.tasks}
    assert all(t.mode == "static" for t in tasks.values())
    for queue in plan.static_queues.values():
        idx = [order[tasks[i].grid] for i in queue]
        assert idx == sorted(idx)
    for name in order:
        if "_attn_" in name:
            assert order[name.replace("_attn_", "_qabs_")] < order[name]
    assert {_req(x) for x in order if x[0] == "r"} == set(range(n))


def test_routing_handoff_stays_adjacent_on_its_worker(lowered):
    """expert_down_cached reads the routing its worker's gate_up left in LDS: nothing may run between them."""
    n, g, plan = lowered
    tasks = {t.id: t for t in plan.tasks}
    count = 0
    for queue in plan.static_queues.values():
        for i, tid in enumerate(queue):
            gu = tasks[tid]
            if "_gate_up_" in gu.grid:
                down = tasks[queue[i + 1]]
                assert down.grid == gu.grid.replace("_gate_up_", "_down_") and down.coord == gu.coord
                count += 1
    moe_layers = sum("_gate_up_" in gr.name for gr in g.grids)
    assert count == moe_layers * 8 * 37 and moe_layers == n * 2


@pytest.mark.parametrize("env,match", [(dict(ETX_DSV2_COUPLE="router"), "ETX_DSV2_COUPLE=none"),
                                       (dict(ETX_DSV2_COUPLE="experts"), "ETX_DSV2_COUPLE=none"),
                                       (dict(ETX_DSV2_COUPLE="none", ETX_DSV2_D4="1"), "one request")])
def test_more_than_two_requests_refuse_what_names_two_or_one(monkeypatch, env, match):
    from examples.dsv2lite import graph
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        graph.build(layers=2, requests=4)


def test_shared_kva_is_allowed_on_a_batched_graph(monkeypatch):
    # docs/BATCH-2026-10-14.md section 10: the batch arms carry K1, so a shared kv_a must build for 4 requests
    from examples.dsv2lite import graph
    for k, v in dict(ETX_DSV2_COUPLE="none", ETX_DSV2_KVA="shared").items():
        monkeypatch.setenv(k, v)
    g = graph.build(layers=2, requests=4)
    assert len(g.grids) > 0


def test_two_requests_keep_the_historic_slots():
    from examples.dsv2lite.graph import CUS, _Req
    assert [(_Req(r, 2).off, _Req(r, 2).p) for r in range(2)] == [(0, "r0_"), (CUS, "r1_")]
    assert _Req(0, 1).off == 0 and _Req(0, 1).p == ""


def test_batch_arms_carry_the_kept_recipe_without_d4():
    kept = ARMS["alld4r0l1"]
    assert BATCH_ARMS == {1: "alld4r0l1kvsmw", 2: "x2", 4: "x4", 8: "x8"} and BATCHES == (1, 2, 4, 8)
    # section 10: the batch arms carry the kept arm's two levers (M1, K1) as well, so a scaling ratio measures
    # batching and not a recipe difference
    for b in (2, 4):
        a = ARMS[f"x{b}"]
        assert a.levers == (*kept.levers, "merge_uv_waves")
        assert "-DETX_DSV2_KVA_SHARED=1" in a.flags and a.env["ETX_DSV2_KVA"] == "shared"
        assert a.flags[:2] == (f"-DETX_DS_REQUESTS={b}", "-DETX_SLOT_BY_CU")
        assert a.rule is None and a.order == ()
    # B = 8: the gate failed with M1 + K1 and with M1 alone, so the four recipes are built and gated side by side
    # (examples/sessions/x8_diag.py); all four carry the same request flags and the same scratch allowance
    for name, extra, k1 in (("x8p", (), False), ("x8m", ("merge_uv_waves",), False),
                            ("x8k", (), True), ("x8", ("merge_uv_waves",), True)):
        a = ARMS[name]
        assert a.levers == (*kept.levers, *extra), name
        assert (("-DETX_DSV2_KVA_SHARED=1",) if k1 else ()) == tuple(f for f in a.flags if "KVA" in f), name
        assert a.env["ETX_DSV2_REQUESTS"] == "8" and (a.env.get("ETX_DSV2_KVA") == "shared") == k1, name
        assert a.scratch_allow == 192, name
    assert [ARMS[f"x{b}"].scratch_allow for b in (2, 4)] == [64, 64] and kept.scratch_allow == 0
    b1 = ARMS[BATCH_ARMS[1]]                 # section 9: B = 1 is the kept arm, K1 + M1 on the same recipe
    assert b1.levers == (*kept.levers, "merge_uv_waves") and "-DETX_DSV2_KVA_SHARED=1" in b1.flags
    assert b1.levers == ARMS["x2"].levers and b1.levers == ARMS["x4"].levers == ARMS["x8"].levers


@pytest.mark.parametrize("config", [None, "", "/models/qwen config.json", "/models/qwen'$(touch $ETX_N6_MARKER);config.json"])
def test_session_start_transports_qwen_config_without_shell_expansion(tmp_path, config):
    import os
    import subprocess
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    ssh.write_text("#!/usr/bin/env python3\n"
                   "import os, subprocess, sys\n"
                   "sys.stdin.read()\n"
                   "env = dict(os.environ)\n"
                   "env.pop('ETX_QWEN_G6_CONFIG', None)\n"
                   "env.pop('ETX_SESS_MAX_SECONDS', None)\n"
                   "probe = \"python3 -c 'import json,os; print(json.dumps({k: os.environ.get(k) for k in "
                   "[\\\"ETX_QWEN_G6_CONFIG\\\", \\\"ETX_SESS_MAX_SECONDS\\\"]}))' > \\\"$TRANSPORT_CAPTURE\\\"\\n\"\n"
                   "raise SystemExit(subprocess.run(['bash', '-c', ' '.join(sys.argv[6:])], "
                   "input=probe, text=True, env=env).returncode)\n")
    ssh.chmod(0o755)
    capture = tmp_path / "transport.json"
    marker = tmp_path / "expanded"
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               TRANSPORT_CAPTURE=str(capture), ETX_N6_MARKER=str(marker), ETX_SESS_MAX_SECONDS="1200")
    env.pop("ETX_QWEN_G6_CONFIG", None)
    if config is not None:
        env["ETX_QWEN_G6_CONFIG"] = config
    result = subprocess.run(["bash", str(ROOT / "tools/session.sh"), "start", "offline", "session.py"],
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(capture.read_text()) == {"ETX_QWEN_G6_CONFIG": config, "ETX_SESS_MAX_SECONDS": "1200"}
    assert not marker.exists()


# ---------------------------------------------------------------- per-request gates (ETX)
def etx_log(b, tokens="32/32", layers=27, warm="32/32", steps="5.120000", intervals=None, extra=""):
    lines = [f"BATCHREQ r={r} tokens={tokens if r == b - 1 else '32/32'} layers={layers} warmup={warm} "
             f"interval_median_ms={(intervals or {}).get(r, '5.10000')} device_median_ms=3.200000" for r in range(b)]
    return "\n".join([*lines, f"BATCH requests={b} tokens=32 step_median_ms={steps}", extra]) + "\n"


def test_etx_result_takes_the_worst_requests_interval_and_the_host_step():
    r = batch.etx_result(etx_log(4, intervals={2: "5.30000"}), 4)
    assert r["step_ms"] == "5.120000" and r["latency_ms"] == "5.30000" and r["worst_request"] == 2
    assert sorted(r["requests"]) == [0, 1, 2, 3]


@pytest.mark.parametrize("log,b", [
    (etx_log(4, tokens="31/32"), 4), (etx_log(4, layers=26), 4), (etx_log(4, warm="31/32"), 4),
    (etx_log(4, extra="r3 token  7:  MISMATCH (HF 5)"), 4), (etx_log(4), 8), (etx_log(4).replace("BATCHREQ r=3", "BATCHREQ r=2"), 4),
    (etx_log(2, steps="0.000000"), 2), (etx_log(2).replace("BATCH requests", "XBATCH requests"), 2)])
def test_one_failing_request_fails_the_invocation(log, b):
    with pytest.raises(batch.Incomplete):
        batch.etx_result(log, b)


# ---------------------------------------------------------------- vLLM batch mode: its lines round-trip into the rule
def vllm_log(label="default", batches=(1, 2, 4, 8), per=("5.1", "5.0", "5.3", "5.2", "5.4"), diverge=None):
    from examples.llm.vllm_bench import batch_lines
    out = []
    for b in batches:
        goldens = [list(range(100 + r, 133 + r)) for r in range(b)]
        for phase in ["warmup", *(f"sample{k}" for k in range(5))]:
            outs = [g[:32] for g in goldens]
            if diverge and diverge == (b, phase):
                outs[-1] = outs[-1][:7] + [0] * 25
            out += batch_lines(label, b, phase, goldens, outs, 32)
            out += batch_lines(label, b, phase + "_short", goldens, [g[:1] for g in goldens], 1)
        out.append(f"VLLMB label={label} batch={b} per_token_ms={','.join(f'{float(x) * b:.6f}' for x in per)}")
    return "\n".join(out) + "\n"


def test_vllm_lines_round_trip_and_the_median_is_one_of_the_samples():
    r = batch.vllm_results(vllm_log(), "default")
    assert sorted(r) == [1, 2, 4, 8] and all(x["legal"] for x in r.values())
    assert r[2]["step_ms"] == "10.400000" and r[8]["step_ms"] == "41.600000"


def test_a_diverged_sequence_makes_that_batch_size_illegal_only():
    r = batch.vllm_results(vllm_log(diverge=(4, "sample3")), "default")
    assert not r[4]["legal"] and "diverged" in r[4]["reason"] and "7/32" in r[4]["reason"]
    assert r[2]["legal"] and r[8]["legal"]


def test_a_missing_phase_is_illegal():
    log = "\n".join(x for x in vllm_log().splitlines() if "phase=sample4 seq=1 " not in x or "batch=2 " not in x)
    assert not batch.vllm_results(log, "default")[2]["legal"]


def test_selection_takes_the_faster_legal_configuration_per_batch_size():
    d = batch.vllm_results(vllm_log("default", per=("6",) * 5), "default")
    a = batch.vllm_results(vllm_log("aiter", per=("5",) * 5, diverge=(8, "warmup")), "aiter")
    sel = batch.select({"default": d, "aiter": a}, BATCHES)
    assert [sel[b]["config"] for b in BATCHES] == ["aiter", "aiter", "aiter", "default"]
    none = batch.select({"default": batch.vllm_results(vllm_log(diverge=(2, "warmup")), "default")}, (2,))
    assert none[2]["config"] is None and none[2]["verdict"] == "NO LEGAL BASELINE"


# ---------------------------------------------------------------- the rule
@pytest.mark.parametrize("gains,outcome", [
    ([Fraction(3, 100)] * 5, "WIN"), ([Fraction(2, 100)] * 5, "WIN"), ([Fraction(199, 10000)] * 5, "TIE"),
    ([Fraction(-3, 100)] * 5, "LOSS"), ([Fraction(x, 100) for x in (5, 6, 7, 8, 9)], "TIE"),        # median 7 <= 2 x 4
    ([Fraction(x, 100) for x in (5, 5, 6, 6, 6)], "WIN"), ([Fraction(x, 100) for x in (-1, 5, 5, 5, 5)], "TIE")])
def test_verdict_needs_one_sign_two_percent_and_twice_the_spread(gains, outcome):
    assert batch.verdict(gains)["outcome"] == outcome


def test_gain_is_exact_and_signed_for_etx():
    assert batch.gain("6.000000", "5.000000") == Fraction(1, 5)
    assert batch.gain("5.000000", "6.000000") == Fraction(-1, 6)


def rounds_for(etx_steps, vllm_steps, kept="3.200000"):
    out = []
    for k in range(5):
        etx = {1: batch.etx_result(etx_log(1, steps=kept, intervals={0: "3.30000"}), 1)}
        for b, s in etx_steps.items():
            etx[b] = batch.etx_result(etx_log(b, steps=s[k] if isinstance(s, list) else s,
                                             intervals={0: s[k][:-1] if isinstance(s, list) else s[:-1]}), b)
        out.append(dict(etx=etx, vllm={"aiter": {b: dict(step_ms=v, legal=True, reason=None) for b, v in vllm_steps.items()}}))
    return out


def test_study_reports_both_quantities_and_the_scalings():
    sel = {b: dict(config="aiter") for b in BATCHES}
    rounds = rounds_for({2: "4.800000", 4: "9.000000", 8: "17.000000"},
                        {1: "4.200000", 2: "5.300000", 4: "5.800000", 8: "7.000000"})
    res = batch.study(rounds, sel, BATCHES)
    assert res[2]["throughput"]["outcome"] == "WIN" and res[4]["throughput"]["outcome"] == "LOSS"
    assert res[8]["latency"]["outcome"] == "LOSS"
    assert res[2]["etx_tokens_per_s"] == pytest.approx(2000 / 4.8)
    assert res[2]["etx_throughput_scaling"] == pytest.approx(2 * 3.2 / 4.8)
    assert "step time" in res[2]["vllm_latency_note"]


def test_one_failed_invocation_makes_its_batch_size_not_complete():
    sel = {b: dict(config="aiter") for b in BATCHES}
    rounds = rounds_for({2: "4.800000", 4: "9.000000", 8: "17.000000"},
                        {1: "4.2", 2: "5.3", 4: "5.8", 8: "7.0"})
    rounds[3]["etx"][4] = dict(error="request 3: tokens 31/32")
    rounds[1]["vllm"]["aiter"] = dict(error="DEADLINE")
    res = batch.study(rounds, sel, BATCHES)
    assert res[4]["status"] == "NOT COMPLETE" and any("31/32" in p for p in res[4]["problems"])
    assert all(res[b]["status"] == "NOT COMPLETE" for b in BATCHES)          # round 2's vLLM process is gone for every B
    sel[8] = dict(config=None, reasons={"aiter": "diverged"})
    assert batch.study(rounds, sel, (8,))[8]["status"] == "NO LEGAL BASELINE"


def test_build_decision_follows_section_5():
    rec = dict(batch=8, status="COMPLETE", throughput=dict(outcome="LOSS"), vllm_step_ms=["7.000000"] * 5)
    yes = batch.build_decision(rec, "3.200000", 13.2e9)          # 3.2 + 13.2 GB / 3.5 TB/s = 6.971 ms > 6.3: no
    assert yes["build"] is False and abs(yes["projection_ms"] - (3.2 + 13.2e9 / 3.5e9)) < 1e-6
    assert batch.build_decision(rec, "3.200000", 10.0e9)["build"] is True       # 6.057 <= 6.3
    assert "PENDING" in batch.build_decision(rec, "3.200000", None)["next"]
    assert "trace" in batch.build_decision(dict(rec, batch=2), "3.2", 1.0)["next"]
    assert "replication" in batch.build_decision(dict(rec, throughput=dict(outcome="WIN")), "3.2", 1.0)["next"]
    assert "rerun" in batch.build_decision(dict(batch=4, status="NOT COMPLETE"), "3.2", 1.0)["next"]


# ---------------------------------------------------------------- goldens, floors, host lines
def test_goldens_refuse_a_wrong_rotation_a_duplicate_and_a_token_file_mismatch():
    from examples.dsv2lite.batch_goldens import check, collisions, rotate
    p0 = list(range(1024))
    prompts = [rotate(p0, r) for r in range(8)]
    gens = [[r] * 33 for r in range(8)]
    assert check(prompts, gens, gens) == []
    assert collisions(gens) == []
    assert any("rotated" in x for x in check([p0, p0, *prompts[2:]], gens, gens))
    assert any("same prompt" in x for x in check([p0, p0, *prompts[2:]], gens, gens))
    assert any("differs" in x for x in check(prompts, gens, [gens[0], [9] * 33, *gens[2:]]))
    # an identical 32-token greedy output is reported as a collision, never refused (docs/BATCH-2026-10-14.md 4a)
    dup = [gens[0], gens[0], *gens[2:]]
    assert check(prompts, dup, dup) == []
    assert collisions(dup) == [[0, 1]]


def test_floors_match_the_design_table():
    from examples.dsv2lite.batch_floor import floors
    f = floors()
    assert [f[b]["replicated_floor_ms"] for b in BATCHES] == [1.148, 2.295, 4.59, 9.18]
    assert [f[b]["batched_floor_ms"] for b in BATCHES] == [1.148, 1.724, 2.721, 4.221]


def test_the_host_prints_exactly_the_fields_the_rule_reads():
    src = (ROOT / "examples/dsv2lite/host.hip").read_text()
    assert 'BATCHREQ r=%d tokens=%d/%d layers=%d warmup=%d/%d interval_median_ms=%llu.%05llu device_median_ms=%.6f' in src
    assert 'BATCH requests=%d tokens=%d step_median_ms=%.6f' in src
    # an identical short greedy output is a printed collision, not a refusal (docs/BATCH-2026-10-14.md 4a); prompt
    # distinctness is enforced by examples/dsv2lite/batch_goldens.py before the host ever runs
    assert "BATCHCOLLISION requests %d and %d" in src and '--req-dir' in src


def test_rerun_includes_required_artifact_producers_before_their_consumers(tmp_path, monkeypatch):
    import runpy
    from etx.tools.gpu_jobs import Runner, load_jobs
    path = ROOT / "examples/sessions/rerun_skips.py"
    all_jobs = runpy.run_path(str(path))["_every"]
    writers = {w.lstrip("?"): name for name, job in all_jobs.items() for w in job.get("writes", [])}
    jobs = load_jobs(path)     # Also rejects a same-GPU producer placed after its consumer.
    by = {j.name: j for j in jobs}
    order = {j.name: i for i, j in enumerate(jobs)}
    assert {"preflight", "build_x2", "build_x4", "build_x8", "build_alld4r0l1kvsmw"} <= set(by)
    for job in jobs:
        for read in job.reads:
            key = read.lstrip("?")
            producer = key if key in all_jobs else None if read.startswith("?") else writers.get(key)
            if producer is not None:
                assert producer in by, (job.name, read, producer)
                assert order[producer] < order[job.name], (job.name, read, producer)
    assert "routing" not in by and "qwen_moerows" not in by
    runner = Runner(jobs, [0], 2, tmp_path)
    present = tmp_path / "present"
    present.touch()
    monkeypatch.setattr(runner, "path", lambda job, key: present)
    for job in jobs:
        job.status = "ok"
    build, gate = by["build_x2"], by["bgate_x2"]
    build.status = "running"
    assert runner._state(gate) == "wait"
    build.status = "fail"
    assert runner._state(gate) == "skip" and "build_x2 fail" in gate.note
    build.status = "ok"
    assert runner._state(gate) == "ready"


# ---------------------------------------------------------------- the session
@pytest.fixture(scope="module")
def session():
    from etx.tools.gpu_jobs import load_session
    jobs, limit = load_session(ROOT / "examples/sessions/ds_batch.py")
    return {j.name: j for j in jobs}, jobs, limit


def test_session_order_correctness_first_then_timings_then_captures(session):
    by, jobs, limit = session
    assert limit == 5400
    assert [j.name for j in jobs if j.lane == "gpu"] == ["prep_ref", "prep_goldens", "bgate_x2", "bgate_x4", "bgate_x8", "boxcontrol",
                                                          "vsel", "brounds", "routing"]
    assert {f"build_{a}" for a in (BATCH_ARMS[1], "alld4r0l1", "x2", "x4", "x8")} <= set(by)
    assert all(by[f"bgate_{a}"].reads[1] == "$ETX_DSV2_DATA/goldens.json" for a in ("x2", "x4", "x8"))
    assert set(by["brounds"].reads) == {"vsel.json", "boxcontrol.json", "b_alld4r0l1kvsmw/run", "?bgate_x2.json", "?bgate_x4.json", "?bgate_x8.json"}
    assert "docker_pull" in by["vsel"].reads and by["breport"].lane == "final"


def test_boxcontrol_job_declares_both_builds_and_preserves_its_recipe_difference(session):
    import re
    by, jobs, _ = session
    control = by["boxcontrol"]
    assert control.cmd.endswith("batch_session boxcontrol")
    assert set(control.reads) == {"inputs.json", f"b_{BATCH_ARMS[1]}/run", "b_alld4r0l1/run"}
    assert control.writes == ["boxcontrol.json"]
    assert "boxcontrol.json" in by["brounds"].reads
    assert any(re.search(pattern, "BOXCONTROL recipe_difference device_ms=0.1") for pattern in control.summary)
    assert jobs.index(by["build_alld4r0l1"]) < jobs.index(control) < jobs.index(by["brounds"])


def test_boxcontrol_prints_exact_host_medians_and_signed_recipe_difference(tmp_path, monkeypatch, capsys):
    import re
    from examples.dsv2lite import batch_session as bs
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path))
    # Follow the registry's current shipped arm, rather than baking today's recipe into the mode.
    monkeypatch.setattr(bs, "BATCH_ARMS", {**BATCH_ARMS, 1: "shipped"})
    s = bs.BatchSession()
    measurements = {"shipped": ("3.200007", "3.300011"), "alld4r0l1": ("3.240013", "3.350017")}
    calls = []

    def fake_run(name, command, **kw):
        arm = Path(command[0]).parent.name.removeprefix("b_")
        calls.append((name, command))
        dev, wall = measurements[arm]
        # Human wall output is a mean; interval latency is yet another basis. Neither is the wall median.
        return etx_log(1, steps=wall, intervals={0: "9.99999"}, extra=
                       "per-token latency (embed -> argmax on device): median 9.000 ms  mean 9.001 ms; "
                       "step wall (launch to end) 9.002 ms/token").replace("device_median_ms=3.200000",
                                                                                    f"device_median_ms={dev}")

    monkeypatch.setattr(s, "run", fake_run)
    s.boxcontrol()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    parsed = {}
    for line in lines[:2]:
        match = re.fullmatch(r"BOXCONTROL arm=(\S+) b=1 device_median_ms=([\d.]+) wall_median_ms=([\d.]+) "
                             r"tokens=32/32 layers=27/27", line)
        assert match, line
        parsed[match[1]] = (match[2], match[3])
    assert parsed == measurements
    offset = re.fullmatch(r"BOXCONTROL recipe_difference device_ms=([+-]?[\d.]+) wall_ms=([+-]?[\d.]+) pct=([+-]?[\d.]+) "
                          r"between shipped and alld4r0l1", lines[2])
    assert offset
    assert offset.group(1, 2) == ("-0.040006", "-0.050006")
    assert float(offset[3]) == pytest.approx(-0.050006 / 3.350017 * 100, abs=0.000001)
    assert len(calls) == 2
    assert [name for name, _ in calls] == ["boxcontrol_shipped", "boxcontrol_alld4r0l1"]
    assert all(command[1:] == s.args() and "--req-dir" not in command for _, command in calls)
    saved = json.loads((tmp_path / "boxcontrol.json").read_text())
    assert set(saved["arms"]) == set(measurements)
    assert "offset" not in saved
    assert saved["measurement_kind"] == "within_session_recipe_difference"
    assert saved["invocations_per_arm"] == 1
    assert saved["machine_offset_measured"] is False
    assert saved["recipe_difference"]["between"] == ["shipped", "alld4r0l1"]
    with pytest.raises(RuntimeError, match="fresh ETX_SESS_LOGDIR"):
        s.boxcontrol()
    assert len(calls) == 2


@pytest.mark.parametrize("missing", [BATCH_ARMS[1], "alld4r0l1"])
def test_boxcontrol_refuses_a_missing_arm_without_partial_numbers(tmp_path, monkeypatch, capsys, missing):
    from examples.dsv2lite import batch_session as bs
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path))
    s = bs.BatchSession()

    def fake_run(name, command, **kw):
        if Path(command[0]).parent.name == f"b_{missing}":
            raise RuntimeError("binary is missing")
        return etx_log(1)

    monkeypatch.setattr(s, "run", fake_run)
    with pytest.raises(RuntimeError, match=f"BOXCONTROL.*{missing}.*binary is missing"):
        s.boxcontrol()
    assert not any(line.startswith("BOXCONTROL ") for line in capsys.readouterr().out.splitlines())
    assert not (tmp_path / "boxcontrol.json").exists()


@pytest.mark.parametrize("damage", [dict(tokens="31/32"), dict(layers=26), dict(warm="31/32")])
def test_boxcontrol_refuses_failed_correctness_gates(tmp_path, monkeypatch, capsys, damage):
    from examples.dsv2lite import batch_session as bs
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path))
    s = bs.BatchSession()
    monkeypatch.setattr(s, "run", lambda *args, **kwargs: etx_log(1, **damage))
    with pytest.raises(RuntimeError, match=f"BOXCONTROL.*{BATCH_ARMS[1]}"):
        s.boxcontrol()
    assert not any(line.startswith("BOXCONTROL ") for line in capsys.readouterr().out.splitlines())
    assert not (tmp_path / "boxcontrol.json").exists()


def test_boxcontrol_cli_runs_without_an_arm(monkeypatch):
    from types import SimpleNamespace
    from examples.dsv2lite import batch_session as bs
    calls = []
    monkeypatch.setattr(bs, "BatchSession", lambda: SimpleNamespace(boxcontrol=lambda: calls.append("boxcontrol")))
    bs.main(["boxcontrol"])
    assert calls == ["boxcontrol"]


def test_rounds_alternate_and_skip_an_arm_whose_gate_failed(tmp_path, monkeypatch):
    from examples.dsv2lite import batch_session as bs
    for k, v in dict(ETX_SESS_LOGDIR=str(tmp_path), ETX_DSV2_FLEET=str(tmp_path), ETX_DSV2_DATA=str(tmp_path)).items():
        monkeypatch.setenv(k, v)
    s = bs.BatchSession()
    (tmp_path / "vsel.json").write_text(json.dumps(dict(selection={str(b): dict(config="aiter") for b in BATCHES})))
    (tmp_path / "bgate_x2.json").write_text(json.dumps(dict(passed=True)))
    (tmp_path / "bgate_x4.json").write_text(json.dumps(dict(passed=False)))   # x8's gate never ran (no file)
    calls = []
    monkeypatch.setattr(s, "etx", lambda b, name: calls.append(name) or dict(step_ms="1.0", latency_ms="1.0"))
    monkeypatch.setattr(s, "vllm", lambda c, name: calls.append(name) or {})
    s.rounds()
    assert calls[:3] == ["b1_alld4r0l1kvsmw", "b1_x2", "b1_vllm_aiter"] and calls[3:6] == ["b2_vllm_aiter", "b2_x2", "b2_alld4r0l1kvsmw"]
    assert len(calls) == 15
    rec = json.loads((tmp_path / "brounds.json").read_text())
    assert rec["skipped"] == ["x4", "x8"]
    with pytest.raises(RuntimeError, match="fresh ETX_SESS_LOGDIR"):
        s.rounds()


def test_gate_runs_the_arm_it_names(tmp_path, monkeypatch):
    """x8p / x8m / x8k / x8 must each gate their own binary.

    The four labels once all executed b_x8/run (the batch number alone selected the arm), so one measurement was
    written four times under four names and the run appeared to say that the plain arm fails exactly like the lever
    arms. This test is the local check that each gate invocation points at b_<arm>/run.
    """
    from examples.dsv2lite import batch_session as bs
    for k, v in dict(ETX_SESS_LOGDIR=str(tmp_path), ETX_DSV2_FLEET=str(tmp_path), ETX_DSV2_DATA=str(tmp_path)).items():
        monkeypatch.setenv(k, v)
    s = bs.BatchSession()
    seen = []

    def fake_run(name, command, **kw):
        seen.append((name, str(command[0])))
        raise RuntimeError("stub: no GPU here")

    monkeypatch.setattr(s, "run", fake_run)
    for arm in bs.X8_CANDIDATES:
        s.gate(arm)
    assert [name for name, _ in seen] == [f"bgate_{a}" for a in ("x8p", "x8m", "x8k", "x8")]
    assert [path for _, path in seen] == [str(tmp_path / f"b_{a}" / "run") for a in ("x8p", "x8m", "x8k", "x8")]
    for arm in ("x8p", "x8m", "x8k", "x8"):
        rec = json.loads((tmp_path / f"bgate_{arm}.json").read_text())
        assert rec["arm"] == arm and rec["batch"] == 8 and rec["passed"] is False


@pytest.mark.parametrize("name", ["x8_diag", "x8_diag_qwen"])
def test_candidate_gate_declares_its_own_build(name):
    from etx.tools.gpu_jobs import load_session
    jobs, _ = load_session(ROOT / f"examples/sessions/{name}.py")
    by = {j.name: j for j in jobs}
    for arm in ("x8p", "x8m", "x8k", "x8"):
        gate = by[f"bgate_{arm}"]
        assert gate.cmd.endswith(f"gate {arm}")
        assert {p for p in gate.reads if p.startswith("b_")} == {f"b_{arm}/run"}


def test_routing_union_uses_the_executed_request_prefix():
    import ast
    from types import SimpleNamespace
    from itertools import combinations
    tree = ast.parse((ROOT / "examples/dsv2lite/routing_overlap.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "mean_union")
    scope = dict(a=SimpleNamespace(batch=8), combinations=combinations)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "routing_overlap.py", "exec"), scope)
    disjoint = [list(range(6 * r, 6 * r + 6)) for r in range(4)]
    step = disjoint + [disjoint[0]] * 4
    assert scope["mean_union"](2, [step]) == 12
    assert scope["mean_union"](4, [step]) == 24
    assert scope["mean_union"](8, [step]) == 24


def test_report_does_not_project_from_legacy_subset_means(tmp_path, monkeypatch):
    from examples.dsv2lite import batch_session as bs
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path))
    s = bs.BatchSession()
    selection = {str(b): dict(config="aiter") for b in BATCHES}
    (tmp_path / "vsel.json").write_text(json.dumps(dict(selection=selection)))
    rounds = rounds_for({4: "9.0"}, {1: "4.2", 4: "6.0"})
    (tmp_path / "brounds.json").write_text(json.dumps(dict(rounds=rounds)))
    (tmp_path / "routing.json").write_text(json.dumps(dict(overall={"4": 15})))
    s.report()
    report = json.loads((tmp_path / "batch_report.json").read_text())
    assert "PENDING" in report["batches"]["4"]["next"]["next"]


@pytest.mark.parametrize("damage", ["mixed", "missing", "prompt"])
def test_golden_certificate_binds_prompt_and_execution_bytes(tmp_path, damage):
    import hashlib
    import numpy as np
    from examples.dsv2lite import batch_goldens as bg
    for r in range(2):
        d = bg.request_dir(tmp_path, r)
        d.mkdir(exist_ok=True)
        prompt = bg.rotate(list(range(128)), r)
        np.savez(d / "golden.npz", prompt_ids=prompt, token_ids=[7] * 33)
        (d / "golden_tokens.txt").write_text(" ".join(["7"] * 33))
        for name in ("fleet_cache.bin", "golden_hidden.bin"):
            (d / name).write_bytes(f"{name}: request {r}".encode())
        cert = dict(prompt_ids=prompt, sha256={f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in bg.FILES})
        (d / "golden.binding.json").write_text(json.dumps(cert))
    if damage == "mixed":
        for name in ("fleet_cache.bin", "golden_hidden.bin"):
            (tmp_path / "r1" / name).write_bytes((tmp_path / name).read_bytes())
    elif damage == "missing":
        (tmp_path / "r1/golden.binding.json").unlink()
    else:
        cert["prompt_ids"] = list(range(128))
        (tmp_path / "r1/golden.binding.json").write_text(json.dumps(cert))
    with pytest.raises(SystemExit, match="request 1:.*binding"):
        bg.main([str(tmp_path), "--requests", "2"])


def test_fresh_golden_binding_verifies_and_exports(tmp_path):
    import numpy as np
    from examples.dsv2lite import batch_goldens as bg
    np.savez(tmp_path / "golden.npz", prompt_ids=list(range(128)), token_ids=[7] * 33)
    (tmp_path / "golden_tokens.txt").write_text(" ".join(["7"] * 33))
    for name in ("fleet_cache.bin", "golden_hidden.bin"):
        (tmp_path / name).write_bytes(b"fresh reference")
    bg.main([str(tmp_path), "--certify"])
    bg.main([str(tmp_path), "--verify-binding"])
    bg.main([str(tmp_path), "--requests", "1"])
    assert (tmp_path / "vllm/golden_r0.txt").read_text().startswith("prompt 0 1 2")


@pytest.mark.parametrize("bad_short", range(6))
def test_each_short_vllm_invocation_is_gated(tmp_path, monkeypatch, capsys, bad_short):
    import sys
    from types import ModuleType, SimpleNamespace
    from examples.llm import vllm_bench as vb
    clock = [0.0]
    shorts = []
    vllm = ModuleType("vllm")
    vllm.__version__ = "test"
    vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)

    class LLM:
        def __init__(self, **kw):
            pass

        def generate(self, prompts, sp, **kw):
            clock[0] += sp.max_tokens * 0.001
            tokens = list(range(100, 100 + sp.max_tokens))
            if sp.max_tokens == 1:
                if len(shorts) == bad_short:
                    tokens = [999]
                shorts.append(tokens)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=tokens)]) for _ in prompts]

    vllm.LLM = LLM
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(vb.time, "perf_counter", lambda: clock[0])
    golden = tmp_path / "golden.txt"
    golden.write_text("prompt 1 2 3\ngen " + " ".join(map(str, range(100, 133))) + "\n")
    a = SimpleNamespace(goldens=[golden], batches="1", model="test", tokens=32, eager=False,
                        echo_config=False, label="default", repeats=5, no_detok=False, json=None)
    vb.batch_mode(a, {})
    assert len(shorts) == 6
    result = batch.vllm_results(capsys.readouterr().out, "default")[1]
    assert not result["legal"] and "diverged" in result["reason"]


@pytest.mark.parametrize("sample", ["0", "-1", "-0.000001"])
def test_nonpositive_vllm_sample_is_illegal(sample):
    log = vllm_log(batches=(1,), per=("5", "5", sample, "5", "5"))
    result = batch.vllm_results(log, "default")[1]
    assert not result["legal"] and "nonpositive" in result["reason"]


@pytest.mark.parametrize("damage", ["failure_first", "failure_last", "extra_index", "extra_phase"])
def test_duplicate_or_unexpected_sequence_gate_cannot_pass(damage):
    log = vllm_log(batches=(1,))
    line = next(x for x in log.splitlines() if "phase=sample0 seq=0 " in x)
    if damage.startswith("failure"):
        line = line.replace("32/32", "7/32")
    elif damage == "extra_index":
        line = line.replace("seq=0", "seq=1")
    else:
        line = line.replace("phase=sample0", "phase=sample99")
    log = line + "\n" + log if damage == "failure_first" else log + line + "\n"
    result = batch.vllm_results(log, "default")[1]
    assert not result["legal"]


def test_legacy_verification_keeps_the_supported_rederivation_marker(tmp_path):
    from etx.session import experiment
    from examples.dsv2lite import arms
    (tmp_path / "SHA256SUMS").write_text("")
    rc, lines = experiment.verify(tmp_path, arms)
    assert rc == 3                         # Legacy byte integrity remains incomplete.
    assert any(line.startswith("RE-DERIVED:") and "every available" not in line for line in lines)


def test_sealed_batch_report_is_explicitly_excluded_from_semantic_verification(tmp_path):
    from etx.session import experiment, record
    from examples.dsv2lite import arms
    (tmp_path / "batch_report.json").write_text(json.dumps(dict(batches={"8": dict(status="COMPLETE", throughput="WIN")})))
    rounds = [dict(etx={"8": dict(error="gate failed")}, vllm={}) for _ in range(5)]
    (tmp_path / "brounds.json").write_text(json.dumps(dict(rounds=rounds)))
    record.seal(tmp_path, jobs={"breport": "ok"}, plan="test")
    rc, lines = experiment.verify(tmp_path, arms)
    assert rc == 3
    assert any("EXCLUDED batch_report.json" in line and "raw" in line for line in lines)
    assert not any("every available verdict" in line for line in lines)


def test_verifier_checks_the_promised_batch_report(tmp_path):
    from etx.session import experiment, record
    from examples.dsv2lite import arms
    record.seal(tmp_path, jobs={"breport": "ok"}, plan="test")
    rc, lines = experiment.verify(tmp_path, arms, ROOT / "examples/sessions/ds_batch.py")
    assert rc == 3
    assert any("MISSING batch_report.json" in line for line in lines)


def test_baseline_refuses_a_model_other_than_the_prepared_model(tmp_path, monkeypatch):
    from examples.dsv2lite import batch_session as bs
    for k in ("ETX_SESS_LOGDIR", "ETX_DSV2_FLEET", "ETX_DSV2_DATA"):
        monkeypatch.setenv(k, str(tmp_path))
    other = tmp_path / "different-model"
    prepared = tmp_path / "prepared-model"
    monkeypatch.setattr(bs, "MODEL", prepared)
    monkeypatch.setenv("ETX_DSV2_MODEL", str(other))
    s = bs.BatchSession()
    calls = []
    monkeypatch.setattr(s, "run", lambda *a, **kw: calls.append(a) or "")
    with pytest.raises(ValueError) as exc:
        s.vllm("default", "model_probe")
    assert str(other) in str(exc.value)
    assert str(prepared) in str(exc.value)
    assert not calls


def test_standalone_diagnostic_prepares_its_goldens():
    from etx.tools.gpu_jobs import load_session
    jobs, _ = load_session(ROOT / "examples/sessions/x8_diag.py")
    names = [j.name for j in jobs]
    assert "prep_goldens" in names
    assert all(names.index("prep_goldens") < names.index(f"bgate_{a}") for a in ("x8p", "x8m", "x8k", "x8"))


@pytest.mark.parametrize("changed", [0, 32])
def test_collision_warning_covers_either_checked_token_window(changed):
    from examples.dsv2lite.batch_goldens import collisions
    goldens = [list(range(33)), list(range(33))]
    goldens[1][changed] = 999
    assert collisions(goldens) == [[0, 1]]


@pytest.mark.parametrize("changed,expected", [(0, "1"), (32, "0")])
def test_host_collision_compares_its_checked_suffix(tmp_path, changed, expected):
    import subprocess
    toolchain = Path("/opt/homebrew/opt/llvm/bin/clang++")
    src = (ROOT / "examples/dsv2lite/host.hip").read_text()
    line = next(x.strip() for x in src.splitlines() if x.strip().startswith("if (") and "q.golden" in x and "reqs[o].golden" in x)
    condition = line[len("if ("):-1]
    cpp = tmp_path / "collision.cpp"
    cpp.write_text('#include <vector>\n#include <algorithm>\n#include <iostream>\n'
                   'int main() { struct Req { std::vector<int> golden; }; Req q; std::vector<Req> reqs(1); '
                   'int o=0, tokens=32; for(int i=0;i<33;++i) { q.golden.push_back(i); reqs[0].golden.push_back(i); } '
                   f'q.golden[{changed}]=999; std::cout << ({condition}); }}\n')
    subprocess.run([str(toolchain) if toolchain.exists() else "c++", "-std=c++17", str(cpp), "-o", str(tmp_path / "collision")],
                   check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "collision")], check=True, capture_output=True, text=True)
    assert out.stdout == expected


@pytest.mark.parametrize("args", [["--goldens", "unused", "--batch", "8"],
                                  ["--golden", "unused", "--batches", "1,2,4,8"]])
def test_vllm_cli_rejects_options_for_the_other_mode(monkeypatch, capsys, args):
    import sys
    from types import ModuleType
    from examples.llm import vllm_bench as vb
    vllm = ModuleType("vllm")
    vllm.LLM = vllm.SamplingParams = object
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = object
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(sys, "argv", ["vllm_bench", "--model", "unused", *args])
    monkeypatch.setattr(vb, "batch_mode", lambda *a: pytest.fail("incompatible option reached batch mode"))
    monkeypatch.setattr(vb, "read_golden", lambda *a: pytest.fail("incompatible option reached legacy mode"))
    with pytest.raises(SystemExit) as exc:
        vb.main()
    assert exc.value.code == 2 and "only valid" in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["rounds", "vsel", "report", "boxcontrol"])
def test_batch_cli_rejects_an_ignored_arm(monkeypatch, capsys, operation):
    from examples.dsv2lite import batch_session as bs
    monkeypatch.setattr(bs, "BatchSession", lambda: pytest.fail("ignored arm reached session construction"))
    with pytest.raises(SystemExit) as exc:
        bs.main([operation, "x8p"])
    assert exc.value.code == 2 and "only valid" in capsys.readouterr().err


def test_bulk_floor_excludes_router_weights_in_the_dense_layer():
    from examples.dsv2lite.batch_floor import step_bytes
    # Independently count the pinned model: 27 attention projections, 26 MoE routers,
    # one dense MLP, 26*(6 routed + 2 shared) experts, embedding and vocabulary projection.
    weights = (27 * (2048 * 16 * 192 + 2048 * 576 + 2 * 512 * 16 * 128 + 16 * 128 * 2048)
               + 26 * 2048 * 64 + 3 * 10944 * 2048 + 26 * 8 * 3 * 1408 * 2048
               + 2048 + 102400 * 2048) * 2
    assert weights == 4_902_621_184
    for b in (1, 2, 4, 8):
        s = step_bytes(b, context=1024)
        assert s["one"] == weights + 27 * 1024 * 576 * 2
        assert s["replicated"] == b * s["one"]
    assert step_bytes(1)["batched"] == step_bytes(1)["one"]
