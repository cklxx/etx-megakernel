"""Exercise real simulator outputs and fail-closed offline reporting."""
import copy
from dataclasses import asdict
import json

import pytest

from etx.tools.validation import bench_sim as bench
from etx.sim import simulate

CASE = {"name": "small", "workload": "splitk", "size": 2, "duration_cv": 0.1}
CONFIG = {"schema_version": 1, "arch": "gfx942", "rounds": 2, "seed": 7, "cases": [CASE]}


@pytest.fixture
def observation():
    plan = bench.make_plan(CASE, "gfx942", "static")
    return bench.contract(plan), asdict(simulate(plan, seed=7))


@pytest.mark.parametrize("mode", bench.MODES)
def test_protocol_completion(mode):
    plan = bench.make_plan(CASE, "gfx942", mode)
    assert bench.check_sample(bench.contract(plan), asdict(simulate(plan))) == []


@pytest.mark.parametrize("fault,expected", [
    ("missing", "coverage"), ("duplicate", "coverage"), ("nan", "nonfinite"),
    ("sync", "dependency"), ("deadlock", "deadlock"), ("budget", "disagrees"),
])
def test_injected_faults_fail(observation, fault, expected):
    spec, result = observation
    task = next(s for s in result["segments"] if s["kind"] == "task")
    if fault == "missing":
        result["segments"].remove(task)
    elif fault == "duplicate":
        result["segments"].append(copy.copy(task))
    elif fault == "nan":
        result["makespan_us"] = float("nan")
    elif fault == "sync":
        cid = next(int(t) for t, deps in spec["dependencies"].items() if deps)
        consumer = next(s for s in result["segments"] if s["kind"] == "task" and s["task"] == cid)
        consumer["start"] = 0
    elif fault == "deadlock":
        result["deadlock"] = True
    elif fault == "budget":
        result["busy_us"] += 1
    assert any(expected in e for e in bench.check_sample(spec, result))


def test_dynamic_waits_for_last_finishing_producer():
    case = {**CASE, "size": 1, "duration_cv": 0}
    plan = bench.make_plan(case, "gfx942", "dynamic")
    producers = [t for t in plan.tasks if t.grid == "partial_sum"]
    # The first visited producer finishes last; enqueue visitation is not readiness.
    producers[0].duration_us = 1000
    result = simulate(plan)
    assert bench.check_sample(bench.contract(plan), asdict(result)) == []
    consumer = next(s for s in result.segments if s.kind == "task" and plan.tasks[s.task].grid == "final_sum")
    latest = max(s.end for s in result.segments if s.kind == "task" and plan.tasks[s.task].grid == "partial_sum")
    assert consumer.start >= latest + plan.machine.t_sync_us(plan.events["E"].scope)


@pytest.fixture
def experiment(tmp_path):
    out = tmp_path / "run"
    assert bench.run(copy.deepcopy(CONFIG), out) == 0
    return out


def rewrite_samples(out, samples):
    path = out / "samples.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["artifacts"][path.name] = bench.digest(path.read_bytes())
    bench.write_json(out / "manifest.json", manifest)


def test_recompute_is_exact_and_rounds_are_paired(experiment):
    before = (experiment / "summary.json").read_bytes()
    samples = [json.loads(s) for s in (experiment / "samples.jsonl").read_text().splitlines()]
    assert [s["mode"] for s in samples] == ["static", "dynamic", "hybrid", "dynamic", "hybrid", "static"]
    assert [s["seed"] for s in samples] == [7, 7, 7, 8, 8, 8]
    assert bench.report(experiment) == 0
    assert (experiment / "summary.json").read_bytes() == before
    manifest = json.loads((experiment / "manifest.json").read_text())
    assert manifest["tensor_correctness"] == "not_executed"
    assert not manifest["gpu_measurement"]
    assert manifest["provenance"]["source_sha256"]["etx/sim/protocol.py"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "seed", "trace", "failure"])
def test_offline_report_rejects_incomplete_or_invalid_pairs(experiment, fault):
    samples = [json.loads(s) for s in (experiment / "samples.jsonl").read_text().splitlines()]
    if fault == "missing":
        samples.pop()
    elif fault == "duplicate":
        samples.append(samples[0])
    elif fault == "seed":
        samples[0]["seed"] += 1
    elif fault == "trace":
        samples[0]["result"]["segments"] = []  # Claimed errors=[] must not bypass revalidation.
    else:
        samples[0]["errors"] = ["injected failure"]
    rewrite_samples(experiment, samples)
    assert bench.report(experiment) == 1
    summary = json.loads((experiment / "summary.json").read_text())
    assert not summary["valid"] and not summary["comparisons"]


def test_corrupt_artifact_invalidates_previous_success(experiment):
    with (experiment / "samples.jsonl").open("a") as stream:
        stream.write("\n")
    assert bench.main(["report", str(experiment)]) == 1
    assert not json.loads((experiment / "summary.json").read_text())["valid"]


def test_refuse_overwrite(experiment):
    before = (experiment / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        bench.run(CONFIG, experiment)
    assert (experiment / "manifest.json").read_bytes() == before


def test_bad_configuration_rejected_before_output(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["cases"][0]["size"] = 0
    with pytest.raises(ValueError, match="size"):
        bench.run(config, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_exception_is_recorded_and_fails_run(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected simulator failure")
    monkeypatch.setattr(bench, "simulate", fail)
    out = tmp_path / "failed"
    assert bench.run(CONFIG, out) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["failures"]) == 6 and not summary["comparisons"]
    assert "injected simulator failure" in (out / "samples.jsonl").read_text()


def test_nonfinite_sample_is_retained_as_strict_json(tmp_path, monkeypatch):
    original = bench.simulate
    def fail(plan, seed):
        result = original(plan, seed=seed)
        result.makespan_us = float("inf")
        return result
    monkeypatch.setattr(bench, "simulate", fail)
    out = tmp_path / "nonfinite"
    assert bench.run(CONFIG, out) == 1
    raw = (out / "samples.jsonl").read_text()
    assert "Infinity" not in raw and "NaN" not in raw
    samples = [json.loads(s) for s in raw.splitlines()]
    assert all(s["result"]["makespan_us"] is None and s["errors"] for s in samples)


def test_inflated_makespan_fails_trace_and_report(observation, experiment):
    spec, result = observation
    result['makespan_us'] *= 100
    assert any('makespan disagrees' in e for e in bench.check_sample(spec, result))
    samples = [json.loads(s) for s in (experiment / 'samples.jsonl').read_text().splitlines()]
    samples[0]['result']['makespan_us'] *= 100
    rewrite_samples(experiment, samples)
    assert bench.report(experiment) == 1
    assert not json.loads((experiment / 'summary.json').read_text())['comparisons']


def test_provenance_binds_non_python_sources(tmp_path, monkeypatch):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    names = ['driver.sh', 'host.hip', 'runtime.h', 'model.tla', 'model.cfg', 'proof.lean', 'workflow.yml']
    for name in names:
        (tmp_path / name).write_text('original\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=test', '-c',
                    'user.email=test@example.com', 'commit', '-qm', 'fixture'], check=True)
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    before = bench.provenance()
    for name in names:
        (tmp_path / name).write_text('changed\n')
    after = bench.provenance()
    assert before['commit'] == after['commit']
    assert all(before['source_sha256'][n] != after['source_sha256'][n] for n in names)
