"""examples/sessions/all_verify.py: DeepSeek's batch study and Qwen on ONE box, one cap, one GPU lane.

The guard is the plan the runner itself resolves (etx.tools.gpu_jobs.load_session), not this test's own reading of the
session file: duplicate names, a missing writer and a wrong GPU order are all refusals of the runner.
"""
from pathlib import Path

import pytest

from etx.tools import gpu_jobs as gj

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "examples/sessions/all_verify.py"
MEASURED = ["pairs_alld4r0l1kvs", "pairs_alld4r0l1mw", "pairs_alld4r0l1noQ", "pairs_alld4r0l1noA",
            "trace_alld4r0l1kvs", "pairs_alld4r0l1kvsmw"]       # results/2026-10-14-k1 and -km: never rerun here
BATCH_GPU = ["prep_goldens", "bgate_x2", "bgate_x4", "bgate_x8", "vsel", "brounds", "routing"]
QWEN_GPU = ["qwen_moerows", "qwen_stage_pack", "qwen_certify_kept", "qwen_certify_kept_l1", "qwen_pairs_kept_l1"]
CAP = 10800


@pytest.fixture(scope="module")
def plan():
    jobs, limit = gj.load_session(SESSION)
    return {j.name: j for j in jobs}, jobs, limit


def test_cap_and_no_duplicate_job_names(plan):
    _, jobs, limit = plan
    assert limit == CAP
    names = [j.name for j in jobs]
    assert len(names) == len(set(names))


def test_the_documented_studies_are_all_present(plan):
    by, _, _ = plan
    for name in BATCH_GPU + QWEN_GPU:
        assert name in by, f"{name} is missing from the combined session"
    import importlib.util
    spec = importlib.util.spec_from_file_location("all_verify", SESSION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.STUDIES == {"DeepSeek batch": ["bgate_x2", "bgate_x4", "bgate_x8", "brounds"],
                           "Qwen": ["qwen_moerows", "qwen_pairs_kept_l1"]}


def test_no_already_measured_deepseek_study_and_no_build_only_it_needs(plan):
    """AGENTS.md 2.5: K1, M1, KM, the ablations and the trace are measured; the kept arm runs only as batch's B = 1."""
    by, jobs, _ = plan
    assert not set(MEASURED) & set(by)
    assert {n for n in by if n.startswith("build_")} == {
        "build_alld4r0l1kvsmw", "build_x2", "build_x4", "build_x8"}
    assert "b_alld4r0l1kvsmw/run" in by["brounds"].reads


def test_one_gpu_lane_in_the_planned_order(plan):
    _, jobs, _ = plan
    assert [j.name for j in jobs if j.lane == "gpu"] == (
        ["prep_ref", "qwen_moerows"] + BATCH_GPU +
        ["qwen_stage_pack", "qwen_certify_kept", "qwen_certify_kept_l1", "qwen_pairs_kept_l1"])


def test_every_read_has_an_earlier_writer(plan):
    by, jobs, _ = plan
    writer = {w: j.name for j in jobs for w in j.writes}
    for job in jobs:
        for r in job.reads:
            if r.startswith("?"):          # optional: absence skips this job, never the run
                continue
            assert r in by or r in writer, f"{job.name} reads {r} and nothing produces it"


def test_the_models_share_no_file_and_qwen_records_under_qwen(plan):
    by, jobs, _ = plan
    writes = [w for j in jobs for w in j.writes]
    assert len(writes) == len(set(writes)), "two jobs write the same path"
    for j in jobs:
        if j.name.startswith("qwen_"):
            assert all(w.startswith("qwen/") or w.startswith("$") or w.startswith("/") for w in j.writes)
        else:
            assert not any(w.startswith("qwen/") for w in j.writes)


def test_a_failed_qwen_step_cannot_gate_deepseek_and_the_reverse(plan):
    """Hard ancestors only: the two models must not depend on each other's artifacts."""
    by, jobs, _ = plan
    writer = {w: j.name for j in jobs for w in j.writes}

    def hard_ancestors(name):
        seen, todo = set(), [name]
        while todo:
            for r in by[todo.pop()].reads:
                if r.startswith("?"):
                    continue
                p = r if r in by else writer.get(r)
                if p and p not in seen:
                    seen.add(p)
                    todo.append(p)
        return seen

    for name in BATCH_GPU:
        assert not any(a.startswith("qwen_") for a in hard_ancestors(name)), f"{name} depends on Qwen"
    for name in QWEN_GPU:
        assert not any(a in BATCH_GPU for a in hard_ancestors(name)), f"{name} depends on DeepSeek studies"
