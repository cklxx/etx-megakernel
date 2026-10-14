"""examples/sessions/joint_ds_qwen.py: six studies on one box, the two models isolated by the actual artifact graph."""
import io
from pathlib import Path

import pytest

from etx.tools import gpu_jobs as gj

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "examples/sessions/joint_ds_qwen.py"
DS_STUDIES = ["pairs_alld4r0l1kvs", "pairs_alld4r0l1mw", "pairs_alld4r0l1noQ", "pairs_alld4r0l1noA", "trace_alld4r0l1kvs"]
QWEN_STUDIES = ["qwen_moerows", "qwen_pairs_kept_l1"]
GATES = ["qwen_certify_kept", "qwen_certify_kept_l1"]


@pytest.fixture(scope="module")
def joint():
    jobs, limit = gj.load_session(SESSION)
    return {j.name: j for j in jobs}, jobs, limit


def hard_ancestors(by, name):
    """Every job a job cannot run without (hard reads: files through their writer, or job names)."""
    writer = {w: j.name for j in by.values() for w in j.writes}
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


def test_exactly_six_studies_and_two_gates(joint):
    by, jobs, limit = joint
    import importlib.util
    spec = importlib.util.spec_from_file_location("joint", SESSION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.STUDIES == {"DeepSeek": DS_STUDIES, "Qwen": QWEN_STUDIES}
    assert sum(map(len, mod.STUDIES.values())) == 7
    assert all(by[n].lane == "gpu" for n in DS_STUDIES + QWEN_STUDIES + GATES)
    assert limit == 7200


def test_one_gpu_lane_in_the_planned_order(joint):
    _, jobs, _ = joint
    assert [j.name for j in jobs if j.lane == "gpu"] == [
        "qwen_moerows", "prep_ref", *DS_STUDIES, "qwen_stage_pack", *GATES, "qwen_pairs_kept_l1"]


def test_no_name_or_file_is_shared_and_qwen_records_under_qwen(joint):
    by, jobs, _ = joint
    assert len(by) == len(jobs)
    writes = [w for j in jobs for w in j.writes]
    assert len(writes) == len(set(writes))
    for j in jobs:
        if j.name.startswith("qwen_"):
            assert all(w.startswith(("qwen/", "$HOME/llm/", "$HOME/vllm/", "$HOME/qwen-llmenv/")) or w.endswith("qwen-g6.json")
                       for w in j.writes), j.name
        else:
            assert not any(w.startswith("qwen/") or "qwen" in w for w in j.writes), j.name


def test_environments_are_model_separated(joint):
    by, jobs, _ = joint
    for j in jobs:
        keys = set(j.env)
        if j.name.startswith("qwen_"):
            assert not any(k.startswith("ETX_DSV2_") for k in keys) and j.env["ETX_QWEN_G6_RECORD"] == "qwen"
        else:
            assert not any(k.startswith("ETX_QWEN_") or k.startswith("ETX_VL_") for k in keys)


def test_the_artifact_graph_isolates_the_models(joint):
    by, _, _ = joint
    for s in DS_STUDIES:
        assert not any(a.startswith("qwen_") for a in hard_ancestors(by, s)), s
    for s in QWEN_STUDIES + GATES:
        assert all(a.startswith("qwen_") for a in hard_ancestors(by, s)), s          # DeepSeek only through "?" waits
    assert "qwen_moerows" not in hard_ancestors(by, "qwen_pairs_kept_l1")         # the microbenchmark gates nothing
    assert {"?apt"} <= set(by["qwen_stage_env"].reads) and "?etx_env" in by["qwen_preflight"].reads


def settled(tmp_path, monkeypatch, failed):
    """Every job finished: ok with its declared outputs on disk (HOME is tmp, so $HOME paths never touch the real one),
    except `failed`. Returns the runner over the real joint graph."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    jobs, _ = gj.load_session(SESSION)
    r = gj.Runner(jobs, [0], 1, tmp_path / "rec", poll_s=.01, out=io.StringIO())
    for j in jobs:
        j.status = "fail" if j.name in failed else "ok"
        if j.status == "ok":
            for w in j.writes:
                f = r.path(j, w.lstrip("?"))
                f.parent.mkdir(parents=True, exist_ok=True)
                f.touch()
    return r


def state_if_pending(r, name):
    j = r.by_name[name]
    for w in j.writes:                                          # it has not run: its outputs do not exist
        f = r.path(j, w.lstrip("?"))
        if f.is_file():
            f.unlink()
    j.status = "pending"
    return r._state(j)


def test_a_failed_qwen_input_skips_only_qwen(tmp_path, monkeypatch):
    r = settled(tmp_path / "q", monkeypatch, failed={"qwen_stage_dl"})
    assert state_if_pending(r, "qwen_stage_pack") == "skip"
    for i, s in enumerate(DS_STUDIES):
        r = settled(tmp_path / str(i), monkeypatch, failed={"qwen_stage_dl"})
        assert state_if_pending(r, s) == "ready", (s, r.by_name[s].note)


def test_a_failed_microbenchmark_or_deepseek_does_not_block_the_qwen_answer(tmp_path, monkeypatch):
    for i, name in enumerate(("qwen_stage_env", "qwen_preflight", "qwen_pairs_kept_l1")):   # each on a settled graph
        r = settled(tmp_path / str(i), monkeypatch, failed={"qwen_moerows", "apt", "etx_env", "preflight", *DS_STUDIES})
        assert state_if_pending(r, name) == "ready", (name, r.by_name[name].note)


def test_a_failed_qwen_preflight_skips_qwen_builds_and_gates(tmp_path, monkeypatch):
    r = settled(tmp_path / "q", monkeypatch, failed={"qwen_preflight"})
    for name in ("qwen_build_kept", "qwen_build_kept_l1"):
        assert state_if_pending(r, name) == "skip", name
    for i, s in enumerate(DS_STUDIES):
        assert state_if_pending(settled(tmp_path / str(i), monkeypatch, failed={"qwen_preflight"}), s) == "ready", s
