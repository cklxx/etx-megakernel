"""D1: every DeepSeek session builds with the safe sc1 acquire; the relaxed sc0 acquire appears only in an arm that is
named for it (examples/dsv2lite/acquire.py, docs/deepseek/DSV2-D1-D5.md)."""
from pathlib import Path

import pytest

from etx.experiments.jobs import load_session
from examples.dsv2lite.arms import ARMS, SINGLE
from examples.dsv2lite.acquire import ACQUIRE, RELAXED

SESSIONS = ["ds_inplace", "ds_model", "ds_interleave", "ds_evstride", "ds_gap"]


@pytest.mark.parametrize("name", SESSIONS)
def test_relaxed_only_in_a_named_arm(name):
    jobs, _ = load_session(Path(f"examples/sessions/{name}.py"))
    builds = [j for j in jobs if "build.sh" in j.cmd or "inplace_session build" in j.cmd]
    assert builds, name
    for j in builds:
        if RELAXED in j.cmd:
            assert "relaxed" in j.name, (name, j.name)


def test_the_default_is_safe_and_relaxed_is_one_explicit_arm():
    assert ACQUIRE == ""                                    # the build default: buffer_inv sc1
    assert RELAXED == "-DETX_ACQUIRE_DOMAIN_RELAXED"
    # relaxed only in arms named for it: D1's comparison, and allrelaxed (the prize measurement, never a candidate)
    assert {a for a, arm in ARMS.items() if RELAXED in arm.flags} == {"relaxed", "allrelaxed"}
    assert all("relaxed" in a for a, arm in ARMS.items() if RELAXED in arm.flags)
    assert ARMS["relaxed"].levers == () and "relaxed" in SINGLE
    jobs, _ = load_session(Path("examples/sessions/ds_inplace.py"))
    by = {j.name: j for j in jobs}
    assert by["build_relaxed"].cmd.endswith("inplace_session build relaxed")
    assert "pairs_relaxed" in by
    gap, _ = load_session(Path("examples/sessions/ds_gap.py"))
    assert any(RELAXED in j.cmd for j in gap if "relaxed" in j.name)   # the 09-27 comparison arm keeps its meaning
