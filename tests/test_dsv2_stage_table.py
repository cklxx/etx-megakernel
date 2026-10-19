"""CPU gates for the per-stage account: the replay reproduces the measured trace, and the stage rows partition it."""
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data():
    from examples.dsv2lite.pinned import data_sets
    try:
        return data_sets()
    except Exception as exc:          # no network / gh: the archived data cannot be fetched
        pytest.skip(f"archived data sets unavailable: {exc}")


def test_replay_reproduces_the_measured_trace(data):
    from examples.dsv2lite.replay import check, default_replay
    r = check(default_replay())
    assert abs(r["error_us"]) < 0.01 and r["clamped_skew_us"] < 0.01, r


def test_every_stage_has_a_lever_and_the_rows_partition_the_token(data, tmp_path):
    from examples.dsv2lite import stage_table as st
    # Write to tmp_path, never to the committed files: other tests (inplace_session report) read them.
    assert st.main(out=tmp_path) == 0
    for name in ("stage_table.json", "stage_table.md"):     # the committed table regenerates byte for byte
        assert (tmp_path / name).read_bytes() == (st.OUT / name).read_bytes(), name
    out = json.loads((tmp_path / "stage_table.json").read_text())
    rows = {r["stage"]: r for r in out["rows"]}
    assert list(rows) == st.ORDER and len(rows) == 14
    assert abs(sum(r["current_us"] for r in rows.values()) - out["summary"]["traced_token_us"]) < 1e-6
    for r in rows.values():
        assert r["lever"] and r["formula"], r["stage"]
        # every stage either has a priced lever, or says which lever was tried and why it was rejected
        assert (r["delta_us"] < 0 and r["basis"] != "-") or "REJECTED" in r["lever"], r["stage"]
    saving = -sum(r["delta_us"] for r in rows.values())
    assert abs(out["summary"]["arithmetic"]["saving_us"] - saving) < 1e-6
    assert out["summary"]["simulated"]["saving_us"] <= saving + 1e-6
