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
    from etx.models.dsv2lite.replay import check, default_replay
    r = check(default_replay())
    assert abs(r["error_us"]) < 0.01 and r["clamped_skew_us"] < 0.01, r


def test_stage_order_is_the_committed_table_order(data):
    # The --out command regenerating the committed table byte for byte is in tests/e2e/test_api_dsv2_stage_table.py.
    from etx.models.dsv2lite import stage_table as st
    rows = json.loads((st.OUT / "stage_table.json").read_text())["rows"]
    assert [r["stage"] for r in rows] == st.ORDER and len(st.ORDER) == 14
