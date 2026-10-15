import copy

import pytest

from examples.dsv2lite.multibatch_trace import aggregate, validate_rows


def fixture():
    return dict(descs=[dict(type=2, coord=[3], worker=7)], task_in=[[]], ev_inst=[]), [(2, 3, 0, 0, 0, 10, 12, 15, 7)]


def test_complete_trace():
    validate_rows(*fixture())


@pytest.mark.parametrize("column,value", [(0, 1), (1, 4), (5, 0), (5, 13), (6, 16), (8, 8)])
def test_rejects_corruption_before_display_clamping(column, value):
    plan, rows = fixture()
    row = list(rows[0]); row[column] = value
    with pytest.raises(ValueError):
        validate_rows(plan, [row])


def test_missing_coverage_and_dependencies():
    plan, rows = fixture()
    with pytest.raises(ValueError):
        validate_rows(plan, [])
    del plan['task_in']
    with pytest.raises(ValueError):
        validate_rows(plan, rows)


def test_family_aggregation_weights_means_and_preserves_request_rows():
    rows = [dict(phase='r0_gate_up', tasks=2, body_us=3., wait_us=4., cp_body_us=5.,
                 cp_dep_gap_us=6., cp_queue_gap_us=7.),
            dict(phase='r1_gate_up', tasks=4, body_us=5., wait_us=6., cp_body_us=7.,
                 cp_dep_gap_us=8., cp_queue_gap_us=9.)]
    original = copy.deepcopy(rows)
    a, = aggregate(rows)
    assert a == dict(family='gate_up', tasks=6, body_work_us=26., wait_work_us=32.,
                     cp_body_us=12., cp_dep_gap_us=14., cp_queue_gap_us=16.)
    assert rows == original
