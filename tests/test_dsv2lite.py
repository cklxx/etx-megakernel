"""Fleet graph/planner conformance through the supported testing surface."""
from etx.testing import dsv2lite_conformance


def test_graph_shape_matches_fleet():
    result = dsv2lite_conformance('shape')
    assert result['verified'] and 50_000 < result['tasks'] < 60_000


def test_simulator_prediction_is_in_fleets_range():
    result = dsv2lite_conformance('simulation')
    assert not result['deadlock']
    assert 80 < result['simulated_us_per_layer'] < 300
    assert not result['device_qualification']


def test_two_request_interleave_graph():
    result = dsv2lite_conformance('interleave')
    assert result['couplings'] == 3 and result['workers_per_domain'] == 76
    assert result['emitted']
