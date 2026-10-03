import numpy as np
import pytest

from etx.bench.numerical import acceptance_suite, fixture, vectorized, scalar_reference, output_errors


def test_complete_acceptance_suite():
    result = acceptance_suite()
    assert result["passed"] and len(result["cases"]) == 24
    assert not result["gpu_measurement"] and not result["model_validation"]


@pytest.mark.parametrize("renormalize", [True, False])
def test_routing_ties_choose_smallest_id(renormalize):
    data = fixture(renormalize=renormalize)
    data["router"][:] = 0
    actual, written, selected = vectorized(data)
    reference, ids = scalar_reference(data)
    assert ids == selected == [0, 1]
    assert output_errors(actual, reference, written) == []


def test_reference_does_not_use_candidate_launch_parameters():
    data = fixture()
    reference, _ = scalar_reference(data)
    bad, mask, _ = vectorized({**data, "top_k": 1})
    assert output_errors(bad, reference, mask) == ["numerical mismatch"]


@pytest.mark.parametrize("fault", ["mask", "shape", "reference", "empty"])
def test_fail_closed_inputs(fault):
    a = np.ones(4)
    expected = a.copy()
    written = np.ones(4, bool)
    if fault == "mask":
        written = written.astype(int)
    elif fault == "shape":
        a = a.reshape(2, 2)
    elif fault == "reference":
        expected[0] = np.nan
    else:
        a = expected = written = np.array([])
    assert output_errors(a, expected, written)


def test_tolerance_cannot_disable_gate():
    with pytest.raises(ValueError):
        output_errors([1], [2], [True], atol=float("inf"))
