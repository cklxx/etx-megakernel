"""Internal seams exercised exclusively through the supported probe facade."""
import pytest
from etx.testing import conformance


@pytest.mark.parametrize('devices', [1, 2, 8])
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
def test_provider_planner_conformance(devices, batch):
    result = conformance('planning', devices=devices, batch=batch)
    assert result.checks == ('verified', 'isolated', 'ownership', 'scope')
    assert result.evidence['backend'] == 'cpu'
    assert result.evidence['tasks'] == devices * batch * 2


def test_private_declaration_storage_is_frozen():
    assert conformance('declarations').checks == ('frozen-storage',)


def test_unsupported_probe_is_not_a_pass():
    with pytest.raises(ValueError, match='unknown conformance seam'):
        conformance('imaginary')
