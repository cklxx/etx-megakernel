"""Fixed external-only CPU workflows run in an installed wheel outside the checkout."""
import os
from pathlib import Path

from etx.testing import installed_package_probe, scaffold_census


ROOT = Path(__file__).resolve().parents[2]


def test_installed_external_workflows_without_examples_or_etx_environment(tmp_path):
    # The DeepSeek part needs the retained pinned fleet export (the same input the dsv2lite tests use).
    fleet = os.environ.get('ETX_DSV2_FLEET_EXPORT')
    probe = installed_package_probe(tmp_path / 'package', declarations=ROOT, fleet=fleet)['probe']
    assert {(row['devices'], row['batch']) for row in probe['rows']} == {
        (devices, batch) for devices in (1, 2, 8) for batch in (1, 2, 4, 8)
    }
    assert all(row['status'] == 'SOURCES' for row in probe['rows'])
    assert len(probe['operators']) == 12
    assert {(row['name'], row['small'], row['size']) for row in probe['operators']} == {
        (name, small, size) for name, small in (
            ('splitk_sum', False), ('moe_layer', False), ('moe_layer', True))
        for size in (1, 2, 4, 8)
    }
    assert all(row['status'] == 'SOURCES' for row in probe['operators'])
    assert probe['gemm_ranks'] == 2
    assert probe['catalog_records'] == 13 and probe['historical_fixture']
    assert probe['relocated_process']
    assert probe['protocol_cases'] == 2 and probe['protocol_replay']
    assert set(probe['studies']) == {'variance', 'routing-skew'}
    assert all(len(study['rows']) == 5 and study['backend'] == 'simulation'
               and not study['gpu_measurement'] for study in probe['studies'].values())
    assert not probe['example_imports'] and not probe['device_qualification']
    assert probe['job_session'] and probe['job_record_verified'] and probe['transport_status']
    # Every set the census counts must have its lifecycle exercised by this installed probe.
    for row in scaffold_census():
        if row['counted']:
            assert all(probe.get(key) for key in row['installed']), (row['id'], 'set ETX_DSV2_FLEET_EXPORT' if row['id'] == 'S5' else '')
