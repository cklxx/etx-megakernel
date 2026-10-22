"""Offline co-residency gate arithmetic and metadata parsing (no ROCm tools, no GPU)."""
import pytest
from examples.glm53 import residency

NOTES = '''
amdhsa.kernels:
  - .agpr_count:     {agpr}
    .args:
      - .address_space:  global
        .name:           p
        .offset:         0
        .size:           8
    .group_segment_fixed_size: {lds}
    .max_flat_workgroup_size: 256
    .name:           etx_megakernel_d0
    .private_segment_fixed_size: 960
    .sgpr_count:     108
    .vgpr_count:     {vgpr}
    .vgpr_spill_count: 271
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .name:           first
    .group_segment_fixed_size: 18944
    .max_flat_workgroup_size: 256
    .name:           etx_unfused_d0
    .sgpr_count:     108
    .vgpr_count:     256
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
'''


def test_parser_keeps_kernel_fields_and_ignores_argument_names():
    got = residency.kernels(NOTES.format(agpr=0, lds=18976, vgpr=256))
    assert [k['name'] for k in got] == ['etx_megakernel_d0', 'etx_unfused_d0']
    assert got[0]['vgpr_count'] == 256 and got[0]['vgpr_spill_count'] == 271 and got[0]['group_segment_fixed_size'] == 18976
    assert got[1]['private_segment_fixed_size'] == 0  # absent fields read as 0


def kernel(vgpr, lds=18976, agpr=0):
    return residency.kernels(NOTES.format(agpr=agpr, lds=lds, vgpr=vgpr))[0]


def test_rocm724_measured_default_reaches_two_workgroups_per_cu():
    r = residency.resident(kernel(256), 304)
    assert r['waves_per_simd'] == 2 and r['wg_per_cu'] == 2 and r['resident'] == 608


def test_rocm641_unified_count_includes_agprs_and_halves_residency():
    # 6.4.1 metadata: .vgpr_count 308 already contains the 128 AGPRs; they are not added again.
    r = residency.resident(kernel(308, agpr=128), 304)
    assert r['waves_per_simd'] == 1 and r['wg_per_cu'] == 1 and r['resident'] == 304


@pytest.mark.parametrize('vgpr,expected', [(128, 2), (256, 2), (257, 1), (512, 1)])
def test_register_granules_and_machine_cap(vgpr, expected):
    assert residency.resident(kernel(vgpr), 304)['wg_per_cu'] == expected


def test_lds_can_be_the_limit():
    assert residency.resident(kernel(128, lds=40000), 304)['wg_per_cu'] == 1


def test_bundles_split_at_every_header():
    blob = residency.MAGIC + b'a' * 16 + residency.MAGIC + b'b' * 8
    assert residency.bundles(blob) == [residency.MAGIC + b'a' * 16, residency.MAGIC + b'b' * 8]


def test_build_script_runs_the_gate_after_the_link():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'examples/glm53/build_model.sh').read_text()
    link = text.index('-pthread -o "$OUT/run"')
    gate = text.index('-m examples.glm53.residency "$OUT"')
    assert link < gate < text.index('finalize(Path(sys.argv[1])')
