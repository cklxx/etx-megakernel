"""Independent loop-count checks plus a real instrumented decode comparison."""
import pytest

from examples.spec.tile_reads import iape_lookups, routed_tiles, row_multiplicity


REC = dict(shape=[1024, 512], dtype='torch.float8_e4m3fn', offset=0)


def test_decode_padded_pairs_and_scale_counts():
    cpu = routed_tiles(REC, projection='gate', tiles=86, selected_slot=0,
                       path='decode_cpu')
    device = routed_tiles(REC, projection='gate', tiles=86, selected_slot=0)
    # Independent arithmetic: slot 0 spans 28 full 36-row tiles and 16
    # rows of tile 28. Eight of those tiles have 35 rows (odd padding).
    assert sum(t.payload_bytes for t in cpu) == (1024+8)*512
    assert sum(t.total_bytes for t in cpu) == 2_641_920
    assert sum(t.total_bytes for t in device) == 660_480
    counts = row_multiplicity(device)
    assert len(counts) == 1024
    assert sum(n-1 for n in counts.values()) == 8
    assert max(counts.values()) == 2
    assert cpu[0].rows == device[0].rows


@pytest.mark.parametrize('projection,shape,tiles', [
    ('gate', [1024, 512], 29), ('up', [1024, 512], 29),
    ('down', [512, 1024], 8)])
def test_verify_has_no_payload_repeats_but_scalar_scales(projection, shape, tiles):
    rec = dict(REC, shape=shape)
    reads = routed_tiles(rec, projection=projection, tiles=tiles, path='verify')
    assert set(row_multiplicity(reads).values()) == {1}
    assert sum(t.payload_bytes for t in reads) == 524_288
    assert sum(t.scale_bytes for t in reads) == 4*524_288


def test_down_and_unaligned_device_fallback():
    reads = routed_tiles(REC, projection='down', tiles=64)
    assert set(row_multiplicity(reads).values()) == {1}
    rec = dict(REC, offset=1)
    fallback = routed_tiles(rec, projection='down', tiles=64)
    assert sum(t.scale_bytes for t in fallback) == 4*524_288


def test_pool_baseline_and_first_active_position():
    assert iape_lookups(pos=0, pool=4, index_topk=64, width=128) == 0
    assert iape_lookups(pos=66, pool=4, index_topk=64, width=128) == 0
    assert iape_lookups(pos=67, pool=4, index_topk=64, width=128, tiles=2) == 3*17*4*128*2


def test_native_cpu_observation_matches_static_tiles(tmp_path):
    from examples.glm53 import module_gate, stage
    from examples.glm53.verify_graph import VerifyGraphCPU, observed_step
    from examples.spec.ledger import ObservedBytes
    module_gate.checkpoint(tmp_path/'checkpoint', variant='module8')
    stage.stage(tmp_path/'checkpoint', tmp_path/'stage', maxlen=160, gate='module8')
    spec = stage.validate(tmp_path/'stage', allow_fixture=True, gate='module8')
    cpu = VerifyGraphCPU(tmp_path/'stage', 0, 1, 'module8', tmp_path/'run', observe=True, tiles='decode')
    sink = ObservedBytes(spec)
    observed_step(cpu, spec, 0, [[1]], sink, tiles='decode')
    ids = cpu.buffers['l1_sel'].view(cpu.np.float32)[:2].astype(int).tolist()
    for slot, expert in enumerate(ids):
        for projection in ('gate', 'up', 'down'):
            key = f'mlp.experts.{expert}.{projection}_proj.weight'
            record = spec['weights']['l1']['tensors'][key]
            grid = cpu.graph.grid('l1_'+('down' if projection == 'down' else 'up'))
            reads = routed_tiles(record, projection=projection, tiles=grid.grid[1],
                                 selected_slot=slot, path='decode_cpu')
            assert sink.snapshot()['l1', key] == sum(t.total_bytes for t in reads)
    assert sink.snapshot()['l1', 'self_attn.indexer.index_kpool_compress_ape'] == 0
