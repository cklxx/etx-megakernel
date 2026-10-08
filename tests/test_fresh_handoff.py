"""CPU fixtures test the evidence checker, never stand in for hardware results."""
import json

import pytest

from etx.tools import fresh_handoff as probe


def fixture(root, monkeypatch, mode='cached', topology='cross'):
    monkeypatch.setattr(probe, 'MIN_WORDS', 32)
    monkeypatch.setattr(probe, 'MIN_HANDOFFS', 2)
    case = dict(bytes=64, handoffs=2, topology=topology, load='idle', mode=mode)
    readers = 7 if topology == 'fanout' else 1
    wrong = 16 if mode in ('forced_old', 'primed_stale') else 0
    records = []
    for epoch in (1, 2):
        r = dict(schema_version=1, epoch=epoch, launch_id=epoch, payload_address=128,
                 **case, readers=readers, words_checked=32*readers, wrong_words=wrong*2*readers,
                 complete=True, placement_valid=True, valid=True, slot_stride_bytes=128,
                 wall_clock_khz=100000, timer_min_ticks=2, gpu='gfx942', load_kind='synthetic_only',
                 ncu=readers+1, cu_coverage=readers+1, busy_bytes=0)
        records.append(r)
        with (root / f'placement-{epoch}.csv').open('w') as f:
            f.write('block,cu_key,role,invalidations\n')
            for role in range(readers+1):
                key = role if topology == 'same' else role*256
                f.write(f'{role},{key},{role},1\n')
        with (root / f'samples-{epoch}.csv').open('w') as f:
            f.write('handoff,reader,first_load_ticks,bulk_checked_ticks,handoff_ack_ticks,wrong_words\n')
            for reader in range(1, readers+1):
                for handoff in (1, 2):
                    f.write(f'{handoff},{reader},15,100,200,{wrong}\n')
    save(root, records)
    return case, records


def save(root, records):
    (root / 'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))


def test_plan_has_full_matrix_and_one_handoff_threshold():
    plan = probe.matrix()
    assert len(plan['cases']) == 144
    assert len({c['id'] for c in plan['cases']}) == 144
    for c in plan['cases']:
        assert c['handoffs'] == 1_000_000        # every mode, not only the negative controls
        assert c['handoffs'] * c['bytes']//4 >= 10_000_000
    assert len(plan['model_followup']) == 2
    assert all('blocked' in m['status'] for m in plan['model_followup'])


@pytest.mark.parametrize('topology', ['same', 'cross', 'fanout'])
@pytest.mark.parametrize('mode', probe.MODES)
def test_valid_evidence(tmp_path, monkeypatch, topology, mode):
    case, _ = fixture(tmp_path, monkeypatch, mode, topology)
    report = probe.validate_case(tmp_path, case)
    assert [e['state'] for e in report] == ['pass', 'pass']
    if mode in ('cached', 'bypass'):
        assert report[0]['timings_us']['first_load_ticks']['p50'] == .15
    else:
        assert 'timings_us' not in report[0]


@pytest.mark.parametrize('field,value', [
    ('launch_id', 1), ('payload_address', 256), ('complete', False),
    ('placement_valid', False), ('words_checked', 31), ('slot_stride_bytes', 64),
    ('cu_coverage', 1), ('wall_clock_khz', 0), ('wall_clock_khz', float('inf')), ('valid', False), ('wrong_words', 1),
])
def test_bad_metadata_rejected(tmp_path, monkeypatch, field, value):
    case, records = fixture(tmp_path, monkeypatch)
    records[1][field] = value
    save(tmp_path, records)
    with pytest.raises(ValueError):
        probe.validate_case(tmp_path, case)


@pytest.mark.parametrize('kind', ['truncate', 'duplicate', 'stale', 'zero_time', 'bad_cu', 'missed_inv'])
def test_corrupt_raw_evidence_rejected(tmp_path, monkeypatch, kind):
    case, _ = fixture(tmp_path, monkeypatch)
    path = tmp_path / ('placement-2.csv' if kind in ('bad_cu', 'missed_inv') else 'samples-2.csv')
    lines = path.read_text().splitlines()
    if kind == 'truncate':
        lines.pop()
    elif kind == 'duplicate':
        lines[-1] = lines[-2]
    elif kind == 'stale':
        lines[-1] = '2,1,15,100,200,1'
    elif kind == 'zero_time':
        lines[-1] = '2,1,0,100,200,0'
    elif kind == 'bad_cu':
        lines[-1] = '1,0,1,1'
    else:
        lines[-1] = '1,256,1,0'
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):
        probe.validate_case(tmp_path, case)


def test_natural_stale_absent_is_inconclusive(tmp_path, monkeypatch):
    case, records = fixture(tmp_path, monkeypatch, 'primed_stale')
    for r in records:
        r.update(wrong_words=0, valid=False)
        path = tmp_path / f"samples-{r['epoch']}.csv"
        path.write_text(path.read_text().replace(',200,16', ',200,0'))
    save(tmp_path, records)
    assert all(e['state'] == 'inconclusive' for e in probe.validate_case(tmp_path, case))


def test_small_fixture_cannot_pass_production_threshold(tmp_path, monkeypatch):
    case, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, 'MIN_WORDS', 10_000_000)
    with pytest.raises(ValueError, match='insufficient'):
        probe.validate_case(tmp_path, case)


def test_handoff_threshold_covers_every_mode(tmp_path, monkeypatch):
    case, _ = fixture(tmp_path, monkeypatch, 'forced_old')
    monkeypatch.setattr(probe, 'MIN_HANDOFFS', 1_000_000)
    with pytest.raises(ValueError, match='insufficient'):
        probe.validate_case(tmp_path, case)


def test_missing_runs_fail_closed(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'plan': probe.matrix()}))
    report = probe.analyze(tmp_path)
    assert report['mechanism_gate'] == 'not_passed'
    assert len(report['cases']) == 144
    assert all(c['state'] == 'fail' for c in report['cases'])
    assert report['paired'] == []


def test_edited_matrix_cannot_pass(tmp_path):
    plan = probe.matrix()
    plan['cases'] = []
    (tmp_path / 'manifest.json').write_text(json.dumps({'plan': plan}))
    with pytest.raises(ValueError, match='frozen'):
        probe.analyze(tmp_path)


@pytest.mark.parametrize('damage', ['none', 'checksum', 'exit', 'missing_inventory'])
def test_archive_analysis(tmp_path, monkeypatch, damage):
    directory = tmp_path / '64-cross-idle-cached-0'
    directory.mkdir()
    case, _ = fixture(directory, monkeypatch)
    case.update(id=directory.name, repeat=0)
    plan = {'cases': [case]}
    monkeypatch.setattr(probe, 'matrix', lambda: plan)
    (tmp_path / 'manifest.json').write_text(json.dumps({'plan': plan}))
    inventory = {p.name: probe.digest(p) for p in directory.iterdir()}
    if damage == 'checksum':
        inventory['samples-2.csv'] = 'bad'
    if damage == 'missing_inventory':
        inventory.pop('placement-1.csv')
    code = 3 if damage == 'exit' else 0
    (tmp_path / f'{directory.name}.status.json').write_text(json.dumps({'returncode': code, 'sha256': inventory}))
    result = probe.analyze(tmp_path)
    assert result['cases'][0]['state'] == ('pass' if damage == 'none' else 'fail')
