"""CPU refusal tests for the paid-session plan; no GPU or network."""
import csv
from pathlib import Path
import pytest
from examples.sessions import qwen300 as s


def test_two_serial_bf16_jobs():
    assert [j['name'] for j in s.JOBS] == ['job1_pricing', 'job2_key32']
    assert [j['timeout'] for j in s.JOBS] == [1800, 2400]
    assert all(j['gpu'] == 0 and j['quiet_s'] <= 180 for j in s.JOBS)
    assert s.JOBS[0]['after_any'] == []
    assert s.JOBS[1]['after_any'] == ['job1_pricing']
    assert 'All configurations require bf16 and 32/32' in s.HEADER
    assert '<=21' in s.HEADER and '>24' in s.HEADER
    assert s.HF == {'bf16': 'Qwen/Qwen3-30B-A3B'}


@pytest.mark.parametrize('gate', ['ok', 'bad_golden', 'missing_trace_correctness'])
def test_pricing_job_prepares_bf16_and_requires_gates(tmp_path, monkeypatch, gate):
    import time
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(s, 'DEADLINE', time.monotonic()+1800)
    calls, priced = [], []
    def step(name, cmd, seconds):
        calls.append((name, cmd))
        return 'TRACE COMPLETE 32/32' if gate != 'missing_trace_correctness' else ''
    monkeypatch.setattr(s, 'step', step)
    monkeypatch.setattr(s, 'price_trace', lambda path, precision: priced.append((path, precision)))
    if gate == 'bad_golden':
        bad = tmp_path/'bad_golden.txt'
        bad.write_text('corrupted reference')
        monkeypatch.setattr(s, 'GOLDEN', bad)
        with pytest.raises(RuntimeError, match='bf16 golden hash mismatch'):
            s.job1()
        assert [name for name, _ in calls] == ['image', 'record', 'bf16_download']
        assert not s.golden('bf16').exists()
    else:
        if gate == 'missing_trace_correctness':
            with pytest.raises(RuntimeError, match='trace correctness missing'):
                s.job1()
        else:
            s.job1()
        assert [name for name, _ in calls] == ['image', 'record', 'bf16_download', 'bf16_bar', 'trace_bf16']
        assert s.golden('bf16').read_bytes() == s.GOLDEN.read_bytes()
        cmd = calls[-1][1]
        assert cmd[:3] == ['sudo', 'docker', 'run']
        assert 'rocprofv3' in cmd and '--kernel-trace' in cmd
        assert cmd[-4:] == ['--inside', 'trace', '--precision', 'bf16']
    assert priced == ([(tmp_path/'trace_bf16', 'bf16')] if gate == 'ok' else [])


@pytest.mark.parametrize('args', [
    ['--job', '3'], ['--inside', 'hf'], ['--inside', 'probe'],
    ['--inside', 'bench', '--precision', 'fp8'],
])
def test_removed_jobs_and_modes_refused(args):
    with pytest.raises(SystemExit) as exc:
        s.main(args)
    assert exc.value.code == 2


def trace(tmp_path, missing=False):
    path = tmp_path/'trace'
    path.mkdir()
    with (path/'kernel_trace.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['Start_Timestamp','End_Timestamp','Kernel_Name'])
        t = 0
        for count in (1,32):
            for token in range(count):
                for layer in range(48):
                    for name, duration in [('unified_attention', 3000), ('attention_reduce', 1000),
                                           ('ck_moe_stage1', 13000), ('ck_moe_stage2', 8000)]:
                        if not (missing and count == 32 and token == 31 and layer == 47 and name == 'ck_moe_stage2'):
                            w.writerow([t,t+duration,name])
                        t += duration
            t += 250_000_000
    return path


def test_trace_layer_identity_and_units(tmp_path, monkeypatch):
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    rows = s.price_trace(trace(tmp_path), 'bf16')
    assert len(rows) == 48
    assert all(r['stage1'] + r['stage2'] == 21 for r in rows)
    assert all(r['attention'] == 3 and r['reduce'] == 1 for r in rows)


def test_missing_stage_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='coverage'):
        s.price_trace(trace(tmp_path, missing=True), 'bf16')
    assert not (tmp_path/'bf16_pricing.json').exists()


def routing(tmp_path):
    for arm in ('off','on'):
        for pos in (0,1040):
            path = tmp_path/f'routing_{arm}_p{pos}'
            path.mkdir()
            for layer in range(48):
                for suffix in ('ids','weights'):
                    (path/f'L{layer}_{suffix}.bin').write_bytes(bytes(256))


def test_routing_all_layers_bit_identity(tmp_path):
    routing(tmp_path)
    s.routing_gate(tmp_path)
    (tmp_path/'routing_on_p1040/L17_weights.bin').write_bytes(bytes(255)+b'1')
    with pytest.raises(RuntimeError, match='mismatch'):
        s.routing_gate(tmp_path)


def test_missing_routing_refuses(tmp_path):
    routing(tmp_path)
    (tmp_path/'routing_off_p0/L0_ids.bin').unlink()
    with pytest.raises(RuntimeError, match='missing/extra'):
        s.routing_gate(tmp_path)


def test_step_failure_retains_full_log(tmp_path, monkeypatch):
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    monkeypatch.setattr(s, 'DEADLINE', float('inf'))
    # Normal operation has a finite deadline.
    import time
    monkeypatch.setattr(s, 'DEADLINE', time.monotonic()+20)
    with pytest.raises(RuntimeError, match='rc=7'):
        s.step('failure', ['bash','-c','echo full-output; exit 7'], 10)
    assert 'full-output' in (tmp_path/'failure.log').read_text()
    with pytest.raises(RuntimeError, match='overwrite'):
        s.step('failure', ['true'], 10)


def test_ck_demangled_symbols_are_attributed_by_validated_order(tmp_path, monkeypatch):
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    directory = trace(tmp_path)
    path = directory/'kernel_trace.csv'
    path.write_text(path.read_text().replace('ck_moe_stage1', 'ck::kernel_moe_gemm<up>')
                    .replace('ck_moe_stage2', 'ck::kernel_moe_gemm<down>'))
    rows = s.price_trace(directory, 'bf16')
    assert all(r['stage1'] == 13 and r['stage2'] == 8 for r in rows)


def test_container_preserves_home_and_hard_stop(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    monkeypatch.setattr(s, 'DEADLINE', time.monotonic()+100)
    monkeypatch.setenv('HOME', str(tmp_path))
    cmd = s.container(['python3','-V'], 600)
    assert f'HOME={tmp_path}' in cmd
    assert cmd[cmd.index(s.IMAGE)+1:cmd.index(s.IMAGE)+3] == ['-k','5']
    assert int(cmd[cmd.index(s.IMAGE)+3]) <= 90
