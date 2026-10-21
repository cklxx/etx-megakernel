from pathlib import Path
import pytest
from examples.experiment_program.router.prepare import source, capture_manifest


def test_actual_router_math_and_protocol_anchors():
    s=source()
    assert 'dsl_run_dsl_gemv_router_fold' in s
    assert 'dsl_run_dsl_topk' in s
    assert 'old == 127' in s and 'r < 8' in s
    assert 'ETX_RELEASE_DOMAIN();' in s and 'ETX_ACQUIRE_DEVICE();' in s
    assert 'domain * 32' in s and 'next[domain]' in s
    assert 'norm' not in s.split('// G01 native subgraph')[1].split('struct G01State')[0]


def test_missing_capture_fails_closed(tmp_path):
    with pytest.raises((FileNotFoundError,ValueError)):capture_manifest(tmp_path)


def native_log(population=2,seed=19):
    from examples.experiment_program.protocol import pair_order
    lines=[f'CHECK G01 case={i} exact=1' for i in range(96)]
    for i,order in enumerate(pair_order(seed)):
        for ch in order:
            arm=int(ch=='B');t=100000-arm*1000
            lines.extend(f'SAMPLE G01 ordinal={i} arm={arm} sample={j} device_ns={t:.9f} wall_ns={t+100:.9f}' for j in range(population))
            lines.append(f'PAIR G01 ordinal={i} arm={arm} device_ns={t:.9f} wall_ns={t+100:.9f} population={population}')
    return '\n'.join(lines)+'\nG01 COMPLETE all96 exact\n'


def test_raw_populations_reject_truncation_duplicates_and_edited_means():
    from examples.experiment_program.router.receipts import parse_native
    raw=native_log()
    assert parse_native(raw,population=2,seed=19)['admitted'] is False # gain<2%
    for changed in (raw.replace('CHECK G01 case=33 exact=1\n',''),
                    raw.replace('sample=1','sample=0',1),
                    raw.replace('device_ns=100000.000000000 wall_ns=100100.000000000 population=2',
                                'device_ns=100001.000000000 wall_ns=100100.000000000 population=2',1)):
        with pytest.raises(ValueError):parse_native(changed,population=2,seed=19)


def test_check_launch_indices_include_inserted_reset_grids():
    # Imported launches after router0 drift by one, after router47 by48.
    # A direct textual-instance index silently checks the wrong launch count.
    host=Path('examples/vllm_llm/host.hip').read_text()
    assert 'etx_launch_count[x.n + 1]' not in host
    assert 'etx_launch_count[physical_launch(li)]' in host
    assert 'physical_launch = [&](int virtual_li, bool reset_router=false)' in host
    assert 'etx_launch_count[physical_launch(x.n + nreq)]' in host
    assert '!= physical_launch((int)insts.size() + 2 * nreq)' in host


@pytest.mark.parametrize('nreq,enabled', [(1, 1), (1, 0), (2, 0), (4, 0), (8, 0)])
def test_physical_launch_mapping_on_cpu(tmp_path, nreq, enabled):
    import shutil
    import subprocess
    compiler=shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    host=Path('examples/vllm_llm/host.hip').read_text()
    mapping=host[host.index('  auto physical_launch ='):host.index('  printf("etx:', host.index('  auto physical_launch ='))]
    source=tmp_path/'mapping.cpp'
    source.write_text('''using int32_t = int;
#define assert(x) ((x) ? (void)0 : __builtin_trap())
#define ETX_CHECK(...) ((void)0)
struct Role { bool router; bool operator==(const char*) const { return router; } };
struct Inst { int n; Role role; };
int main() {
''' + f'#define ETX_EMIT_LAST_ROUTER {enabled}\nint nreq={nreq};\n' + '''
Inst insts[96];
for (int i=0;i<96;++i) insts[i]={i,{i%2==0}};
int etx_launch_first[256]={};
struct {int p;} hst;
''' + mapping + '''
for(int li=0;li<96+2*nreq;++li){
  int resets=0;
  for(const Inst& x:insts) if(ETX_EMIT_LAST_ROUTER && x.role=="router" && x.n+nreq<=li) ++resets;
  assert(physical_launch(li)==li+resets);
  assert(physical_launch(li,true)==li+resets);
}
assert(physical_launch(96+2*nreq)==96+2*nreq+48*ETX_EMIT_LAST_ROUTER);
}
''')
    binary=tmp_path/'mapping'
    subprocess.run([compiler,'-std=c++17',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_replay_poisons_every_output_before_timing():
    host=Path('examples/experiment_program/router/replay.cpp').read_text()
    assert 'b^=0xff' in host
    assert host.index('c.observed_poison.data()')<host.index('auto host=Clock::now()')
    assert 'wait(stream); // diagnostic poisoning' in host


def test_separate_probe_never_enters_decisive_source():
    assert '#define G01_PROBE 1' not in source()
    assert '#define G01_PROBE 1' in source(probe=True)
    assert 'g01_stamp(trace,256+domain,7,local_worker,epoch)' in source(probe=True)


def test_unparsed_or_diagnostic_rows_cannot_be_native_admission():
    from examples.experiment_program.router.receipts import parse_native
    raw=native_log()
    for extra in ('SAMPLE G01 ordinal=0 arm=0 sample=4 device_ns=NaN wall_ns=1\n',
                  'PAIR G01 ordinal=0 arm=0 device_ns=NaN wall_ns=1 population=2\n',
                  'CHECK G01 case=3 exact=1 truncated\n'):
        with pytest.raises(ValueError):parse_native(raw+extra,population=2,seed=19)
    with pytest.raises(ValueError):parse_native(raw.replace('G01 COMPLETE','G01 DIAGNOSTIC COMPLETE'),population=2,seed=19)
